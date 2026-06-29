import os
import argparse
import json
from datetime import datetime
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from scipy.signal import find_peaks

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
TASK = 'mackey_glass'

from utils.run_dirs import make_run_dir, sweep_summary_path, epoch_dir, save_training_checkpoint, promote_epoch_artifacts
from utils.manifold_dimension_analysis import analyze_manifold_dimension, collect_and_save_final_states
from utils.slon_analysis import extract_model_parameters, compute_parameter_statistics
from utils.plotting_utils import (
    plot_regression_metrics,
    plot_mackey_glass_snapshots,
    plot_mackey_glass_parameter_analysis,
    plot_epoch_weight_heatmaps,
    create_mackey_glass_gifs,
    plot_signal_stages_for_example,
)


class MackeyGlassDataset(Dataset):
    def __init__(self, series, input_length, horizon, sample_step):
        self.series = series.astype(np.float32)
        self.input_length = input_length
        self.horizon = horizon
        self.sample_step = int(sample_step)

        sampled_indices = []
        idx = 0
        while idx < len(series):
            sampled_indices.append(idx)
            idx += self.sample_step

        self.sampled_indices = np.array(sampled_indices, dtype=np.int64)

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        start_idx = max(0, idx - self.input_length + 1)
        input_sampled_indices = self.sampled_indices[start_idx:idx+1]
        
        if len(input_sampled_indices) < self.input_length:
            padding_needed = self.input_length - len(input_sampled_indices)
            first_idx = input_sampled_indices[0] if len(input_sampled_indices) > 0 else 0
            padding = np.full(padding_needed, first_idx, dtype=np.int64)
            input_sampled_indices = np.concatenate([padding, input_sampled_indices])
        
        last_sampled_idx = self.sampled_indices[idx]
        target_idx = last_sampled_idx + self.horizon
        if target_idx >= len(self.series):
            target_idx = len(self.series) - 1
        
        x = self.series[input_sampled_indices]
        y = self.series[target_idx]
        prev_value = self.series[last_sampled_idx]
        
        x = torch.from_numpy(x).unsqueeze(-1)
        y = torch.tensor([y], dtype=torch.float32)
        prev_value_tensor = torch.tensor([prev_value], dtype=torch.float32)
        return x, y, last_sampled_idx, prev_value_tensor


def generate_mackey_glass(
    length,
    tau=17.0,
    delta_t=1.0,
    beta=0.2,
    gamma=0.1,
    n=10,
    x0=1.2,
):
    delay_steps = int(tau / delta_t)
    total_len = length + delay_steps + 1
    x = np.zeros(total_len, dtype=np.float64)
    x[: delay_steps + 1] = x0

    for t in range(delay_steps, total_len - 1):
        x_tau = x[t - delay_steps]
        dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + delta_t * dx

    return x[delay_steps + 1 :]


def remove_top_frequencies(series, delta_t, n_freqs=3):
    if n_freqs == 0:
        return series.copy(), np.array([]), np.array([])
    
    fft_vals = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series), delta_t)
    power = np.abs(fft_vals) ** 2
    positive_freqs = freqs[: len(freqs) // 2]
    positive_power = power[: len(power) // 2]
    
    low_freq_mask = positive_freqs <= 0.2
    low_freq_indices = np.where(low_freq_mask)[0]
    low_freq_power = positive_power[low_freq_mask]
    
    top_local_indices = np.argsort(low_freq_power)[-n_freqs:]
    top_global_indices = low_freq_indices[top_local_indices]
    
    top_freqs = positive_freqs[top_global_indices]
    top_powers = positive_power[top_global_indices]
    
    filtered_fft = np.zeros_like(fft_vals)
    for idx in top_global_indices:
        filtered_fft[idx] = fft_vals[idx]
        if idx > 0:
            neg_idx = len(freqs) - idx
            if neg_idx < len(freqs):
                filtered_fft[neg_idx] = fft_vals[neg_idx]
    
    filtered_series = np.real(np.fft.ifft(filtered_fft))
    residual = series - filtered_series
    return residual, top_freqs, top_powers


def build_dataloaders(
    series_length,
    input_length,
    horizon,
    batch_size,
    val_fraction,
    test_fraction,
    tau,
    delta_t,
    beta,
    gamma_mg,
    n,
    x0,
    seed,
    remove_top_n_freqs=0,
):
    series = generate_mackey_glass(
        length=series_length,
        tau=tau,
        delta_t=delta_t,
        beta=beta,
        gamma=gamma_mg,
        n=n,
        x0=x0,
    )

    removed_freqs = None
    removed_powers = None
    if remove_top_n_freqs > 0:
        series, removed_freqs, removed_powers = remove_top_frequencies(series, delta_t, n_freqs=remove_top_n_freqs)

    total_len = len(series)
    sample_step = 1
    cutoff_idx = int(total_len * (1 - test_fraction - val_fraction))
    val_cutoff_idx = int(total_len * (1 - test_fraction))

    train_series = series[:cutoff_idx]
    val_series = series[cutoff_idx:val_cutoff_idx]
    test_series = series[val_cutoff_idx:]

    train_ds = MackeyGlassDataset(train_series, input_length, horizon, sample_step)
    val_ds = MackeyGlassDataset(val_series, input_length, horizon, sample_step)
    test_ds = MackeyGlassDataset(test_series, input_length, horizon, sample_step)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, series, removed_freqs, removed_powers


def evaluate_model(data_loader, model, loss_fn, batch_size_test, mse_loss_fn=None):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                inputs, targets, _, prev_values = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
                prev_values = None
            else:
                inputs, targets = batch
                prev_values = None
            inputs = inputs.permute(1, 0, 2)
            targets = targets

            if inputs.size(1) != batch_size_test:
                batch_size_current = inputs.size(1)
            else:
                batch_size_current = batch_size_test

            out = model(inputs)
            preds = out["output"]

            if prev_values is not None and isinstance(loss_fn, NormalizedErrorLoss):
                loss = loss_fn(preds, targets, prev_values)
            else:
                if mse_loss_fn is not None:
                    loss = mse_loss_fn(preds, targets)
                else:
                    loss = loss_fn(preds, targets)
            
            total_loss += loss.item() * batch_size_current
            total_count += batch_size_current

    return total_loss / max(total_count, 1)


def evaluate_r2(data_loader, model, batch_size_test):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                inputs, targets, _, _ = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs = inputs.permute(1, 0, 2)

            out = model(inputs)
            preds = out["output"]

            preds_all.append(preds.detach().cpu().numpy().reshape(-1))
            targets_all.append(targets.detach().cpu().numpy().reshape(-1))

    if not preds_all:
        return 0.0

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)

    y_mean = y_true.mean()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


class NormalizedErrorLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3, max_ratio=100.0, min_denominator=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio
        self.min_denominator = min_denominator
    
    def forward(self, preds, targets, prev_values):
        pred_errors = torch.abs(preds - targets)
        time_step_changes = torch.abs(targets - prev_values)
        denominator = torch.clamp(time_step_changes, min=self.min_denominator) + self.epsilon
        normalized = pred_errors / denominator
        normalized = torch.clamp(normalized, 0.0, self.max_ratio)
        normalized = torch.log(1.0 + normalized)
        return torch.mean(normalized)


def evaluate_normalized_error(data_loader, model, batch_size_test, epsilon=1e-8):
    model.eval()
    normalized_errors = []
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                inputs, targets, _, prev_values = batch
            else:
                continue
            
            inputs = inputs.permute(1, 0, 2)
            
            if inputs.size(1) != batch_size_test:
                batch_size_current = inputs.size(1)
            else:
                batch_size_current = batch_size_test

            out = model(inputs)
            preds = out["output"]
            
            pred_errors = torch.abs(preds - targets)
            time_step_changes = torch.abs(targets - prev_values)
            
            normalized = pred_errors / (time_step_changes + epsilon)
            normalized_errors.append(normalized.detach().cpu().numpy().reshape(-1))
            total_count += batch_size_current

    if not normalized_errors:
        return float('inf')
    
    all_normalized = np.concatenate(normalized_errors)
    return np.mean(all_normalized)


def collect_all_test_predictions(test_loader, model, batch_size_test, test_start_idx, horizon):
    model.eval()
    all_predictions = []
    all_targets = []
    all_target_indices = []
    all_input_windows = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                inputs, targets, sampled_indices, prev_values = batch
            elif len(batch) == 3:
                inputs, targets, sampled_indices = batch
            else:
                inputs, targets = batch
                sampled_indices = None
            
            inputs = inputs.permute(1, 0, 2)
            
            if inputs.size(1) != batch_size_test:
                batch_size_current = inputs.size(1)
            else:
                batch_size_current = batch_size_test

            out = model(inputs)
            preds = out["output"]
            
            preds_np = preds.detach().cpu().numpy().reshape(-1)
            targets_np = targets.detach().cpu().numpy().reshape(-1)
            
            all_predictions.append(preds_np)
            all_targets.append(targets_np)
            
            if sampled_indices is not None:
                sampled_indices_np = sampled_indices.detach().cpu().numpy()
                for i, sampled_idx in enumerate(sampled_indices_np):
                    last_sampled_idx = int(sampled_idx)
                    target_idx_in_test = last_sampled_idx + horizon
                    target_idx_absolute = test_start_idx + target_idx_in_test
                    all_target_indices.append(target_idx_absolute)
                    input_window = inputs[:, i, :].detach().cpu().numpy().squeeze()
                    all_input_windows.append(input_window)
            else:
                for i in range(len(preds_np)):
                    all_target_indices.append(None)
                    input_window = inputs[:, i, :].detach().cpu().numpy().squeeze()
                    all_input_windows.append(input_window)
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    return all_predictions, all_targets, all_target_indices, all_input_windows


def train_with_params(
    args, omega_value, lambda_value=None, sweep_idx=None, sweep_type=None
):
    lambda_param = lambda_value if lambda_value is not None else args.lambda_param

    from models import SLON

    model = SLON(
            1,
            args.num_hidden,
            1,
            args.h,
            args.alpha,
            omega_value,
            args.gamma,
            lambda_param=lambda_param,
            gamma_real=args.gamma_real,
            gamma_imag=args.gamma_imag,
        )

    mse_loss_fn = torch.nn.MSELoss()
    loss_fn = NormalizedErrorLoss(epsilon=1e-3, max_ratio=100.0, min_denominator=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    warmup_epochs = max(5, args.epochs // 20)
    cosine_epochs = args.epochs - warmup_epochs
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / cosine_epochs
            progress_powered = progress ** args.lr_decay_power
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress_powered))
            return max(cosine_decay, args.min_lr_ratio)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if sweep_idx is not None:
        if sweep_type == "omega":
            sweep_suffix = f"omega{omega_value:.6f}"
        elif sweep_type == "lambda":
            sweep_suffix = f"lambda{lambda_param:.6f}"
        else:
            sweep_suffix = None
        output_dir = make_run_dir(PROJECT_ROOT, TASK, timestamp, sweep_idx, sweep_suffix)
    else:
        output_dir = make_run_dir(PROJECT_ROOT, TASK, timestamp)

    fh_log = open(f"{output_dir}/log.txt", "a")
    fh_log.write("=" * 60 + "\n")
    fh_log.write("Command-line Arguments:\n")
    fh_log.write("=" * 60 + "\n")
    for key, value in sorted(vars(args).items()):
        fh_log.write(f"{key}: {value}\n")
    fh_log.write("=" * 60 + "\n")
    fh_log.write(f"omega: {omega_value:.6f}\n")
    fh_log.write(f"lambda: {lambda_param:.6f}\n")
    fh_log.write(f"Initial learning rate: {args.lr:.2e}\n")
    fh_log.write(f"Warmup epochs: {warmup_epochs}, Cosine annealing epochs: {cosine_epochs}\n")
    fh_log.write(f"LR decay power: {args.lr_decay_power:.3f}, Min LR ratio: {args.min_lr_ratio:.3f}\n")
    fh_log.write(f"Training loss: NormalizedErrorLoss (epsilon=1e-3, max_ratio=100.0, min_denominator=1e-3)\n")
    fh_log.write("=" * 60 + "\n")
    fh_log.flush()
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr

    return model, loss_fn, mse_loss_fn, optimizer, scheduler, fh_log, output_dir, args.batch_size


def run_training(
    args,
    omega_value,
    lambda_value=None,
    sweep_idx=None,
    sweep_type=None,
):
    (
        train_loader,
        val_loader,
        test_loader,
        full_series,
        removed_freqs,
        removed_powers,
    ) = build_dataloaders(
        series_length=args.series_length,
        input_length=args.input_length,
        horizon=args.horizon,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        tau=args.mg_tau,
        delta_t=args.mg_delta_t,
        beta=args.mg_beta,
        gamma_mg=args.mg_gamma,
        n=args.mg_n,
        x0=args.mg_x0,
        seed=args.seed,
        remove_top_n_freqs=args.remove_top_n_freqs,
    )

    model, loss_fn, mse_loss_fn, optimizer, scheduler, fh_log, output_dir, batch_size = train_with_params(
        args, omega_value, lambda_value, sweep_idx, sweep_type
    )
    
    if removed_freqs is not None and len(removed_freqs) > 0:
        freq_info = "Removed frequencies:\n"
        for i, (freq, power) in enumerate(zip(removed_freqs, removed_powers)):
            freq_info += f"  {i+1}. Frequency: {freq:.6f}, Power: {power:.2e}\n"
        print(freq_info)
        fh_log.write(freq_info)
        fh_log.flush()

    example_batch = next(iter(test_loader))
    if len(example_batch) == 4:
        example_inputs, example_targets, example_sampled_indices, _ = example_batch
        example_sampled_idx = example_sampled_indices[0].item()
    elif len(example_batch) == 3:
        example_inputs, example_targets, example_sampled_indices = example_batch
        example_sampled_idx = example_sampled_indices[0].item()
    else:
        example_inputs, example_targets = example_batch
        example_sampled_idx = 0
    example_input = example_inputs[0]
    example_target = example_targets[0]
    example_signal_inputs = example_input.unsqueeze(0).permute(1, 0, 2)
    
    tau_steps = int(args.mg_tau / args.mg_delta_t)

    best_val_normalized = float("inf")
    best_test_normalized = float("inf")
    best_val_mse = float("inf")
    best_test_mse = float("inf")

    train_losses = []
    val_losses = []
    test_losses = []
    val_r2_scores = []
    test_r2_scores = []
    val_normalized_errors = []
    test_normalized_errors = []
    grad_norms = []
    parameters_history = []

    if not args.skip_epoch_plots:
        print("Generating initial plots before training...")
        model.eval()
        with torch.no_grad():
            ex_batch = example_input.unsqueeze(0)
            ex_batch = ex_batch.permute(1, 0, 2)
            ex_out = model(ex_batch)
            ex_pred = ex_out["output"][0, 0].item()

        window = example_input.squeeze(-1).detach().cpu().numpy()
        target_val = example_target.squeeze().item()

        total_len = int(args.series_length * (1 - args.test_fraction - args.val_fraction))
        val_len = int(args.series_length * args.val_fraction)
        test_start = total_len + val_len
        
        train_series = full_series[:total_len]
        test_series = full_series[test_start:]
        
        train_sampled_indices = np.arange(0, total_len, 1)
        test_sampled_indices = np.arange(test_start, len(full_series), 1)
        
        example_sampled_idx_full = test_start + example_sampled_idx
        start_idx = max(0, example_sampled_idx_full - args.input_length + 1)
        example_input_indices = np.arange(start_idx, example_sampled_idx_full + 1)
        
        example_target_idx = example_sampled_idx_full + args.horizon
        if example_target_idx >= len(full_series):
            example_target_idx = len(full_series) - 1

        all_test_preds, all_test_targets, all_test_target_indices, all_test_input_windows = collect_all_test_predictions(
            test_loader, model, batch_size, test_start, args.horizon
        )
        
        plot_mackey_glass_snapshots(
            output_dir=output_dir,
            full_series=full_series,
            example_input_indices=example_input_indices,
            example_target_idx=example_target_idx,
            ex_pred=ex_pred,
            example_sampled_idx_full=example_sampled_idx_full,
            start_idx=start_idx,
            tau_steps=tau_steps,
            train_sampled_indices=train_sampled_indices,
            test_start=test_start,
            args=args,
            all_test_preds=all_test_preds,
            all_test_targets=all_test_targets,
            all_test_target_indices=all_test_target_indices,
            is_initial=True,
        )

        try:
            plot_signal_stages_for_example(
                model,
                example_signal_inputs,
                output_dir,
                epoch=None,
                input_mode="scalar",
                task_type="regression",
                target_value=float(example_target.squeeze().item()),
                num_units_plot=min(5, args.num_hidden),
                raw_input_label="MG value",
                title_suffix="initial (before training)",
            )
        except Exception as e:
            print(f"Warning: Failed to generate initial signal stage plots: {e}")

        print("Initial plots generated.")

    scatter_xlim = None
    scatter_ylim = None
    if not args.skip_epoch_plots:
        all_test_targets_for_limits = []
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) >= 2:
                    targets = batch[1]
                    all_test_targets_for_limits.append(targets.detach().cpu().numpy().reshape(-1))
        if all_test_targets_for_limits:
            all_test_targets_flat = np.concatenate(all_test_targets_for_limits)
            scatter_min = np.min(all_test_targets_flat)
            scatter_max = np.max(all_test_targets_flat)
            scatter_margin = (scatter_max - scatter_min) * 0.05
            scatter_xlim = [scatter_min - scatter_margin, scatter_max + scatter_margin]
            scatter_ylim = [scatter_min - scatter_margin, scatter_max + scatter_margin]

    total_len = int(args.series_length * (1 - args.test_fraction - args.val_fraction))
    val_len = int(args.series_length * args.val_fraction)
    test_start = total_len + val_len
    pred_ymin = np.min(full_series[test_start:])
    pred_ymax = np.max(full_series[test_start:])
    pred_ymargin = (pred_ymax - pred_ymin) * 0.1
    pred_ylim = [pred_ymin - pred_ymargin, pred_ymax + pred_ymargin]

    param_str = f"omega={omega_value:.6f}"
    if lambda_value is not None:
        param_str += f", lambda={lambda_value:.6f}"

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        tqdm.write(f"epoch {epoch} ({param_str})")
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        count = 0
        epoch_grad_norms = []

        for batch in train_loader:
            if len(batch) == 4:
                inputs, targets, _, prev_values = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
                prev_values = None
            else:
                inputs, targets = batch
                prev_values = None
            inputs = inputs.permute(1, 0, 2)
            optimizer.zero_grad()
            out = model(inputs)
            preds = out["output"]
            
            if prev_values is not None and isinstance(loss_fn, NormalizedErrorLoss):
                loss = loss_fn(preds, targets, prev_values)
            else:
                loss = mse_loss_fn(preds, targets)
            
            mse_batch = mse_loss_fn(preds, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(
                    f"ERROR: NaN/Inf loss detected at epoch {epoch}. Stopping training."
                )
                fh_log.write(
                    f"ERROR: Training stopped at epoch {epoch} due to NaN/Inf loss\n"
                )
                fh_log.flush()
                fh_log.close()
                raise ValueError("NaN/Inf loss detected - training stopped")

            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                tqdm.write(
                    f"ERROR: NaN/Inf gradient detected at epoch {epoch}. Skipping batch."
                )
                optimizer.zero_grad()
                continue
            
            optimizer.step()

            batch_size_current = inputs.size(1)
            epoch_loss += loss.item() * batch_size_current
            epoch_mse += mse_batch.item() * batch_size_current
            count += batch_size_current
            epoch_grad_norms.append(grad_norm.item())

        avg_train_mse = epoch_mse / max(count, 1)
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        grad_norms.append(avg_grad_norm)

        val_mse = evaluate_model(
            val_loader, model, mse_loss_fn, batch_size_test=batch_size, mse_loss_fn=mse_loss_fn
        )
        test_mse = evaluate_model(
            test_loader, model, mse_loss_fn, batch_size_test=batch_size, mse_loss_fn=mse_loss_fn
        )
        
        val_normalized = evaluate_normalized_error(val_loader, model, batch_size_test=batch_size)
        test_normalized = evaluate_normalized_error(test_loader, model, batch_size_test=batch_size)
        
        avg_train_normalized = evaluate_normalized_error(train_loader, model, batch_size_test=batch_size)

        val_r2 = evaluate_r2(val_loader, model, batch_size_test=batch_size)
        test_r2 = evaluate_r2(test_loader, model, batch_size_test=batch_size)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch == 0 or (epoch + 1) % 10 == 0 or current_lr < optimizer.param_groups[0].get('initial_lr', args.lr) * 0.1:
            fh_log.write(f"Epoch {epoch}: Learning rate = {current_lr:.2e}\n")
            fh_log.flush()

        train_losses.append(avg_train_mse)
        val_losses.append(val_mse)
        test_losses.append(test_mse)
        val_r2_scores.append(val_r2)
        test_r2_scores.append(test_r2)
        val_normalized_errors.append(val_normalized)
        test_normalized_errors.append(test_normalized)

        if not args.skip_epoch_plots:
            plot_regression_metrics(
                train_losses, val_losses, test_losses,
                test_r2_scores=test_r2_scores,
                test_normalized_errors=test_normalized_errors,
                val_r2_scores=val_r2_scores,
                val_normalized_errors=val_normalized_errors,
                output_dir=output_dir
            )

        metrics_file = f"{output_dir}/metrics.json"
        metrics_data = {
            "epoch": epoch,
            "train_normalized_error": float(avg_train_normalized),
            "train_mse": float(avg_train_mse),
            "val_normalized_error": float(val_normalized) if np.isfinite(val_normalized) else None,
            "test_normalized_error": float(test_normalized) if np.isfinite(test_normalized) else None,
            "val_mse": float(val_mse),
            "test_mse": float(test_mse),
            "val_r2": float(val_r2),
            "test_r2": float(test_r2),
            "avg_grad_norm": float(avg_grad_norm) if np.isfinite(avg_grad_norm) else None,
            "learning_rate": float(current_lr),
        }
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
            all_metrics.append(metrics_data)
        else:
            all_metrics = [metrics_data]
        
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        model_params = extract_model_parameters(model, "sl")
        param_stats = compute_parameter_statistics(model_params)
        parameters_history.append({
            "epoch": epoch,
            "params": model_params,
            "stats": param_stats
        })
        
        params_file = f"{output_dir}/parameters.json"
        with open(params_file, "w") as f:
            json.dump(parameters_history, f, indent=2)
        
        if not args.skip_epoch_plots:
            plot_mackey_glass_parameter_analysis(parameters_history, output_dir, epoch, fh_log=fh_log)

            model.eval()
            with torch.no_grad():
                ex_batch = example_input.unsqueeze(0)
                ex_batch = ex_batch.permute(1, 0, 2)
                ex_out = model(ex_batch)
                ex_pred = ex_out["output"][0, 0].item()

            window = example_input.squeeze(-1).detach().cpu().numpy()
            target_val = example_target.squeeze().item()

            total_len = int(args.series_length * (1 - args.test_fraction - args.val_fraction))
            val_len = int(args.series_length * args.val_fraction)
            test_start = total_len + val_len
            
            train_series = full_series[:total_len]
            test_series = full_series[test_start:]
            
            train_sampled_indices = np.arange(0, total_len, 1)
            test_sampled_indices = np.arange(test_start, len(full_series), 1)
            
            example_sampled_idx_full = test_start + example_sampled_idx
            start_idx = max(0, example_sampled_idx_full - args.input_length + 1)
            example_input_indices = np.arange(start_idx, example_sampled_idx_full + 1)
            
            example_target_idx = example_sampled_idx_full + args.horizon
            if example_target_idx >= len(full_series):
                example_target_idx = len(full_series) - 1

            all_test_preds, all_test_targets, all_test_target_indices, all_test_input_windows = collect_all_test_predictions(
                test_loader, model, batch_size, test_start, args.horizon
            )
            
            ep_dir = epoch_dir(output_dir, epoch)
            plot_epoch_weight_heatmaps(parameters_history, ep_dir, epoch)
            plot_mackey_glass_snapshots(
                output_dir=ep_dir,
                full_series=full_series,
                example_input_indices=example_input_indices,
                example_target_idx=example_target_idx,
                ex_pred=ex_pred,
                example_sampled_idx_full=example_sampled_idx_full,
                start_idx=start_idx,
                tau_steps=tau_steps,
                train_sampled_indices=train_sampled_indices,
                test_start=test_start,
                args=args,
                all_test_preds=all_test_preds,
                all_test_targets=all_test_targets,
                all_test_target_indices=all_test_target_indices,
                epoch=epoch,
                scatter_xlim=scatter_xlim,
                scatter_ylim=scatter_ylim,
            )

            try:
                plot_signal_stages_for_example(
                    model,
                    example_signal_inputs,
                    ep_dir,
                    epoch,
                    input_mode="scalar",
                    task_type="regression",
                    target_value=float(example_target.squeeze().item()),
                    num_units_plot=min(5, args.num_hidden),
                    raw_input_label="MG value",
                    title_suffix=f"target = {float(example_target.squeeze().item()):.4f}",
                )
            except Exception as e:
                tqdm.write(f"Warning: Failed to generate signal stage plots at epoch {epoch}: {e}")

        is_best = val_normalized < best_val_normalized
        if is_best:
            best_val_normalized = val_normalized
            best_test_normalized = test_normalized
            best_val_mse = val_mse
            best_test_mse = test_mse

        msg = (
            f"epoch {epoch}: train_normalized_error: {avg_train_normalized:.6f}, train_mse: {avg_train_mse:.6f}, "
            f"val_normalized_error: {val_normalized:.6f}, test_normalized_error: {test_normalized:.6f}, "
            f"val_mse: {val_mse:.6f}, test_mse: {test_mse:.6f}, "
            f"val_r2: {val_r2:.6f}, test_r2: {test_r2:.6f}, "
            f"grad_norm: {avg_grad_norm:.6f}, lr: {current_lr:.2e}"
        )
        if val_normalized == best_val_normalized:
            msg += " [BEST]"
        fh_log.write(msg + "\n")
        fh_log.flush()
        tqdm.write(msg)

        save_training_checkpoint(model, output_dir, is_best=is_best)
        tqdm.write(f'wrote checkpoint last_model.pt{" + best_model.pt" if is_best else ""}')

        if not args.skip_epoch_plots and epoch == args.epochs - 1:
            promote_epoch_artifacts(ep_dir, output_dir, {
                f"mg_pred_epoch{epoch:02d}.png": "mg_pred.png",
                f"mg_pred_epoch{epoch:02d}_zoom.png": "mg_pred_zoom.png",
                f"all_predictions_epoch{epoch:02d}.png": "all_predictions.png",
                f"scatter_epoch{epoch:02d}.png": "scatter.png",
                f"predictions_on_test_epoch{epoch:02d}.png": "predictions_on_test.png",
                f"predictions_on_test_epoch{epoch:02d}_avg.png": "predictions_on_test_avg.png",
            })
        
        if args.analyze_manifold and not args.skip_epoch_plots:
            try:
                tqdm.write(f"\nRunning manifold dimension analysis at epoch {epoch}...")
                manifold_results_epoch = analyze_manifold_dimension(
                    test_loader,
                    model,
                    "sl",
                    ep_dir,
                    epoch=epoch,
                    batch_size_test=batch_size,
                    max_samples=5000,
                    variance_threshold=0.95
                )
                if manifold_results_epoch is not None:
                    tqdm.write(f"  PCA effective dim: {manifold_results_epoch['effective_dim_pca']}, "
                             f"Correlation dim: {manifold_results_epoch['correlation_dim']:.4f}" 
                             if manifold_results_epoch['correlation_dim'] is not None 
                             else f"  PCA effective dim: {manifold_results_epoch['effective_dim_pca']}")
            except Exception as e:
                tqdm.write(f"Warning: Manifold dimension analysis failed at epoch {epoch}: {e}")


    if len(parameters_history) > 0 and not args.skip_epoch_plots:
        create_mackey_glass_gifs(output_dir)
    

    msg = f"best test_normalized_error: {best_test_normalized:.6f} (val_normalized_error: {best_val_normalized:.6f}), best test_mse: {best_test_mse:.6f} (val_mse: {best_val_mse:.6f})"
    fh_log.write(msg + "\n")
    fh_log.flush()
    
    if args.analyze_manifold and not args.skip_epoch_plots:
        print("\n" + "=" * 60)
        print("Computing final manifold dimension analysis...")
        print("=" * 60 + "\n")
        
        try:
            manifold_results = analyze_manifold_dimension(
                test_loader, 
                model, 
                "sl", 
                output_dir,
                epoch=None,
                batch_size_test=batch_size,
                max_samples=10000,
                variance_threshold=0.95
            )
            
            if manifold_results is not None:
                manifold_file = f"{output_dir}/manifold_dimension_results.json"
                with open(manifold_file, "w") as f:
                    json.dump(manifold_results, f, indent=2)
                
                fh_log.write("\n" + "=" * 60 + "\n")
                fh_log.write("Manifold Dimension Analysis Results:\n")
                fh_log.write("=" * 60 + "\n")
                fh_log.write(f"PCA effective dimension (95% variance): {manifold_results['effective_dim_pca']}\n")
                fh_log.write(f"Correlation dimension: {manifold_results['correlation_dim']:.4f}\n" if manifold_results['correlation_dim'] is not None else "Correlation dimension: N/A\n")
                fh_log.write(f"State space dimension: {manifold_results['state_dim']}\n")
                fh_log.write(f"Number of samples analyzed: {manifold_results['n_samples']}\n")
                fh_log.write(f"Explained variance at 95% threshold: {manifold_results['explained_variance_95']:.4f}\n")
                fh_log.write("=" * 60 + "\n")
                fh_log.flush()
                
                print(f"Manifold dimension analysis complete!")
                print(f"  PCA effective dimension: {manifold_results['effective_dim_pca']}")
                if manifold_results['correlation_dim'] is not None:
                    print(f"  Correlation dimension: {manifold_results['correlation_dim']:.4f}")
                print(f"  State space dimension: {manifold_results['state_dim']}")
                print(f"  Results saved to: {manifold_file}")
            else:
                fh_log.write("\nWarning: Manifold dimension analysis failed or returned no results\n")
                fh_log.flush()
                print("Warning: Manifold dimension analysis failed or returned no results")
        except Exception as e:
            import traceback
            error_msg = f"Error during manifold dimension analysis: {e}\n{traceback.format_exc()}"
            fh_log.write(f"\n{error_msg}\n")
            fh_log.flush()
            print(f"Error during manifold dimension analysis: {e}")
            tqdm.write(traceback.format_exc())
        
        # Collect and save final states with animation
        try:
            print("\n" + "=" * 60)
            print("Collecting final states and creating animation...")
            print("=" * 60 + "\n")
            collect_and_save_final_states(
                test_loader,
                model,
                "sl",
                output_dir,
                batch_size_test=batch_size,
                max_samples=50000,
                is_imdb=False
            )
        except Exception as e:
            import traceback
            print(f"Warning: Failed to collect final states or create animation: {e}")
            fh_log.write(f"\nWarning: Failed to collect final states: {e}\n")
            fh_log.flush()
    
    fh_log.close()

    if lambda_value is not None:
        return {
            "omega": omega_value,
            "lambda": lambda_value,
            "best_val_normalized_error": best_val_normalized,
            "best_test_normalized_error": best_test_normalized,
            "best_val_mse": best_val_mse,
            "best_test_mse": best_test_mse,
        }
    else:
        return {
            "omega": omega_value,
            "best_val_normalized_error": best_val_normalized,
            "best_test_normalized_error": best_test_normalized,
            "best_val_mse": best_val_mse,
            "best_test_mse": best_test_mse,
        }


def main():
    parser = argparse.ArgumentParser(
        description="SLON training script for Mackey-Glass prediction"
    )
    parser.add_argument("--num-hidden", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--omega", type=float, default=0.224)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--lambda-param", type=float, default=-0.04)
    parser.add_argument("--gamma-real", type=float, default=-0.05)
    parser.add_argument("--gamma-imag", type=float, default=0.1)

    parser.add_argument("--sweep-omega", action="store_true")
    parser.add_argument("--omega-min", type=float, default=None)
    parser.add_argument("--omega-max", type=float, default=None)
    parser.add_argument("--omega-steps", type=int, default=10)
    parser.add_argument("--sweep-lambda", action="store_true")
    parser.add_argument("--lambda-min", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--lambda-steps", type=int, default=10)

    parser.add_argument("--series-length", type=int, default=20000)
    parser.add_argument("--input-length", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)

    parser.add_argument("--mg-tau", type=float, default=17.0)
    parser.add_argument("--mg-delta-t", type=float, default=1.0)
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--mg-gamma", type=float, default=0.1)
    parser.add_argument("--mg-n", type=float, default=10.0)
    parser.add_argument("--mg-x0", type=float, default=1.2)
    parser.add_argument("--remove-top-n-freqs", type=int, default=0,
                        help="Number of top frequencies to remove from time series (increases difficulty, default: 0)")

    parser.add_argument("--lr-decay-power", type=float, default=1.0,
                        help="Power factor for cosine decay (lower = slower decay, default: 1.0)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0,
                        help="Minimum LR as fraction of initial LR (default: 0.0)")
    parser.add_argument("--analyze-manifold", action="store_true",
                        help="Enable manifold dimension analysis (runs at end of training and every 10 epochs)", default=True)
    parser.add_argument("--skip-epoch-plots", action="store_true",
                        help="Skip per-epoch plots and manifold analysis (metrics still saved)")

    args = parser.parse_args()

    if args.sweep_omega and args.sweep_lambda:
        raise ValueError("Cannot sweep both omega and lambda simultaneously.")

    torch.manual_seed(args.seed)

    if args.sweep_omega:
        if args.omega_min is None or args.omega_max is None:
            raise ValueError(
                "--omega-min and --omega-max must be specified when --sweep-omega is enabled"
            )

        omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_steps)
        print(
            f"Starting omega sweep: {args.omega_steps} steps from {args.omega_min:.6f} to {args.omega_max:.6f}"
        )
        print(f"Omega values: {omega_values}")

        sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_results = []

        for sweep_idx, omega_val in enumerate(omega_values):
            print("\n" + "=" * 60)
            print(
                f"Sweep iteration {sweep_idx + 1}/{args.omega_steps}: omega = {omega_val:.6f}"
            )
            print("=" * 60 + "\n")

            result = run_training(
                args, omega_val, sweep_idx=sweep_idx, sweep_type="omega"
            )
            sweep_results.append(result)

            print(
                f"\nCompleted sweep {sweep_idx + 1}/{args.omega_steps}: omega={result['omega']:.6f}, "
                f"val_normalized_error={result['best_val_normalized_error']:.6f}, test_normalized_error={result['best_test_normalized_error']:.6f}, "
                f"val_mse={result['best_val_mse']:.6f}, test_mse={result['best_test_mse']:.6f}\n"
            )

        sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, "omega", sweep_timestamp)
        with open(sweep_summary_file, "w") as f:
            f.write("Omega Sweep Summary (Mackey-Glass)\n")
            f.write(f"Timestamp: {sweep_timestamp}\n")
            f.write(f"Range: {args.omega_min:.6f} to {args.omega_max:.6f}\n")
            f.write(f"Steps: {args.omega_steps}\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"{'Omega':<15} {'Val Norm Error':<15} {'Test Norm Error':<15} {'Val MSE':<15} {'Test MSE':<15}\n"
            )
            f.write("-" * 90 + "\n")

            best_overall = min(sweep_results, key=lambda x: x["best_val_normalized_error"])

            for result in sweep_results:
                marker = " <-- BEST" if result == best_overall else ""
                f.write(
                    f"{result['omega']:<15.6f} {result['best_val_normalized_error']:<15.6f} "
                    f"{result['best_test_normalized_error']:<15.6f} {result['best_val_mse']:<15.6f} "
                    f"{result['best_test_mse']:<15.6f}{marker}\n"
                )

            f.write("=" * 90 + "\n")
            f.write(
                f"Best overall: omega={best_overall['omega']:.6f}, "
                f"val_normalized_error={best_overall['best_val_normalized_error']:.6f}, "
                f"test_normalized_error={best_overall['best_test_normalized_error']:.6f}, "
                f"val_mse={best_overall['best_val_mse']:.6f}, "
                f"test_mse={best_overall['best_test_mse']:.6f}\n"
            )

        print("\n" + "=" * 60)
        print(f"Sweep complete! Summary saved to {sweep_summary_file}")
        print(
            f"Best result: omega={best_overall['omega']:.6f}, "
            f"val_normalized_error={best_overall['best_val_normalized_error']:.6f}, "
            f"test_normalized_error={best_overall['best_test_normalized_error']:.6f}, "
            f"val_mse={best_overall['best_val_mse']:.6f}, "
            f"test_mse={best_overall['best_test_mse']:.6f}"
        )
        print("=" * 60 + "\n")
    elif args.sweep_lambda:
        if args.lambda_min is None or args.lambda_max is None:
            raise ValueError(
                "--lambda-min and --lambda-max must be specified when --sweep-lambda is enabled"
            )

        lambda_values = np.linspace(
            args.lambda_min, args.lambda_max, args.lambda_steps
        )
        print(
            f"Starting lambda sweep: {args.lambda_steps} steps from {args.lambda_min:.6f} to {args.lambda_max:.6f}"
        )
        print(f"Lambda values: {lambda_values}")

        sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_results = []

        for sweep_idx, lambda_val in enumerate(lambda_values):
            print("\n" + "=" * 60)
            print(
                f"Sweep iteration {sweep_idx + 1}/{args.lambda_steps}: lambda = {lambda_val:.6f}"
            )
            print("=" * 60 + "\n")

            result = run_training(
                args,
                args.omega,
                lambda_value=lambda_val,
                sweep_idx=sweep_idx,
                sweep_type="lambda",
            )
            sweep_results.append(result)

            print(
                f"\nCompleted sweep {sweep_idx + 1}/{args.lambda_steps}: lambda={result['lambda']:.6f}, "
                f"val_normalized_error={result['best_val_normalized_error']:.6f}, test_normalized_error={result['best_test_normalized_error']:.6f}, "
                f"val_mse={result['best_val_mse']:.6f}, test_mse={result['best_test_mse']:.6f}\n"
            )

        sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, "lambda", sweep_timestamp)
        with open(sweep_summary_file, "w") as f:
            f.write("Lambda Sweep Summary (Mackey-Glass)\n")
            f.write(f"Timestamp: {sweep_timestamp}\n")
            f.write(f"Range: {args.lambda_min:.6f} to {args.lambda_max:.6f}\n")
            f.write(f"Steps: {args.lambda_steps}\n")
            f.write("=" * 75 + "\n")
            f.write(
                f"{'Lambda':<15} {'Val Norm Error':<15} {'Test Norm Error':<15} {'Val MSE':<15} {'Test MSE':<15}\n"
            )
            f.write("-" * 90 + "\n")

            best_overall = min(sweep_results, key=lambda x: x["best_val_normalized_error"])

            for result in sweep_results:
                marker = " <-- BEST" if result == best_overall else ""
                f.write(
                    f"{result['lambda']:<15.6f} {result['best_val_normalized_error']:<15.6f} "
                    f"{result['best_test_normalized_error']:<15.6f} {result['best_val_mse']:<15.6f} "
                    f"{result['best_test_mse']:<15.6f}{marker}\n"
                )

            f.write("=" * 90 + "\n")
            f.write(
                f"Best overall: lambda={best_overall['lambda']:.6f}, "
                f"val_normalized_error={best_overall['best_val_normalized_error']:.6f}, "
                f"test_normalized_error={best_overall['best_test_normalized_error']:.6f}, "
                f"val_mse={best_overall['best_val_mse']:.6f}, "
                f"test_mse={best_overall['best_test_mse']:.6f}\n"
            )

        print("\n" + "=" * 60)
        print(f"Sweep complete! Summary saved to {sweep_summary_file}")
        print(
            f"Best result: lambda={best_overall['lambda']:.6f}, "
            f"val_normalized_error={best_overall['best_val_normalized_error']:.6f}, "
            f"test_normalized_error={best_overall['best_test_normalized_error']:.6f}, "
            f"val_mse={best_overall['best_val_mse']:.6f}, "
            f"test_mse={best_overall['best_test_mse']:.6f}"
        )
        print("=" * 60 + "\n")
    else:
        result = run_training(args, args.omega)
        print(
            f"best test_normalized_error: {result['best_test_normalized_error']:.6f} "
            f"(val_normalized_error: {result['best_val_normalized_error']:.6f}), "
            f"best test_mse: {result['best_test_mse']:.6f} "
            f"(val_mse: {result['best_val_mse']:.6f})"
        )


if __name__ == "__main__":
    main()


