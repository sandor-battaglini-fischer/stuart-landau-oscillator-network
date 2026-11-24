import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
TASK = 'mackey_glass'

from utils.device import resolve_device, seed_all
from utils.manifold_dimension_analysis import analyze_manifold_dimension, run_final_manifold_analysis
from utils.model_factory import build_oscillator, manifold_dynamics_type
from utils.plotting_utils import (
    create_mackey_glass_gifs,
    plot_epoch_weight_heatmaps,
    plot_mackey_glass_parameter_analysis,
    plot_mackey_glass_snapshots,
    plot_regression_metrics,
    plot_signal_stages_for_example,
)
from utils.run_dirs import (
    atomic_write_json,
    epoch_dir,
    make_run_dir,
    promote_epoch_artifacts,
    save_initial_checkpoint,
    save_training_checkpoint,
    write_log_header,
)
from utils.slon_analysis import compute_parameter_statistics, extract_model_parameters


class MackeyGlassDataset(Dataset):
    """Sliding windows over a series; item = (window, target, last index, last value)."""

    def __init__(self, series, input_length, horizon, sample_step):
        self.series = series.astype(np.float32)
        self.input_length = input_length
        self.horizon = horizon
        self.sample_step = int(sample_step)
        self.sampled_indices = np.arange(0, len(series), self.sample_step, dtype=np.int64)

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        window = self.sampled_indices[max(0, idx - self.input_length + 1):idx + 1]
        if len(window) < self.input_length:
            pad = np.full(self.input_length - len(window), window[0] if len(window) else 0, dtype=np.int64)
            window = np.concatenate([pad, window])
        last_idx = self.sampled_indices[idx]
        target_idx = min(last_idx + self.horizon, len(self.series) - 1)
        x = torch.from_numpy(self.series[window]).unsqueeze(-1)
        y = torch.tensor([self.series[target_idx]], dtype=torch.float32)
        prev = torch.tensor([self.series[last_idx]], dtype=torch.float32)
        return x, y, last_idx, prev


def generate_mackey_glass(length, tau=17.0, delta_t=1.0, beta=0.2, gamma=0.1, n=10, x0=1.2):
    delay = int(tau / delta_t)
    total_len = length + delay + 1
    x = np.zeros(total_len, dtype=np.float64)
    x[:delay + 1] = x0
    for t in range(delay, total_len - 1):
        x_tau = x[t - delay]
        x[t + 1] = x[t] + delta_t * (beta * x_tau / (1.0 + x_tau ** n) - gamma * x[t])
    return x[delay + 1:]


def remove_top_frequencies(series, delta_t, n_freqs=3):
    """Subtract the n strongest low-frequency (<= 0.2) Fourier components."""
    if n_freqs == 0:
        return series.copy(), np.array([]), np.array([])
    fft_vals = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series), delta_t)
    power = np.abs(fft_vals) ** 2
    half = len(freqs) // 2
    low = np.where(freqs[:half] <= 0.2)[0]
    top = low[np.argsort(power[:half][low])[-n_freqs:]]

    kept = np.zeros_like(fft_vals)
    for idx in top:
        kept[idx] = fft_vals[idx]
        if idx > 0:
            kept[len(freqs) - idx] = fft_vals[len(freqs) - idx]
    residual = series - np.real(np.fft.ifft(kept))
    return residual, freqs[top], power[top]


def split_bounds(series_length, val_fraction, test_fraction):
    train_end = int(series_length * (1 - test_fraction - val_fraction))
    val_end = int(series_length * (1 - test_fraction))
    return train_end, val_end


def build_dataloaders(series_length, input_length, horizon, batch_size, val_fraction, test_fraction,
                      tau, delta_t, beta, gamma_mg, n, x0, seed, remove_top_n_freqs=0):
    series = generate_mackey_glass(series_length, tau=tau, delta_t=delta_t, beta=beta, gamma=gamma_mg, n=n, x0=x0)
    removed_freqs = removed_powers = None
    if remove_top_n_freqs > 0:
        series, removed_freqs, removed_powers = remove_top_frequencies(series, delta_t, n_freqs=remove_top_n_freqs)

    train_end, val_end = split_bounds(len(series), val_fraction, test_fraction)
    datasets = [MackeyGlassDataset(part, input_length, horizon, 1)
                for part in (series[:train_end], series[train_end:val_end], series[val_end:])]
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, series, removed_freqs, removed_powers


class NormalizedErrorLoss(torch.nn.Module):
    """log(1 + |pred - target| / |target - previous|), clamped. Penalises copying the last value."""

    def __init__(self, epsilon=1e-3, max_ratio=100.0, min_denominator=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio
        self.min_denominator = min_denominator

    def forward(self, preds, targets, prev_values):
        denominator = torch.clamp(torch.abs(targets - prev_values), min=self.min_denominator) + self.epsilon
        ratio = torch.clamp(torch.abs(preds - targets) / denominator, 0.0, self.max_ratio)
        return torch.mean(torch.log1p(ratio))


def _device(model):
    return next(model.parameters()).device


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _predict(loader, model):
    """Yields (preds, targets, last_indices, prev_values, inputs) per batch, inputs as (T, B, 1)."""
    model.eval()
    device = _device(model)
    with torch.no_grad():
        for inputs, targets, last_idx, prev in loader:
            inputs = inputs.to(device).permute(1, 0, 2)
            yield model(inputs)['output'], targets.to(device), last_idx, prev.to(device), inputs


def evaluate_mse(loader, model):
    total, count = 0.0, 0
    for preds, targets, _, _, _ in _predict(loader, model):
        total += torch.nn.functional.mse_loss(preds, targets).item() * len(targets)
        count += len(targets)
    return total / max(count, 1)


def evaluate_r2(loader, model):
    preds, targets = [], []
    for p, t, _, _, _ in _predict(loader, model):
        preds.append(p.cpu().numpy().reshape(-1))
        targets.append(t.cpu().numpy().reshape(-1))
    if not preds:
        return 0.0
    y_pred, y_true = np.concatenate(preds), np.concatenate(targets)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0.0 else 1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot


def evaluate_normalized_error(loader, model, epsilon=1e-8):
    errors = [(torch.abs(p - t) / (torch.abs(t - prev) + epsilon)).cpu().numpy().reshape(-1)
              for p, t, _, prev, _ in _predict(loader, model)]
    return float(np.mean(np.concatenate(errors))) if errors else float('inf')


def collect_all_test_predictions(test_loader, model, test_start_idx, horizon):
    preds, targets, target_indices, windows = [], [], [], []
    for p, t, last_idx, _, inputs in _predict(test_loader, model):
        preds.append(p.cpu().numpy().reshape(-1))
        targets.append(t.cpu().numpy().reshape(-1))
        for i, idx in enumerate(last_idx.numpy()):
            target_indices.append(test_start_idx + int(idx) + horizon)
            windows.append(inputs[:, i, :].cpu().numpy().squeeze())
    return np.concatenate(preds), np.concatenate(targets), target_indices, windows


def make_scheduler(optimizer, args):
    """Linear warmup followed by a (power-warped) cosine decay floored at min_lr_ratio."""
    warmup = max(5, args.epochs // 20)
    cosine = args.epochs - warmup

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = ((epoch - warmup) / cosine) ** args.lr_decay_power
        return max(0.5 * (1 + np.cos(np.pi * progress)), args.min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_training(args, return_model=False):
    device = resolve_device(args.device)
    train_loader, val_loader, test_loader, full_series, removed_freqs, removed_powers = build_dataloaders(
        series_length=args.series_length, input_length=args.input_length, horizon=args.horizon,
        batch_size=args.batch_size, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        tau=args.mg_tau, delta_t=args.mg_delta_t, beta=args.mg_beta, gamma_mg=args.mg_gamma,
        n=args.mg_n, x0=args.mg_x0, seed=args.seed, remove_top_n_freqs=args.remove_top_n_freqs,
    )

    model = build_oscillator(
        args.dynamics, 1, args.num_hidden, 1, args.h, args.alpha, args.omega, args.gamma,
        lambda_param=args.lambda_param, gamma_real=args.gamma_real, gamma_imag=args.gamma_imag,
        use_tanh=not args.no_tanh,
    ).to(device)
    loss_fn = NormalizedErrorLoss(epsilon=1e-3, max_ratio=100.0, min_denominator=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = make_scheduler(optimizer, args)

    output_dir = make_run_dir(PROJECT_ROOT, TASK, args.output_dir)
    if not args.sweep_mode:
        save_initial_checkpoint(model, output_dir)
    fh_log = open(os.path.join(output_dir, 'log.txt'), 'a')
    write_log_header(fh_log, dict(sorted(vars(args).items())))

    def log(msg):
        tqdm.write(msg)
        fh_log.write(msg + '\n')
        fh_log.flush()

    if removed_freqs is not None and len(removed_freqs):
        log('Removed frequencies:\n' + '\n'.join(
            f'  {i + 1}. Frequency: {f:.6f}, Power: {p:.2e}' for i, (f, p) in enumerate(zip(removed_freqs, removed_powers))))

    train_end, val_end = split_bounds(args.series_length, args.val_fraction, args.test_fraction)
    test_start = val_end
    plots = None if args.skip_epoch_plots else EpochPlots(args, model, test_loader, full_series, test_start, output_dir)

    best = {'val_r2': -float('inf'), 'test_r2': -float('inf'), 'val_mse': float('inf'), 'test_mse': float('inf'),
            'val_norm': float('inf'), 'test_norm': float('inf'), 'epoch': 0}
    patience_counter = 0
    numerical_error = False
    history = {k: [] for k in ('train_mse', 'val_mse', 'test_mse', 'val_r2', 'test_r2', 'val_norm', 'test_norm')}
    parameters_history = []
    all_metrics = []

    for epoch in tqdm(range(args.epochs), total=args.epochs, disable=args.sweep_mode):
        model.train()
        epoch_mse, count, grad_norms = 0.0, 0, []
        for batch_idx, (inputs, targets, _, prev_values) in enumerate(train_loader):
            inputs = inputs.to(device).permute(1, 0, 2)
            targets, prev_values = targets.to(device), prev_values.to(device)
            optimizer.zero_grad()
            preds = model(inputs)['output']
            loss = loss_fn(preds, targets, prev_values)
            if not torch.isfinite(loss):
                log(f'ERROR: Training stopped at epoch {epoch} due to NaN/Inf loss')
                numerical_error = True
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            epoch_mse += torch.nn.functional.mse_loss(preds, targets).item() * inputs.size(1)
            count += inputs.size(1)
            grad_norms.append(grad_norm.item())
            if args.sweep_mode and batch_idx == 0:
                print(f'epoch {epoch}: first batch done (loss={loss.item():.6f})', flush=True)
        if numerical_error:
            break

        train_mse = epoch_mse / max(count, 1)
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        val_mse, test_mse = evaluate_mse(val_loader, model), evaluate_mse(test_loader, model)
        val_norm, test_norm = evaluate_normalized_error(val_loader, model), evaluate_normalized_error(test_loader, model)
        train_norm = evaluate_normalized_error(train_loader, model)
        val_r2, test_r2 = evaluate_r2(val_loader, model), evaluate_r2(test_loader, model)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        for key, value in zip(history, (train_mse, val_mse, test_mse, val_r2, test_r2, val_norm, test_norm)):
            history[key].append(value)

        all_metrics.append({
            'epoch': epoch, 'train_normalized_error': float(train_norm), 'train_mse': float(train_mse),
            'val_normalized_error': _finite(val_norm), 'test_normalized_error': _finite(test_norm),
            'val_mse': float(val_mse), 'test_mse': float(test_mse), 'val_r2': float(val_r2), 'test_r2': float(test_r2),
            'avg_grad_norm': _finite(avg_grad_norm), 'learning_rate': float(lr),
        })
        atomic_write_json(os.path.join(output_dir, 'metrics.json'), all_metrics)

        if plots is not None:
            plot_regression_metrics(
                history['train_mse'], history['val_mse'], history['test_mse'],
                test_r2_scores=history['test_r2'], test_normalized_errors=history['test_norm'],
                val_r2_scores=history['val_r2'], val_normalized_errors=history['val_norm'], output_dir=output_dir,
            )
            model_params = extract_model_parameters(model, manifold_dynamics_type(args.dynamics))
            parameters_history.append({'epoch': epoch, 'params': model_params,
                                       'stats': compute_parameter_statistics(model_params)})
            atomic_write_json(os.path.join(output_dir, 'parameters.json'), parameters_history)
            plot_mackey_glass_parameter_analysis(parameters_history, output_dir, epoch, fh_log=fh_log)
            ep_dir = epoch_dir(output_dir, epoch)
            plot_epoch_weight_heatmaps(parameters_history, ep_dir, epoch)
            plots.snapshot(ep_dir, epoch)

        # select on val R^2: the normalised error can get worse while MSE/R^2 improve
        is_best = np.isfinite(val_r2) and val_r2 > best['val_r2']
        if is_best:
            best.update(val_r2=val_r2, test_r2=test_r2, val_mse=val_mse, test_mse=test_mse,
                        val_norm=val_norm, test_norm=test_norm, epoch=epoch)
            patience_counter = 0
        else:
            patience_counter += 1

        log(f'epoch {epoch}: train_normalized_error: {train_norm:.6f}, train_mse: {train_mse:.6f}, '
            f'val_normalized_error: {val_norm:.6f}, test_normalized_error: {test_norm:.6f}, '
            f'val_mse: {val_mse:.6f}, test_mse: {test_mse:.6f}, val_r2: {val_r2:.6f}, test_r2: {test_r2:.6f}, '
            f'grad_norm: {avg_grad_norm:.6f}, lr: {lr:.2e}' + (' [BEST]' if is_best else ''))

        if not args.sweep_mode:
            save_training_checkpoint(model, output_dir, is_best=is_best)

        stopped_early = patience_counter >= args.early_stop_patience
        if plots is not None and (epoch == args.epochs - 1 or stopped_early):
            promote_epoch_artifacts(ep_dir, output_dir, {
                f'mg_pred_epoch{epoch:02d}.png': 'mg_pred.png',
                f'mg_pred_epoch{epoch:02d}_zoom.png': 'mg_pred_zoom.png',
                f'all_predictions_epoch{epoch:02d}.png': 'all_predictions.png',
                f'scatter_epoch{epoch:02d}.png': 'scatter.png',
                f'predictions_on_test_epoch{epoch:02d}.png': 'predictions_on_test.png',
                f'predictions_on_test_epoch{epoch:02d}_avg.png': 'predictions_on_test_avg.png',
            })

        if plots is not None and args.analyze_manifold:
            try:
                analyze_manifold_dimension(test_loader, model, manifold_dynamics_type(args.dynamics), ep_dir,
                                           epoch=epoch, batch_size_test=args.batch_size, max_samples=5000,
                                           variance_threshold=0.95)
            except Exception as e:
                tqdm.write(f'Warning: manifold analysis failed at epoch {epoch}: {e}')

        if stopped_early:
            log(f"early stop at epoch {epoch}: best test_r2={best['test_r2']:.4f} "
                f"(best_val_r2={best['val_r2']:.6f}, best_epoch={best['epoch']})")
            break

    if plots is not None and parameters_history:
        create_mackey_glass_gifs(output_dir)

    fh_log.write(f"best test_normalized_error: {best['test_norm']:.6f} (val_normalized_error: {best['val_norm']:.6f}), "
                 f"best test_mse: {best['test_mse']:.6f} (val_mse: {best['val_mse']:.6f})\n")
    if np.isfinite(best['test_r2']):
        fh_log.write(f"best test_r2: {best['test_r2']:.6f} (epoch {best['epoch']})\n")
        print(f"best test_r2: {best['test_r2']:.6f}")
    if numerical_error:
        fh_log.write('WARNING: Training stopped due to numerical error (NaN/Inf)\n')
    fh_log.flush()

    if not all_metrics:
        atomic_write_json(os.path.join(output_dir, 'metrics.json'), [{
            'epoch': -1, 'train_normalized_error': None, 'train_mse': None, 'val_normalized_error': None,
            'test_normalized_error': None, 'val_mse': None, 'test_mse': None, 'val_r2': None,
            'test_r2': 0.0 if numerical_error else None, 'avg_grad_norm': None, 'learning_rate': None,
            'numerical_error': True,
        }])

    if plots is not None and args.analyze_manifold:
        run_final_manifold_analysis(test_loader, model, args.dynamics, output_dir, args.batch_size, fh_log)
    fh_log.close()

    result = {
        'best_val_normalized_error': best['val_norm'], 'best_test_normalized_error': best['test_norm'],
        'best_val_mse': best['val_mse'], 'best_test_mse': best['test_mse'],
        'best_test_r2': float(best['test_r2']) if np.isfinite(best['test_r2']) else 0.0,
        'best_epoch': int(best['epoch']), 'numerical_error': bool(numerical_error), 'output_dir': output_dir,
    }
    if return_model:
        return result, model, test_loader, full_series
    return result


class EpochPlots:
    """Prediction snapshots of one fixed test window plus the whole test segment."""

    def __init__(self, args, model, test_loader, full_series, test_start, output_dir):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.full_series = full_series
        self.test_start = test_start

        inputs, targets, last_idx, _ = next(iter(test_loader))
        self.example_input = inputs[0]
        self.example_target = float(targets[0].item())
        self.example_idx = test_start + int(last_idx[0].item())

        test_targets = np.concatenate([t.numpy().reshape(-1) for _, t, _, _ in test_loader])
        margin = (test_targets.max() - test_targets.min()) * 0.05
        self.scatter_lim = [test_targets.min() - margin, test_targets.max() + margin]

        self.snapshot(output_dir, epoch=None)

    def snapshot(self, out_dir, epoch):
        args = self.args
        self.model.eval()
        with torch.no_grad():
            ex_pred = self.model(self.example_input.unsqueeze(1).to(_device(self.model)))['output'][0, 0].item()
        start_idx = max(0, self.example_idx - args.input_length + 1)
        preds, targets, target_indices, _ = collect_all_test_predictions(
            self.test_loader, self.model, self.test_start, args.horizon)
        plot_mackey_glass_snapshots(
            output_dir=out_dir, full_series=self.full_series,
            example_input_indices=np.arange(start_idx, self.example_idx + 1),
            example_target_idx=min(self.example_idx + args.horizon, len(self.full_series) - 1),
            ex_pred=ex_pred, example_sampled_idx_full=self.example_idx, start_idx=start_idx,
            tau_steps=int(args.mg_tau / args.mg_delta_t),
            train_sampled_indices=np.arange(0, split_bounds(args.series_length, args.val_fraction, args.test_fraction)[0]),
            test_start=self.test_start, args=args, all_test_preds=preds, all_test_targets=targets,
            all_test_target_indices=target_indices,
            is_initial=epoch is None,
            **({} if epoch is None else {'epoch': epoch, 'scatter_xlim': self.scatter_lim, 'scatter_ylim': self.scatter_lim}),
        )
        try:
            plot_signal_stages_for_example(
                self.model, self.example_input.unsqueeze(1), out_dir, epoch, input_mode='scalar',
                task_type='regression', target_value=self.example_target, num_units_plot=min(5, args.num_hidden),
                raw_input_label='MG value',
                title_suffix='initial (before training)' if epoch is None else f'target = {self.example_target:.4f}',
            )
        except Exception as e:
            tqdm.write(f'Warning: signal stage plots failed: {e}')


def build_parser():
    p = argparse.ArgumentParser(description='SLON training on Mackey-Glass prediction')
    p.add_argument('--num-hidden', type=int, default=50)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--h', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.04)
    p.add_argument('--omega', type=float, default=0.224)
    p.add_argument('--gamma', type=float, default=0.01)
    p.add_argument('--lambda-param', type=float, default=-0.04)
    p.add_argument('--gamma-real', type=float, default=-0.05)
    p.add_argument('--gamma-imag', type=float, default=0.1)
    p.add_argument('--dynamics', type=str, default='sl', choices=['sl', 'lo', 'dho'])
    p.add_argument('--no-tanh', action='store_true')
    p.add_argument('--sweep-mode', action='store_true', help='only write log.txt and metrics.json')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--series-length', type=int, default=20000)
    p.add_argument('--input-length', type=int, default=50)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--test-fraction', type=float, default=0.1)
    p.add_argument('--mg-tau', type=float, default=17.0)
    p.add_argument('--mg-delta-t', type=float, default=1.0)
    p.add_argument('--mg-beta', type=float, default=0.2)
    p.add_argument('--mg-gamma', type=float, default=0.1)
    p.add_argument('--mg-n', type=float, default=10.0)
    p.add_argument('--mg-x0', type=float, default=1.2)
    p.add_argument('--remove-top-n-freqs', type=int, default=0, help='subtract the strongest Fourier modes')
    p.add_argument('--lr-decay-power', type=float, default=1.0)
    p.add_argument('--min-lr-ratio', type=float, default=0.0)
    p.add_argument('--analyze-manifold', action='store_true', default=True)
    p.add_argument('--skip-epoch-plots', action='store_true')
    p.add_argument('--early-stop-patience', type=int, default=5, help='epochs without val R^2 improvement')
    p.add_argument('--device', type=str, default='cpu', choices=['auto', 'cpu', 'cuda'])
    return p


def main():
    args = build_parser().parse_args()
    if args.sweep_mode:
        args.skip_epoch_plots = True
    # grid workers run many CPU jobs side by side; one thread each unless told otherwise
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
    device = resolve_device(args.device)
    seed_all(args.seed, device)
    print(f'Using device: {device}')
    result = run_training(args)
    print(f"best test_normalized_error: {result['best_test_normalized_error']:.6f}, "
          f"best test_mse: {result['best_test_mse']:.6f}")


if __name__ == '__main__':
    main()
