import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()

from training.train_mackey_glass import (
    build_dataloaders,
    train_with_params,
    evaluate_model,
    evaluate_r2,
    evaluate_normalized_error,
    NormalizedErrorLoss,
)


def run_training_grid(
    args,
    omega_value,
    lambda_value=None,
    gamma_real_value=None,
    gamma_imag_value=None,
    run_idx=0,
):
    (
        train_loader,
        val_loader,
        test_loader,
        _,
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

    lambda_param = lambda_value if lambda_value is not None else args.lambda_param
    gamma_real_param = gamma_real_value if gamma_real_value is not None else args.gamma_real
    gamma_imag_param = gamma_imag_value if gamma_imag_value is not None else args.gamma_imag
    if args.dynamics == "sl":
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
            gamma_real=gamma_real_param,
            gamma_imag=gamma_imag_param,
        )
    else:
        from models import SLON

        model = SLON(1, args.num_hidden, 1, args.h, args.alpha, omega_value, args.gamma)

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
    label_parts = [f"omega{omega_value:.6f}"]
    if lambda_value is not None:
        label_parts.append(f"lambda{lambda_value:.6f}")
    if gamma_real_value is not None:
        label_parts.append(f"gr{gamma_real_value:.3f}")
    if gamma_imag_value is not None:
        label_parts.append(f"gi{gamma_imag_value:.3f}")
    output_dir = f"results/mackey_glass/{timestamp}_grid{run_idx:03d}_" + "_".join(label_parts)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_path = Path(output_dir) / "log.txt"
    best_val_normalized = float("inf")
    best_test_normalized = float("inf")
    best_val_mse = float("inf")
    best_test_mse = float("inf")
    best_test_r2 = -float("inf")
    best_epoch = 0
    numerical_error = False

    with open(log_path, "a") as fh_log:
        fh_log.write("=" * 60 + "\n")
        fh_log.write("Command-line Arguments:\n")
        fh_log.write("=" * 60 + "\n")
        for key, value in sorted(vars(args).items()):
            fh_log.write(f"{key}: {value}\n")
        fh_log.write("=" * 60 + "\n")
        fh_log.write(f"omega: {omega_value:.6f}\n")
        if lambda_value is not None:
            fh_log.write(f"lambda: {lambda_param:.6f}\n")
        if gamma_real_value is not None:
            fh_log.write(f"gamma_real: {gamma_real_value:.6f}\n")
        if gamma_imag_value is not None:
            fh_log.write(f"gamma_imag: {gamma_imag_value:.6f}\n")
        fh_log.write(f"Initial learning rate: {args.lr:.2e}\n")
        fh_log.write(f"Warmup epochs: {warmup_epochs}, Cosine annealing epochs: {cosine_epochs}\n")
        fh_log.write(f"LR decay power: {args.lr_decay_power:.3f}, Min LR ratio: {args.min_lr_ratio:.3f}\n")
        fh_log.write(f"Training loss: NormalizedErrorLoss (epsilon=1e-3, max_ratio=100.0, min_denominator=1e-3)\n")
        if removed_freqs is not None and len(removed_freqs) > 0:
            fh_log.write("Removed frequencies:\n")
            for i, (freq, power) in enumerate(zip(removed_freqs, removed_powers)):
                fh_log.write(f"  {i+1}. Frequency: {freq:.6f}, Power: {power:.2e}\n")
            print("Removed frequencies:")
            for i, (freq, power) in enumerate(zip(removed_freqs, removed_powers)):
                print(f"  {i+1}. Frequency: {freq:.6f}, Power: {power:.2e}")
        fh_log.write("=" * 60 + "\n")
        fh_log.flush()
        
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = args.lr

        for epoch in tqdm(range(args.epochs), total=args.epochs, desc=f"run {run_idx:03d}"):
            model.train()
            epoch_loss = 0.0
            count = 0

            try:
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
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        numerical_error = True
                        break

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()

                    batch_size_current = inputs.size(1)
                    epoch_loss += loss.item() * batch_size_current
                    count += batch_size_current

                if numerical_error:
                    break

                avg_train_loss = epoch_loss / max(count, 1)

                val_mse = evaluate_model(
                    val_loader, model, mse_loss_fn, batch_size_test=args.batch_size, mse_loss_fn=mse_loss_fn
                )
                test_mse = evaluate_model(
                    test_loader, model, mse_loss_fn, batch_size_test=args.batch_size, mse_loss_fn=mse_loss_fn
                )
                
                val_normalized = evaluate_normalized_error(val_loader, model, batch_size_test=args.batch_size)
                test_normalized = evaluate_normalized_error(test_loader, model, batch_size_test=args.batch_size)
                
                test_r2 = evaluate_r2(
                    test_loader, model, batch_size_test=args.batch_size
                )

                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                if val_normalized < best_val_normalized:
                    best_val_normalized = val_normalized
                    best_test_normalized = test_normalized
                    best_val_mse = val_mse
                    best_test_mse = test_mse
                    best_test_r2 = test_r2
                    best_epoch = epoch

                msg = (
                    f"epoch {epoch}: train_loss: {avg_train_loss:.6f}, "
                    f"val_normalized_error: {val_normalized:.6f}, test_normalized_error: {test_normalized:.6f}, "
                    f"val_mse: {val_mse:.6f}, test_mse: {test_mse:.6f}, "
                    f"test_r2: {test_r2:.6f}, lr: {current_lr:.2e}"
                )
                if val_normalized == best_val_normalized:
                    msg += " [BEST]"
                fh_log.write(msg + "\n")
                fh_log.flush()

            except (RuntimeError, ValueError) as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    numerical_error = True
                    break
                raise

        fh_log.write(
            f"best test_normalized_error: {best_test_normalized:.6f} (val_normalized_error: {best_val_normalized:.6f}), "
            f"best test_mse: {best_test_mse:.6f} (val_mse: {best_val_mse:.6f}, epoch: {best_epoch})\n"
        )
        if numerical_error:
            fh_log.write("WARNING: Training stopped due to numerical error (NaN/Inf)\n")

    result = {
        "omega": float(omega_value),
        "best_val_normalized_error": float(best_val_normalized) if np.isfinite(best_val_normalized) else float('inf'),
        "best_test_normalized_error": float(best_test_normalized) if np.isfinite(best_test_normalized) else float('inf'),
        "best_val_mse": float(best_val_mse),
        "best_test_mse": float(best_test_mse),
        "best_test_r2": float(best_test_r2),
        "best_epoch": int(best_epoch),
        "numerical_error": numerical_error,
    }
    if lambda_value is not None:
        result["lambda"] = float(lambda_value)
    if gamma_real_value is not None:
        result["gamma_real"] = float(gamma_real_value)
    if gamma_imag_value is not None:
        result["gamma_imag"] = float(gamma_imag_value)
    return result


def generate_plots(results, output_dir, omega_fixed=False, timestamp=None, config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not results:
        return
    
    def range_from_config(key, fallback_vals):
        if config and config.get(key):
            start, end, steps = config[key]
            return np.linspace(start, end, int(steps)).tolist()
        return fallback_vals

    omega_vals = range_from_config('omega_range', sorted(set(r['omega'] for r in results)))
    lambda_vals = range_from_config('lambda_range', sorted(set(r.get('lambda', 0.0) for r in results)))
    gamma_real_vals = range_from_config('gamma_real_range', sorted(set(r.get('gamma_real', 0.0) for r in results)))
    gamma_imag_vals = range_from_config('gamma_imag_range', sorted(set(r.get('gamma_imag', 0.0) for r in results)))

    def find_closest_idx(val, vals, tol=1e-6):
        for idx, v in enumerate(vals):
            if abs(v - val) < tol:
                return idx
        return None
    
    valid_results = [r for r in results if not r.get('numerical_error', False)]
    
    if omega_fixed or len(omega_vals) == 1:
        param_pairs = [
            ('lambda', 'gamma_real', lambda_vals, gamma_real_vals, 'gamma_imag', None, gamma_imag_vals, None),
            ('lambda', 'gamma_imag', lambda_vals, gamma_imag_vals, 'gamma_real', None, gamma_real_vals, None),
            ('gamma_real', 'gamma_imag', gamma_real_vals, gamma_imag_vals, 'lambda', None, lambda_vals, None),
        ]
    else:
        param_pairs = [
            ('lambda', 'omega', lambda_vals, omega_vals, 'gamma_real', 'gamma_imag', gamma_real_vals, gamma_imag_vals),
            ('lambda', 'gamma_real', lambda_vals, gamma_real_vals, 'omega', 'gamma_imag', omega_vals, gamma_imag_vals),
            ('lambda', 'gamma_imag', lambda_vals, gamma_imag_vals, 'omega', 'gamma_real', omega_vals, gamma_real_vals),
            ('omega', 'gamma_real', omega_vals, gamma_real_vals, 'lambda', 'gamma_imag', lambda_vals, gamma_imag_vals),
            ('omega', 'gamma_imag', omega_vals, gamma_imag_vals, 'lambda', 'gamma_real', lambda_vals, gamma_real_vals),
            ('gamma_real', 'gamma_imag', gamma_real_vals, gamma_imag_vals, 'lambda', 'omega', lambda_vals, omega_vals),
        ]
    
    def format_val(v, param_name):
        if param_name == 'omega':
            return f'{v:.4f}'
        else:
            return f'{v:.3f}'
    
    for param1_name, param2_name, param1_vals, param2_vals, avg_param1_name, avg_param2_name, avg_param1_vals, avg_param2_vals in param_pairs:
        if len(param1_vals) <= 1 or len(param2_vals) <= 1:
            continue

        def build_and_save_heatmap(current_results, suffix='', title_suffix=''):
            heatmap_data = np.full((len(param2_vals), len(param1_vals)), np.nan)
            count_data = np.zeros((len(param2_vals), len(param1_vals)), dtype=int)
            max_data = np.full((len(param2_vals), len(param1_vals)), np.nan)
            
            for r in current_results:
                param1_val = r[param1_name]
                param2_val = r[param2_name]
                
                idx1 = find_closest_idx(param1_val, param1_vals)
                idx2 = find_closest_idx(param2_val, param2_vals)
                
                if idx1 is not None and idx2 is not None:
                    if r.get('numerical_error', False):
                        r2 = 0.0
                    elif 'best_test_r2' in r:
                        r2 = max(0.0, r['best_test_r2'])
                    else:
                        r2 = 0.0
                    
                    if np.isnan(heatmap_data[idx2, idx1]):
                        heatmap_data[idx2, idx1] = r2
                        max_data[idx2, idx1] = r2
                        count_data[idx2, idx1] = 1
                    else:
                        heatmap_data[idx2, idx1] += r2
                        max_data[idx2, idx1] = max(max_data[idx2, idx1], r2)
                        count_data[idx2, idx1] += 1
            
            heatmap_data_local = np.divide(heatmap_data, count_data, out=np.full_like(heatmap_data, np.nan), where=count_data!=0)
            
            missing_mask = np.isnan(heatmap_data_local)
            
            heatmap_data_percent = heatmap_data_local.copy() * 100.0
            
            valid_data = heatmap_data_percent[~missing_mask]
            
            if len(valid_data) > 0:
                vmax = max(100.0, np.max(valid_data))
            else:
                vmax = 100.0
            
            data_for_plot = np.ma.array(heatmap_data_percent, mask=missing_mask)
            
            threshold = 90.0
            
            cdict = {
                'red': [
                    (0.0, 0.0, 0.0),
                    ((threshold / (2.0 * vmax)), thesis_blue[0], thesis_blue[0]),
                    ((threshold / vmax), ifisc_green[0], ifisc_green[0]),
                    (1.0, thesis_red[0], thesis_red[0])
                ],
                'green': [
                    (0.0, 0.0, 0.0),
                    ((threshold / (2.0 * vmax)), thesis_blue[1], thesis_blue[1]),
                    ((threshold / vmax), ifisc_green[1], ifisc_green[1]),
                    (1.0, thesis_red[1], thesis_red[1])
                ],
                'blue': [
                    (0.0, 0.0, 0.0),
                    ((threshold / (2.0 * vmax)), thesis_blue[2], thesis_blue[2]),
                    ((threshold / vmax), ifisc_green[2], ifisc_green[2]),
                    (1.0, thesis_red[2], thesis_red[2])
                ]
            }
            
            custom_cmap = mcolors.LinearSegmentedColormap('custom_heatmap', cdict, N=256)
            custom_cmap.set_bad((1.0, 1.0, 1.0, 1.0))
            
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(data_for_plot, aspect='auto', origin='lower', cmap=custom_cmap, norm=norm, interpolation='nearest')
            
            num_ticks_x = min(16, len(param1_vals))
            num_ticks_y = min(16, len(param2_vals))
            
            if len(param1_vals) > 16:
                x_indices = np.linspace(0, len(param1_vals) - 1, num_ticks_x, dtype=int)
                ax.set_xticks(x_indices)
                ax.set_xticklabels([format_val(param1_vals[i], param1_name) for i in x_indices], rotation=45, ha='right', fontsize=28)
            else:
                ax.set_xticks(np.arange(len(param1_vals)))
                ax.set_xticklabels([format_val(v, param1_name) for v in param1_vals], rotation=45, ha='right', fontsize=28)
            
            if len(param2_vals) > 16:
                y_indices = np.linspace(0, len(param2_vals) - 1, num_ticks_y, dtype=int)
                ax.set_yticks(y_indices)
                ax.set_yticklabels([format_val(param2_vals[i], param2_name) for i in y_indices], fontsize=28)
            else:
                ax.set_yticks(np.arange(len(param2_vals)))
                ax.set_yticklabels([format_val(v, param2_name) for v in param2_vals], fontsize=28)
            
            param1_label = param1_name.replace('_', ' ').title()
            param2_label = param2_name.replace('_', ' ').title()
            if param1_name == 'lambda':
                param1_label = 'Lambda ($\\lambda$)'
            elif param1_name == 'omega':
                param1_label = 'Omega ($\\omega$)'
            elif param1_name == 'gamma_real':
                param1_label = 'Gamma Real ($\\gamma_r$)'
            elif param1_name == 'gamma_imag':
                param1_label = 'Gamma Imaginary ($\\gamma_i$)'
                
            if param2_name == 'lambda':
                param2_label = 'Lambda ($\\lambda$)'
            elif param2_name == 'omega':
                param2_label = 'Omega ($\\omega$)'
            elif param2_name == 'gamma_real':
                param2_label = 'Gamma Real ($\\gamma_r$)'
            elif param2_name == 'gamma_imag':
                param2_label = 'Gamma Imaginary ($\\gamma_i$)'
            
            ax.set_xlabel(param1_label, fontsize=30)
            ax.set_ylabel(param2_label, fontsize=30)
            
            for j in range(len(param2_vals)):
                for i in range(len(param1_vals)):
                    if missing_mask[j, i]:
                        continue
                    avg_val = heatmap_data_local[j, i] * 100.0
                    max_val = max_data[j, i] * 100.0
                    if np.isnan(avg_val) or np.isnan(max_val):
                        continue
                    text = f'{avg_val:.1f}\n{max_val:.1f}'
                    ax.text(i, j, text, ha='center', va='center', color='white', fontsize=6)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=28)
            cbar.set_label('Test R² (\\%)', fontsize=30)
            
            if current_results:
                current_valid = [r for r in current_results if not r.get('numerical_error', False)]
                if current_valid:
                    best_result = max(current_valid, key=lambda x: x.get('best_test_r2', 0.0))
                    best_idx1 = find_closest_idx(best_result[param1_name], param1_vals)
                    best_idx2 = find_closest_idx(best_result[param2_name], param2_vals)
                    if best_idx1 is not None and best_idx2 is not None:
                        rect = mpatches.Rectangle(
                            (best_idx1 - 0.5, best_idx2 - 0.5),
                            1.0,
                            1.0,
                            linewidth=2,
                            edgecolor='white',
                            facecolor='none',
                            zorder=10,
                        )
                        ax.add_patch(rect)
            
            plt.tight_layout()
            base_name = f'heatmap_{param1_name}_vs_{param2_name}'
            if suffix:
                base_name += f'_{suffix}'
            output_path = output_dir / f'{base_name}_{timestamp}.png'
            plt.savefig(output_path)
            plt.close()
            print(f"Generated heatmap: {output_path}")

        build_and_save_heatmap(results)

        if avg_param1_name and avg_param1_vals:
            for v3 in avg_param1_vals:
                filtered = [r for r in results if abs(r.get(avg_param1_name, 0.0) - v3) < 1e-6]
                if not filtered:
                    continue
                label_val = format_val(v3, avg_param1_name)
                if avg_param1_name == 'lambda':
                    title_suffix = f'Lambda ($\\lambda$) = {label_val}'
                elif avg_param1_name == 'omega':
                    title_suffix = f'Omega ($\\omega$) = {label_val}'
                elif avg_param1_name == 'gamma_real':
                    title_suffix = f'Gamma Real ($\\gamma_r$) = {label_val}'
                elif avg_param1_name == 'gamma_imag':
                    title_suffix = f'Gamma Imaginary ($\\gamma_i$) = {label_val}'
                else:
                    title_suffix = f'{avg_param1_name} = {label_val}'
                build_and_save_heatmap(
                    filtered,
                    suffix=f'{avg_param1_name}_{label_val.replace(".", "p")}',
                    title_suffix=title_suffix,
                )


def main():
    parser = argparse.ArgumentParser(
        description="HORN grid search for Mackey-Glass prediction"
    )
    parser.add_argument(
        "--dynamics", type=str, default="sl", choices=["dho", "sl"]
    )
    parser.add_argument("--num-hidden", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--omega-min", type=float, default=0.02)
    parser.add_argument("--omega-max", type=float, default=0.08)
    parser.add_argument("--omega-steps", type=int, default=6)
    parser.add_argument("--lambda-min", type=float, default=-0.1)
    parser.add_argument("--lambda-max", type=float, default=-0.1)
    parser.add_argument("--lambda-steps", type=int, default=1)
    parser.add_argument("--gamma-real-min", type=float, default=0.0)
    parser.add_argument("--gamma-real-max", type=float, default=0.0)
    parser.add_argument("--gamma-real-steps", type=int, default=1)
    parser.add_argument("--gamma-imag-min", type=float, default=0.0)
    parser.add_argument("--gamma-imag-max", type=float, default=0.0)
    parser.add_argument("--gamma-imag-steps", type=int, default=1)
    parser.add_argument("--lambda-param", type=float, default=-0.04)
    parser.add_argument("--gamma-real", type=float, default=-0.05)
    parser.add_argument("--gamma-imag", type=float, default=0.1)
    parser.add_argument("--lr-decay-power", type=float, default=1.0,
                        help="Power factor for cosine decay (lower = slower decay, default: 1.0)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0,
                        help="Minimum LR as fraction of initial LR (default: 0.0)")
    parser.add_argument("--results-dir", type=str, default="grid_mg_results")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--plot-only", type=str, default=None)

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

    args = parser.parse_args()

    if args.plot_only:
        with open(args.plot_only, 'r') as f:
            data = json.load(f)
        results = data.get('results', [])
        base_results_dir = Path(args.plot_only).parent
        print(f"Generating heatmaps from {args.plot_only}...")
        print(f"Found {len(results)} results")
        config = data.get('config', {})
        omega_fixed = config.get('omega_fixed', False) or (len(config.get('omega_range', [0, 0, 1])) > 0 and config.get('omega_range', [0, 0, 1])[2] == 1)
        
        json_filename = Path(args.plot_only).stem
        if 'grid_mg_results_' in json_filename:
            run_id = json_filename.replace('grid_mg_results_', '')
        else:
            run_id = json_filename
        timestamp = run_id
        plots_dir = base_results_dir / run_id
        
        generate_plots(results, plots_dir, omega_fixed=omega_fixed, timestamp=timestamp, config=config)
        print(f"\nPlots saved in {plots_dir}")
        return

    torch.manual_seed(args.seed)

    if args.dynamics == "sl":
        omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_steps)
        lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
        gamma_real_values = np.linspace(
            args.gamma_real_min, args.gamma_real_max, args.gamma_real_steps
        )
        gamma_imag_values = np.linspace(
            args.gamma_imag_min, args.gamma_imag_max, args.gamma_imag_steps
        )
    else:
        omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_steps)
        lambda_values = [None]
        gamma_real_values = [None]
        gamma_imag_values = [None]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"grid_mg_results_{timestamp}.json"
    completed_file = results_dir / f"grid_mg_completed_{timestamp}.json"

    if args.dynamics == "sl":
        all_combinations = [
            (o, l, gr, gi)
            for o in omega_values
            for l in lambda_values
            for gr in gamma_real_values
            for gi in gamma_imag_values
        ]
    else:
        all_combinations = [(o, None, None, None) for o in omega_values]

    total_combinations = len(all_combinations)
    
    print("Grid search configuration:")
    print(f"  Omega: {len(omega_values)} values from {omega_values[0]:.6f} to {omega_values[-1]:.6f}")
    if args.dynamics == "sl":
        print(f"  Lambda: {len(lambda_values)} values from {lambda_values[0]:.3f} to {lambda_values[-1]:.3f}")
        print(f"  Gamma_real: {len(gamma_real_values)} values from {gamma_real_values[0]:.3f} to {gamma_real_values[-1]:.3f}")
        print(f"  Gamma_imag: {len(gamma_imag_values)} values from {gamma_imag_values[0]:.3f} to {gamma_imag_values[-1]:.3f}")
    else:
        print("  Lambda/Gamma: not swept (DHO dynamics)")

    results = []
    completed_indices = set()
    if args.resume:
        with open(args.resume, "r") as f:
            resume_data = json.load(f)
        results = resume_data.get("results", [])
        completed_indices = set(resume_data.get("completed_indices", []))
        print(
            f"Resuming from {args.resume}: loaded {len(results)} results, "
            f"{len(completed_indices)} completed indices"
        )

    print(f"\nStarting grid search: {total_combinations} total combinations")
    print(f"  {len(completed_indices)} already completed")
    print(f"  {total_combinations - len(completed_indices)} remaining\n")

    pbar = tqdm(
        total=len(all_combinations),
        initial=len(completed_indices),
        desc="Grid search",
    )
    for idx, (omega_val, lambda_val, gamma_real_val, gamma_imag_val) in enumerate(
        all_combinations
    ):
        if idx in completed_indices:
            continue

        res = run_training_grid(
            args,
            omega_val,
            lambda_val,
            gamma_real_val,
            gamma_imag_val,
            run_idx=idx,
        )
        res["index"] = idx
        results.append(res)
        completed_indices.add(idx)
        pbar.update(1)

        if (len(results) % args.save_interval == 0) or (
            idx == len(all_combinations) - 1
        ):
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "config": {
                            "omega_range": [
                                float(omega_values[0]),
                                float(omega_values[-1]),
                                len(omega_values),
                            ],
                            "lambda_range": [
                                float(args.lambda_min),
                                float(args.lambda_max),
                                args.lambda_steps,
                            ]
                            if args.dynamics == "sl"
                            else None,
                            "gamma_real_range": [
                                float(args.gamma_real_min),
                                float(args.gamma_real_max),
                                args.gamma_real_steps,
                            ]
                            if args.dynamics == "sl"
                            else None,
                            "gamma_imag_range": [
                                float(args.gamma_imag_min),
                                float(args.gamma_imag_max),
                                args.gamma_imag_steps,
                            ]
                            if args.dynamics == "sl"
                            else None,
                            "num_hidden": args.num_hidden,
                            "epochs": args.epochs,
                            "dynamics": args.dynamics,
                        },
                        "results": results,
                        "completed_indices": list(completed_indices),
                    },
                    f,
                    indent=2,
                )
            pbar.write(
                f"Saved results to {results_file} "
                f"({len(results)}/{len(all_combinations)})"
            )

    pbar.close()

    with open(completed_file, "w") as f:
        json.dump(
            {
                "config": {
                    "omega_range": [
                        float(omega_values[0]),
                        float(omega_values[-1]),
                        len(omega_values),
                    ],
                    "lambda_range": [
                        float(args.lambda_min),
                        float(args.lambda_max),
                        args.lambda_steps,
                    ]
                    if args.dynamics == "sl"
                    else None,
                    "gamma_real_range": [
                        float(args.gamma_real_min),
                        float(args.gamma_real_max),
                        args.gamma_real_steps,
                    ]
                    if args.dynamics == "sl"
                    else None,
                    "gamma_imag_range": [
                        float(args.gamma_imag_min),
                        float(args.gamma_imag_max),
                        args.gamma_imag_steps,
                    ]
                    if args.dynamics == "sl"
                    else None,
                    "num_hidden": args.num_hidden,
                    "epochs": args.epochs,
                    "dynamics": args.dynamics,
                },
                "results": results,
                "completed_indices": list(completed_indices),
            },
            f,
            indent=2,
        )

    print(f"\nGrid search complete! Results saved to {results_file}")
    print(f"Completed snapshot saved to {completed_file}")

    plot_dir = results_dir / timestamp
    generate_plots(
        results,
        plot_dir,
        omega_fixed=False,
        timestamp=timestamp,
        config={
            "omega_range": [
                float(omega_values[0]),
                float(omega_values[-1]),
                len(omega_values),
            ],
            "lambda_range": [
                float(args.lambda_min),
                float(args.lambda_max),
                args.lambda_steps,
            ]
            if args.dynamics == "sl"
            else None,
            "gamma_real_range": [
                float(args.gamma_real_min),
                float(args.gamma_real_max),
                args.gamma_real_steps,
            ]
            if args.dynamics == "sl"
            else None,
            "gamma_imag_range": [
                float(args.gamma_imag_min),
                float(args.gamma_imag_max),
                args.gamma_imag_steps,
            ]
            if args.dynamics == "sl"
            else None,
            "num_hidden": args.num_hidden,
            "epochs": args.epochs,
            "dynamics": args.dynamics,
        },
    )
    print("\nGenerated plots in", plot_dir)


if __name__ == "__main__":
    main()


