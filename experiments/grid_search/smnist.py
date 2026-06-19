import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch
import torchvision
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()

parser = argparse.ArgumentParser(description='SLON grid search for sMNIST')
parser.add_argument('--dynamics', type=str, default='sl', choices=['dho', 'sl'])
parser.add_argument('--num-hidden', type=int, default=50)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--h', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=0.04)
parser.add_argument('--gamma', type=float, default=0.01)
parser.add_argument('--omega-min', type=float, default=0.02)
parser.add_argument('--omega-max', type=float, default=0.08)
parser.add_argument('--omega-steps', type=int, default=6)
parser.add_argument('--lambda-min', type=float, default=-0.1)
parser.add_argument('--lambda-max', type=float, default=-0.1)
parser.add_argument('--lambda-steps', type=int, default=1)
parser.add_argument('--gamma-real-min', type=float, default=0)
parser.add_argument('--gamma-real-max', type=float, default=0.00)
parser.add_argument('--gamma-real-steps', type=int, default=1)
parser.add_argument('--gamma-imag-min', type=float, default=0.0)
parser.add_argument('--gamma-imag-max', type=float, default=0.0)
parser.add_argument('--gamma-imag-steps', type=int, default=1)
parser.add_argument('--fix-omega', action='store_true')
parser.add_argument('--omega', type=float, default=(2 * math.pi) / 28)
parser.add_argument('--lambda-param', type=float, default=None)
parser.add_argument('--gamma-real', type=float, default=None)
parser.add_argument('--gamma-imag', type=float, default=0.1)
parser.add_argument('--early-stop-patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay for regularization')
parser.add_argument('--results-dir', type=str, default='grid_smnist_results')
parser.add_argument('--save-interval', type=int, default=1)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--plot-only', type=str, default=None)
args = parser.parse_args()

from models import SLON

torch.manual_seed(args.seed)

dim_input = 1
dim_output = 10
batch_size_train = args.batch_size
batch_size_test = 1000

perm = None
if args.shuffle:
    perm = torch.randperm(784)

size_validation = 1000
train_set = torchvision.datasets.MNIST(root='data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root='data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - size_validation, size_validation])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size_test, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

def evaluate_model(data_loader, model, loss_fn):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.reshape(batch_size_test, 1, 784).permute(2, 0, 1)
            if perm is not None:
                images = images[perm, :, :]
            output = model(images, record=False)
            prediction = output['output']
            pred_label = prediction.data.max(1, keepdim=True)[1]
            correct += pred_label.eq(labels.data.view_as(pred_label)).sum()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

def run_training(omega_value, lambda_value=None, gamma_real_value=None, gamma_imag_value=None, run_idx=0):
    lambda_param = lambda_value if lambda_value is not None else (args.lambda_param if args.lambda_param is not None else None)
    if args.dynamics == 'sl':
        model = SLON(dim_input, args.num_hidden, dim_output, args.h, args.alpha, omega_value, args.gamma, lambda_param=lambda_param, gamma_real=gamma_real_value, gamma_imag=gamma_imag_value)
    else:
        model = SLON(dim_input, args.num_hidden, dim_output, args.h, args.alpha, omega_value, args.gamma)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    label_parts = [f'omega{omega_value:.6f}']
    if lambda_value is not None:
        label_parts.append(f'lambda{lambda_value:.6f}')
    if gamma_real_value is not None:
        label_parts.append(f'gr{gamma_real_value:.3f}')
    if gamma_imag_value is not None:
        label_parts.append(f'gi{gamma_imag_value:.3f}')
    output_dir = f'results/smnist/{timestamp}_grid{run_idx:03d}_' + '_'.join(label_parts)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / 'log.txt'
    with open(log_path, 'a') as fh_log:
        fh_log.write(f'omega: {omega_value:.6f}\n')
        if lambda_value is not None:
            fh_log.write(f'lambda: {lambda_param:.6f}\n')
        if gamma_real_value is not None:
            fh_log.write(f'gamma_real: {gamma_real_value:.6f}\n')
        if gamma_imag_value is not None:
            fh_log.write(f'gamma_imag: {gamma_imag_value:.6f}\n')
        best_eval = 0.0
        final_test_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        numerical_error = False
        
        for epoch in tqdm(range(args.epochs), total=args.epochs, desc=f'run {run_idx:03d}'):
            epoch_train_loss = 0.0
            model.train()
            
            try:
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.reshape(-1, 1, 784).permute(2, 0, 1)
                    if perm is not None:
                        images = images[perm, :, :]
                    
                    optimizer.zero_grad()
                    output = model(images)
                    prediction = output['output']
                    train_loss = loss_fn(prediction, labels)
                    
                    if not torch.isfinite(train_loss):
                        numerical_error = True
                        break
                    
                    epoch_train_loss += train_loss.item()
                    
                    train_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                if numerical_error:
                    break
                
                valid_acc = evaluate_model(valid_loader, model, loss_fn)
                test_acc = evaluate_model(test_loader, model, loss_fn)
                
                if valid_acc > best_eval:
                    best_eval = valid_acc
                    final_test_acc = test_acc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(valid_acc)
                
                avg_train_loss = epoch_train_loss / len(train_loader)
                msg = f'epoch {epoch}: train_loss: {avg_train_loss:.4f}, val: {valid_acc:.4f}, test: {test_acc:.4f}'
                if valid_acc == best_eval:
                    msg += ' [BEST]'
                fh_log.write(msg + '\n')
                
                if patience_counter >= args.early_stop_patience:
                    break
                    
            except (RuntimeError, ValueError) as e:
                if 'nan' in str(e).lower() or 'inf' in str(e).lower():
                    numerical_error = True
                    break
                raise
        
        fh_log.write(f'best test: {final_test_acc:.2f} (val: {best_eval:.4f}, epoch: {best_epoch})\n')
        if numerical_error:
            fh_log.write('WARNING: Training stopped due to numerical error (NaN/Inf)\n')
    
    result = {
        'omega': float(omega_value),
        'best_val_acc': float(best_eval),
        'best_test_acc': float(final_test_acc),
        'best_epoch': int(best_epoch),
        'numerical_error': numerical_error
    }
    if lambda_value is not None:
        result['lambda'] = float(lambda_value)
    if gamma_real_value is not None:
        result['gamma_real'] = float(gamma_real_value)
    if gamma_imag_value is not None:
        result['gamma_imag'] = float(gamma_imag_value)
    return result

def generate_plots(results, output_dir, omega_fixed=False, timestamp=None, config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not results:
        return
    
    def _safe_float(v):
        return float(v) if v is not None else 0.0

    def range_from_config(key, fallback_vals):
        if config and config.get(key):
            rng = config[key]
            if not rng or len(rng) < 3:
                return fallback_vals
            start, end, steps = rng[0], rng[1], int(rng[2])
            if steps <= 1:
                return fallback_vals
            return np.linspace(start, end, steps).tolist()
        return fallback_vals

    omega_vals = range_from_config('omega_range', sorted(set(_safe_float(r.get('omega')) for r in results)))
    lambda_vals = range_from_config('lambda_range', sorted(set(_safe_float(r.get('lambda')) for r in results)))
    gamma_real_vals = range_from_config('gamma_real_range', sorted(set(_safe_float(r.get('gamma_real')) for r in results)))
    gamma_imag_vals = range_from_config('gamma_imag_range', sorted(set(_safe_float(r.get('gamma_imag')) for r in results)))

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
                    acc = 0.0 if r.get('numerical_error', False) else r['best_test_acc']
                    
                    if np.isnan(heatmap_data[idx2, idx1]):
                        heatmap_data[idx2, idx1] = acc
                        max_data[idx2, idx1] = acc
                        count_data[idx2, idx1] = 1
                    else:
                        heatmap_data[idx2, idx1] += acc
                        max_data[idx2, idx1] = max(max_data[idx2, idx1], acc)
                        count_data[idx2, idx1] += 1
            
            heatmap_data_local = np.divide(heatmap_data, count_data, out=np.full_like(heatmap_data, np.nan), where=count_data!=0)
            
            missing_mask = np.isnan(heatmap_data_local)
            
            zero_mask = (heatmap_data_local == 0) & (~missing_mask)
            below_10_mask = (heatmap_data_local > 0) & (heatmap_data_local < 10) & (~missing_mask)
            
            valid_data = heatmap_data_local[~missing_mask]
            
            if len(valid_data) > 0:
                vmax = max(100.0, np.max(valid_data))
            else:
                vmax = 100.0
            
            heatmap_data_normalized = heatmap_data_local.copy()
            heatmap_data_normalized[below_10_mask] = 9.9
            heatmap_data_normalized[zero_mask] = -1
            heatmap_data_normalized[missing_mask] = -2
            
            vmin_norm = -2
            vmax_norm = vmax
            
            def normalize_value(val):
                return (val - vmin_norm) / (vmax_norm - vmin_norm)
            
            missing_pos = normalize_value(-2)
            zero_pos = normalize_value(-1)
            blue_pos = normalize_value(9.9)
            green_pos = normalize_value(10.0)
            
            cdict = {
                'red': [
                    (missing_pos, 1.0, 1.0),
                    (zero_pos, 0.0, 0.0),
                    (blue_pos, thesis_blue[0], thesis_blue[0]),
                    (green_pos, ifisc_green[0], ifisc_green[0]),
                    (1.0, thesis_red[0], thesis_red[0])
                ],
                'green': [
                    (missing_pos, 1.0, 1.0),
                    (zero_pos, 0.0, 0.0),
                    (blue_pos, thesis_blue[1], thesis_blue[1]),
                    (green_pos, ifisc_green[1], ifisc_green[1]),
                    (1.0, thesis_red[1], thesis_red[1])
                ],
                'blue': [
                    (missing_pos, 1.0, 1.0),
                    (zero_pos, 0.0, 0.0),
                    (blue_pos, thesis_blue[2], thesis_blue[2]),
                    (green_pos, ifisc_green[2], ifisc_green[2]),
                    (1.0, thesis_red[2], thesis_red[2])
                ]
            }
            
            custom_cmap = mcolors.LinearSegmentedColormap('custom_heatmap', cdict, N=256)
            
            norm = mcolors.Normalize(vmin=vmin_norm, vmax=vmax_norm)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(heatmap_data_normalized, aspect='auto', origin='lower', cmap=custom_cmap, norm=norm, interpolation='nearest')
            
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
            title = f'Test Accuracy Heatmap: {param1_label} vs {param2_label}'
            if title_suffix:
                title += f' ({title_suffix})'
            # ax.set_title(title)
            
            for j in range(len(param2_vals)):
                for i in range(len(param1_vals)):
                    if missing_mask[j, i]:
                        continue
                    avg_val = heatmap_data_local[j, i]
                    max_val = max_data[j, i]
                    if np.isnan(avg_val) or np.isnan(max_val):
                        continue
                    text = f'{avg_val:.1f}\n{max_val:.1f}'
                    ax.text(i, j, text, ha='center', va='center', color='white', fontsize=6)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=28)
            cbar.set_label('Test Accuracy (\\%)', fontsize=30)
            
            if current_results:
                current_valid = [r for r in current_results if not r.get('numerical_error', False)]
                if current_valid:
                    best_result = max(current_valid, key=lambda x: x['best_test_acc'])
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


if args.plot_only:
    with open(args.plot_only, 'r') as f:
        data = json.load(f)
    results = data.get('results', [])
    base_results_dir = Path(args.plot_only).parent
    print(f"Generating heatmaps from {args.plot_only}...")
    print(f"Found {len(results)} results")
    config = data.get('config', {})
    omega_fixed = config.get('omega_fixed', False)
    
    json_filename = Path(args.plot_only).stem
    if 'grid_smnist_results_' in json_filename:
        run_id = json_filename.replace('grid_smnist_results_', '')
    else:
        run_id = json_filename
    timestamp = run_id
    plots_dir = base_results_dir / run_id
    
    generate_plots(results, plots_dir, omega_fixed=omega_fixed, timestamp=timestamp, config=config)
    print(f"\nPlots saved in {plots_dir}")
    exit(0)

resume_data = None
if args.resume:
    with open(args.resume, 'r') as f:
        resume_data = json.load(f)

def build_grid_from_config(config):
    if not config:
        return None
    o = config.get('omega_range')
    if not o or len(o) < 3:
        return None
    omega_vals = np.linspace(float(o[0]), float(o[1]), int(o[2])).tolist()
    if config.get('omega_fixed') and len(omega_vals) >= 1:
        omega_vals = [omega_vals[0]]
    dynamics = config.get('dynamics', 'sl')
    if dynamics == 'sl':
        lr = config.get('lambda_range')
        gr = config.get('gamma_real_range')
        gi = config.get('gamma_imag_range')
        lambda_vals = np.linspace(lr[0], lr[1], int(lr[2])).tolist() if lr and len(lr) >= 3 else [None]
        gamma_real_vals = np.linspace(gr[0], gr[1], int(gr[2])).tolist() if gr and len(gr) >= 3 else [None]
        gamma_imag_vals = np.linspace(gi[0], gi[1], int(gi[2])).tolist() if gi and len(gi) >= 3 else [None]
        return omega_vals, lambda_vals, gamma_real_vals, gamma_imag_vals, dynamics
    return omega_vals, [None], [None], [None], dynamics

if resume_data:
    cfg = resume_data.get('config', {})
    built = build_grid_from_config(cfg)
    if built:
        omega_values = built[0]
        lambda_values = built[1]
        gamma_real_values = built[2]
        gamma_imag_values = built[3]
        resume_dynamics = built[4]
        args.dynamics = resume_dynamics
    else:
        built = None
else:
    built = None

if not built:
    omega_values = [args.omega] if args.fix_omega else np.linspace(args.omega_min, args.omega_max, args.omega_steps)
    if args.dynamics == 'sl':
        lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
        gamma_real_values = np.linspace(args.gamma_real_min, args.gamma_real_max, args.gamma_real_steps)
        gamma_imag_values = np.linspace(args.gamma_imag_min, args.gamma_imag_max, args.gamma_imag_steps)
    else:
        lambda_values = [None]
        gamma_real_values = [None]
        gamma_imag_values = [None]
    resume_dynamics = args.dynamics

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path(args.results_dir)
results_dir.mkdir(exist_ok=True)
if args.resume:
    resume_path = Path(args.resume)
    results_file = resume_path
    completed_file = resume_path.parent / f'grid_smnist_completed_{resume_path.stem.replace("grid_smnist_results_", "")}.json'
else:
    results_file = results_dir / f'grid_smnist_results_{timestamp}.json'
    completed_file = results_dir / f'grid_smnist_completed_{timestamp}.json'

print("Grid search configuration:")
if args.fix_omega or (resume_data and resume_data.get('config', {}).get('omega_fixed')):
    print(f"  Omega: FIXED at {omega_values[0]:.6f}")
else:
    print(f"  Omega: {len(omega_values)} values from {omega_values[0]:.6f} to {omega_values[-1]:.6f}")
if args.dynamics == 'sl':
    print(f"  Lambda: {len(lambda_values)} values from {lambda_values[0]:.3f} to {lambda_values[-1]:.3f}")
    print(f"  Gamma_real: {len(gamma_real_values)} values from {gamma_real_values[0]:.3f} to {gamma_real_values[-1]:.3f}")
    print(f"  Gamma_imag: {len(gamma_imag_values)} values from {gamma_imag_values[0]:.3f} to {gamma_imag_values[-1]:.3f}")
else:
    print("  Lambda/Gamma: not swept (DHO dynamics)")

if args.dynamics == 'sl':
    all_combinations = list(itertools.product(omega_values, lambda_values, gamma_real_values, gamma_imag_values))
else:
    all_combinations = [(omega_val, None, None, None) for omega_val in omega_values]

total_combinations = len(all_combinations)

def _range_to_config(vals):
    if not vals or vals[0] is None:
        return None
    return [float(vals[0]), float(vals[-1]), len(vals)]

results = []
completed_indices = set()
if args.resume and resume_data:
    results = resume_data.get('results', [])
    completed_indices = set(resume_data.get('completed_indices', []))
    if not completed_indices and results:
        completed_indices = set(r['index'] for r in results if 'index' in r)
        if completed_indices:
            print(f"Resuming: inferred completed_indices from results ({len(completed_indices)} indices)")
    completed_indices = {i for i in completed_indices if i < total_combinations}
    print(f"Resuming from {args.resume}: loaded {len(results)} results, {len(completed_indices)} completed indices (grid size {total_combinations})")

print(f"\nStarting grid search: {total_combinations} total combinations")
print(f"  {len(completed_indices)} already completed")
print(f"  {total_combinations - len(completed_indices)} remaining\n")

pbar = tqdm(total=len(all_combinations), initial=len(completed_indices), desc='Grid search')
for idx, (omega_val, lambda_val, gamma_real_val, gamma_imag_val) in enumerate(all_combinations):
    if idx in completed_indices:
        continue
    res = run_training(omega_val, lambda_val, gamma_real_val, gamma_imag_val, run_idx=idx)
    res['index'] = idx
    results.append(res)
    completed_indices.add(idx)
    pbar.update(1)
    if (len(results) % args.save_interval == 0) or (idx == len(all_combinations) - 1):
        cfg = {
            'omega_range': [float(omega_values[0]), float(omega_values[-1]), len(omega_values)],
            'lambda_range': _range_to_config(lambda_values) if args.dynamics == 'sl' else None,
            'gamma_real_range': _range_to_config(gamma_real_values) if args.dynamics == 'sl' else None,
            'gamma_imag_range': _range_to_config(gamma_imag_values) if args.dynamics == 'sl' else None,
            'omega_fixed': args.fix_omega,
            'num_hidden': args.num_hidden,
            'epochs': args.epochs,
            'early_stop_patience': args.early_stop_patience,
            'shuffle': args.shuffle,
            'dynamics': args.dynamics,
        }
        with open(results_file, 'w') as f:
            json.dump({'config': cfg, 'results': results, 'completed_indices': list(completed_indices)}, f, indent=2)
        pbar.write(f'Saved results to {results_file} ({len(results)}/{len(all_combinations)})')
pbar.close()

with open(completed_file, 'w') as f:
    json.dump({
        'config': {
            'omega_range': [float(omega_values[0]), float(omega_values[-1]), len(omega_values)],
            'lambda_range': _range_to_config(lambda_values) if args.dynamics == 'sl' else None,
            'gamma_real_range': _range_to_config(gamma_real_values) if args.dynamics == 'sl' else None,
            'gamma_imag_range': _range_to_config(gamma_imag_values) if args.dynamics == 'sl' else None,
            'omega_fixed': args.fix_omega,
            'num_hidden': args.num_hidden,
            'epochs': args.epochs,
            'shuffle': args.shuffle,
            'dynamics': args.dynamics,
        },
        'results': results,
        'completed_indices': list(completed_indices)
    }, f, indent=2)

print(f"\nGrid search complete! Results saved to {results_file}")
print(f"Completed snapshot saved to {completed_file}")

plot_dir = results_dir / timestamp
generate_plots(results, plot_dir, omega_fixed=args.fix_omega or args.dynamics != 'sl', timestamp=timestamp, config={
    'omega_range': [float(omega_values[0]), float(omega_values[-1]), len(omega_values)],
    'lambda_range': _range_to_config(lambda_values) if args.dynamics == 'sl' else None,
    'gamma_real_range': _range_to_config(gamma_real_values) if args.dynamics == 'sl' else None,
    'gamma_imag_range': _range_to_config(gamma_imag_values) if args.dynamics == 'sl' else None,
    'omega_fixed': args.fix_omega,
    'num_hidden': args.num_hidden,
    'epochs': args.epochs,
    'early_stop_patience': args.early_stop_patience,
    'shuffle': args.shuffle,
    'dynamics': args.dynamics,
})
print("\nGenerated plots in", plot_dir)
