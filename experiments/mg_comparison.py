import os
import sys
import argparse
import json
from datetime import datetime
from itertools import product

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()

from training.train_mackey_glass import build_parser as build_mg_parser
from training.train_mackey_glass import collect_all_test_predictions, run_training, split_bounds


def run_single_training(args, param_overrides, run_idx, total_runs, output_dir):
    """Train one configuration with the standard Mackey-Glass trainer and collect its curves and predictions."""
    run_args = vars(build_mg_parser().parse_args([]))
    run_args.update({k: v for k, v in vars(args).items() if k in run_args})
    run_args.update(param_overrides)
    run_args.update(
        output_dir=os.path.join(output_dir, 'runs', f'run{run_idx:03d}'),
        early_stop_patience=args.epochs + 1,
        sweep_mode=True,
        skip_epoch_plots=True,
    )
    run_args = argparse.Namespace(**run_args)
    print(f'Run {run_idx + 1}/{total_runs} -> {run_args.output_dir}')

    _, model, test_loader, full_series = run_training(run_args, return_model=True)
    with open(os.path.join(run_args.output_dir, 'metrics.json')) as f:
        metrics = [m for m in json.load(f) if m['epoch'] >= 0]

    test_start = split_bounds(run_args.series_length, run_args.val_fraction, run_args.test_fraction)[1]
    preds, targets, target_indices, _ = collect_all_test_predictions(test_loader, model, test_start, run_args.horizon)
    return {
        'test_r2_scores': [m['test_r2'] for m in metrics],
        'val_r2_scores': [m['val_r2'] for m in metrics],
        'test_mse_scores': [m['test_mse'] for m in metrics],
        'val_mse_scores': [m['val_mse'] for m in metrics],
        'param_overrides': param_overrides,
        'test_predictions': preds,
        'test_targets': targets,
        'test_indices': target_indices,
        'full_series': full_series,
        'test_start_idx': test_start,
    }


def generate_param_combinations(args):
    def values(list_arg, default, cast=float):
        return [cast(x) for x in list_arg.split(',')] if list_arg else [default]

    grid = {
        'num_hidden': values(args.num_hidden_list, args.num_hidden, int),
        'h': values(args.h_list, args.h),
        'alpha': values(args.alpha_list, args.alpha),
        'omega': values(args.omega_list, args.omega),
        'gamma': values(args.gamma_list, args.gamma),
        'lambda_param': values(args.lambda_list, args.lambda_param),
        'mg_tau': values(args.mg_tau_list, args.mg_tau),
        'horizon': values(args.horizon_list, args.horizon, int),
        'remove_top_n_freqs': values(args.remove_top_n_freqs_list, args.remove_top_n_freqs, int),
    }
    return [dict(zip(grid, combo)) for combo in product(*grid.values())]


def create_label(param_overrides):
    parts = []
    if 'num_hidden' in param_overrides:
        parts.append(f"$N={param_overrides['num_hidden']}$")
    if 'h' in param_overrides:
        parts.append(f"$h={param_overrides['h']:.3f}$")
    if 'alpha' in param_overrides:
        parts.append(f"$\\alpha={param_overrides['alpha']:.4f}$")
    if 'omega' in param_overrides:
        parts.append(f"$\\omega={param_overrides['omega']:.4f}$")
    if 'gamma' in param_overrides:
        parts.append(f"$\\gamma={param_overrides['gamma']:.4f}$")
    if 'lambda_param' in param_overrides:
        parts.append(f"$\\lambda={param_overrides['lambda_param']:.4f}$")
    if 'mg_tau' in param_overrides:
        parts.append(f"$\\tau={param_overrides['mg_tau']:.1f}$")
    if 'horizon' in param_overrides:
        parts.append(f"$H={param_overrides['horizon']}$")
    if 'remove_top_n_freqs' in param_overrides:
        parts.append(f"$R={param_overrides['remove_top_n_freqs']}$")
    
    return ", ".join(parts) if parts else "default"


def get_line_style_and_alpha(param_overrides, varying_param):
    if varying_param == 'mg_tau':
        tau = param_overrides.get('mg_tau', 17.0)
        if tau == 17.0:
            return '-', 1.0
        elif tau == 34.0:
            return '--', 1.0
        else:
            return '-', 0.7
    elif varying_param == 'horizon':
        horizon = param_overrides.get('horizon', 1)
        styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        alphas = [1.0, 0.9, 0.8, 0.7, 0.6]
        idx = min(horizon - 1, len(styles) - 1) if horizon >= 1 else 0
        return styles[idx], alphas[idx]
    elif varying_param == 'lambda_param':
        lambda_val = param_overrides.get('lambda_param', 0.1)
        if lambda_val == 0.1:
            return '-', 1.0
        elif lambda_val == -0.1:
            return '--', 1.0
        else:
            return '-', 0.7
    return '-', 1.0


def create_scaling_plots(all_results, output_dir, update_mode=False):
    if not all_results:
        return
    
    num_hidden_values = sorted(set(r['param_overrides'].get('num_hidden', 50) for r in all_results))
    
    param_display_names = {
        'mg_tau': 'MG Delay ($\\tau$)',
        'horizon': 'Prediction Horizon ($H$)',
        'lambda_param': 'Damping ($\\lambda$)',
        'h': 'Step size ($h$)',
    }
    
    for varying_param, param_name in param_display_names.items():
        param_values = sorted(
            {
                r['param_overrides'].get(varying_param)
                for r in all_results
                if varying_param in r['param_overrides']
            }
        )
        if len(param_values) <= 1:
            continue
        
        filtered_results = [
            r for r in all_results
            if varying_param in r['param_overrides']
            and r['param_overrides'].get(varying_param) in param_values
        ]
        if not filtered_results:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig_mse, ax_mse = plt.subplots(figsize=(12, 8))
        all_r2_values = []
        all_mse_values = []
        
        for num_hidden in num_hidden_values:
            for param_val in param_values:
                matching_results = [
                    r for r in filtered_results
                    if r['param_overrides'].get('num_hidden') == num_hidden
                    and r['param_overrides'].get(varying_param) == param_val
                ]
                if not matching_results:
                    continue
                
                result = matching_results[0]
                all_r2_values.extend(result['test_r2_scores'])
                epochs = np.arange(len(result['test_r2_scores']))
                
                mse_values = np.array(result['test_mse_scores'], dtype=float)
                mse_values = np.clip(mse_values, 1e-12, None)
                all_mse_values.extend(mse_values.tolist())
                
                param_idx = param_values.index(param_val)
                frac = 1.0 - (param_idx / max(1, len(param_values) - 1)) if len(param_values) > 1 else 1.0
                color = mycmap(frac)
                if varying_param == 'h':
                    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
                    alphas = [1.0, 0.9, 0.8, 0.7, 0.6]
                    idx = min(param_values.index(param_val), len(styles) - 1)
                    linestyle, alpha = styles[idx], alphas[idx]
                else:
                    linestyle, alpha = get_line_style_and_alpha(result['param_overrides'], varying_param)
                
                if varying_param == 'mg_tau':
                    label = f"$N={num_hidden}$, $\\tau={param_val:.0f}$"
                elif varying_param == 'horizon':
                    label = f"$N={num_hidden}$, $H={param_val}$"
                elif varying_param == 'lambda_param':
                    label = f"$N={num_hidden}$, $\\lambda={param_val:.1f}$"
                elif varying_param == 'h':
                    label = f"$N={num_hidden}$, $h={param_val:.3f}$"
                else:
                    label = f"$N={num_hidden}$"
                
                ax.plot(epochs, result['test_r2_scores'], label=label, color=color, 
                       linewidth=2.5, linestyle=linestyle, alpha=alpha)
                ax_mse.plot(epochs, mse_values, label=label, color=color,
                            linewidth=2.5, linestyle=linestyle, alpha=alpha)
        
        ax.set_xlabel("epoch", fontsize=14)
        ax.set_ylabel("test $R^2$", fontsize=14)
        ax.set_title(f"Scaling: {param_name}", fontsize=16)
        ax.legend(loc="best", fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.0, 1.0)
        
        fig.tight_layout()
        plot_name = f"scaling_{varying_param}.png"
        fig.savefig(f"{output_dir}/{plot_name}")

        if all_r2_values and max(all_r2_values) >= 0.5:
            ax.set_ylim(0.5, 1.0)
            fig.tight_layout()
            plot_name_zoom = f"scaling_{varying_param}_zoom.png"
            fig.savefig(f"{output_dir}/{plot_name_zoom}")
        
        plt.close(fig)
        
        ax_mse.set_xlabel("epoch", fontsize=14)
        ax_mse.set_ylabel("test MSE", fontsize=14)
        ax_mse.set_title(f"Scaling (log scale): {param_name}", fontsize=16)
        ax_mse.set_yscale("log")
        ax_mse.legend(loc="best", fontsize=10, ncol=2)
        ax_mse.grid(True, which="both", alpha=0.3)
        
        if all_mse_values:
            ymin = min(all_mse_values)
            ymax = max(all_mse_values)
            if ymin > 0 and ymax > ymin:
                ax_mse.set_ylim(ymin, ymax)
        
        fig_mse.tight_layout()
        plot_name_mse = f"scaling_{varying_param}_mse_log.png"
        fig_mse.savefig(f"{output_dir}/{plot_name_mse}")
        plt.close(fig_mse)
        if not update_mode:
            print(f"Saved scaling plot (log MSE): {output_dir}/{plot_name_mse}")
        if not update_mode:
            print(f"Saved scaling plot: {output_dir}/{plot_name}")


def create_heatmap_plot(all_results, output_dir):
    num_hidden_results = [
        r for r in all_results
        if 'num_hidden' in r['param_overrides']
    ]
    remove_freqs_results = [
        r for r in all_results
        if 'remove_top_n_freqs' in r['param_overrides']
    ]
    
    if not num_hidden_results or not remove_freqs_results:
        return
    
    num_hidden_values = sorted(set(r['param_overrides'].get('num_hidden') for r in num_hidden_results))
    remove_freqs_values = sorted(set(r['param_overrides'].get('remove_top_n_freqs') for r in remove_freqs_results))
    
    if len(num_hidden_values) <= 1 or len(remove_freqs_values) <= 1:
        return
    
    heatmap_data = np.full((len(remove_freqs_values), len(num_hidden_values)), np.nan)
    count_data = np.zeros((len(remove_freqs_values), len(num_hidden_values)), dtype=int)
    max_data = np.full((len(remove_freqs_values), len(num_hidden_values)), np.nan)
    
    def find_closest_idx(val, vals, tol=1e-6):
        for idx, v in enumerate(vals):
            if abs(v - val) < tol:
                return idx
        return None
    
    for r in all_results:
        num_hidden = r['param_overrides'].get('num_hidden')
        remove_freqs = r['param_overrides'].get('remove_top_n_freqs')
        
        if num_hidden is None or remove_freqs is None:
            continue
        
        idx1 = find_closest_idx(num_hidden, num_hidden_values)
        idx2 = find_closest_idx(remove_freqs, remove_freqs_values)
        
        if idx1 is not None and idx2 is not None:
            final_r2 = r['test_r2_scores'][-1]
            r2 = max(0.0, final_r2)
            
            if np.isnan(heatmap_data[idx2, idx1]):
                heatmap_data[idx2, idx1] = r2
                max_data[idx2, idx1] = r2
                count_data[idx2, idx1] = 1
            else:
                heatmap_data[idx2, idx1] += r2
                max_data[idx2, idx1] = max(max_data[idx2, idx1], r2)
                count_data[idx2, idx1] += 1
    
    heatmap_data_avg = np.divide(heatmap_data, count_data, out=np.full_like(heatmap_data, np.nan), where=count_data!=0)
    
    heatmap_data_percent = heatmap_data_avg.copy() * 100.0
    
    vmin = 97.0
    vmax = 100.0
    
    missing_mask = np.isnan(heatmap_data_percent) | (heatmap_data_percent < vmin)
    data_for_plot = np.ma.array(heatmap_data_percent, mask=missing_mask)
    
    threshold = 95.0
    
    cdict = {
        'red': [
            (0.0, thesis_blue[0], thesis_blue[0]),
            (0.5, ifisc_green[0], ifisc_green[0]),
            (1.0, thesis_red[0], thesis_red[0])
        ],
        'green': [
            (0.0, thesis_blue[1], thesis_blue[1]),
            (0.5, ifisc_green[1], ifisc_green[1]),
            (1.0, thesis_red[1], thesis_red[1])
        ],
        'blue': [
            (0.0, thesis_blue[2], thesis_blue[2]),
            (0.5, ifisc_green[2], ifisc_green[2]),
            (1.0, thesis_red[2], thesis_red[2])
        ]
    }
    
    custom_cmap = mcolors.LinearSegmentedColormap('comparison_heatmap', cdict, N=256)
    custom_cmap.set_bad((1.0, 1.0, 1.0, 1.0))
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(data_for_plot, aspect='auto', origin='lower', cmap=custom_cmap, norm=norm, interpolation='nearest')
    
    ax.set_xticks(np.arange(len(num_hidden_values)))
    ax.set_xticklabels([f'{v}' for v in num_hidden_values], fontsize=14)
    ax.set_yticks(np.arange(len(remove_freqs_values)))
    ax.set_yticklabels([f'{v}' for v in remove_freqs_values], fontsize=14)
    
    ax.set_xlabel('Number of Hidden Nodes ($N$)', fontsize=16)
    ax.set_ylabel('Removed Top Frequencies ($R$)', fontsize=16)
    ax.set_title('Final Test $R^2$: Hidden Nodes vs Removed Frequencies', fontsize=18, fontweight='bold')
    
    for j in range(len(remove_freqs_values)):
        for i in range(len(num_hidden_values)):
            if missing_mask[j, i]:
                continue
            avg_val = heatmap_data_percent[j, i]
            text = f'{avg_val:.4f}'
            text_color = 'white' if avg_val > 95.0 else 'black'
            ax.text(i, j, text, ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(np.linspace(vmin, vmax, 11))
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Test $R^2$ (%)', fontsize=14)
    
    valid_results = [r for r in all_results if 'num_hidden' in r['param_overrides'] and 'remove_top_n_freqs' in r['param_overrides']]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['test_r2_scores'][-1])
        best_num_hidden = best_result['param_overrides']['num_hidden']
        best_remove_freqs = best_result['param_overrides']['remove_top_n_freqs']
        
        best_idx1 = find_closest_idx(best_num_hidden, num_hidden_values)
        best_idx2 = find_closest_idx(best_remove_freqs, remove_freqs_values)
        
        if best_idx1 is not None and best_idx2 is not None:
            rect = mpatches.Rectangle(
                (best_idx1 - 0.5, best_idx2 - 0.5),
                1.0,
                1.0,
                linewidth=3,
                edgecolor='white',
                facecolor='none',
                zorder=10,
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    plot_name = f"{output_dir}/heatmap_num_hidden_vs_remove_freqs.png"
    fig.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap plot: {plot_name}")


def create_horizon_plot(all_results, output_dir):
    horizon_results = [
        r for r in all_results
        if 'horizon' in r['param_overrides']
    ]
    
    if not horizon_results:
        return
    
    horizon_data = {}
    for result in horizon_results:
        horizon = result['param_overrides']['horizon']
        final_r2 = result['test_r2_scores'][-1]
        
        if horizon not in horizon_data:
            horizon_data[horizon] = []
        horizon_data[horizon].append(final_r2)
    
    if not horizon_data:
        return
    
    horizons = sorted(horizon_data.keys())
    mean_r2_values = [np.mean(horizon_data[h]) for h in horizons]
    
    all_r2_vals = [r2 for r2_list in horizon_data.values() for r2 in r2_list]
    vmin = min(0.0, min(all_r2_vals)) if all_r2_vals else 0.0
    vmax = max(1.0, max(all_r2_vals)) if all_r2_vals else 1.0
    
    threshold = 0.9
    
    threshold_pos = (threshold - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    threshold_half_pos = threshold_pos / 2.0
    
    cdict = {
        'red': [
            (0.0, 0.0, 0.0),
            (threshold_half_pos, thesis_blue[0], thesis_blue[0]),
            (threshold_pos, ifisc_green[0], ifisc_green[0]),
            (1.0, thesis_red[0], thesis_red[0])
        ],
        'green': [
            (0.0, 0.0, 0.0),
            (threshold_half_pos, thesis_blue[1], thesis_blue[1]),
            (threshold_pos, ifisc_green[1], ifisc_green[1]),
            (1.0, thesis_red[1], thesis_red[1])
        ],
        'blue': [
            (0.0, 0.0, 0.0),
            (threshold_half_pos, thesis_blue[2], thesis_blue[2]),
            (threshold_pos, ifisc_green[2], ifisc_green[2]),
            (1.0, thesis_red[2], thesis_red[2])
        ]
    }
    
    custom_cmap = mcolors.LinearSegmentedColormap('horizon_cmap', cdict, N=256)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for horizon in horizons:
        r2_values = horizon_data[horizon]
        mean_r2 = np.mean(r2_values)
        
        color = custom_cmap(norm(mean_r2))
        
        if len(r2_values) > 1:
            std_r2 = np.std(r2_values)
            ax.errorbar(horizon, mean_r2, yerr=std_r2, color=color, capsize=5, capthick=2, alpha=0.6, zorder=2)
            for r2_val in r2_values:
                point_color = custom_cmap(norm(r2_val))
                ax.plot(horizon, r2_val, 'o', color=point_color, markersize=4, alpha=0.4, zorder=1)
        
        ax.plot(horizon, mean_r2, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5, zorder=3, label=f'$H={horizon}$' if horizon == horizons[0] or horizon == horizons[-1] else '')
    
    ax.plot(horizons, mean_r2_values, '-', color='gray', linewidth=1.5, alpha=0.5, zorder=0)
    
    ax.set_xlabel("Horizon ($H$)", fontsize=14)
    ax.set_ylabel("Final Test $R^2$", fontsize=14)
    ax.set_title("Final Test Accuracy vs Prediction Horizon", fontsize=16)
    ax.grid(True, alpha=0.3)
    
    if len(horizons) > 10:
        step = max(1, len(horizons) // 10)
        tick_positions = horizons[::step]
        if horizons[-1] not in tick_positions:
            tick_positions.append(horizons[-1])
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_xticks(horizons)
        if len(horizons) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Test $R^2$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    fig.tight_layout()
    plot_name = f"{output_dir}/horizon_vs_accuracy.png"
    fig.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved horizon plot: {plot_name}")


def create_horizon_predictions_plot(all_results, output_dir, args=None, config=None):
    horizon_results = [
        r for r in all_results
        if 'horizon' in r['param_overrides']
    ]
    
    if not horizon_results:
        return
    
    horizon_groups = {}
    for result in horizon_results:
        horizon = result['param_overrides']['horizon']
        if horizon not in horizon_groups:
            horizon_groups[horizon] = []
        horizon_groups[horizon].append(result)
    
    if not horizon_groups:
        return
    
    horizons = sorted(horizon_groups.keys())
    
    result = horizon_results[0]
    full_series = result.get('full_series')
    test_start_idx = result.get('test_start_idx')
    
    if full_series is None or test_start_idx is None:
        params = result['param_overrides']
        if args is None:
            import argparse
            args = argparse.Namespace()
        
        if config:
            args.series_length = config.get('series_length', params.get('series_length', 20000))
            args.input_length = config.get('input_length', params.get('input_length', 100))
            args.batch_size = config.get('batch_size', params.get('batch_size', 64))
            args.val_fraction = config.get('val_fraction', params.get('val_fraction', 0.1))
            args.test_fraction = config.get('test_fraction', params.get('test_fraction', 0.1))
            args.mg_tau = config.get('mg_tau', params.get('mg_tau', 17.0))
            args.mg_delta_t = config.get('mg_delta_t', params.get('mg_delta_t', 1.0))
            args.mg_beta = config.get('mg_beta', params.get('mg_beta', 0.2))
            args.mg_gamma = config.get('mg_gamma', params.get('mg_gamma', 0.1))
            args.mg_n = config.get('mg_n', params.get('mg_n', 10.0))
            args.mg_x0 = config.get('mg_x0', params.get('mg_x0', 1.2))
            args.seed = config.get('seed', params.get('seed', 1))
            args.remove_top_n_freqs = config.get('remove_top_n_freqs', params.get('remove_top_n_freqs', 0))
        else:
            args.series_length = getattr(args, 'series_length', params.get('series_length', 20000))
            args.input_length = getattr(args, 'input_length', params.get('input_length', 100))
            args.batch_size = getattr(args, 'batch_size', params.get('batch_size', 64))
            args.val_fraction = getattr(args, 'val_fraction', params.get('val_fraction', 0.1))
            args.test_fraction = getattr(args, 'test_fraction', params.get('test_fraction', 0.1))
            args.mg_tau = getattr(args, 'mg_tau', params.get('mg_tau', 17.0))
            args.mg_delta_t = getattr(args, 'mg_delta_t', params.get('mg_delta_t', 1.0))
            args.mg_beta = getattr(args, 'mg_beta', params.get('mg_beta', 0.2))
            args.mg_gamma = getattr(args, 'mg_gamma', params.get('mg_gamma', 0.1))
            args.mg_n = getattr(args, 'mg_n', params.get('mg_n', 10.0))
            args.mg_x0 = getattr(args, 'mg_x0', params.get('mg_x0', 1.2))
            args.seed = getattr(args, 'seed', params.get('seed', 1))
            args.remove_top_n_freqs = getattr(args, 'remove_top_n_freqs', params.get('remove_top_n_freqs', 0))
        
        try:
            _, _, test_loader, full_series, _, _ = build_dataloaders(
                series_length=getattr(args, 'series_length', params.get('series_length', 20000)),
                input_length=getattr(args, 'input_length', params.get('input_length', 100)),
                horizon=horizons[0],
                batch_size=getattr(args, 'batch_size', params.get('batch_size', 64)),
                val_fraction=getattr(args, 'val_fraction', params.get('val_fraction', 0.1)),
                test_fraction=getattr(args, 'test_fraction', params.get('test_fraction', 0.1)),
                tau=getattr(args, 'mg_tau', params.get('mg_tau', 17.0)),
                delta_t=getattr(args, 'mg_delta_t', params.get('mg_delta_t', 1.0)),
                beta=getattr(args, 'mg_beta', params.get('mg_beta', 0.2)),
                gamma_mg=getattr(args, 'mg_gamma', params.get('mg_gamma', 0.1)),
                n=getattr(args, 'mg_n', params.get('mg_n', 10.0)),
                x0=getattr(args, 'mg_x0', params.get('mg_x0', 1.2)),
                seed=getattr(args, 'seed', params.get('seed', 1)),
                remove_top_n_freqs=getattr(args, 'remove_top_n_freqs', params.get('remove_top_n_freqs', 0)),
            )
            
            series_length = getattr(args, 'series_length', params.get('series_length', 20000))
            test_fraction = getattr(args, 'test_fraction', params.get('test_fraction', 0.1))
            val_fraction = getattr(args, 'val_fraction', params.get('val_fraction', 0.1))
            total_len = int(series_length * (1 - test_fraction - val_fraction))
            val_len = int(series_length * val_fraction)
            test_start_idx = total_len + val_len
            
            for r in horizon_results:
                r['full_series'] = full_series
                r['test_start_idx'] = test_start_idx
        except Exception as e:
            print(f"Warning: Could not regenerate series for horizon predictions plot: {e}")
            return
    
    all_final_r2 = [r['test_r2_scores'][-1] for r in horizon_results]
    vmin = min(0.0, min(all_final_r2)) if all_final_r2 else 0.0
    vmax = max(1.0, max(all_final_r2)) if all_final_r2 else 1.0
    
    threshold = 0.9
    threshold_pos = (threshold - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    threshold_half_pos = threshold_pos / 2.0
    
    cdict = {
        'red': [
            (0.0, 0.0, 0.0),
            (threshold_half_pos, thesis_blue[0], thesis_blue[0]),
            (threshold_pos, ifisc_green[0], ifisc_green[0]),
            (1.0, thesis_red[0], thesis_red[0])
        ],
        'green': [
            (0.0, 0.0, 0.0),
            (threshold_half_pos, thesis_blue[1], thesis_blue[1]),
            (threshold_pos, ifisc_green[1], ifisc_green[1]),
            (1.0, thesis_red[1], thesis_red[1])
        ],
        'blue': [
            (0.0, 0.0, 0.0),
            (threshold_half_pos, thesis_blue[2], thesis_blue[2]),
            (threshold_pos, ifisc_green[2], ifisc_green[2]),
            (1.0, thesis_red[2], thesis_red[2])
        ]
    }
    
    custom_cmap = mcolors.LinearSegmentedColormap('horizon_pred_cmap', cdict, N=256)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    actual_series_length = len(full_series)
    max_horizon = max(horizons)
    plot_end = min(actual_series_length, test_start_idx + max_horizon + 100)
    
    t_test = np.arange(test_start_idx, plot_end)
    series_test = full_series[test_start_idx:plot_end]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(t_test, series_test, color=thesis_blue, linewidth=1.5, alpha=0.7, label='Mackey-Glass trajectory')
    ax.axvline(test_start_idx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Test start')
    
    dot_positions = []
    dot_colors = []
    
    for horizon in horizons:
        group_results = horizon_groups[horizon]
        best_result = max(group_results, key=lambda x: x['test_r2_scores'][-1])
        
        final_r2 = best_result['test_r2_scores'][-1]
        color = custom_cmap(norm(final_r2))
        
        position = test_start_idx + horizon
        if position < actual_series_length:
            dot_positions.append(position)
            dot_colors.append(color)
    
    if dot_positions:
        ax.scatter(
            dot_positions,
            [full_series[pos] for pos in dot_positions],
            c=dot_colors,
            s=200,
            marker='o',
            edgecolors='black',
            linewidths=2.5,
            alpha=0.95,
            zorder=5
        )
    
    ax.set_xlim(test_start_idx, plot_end)
    y_min = np.min(series_test)
    y_max = np.max(series_test)
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    ax.set_xlabel('Time (simulation steps)', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Test Segment: Horizon Points on Mackey-Glass Trajectory (Colored by Final Test $R^2$)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Final Test $R^2$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    fig.tight_layout()
    plot_name = f"{output_dir}/horizon_predictions_comparison.png"
    fig.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved horizon predictions comparison plot: {plot_name}")


def regenerate_plots(data_file, output_dir=None):
    with open(data_file, "r") as f:
        full_data = json.load(f)
    
    if output_dir is None:
        data_dir = os.path.dirname(os.path.abspath(data_file))
        output_dir = data_dir
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    all_results = []
    for result_data in full_data['results']:
        result = {
            'param_overrides': result_data['parameters'],
            'test_r2_scores': result_data['test_r2_scores'],
            'val_r2_scores': result_data['val_r2_scores'],
            'test_mse_scores': result_data['test_mse_scores'],
            'val_mse_scores': result_data['val_mse_scores'],
        }
        if 'test_predictions' in result_data:
            result['test_predictions'] = np.array(result_data['test_predictions'])
        if 'test_targets' in result_data:
            result['test_targets'] = np.array(result_data['test_targets'])
        if 'test_indices' in result_data:
            result['test_indices'] = [int(x) if x is not None else None for x in result_data['test_indices']]
        if 'test_start_idx' in result_data:
            result['test_start_idx'] = int(result_data['test_start_idx'])
        all_results.append(result)
    
    if 'full_series' in full_data:
        full_series = np.array(full_data['full_series'])
        test_start_idx = full_data.get('test_start_idx', None)
        for result in all_results:
            result['full_series'] = full_series
            if test_start_idx is not None:
                result['test_start_idx'] = test_start_idx
    
    print(f"Regenerating plots from {data_file}")
    print(f"Total runs: {len(all_results)}")
    print("=" * 80)
    
    n_results = len(all_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_test_r2, ax_val_r2, ax_test_mse, ax_val_mse = axes.flatten()
    
    for idx, result in enumerate(all_results):
        epochs = np.arange(len(result['test_r2_scores']))
        label = create_label(result['param_overrides'])
        frac = idx / max(1, n_results - 1) if n_results > 1 else 0.5
        color = mycmap(frac)
        
        ax_test_r2.plot(epochs, result['test_r2_scores'], label=label, color=color, linewidth=2)
        ax_val_r2.plot(epochs, result['val_r2_scores'], label=label, color=color, linewidth=2, linestyle='--')
        ax_test_mse.plot(epochs, result['test_mse_scores'], label=label, color=color, linewidth=2)
        ax_val_mse.plot(epochs, result['val_mse_scores'], label=label, color=color, linewidth=2, linestyle='--')
    
    ax_test_r2.set_xlabel("epoch")
    ax_test_r2.set_ylabel("test $R^2$")
    ax_test_r2.set_title("Test $R^2$ Over Time (Comparison)")
    ax_test_r2.legend(loc="best", fontsize=9)
    ax_test_r2.grid(True, alpha=0.3)
    ax_test_r2.set_ylim(-2.0, 1.0)
    
    ax_val_r2.set_xlabel("epoch")
    ax_val_r2.set_ylabel("val $R^2$")
    ax_val_r2.set_title("Validation $R^2$ Over Time (Comparison)")
    ax_val_r2.legend(loc="best", fontsize=9)
    ax_val_r2.grid(True, alpha=0.3)
    ax_val_r2.set_ylim(-2.0, 1.0)
    
    ax_test_mse.set_xlabel("epoch")
    ax_test_mse.set_ylabel("test MSE")
    ax_test_mse.set_title("Test MSE Over Time (Comparison)")
    ax_test_mse.legend(loc="best", fontsize=9)
    ax_test_mse.grid(True, alpha=0.3)
    
    ax_val_mse.set_xlabel("epoch")
    ax_val_mse.set_ylabel("val MSE")
    ax_val_mse.set_title("Validation MSE Over Time (Comparison)")
    ax_val_mse.legend(loc="best", fontsize=9)
    ax_val_mse.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_all_metrics.png")

    all_test_r2_vals = [v for r in all_results for v in r['test_r2_scores']]
    all_val_r2_vals = [v for r in all_results for v in r['val_r2_scores']]
    if all_test_r2_vals and max(all_test_r2_vals) >= 0.5:
        ax_test_r2.set_ylim(0.5, 1.0)
    if all_val_r2_vals and max(all_val_r2_vals) >= 0.5:
        ax_val_r2.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_all_metrics_zoom.png")

    plt.close(fig)
    print(f"Saved comparison plot: {output_dir}/comparison_all_metrics.png")
    
    fig_single, ax_single = plt.subplots(figsize=(12, 8))
    
    for idx, result in enumerate(all_results):
        epochs = np.arange(len(result['test_r2_scores']))
        label = create_label(result['param_overrides'])
        frac = idx / max(1, n_results - 1) if n_results > 1 else 0.5
        color = mycmap(frac)
        ax_single.plot(epochs, result['test_r2_scores'], label=label, color=color, linewidth=2.5)
    
    ax_single.set_xlabel("epoch", fontsize=14)
    ax_single.set_ylabel("test $R^2$", fontsize=14)
    ax_single.set_title("Test Accuracy Over Time (Comparison)", fontsize=16)
    ax_single.legend(loc="best", fontsize=10, ncol=2)
    ax_single.grid(True, alpha=0.3)
    
    fig_single.tight_layout()
    fig_single.savefig(f"{output_dir}/comparison_test_r2.png")

    if all_test_r2_vals and max(all_test_r2_vals) >= 0.5:
        ax_single.set_ylim(0.5, 1.0)
        fig_single.tight_layout()
        fig_single.savefig(f"{output_dir}/comparison_test_r2_zoom.png")
    
    plt.close(fig_single)
    print(f"Saved test R² comparison plot: {output_dir}/comparison_test_r2.png")
    
    config = full_data.get('config', {})
    create_horizon_plot(all_results, output_dir)
    create_heatmap_plot(all_results, output_dir)
    create_horizon_predictions_plot(all_results, output_dir, args=None, config=config)
    
    create_scaling_plots(all_results, output_dir, update_mode=False)
    
    print("\n" + "=" * 80)
    print("Plots regenerated successfully!")
    print(f"Plots saved in: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train Mackey-Glass models over a grid of parameter values and compare them"
    )
    parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    parser.add_argument("--no-tanh", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-hidden", type=int, default=50)
    parser.add_argument("--num-hidden-list", type=str, default=None,
                        help="Comma-separated list of num_hidden values (e.g., '25,50,100')")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--h-list", type=str, default=None,
                        help="Comma-separated list of h values (e.g., '0.5,1.0,2.0')")
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--alpha-list", type=str, default=None,
                        help="Comma-separated list of alpha values")
    parser.add_argument("--omega", type=float, default=0.15)
    parser.add_argument("--omega-list", type=str, default=None,
                        help="Comma-separated list of omega values")
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--gamma-list", type=str, default=None,
                        help="Comma-separated list of gamma values")
    parser.add_argument("--lambda-param", type=float, default=-0.1)
    parser.add_argument("--lambda-list", type=str, default=None,
                        help="Comma-separated list of lambda values (only for sl dynamics)")
    parser.add_argument("--gamma-real", type=float, default=-0.1)
    parser.add_argument("--gamma-imag", type=float, default=0.0)

    parser.add_argument("--series-length", type=int, default=20000)
    parser.add_argument("--input-length", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)

    parser.add_argument("--mg-tau", type=float, default=17.0)
    parser.add_argument("--mg-tau-list", type=str, default=None,
                        help="Comma-separated list of MG tau values (e.g., '17,34')")
    parser.add_argument("--mg-delta-t", type=float, default=1.0)
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--mg-gamma", type=float, default=0.1)
    parser.add_argument("--mg-n", type=float, default=10.0)
    parser.add_argument("--mg-x0", type=float, default=1.2)
    parser.add_argument("--remove-top-n-freqs", type=int, default=0,
                        help="Number of top frequencies to remove from time series (increases difficulty, default: 0)")
    parser.add_argument("--remove-top-n-freqs-list", type=str, default=None,
                        help="Comma-separated list of remove_top_n_freqs values (e.g., '0,1,2')")
    
    parser.add_argument("--horizon-list", type=str, default=None,
                        help="Comma-separated list of horizon values (e.g., '1,2,3,4,5')")

    parser.add_argument("--lr-decay-power", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: auto-generated)")
    parser.add_argument("--regenerate-plots", type=str, default=None,
                        help="Path to full_results.json file to regenerate plots from saved data")

    args = parser.parse_args()
    
    if args.regenerate_plots:
        regenerate_plots(args.regenerate_plots, args.output_dir)
        return

    torch.manual_seed(args.seed)

    param_configs = generate_param_combinations(args)
    total_runs = len(param_configs)
    
    print(f"Running {total_runs} training runs with different parameter combinations...")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results/mackey_glass/comparison_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_results = []
    full_data_file = f"{output_dir}/full_results.json"
    
    for run_idx, param_overrides in enumerate(param_configs):
        print(f"\nRun {run_idx + 1}/{total_runs}: {create_label(param_overrides)}")
        print("-" * 80)
        
        try:
            result = run_single_training(args, param_overrides, run_idx, total_runs, output_dir)
            all_results.append(result)
            print(f"Completed: test R² final = {result['test_r2_scores'][-1]:.6f}")
            
            full_data = {
                'timestamp': timestamp,
                'total_runs': len(all_results),
                'config': {
                    'series_length': args.series_length,
                    'input_length': args.input_length,
                    'batch_size': args.batch_size,
                    'val_fraction': args.val_fraction,
                    'test_fraction': args.test_fraction,
                    'mg_tau': args.mg_tau,
                    'mg_delta_t': args.mg_delta_t,
                    'mg_beta': args.mg_beta,
                    'mg_gamma': args.mg_gamma,
                    'mg_n': args.mg_n,
                    'mg_x0': args.mg_x0,
                    'seed': args.seed,
                    'remove_top_n_freqs': getattr(args, 'remove_top_n_freqs', 0),
                },
                'results': []
            }
            for r in all_results:
                result_data = {
                    'parameters': r['param_overrides'],
                    'label': create_label(r['param_overrides']),
                    'test_r2_scores': [float(x) for x in r['test_r2_scores']],
                    'val_r2_scores': [float(x) for x in r['val_r2_scores']],
                    'test_mse_scores': [float(x) for x in r['test_mse_scores']],
                    'val_mse_scores': [float(x) for x in r['val_mse_scores']],
                }
                if 'test_predictions' in r and 'test_targets' in r and 'test_indices' in r:
                    result_data['test_predictions'] = [float(x) for x in r['test_predictions']]
                    result_data['test_targets'] = [float(x) for x in r['test_targets']]
                    result_data['test_indices'] = [int(x) if x is not None else None for x in r['test_indices']]
                if 'test_start_idx' in r:
                    result_data['test_start_idx'] = int(r['test_start_idx'])
                full_data['results'].append(result_data)
            
            if all_results and 'full_series' in all_results[0]:
                full_data['full_series'] = [float(x) for x in all_results[0]['full_series']]
                full_data['test_start_idx'] = int(all_results[0]['test_start_idx'])
            with open(full_data_file, "w") as f:
                json.dump(full_data, f, indent=2)
            
            create_scaling_plots(all_results, output_dir, update_mode=True)
            
        except Exception as e:
            print(f"ERROR: Run {run_idx + 1} failed: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("All runs completed. Generating final comparison plots...")
    
    create_scaling_plots(all_results, output_dir, update_mode=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_test_r2, ax_val_r2, ax_test_mse, ax_val_mse = axes.flatten()
    
    n_results = len(all_results)
    
    for idx, result in enumerate(all_results):
        epochs = np.arange(len(result['test_r2_scores']))
        label = create_label(result['param_overrides'])
        frac = idx / max(1, n_results - 1) if n_results > 1 else 0.5
        color = mycmap(frac)
        
        ax_test_r2.plot(epochs, result['test_r2_scores'], label=label, color=color, linewidth=2)
        ax_val_r2.plot(epochs, result['val_r2_scores'], label=label, color=color, linewidth=2, linestyle='--')
        ax_test_mse.plot(epochs, result['test_mse_scores'], label=label, color=color, linewidth=2)
        ax_val_mse.plot(epochs, result['val_mse_scores'], label=label, color=color, linewidth=2, linestyle='--')
    
    ax_test_r2.set_xlabel("epoch")
    ax_test_r2.set_ylabel("test $R^2$")
    ax_test_r2.set_title("Test $R^2$ Over Time (Comparison)")
    ax_test_r2.legend(loc="best", fontsize=9)
    ax_test_r2.grid(True, alpha=0.3)
    ax_test_r2.set_ylim(-2.0, 1.0)
    
    ax_val_r2.set_xlabel("epoch")
    ax_val_r2.set_ylabel("val $R^2$")
    ax_val_r2.set_title("Validation $R^2$ Over Time (Comparison)")
    ax_val_r2.legend(loc="best", fontsize=9)
    ax_val_r2.grid(True, alpha=0.3)
    ax_val_r2.set_ylim(-2.0, 1.0)
    
    ax_test_mse.set_xlabel("epoch")
    ax_test_mse.set_ylabel("test MSE")
    ax_test_mse.set_title("Test MSE Over Time (Comparison)")
    ax_test_mse.legend(loc="best", fontsize=9)
    ax_test_mse.grid(True, alpha=0.3)
    
    ax_val_mse.set_xlabel("epoch")
    ax_val_mse.set_ylabel("val MSE")
    ax_val_mse.set_title("Validation MSE Over Time (Comparison)")
    ax_val_mse.legend(loc="best", fontsize=9)
    ax_val_mse.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_all_metrics.png")

    all_test_r2_vals = [v for r in all_results for v in r['test_r2_scores']]
    all_val_r2_vals = [v for r in all_results for v in r['val_r2_scores']]
    if all_test_r2_vals and max(all_test_r2_vals) >= 0.5:
        ax_test_r2.set_ylim(0.5, 1.0)
    if all_val_r2_vals and max(all_val_r2_vals) >= 0.5:
        ax_val_r2.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_all_metrics_zoom.png")

    plt.close(fig)
    print(f"Saved comparison plot: {output_dir}/comparison_all_metrics.png")
    
    fig_single, ax_single = plt.subplots(figsize=(12, 8))
    
    for idx, result in enumerate(all_results):
        epochs = np.arange(len(result['test_r2_scores']))
        label = create_label(result['param_overrides'])
        frac = idx / max(1, n_results - 1) if n_results > 1 else 0.5
        color = mycmap(frac)
        ax_single.plot(epochs, result['test_r2_scores'], label=label, color=color, linewidth=2.5)
    
    ax_single.set_xlabel("epoch", fontsize=14)
    ax_single.set_ylabel("test $R^2$", fontsize=14)
    ax_single.set_title("Test Accuracy Over Time (Comparison)", fontsize=16)
    ax_single.legend(loc="best", fontsize=10, ncol=2)
    ax_single.grid(True, alpha=0.3)
    ax_single.set_ylim(-2.0, 1.0)
    
    fig_single.tight_layout()
    fig_single.savefig(f"{output_dir}/comparison_test_r2.png")

    if all_test_r2_vals and max(all_test_r2_vals) >= 0.5:
        ax_single.set_ylim(0.5, 1.0)
        fig_single.tight_layout()
        fig_single.savefig(f"{output_dir}/comparison_test_r2_zoom.png")
    
    plt.close(fig_single)
    print(f"Saved test R² comparison plot: {output_dir}/comparison_test_r2.png")
    
    create_horizon_plot(all_results, output_dir)
    create_heatmap_plot(all_results, output_dir)
    create_horizon_predictions_plot(all_results, output_dir, args)
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'parameters': result['param_overrides'],
            'label': create_label(result['param_overrides']),
            'final_test_r2': float(result['test_r2_scores'][-1]),
            'final_val_r2': float(result['val_r2_scores'][-1]),
            'final_test_mse': float(result['test_mse_scores'][-1]),
            'final_val_mse': float(result['val_mse_scores'][-1]),
            'best_test_r2': float(max(result['test_r2_scores'])),
            'best_val_r2': float(max(result['val_r2_scores'])),
        })
    
    summary_file = f"{output_dir}/summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary: {summary_file}")
    
    print(f"Full results data already saved: {full_data_file}")
    
    summary_txt = f"{output_dir}/summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Parameter Comparison Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total runs: {len(all_results)}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        sorted_results = sorted(summary_data, key=lambda x: x['final_test_r2'], reverse=True)
        
        f.write(f"{'Label':<40} {'Final Test R²':<15} {'Best Test R²':<15} {'Final Val R²':<15}\n")
        f.write("-" * 80 + "\n")
        
        for result in sorted_results:
            f.write(
                f"{result['label']:<40} {result['final_test_r2']:<15.6f} "
                f"{result['best_test_r2']:<15.6f} {result['final_val_r2']:<15.6f}\n"
            )
    
    print(f"Saved text summary: {summary_txt}")
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
