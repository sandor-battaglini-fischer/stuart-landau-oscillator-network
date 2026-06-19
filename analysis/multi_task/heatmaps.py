import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


thesis_blue = (0, 0.38, 0.68)
ifisc_green = (0.73, 0.83, 0.01)


def load_json_with_fallback(json_path, fallback_paths=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    config = data.get('config', {})
    
    if fallback_paths:
        for fallback_path in fallback_paths:
            if fallback_path and Path(fallback_path).exists():
                with open(fallback_path, 'r') as f:
                    fallback_data = json.load(f)
                fallback_results = fallback_data.get('results', [])
                
                existing_keys = set()
                for r in results:
                    key = (r.get('omega'), r.get('lambda'), r.get('gamma_real'), r.get('gamma_imag'))
                    existing_keys.add(key)
                
                for r in fallback_results:
                    key = (r.get('omega'), r.get('lambda'), r.get('gamma_real'), r.get('gamma_imag'))
                    if key not in existing_keys:
                        results.append(r)
    
    return results, config


def determine_common_ranges(all_results):
    all_omega = set()
    all_lambda = set()
    all_gamma_real = set()
    all_gamma_imag = set()
    
    for results in all_results:
        for r in results:
            if 'omega' in r:
                all_omega.add(r['omega'])
            if 'lambda' in r:
                all_lambda.add(r['lambda'])
            if 'gamma_real' in r:
                all_gamma_real.add(r['gamma_real'])
            if 'gamma_imag' in r:
                all_gamma_imag.add(r['gamma_imag'])
    
    omega_vals = sorted(all_omega) if all_omega else []
    lambda_vals = sorted(all_lambda) if all_lambda else []
    gamma_real_vals = sorted(all_gamma_real) if all_gamma_real else []
    gamma_imag_vals = sorted(all_gamma_imag) if all_gamma_imag else []
    
    return omega_vals, lambda_vals, gamma_real_vals, gamma_imag_vals


def range_from_config(key, config, fallback_vals):
    if config and config.get(key):
        start, end, steps = config[key]
        if steps > 1:
            return np.linspace(start, end, int(steps)).tolist()
        else:
            return [start]
    return fallback_vals


def find_closest_idx(val, vals, tol=None):
    if tol is None:
        unique_vals = sorted(set(vals))
        if len(unique_vals) <= 1:
            tol = 1e-6
        else:
            diffs = [
                abs(unique_vals[i + 1] - unique_vals[i])
                for i in range(len(unique_vals) - 1)
                if abs(unique_vals[i + 1] - unique_vals[i]) > 1e-9
            ]
            if diffs:
                tol = max(1e-6, 0.51 * min(diffs))
            else:
                tol = 1e-6

    best_idx = None
    best_diff = None
    for idx, v in enumerate(vals):
        d = abs(v - val)
        if d <= tol and (best_diff is None or d < best_diff):
            best_idx = idx
            best_diff = d
    return best_idx


def format_val(v, param_name):
    if param_name == 'omega':
        return f'{v:.4f}'
    else:
        return f'{v:.3f}'


def grid_edges(n):
    return np.arange(n + 1, dtype=float)


def zero_param_index(vals):
    x_array = np.asarray(vals, dtype=float)
    idx = np.where(np.isclose(x_array, 0.0, atol=1e-9))[0]
    return int(idx[0]) if len(idx) > 0 else None


def draw_zero_parameter_lines(ax, x_vals, y_vals, color='white', linewidth=2.5):
    x_zero_idx = zero_param_index(x_vals)
    if x_zero_idx is not None:
        ax.axvline(float(x_zero_idx), color=color, linewidth=linewidth, alpha=0.95)

    y_zero_idx = zero_param_index(y_vals)
    if y_zero_idx is not None:
        ax.axhline(float(y_zero_idx), color=color, linewidth=linewidth, alpha=0.95)


HEATMAP_XLABEL_PAD = 65


def set_heatmap_ticks(ax, param1_vals, param2_vals, param1_name, param2_name):
    num_ticks_x = min(16, len(param1_vals))
    num_ticks_y = min(16, len(param2_vals))

    # Move the tick labels to the right by a set amount (using labelpad)
    labelpad_shift = -40  # Adjust this value as needed for desired shift

    if len(param1_vals) > 16:
        x_indices = np.linspace(0, len(param1_vals) - 1, num_ticks_x, dtype=int)
        ax.set_xticks(x_indices)
        xtick_labels = [
            format_val(param1_vals[i], param1_name) for i in x_indices
        ]
    else:
        x_indices = np.arange(len(param1_vals))
        ax.set_xticks(x_indices)
        xtick_labels = [
            format_val(v, param1_name) for v in param1_vals
        ]

    # Set the tick labels, but use empty strings to suppress default labels,
    # then manually place moved-over labels as annotation for visual shift
    ax.set_xticklabels(['' for _ in xtick_labels])

    for idx, label in zip(x_indices, xtick_labels):
        ax.annotate(
            label,
            (idx, 0),  # Data coordinate at the tick
            xytext=(labelpad_shift, -5),  # Shift right in points
            textcoords='offset points',
            rotation=60,
            ha='left',
            va='top',
            fontsize=28,
            annotation_clip=False,
            xycoords=('data', 'axes fraction')
        )


    if len(param2_vals) > 16:
        y_indices = np.linspace(0, len(param2_vals) - 1, num_ticks_y, dtype=int)
        ax.set_yticks(y_indices)
        ax.set_yticklabels(
            [format_val(param2_vals[i], param2_name) for i in y_indices],
            fontsize=28
        )
    else:
        ax.set_yticks(np.arange(len(param2_vals)))
        ax.set_yticklabels(
            [format_val(v, param2_name) for v in param2_vals],
            fontsize=28
        )


def get_metric_value(result, task_name):
    if task_name == 'mg':
        if result.get('numerical_error', False):
            return 0.0
        return max(0.0, result.get('best_test_r2', 0.0)) * 100.0
    else:
        if result.get('numerical_error', False):
            return 0.0
        acc = result.get('best_test_acc', 0.0)
        if acc > 1.0:
            return acc
        else:
            return acc * 100.0


def get_threshold(task_name):
    if task_name == 'mg':
        return 90.0
    elif task_name == 'imdb':
        return 50.0
    elif task_name == 'smnist':
        return 10.0
    else:
        return 0.0


def get_metric_label(task_name):
    if task_name == 'mg':
        return 'Test R² (\\%)'
    else:
        return 'Test Accuracy (\\%)'


def get_task_label(task_name):
    labels = {
        'mg': 'Mackey-Glass',
        'imdb': 'IMDB',
        'smnist': 'sMNIST'
    }
    return labels.get(task_name, task_name)


def compute_heatmap_data(
    results,
    param1_name,
    param2_name,
    param1_vals,
    param2_vals,
    task_name,
):
    heatmap_data = np.full((len(param2_vals), len(param1_vals)), np.nan)
    count_data = np.zeros((len(param2_vals), len(param1_vals)), dtype=int)

    for r in results:
        param1_val = r.get(param1_name)
        param2_val = r.get(param2_name)

        if param1_val is None or param2_val is None:
            continue

        idx1 = find_closest_idx(param1_val, param1_vals)
        idx2 = find_closest_idx(param2_val, param2_vals)

        if idx1 is not None and idx2 is not None:
            metric_val = get_metric_value(r, task_name)

            if np.isnan(heatmap_data[idx2, idx1]):
                heatmap_data[idx2, idx1] = metric_val
                count_data[idx2, idx1] = 1
            else:
                heatmap_data[idx2, idx1] += metric_val
                count_data[idx2, idx1] += 1

    heatmap_data_local = np.divide(
        heatmap_data,
        count_data,
        out=np.full_like(heatmap_data, np.nan),
        where=count_data != 0,
    )

    return heatmap_data_local


def build_heatmap(
    results,
    param1_name,
    param2_name,
    param1_vals,
    param2_vals,
    task_name,
    output_path,
    title_suffix='\n'
):
    heatmap_data_local = compute_heatmap_data(
        results,
        param1_name,
        param2_name,
        param1_vals,
        param2_vals,
        task_name,
    )
    
    missing_mask = np.isnan(heatmap_data_local)
    
    valid_data = heatmap_data_local[~missing_mask]
    
    if len(valid_data) > 0:
        vmax = max(100.0, np.max(valid_data))
    else:
        vmax = 100.0
    
    data_for_plot = np.ma.array(heatmap_data_local, mask=missing_mask)
    
    threshold = get_threshold(task_name)
    
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
    
    im = ax.pcolormesh(
        grid_edges(len(param1_vals)),
        grid_edges(len(param2_vals)),
        data_for_plot,
        cmap=custom_cmap,
        norm=norm,
        shading='flat',
    )
    ax.set_aspect('auto')
    ax.grid(False)

    set_heatmap_ticks(ax, param1_vals, param2_vals, param1_name, param2_name)
    draw_zero_parameter_lines(ax, param1_vals, param2_vals)
    
    param1_label = param1_name.replace('_', ' ').title()
    param2_label = param2_name.replace('_', ' ').title()
    if param1_name == 'lambda':
        param1_label = '$\\lambda$'
    elif param1_name == 'omega':
        param1_label = '$\\omega$'
    elif param1_name == 'gamma_real':
        param1_label = r'$\gamma_{\mathrm{real}}$'
    elif param1_name == 'gamma_imag':
        param1_label = r'$\gamma_{\mathrm{imag}}$'
    
    if param2_name == 'lambda':
        param2_label = '$\\lambda$'
    elif param2_name == 'omega':
        param2_label = '$\\omega$'
    elif param2_name == 'gamma_real':
        param2_label = r'$\gamma_{\mathrm{real}}$'
    elif param2_name == 'gamma_imag':
        param2_label = r'$\gamma_{\mathrm{imag}}$'
    
    ax.set_xlabel(param1_label, fontsize=30, labelpad=HEATMAP_XLABEL_PAD)
    ax.set_ylabel(param2_label, fontsize=30)
    
    if title_suffix:
        ax.set_title(f'{get_task_label(task_name)}: {title_suffix}', fontsize=32)
    else:
        ax.set_title(get_task_label(task_name), fontsize=32)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(get_metric_label(task_name), fontsize=30)
    
    if task_name == 'imdb':
        ticks = cbar.get_ticks()
        if len(ticks) > 0:
            span = vmax - 50.0
            if span <= 0:
                labels = [50.0 for _ in ticks]
            else:
                labels = [50.0 + (t / vmax) * span for t in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{lbl:.0f}" for lbl in labels])
    
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()
    print(f"Generated heatmap: {output_path}")


def _fill_matrix_nearest_param(matrix, param1_vals, param2_vals):
    out = matrix.copy()
    nan_mask = np.isnan(matrix)
    if not np.any(nan_mask):
        return out
    valid = ~nan_mask
    if not np.any(valid):
        return out
    rows, cols = np.where(nan_mask)
    valid_rows, valid_cols = np.where(valid)
    p1 = np.array(param1_vals, dtype=float)
    p2 = np.array(param2_vals, dtype=float)
    for r, c in zip(rows, cols):
        x, y = p1[c], p2[r]
        dist_sq = (p1[valid_cols] - x) ** 2 + (p2[valid_rows] - y) ** 2
        idx = np.argmin(dist_sq)
        out[r, c] = matrix[valid_rows[idx], valid_cols[idx]]
    return out


def build_multi_task_average_heatmap(
    task_results_dict,
    param1_name,
    param2_name,
    param1_vals,
    param2_vals,
    output_path,
    use_full_grid=False,
):
    task_matrices = {}
    task_maxima = {}

    for task_name, task_results in task_results_dict.items():
        matrix = compute_heatmap_data(
            task_results,
            param1_name,
            param2_name,
            param1_vals,
            param2_vals,
            task_name,
        )
        task_matrices[task_name] = matrix

        if np.all(np.isnan(matrix)):
            task_maxima[task_name] = np.nan
        else:
            task_maxima[task_name] = np.nanmax(matrix)

    valid_tasks = [
        name
        for name, max_val in task_maxima.items()
        if max_val is not None and not np.isnan(max_val) and max_val > 0.0
    ]

    if len(valid_tasks) == 0:
        return

    base_shape = task_matrices[valid_tasks[0]].shape
    for name in valid_tasks[1:]:
        if task_matrices[name].shape != base_shape:
            return

    if use_full_grid and 'mg' in task_matrices:
        mg_mat = task_matrices['mg']
        if np.any(~np.isnan(mg_mat)):
            task_matrices['mg'] = _fill_matrix_nearest_param(
                mg_mat, param1_vals, param2_vals
            )

    stacked_normalized = []
    for name in valid_tasks:
        matrix = task_matrices[name]
        max_val = task_maxima[name]
        norm_matrix = matrix / max_val
        stacked_normalized.append(norm_matrix)

    stacked = np.stack(stacked_normalized, axis=0)
    avg_norm = np.nanmean(stacked, axis=0)

    combined_mask = np.ones_like(avg_norm, dtype=bool)
    for name in valid_tasks:
        combined_mask &= ~np.isnan(task_matrices[name])

    if use_full_grid:
        if 'imdb' in task_matrices and 'smnist' in task_matrices:
            base_mask = (~np.isnan(task_matrices['imdb'])) & (
                ~np.isnan(task_matrices['smnist'])
            )
        else:
            base_mask = np.any(~np.isnan(stacked), axis=0)

        data_for_plot = np.ma.array(avg_norm, mask=~base_mask)
        plot_param1_vals = param1_vals
        plot_param2_vals = param2_vals
    else:
        candidate_rows = list(np.where(combined_mask.any(axis=1))[0])
        candidate_cols = list(np.where(combined_mask.any(axis=0))[0])

        changed = True
        while changed and candidate_rows and candidate_cols:
            changed = False

            bad_rows = [
                i for i in candidate_rows
                if not np.all(combined_mask[i, candidate_cols])
            ]
            if bad_rows:
                for i in bad_rows:
                    candidate_rows.remove(i)
                changed = True

            if not candidate_rows:
                break

            bad_cols = [
                j for j in candidate_cols
                if not np.all(combined_mask[candidate_rows, j])
            ]
            if bad_cols:
                for j in bad_cols:
                    candidate_cols.remove(j)
                changed = True

        if not candidate_rows or not candidate_cols:
            return

        candidate_rows = np.array(candidate_rows, dtype=int)
        candidate_cols = np.array(candidate_cols, dtype=int)

        avg_norm_cropped = avg_norm[np.ix_(candidate_rows, candidate_cols)]
        data_for_plot = avg_norm_cropped
        plot_param1_vals = [param1_vals[i] for i in candidate_cols]
        plot_param2_vals = [param2_vals[i] for i in candidate_rows]

    vmax = 1.0
    threshold = 0.7 * vmax

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

    custom_cmap = mcolors.LinearSegmentedColormap('multi_task_heatmap', cdict, N=256)
    custom_cmap.set_bad((1.0, 1.0, 1.0, 1.0))

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.pcolormesh(
        grid_edges(len(plot_param1_vals)),
        grid_edges(len(plot_param2_vals)),
        data_for_plot,
        cmap=custom_cmap,
        norm=norm,
        shading='flat',
    )
    ax.set_aspect('auto')
    ax.grid(False)

    set_heatmap_ticks(ax, plot_param1_vals, plot_param2_vals, param1_name, param2_name)
    draw_zero_parameter_lines(ax, plot_param1_vals, plot_param2_vals)

    param1_label = param1_name.replace('_', ' ').title()
    param2_label = param2_name.replace('_', ' ').title()
    if param1_name == 'lambda':
        param1_label = '$\\lambda$'
    elif param1_name == 'omega':
        param1_label = '$\\omega$'
    elif param1_name == 'gamma_real':
        param1_label = r'$\gamma_{\mathrm{real}}$'
    elif param1_name == 'gamma_imag':
        param1_label = r'$\gamma_{\mathrm{imag}}$'

    if param2_name == 'lambda':
        param2_label = '$\\lambda$'
    elif param2_name == 'omega':
        param2_label = '$\\omega$'
    elif param2_name == 'gamma_real':
        param2_label = r'$\gamma_{\mathrm{real}}$'
    elif param2_name == 'gamma_imag':
        param2_label = r'$\gamma_{\mathrm{imag}}$'

    ax.set_xlabel(param1_label, fontsize=30, labelpad=HEATMAP_XLABEL_PAD)
    ax.set_ylabel(param2_label, fontsize=30)

    ax.set_title('Multi-task average (normalized)', fontsize=32)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label('Normalized performance (fraction of task best)', fontsize=30)

    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()
    print(f"Generated multi-task average heatmap: {output_path}")


def generate_multi_task_heatmaps(
    mg_json,
    imdb_json,
    smnist_json,
    output_dir,
    mg_fallback=None,
    imdb_fallback=None,
    smnist_fallback=None,
    omega_range=None,
    lambda_range=None,
    gamma_real_range=None,
    gamma_imag_range=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading JSON files...")
    mg_results, mg_config = load_json_with_fallback(mg_json, [mg_fallback] if mg_fallback else None)
    imdb_results, imdb_config = load_json_with_fallback(imdb_json, [imdb_fallback] if imdb_fallback else None)
    smnist_results, smnist_config = load_json_with_fallback(smnist_json, [smnist_fallback] if smnist_fallback else None)
    
    print(f"Loaded {len(mg_results)} MG results")
    print(f"Loaded {len(imdb_results)} IMDB results")
    print(f"Loaded {len(smnist_results)} sMNIST results")
    
    all_results = [mg_results, imdb_results, smnist_results]
    all_configs = [mg_config, imdb_config, smnist_config]
    
    print("\nDetermining common parameter ranges...")
    omega_vals_raw, lambda_vals_raw, gamma_real_vals_raw, gamma_imag_vals_raw = determine_common_ranges(all_results)
    
    if omega_range:
        omega_vals = np.linspace(omega_range[0], omega_range[1], int(omega_range[2])).tolist()
    else:
        omega_vals = omega_vals_raw
        for config in all_configs:
            if config:
                omega_vals = range_from_config('omega_range', config, omega_vals)
        omega_vals = sorted(set(omega_vals))
    
    if lambda_range:
        lambda_vals = np.linspace(lambda_range[0], lambda_range[1], int(lambda_range[2])).tolist()
    else:
        lambda_vals = lambda_vals_raw
        for config in all_configs:
            if config:
                lambda_vals = range_from_config('lambda_range', config, lambda_vals)
        lambda_vals = sorted(set(lambda_vals))
    
    if gamma_real_range:
        gamma_real_vals = np.linspace(gamma_real_range[0], gamma_real_range[1], int(gamma_real_range[2])).tolist()
    else:
        gamma_real_vals = gamma_real_vals_raw
        for config in all_configs:
            if config:
                gamma_real_vals = range_from_config('gamma_real_range', config, gamma_real_vals)
        gamma_real_vals = sorted(set(gamma_real_vals))
    
    if gamma_imag_range:
        gamma_imag_vals = np.linspace(gamma_imag_range[0], gamma_imag_range[1], int(gamma_imag_range[2])).tolist()
    else:
        gamma_imag_vals = gamma_imag_vals_raw
        for config in all_configs:
            if config:
                gamma_imag_vals = range_from_config('gamma_imag_range', config, gamma_imag_vals)
        gamma_imag_vals = sorted(set(gamma_imag_vals))
    
    print(f"  Omega: {len(omega_vals)} values")
    print(f"  Lambda: {len(lambda_vals)} values")
    print(f"  Gamma Real: {len(gamma_real_vals)} values")
    print(f"  Gamma Imag: {len(gamma_imag_vals)} values")
    
    omega_fixed = len(omega_vals) == 1
    
    if omega_fixed:
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
    
    task_data = [
        ('mg', mg_results),
        ('imdb', imdb_results),
        ('smnist', smnist_results),
    ]
    
    mt_lambda_vals = list(np.linspace(-0.5, 0.5, 11))
    mt_gamma_real_vals = list(np.linspace(-0.1, 0.1, 11))

    print("\nGenerating unified heatmaps...")
    for param1_name, param2_name, param1_vals, param2_vals, avg_param1_name, avg_param2_name, avg_param1_vals, avg_param2_vals in param_pairs:
        if len(param1_vals) <= 1 or len(param2_vals) <= 1:
            continue

        if param1_name == 'lambda' and param2_name == 'gamma_real':
            mt_param1_vals = mt_lambda_vals
            mt_param2_vals = mt_gamma_real_vals
            mt_use_full_grid = True
        else:
            mt_param1_vals = param1_vals
            mt_param2_vals = param2_vals
            mt_use_full_grid = False

        multi_task_results = {name: res for name, res in task_data}
        multi_task_output = output_dir / f'heatmap_{param1_name}_vs_{param2_name}_multi_task_avg.png'
        build_multi_task_average_heatmap(
            multi_task_results,
            param1_name,
            param2_name,
            mt_param1_vals,
            mt_param2_vals,
            multi_task_output,
            use_full_grid=mt_use_full_grid,
        )

        for task_name, task_results in task_data:
            output_path = output_dir / f'heatmap_{param1_name}_vs_{param2_name}_{task_name}.png'
            build_heatmap(
                task_results,
                param1_name,
                param2_name,
                param1_vals,
                param2_vals,
                task_name,
                output_path
            )
    
    has_cmd_ranges = omega_range or lambda_range or gamma_real_range or gamma_imag_range
    if has_cmd_ranges:
        print("\nGenerating task-specific heatmaps with full ranges from JSON...")
        task_configs = [
            ('mg', mg_results, mg_config),
            ('imdb', imdb_results, imdb_config),
            ('smnist', smnist_results, smnist_config),
        ]
        
        for task_name, task_results, task_config in task_configs:
            if not task_config:
                continue
            
            task_omega_raw = sorted(set(r.get('omega', 0.0) for r in task_results if 'omega' in r))
            task_lambda_raw = sorted(set(r.get('lambda', 0.0) for r in task_results if 'lambda' in r))
            task_gamma_real_raw = sorted(set(r.get('gamma_real', 0.0) for r in task_results if 'gamma_real' in r))
            task_gamma_imag_raw = sorted(set(r.get('gamma_imag', 0.0) for r in task_results if 'gamma_imag' in r))
            
            task_omega_vals = range_from_config('omega_range', task_config, task_omega_raw)
            task_lambda_vals = range_from_config('lambda_range', task_config, task_lambda_raw)
            task_gamma_real_vals = range_from_config('gamma_real_range', task_config, task_gamma_real_raw)
            task_gamma_imag_vals = range_from_config('gamma_imag_range', task_config, task_gamma_imag_raw)
            
            task_omega_vals = sorted(set(task_omega_vals))
            task_lambda_vals = sorted(set(task_lambda_vals))
            task_gamma_real_vals = sorted(set(task_gamma_real_vals))
            task_gamma_imag_vals = sorted(set(task_gamma_imag_vals))
            
            task_omega_fixed = len(task_omega_vals) == 1
            
            if task_omega_fixed:
                task_param_pairs = [
                    ('lambda', 'gamma_real', task_lambda_vals, task_gamma_real_vals, 'gamma_imag', None, task_gamma_imag_vals, None),
                    ('lambda', 'gamma_imag', task_lambda_vals, task_gamma_imag_vals, 'gamma_real', None, task_gamma_real_vals, None),
                    ('gamma_real', 'gamma_imag', task_gamma_real_vals, task_gamma_imag_vals, 'lambda', None, task_lambda_vals, None),
                ]
            else:
                task_param_pairs = [
                    ('lambda', 'omega', task_lambda_vals, task_omega_vals, 'gamma_real', 'gamma_imag', task_gamma_real_vals, task_gamma_imag_vals),
                    ('lambda', 'gamma_real', task_lambda_vals, task_gamma_real_vals, 'omega', 'gamma_imag', task_omega_vals, task_gamma_imag_vals),
                    ('lambda', 'gamma_imag', task_lambda_vals, task_gamma_imag_vals, 'omega', 'gamma_real', task_omega_vals, task_gamma_real_vals),
                    ('omega', 'gamma_real', task_omega_vals, task_gamma_real_vals, 'lambda', 'gamma_imag', task_lambda_vals, task_gamma_imag_vals),
                    ('omega', 'gamma_imag', task_omega_vals, task_gamma_imag_vals, 'lambda', 'gamma_real', task_lambda_vals, task_gamma_real_vals),
                    ('gamma_real', 'gamma_imag', task_gamma_real_vals, task_gamma_imag_vals, 'lambda', 'omega', task_lambda_vals, task_omega_vals),
                ]
            
            for param1_name, param2_name, param1_vals, param2_vals, avg_param1_name, avg_param2_name, avg_param1_vals, avg_param2_vals in task_param_pairs:
                if len(param1_vals) <= 1 or len(param2_vals) <= 1:
                    continue
                
                output_path = output_dir / f'heatmap_{param1_name}_vs_{param2_name}_{task_name}_fullrange.png'
                build_heatmap(
                    task_results,
                    param1_name,
                    param2_name,
                    param1_vals,
                    param2_vals,
                    task_name,
                    output_path
                )
    
    print(f"\nAll heatmaps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-task heatmaps with unified parameter ranges"
    )
    parser.add_argument(
        '--mg-json',
        type=str,
        required=True,
        help='Path to Mackey-Glass results JSON file'
    )
    parser.add_argument(
        '--imdb-json',
        type=str,
        required=True,
        help='Path to IMDB results JSON file'
    )
    parser.add_argument(
        '--smnist-json',
        type=str,
        required=True,
        help='Path to sMNIST results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for heatmaps'
    )
    parser.add_argument(
        '--mg-fallback',
        type=str,
        default=None,
        help='Fallback JSON file for missing MG data'
    )
    parser.add_argument(
        '--imdb-fallback',
        type=str,
        default=None,
        help='Fallback JSON file for missing IMDB data'
    )
    parser.add_argument(
        '--smnist-fallback',
        type=str,
        default=None,
        help='Fallback JSON file for missing sMNIST data'
    )
    parser.add_argument(
        '--omega-min',
        type=float,
        default=None,
        help='Minimum omega value for axis range'
    )
    parser.add_argument(
        '--omega-max',
        type=float,
        default=None,
        help='Maximum omega value for axis range'
    )
    parser.add_argument(
        '--omega-steps',
        type=int,
        default=None,
        help='Number of steps for omega axis range'
    )
    parser.add_argument(
        '--lambda-min',
        type=float,
        default=None,
        help='Minimum lambda value for axis range'
    )
    parser.add_argument(
        '--lambda-max',
        type=float,
        default=None,
        help='Maximum lambda value for axis range'
    )
    parser.add_argument(
        '--lambda-steps',
        type=int,
        default=None,
        help='Number of steps for lambda axis range'
    )
    parser.add_argument(
        '--gamma-real-min',
        type=float,
        default=None,
        help='Minimum gamma_real value for axis range'
    )
    parser.add_argument(
        '--gamma-real-max',
        type=float,
        default=None,
        help='Maximum gamma_real value for axis range'
    )
    parser.add_argument(
        '--gamma-real-steps',
        type=int,
        default=None,
        help='Number of steps for gamma_real axis range'
    )
    parser.add_argument(
        '--gamma-imag-min',
        type=float,
        default=None,
        help='Minimum gamma_imag value for axis range'
    )
    parser.add_argument(
        '--gamma-imag-max',
        type=float,
        default=None,
        help='Maximum gamma_imag value for axis range'
    )
    parser.add_argument(
        '--gamma-imag-steps',
        type=int,
        default=None,
        help='Number of steps for gamma_imag axis range'
    )
    
    args = parser.parse_args()
    
    omega_range = None
    if args.omega_min is not None and args.omega_max is not None and args.omega_steps is not None:
        omega_range = (args.omega_min, args.omega_max, args.omega_steps)
    
    lambda_range = None
    if args.lambda_min is not None and args.lambda_max is not None and args.lambda_steps is not None:
        lambda_range = (args.lambda_min, args.lambda_max, args.lambda_steps)
    
    gamma_real_range = None
    if args.gamma_real_min is not None and args.gamma_real_max is not None and args.gamma_real_steps is not None:
        gamma_real_range = (args.gamma_real_min, args.gamma_real_max, args.gamma_real_steps)
    
    gamma_imag_range = None
    if args.gamma_imag_min is not None and args.gamma_imag_max is not None and args.gamma_imag_steps is not None:
        gamma_imag_range = (args.gamma_imag_min, args.gamma_imag_max, args.gamma_imag_steps)
    
    generate_multi_task_heatmaps(
        args.mg_json,
        args.imdb_json,
        args.smnist_json,
        args.output_dir,
        mg_fallback=args.mg_fallback,
        imdb_fallback=args.imdb_fallback,
        smnist_fallback=args.smnist_fallback,
        omega_range=omega_range,
        lambda_range=lambda_range,
        gamma_real_range=gamma_real_range,
        gamma_imag_range=gamma_imag_range,
    )


if __name__ == '__main__':
    main()
