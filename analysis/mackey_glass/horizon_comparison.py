import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from training.train_mackey_glass import build_dataloaders

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


def load_results_from_directory(data_dir):
    full_results_file = os.path.join(data_dir, "full_results.json")
    if not os.path.exists(full_results_file):
        raise FileNotFoundError(f"full_results.json not found in {data_dir}")
    
    with open(full_results_file, "r") as f:
        full_data = json.load(f)
    
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
    
    config = full_data.get('config', {})
    
    remove_freqs = None
    for result in all_results:
        if 'remove_top_n_freqs' in result['param_overrides']:
            remove_freqs = result['param_overrides']['remove_top_n_freqs']
            break
    
    if remove_freqs is None:
        remove_freqs = config.get('remove_top_n_freqs', 0)
    
    return all_results, config, remove_freqs


def create_horizon_comparison_plot(results_by_freqs, output_dir):
    if not results_by_freqs:
        return
    
    remove_freqs_values = sorted(results_by_freqs.keys())
    
    horizon_data_by_freqs = {}
    for remove_freqs in remove_freqs_values:
        horizon_results = [
            r for r in results_by_freqs[remove_freqs]
            if 'horizon' in r['param_overrides']
        ]
        
        if not horizon_results:
            continue
        
        horizon_data = {}
        for result in horizon_results:
            horizon = result['param_overrides']['horizon']
            final_r2 = result['test_r2_scores'][-1]
            
            if horizon not in horizon_data:
                horizon_data[horizon] = []
            horizon_data[horizon].append(final_r2)
        
        if horizon_data:
            horizon_data_by_freqs[remove_freqs] = horizon_data
    
    if not horizon_data_by_freqs:
        return
    
    all_horizons = set()
    for horizon_data in horizon_data_by_freqs.values():
        all_horizons.update(horizon_data.keys())
    horizons = sorted(all_horizons)
    
    colors_map = {0: thesis_blue, 1: ifisc_green, 2: thesis_red}
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for idx, remove_freqs in enumerate(remove_freqs_values):
        if remove_freqs not in horizon_data_by_freqs:
            continue
        
        horizon_data = horizon_data_by_freqs[remove_freqs]
        mean_r2_values = []
        valid_horizons = []
        
        for horizon in horizons:
            if horizon in horizon_data:
                r2_values = horizon_data[horizon]
                mean_r2 = np.mean(r2_values)
                mean_r2_values.append(mean_r2)
                valid_horizons.append(horizon)
        
        if not valid_horizons:
            continue
        
        color = colors_map.get(idx % len(colors_map), thesis_blue)
        label = f"$R={remove_freqs}$"
        
        ax.plot(
            valid_horizons, mean_r2_values,
            color=color, linestyle='-', marker='o',
            markersize=10, linewidth=3.5,
            label=label, alpha=0.9, zorder=3-idx
        )
    
    ax.set_xlabel("Horizon ($H$)", fontsize=24)
    ax.set_ylabel("Final Test $R^2$", fontsize=24)
    ax.set_title("Final Test Accuracy vs Prediction Horizon (Multiple Frequency Removals)", fontsize=28)
    ax.legend(loc="best", fontsize=20, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    max_horizon = max(horizons) if horizons else 0
    tick_step = 10
    tick_positions = list(range(0, max_horizon + tick_step, tick_step))
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    fig.tight_layout()
    plot_name = f"{output_dir}/horizon_comparison_multi_freqs.png"
    fig.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved horizon comparison plot: {plot_name}")


def create_horizon_comparison_plot_limited(results_by_freqs, output_dir, max_horizon=10):
    if not results_by_freqs:
        return
    
    remove_freqs_values = sorted(results_by_freqs.keys())
    
    horizon_data_by_freqs = {}
    for remove_freqs in remove_freqs_values:
        horizon_results = [
            r for r in results_by_freqs[remove_freqs]
            if 'horizon' in r['param_overrides'] and r['param_overrides']['horizon'] <= max_horizon
        ]
        
        if not horizon_results:
            continue
        
        horizon_data = {}
        for result in horizon_results:
            horizon = result['param_overrides']['horizon']
            final_r2 = result['test_r2_scores'][-1]
            
            if horizon not in horizon_data:
                horizon_data[horizon] = []
            horizon_data[horizon].append(final_r2)
        
        if horizon_data:
            horizon_data_by_freqs[remove_freqs] = horizon_data
    
    if not horizon_data_by_freqs:
        return
    
    all_horizons = set()
    for horizon_data in horizon_data_by_freqs.values():
        all_horizons.update(horizon_data.keys())
    horizons = sorted([h for h in all_horizons if h <= max_horizon])
    
    colors_map = {0: thesis_blue, 1: ifisc_green, 2: thesis_red}
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for idx, remove_freqs in enumerate(remove_freqs_values):
        if remove_freqs not in horizon_data_by_freqs:
            continue
        
        horizon_data = horizon_data_by_freqs[remove_freqs]
        mean_r2_values = []
        valid_horizons = []
        
        for horizon in horizons:
            if horizon in horizon_data:
                r2_values = horizon_data[horizon]
                mean_r2 = np.mean(r2_values)
                mean_r2_values.append(mean_r2)
                valid_horizons.append(horizon)
        
        if not valid_horizons:
            continue
        
        color = colors_map.get(idx % len(colors_map), thesis_blue)
        label = f"$R={remove_freqs}$"
        
        ax.plot(
            valid_horizons, mean_r2_values,
            color=color, linestyle='-', marker='o',
            markersize=10, linewidth=3.5,
            label=label, alpha=0.9, zorder=3-idx
        )
    
    ax.set_xlabel("Horizon ($H$)", fontsize=24)
    ax.set_ylabel("Final Test $R^2$", fontsize=24)
    ax.set_title(f"Final Test Accuracy vs Prediction Horizon (Multiple Frequency Removals, $H \\leq {max_horizon}$)", fontsize=28)
    ax.legend(loc="best", fontsize=20, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    tick_step = 10
    tick_positions = list(range(0, max_horizon + tick_step, tick_step))
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    fig.tight_layout()
    plot_name = f"{output_dir}/horizon_comparison_multi_freqs_limited.png"
    fig.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved horizon comparison plot (limited to H≤{max_horizon}): {plot_name}")


def create_horizon_predictions_comparison_plot(results_by_freqs, output_dir, configs_by_freqs):
    if not results_by_freqs:
        return
    
    remove_freqs_values = sorted(results_by_freqs.keys())
    
    for remove_freqs in remove_freqs_values:
        horizon_results = [
            r for r in results_by_freqs[remove_freqs]
            if 'horizon' in r['param_overrides']
        ]
        
        if not horizon_results:
            continue
        
        horizon_groups = {}
        for result in horizon_results:
            horizon = result['param_overrides']['horizon']
            if horizon not in horizon_groups:
                horizon_groups[horizon] = []
            horizon_groups[horizon].append(result)
        
        if not horizon_groups:
            continue
        
        horizons = sorted(horizon_groups.keys())
        
        result = horizon_results[0]
        full_series = result.get('full_series')
        test_start_idx = result.get('test_start_idx')
        
        if full_series is None or test_start_idx is None:
            config = configs_by_freqs.get(remove_freqs, {})
            try:
                from training.train_mackey_glass import build_dataloaders
                _, _, test_loader, full_series, _, _ = build_dataloaders(
                    series_length=config.get('series_length', 20000),
                    input_length=config.get('input_length', 100),
                    horizon=horizons[0],
                    batch_size=config.get('batch_size', 64),
                    val_fraction=config.get('val_fraction', 0.1),
                    test_fraction=config.get('test_fraction', 0.1),
                    tau=config.get('mg_tau', 17.0),
                    delta_t=config.get('mg_delta_t', 1.0),
                    beta=config.get('mg_beta', 0.2),
                    gamma_mg=config.get('mg_gamma', 0.1),
                    n=config.get('mg_n', 10.0),
                    x0=config.get('mg_x0', 1.2),
                    seed=config.get('seed', 1),
                    remove_top_n_freqs=remove_freqs,
                )
                
                series_length = config.get('series_length', 20000)
                test_fraction = config.get('test_fraction', 0.1)
                val_fraction = config.get('val_fraction', 0.1)
                total_len = int(series_length * (1 - test_fraction - val_fraction))
                val_len = int(series_length * val_fraction)
                test_start_idx = total_len + val_len
                
                for r in horizon_results:
                    r['full_series'] = full_series
                    r['test_start_idx'] = test_start_idx
            except Exception as e:
                print(f"Warning: Could not regenerate series for remove_freqs={remove_freqs}: {e}")
                continue
        
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
        
        custom_cmap = mcolors.LinearSegmentedColormap(f'horizon_pred_cmap_{remove_freqs}', cdict, N=256)
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
        
        ax.set_xlabel('Time (simulation steps)', fontsize=24)
        ax.set_ylabel('Value', fontsize=24)
        ax.set_title(f'Test Segment: Horizon Points (Remove Top $R={remove_freqs}$ Frequencies)', fontsize=28, fontweight='bold')
        ax.legend(loc='upper right', fontsize=20, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        fig.tight_layout()
        plot_name = f"{output_dir}/horizon_predictions_R{remove_freqs}.png"
        fig.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved horizon predictions plot for R={remove_freqs}: {plot_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create horizon comparison plots from multiple directories with different remove_top_n_freqs values"
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing full_results.json files (one for each remove_top_n_freqs value)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: first input directory)"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.dirs[0]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    results_by_freqs = {}
    configs_by_freqs = {}
    
    print("Loading results from directories...")
    print("=" * 80)
    
    for data_dir in args.dirs:
        if not os.path.isdir(data_dir):
            print(f"Warning: {data_dir} is not a directory, skipping...")
            continue
        
        try:
            all_results, config, remove_freqs = load_results_from_directory(data_dir)
            results_by_freqs[remove_freqs] = all_results
            configs_by_freqs[remove_freqs] = config
            print(f"Loaded {len(all_results)} results from {data_dir} (R={remove_freqs})")
        except Exception as e:
            print(f"Error loading {data_dir}: {e}")
            continue
    
    if not results_by_freqs:
        print("No results loaded. Exiting.")
        return
    
    print("\n" + "=" * 80)
    print(f"Loaded results for {len(results_by_freqs)} different remove_freqs values: {sorted(results_by_freqs.keys())}")
    print("=" * 80)
    
    print("\nGenerating horizon comparison plots...")
    create_horizon_comparison_plot(results_by_freqs, args.output_dir)
    create_horizon_comparison_plot_limited(results_by_freqs, args.output_dir, max_horizon=25)
    create_horizon_predictions_comparison_plot(results_by_freqs, args.output_dir, configs_by_freqs)
    
    print("\n" + "=" * 80)
    print("Plots generated successfully!")
    print(f"Plots saved in: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
