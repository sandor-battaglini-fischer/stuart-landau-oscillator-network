import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import argparse
import re
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


def parse_sweep_summary(summary_path):
    results = []
    metadata = {}
    sweep_type = None
    has_epoch = None
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('Omega Sweep Summary') or 'Omega' in line and 'Sweep' in line:
            sweep_type = 'omega'
            metadata['sweep_type'] = 'omega'
        elif line.startswith('Lambda Sweep Summary') or 'Lambda' in line and 'Sweep' in line:
            sweep_type = 'lambda'
            metadata['sweep_type'] = 'lambda'
        elif line.startswith('Timestamp:'):
            metadata['timestamp'] = line.split(':', 1)[1].strip()
        elif line.startswith('Range:'):
            range_str = line.split(':', 1)[1].strip()
            parts = range_str.split(' to ')
            param_min = float(parts[0])
            param_max = float(parts[1])
            if sweep_type == 'lambda':
                metadata['lambda_min'] = param_min
                metadata['lambda_max'] = param_max
            else:
                metadata['omega_min'] = param_min
                metadata['omega_max'] = param_max
        elif line.startswith('Steps:'):
            metadata['steps'] = int(line.split(':', 1)[1].strip())
        elif 'Best Epoch' in line:
            has_epoch = True
        elif re.match(r'^-?\d+\.\d+', line):
            parts = line.split()
            is_best = '<-- BEST' in line
            
            if is_best:
                parts = [p for p in parts if p != '<--' and p != 'BEST']
            
            if len(parts) >= 3:
                param_val = float(parts[0])
                val_acc = float(parts[1])
                test_acc = float(parts[2])
                
                if len(parts) >= 4 and has_epoch is not False:
                    try:
                        best_epoch = int(parts[3])
                        has_epoch = True
                    except (ValueError, IndexError):
                        best_epoch = None
                        has_epoch = False
                else:
                    best_epoch = None
                    if has_epoch is None:
                        has_epoch = False
                
                result = {
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'is_best': is_best
                }
                
                if best_epoch is not None:
                    result['best_epoch'] = best_epoch
                
                if sweep_type == 'lambda':
                    result['lambda'] = param_val
                else:
                    result['omega'] = param_val
                
                results.append(result)
    
    if sweep_type is None:
        sweep_type = 'omega'
        metadata['sweep_type'] = 'omega'
    
    metadata['has_epoch'] = has_epoch if has_epoch is not None else False
    
    return results, metadata


def find_sweep_summaries(directory='.'):
    patterns = [
        os.path.join(directory, 'results/smnist/sweep_omega_*_summary.txt'),
        os.path.join(directory, 'results/smnist/sweep_lambda_*_summary.txt'),
        os.path.join(directory, 'results/smnist/sweep_*_summary.txt')
    ]
    summaries = set()
    for pattern in patterns:
        summaries.update(glob.glob(pattern))
    return sorted(summaries)


def plot_param_vs_accuracy(results, metadata, output_path):
    sweep_type = metadata.get('sweep_type', 'omega')
    
    if sweep_type == 'lambda':
        param_name = 'lambda'
        param_label = 'Lambda ($\\lambda$)'
        param_values = [r['lambda'] for r in results]
        param_min = metadata.get('lambda_min', min(param_values))
        param_max = metadata.get('lambda_max', max(param_values))
        title_prefix = 'Lambda Sweep'
    else:
        param_name = 'omega'
        param_label = 'Omega ($\\omega$)'
        param_values = [r['omega'] for r in results]
        param_min = metadata.get('omega_min', min(param_values))
        param_max = metadata.get('omega_max', max(param_values))
        title_prefix = 'Omega Sweep'
    
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, val_accs, color=thesis_red, label='Validation Accuracy', marker='o', markersize=6, linewidth=2)
    ax.plot(param_values, test_accs, color=ifisc_green, label='Test Accuracy', marker='s', markersize=6, linewidth=2)
    
    param_symbol = '$\\lambda$' if sweep_type == 'lambda' else '$\\omega$'
    ax.scatter([param_values[best_idx]], [val_accs[best_idx]], 
              color=thesis_blue, s=200, zorder=5, marker='*', 
              label=f'Best ({param_symbol}={param_values[best_idx]:.4f})', edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel(param_label)
    ax.set_ylabel('Accuracy (\\%)')
    param_label_plain = param_label.replace("$", "").replace("\\", "")
    ax.set_title(f'{title_prefix}: Accuracy vs {param_label_plain}\nRange: {param_min:.4f} to {param_max:.4f}, Steps: {metadata["steps"]}')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_param_vs_epoch(results, metadata, output_path):
    if not metadata.get('has_epoch', False):
        return
    
    sweep_type = metadata.get('sweep_type', 'omega')
    
    if sweep_type == 'lambda':
        param_name = 'lambda'
        param_label = 'Lambda ($\\lambda$)'
        param_values = [r['lambda'] for r in results]
        title_prefix = 'Lambda Sweep'
    else:
        param_name = 'omega'
        param_label = 'Omega ($\\omega$)'
        param_values = [r['omega'] for r in results]
        title_prefix = 'Omega Sweep'
    
    epochs = [r.get('best_epoch', 0) for r in results]
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, epochs, color=thesis_blue, marker='o', markersize=6, linewidth=2)
    
    param_symbol = '$\\lambda$' if sweep_type == 'lambda' else '$\\omega$'
    ax.scatter([param_values[best_idx]], [epochs[best_idx]], 
              color=thesis_red, s=200, zorder=5, marker='*', 
              label=f'Best ({param_symbol}={param_values[best_idx]:.4f})', edgecolors='black', linewidth=1.5)
    ax.legend()
    
    ax.set_xlabel(param_label)
    ax.set_ylabel('Best Epoch')
    param_label_plain = param_label.replace("$", "").replace("\\", "")
    ax.set_title(f'{title_prefix}: Best Epoch vs {param_label_plain}')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_comparison(results, metadata, output_path):
    sweep_type = metadata.get('sweep_type', 'omega')
    
    if sweep_type == 'lambda':
        param_values = [r['lambda'] for r in results]
        title_prefix = 'Lambda Sweep'
        param_label = 'Lambda'
    else:
        param_values = [r['omega'] for r in results]
        title_prefix = 'Omega Sweep'
        param_label = 'Omega'
    
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    x = np.arange(len(param_values))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation Accuracy', color=thesis_red, alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color=ifisc_green, alpha=0.8)
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
    bars1[best_idx].set_edgecolor(thesis_blue)
    bars1[best_idx].set_linewidth(3)
    bars2[best_idx].set_edgecolor(thesis_blue)
    bars2[best_idx].set_linewidth(3)
    
    ax.set_xlabel(f'{param_label} Index')
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_title(f'{title_prefix}: Validation vs Test Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.4f}' for p in param_values], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scatter_accuracy(results, metadata, output_path):
    sweep_type = metadata.get('sweep_type', 'omega')
    
    if sweep_type == 'lambda':
        param_values = [r['lambda'] for r in results]
        param_label = 'Lambda ($\\lambda$)'
        title_prefix = 'Lambda Sweep'
    else:
        param_values = [r['omega'] for r in results]
        param_label = 'Omega ($\\omega$)'
        title_prefix = 'Omega Sweep'
    
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(val_accs, test_accs, c=param_values, cmap=mycmap, 
                        s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    param_symbol = '$\\lambda$' if sweep_type == 'lambda' else '$\\omega$'
    ax.scatter([val_accs[best_idx]], [test_accs[best_idx]], 
              color=thesis_blue, s=300, zorder=5, marker='*', 
              label=f'Best ({param_symbol}={param_values[best_idx]:.4f})', edgecolors='black', linewidth=2)
    ax.legend()
    
    ax.plot([min(val_accs), max(val_accs)], [min(val_accs), max(val_accs)], 
           'k--', alpha=0.3, linewidth=1, label='y=x')
    
    ax.set_xlabel('Validation Accuracy (\\%)')
    ax.set_ylabel('Test Accuracy (\\%)')
    ax.set_title(f'{title_prefix}: Test vs Validation Accuracy')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(param_label)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_combined_overview(results, metadata, output_path):
    sweep_type = metadata.get('sweep_type', 'omega')
    has_epoch = metadata.get('has_epoch', False)
    
    if sweep_type == 'lambda':
        param_values = [r['lambda'] for r in results]
        param_label = 'Lambda ($\\lambda$)'
        param_label_plain = 'Lambda'
        param_min = metadata.get('lambda_min', min(param_values))
        param_max = metadata.get('lambda_max', max(param_values))
        title_prefix = 'Lambda Sweep'
    else:
        param_values = [r['omega'] for r in results]
        param_label = 'Omega ($\\omega$)'
        param_label_plain = 'Omega'
        param_min = metadata.get('omega_min', min(param_values))
        param_max = metadata.get('omega_max', max(param_values))
        title_prefix = 'Omega Sweep'
    
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
    
    if has_epoch:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        epochs = [r.get('best_epoch', 0) for r in results]
    else:
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(param_values, val_accs, color=thesis_red, label='Validation', marker='o', markersize=5, linewidth=2)
    ax1.plot(param_values, test_accs, color=ifisc_green, label='Test', marker='s', markersize=5, linewidth=2)
    ax1.scatter([param_values[best_idx]], [val_accs[best_idx]], 
               color=thesis_blue, s=150, zorder=5, marker='*', edgecolors='black', linewidth=1)
    ax1.set_xlabel(param_label)
    ax1.set_ylabel('Accuracy (\\%)')
    ax1.set_title(f'Accuracy vs {param_label_plain}')
    ax1.legend()
    
    if has_epoch:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(param_values, epochs, color=thesis_blue, marker='o', markersize=5, linewidth=2)
        ax2.scatter([param_values[best_idx]], [epochs[best_idx]], 
                   color=thesis_red, s=150, zorder=5, marker='*', edgecolors='black', linewidth=1)
        ax2.set_xlabel(param_label)
        ax2.set_ylabel('Best Epoch')
        ax2.set_title(f'Best Epoch vs {param_label_plain}')
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = None
    
    scatter = ax3.scatter(val_accs, test_accs, c=param_values, cmap=mycmap, 
                         s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax3.scatter([val_accs[best_idx]], [test_accs[best_idx]], 
               color=thesis_blue, s=200, zorder=5, marker='*', edgecolors='black', linewidth=2)
    ax3.plot([min(val_accs), max(val_accs)], [min(val_accs), max(val_accs)], 
            'k--', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Validation Accuracy (\\%)')
    ax3.set_ylabel('Test Accuracy (\\%)')
    ax3.set_title('Test vs Validation Accuracy')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label(param_label)
    
    if ax4 is not None:
        x = np.arange(len(param_values))
        width = 0.35
        bars1 = ax4.bar(x - width/2, val_accs, width, label='Validation', color=thesis_red, alpha=0.8)
        bars2 = ax4.bar(x + width/2, test_accs, width, label='Test', color=ifisc_green, alpha=0.8)
        bars1[best_idx].set_edgecolor(thesis_blue)
        bars1[best_idx].set_linewidth(3)
        bars2[best_idx].set_edgecolor(thesis_blue)
        bars2[best_idx].set_linewidth(3)
        ax4.set_xlabel(f'{param_label_plain} Index')
        ax4.set_ylabel('Accuracy (\\%)')
        ax4.set_title('Accuracy Comparison')
        ax4.set_xticks(x[::max(1, len(x)//10)])
        ax4.set_xticklabels([f'{param_values[i]:.3f}' for i in x[::max(1, len(x)//10)]], rotation=45, ha='right')
        ax4.legend()
    
    fig.suptitle(f'{title_prefix} Analysis: {param_min:.4f} to {param_max:.4f} ({metadata["steps"]} steps)', 
                 fontsize=16, y=0.995)
    
    plt.savefig(output_path)
    plt.close()


def analyze_sweep(summary_path, output_dir=None):
    results, metadata = parse_sweep_summary(summary_path)
    
    if not results:
        print(f"Warning: No results found in {summary_path}")
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(summary_path) or '.'
    
    base_name = os.path.splitext(os.path.basename(summary_path))[0]
    base_name = base_name.replace('_summary', '')
    
    sweep_type = metadata.get('sweep_type', 'omega')
    print(f"Analyzing sweep: {base_name}")
    print(f"  Sweep type: {sweep_type}")
    print(f"  Found {len(results)} results")
    if sweep_type == 'lambda':
        print(f"  Lambda range: {metadata.get('lambda_min', 'N/A'):.6f} to {metadata.get('lambda_max', 'N/A'):.6f}")
    else:
        print(f"  Omega range: {metadata.get('omega_min', 'N/A'):.6f} to {metadata.get('omega_max', 'N/A'):.6f}")
    
    sweep_type = metadata.get('sweep_type', 'omega')
    param_name = 'lambda' if sweep_type == 'lambda' else 'omega'
    
    plot_param_vs_accuracy(results, metadata, 
                          os.path.join(output_dir, f'{base_name}_accuracy_vs_{param_name}.png'))
    
    if metadata.get('has_epoch', False):
        plot_param_vs_epoch(results, metadata, 
                           os.path.join(output_dir, f'{base_name}_epoch_vs_{param_name}.png'))
    
    plot_accuracy_comparison(results, metadata, 
                            os.path.join(output_dir, f'{base_name}_accuracy_comparison.png'))
    plot_scatter_accuracy(results, metadata, 
                         os.path.join(output_dir, f'{base_name}_scatter_accuracy.png'))
    plot_combined_overview(results, metadata, 
                          os.path.join(output_dir, f'{base_name}_overview.png'))
    
    print(f"  Generated plots in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze smnist sweep results')
    parser.add_argument('--summary', type=str, default=None, 
                       help='Path to specific sweep summary file (if not provided, analyzes all found)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as summary file)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all sweep summaries found in current directory')
    
    args = parser.parse_args()
    
    if args.summary:
        analyze_sweep(args.summary, args.output_dir)
    elif args.all:
        summaries = find_sweep_summaries()
        if not summaries:
            print("No sweep summary files found!")
        else:
            print(f"Found {len(summaries)} sweep summary files")
            for summary_path in summaries:
                analyze_sweep(summary_path, args.output_dir)
                print()
    else:
        summaries = find_sweep_summaries()
        if not summaries:
            print("No sweep summary files found!")
            print("Usage: python smnist_sweep_analysis.py --all")
        elif len(summaries) == 1:
            print(f"Found 1 sweep summary file, analyzing...")
            analyze_sweep(summaries[0], args.output_dir)
        else:
            print(f"Found {len(summaries)} sweep summary files:")
            for i, s in enumerate(summaries, 1):
                print(f"  {i}. {s}")
            print("\nUse --all to analyze all, or --summary <path> for a specific file")

