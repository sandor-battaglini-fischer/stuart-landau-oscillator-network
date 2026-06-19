import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


def parse_log_file(log_path):
    epochs = []
    val_accuracies = []
    test_accuracies = []
    best_test = None
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('epoch'):
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                
                epoch_parts = parts[0].split()
                if len(epoch_parts) < 2:
                    continue
                
                try:
                    epoch = int(epoch_parts[1])
                except (ValueError, IndexError):
                    continue
                
                rest = parts[1].strip()
                rest_parts = rest.split(',')
                
                if len(rest_parts) < 2:
                    continue
                
                try:
                    val_str = rest_parts[0].strip()
                    val = float(val_str.split(':', 1)[1].strip())
                    
                    test_str = rest_parts[1].strip()
                    test = float(test_str.split(':', 1)[1].strip().split()[0])
                    
                    epochs.append(epoch)
                    val_accuracies.append(val)
                    test_accuracies.append(test)
                except (ValueError, IndexError):
                    continue
    
    if test_accuracies:
        best_test = max(test_accuracies)
    
    return epochs, val_accuracies, test_accuracies, best_test


def plot_progressions(log_items, output_path='comparison.png'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax_test, ax_val = axes
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(log_items))))
    
    for idx, (label, log_path) in enumerate(log_items):
        epochs, val_accuracies, test_accuracies, best_test = parse_log_file(log_path)
        color = colors[idx % len(colors)]
        if best_test is not None:
            test_label = f'{label} Test (best {best_test:.2f})'
        else:
            test_label = f'{label} Test'
        ax_test.plot(epochs, test_accuracies, color=color, linestyle='-', label=test_label)
        ax_val.plot(epochs, val_accuracies, color=color, linestyle=':', label=f'{label} Val')
    
    ax_test.set_xlabel('Epoch')
    ax_test.set_ylabel('Accuracy (\\%)')
    ax_test.set_title('Test Accuracy')
    ax_test.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    ax_val.set_xlabel('Epoch')
    ax_val.set_ylabel('Accuracy (\\%)')
    ax_val.set_title('Validation Accuracy')
    ax_val.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_single(label, log_path, output_path):
    epochs, val_accuracies, test_accuracies, best_test = parse_log_file(log_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, test_accuracies, color=thesis_blue, linestyle='-', label='Test')
    ax.plot(epochs, val_accuracies, color=thesis_red, linestyle=':', label='Val')
    if best_test is not None:
        ax.axhline(y=best_test, color=ifisc_green, linestyle='--', linewidth=1.2, label=f'Best Test: {best_test:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_title(f'{label} Validation and Test')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    import sys
    
    args = sys.argv[1:]
    output_path = 'comparison.png'
    log_items = []
    
    for arg in args:
        if arg.startswith('--out='):
            output_path = arg.split('=', 1)[1]
            continue
        if '=' in arg:
            label, log_path = arg.split('=', 1)
        else:
            log_path = arg
            label = Path(log_path).stem
        log_items.append((label, log_path))
    
    if not log_items:
        log_items = [('default', 'results/smnist/20260120_203300/log.txt')]
    
    if output_path == 'comparison.png' and log_items:
        first_log_dir = Path(log_items[0][1]).parent
        if str(first_log_dir) != '.':
            output_path = str(first_log_dir / 'comparison.png')
    
    plot_progressions(log_items, output_path)
    if len(log_items) > 1:
        for label, log_path in log_items:
            log_dir = Path(log_path).parent
            output_file = log_dir / f'{label}_progression.png'
            plot_single(label, log_path, str(output_file))
    else:
        label, log_path = log_items[0]
        log_dir = Path(log_path).parent
        output_file = log_dir / f'{label}_progression.png'
        plot_single(label, log_path, str(output_file))

