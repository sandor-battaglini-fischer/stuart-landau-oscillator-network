import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


def parse_imdb_log(log_path: Path):
    """Parse IMDb log.txt to extract test accuracy per epoch."""
    if not log_path.exists():
        return None
    
    epochs = []
    test_accuracies = []
    
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
                
                for part in rest_parts:
                    if 'test:' in part:
                        test_str = part.strip()
                        test_value_str = test_str.split(':', 1)[1].strip().split()[0]
                        test = float(test_value_str)
                        epochs.append(epoch)
                        test_accuracies.append(test)
                        break
    
    if len(test_accuracies) == 0:
        return None
    
    if len(epochs) != len(test_accuracies):
        return None
    
    sorted_data = sorted(zip(epochs, test_accuracies))
    sorted_epochs, sorted_test = zip(*sorted_data)
    
    return np.asarray(sorted_test, dtype=float)


def parse_smnist_log(log_path: Path):
    """Parse sMNIST log.txt to extract test accuracy per epoch."""
    if not log_path.exists():
        return None
    
    epochs = []
    test_accuracies = []
    
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
                
                for part in rest_parts:
                    if 'test:' in part:
                        test_str = part.strip()
                        test_value_str = test_str.split(':', 1)[1].strip().split()[0]
                        test = float(test_value_str)
                        epochs.append(epoch)
                        test_accuracies.append(test)
                        break
    
    if len(test_accuracies) == 0:
        return None
    
    if len(epochs) != len(test_accuracies):
        return None
    
    sorted_data = sorted(zip(epochs, test_accuracies))
    sorted_epochs, sorted_test = zip(*sorted_data)
    
    return np.asarray(sorted_test, dtype=float)


def load_mg_results(results_path: Path, num_hidden_list):
    """Load Mackey-Glass results from full_results.json for specified node counts."""
    if not results_path.exists():
        return {}
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    results = data.get("results", [])
    by_n = {}
    
    for r in results:
        params = r.get("parameters", {})
        n = params.get("num_hidden")
        if n is None or n not in num_hidden_list:
            continue
        
        test_r2 = r.get("test_r2_scores", [])
        if not test_r2:
            continue
        
        if n not in by_n:
            by_n[n] = np.asarray(test_r2, dtype=float)
        else:
            existing = by_n[n]
            new_scores = np.asarray(test_r2, dtype=float)
            if len(new_scores) > len(existing) or np.max(new_scores) > np.max(existing):
                by_n[n] = new_scores
    
    return by_n


def plot_task_scaling(data_by_n, task_name, output_path: Path, use_r2=False):
    """Plot scaling curves for a single task."""
    if not data_by_n:
        print(f"  No data found for {task_name}, skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ns = sorted(data_by_n.keys())
    n_results = len(ns)
    
    for idx, n in enumerate(ns):
        test_values = data_by_n[n]
        epochs = np.arange(len(test_values))
        frac = idx / max(1, n_results - 1) if n_results > 1 else 0.5
        color = mycmap(frac)
        
        label = rf"$N={n}$"
        ax.plot(epochs, test_values, label=label, color=color, linewidth=2.5)
    
    ax.set_xlabel("Epoch")
    
    title_map = {
        "imdb": "IMDb",
        "smnist": "sMNIST",
        "mg": "Mackey-Glass",
    }
    title = title_map.get(task_name, task_name)
    
    if use_r2:
        ax.set_ylabel("Test $R^2$")
        ax.set_title(f"{title}: Test $R^2$ vs Epoch")
    else:
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{title}: Test Accuracy vs Epoch")
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=18, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    base_dir = Path(__file__).parent
    runs_old_dir = base_dir / "runs-old"
    
    imdb_dirs = {
        1: "results/imdb/20251207_231522",
        2: "results/imdb/20251207_231503",
        4: "results/imdb/20251208_104915",
        9: "results/imdb/20251207_221754",
        50: "results/imdb/20251207_221628",
        128: "results/imdb/20251210_172243",
    }
    
    smnist_dirs = {
        1: "results/smnist/20251218_114219",
        2: "results/smnist/20251218_114204",
        4: "results/smnist/20251218_114148",
        9: "results/smnist/20251218_114028",
        16: "results/smnist/20260312_220150",
        25: "results/smnist/20260312_110818",
        50: "results/smnist/20251218_115016",
        128: "results/smnist/20251218_120601",
    }
    
    mg_results_file = base_dir / "results/mackey_glass/comparison_20260203_184809" / "full_results.json"
    mg_node_counts = [1, 2, 4, 9, 50, 128]
    
    imdb_data = {}
    for n, dirname in imdb_dirs.items():
        log_path = runs_old_dir / dirname / "log.txt"
        test_acc = parse_imdb_log(log_path)
        if test_acc is not None:
            imdb_data[n] = test_acc
            print(f"Loaded IMDb N={n}: {len(test_acc)} epochs")
        else:
            print(f"Warning: Could not load IMDb N={n} from {log_path}")
    
    smnist_data = {}
    for n, dirname in smnist_dirs.items():
        log_path = runs_old_dir / dirname / "log.txt"
        test_acc = parse_smnist_log(log_path)
        if test_acc is not None:
            smnist_data[n] = test_acc
            print(f"Loaded sMNIST N={n}: {len(test_acc)} epochs")
        else:
            print(f"Warning: Could not load sMNIST N={n} from {log_path}")
    
    mg_data = load_mg_results(mg_results_file, mg_node_counts)
    for n, test_r2 in mg_data.items():
        print(f"Loaded Mackey-Glass N={n}: {len(test_r2)} epochs")
    
    output_dir = base_dir / "comparison_plots_final"
    output_dir.mkdir(exist_ok=True)
    
    plot_task_scaling(imdb_data, "imdb", output_dir / "imdb_scaling.png", use_r2=False)
    plot_task_scaling(smnist_data, "smnist", output_dir / "smnist_scaling.png", use_r2=False)
    plot_task_scaling(mg_data, "mg", output_dir / "mg_scaling.png", use_r2=True)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
