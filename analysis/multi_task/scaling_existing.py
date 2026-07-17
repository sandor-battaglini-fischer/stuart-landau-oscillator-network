#!/usr/bin/env python3
"""Plot scaling curves from legacy runs or multi-seed sweep results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.sweep_common import (
    RANDOM_GUESS_BASELINE,
    add_stats_legend,
    aggregate_epoch_curves,
    load_results,
    plot_epoch_curves_with_stats,
)
from utils.plotting_utils.style import apply_style, mycmap

apply_style()

SWEEP_TASK_MAP = {
    "imdb": "imdb",
    "smnist": "smnist",
    "mg": "mackey_glass",
}


def parse_imdb_log(log_path: Path):
    if not log_path.exists():
        return None

    epochs = []
    test_accuracies = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("epoch"):
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                epoch_parts = parts[0].split()
                if len(epoch_parts) < 2:
                    continue

                try:
                    epoch = int(epoch_parts[1])
                except (ValueError, IndexError):
                    continue

                for part in parts[1].strip().split(","):
                    if "test:" in part:
                        test_str = part.strip()
                        test_value_str = test_str.split(":", 1)[1].strip().split()[0]
                        test = float(test_value_str)
                        epochs.append(epoch)
                        test_accuracies.append(test)
                        break

    if len(test_accuracies) == 0 or len(epochs) != len(test_accuracies):
        return None

    sorted_data = sorted(zip(epochs, test_accuracies))
    return np.asarray([x[1] for x in sorted_data], dtype=float)


parse_smnist_log = parse_imdb_log


def load_mg_results(results_path: Path, num_hidden_list):
    if not results_path.exists():
        return {}

    with open(results_path) as f:
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


def plot_task_scaling_legacy(data_by_n, task_name, output_path: Path, use_r2=False):
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

    title_map = {"imdb": "IMDb", "smnist": "sMNIST", "mg": "Mackey-Glass"}
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


def plot_task_scaling_from_sweep(results: dict, task_name: str, output_path: Path, use_r2: bool):
    sweep_key = SWEEP_TASK_MAP[task_name]
    curves = aggregate_epoch_curves(results.get(sweep_key, []), "num_hidden")
    if not curves:
        print(f"  No sweep data for {task_name}, skipping plot.")
        return

    ns = sorted(curves.keys())
    n_groups = len(ns)
    colors = [mycmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
    labels = [rf"$N={n}$" for n in ns]
    ordered = {n: curves[n] for n in ns}

    if use_r2:
        ordered = {
            n: {
                **stats,
                "mean": stats["mean"] * 100,
                "std": stats["std"] * 100,
                "ci95": stats["ci95"] * 100,
            }
            for n, stats in ordered.items()
        }

    guess = RANDOM_GUESS_BASELINE.get(task_name, 0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_epoch_curves_with_stats(
        ax,
        ordered,
        colors,
        labels=labels,
        ylabel=r"test $R^2$ (\%)" if use_r2 else r"test accuracy ($\%$)",
        ylim=None if use_r2 else (guess, 100.0),
        random_guess=guess if not use_r2 else 0.0,
    )
    title_map = {"imdb": "IMDb", "smnist": "sMNIST", "mg": "Mackey-Glass"}
    ax.set_title(f"{title_map.get(task_name, task_name)}: scaling")
    add_stats_legend(fig, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot scaling training curves.")
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Path to scaling_sweep results (uses multi-seed statistics)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        results = load_results(sweep_dir / "results.json", list(SWEEP_TASK_MAP.values()))
        output_dir = Path(args.output_dir) if args.output_dir else sweep_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for task, use_r2 in [("imdb", False), ("smnist", False), ("mg", True)]:
            plot_task_scaling_from_sweep(
                results, task, output_dir / f"{task}_scaling.png", use_r2=use_r2
            )
        print("\nAll scaling plots generated from sweep results.")
        return

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

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "comparison_plots_final"
    output_dir.mkdir(exist_ok=True)

    plot_task_scaling_legacy(imdb_data, "imdb", output_dir / "imdb_scaling.png", use_r2=False)
    plot_task_scaling_legacy(smnist_data, "smnist", output_dir / "smnist_scaling.png", use_r2=False)
    plot_task_scaling_legacy(mg_data, "mg", output_dir / "mg_scaling.png", use_r2=True)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
