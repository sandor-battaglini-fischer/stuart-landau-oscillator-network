#!/usr/bin/env python3
"""Plot nonlinearity ablation curves from legacy runs or multi-seed sweep results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.nonlinearity_ablation_sweep import ABLATION_COLORS, VARIANTS, ablation_colors_for
from experiments.sweeps.sweep_common import (
    RANDOM_GUESS_BASELINE,
    add_stats_legend,
    aggregate_epoch_curves,
    load_results,
    plot_epoch_curves_with_stats,
)
from utils.plotting_utils.style import apply_style

apply_style()

ABLATION_RUNS = {
    "imdb": {
        "SL": "results/imdb/20260307_215103",
        "SL w/o tanh": "results/imdb/20260307_215433",
        "DHO": "results/imdb/20260307_215834",
        "DHO w/o tanh": "results/imdb/20260307_220047",
    },
    "smnist": {
        "SL": "results/smnist/20260307_215124",
        "SL w/o tanh": "results/smnist/20260307_215454",
        "DHO": "results/smnist/20260307_215854",
        "DHO w/o tanh": "results/smnist/20260307_220026",
    },
    "mg": {
        "SL": "results/mackey_glass/20260307_215156",
        "SL w/o tanh": "results/mackey_glass/20260307_215517",
        "DHO": "results/mackey_glass/20260307_215920",
        "DHO w/o tanh": "results/mackey_glass/20260307_220052",
    },
}

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
            if not line or not line.startswith("epoch"):
                continue
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
                    test_value_str = part.strip().split(":", 1)[1].strip().split()[0]
                    test_accuracies.append(float(test_value_str))
                    epochs.append(epoch)
                    break
    if not test_accuracies or len(epochs) != len(test_accuracies):
        return None
    sorted_data = sorted(zip(epochs, test_accuracies))
    return np.asarray([x[1] for x in sorted_data], dtype=float)


parse_smnist_log = parse_imdb_log


def load_mg_run_metrics(run_path: Path):
    metrics_path = run_path / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        data = json.load(f)
    if not data:
        return None
    test_r2 = [d["test_r2"] for d in data if "test_r2" in d]
    return np.asarray(test_r2, dtype=float) if test_r2 else None


def find_run_dir(base_dir: Path, run_name: str) -> Path | None:
    search_roots = (
        base_dir,
        base_dir / "Nonlinearity Ablation runs",
        base_dir / "runs-old",
        REPO_ROOT,
    )
    for parent in search_roots:
        candidate = parent / run_name
        if candidate.is_dir():
            return candidate
    return None


def plot_ablation_legacy(series_by_label, task_name: str, output_path: Path, use_r2: bool):
    if not series_by_label:
        print(f"  No data for {task_name}, skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (label, values) in enumerate(series_by_label.items()):
        if values is None:
            continue
        color = ABLATION_COLORS[idx % len(ABLATION_COLORS)]
        epochs = np.arange(len(values))
        ax.plot(epochs, values, label=label, color=color, linewidth=2.5)
    ax.set_xlabel("Epoch")
    title_map = {"imdb": "IMDb", "smnist": "sMNIST", "mg": "Mackey-Glass"}
    title = title_map.get(task_name, task_name)
    if use_r2:
        ax.set_ylabel("Test $R^2$")
        ax.set_title(f"{title}: Test $R^2$ vs Epoch (ablation)")
    else:
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{title}: Test Accuracy vs Epoch (ablation)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=18, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_ablation_from_sweep(results: dict, task_name: str, output_path: Path, use_r2: bool):
    sweep_key = SWEEP_TASK_MAP[task_name]
    curves = aggregate_epoch_curves(results.get(sweep_key, []), "variant")
    if not curves:
        print(f"  No sweep data for {task_name}, skipping.")
        return

    variant_order = list(VARIANTS.keys())
    ordered = {k: curves[k] for k in variant_order if k in curves}

    if use_r2:
        ordered = {
            label: {
                **stats,
                "mean": stats["mean"] * 100,
                "std": stats["std"] * 100,
                "ci95": stats["ci95"] * 100,
            }
            for label, stats in ordered.items()
        }

    guess = RANDOM_GUESS_BASELINE.get(task_name, 0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ordered_keys = list(ordered.keys())
    plot_epoch_curves_with_stats(
        ax,
        ordered,
        ablation_colors_for(ordered_keys),
        labels=ordered_keys,
        ylabel=r"test $R^2$ (\%)" if use_r2 else r"test accuracy ($\%$)",
        ylim=None if use_r2 else (guess, 100.0),
        random_guess=guess if not use_r2 else 0.0,
    )
    title_map = {"imdb": "IMDb", "smnist": "sMNIST", "mg": "Mackey-Glass"}
    ax.set_title(f"{title_map.get(task_name, task_name)}: nonlinearity ablation")
    add_stats_legend(fig, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot nonlinearity ablation training curves.")
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Path to nonlinearity_ablation_sweep results (uses multi-seed statistics)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: comparison_plots_final or sweep dir)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        results_path = sweep_dir / "results.json"
        results = load_results(results_path, list(SWEEP_TASK_MAP.values()))
        output_dir = Path(args.output_dir) if args.output_dir else sweep_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for task in ABLATION_RUNS:
            use_r2 = task == "mg"
            out_name = f"{task}_ablation_ta.png"
            plot_ablation_from_sweep(results, task, output_dir / out_name, use_r2=use_r2)
        print("\nAll ablation plots generated from sweep results.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "comparison_plots_final"
    output_dir.mkdir(exist_ok=True)

    for task, run_map in ABLATION_RUNS.items():
        print(f"\n{task.upper()}")
        series_by_label = {}
        for label, run_name in run_map.items():
            run_path = find_run_dir(base_dir, run_name)
            if run_path is None:
                print(f"Warning: run not found {run_name}")
                series_by_label[label] = None
                continue
            if task == "imdb":
                data = parse_imdb_log(run_path / "log.txt")
            elif task == "smnist":
                data = parse_smnist_log(run_path / "log.txt")
            else:
                data = load_mg_run_metrics(run_path)
            if data is not None:
                final = float(data[-1])
                metric = "test R²" if task == "mg" else "test accuracy"
                print(f"  {run_name} ({label}): final {metric} = {final:.4f}")
            series_by_label[label] = data
        use_r2 = task == "mg"
        out_name = f"{task}_ablation_ta.png"
        plot_ablation_legacy(series_by_label, task, output_dir / out_name, use_r2=use_r2)
    print("\nAll ablation TA plots generated.")


if __name__ == "__main__":
    main()
