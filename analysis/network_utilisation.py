import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, ifisc_green, thesis_blue, thesis_red

apply_style()

LAYERS = ["i2h", "h2h", "h2o"]
LAYER_TITLES = {"i2h": "Input weights", "h2h": "Hidden weights", "h2o": "Output weights"}
TASK_COLORS = {"IMDB": thesis_red, "sMNIST": thesis_blue, "Mackey-Glass": ifisc_green}
SWEEP_TASK_LABELS = {"imdb": "IMDB", "smnist": "sMNIST", "mackey_glass": "Mackey-Glass"}


def load_run(label, run_dir):
    with open(os.path.join(run_dir, "manifold_dimension_results.json")) as f:
        manifold = json.load(f)
    with open(os.path.join(run_dir, "parameters.json")) as f:
        history = sorted(json.load(f), key=lambda e: e["epoch"])

    layers = {}
    for layer in LAYERS:
        rows = []
        for entry in history:
            w = np.asarray(entry["params"].get(f"{layer}_weight", []), dtype=float).ravel()
            if w.size == 0 or np.max(np.abs(w)) == 0:
                continue
            thresh = 0.2 * np.max(np.abs(w))
            rows.append((entry["epoch"], np.mean(np.abs(w) > thresh), np.mean(w > thresh), np.mean(w < -thresh)))
        if rows:
            cols = np.asarray(rows, dtype=float).T
            layers[layer] = dict(zip(("epochs", "frac_active", "frac_pos", "frac_neg"), cols))

    return {"label": label, "corr_dim": manifold.get("correlation_dim"),
            "eff_dim": manifold.get("effective_dim_pca"), "layers": layers}


def effective_system_dims(results_paths, threshold=0.90):
    """2N for the smallest N whose seed-averaged score is within `threshold` of the best N."""
    scores = defaultdict(lambda: defaultdict(list))
    for path in results_paths:
        with open(path) as f:
            results = json.load(f)
        for task, records in results.items():
            for r in records:
                if r.get("metric") is not None and r.get("num_hidden") is not None:
                    scores[task][int(r["num_hidden"])].append(float(r["metric"]))

    dims = {}
    for task, by_n in scores.items():
        mean_by_n = {n: np.mean(v) for n, v in by_n.items()}
        best = max(mean_by_n.values())
        dims[SWEEP_TASK_LABELS.get(task, task)] = 2 * min(n for n, v in mean_by_n.items() if v >= threshold * best)
    return dims


def plot_dimension_comparison(runs, system_dims, out_dir):
    labels = [r["label"] for r in runs]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.bar(x - width, [r["corr_dim"] for r in runs], width, label="Correlation dim", color=thesis_blue)
    ax.bar(x, [r["eff_dim"] for r in runs], width, label="Effective PCA dim", color=thesis_red)
    if system_dims:
        ax.bar(x + width, [system_dims.get(l, np.nan) for l in labels], width,
               label="Effective system dim (2N)", color=ifisc_green)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dimension")
    ax.set_title("Manifold dimensions across tasks")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task_dimension_comparison.png"), dpi=600, transparent=True)
    plt.close(fig)


def _layer_figure(runs, out_path, ylabel, plot_fn, legend_y, legend_ncol=None, rect_top=0.85, dpi=600):
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 8))
    for idx, (ax, layer) in enumerate(zip(axes, LAYERS)):
        for run in runs:
            if layer in run["layers"]:
                plot_fn(ax, run, run["layers"][layer])
        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(LAYER_TITLES[layer])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                   ncol=legend_ncol or len(handles))
    fig.tight_layout(rect=[0, 0, 1, rect_top])
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def plot_weight_usage(runs, out_dir):
    def active(ax, run, s):
        ax.plot(s["epochs"], s["frac_active"], label=run["label"], color=TASK_COLORS.get(run["label"]), linewidth=2)

    def signed(ax, run, s):
        color = TASK_COLORS.get(run["label"])
        ax.plot(s["epochs"], s["frac_pos"], label=f"{run['label']} strong +", color=color, linewidth=2)
        ax.plot(s["epochs"], s["frac_neg"], label=f"{run['label']} strong -", color=color, linewidth=1.5, linestyle="--")

    _layer_figure(runs, os.path.join(out_dir, "task_weight_usage_active.png"),
                  "Fraction of active weights", active, legend_y=1.15, dpi=300)
    _layer_figure(runs, os.path.join(out_dir, "task_weight_usage_strong_sign.png"),
                  "Fraction of strongly signed weights", signed, legend_y=1.18, legend_ncol=3, rect_top=0.83)


def layer_variances(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    prefix = "slon." if "slon.i2h.weight" in state else ""
    return {layer: float(state[f"{prefix}{layer}.weight"].float().var(unbiased=False)) for layer in LAYERS}


def load_variance_group(label, pattern, checkpoint):
    """Per-seed weight variance before (initial_model.pt) and after training for all run dirs matching pattern."""
    before, after = [], []
    for run_dir in sorted(glob.glob(pattern)):
        init_path = os.path.join(run_dir, "initial_model.pt")
        final_path = os.path.join(run_dir, f"{checkpoint}_model.pt")
        if os.path.isfile(init_path) and os.path.isfile(final_path):
            before.append(layer_variances(init_path))
            after.append(layer_variances(final_path))
    if not before:
        raise FileNotFoundError(f"no runs with initial_model.pt and {checkpoint}_model.pt match {pattern}")
    return {"label": label, "n": len(before),
            "before": {l: np.array([v[l] for v in before]) for l in LAYERS},
            "after": {l: np.array([v[l] for v in after]) for l in LAYERS}}


def _mean_ci(values):
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return values.mean(), 1.96 * sem


def plot_weight_variance(groups, out_dir):
    """Weight variance per layer before and after training, mean and 95% CI over seeds."""
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 6), sharey=True)
    x = np.arange(len(groups))
    for idx, (ax, layer) in enumerate(zip(axes, LAYERS)):
        for offset, key, color in ((-0.175, "before", "0.8"), (0.175, "after", "0.3")):
            stats = [_mean_ci(g[key][layer]) for g in groups]
            ax.bar(x + offset, [m for m, _ in stats], 0.35, yerr=[c for _, c in stats], color=color,
                   edgecolor="0.2", linewidth=0.5, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g['label']}\n($n={g['n']}$)" for g in groups])
        ax.set_xlabel("Task")
        if idx == 0:
            ax.set_ylabel("Weight variance")
        ax.set_title(LAYER_TITLES[layer])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.grid(True, axis="y", alpha=0.3)
    fig.legend(handles=[Patch(facecolor="0.8", edgecolor="0.2", label="Before Training"),
                        Patch(facecolor="0.3", edgecolor="0.2", label="After Training"),
                        plt.Line2D([], [], color="black", marker="_", linestyle="-", label=r"95\% CI")],
               loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out_dir, "task_weight_variance_before_after.png"), dpi=600, transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare manifold dimension and weight usage across tasks.")
    parser.add_argument("--runs", nargs="*", default=[], metavar="LABEL=RUN_DIR",
                        help="one full training run per task (parameters.json, manifold_dimension_results.json); "
                             "labels IMDB, sMNIST, Mackey-Glass")
    parser.add_argument("--scaling-results", nargs="*", default=[],
                        help="results.json files from experiments/sweeps/scaling_sweep.py")
    parser.add_argument("--variance-runs", nargs="*", default=[], metavar="LABEL=GLOB",
                        help="seed runs per task for the before/after weight variance, e.g. 'IMDb=results/s6/imdb_*'")
    parser.add_argument("--checkpoint", choices=["last", "best"], default="last",
                        help="trained weights used as 'after training'")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "network_utilisation"))
    args = parser.parse_args()
    if not args.runs and not args.variance_runs:
        parser.error("give --runs and/or --variance-runs")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.runs:
        runs = [load_run(*spec.rsplit("=", 1)) for spec in args.runs]
        plot_dimension_comparison(runs, effective_system_dims(args.scaling_results), args.output_dir)
        plot_weight_usage(runs, args.output_dir)
    if args.variance_runs:
        groups = [load_variance_group(*spec.rsplit("=", 1), args.checkpoint) for spec in args.variance_runs]
        plot_weight_variance(groups, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
