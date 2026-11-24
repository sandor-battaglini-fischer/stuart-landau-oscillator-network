#!/usr/bin/env python3
"""Sweep omega or alpha on IMDb, sMNIST and Mackey-Glass (H=1 and H=25) with several seeds per value.

Defaults follow the omega sweep of the supplemental material (Table V: N=100, 25 epochs, 10 seeds).

  python experiments/sweeps/param_sweep.py --param omega --sweep-dir results/sweeps/omega
  python experiments/sweeps/param_sweep.py --param omega --sweep-dir results/sweeps/omega --plot-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.sweep_common import (
    aggregate_scalar_results,
    cmd_with_overrides,
    load_results,
    parse_classification_metric,
    parse_mackey_glass_best_metrics,
    run_training_cmd,
    save_csv,
    save_results,
)
from utils.plotting_utils.style import ifisc_green, thesis_blue, thesis_red

TASK_ARGS = {
    "imdb": ["training/train_imdb.py", "--lr", "1e-3", "--lambda-param", "-0.08", "--gamma-real", "-0.1",
             "--gamma-imag", "0.1"],
    "smnist": ["training/train_smnist.py", "--lr", "1e-2", "--lambda-param", "0.0", "--gamma-real", "-0.1",
               "--gamma-imag", "0.0"],
    "mackey_glass": ["training/train_mackey_glass.py", "--lr", "1e-2", "--omega", "0.15", "--lambda-param", "-0.2",
                     "--gamma-real", "-0.1", "--gamma-imag", "0.0", "--mg-tau", "17", "--input-length", "100"],
}
NAMES = {"imdb": "IMDB", "smnist": "sMNIST", "mackey_glass": "Mackey-Glass"}

# value of each parameter used everywhere else (dashed lines) and the default (min, max, step) grid
PARAMS = {
    "omega": {"symbol": r"\omega", "center": {"imdb": 2 * np.pi / 175, "smnist": 2 * np.pi / 28, "mackey_glass": 0.15},
              "grid": (0.0, 0.9, 0.1)},
    "alpha": {"symbol": r"\alpha", "center": {"imdb": 0.5, "smnist": 0.05, "mackey_glass": 0.04},
              "grid": (0.0, 5.0, None)},
}


def arange(lo, hi, step):
    values = [round(float(v), 10) for v in np.arange(lo, hi + 0.5 * step, step)]
    return [v for v in values if v <= hi + 1e-9]


def value_grid(lo, hi, step):
    if step is not None:
        return arange(lo, hi, step)
    # default: 0.1 steps up to 1, then 0.5 steps
    fine = arange(lo, min(hi, 1.0), 0.1) if lo < 1.0 else []
    coarse = arange(max(lo, 1.0), hi, 0.5) if hi > 1.0 else []
    if fine and coarse and abs(coarse[0] - fine[-1]) < 1e-9:
        coarse = coarse[1:]
    return fine + coarse


def series_list(tasks, mg_horizons):
    """(task, horizon) pairs: one curve per classification task, one per Mackey-Glass horizon."""
    return [(t, h) for t in tasks for h in (mg_horizons if t == "mackey_glass" else [None])]


def run_single(args, task, horizon, value, seed, sweep_dir):
    tag = f"{task}" + (f"_h{horizon}" if horizon is not None else "")
    run_name = f"{tag}_{args.param}{value:.6f}_seed{seed}"
    output_dir = sweep_dir / "runs" / run_name
    overrides = {"--seed": seed, f"--{args.param}": value, "--num-hidden": args.num_hidden, "--epochs": args.epochs,
                 "--early-stop-patience": args.early_stop_patience, "--output-dir": output_dir}
    if horizon is not None:
        overrides["--horizon"] = horizon
    flags = [] if args.with_manifold else ["--sweep-mode"]
    cmd = [sys.executable, *cmd_with_overrides(TASK_ARGS[task] + flags, overrides)]
    print(f"\n{run_name}")
    record = run_training_cmd(cmd, sweep_dir / "logs" / f"{run_name}.log", "", task, "metric",
                              args.dry_run, output_dir=output_dir)
    record.update(task=task, horizon=horizon, seed=seed, **{args.param: value})
    if record["status"] != "ok":
        return record

    # score a run by the test metric at its best validation epoch
    if task == "mackey_glass":
        best = parse_mackey_glass_best_metrics(output_dir / "metrics.json")
        record.update(best)
        record["metric"] = best["test_r2"]
    else:
        record["metric"] = parse_classification_metric((output_dir / "log.txt").read_text())
        record["test_acc"] = record["metric"]
    record.pop("epoch_curve", None)
    if record["metric"] is None:
        record["status"] = "parse_failed"
    return record


def plot_results(param, results, series, plot_path):
    """Accuracy on the left axis, Mackey-Glass R^2 on the right, mean and 95% CI over seeds."""
    spec = PARAMS[param]
    fig, ax_acc = plt.subplots(figsize=(9, 6))
    ax_r2 = ax_acc.twinx()
    # darker red for shorter horizons
    mg_horizons = [h for t, h in series if t == "mackey_glass"]
    shades = np.linspace(0.0, 0.55, max(len(mg_horizons), 1))
    colors = {("imdb", None): thesis_blue, ("smnist", None): ifisc_green}
    colors.update({("mackey_glass", h): tuple((1 - s) * c + s for c in thesis_red) for h, s in zip(mg_horizons, shades)})

    handles, centers_drawn = [], set()
    for task, horizon in series:
        entries = [e for e in results.get(task, []) if e.get("horizon") == horizon]
        summary = aggregate_scalar_results(entries, param)
        if not summary:
            continue
        ax = ax_r2 if task == "mackey_glass" else ax_acc
        color = colors[(task, horizon)]
        xs = [row[param] for row in summary]
        means = np.array([row["mean"] for row in summary])
        ci = np.array([row["ci95"] for row in summary])
        label = NAMES[task] + (f" $H = {horizon}$" if horizon is not None else "")
        handles += ax.plot(xs, means, "o-", color=color, linewidth=2, markersize=5, label=label)
        ax.fill_between(xs, means - ci, means + ci, color=color, alpha=0.25, linewidth=0)
        if task not in centers_drawn:
            ax_acc.axvline(spec["center"][task], color=color, linestyle="--", linewidth=1.5)
            centers_drawn.add(task)
    ax_acc.set_xlabel(rf"${spec['symbol']}$")
    ax_acc.set_ylabel(r"Test accuracy (\%)")
    ax_r2.set_ylabel(r"Test $R^2$")
    ax_acc.set_ylim(50, 100)
    ax_r2.set_ylim(0, 1)
    ax_r2.grid(False)
    if handles:
        ax_acc.legend(handles=handles, loc="lower left", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep omega or alpha across tasks.")
    parser.add_argument("--param", choices=list(PARAMS), required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_ARGS), default=list(TASK_ARGS))
    parser.add_argument("--mg-horizons", nargs="+", type=int, default=[1, 25])
    parser.add_argument("--min", type=float, default=None, help="override the default grid")
    parser.add_argument("--max", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--num-hidden", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=10, help="seeds per value")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--with-manifold", action="store_true", help="full training outputs instead of --sweep-mode")
    args = parser.parse_args()

    param = args.param
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else REPO_ROOT / "results" / "sweeps" / f"{param}_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"
    results = load_results(results_path, list(TASK_ARGS))
    seeds = args.seeds or list(range(1, args.num_runs + 1))
    series = series_list(args.tasks, args.mg_horizons)

    if not args.plot_only:
        lo, hi, step = PARAMS[param]["grid"]
        values = value_grid(lo if args.min is None else args.min, hi if args.max is None else args.max,
                            step if args.step is None else args.step)
        config = {k: v for k, v in vars(args).items() if k not in ("dry_run", "plot_only", "skip_existing")}
        with (sweep_dir / "sweep_config.json").open("w") as f:
            json.dump({**config, "seeds": seeds, "values": values}, f, indent=2)

        def key(e):
            return round(float(e[param]), 10), int(e["seed"]), e.get("horizon")

        for task, horizon in series:
            entries = results.setdefault(task, [])
            done = {key(e) for e in entries if e.get("status") == "ok"}
            for value in values:
                for seed in seeds:
                    k = (round(value, 10), seed, horizon)
                    if args.skip_existing and k in done:
                        continue
                    record = run_single(args, task, horizon, value, seed, sweep_dir)
                    if record["status"] == "dry_run":
                        continue
                    entries[:] = [e for e in entries if key(e) != k]
                    entries.append(record)
                    save_results(results_path, results)
                    save_csv(results_path, results, extra_fields=[param, "horizon"])

    summary = {f"{t}" + (f"_h{h}" if h is not None else ""):
               aggregate_scalar_results([e for e in results.get(t, []) if e.get("horizon") == h], param)
               for t, h in series}
    with (sweep_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    plot_results(param, results, series, sweep_dir / f"{param}_sweep_performance.png")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
