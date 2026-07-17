#!/usr/bin/env python3
"""Sweep hidden size N across IMDB, sMNIST, and Mackey-Glass with multi-seed statistics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.sweep_common import (
    aggregate_epoch_curves,
    aggregate_scalar_results,
    cmd_with_overrides,
    load_results,
    plot_epoch_curves_with_stats,
    plot_scalar_sweep,
    run_training_cmd,
    save_csv,
    save_results,
    save_summary,
)
from utils.plotting_utils.style import mycmap

GROUP_KEY = "num_hidden"


@dataclass
class TaskConfig:
    name: str
    output_glob: str
    metric_label: str
    epoch_ylabel: str
    use_r2: bool
    random_guess: float
    ylim: tuple[float, float]
    default_hidden: list[int] = field(default_factory=list)
    base_cmd: list[str] = field(default_factory=list)


TASKS: dict[str, TaskConfig] = {
    "imdb": TaskConfig(
        name="IMDB",
        output_glob="results/imdb/*",
        metric_label=r"Test accuracy ($\%$)",
        epoch_ylabel=r"Test accuracy ($\%$)",
        use_r2=False,
        random_guess=50.0,
        ylim=(50.0, 90.0),
        default_hidden=[1, 2, 4, 9, 50, 128],
        base_cmd=[
            sys.executable,
            "training/train_imdb.py",
            "--epochs", "10",
            "--batch-size", "64",
            "--lr", "1e-3",
            "--embed-dim", "100",
            "--max-len", "175",
            "--glove", "glove.6B.100d.txt",
            "--lambda-param", "-0.05",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.1",
            "--omega", "0.035904",
            "--sweep-mode",
        ],
    ),
    "smnist": TaskConfig(
        name="sMNIST",
        output_glob="results/smnist/*",
        metric_label=r"Test accuracy ($\%$)",
        epoch_ylabel=r"Test accuracy ($\%$)",
        use_r2=False,
        random_guess=10.0,
        ylim=(10.0, 100.0),
        default_hidden=[1, 2, 4, 9, 16, 25, 50, 128],
        base_cmd=[
            sys.executable,
            "training/train_smnist.py",
            "--lambda-param", "0.0",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--epochs", "10",
            "--omega", "0.224",
            "--sweep-mode",
        ],
    ),
    "mackey_glass": TaskConfig(
        name="Mackey-Glass",
        output_glob="results/mackey_glass/*",
        metric_label=r"Test $R^2$",
        epoch_ylabel=r"Test $R^2$",
        use_r2=True,
        random_guess=0.0,
        ylim=(0.0, 1.0),
        default_hidden=[1, 2, 4, 9, 50, 128],
        base_cmd=[
            sys.executable,
            "training/train_mackey_glass.py",
            "--epochs", "10",
            "--input-length", "100",
            "--gamma", "0.01",
            "--lambda-param", "-0.1",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--omega", "0.15",
            "--mg-tau", "17.0",
            "--horizon", "1",
            "--series-length", "20000",
            "--val-fraction", "0.1",
            "--test-fraction", "0.1",
            "--lr", "1e-2",
            "--batch-size", "64",
            "--lr-decay-power", "1.0",
            "--min-lr-ratio", "0.0",
            "--sweep-mode",
        ],
    ),
}

def existing_completions(results: dict, task_key: str) -> set[tuple[int, int]]:
    return {
        (int(entry[GROUP_KEY]), int(entry.get("seed", 1)))
        for entry in results.get(task_key, [])
        if entry.get("status") == "ok"
    }


def run_single(
    task_key: str,
    num_hidden: int,
    seed: int,
    sweep_dir: Path,
    dry_run: bool,
    epochs: int | None = None,
) -> dict:
    task = TASKS[task_key]
    run_name = f"{task_key}_N{num_hidden}_seed{seed}"
    run_output_dir = sweep_dir / "runs" / run_name
    overrides: dict = {"--num-hidden": num_hidden, "--seed": seed, "--output-dir": run_output_dir}
    if epochs is not None:
        overrides["--epochs"] = epochs
    cmd = cmd_with_overrides(task.base_cmd, overrides)
    log_path = sweep_dir / "logs" / f"{run_name}.log"

    record = {
        "task": task_key,
        GROUP_KEY: num_hidden,
        "seed": seed,
        **run_training_cmd(
            cmd, log_path, task.output_glob, task_key, "metric", dry_run,
            output_dir=run_output_dir,
        ),
    }
    if record.get("status") == "ok":
        print(f"  -> {task.metric_label}: {record['metric']:.4f}")
    return record


def plot_summary(results: dict, plot_path: Path) -> None:
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(7, 3.2 * n_tasks), sharex=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, (task_key, task) in zip(axes, TASKS.items()):
        summary = aggregate_scalar_results(results.get(task_key, []), GROUP_KEY)
        plot_scalar_sweep(
            ax,
            summary,
            GROUP_KEY,
            "black",
            task.metric_label,
            ylim=task.ylim,
            title=task.name,
        )
        ax.set_xscale("log")

    axes[-1].set_xlabel(r"$N$ (hidden units)")
    fig.suptitle("Performance vs hidden size $N$", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_epoch_results(results: dict, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(7, 3.2 * n_tasks), sharex=True)
    if n_tasks == 1:
        axes = [axes]
    has_data = False

    for ax, (task_key, task) in zip(axes, TASKS.items()):
        ax.set_title(task.name, loc="left")
        ax.set_ylabel(task.epoch_ylabel)
        ax.set_ylim(*task.ylim)
        ax.grid(True, alpha=0.3)

        curves = aggregate_epoch_curves(results.get(task_key, []), GROUP_KEY)
        if not curves:
            continue

        has_data = True
        ns = sorted(curves.keys())
        n_groups = len(ns)
        colors = [mycmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
        labels = [rf"$N={n}$" for n in ns]
        ordered = {n: curves[n] for n in ns}

        plot_epoch_curves_with_stats(
            ax,
            ordered,
            colors,
            labels=labels,
            ylim=task.ylim,
            random_guess=task.random_guess,
        )

    if not has_data:
        plt.close(fig)
        return

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Epoch")
    for ax in axes:
        ax.set_xlim(0, 40)
    fig.suptitle("Test performance vs epoch", y=1.02)
    fig.tight_layout()
    out = plot_dir / "scaling_epochs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed hidden-size scaling sweep.")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()), default=list(TASKS.keys()))
    parser.add_argument(
        "--num-hidden-list",
        type=str,
        default=None,
        help="Comma-separated hidden sizes (default: per-task list)",
    )
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for all tasks")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else REPO_ROOT / f"results/sweeps/scaling_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"

    results = load_results(results_path, list(TASKS.keys()))
    seeds = args.seeds if args.seeds is not None else list(range(1, args.num_runs + 1))

    task_grids = {}
    for task_key in args.tasks:
        if args.num_hidden_list:
            task_grids[task_key] = [int(x.strip()) for x in args.num_hidden_list.split(",")]
        else:
            task_grids[task_key] = TASKS[task_key].default_hidden

    if not args.plot_only:
        with (sweep_dir / "sweep_config.json").open("w") as f:
            json.dump(
                {
                    "num_runs": args.num_runs,
                    "seeds": seeds,
                    "epochs": args.epochs,
                    "task_grids": task_grids,
                    "tasks": args.tasks,
                },
                f,
                indent=2,
            )

        for task_key in args.tasks:
            if task_key not in results:
                results[task_key] = []
            done = existing_completions(results, task_key) if args.skip_existing else set()

            for num_hidden in task_grids[task_key]:
                for seed in seeds:
                    key = (num_hidden, seed)
                    if key in done:
                        print(f"Skipping {task_key} N={num_hidden} seed={seed} (already completed)")
                        continue

                    task = TASKS[task_key]
                    print(f"\n{'=' * 60}\n{task.name} | N={num_hidden} | seed={seed}\n{'=' * 60}")
                    record = run_single(task_key, num_hidden, seed, sweep_dir, args.dry_run, epochs=args.epochs)
                    if record["status"] != "dry_run":
                        results[task_key] = [
                            e for e in results[task_key]
                            if not (int(e.get(GROUP_KEY, -1)) == num_hidden and int(e.get("seed", -1)) == seed)
                        ]
                        results[task_key].append(record)
                        save_results(results_path, results)
                        save_csv(results_path, results, extra_fields=[GROUP_KEY])
                        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))

        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=[GROUP_KEY])
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
    else:
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))

    plot_summary(results, sweep_dir / "scaling_performance.png")
    plot_epoch_results(results, sweep_dir)
    print(f"\nResults: {results_path}")
    print(f"Summary: {results_path.with_name('summary.json')}")


if __name__ == "__main__":
    main()
