#!/usr/bin/env python3
"""Sweep natural frequency omega for IMDB, sMNIST, and Mackey-Glass training runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.plotting_utils.style import thesis_red, thesis_blue, ifisc_green

BEST_TEST_RE = re.compile(r"best test:\s*([\d.]+)", re.IGNORECASE)


@dataclass
class TaskConfig:
    name: str
    script: str
    output_glob: str
    metric_label: str
    omega_center: float
    omega_min: float
    omega_max: float
    omega_step: float
    base_cmd: list[str] = field(default_factory=list)


TASKS: dict[str, TaskConfig] = {
    "imdb": TaskConfig(
        name="IMDB",
        script="training/train_imdb.py",
        output_glob="results/imdb/*",
        metric_label=r"test accuracy ($\%$)",
        omega_center=0.035904,
        omega_min=1.0,
        omega_max=10.0,
        omega_step=1,
        base_cmd=[
            sys.executable,
            "training/train_imdb.py",
            "--num-hidden", "128",
            "--epochs", "10",
            "--batch-size", "64",
            "--lr", "1e-3",
            "--embed-dim", "100",
            "--max-len", "175",
            "--glove", "glove.6B.100d.txt",
            "--lambda-param", "-0.05",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.1",
            "--skip-epoch-plots",
        ],
    ),
    "smnist": TaskConfig(
        name="sMNIST",
        script="training/train_smnist.py",
        output_glob="results/smnist/*",
        metric_label=r"test accuracy ($\%$)",
        omega_center=0.224,
        omega_min=0.0,
        omega_max=1.0,
        omega_step=0.1,
        base_cmd=[
            sys.executable,
            "training/train_smnist.py",
            "--lambda-param", "0.0",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--epochs", "10",
            "--num-hidden", "128",
            "--skip-epoch-plots",
        ],
    ),
    "mackey_glass": TaskConfig(
        name="Mackey-Glass",
        script="training/train_mackey_glass.py",
        output_glob="results/mackey_glass/*",
        metric_label=r"test $R^2$",
        omega_center=0.15,
        omega_min=0.0,
        omega_max=1.0,
        omega_step=0.1,
        base_cmd=[
            sys.executable,
            "training/train_mackey_glass.py",
            "--epochs", "10",
            "--num-hidden", "100",
            "--input-length", "100",
            "--gamma", "0.01",
            "--lambda-param", "-0.1",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--mg-tau", "34.0",
            "--horizon", "1",
            "--series-length", "20000",
            "--val-fraction", "0.1",
            "--test-fraction", "0.1",
            "--lr", "1e-2",
            "--batch-size", "64",
            "--seed", "1",
            "--lr-decay-power", "1.0",
            "--min-lr-ratio", "0.0",
            "--skip-epoch-plots",
        ],
    ),
}


def omega_values(omega_min: float, omega_max: float, omega_step: float) -> list[float]:
    values = np.arange(omega_min, omega_max + 0.5 * omega_step, omega_step)
    rounded = [round(float(v), 10) for v in values]
    return [v for v in rounded if v <= omega_max + 1e-9]


def task_omega_grid(
    task: TaskConfig,
    omega_min: float | None,
    omega_max: float | None,
    omega_step: float | None,
) -> list[float]:
    lo = task.omega_min if omega_min is None else omega_min
    hi = task.omega_max if omega_max is None else omega_max
    step = task.omega_step if omega_step is None else omega_step
    return omega_values(lo, hi, step)


def parse_classification_metric(text: str) -> float | None:
    matches = BEST_TEST_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_mackey_glass_metrics(metrics_path: Path) -> dict[str, float] | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open() as f:
        epochs = json.load(f)
    if not epochs:
        return None
    best_epoch = min(
        epochs,
        key=lambda row: row["val_normalized_error"]
        if row.get("val_normalized_error") is not None
        else float("inf"),
    )
    return {
        "test_r2": float(best_epoch["test_r2"]),
        "val_r2": float(best_epoch["val_r2"]),
        "test_normalized_error": float(best_epoch["test_normalized_error"]),
        "val_normalized_error": float(best_epoch["val_normalized_error"]),
        "test_mse": float(best_epoch["test_mse"]),
        "val_mse": float(best_epoch["val_mse"]),
        "best_epoch": int(best_epoch["epoch"]),
    }


def find_newest_output_dir(pattern: str, since: float) -> Path | None:
    candidates = [
        p for p in REPO_ROOT.glob(pattern)
        if p.is_dir() and p.stat().st_mtime >= since - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_results(results_path: Path) -> dict:
    if results_path.is_file():
        with results_path.open() as f:
            return json.load(f)
    return {task: [] for task in TASKS}


def save_results(results_path: Path, results: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)


def save_csv(results_path: Path, results: dict) -> None:
    csv_path = results_path.with_suffix(".csv")
    rows = []
    for task_key, entries in results.items():
        for entry in entries:
            row = {
                "task": task_key,
                "omega": entry.get("omega"),
                "seed": entry.get("seed"),
                "metric": entry.get("metric"),
                "status": entry.get("status"),
                "output_dir": entry.get("output_dir"),
            }
            if task_key == "mackey_glass":
                row.update(
                    {
                        "test_r2": entry.get("test_r2"),
                        "test_normalized_error": entry.get("test_normalized_error"),
                        "test_mse": entry.get("test_mse"),
                        "best_epoch": entry.get("best_epoch"),
                    }
                )
            rows.append(row)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_task_results(entries: list[dict]) -> list[dict]:
    by_omega: dict[float, list[float]] = {}
    for entry in entries:
        if entry.get("status") != "ok" or entry.get("metric") is None:
            continue
        omega = round(float(entry["omega"]), 10)
        by_omega.setdefault(omega, []).append(float(entry["metric"]))

    summary = []
    for omega in sorted(by_omega):
        metrics = np.asarray(by_omega[omega], dtype=float)
        n = len(metrics)
        mean = float(np.mean(metrics))
        std = float(np.std(metrics, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * sem
        summary.append(
            {
                "omega": omega,
                "n_runs": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": ci95,
                "min": float(np.min(metrics)),
                "max": float(np.max(metrics)),
            }
        )
    return summary


def save_summary(results_path: Path, results: dict) -> None:
    summary = {
        task_key: aggregate_task_results(results.get(task_key, []))
        for task_key in TASKS
    }
    summary_path = results_path.with_name("summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


def existing_completions(results: dict, task_key: str) -> set[tuple[float, int]]:
    return {
        (round(float(entry["omega"]), 10), int(entry.get("seed", 1)))
        for entry in results.get(task_key, [])
        if entry.get("status") == "ok"
    }


def cmd_with_overrides(base_cmd: list[str], seed: int, omega: float) -> list[str]:
    cmd: list[str] = []
    i = 0
    while i < len(base_cmd):
        if base_cmd[i] in {"--seed", "--omega"}:
            i += 2
            continue
        cmd.append(base_cmd[i])
        i += 1
    return [*cmd, "--seed", str(seed), "--omega", str(omega)]


def run_single(
    task_key: str,
    omega: float,
    seed: int,
    sweep_dir: Path,
    dry_run: bool,
) -> dict:
    task = TASKS[task_key]
    cmd = cmd_with_overrides(task.base_cmd, seed, omega)
    run_name = f"{task_key}_omega{omega:.6f}_seed{seed}"
    log_path = sweep_dir / "logs" / f"{run_name}.log"

    record = {
        "task": task_key,
        "omega": omega,
        "seed": seed,
        "status": "pending",
        "command": cmd,
        "log_path": str(log_path.relative_to(sweep_dir)),
    }

    if dry_run:
        record["status"] = "dry_run"
        print("DRY RUN:", " ".join(cmd))
        return record

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"\n{'=' * 60}\n{task.name} | omega={omega:.6f} | seed={seed}\n{'=' * 60}")
    print(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_path.write_text(stdout + ("\n" + stderr if stderr else ""))

    record["returncode"] = proc.returncode
    output_dir = find_newest_output_dir(task.output_glob, start)
    if output_dir is not None:
        record["output_dir"] = str(output_dir.relative_to(REPO_ROOT))

    if proc.returncode != 0:
        record["status"] = "failed"
        record["error"] = stderr[-2000:] if stderr else stdout[-2000:]
        return record

    if task_key == "mackey_glass":
        metrics_path = output_dir / "metrics.json" if output_dir else None
        mg_metrics = parse_mackey_glass_metrics(metrics_path) if metrics_path else None
        if mg_metrics is None:
            record["status"] = "parse_failed"
            return record
        record.update(mg_metrics)
        record["metric"] = mg_metrics["test_r2"]
    else:
        metric = parse_classification_metric(stdout)
        if metric is None and output_dir is not None:
            log_file = output_dir / "log.txt"
            if log_file.is_file():
                metric = parse_classification_metric(log_file.read_text())
        if metric is None:
            record["status"] = "parse_failed"
            return record
        record["metric"] = metric
        record["test_acc"] = metric

    record["status"] = "ok"
    print(f"  -> {task.metric_label}: {record['metric']:.4f}")
    return record


def plot_results(results: dict, plot_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {
        "imdb": thesis_blue,
        "smnist": ifisc_green,
        "mackey_glass": thesis_red,
    }

    for ax, (task_key, task) in zip(axes, TASKS.items()):
        summary = aggregate_task_results(results.get(task_key, []))
        if not summary:
            ax.set_title(f"{task.name}\n(no successful runs)")
            ax.set_xlabel(r"$\omega$")
            continue

        omegas = [row["omega"] for row in summary]
        means = [row["mean"] for row in summary]
        ci95 = [row["ci95"] for row in summary]
        stds = [row["std"] for row in summary]
        color = colors[task_key]

        lower_ci = [m - c for m, c in zip(means, ci95)]
        upper_ci = [m + c for m, c in zip(means, ci95)]
        lower_std = [m - s for m, s in zip(means, stds)]
        upper_std = [m + s for m, s in zip(means, stds)]

        ax.fill_between(omegas, lower_std, upper_std, color=color, alpha=0.15, linewidth=0)
        ax.fill_between(omegas, lower_ci, upper_ci, color=color, alpha=0.25, linewidth=0)
        ax.plot(omegas, means, "o-", color=color, linewidth=2, markersize=5, label="mean")
        ax.axvline(task.omega_center, color=color, linestyle="--", alpha=0.5, linewidth=1)

        n_runs = summary[0]["n_runs"]
        # ax.set_title(f"{task.name}\n({n_runs} runs per $\\omega$)")
        ax.set_ylabel(task.metric_label)
        ax.set_xlabel(r"$\omega$")
        ax.grid(True, alpha=0.3)
        if task_key == "mackey_glass":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(50.0, 100.0)

    handles = [
        plt.Line2D([0], [0], color="gray", linewidth=2, marker="o", label="mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.25, label="$95\%$ CI"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.15, label=r"$\pm 1\sigma$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.suptitle("Performance vs natural frequency $\\omega$", y=1.14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep omega across HORN training tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS.keys()),
        default=list(TASKS.keys()),
        help="Tasks to run (default: all)",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=None,
        help="Override minimum omega for all tasks (default: per-task range around chosen value)",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=None,
        help="Override maximum omega for all tasks (default: per-task range around chosen value)",
    )
    parser.add_argument(
        "--omega-step",
        type=float,
        default=None,
        help="Override omega step for all tasks (default: per-task step)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory for sweep outputs (default: results/sweeps/omega_sweep_<timestamp>)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of independent runs per (task, omega) with different seeds (default: 10)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seeds to use (default: 1..num-runs)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (task, omega, seed) combinations already in results.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing results.json")
    parser.add_argument(
        "--with-manifold",
        action="store_true",
        help="Enable manifold analysis (slow; omitted by default)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else REPO_ROOT / f"results/sweeps/omega_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"
    plot_path = sweep_dir / "omega_sweep_performance.png"

    results = load_results(results_path)
    seeds = args.seeds if args.seeds is not None else list(range(1, args.num_runs + 1))

    if args.with_manifold:
        for task in TASKS.values():
            task.base_cmd = [c for c in task.base_cmd if c != "--skip-epoch-plots"]

    if not args.plot_only:
        task_grids = {
            task_key: task_omega_grid(
                TASKS[task_key],
                args.omega_min,
                args.omega_max,
                args.omega_step,
            )
            for task_key in args.tasks
        }
        config_path = sweep_dir / "sweep_config.json"
        with config_path.open("w") as f:
            json.dump(
                {
                    "omega_min": args.omega_min,
                    "omega_max": args.omega_max,
                    "omega_step": args.omega_step,
                    "num_runs": args.num_runs,
                    "seeds": seeds,
                    "task_grids": {
                        task_key: {
                            "omega_center": TASKS[task_key].omega_center,
                            "omega_min": TASKS[task_key].omega_min
                            if args.omega_min is None
                            else args.omega_min,
                            "omega_max": TASKS[task_key].omega_max
                            if args.omega_max is None
                            else args.omega_max,
                            "omega_step": TASKS[task_key].omega_step
                            if args.omega_step is None
                            else args.omega_step,
                            "omegas": task_grids[task_key],
                        }
                        for task_key in args.tasks
                    },
                    "tasks": args.tasks,
                },
                f,
                indent=2,
            )

        for task_key in args.tasks:
            if task_key not in results:
                results[task_key] = []
            done = existing_completions(results, task_key) if args.skip_existing else set()

            for omega in task_grids[task_key]:
                omega_key = round(float(omega), 10)
                for seed in seeds:
                    if (omega_key, seed) in done:
                        print(
                            f"Skipping {task_key} omega={omega:.6f} seed={seed} (already completed)"
                        )
                        continue

                    record = run_single(task_key, omega, seed, sweep_dir, args.dry_run)
                    if record["status"] != "dry_run":
                        results[task_key] = [
                            e
                            for e in results[task_key]
                            if not (
                                round(float(e.get("omega", -1)), 10) == omega_key
                                and int(e.get("seed", -1)) == seed
                            )
                        ]
                        results[task_key].append(record)
                        save_results(results_path, results)
                        save_csv(results_path, results)
                        save_summary(results_path, results)

        save_results(results_path, results)
        save_csv(results_path, results)
        save_summary(results_path, results)

    else:
        save_summary(results_path, results)

    plot_results(results, plot_path)
    print(f"\nResults: {results_path}")
    print(f"Summary: {results_path.with_name('summary.json')}")
    print(f"CSV:     {results_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
