#!/usr/bin/env python3
"""Sweep excitability alpha for IMDB, sMNIST, and Mackey-Glass training runs."""

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
    base_cmd: list[str] = field(default_factory=list)


TASKS: dict[str, TaskConfig] = {
    "imdb": TaskConfig(
        name="IMDB",
        script="training/train_imdb.py",
        output_glob="results/imdb/*",
        metric_label="test accuracy (%)",
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
        ],
    ),
    "smnist": TaskConfig(
        name="sMNIST",
        script="training/train_smnist.py",
        output_glob="results/smnist/*",
        metric_label="test accuracy (%)",
        base_cmd=[
            sys.executable,
            "training/train_smnist.py",
            "--lambda-param", "0.0",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--epochs", "10",
            "--num-hidden", "128",
        ],
    ),
    "mackey_glass": TaskConfig(
        name="Mackey-Glass",
        script="training/train_mackey_glass.py",
        output_glob="results/mackey_glass/*",
        metric_label=r"test $R^2$",
        base_cmd=[
            sys.executable,
            "training/train_mackey_glass.py",
            "--epochs", "10",
            "--num-hidden", "100",
            "--input-length", "100",
            "--omega", "0.15",
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
        ],
    ),
}


def alpha_values(alpha_min: float, alpha_max: float, alpha_step: float) -> list[float]:
    values = np.arange(alpha_min, alpha_max + 0.5 * alpha_step, alpha_step)
    rounded = [round(float(v), 10) for v in values]
    return [v for v in rounded if v <= alpha_max + 1e-9]


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
                "alpha": entry.get("alpha"),
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


def existing_alphas(results: dict, task_key: str) -> set[float]:
    return {
        round(float(entry["alpha"]), 10)
        for entry in results.get(task_key, [])
        if entry.get("status") == "ok"
    }


def run_single(
    task_key: str,
    alpha: float,
    sweep_dir: Path,
    dry_run: bool,
) -> dict:
    task = TASKS[task_key]
    cmd = [*task.base_cmd, "--alpha", str(alpha)]
    run_name = f"{task_key}_alpha{alpha:.2f}"
    log_path = sweep_dir / "logs" / f"{run_name}.log"

    record = {
        "task": task_key,
        "alpha": alpha,
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
    print(f"\n{'=' * 60}\n{task.name} | alpha={alpha:.2f}\n{'=' * 60}")
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    colors = {
        "imdb": thesis_blue,
        "smnist": ifisc_green,
        "mackey_glass": thesis_red,
    }

    for ax, (task_key, task) in zip(axes, TASKS.items()):
        entries = sorted(
            [e for e in results.get(task_key, []) if e.get("status") == "ok"],
            key=lambda e: e["alpha"],
        )
        if not entries:
            ax.set_title(f"{task.name}\n(no successful runs)")
            ax.set_xlabel(r"$\alpha$")
            continue

        alphas = [e["alpha"] for e in entries]
        metrics = [e["metric"] for e in entries]
        ax.plot(alphas, metrics, "o-", color=colors[task_key], linewidth=2, markersize=5)
        ax.set_title(task.name)
        ax.set_ylabel(task.metric_label)
        ax.set_xlabel(r"$\alpha$ (excitability)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Performance vs pre-input excitability $\\alpha$", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep alpha across HORN training tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS.keys()),
        default=list(TASKS.keys()),
        help="Tasks to run (default: all)",
    )
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=5.0)
    parser.add_argument("--alpha-step", type=float, default=0.25)
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory for sweep outputs (default: results/sweeps/alpha_sweep_<timestamp>)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip alphas already in results.json")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing results.json")
    parser.add_argument(
        "--with-manifold",
        action="store_true",
        help="Enable manifold analysis (slow; omitted by default)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else REPO_ROOT / f"results/sweeps/alpha_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"
    plot_path = sweep_dir / "alpha_sweep_performance.png"

    results = load_results(results_path)

    if args.with_manifold:
        for task in TASKS.values():
            task.base_cmd = [c for c in task.base_cmd if c != "--no-analyze-manifold"]

    if not args.plot_only:
        alphas = alpha_values(args.alpha_min, args.alpha_max, args.alpha_step)
        config_path = sweep_dir / "sweep_config.json"
        with config_path.open("w") as f:
            json.dump(
                {
                    "alpha_min": args.alpha_min,
                    "alpha_max": args.alpha_max,
                    "alpha_step": args.alpha_step,
                    "alphas": alphas,
                    "tasks": args.tasks,
                },
                f,
                indent=2,
            )

        for task_key in args.tasks:
            if task_key not in results:
                results[task_key] = []
            done = existing_alphas(results, task_key) if args.skip_existing else set()

            for alpha in alphas:
                alpha_key = round(float(alpha), 10)
                if alpha_key in done:
                    print(f"Skipping {task_key} alpha={alpha:.2f} (already completed)")
                    continue

                record = run_single(task_key, alpha, sweep_dir, args.dry_run)
                if record["status"] != "dry_run":
                    results[task_key] = [
                        e for e in results[task_key] if round(float(e.get("alpha", -1)), 10) != alpha_key
                    ]
                    results[task_key].append(record)
                    save_results(results_path, results)
                    save_csv(results_path, results)

        save_results(results_path, results)
        save_csv(results_path, results)

    plot_results(results, plot_path)
    print(f"\nResults: {results_path}")
    print(f"CSV:     {results_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
