#!/usr/bin/env python3
"""Parallel lambda / gamma_real / alpha (/ seed) grid search.

Primary entry for the cluster `run` wrapper (shared 32-core node):

  run -t 100:00 -c 32 -m 32 -j lga-grid -o lga-grid.out -e lga-grid.err -- \\
    .venv/bin/python experiments/grid_search/lga_grid.py local \\
      --tasks imdb smnist mackey_glass \\
      --workers 32 --run-dir results/grid_search/lga_main

Resume the same run later with the same --run-dir (completed jobs are skipped).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.grid_search.lga_heatmaps import generate_lga_heatmaps

BEST_TEST_RE = re.compile(r"best test:\s*([\d.]+)", re.IGNORECASE)
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable

DEFAULT_NUM_HIDDEN = 25
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5


@dataclass
class TaskConfig:
    name: str
    script: str
    output_glob: str
    metric_key: str
    metric_label: str
    chance_level: float
    scale: float
    omega: float
    gamma_imag: float
    base_cmd: list[str] = field(default_factory=list)


def _task_base(
    script: str,
    extra: list[str],
    *,
    num_hidden: int = DEFAULT_NUM_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
) -> list[str]:
    return [
        PYTHON,
        script,
        "--num-hidden",
        str(num_hidden),
        "--epochs",
        str(epochs),
        "--early-stop-patience",
        str(patience),
        "--skip-epoch-plots",
        "--sweep-mode",
        *extra,
    ]


TASKS: dict[str, TaskConfig] = {
    "imdb": TaskConfig(
        name="IMDB",
        script="training/train_imdb.py",
        output_glob="results/imdb/*",
        metric_key="best_test_acc",
        metric_label="Test Accuracy (%)",
        chance_level=50.0,
        scale=1.0,
        omega=0.035904,
        gamma_imag=0.1,
        base_cmd=_task_base(
            "training/train_imdb.py",
            [
                "--batch-size",
                "64",
                "--lr",
                "1e-3",
                "--embed-dim",
                "100",
                "--max-len",
                "175",
                "--glove",
                "glove.6B.100d.txt",
            ],
        ),
    ),
    "smnist": TaskConfig(
        name="sMNIST",
        script="training/train_smnist.py",
        output_glob="results/smnist/*",
        metric_key="best_test_acc",
        metric_label="Test Accuracy (%)",
        chance_level=10.0,
        scale=1.0,
        omega=0.224,
        gamma_imag=0.0,
        base_cmd=_task_base("training/train_smnist.py", ["--lr", "1e-2"]),
    ),
    "mackey_glass": TaskConfig(
        name="Mackey-Glass",
        script="training/train_mackey_glass.py",
        output_glob="results/mackey_glass/*",
        metric_key="best_test_r2",
        metric_label=r"Test $R^2$ (%)",
        chance_level=90.0,
        scale=100.0,
        omega=0.15,
        gamma_imag=0.0,
        base_cmd=_task_base(
            "training/train_mackey_glass.py",
            [
                "--input-length",
                "100",
                "--gamma",
                "0.01",
                "--mg-tau",
                "34.0",
                "--horizon",
                "25",
                "--series-length",
                "20000",
                "--val-fraction",
                "0.1",
                "--test-fraction",
                "0.1",
                "--lr",
                "1e-2",
                "--batch-size",
                "64",
                "--lr-decay-power",
                "1.0",
                "--min-lr-ratio",
                "0.0",
            ],
        ),
    ),
}


def arange_inclusive(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    n = int(np.floor((hi - lo) / step + 0.5)) + 1
    vals = [round(float(lo + i * step), 10) for i in range(max(n, 1))]
    if vals[-1] < hi - 1e-12:
        vals.append(round(float(hi), 10))
    elif vals[-1] > hi + 1e-12:
        vals[-1] = round(float(hi), 10)
    # dedupe while preserving order
    out: list[float] = []
    for v in vals:
        if not out or abs(out[-1] - v) > 1e-12:
            out.append(v)
    return out


def build_combinations(
    tasks: list[str],
    lambda_vals: list[float],
    gamma_real_vals: list[float],
    alpha_vals: list[float],
    seeds: list[int],
) -> list[dict]:
    combos = []
    idx = 0
    for task in tasks:
        for lam in lambda_vals:
            for gr in gamma_real_vals:
                for alpha in alpha_vals:
                    for seed in seeds:
                        combos.append(
                            {
                                "index": idx,
                                "task": task,
                                "lambda": float(lam),
                                "gamma_real": float(gr),
                                "alpha": float(alpha),
                                "seed": int(seed),
                            }
                        )
                        idx += 1
    return combos


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def job_result_path(run_dir: Path, index: int) -> Path:
    return run_dir / "jobs" / f"{index:05d}.json"


def parse_classification_metric(text: str) -> float | None:
    matches = BEST_TEST_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_mackey_glass_metrics(metrics_path: Path) -> dict[str, float] | None:
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open() as f:
            epochs = json.load(f)
    except json.JSONDecodeError:
        return None
    if not epochs:
        return None
    valid = [
        row
        for row in epochs
        if row.get("val_normalized_error") is not None
        and row.get("test_r2") is not None
    ]
    if not valid:
        # NaN-aborted run with stub metrics
        last = epochs[-1]
        return {
            "best_test_r2": float(last.get("test_r2") or 0.0),
            "best_val_r2": float(last.get("val_r2") or 0.0),
            "best_test_normalized_error": float("inf"),
            "best_val_normalized_error": float("inf"),
            "best_test_mse": float("inf"),
            "best_val_mse": float("inf"),
            "best_epoch": int(last.get("epoch", -1)),
            "numerical_error": True,
        }
    best_epoch = min(valid, key=lambda row: row["val_normalized_error"])
    return {
        "best_test_r2": float(best_epoch["test_r2"]),
        "best_val_r2": float(best_epoch["val_r2"]),
        "best_test_normalized_error": float(best_epoch["test_normalized_error"]),
        "best_val_normalized_error": float(best_epoch["val_normalized_error"]),
        "best_test_mse": float(best_epoch["test_mse"]),
        "best_val_mse": float(best_epoch["val_mse"]),
        "best_epoch": int(best_epoch["epoch"]),
    }


def parse_classification_metrics_json(metrics_path: Path) -> dict[str, float] | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open() as f:
        epochs = json.load(f)
    if not epochs:
        return None
    best = max(epochs, key=lambda row: row.get("val_acc", float("-inf")))
    return {
        "best_val_acc": float(best["val_acc"]),
        "best_test_acc": float(best["test_acc"]),
        "best_epoch": int(best["epoch"]),
        "best_train_acc": float(best.get("train_acc", 0.0)),
        "best_test_loss": float(best.get("test_loss", 0.0)),
        "best_val_loss": float(best.get("val_loss", 0.0)),
    }


def find_newest_output_dir(pattern: str, since: float) -> Path | None:
    candidates = [
        p for p in REPO_ROOT.glob(pattern) if p.is_dir() and p.stat().st_mtime >= since - 1.0
    ]
    if not candidates:
        return None
    # Prefer uniquely named parallel run dirs (contain _p<pid>_).
    unique = [p for p in candidates if "_p" in p.name]
    pool = unique if unique else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def cmd_with_params(
    base_cmd: list[str],
    *,
    seed: int,
    omega: float,
    alpha: float,
    lambda_param: float,
    gamma_real: float,
    gamma_imag: float,
    num_hidden: int,
    epochs: int,
    patience: int,
) -> list[str]:
    skip = {
        "--seed",
        "--omega",
        "--alpha",
        "--lambda-param",
        "--gamma-real",
        "--gamma-imag",
        "--num-hidden",
        "--epochs",
        "--early-stop-patience",
    }
    cmd: list[str] = []
    i = 0
    while i < len(base_cmd):
        if base_cmd[i] in skip:
            i += 2
            continue
        cmd.append(base_cmd[i])
        i += 1
    return [
        *cmd,
        "--num-hidden",
        str(num_hidden),
        "--epochs",
        str(epochs),
        "--early-stop-patience",
        str(patience),
        "--seed",
        str(seed),
        "--omega",
        str(omega),
        "--alpha",
        str(alpha),
        "--lambda-param",
        str(lambda_param),
        "--gamma-real",
        str(gamma_real),
        "--gamma-imag",
        str(gamma_imag),
    ]


def collect_results(run_dir: Path) -> dict[str, list[dict]]:
    config = load_json(run_dir / "config.json")
    by_task = {task: [] for task in config["tasks"]}
    jobs_dir = run_dir / "jobs"
    if not jobs_dir.is_dir():
        return by_task
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            record = load_json(path)
        except json.JSONDecodeError:
            continue
        task = record.get("task")
        if task in by_task:
            by_task[task].append(record)
    return by_task


def save_aggregated(run_dir: Path, by_task: dict[str, list[dict]], config: dict) -> None:
    all_results = []
    for task, rows in by_task.items():
        payload = {
            "config": {
                **config.get("grid", {}),
                "task": task,
                "omega": TASKS[task].omega,
                "gamma_imag": TASKS[task].gamma_imag,
                "fixed_params": config.get("fixed_params", {}).get(task, {}),
                "num_hidden": config.get("num_hidden", DEFAULT_NUM_HIDDEN),
                "epochs": config.get("epochs", DEFAULT_EPOCHS),
                "early_stop_patience": config.get("early_stop_patience", DEFAULT_PATIENCE),
                "seeds": config.get("seeds", []),
            },
            "results": rows,
            "completed_indices": [r["index"] for r in rows if r.get("status") == "ok"],
        }
        atomic_write_json(run_dir / f"results_{task}.json", payload)
        all_results.extend(rows)

    atomic_write_json(run_dir / "results_all.json", {"config": config, "results": all_results})

    csv_path = run_dir / "results_all.csv"
    fieldnames = [
        "index",
        "task",
        "lambda",
        "gamma_real",
        "alpha",
        "seed",
        "metric",
        "best_test_acc",
        "best_val_acc",
        "best_test_r2",
        "best_val_r2",
        "best_test_normalized_error",
        "best_val_normalized_error",
        "best_test_mse",
        "best_val_mse",
        "best_epoch",
        "status",
        "numerical_error",
        "runtime_sec",
        "output_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)


def _heatmap_config_from_grid(grid: dict) -> dict:
    return {
        "lambda_range": [
            grid["lambda_values"][0],
            grid["lambda_values"][-1],
            len(grid["lambda_values"]),
        ],
        "gamma_real_range": [
            grid["gamma_real_values"][0],
            grid["gamma_real_values"][-1],
            len(grid["gamma_real_values"]),
        ],
        "alpha_range": [
            grid["alpha_values"][0],
            grid["alpha_values"][-1],
            len(grid["alpha_values"]),
        ],
        "lambda_values": grid["lambda_values"],
        "gamma_real_values": grid["gamma_real_values"],
        "alpha_values": grid["alpha_values"],
    }


def update_heatmaps(
    run_dir: Path,
    by_task: dict[str, list[dict]],
    config: dict,
    *,
    destinations: list[Path] | None = None,
    timestamp: str = "live",
    clip_nonpositive_tasks: set[str] | None = None,
    annotate: bool = True,
) -> None:
    heatmap_config = _heatmap_config_from_grid(config["grid"])
    if destinations is None:
        destinations = [run_dir / "heatmaps"]
        shared = config.get("heatmap_dir")
        if shared:
            destinations.append(Path(shared))
    clip_tasks = clip_nonpositive_tasks or set()

    for task_key, rows in by_task.items():
        task = TASKS[task_key]
        for base in destinations:
            out_dir = Path(base) / task_key
            generate_lga_heatmaps(
                rows,
                out_dir,
                heatmap_config,
                metric_key=task.metric_key,
                metric_label=task.metric_label,
                chance_level=task.chance_level,
                scale=task.scale,
                timestamp=timestamp,
                clip_nonpositive=task_key in clip_tasks,
                annotate=annotate,
                task_name=task_key,
            )


def aggregate_with_lock(run_dir: Path, *, blocking: bool = True) -> dict[str, list[dict]] | None:
    lock_path = run_dir / ".aggregate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(lock_f.fileno(), flags)
        except BlockingIOError:
            return None
        config = load_json(run_dir / "config.json")
        by_task = collect_results(run_dir)
        save_aggregated(run_dir, by_task, config)
        update_heatmaps(run_dir, by_task, config)
        n_ok = sum(1 for rows in by_task.values() for r in rows if r.get("status") == "ok")
        n_total = len(config["combinations"])
        print(f"Aggregated {n_ok}/{n_total} completed combos -> {run_dir}", flush=True)
        return by_task


def run_worker(run_dir: str | Path, index: int, dry_run: bool = False, aggregate: bool = True) -> dict:
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.json")
    combos = config["combinations"]
    if index < 0 or index >= len(combos):
        raise IndexError(f"index {index} out of range [0, {len(combos)})")

    combo = combos[index]
    result_path = job_result_path(run_dir, index)
    if result_path.is_file():
        existing = load_json(result_path)
        if existing.get("status") == "ok":
            print(f"Skipping index={index} (already ok)", flush=True)
            return existing

    task_key = combo["task"]
    task = TASKS[task_key]
    seed = int(combo.get("seed", config.get("seed", 1)))
    num_hidden = int(config.get("num_hidden", DEFAULT_NUM_HIDDEN))
    epochs = int(config.get("epochs", DEFAULT_EPOCHS))
    patience = int(config.get("early_stop_patience", DEFAULT_PATIENCE))

    cmd = cmd_with_params(
        task.base_cmd,
        seed=seed,
        omega=task.omega,
        alpha=float(combo["alpha"]),
        lambda_param=float(combo["lambda"]),
        gamma_real=float(combo["gamma_real"]),
        gamma_imag=task.gamma_imag,
        num_hidden=num_hidden,
        epochs=epochs,
        patience=patience,
    )

    record = {
        "index": index,
        "task": task_key,
        "lambda": float(combo["lambda"]),
        "gamma_real": float(combo["gamma_real"]),
        "alpha": float(combo["alpha"]),
        "omega": task.omega,
        "gamma_imag": task.gamma_imag,
        "seed": seed,
        "num_hidden": num_hidden,
        "epochs": epochs,
        "early_stop_patience": patience,
        "status": "pending",
        "command": cmd,
    }

    log_path = run_dir / "logs" / f"{index:05d}_{task_key}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["log_path"] = str(log_path.relative_to(run_dir))

    if dry_run:
        record["status"] = "dry_run"
        atomic_write_json(result_path, record)
        print("DRY RUN:", " ".join(cmd), flush=True)
        return record

    print(
        f"[{index}] {task.name} | λ={combo['lambda']:.4f} "
        f"γr={combo['gamma_real']:.4f} α={combo['alpha']:.4f} seed={seed}",
        flush=True,
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_path.write_text(stdout + ("\n" + stderr if stderr else ""))
    record["returncode"] = proc.returncode
    record["runtime_sec"] = time.time() - start

    output_dir = find_newest_output_dir(task.output_glob, start)
    if output_dir is not None:
        record["output_dir"] = str(output_dir.relative_to(REPO_ROOT))

    if proc.returncode != 0:
        record["status"] = "failed"
        record["error"] = (stderr or stdout)[-2000:]
        record["numerical_error"] = "nan" in record["error"].lower() or "inf" in record["error"].lower()
        atomic_write_json(result_path, record)
        if aggregate:
            aggregate_with_lock(run_dir, blocking=False)
        return record

    if task_key == "mackey_glass":
        metrics_path = output_dir / "metrics.json" if output_dir else None
        mg = parse_mackey_glass_metrics(metrics_path) if metrics_path else None
        if mg is None:
            # Fall back to log parsing for best test_r2
            m = re.search(r"best test_r2:\s*([-\d.]+)", stdout + "\n" + log_path.read_text(errors="ignore"))
            if m:
                record["best_test_r2"] = float(m.group(1))
                record["metric"] = float(m.group(1))
                record["numerical_error"] = "numerical error" in (stdout + stderr).lower() or "NaN/Inf" in (stdout + stderr)
                record["status"] = "ok"
                atomic_write_json(result_path, record)
                if aggregate:
                    aggregate_with_lock(run_dir, blocking=False)
                return record
            record["status"] = "parse_failed"
            atomic_write_json(result_path, record)
            if aggregate:
                aggregate_with_lock(run_dir, blocking=False)
            return record
        record.update(mg)
        record["metric"] = mg["best_test_r2"]
        record["numerical_error"] = bool(mg.get("numerical_error", False)) or (
            "numerical error" in (stdout + stderr).lower()
        )
    else:
        metric = parse_classification_metric(stdout)
        metrics_path = output_dir / "metrics.json" if output_dir else None
        cls = parse_classification_metrics_json(metrics_path) if metrics_path else None
        if cls is not None:
            record.update(cls)
            metric = cls["best_test_acc"]
        elif metric is None and output_dir is not None:
            log_file = output_dir / "log.txt"
            if log_file.is_file():
                metric = parse_classification_metric(log_file.read_text())
        if metric is None:
            record["status"] = "parse_failed"
            atomic_write_json(result_path, record)
            if aggregate:
                aggregate_with_lock(run_dir, blocking=False)
            return record
        record["metric"] = metric
        record["best_test_acc"] = metric
        log_text = stdout + "\n" + stderr
        record["numerical_error"] = (
            "numerical error" in log_text.lower()
            or ("NaN/Inf" in log_text and float(metric) == 0.0)
        )

    record["status"] = "ok"
    atomic_write_json(result_path, record)
    print(f"  -> {task.metric_label}: {record['metric']:.4f}", flush=True)
    if aggregate:
        aggregate_with_lock(run_dir, blocking=False)
    return record


def _worker_entry(args: tuple[str, int]) -> dict:
    run_dir, index = args
    return run_worker(run_dir, index, dry_run=False, aggregate=True)


def resolve_grid(args: argparse.Namespace) -> tuple[list[float], list[float], list[float], list[int]]:
    if getattr(args, "lambda_step", None) is not None:
        lambda_vals = arange_inclusive(args.lambda_min, args.lambda_max, args.lambda_step)
    else:
        steps = args.lambda_steps if args.lambda_steps is not None else 21
        lambda_vals = (
            [float(args.lambda_min)]
            if steps <= 1
            else [round(float(v), 10) for v in np.linspace(args.lambda_min, args.lambda_max, steps)]
        )

    if getattr(args, "gamma_real_step", None) is not None:
        gamma_real_vals = arange_inclusive(
            args.gamma_real_min, args.gamma_real_max, args.gamma_real_step
        )
    else:
        steps = args.gamma_real_steps if args.gamma_real_steps is not None else 21
        gamma_real_vals = (
            [float(args.gamma_real_min)]
            if steps <= 1
            else [
                round(float(v), 10)
                for v in np.linspace(args.gamma_real_min, args.gamma_real_max, steps)
            ]
        )

    if args.alpha_values:
        alpha_vals = [float(v) for v in args.alpha_values]
    else:
        steps = args.alpha_steps if args.alpha_steps is not None else 4
        alpha_vals = (
            [float(args.alpha_min)]
            if steps <= 1
            else [round(float(v), 10) for v in np.linspace(args.alpha_min, args.alpha_max, steps)]
        )

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = list(range(1, args.num_seeds + 1))
    return lambda_vals, gamma_real_vals, alpha_vals, seeds


def _resolve_heatmap_dir(args: argparse.Namespace) -> str | None:
    if not getattr(args, "heatmap_dir", None):
        return None
    path = Path(args.heatmap_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def create_or_load_run_dir(args: argparse.Namespace) -> Path:
    heatmap_dir = _resolve_heatmap_dir(args)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if run_dir.is_dir() and (run_dir / "config.json").is_file():
            print(f"Resuming existing run directory: {run_dir}", flush=True)
            if heatmap_dir:
                config = load_json(run_dir / "config.json")
                config["heatmap_dir"] = heatmap_dir
                atomic_write_json(run_dir / "config.json", config)
                print(f"Shared live heatmaps -> {heatmap_dir}", flush=True)
            return run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / "grid_search" / f"lga_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "jobs").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "heatmaps").mkdir(exist_ok=True)

    lambda_vals, gamma_real_vals, alpha_vals, seeds = resolve_grid(args)
    combinations = build_combinations(args.tasks, lambda_vals, gamma_real_vals, alpha_vals, seeds)

    config = {
        "created": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "tasks": args.tasks,
        "num_hidden": args.num_hidden,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "seeds": seeds,
        "grid": {
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "lambda_step": args.lambda_step,
            "gamma_real_min": args.gamma_real_min,
            "gamma_real_max": args.gamma_real_max,
            "gamma_real_step": args.gamma_real_step,
            "lambda_values": lambda_vals,
            "gamma_real_values": gamma_real_vals,
            "alpha_values": alpha_vals,
        },
        "fixed_params": {
            task: {"omega": TASKS[task].omega, "gamma_imag": TASKS[task].gamma_imag}
            for task in args.tasks
        },
        "combinations": combinations,
        "n_combinations": len(combinations),
        "workers": getattr(args, "workers", None),
        "heatmap_dir": heatmap_dir,
    }
    atomic_write_json(run_dir / "config.json", config)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Combinations: {len(combinations)}", flush=True)
    for task in args.tasks:
        n = sum(1 for c in combinations if c["task"] == task)
        print(f"  {task}: {n}", flush=True)
    print(
        f"  λ: {len(lambda_vals)} values {lambda_vals[0]} .. {lambda_vals[-1]}\n"
        f"  γr: {len(gamma_real_vals)} values {gamma_real_vals[0]} .. {gamma_real_vals[-1]}\n"
        f"  α: {alpha_vals}\n"
        f"  seeds: {seeds}\n"
        f"  n={args.num_hidden}, epochs={args.epochs}, patience={args.early_stop_patience}",
        flush=True,
    )
    if heatmap_dir:
        print(f"Shared live heatmaps -> {heatmap_dir}", flush=True)
    return run_dir


def pending_indices(run_dir: Path) -> list[int]:
    config = load_json(run_dir / "config.json")
    pending = []
    for combo in config["combinations"]:
        idx = int(combo["index"])
        path = job_result_path(run_dir, idx)
        if path.is_file():
            try:
                if load_json(path).get("status") == "ok":
                    continue
            except json.JSONDecodeError:
                pass
        pending.append(idx)
    return pending


def cmd_local(args: argparse.Namespace) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    run_dir = create_or_load_run_dir(args)
    config = load_json(run_dir / "config.json")
    pending = pending_indices(run_dir)
    n_total = int(config["n_combinations"])
    print(f"Pending: {len(pending)} / {n_total}", flush=True)

    if args.dry_run:
        for idx in pending[: min(3, len(pending))]:
            run_worker(run_dir, idx, dry_run=True, aggregate=False)
        print(f"Dry-run sample done ({min(3, len(pending))} shown). No training executed.")
        return

    if not pending:
        print("Nothing pending; aggregating heatmaps.")
        aggregate_with_lock(run_dir, blocking=True)
        return

    workers = max(1, int(args.workers))
    print(f"Running with {workers} parallel workers", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker_entry, (str(run_dir), idx)): idx for idx in pending
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                record = fut.result()
                status = record.get("status")
            except Exception as exc:
                status = f"exception:{exc}"
                print(f"[{idx}] worker crashed: {exc}", flush=True)
            completed += 1
            if completed % max(1, min(4, workers)) == 0 or completed == len(pending):
                aggregate_with_lock(run_dir, blocking=False)
            if completed % max(1, workers) == 0 or completed == len(pending):
                print(f"Progress: {completed}/{len(pending)} this session", flush=True)

    aggregate_with_lock(run_dir, blocking=True)
    heatmap_dir = config.get("heatmap_dir") or (run_dir / "heatmaps")
    print(f"Done. Results in {run_dir}", flush=True)
    print(f"Live heatmaps in {heatmap_dir}", flush=True)


def cmd_worker(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    index = args.index
    if index is None:
        index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    run_worker(run_dir, index, dry_run=args.dry_run, aggregate=True)


def cmd_aggregate(args: argparse.Namespace) -> None:
    aggregate_with_lock(Path(args.run_dir), blocking=True)


def cmd_summary(args: argparse.Namespace) -> None:
    """Regenerate clean summary heatmaps (no cell text; MG negatives -> grey)."""
    run_dirs = [Path(p) for p in args.run_dirs]
    run_dirs = [p if p.is_absolute() else REPO_ROOT / p for p in run_dirs]
    out_root = REPO_ROOT / "results" / "grid_search" / "summary"
    out_root.mkdir(parents=True, exist_ok=True)
    for run_dir in run_dirs:
        config = load_json(run_dir / "config.json")
        by_task = collect_results(run_dir)
        for task_key, rows in by_task.items():
            update_heatmaps(
                run_dir,
                {task_key: rows},
                config,
                destinations=[out_root],
                timestamp=task_key,
                clip_nonpositive_tasks={"mackey_glass"},
                annotate=False,
            )
        tasks = ", ".join(sorted(by_task))
        print(f"Summary heatmaps [{tasks}] -> {out_root}", flush=True)


def cmd_watch(args: argparse.Namespace) -> None:
    run_dirs = [Path(p) for p in (args.run_dirs or ([args.run_dir] if args.run_dir else []))]
    if not run_dirs:
        raise SystemExit("Provide --run-dir or --run-dirs")
    run_dirs = [p if p.is_absolute() else REPO_ROOT / p for p in run_dirs]
    shared = None
    if args.heatmap_dir:
        shared = Path(args.heatmap_dir)
        if not shared.is_absolute():
            shared = REPO_ROOT / shared
        shared.mkdir(parents=True, exist_ok=True)
        shared = shared.resolve()
    print(f"Watching {len(run_dirs)} run dir(s), interval={args.interval}s", flush=True)
    if shared:
        print(f"Shared live heatmaps -> {shared}", flush=True)
    while True:
        all_done = True
        stamp = datetime.now().strftime("%H:%M:%S")
        for run_dir in run_dirs:
            if not (run_dir / "config.json").is_file():
                print(f"[{stamp}] waiting for {run_dir}", flush=True)
                all_done = False
                continue
            config = load_json(run_dir / "config.json")
            if shared is not None:
                config["heatmap_dir"] = str(shared)
                atomic_write_json(run_dir / "config.json", config)
            by_task = aggregate_with_lock(run_dir, blocking=True)
            n_total = int(config["n_combinations"])
            n_ok = sum(1 for rows in by_task.values() for r in rows if r.get("status") == "ok")
            print(f"[{stamp}] {run_dir.name}: ok={n_ok}/{n_total}", flush=True)
            if n_ok < n_total:
                all_done = False
        if all_done:
            print("All watched run directories completed.")
            break
        time.sleep(args.interval)


def add_grid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()), default=list(TASKS.keys()))
    parser.add_argument("--lambda-min", type=float, default=-1.0)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--lambda-step", type=float, default=0.1,
                        help="step for lambda grid (default: 0.1 over [-1,1] => 21 values)")
    parser.add_argument("--lambda-steps", type=int, default=None,
                        help="alternative to --lambda-step: number of linspace points")
    parser.add_argument("--gamma-real-min", type=float, default=-0.2)
    parser.add_argument("--gamma-real-max", type=float, default=0.2)
    parser.add_argument("--gamma-real-step", type=float, default=0.02)
    parser.add_argument("--gamma-real-steps", type=int, default=None)
    parser.add_argument("--alpha-min", type=float, default=0.04)
    parser.add_argument("--alpha-max", type=float, default=5.0)
    parser.add_argument("--alpha-steps", type=int, default=None)
    parser.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=[0.04, 0.5, 1.0, 5.0],
        help="explicit alpha list (default: 0.04 0.5 1 5)",
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="seeds 1..N (default: 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="explicit seed list")
    parser.add_argument("--num-hidden", type=int, default=DEFAULT_NUM_HIDDEN)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--heatmap-dir",
        type=str,
        default=None,
        help="shared live heatmap output (also written under each run-dir/heatmaps)",
    )
    parser.add_argument("--workers", type=int, default=32, help="parallel workers for `local`")
    parser.add_argument("--dry-run", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda / gamma_real / alpha grid search")
    sub = parser.add_subparsers(dest="command", required=True)

    p_local = sub.add_parser("local", help="Run grid locally with N parallel workers (for `run -c 32`)")
    add_grid_args(p_local)

    p_worker = sub.add_parser("worker", help="Run one grid combination")
    p_worker.add_argument("--run-dir", type=str, required=True)
    p_worker.add_argument("--index", type=int, default=None)
    p_worker.add_argument("--dry-run", action="store_true")

    p_agg = sub.add_parser("aggregate", help="Merge job results and refresh heatmaps")
    p_agg.add_argument("--run-dir", type=str, required=True)

    p_sum = sub.add_parser(
        "summary",
        help="Write clean heatmaps under results/grid_search/summary/<task>/ (MG clipped, no cell text)",
    )
    p_sum.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="one or more lga_* run directories",
    )

    p_watch = sub.add_parser("watch", help="Periodically aggregate while jobs run")
    p_watch.add_argument("--run-dir", type=str, default=None)
    p_watch.add_argument("--run-dirs", nargs="+", default=None, help="multiple run dirs (split jobs)")
    p_watch.add_argument(
        "--heatmap-dir",
        type=str,
        default=None,
        help="optional shared live heatmap directory for all watched runs",
    )
    p_watch.add_argument("--interval", type=int, default=60)

    args = parser.parse_args()
    if args.command == "local":
        # if user passed --lambda-steps, disable default step mode
        if args.lambda_steps is not None:
            args.lambda_step = None
        if args.gamma_real_steps is not None:
            args.gamma_real_step = None
        if args.alpha_steps is not None:
            args.alpha_values = None
        cmd_local(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "watch":
        cmd_watch(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
