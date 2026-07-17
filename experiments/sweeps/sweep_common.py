"""Shared helpers for multi-seed parameter sweeps with statistical aggregation."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

SWEEP_TRAIN_FLAGS = ["--sweep-mode"]

RANDOM_GUESS_BASELINE: dict[str, float] = {
    "imdb": 50.0,
    "smnist": 10.0,
    "mackey_glass": 0.0,
    "mg": 0.0,
}

BEST_TEST_RE = re.compile(r"best test:\s*([\d.]+)", re.IGNORECASE)
LOG_PARAM_RE = re.compile(r"^([^:]+):\s*(.+)$")


def parse_run_log_header(log_path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    if not log_path.is_file():
        return params
    in_params = False
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.rstrip(":") in ("Training Parameters", "Command-line Arguments"):
            in_params = True
            continue
        if not in_params:
            continue
        if stripped.startswith("===="):
            if params:
                break
            continue
        match = LOG_PARAM_RE.match(stripped)
        if match:
            params[match.group(1).strip()] = match.group(2).strip()
    return params


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_classification_best_metrics(metrics_path: Path) -> dict[str, float] | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open() as f:
        epochs = json.load(f)
    if not epochs:
        return None
    best_epoch = max(epochs, key=lambda row: row.get("val_acc", float("-inf")))
    return {
        "test_acc": float(best_epoch["test_acc"]),
        "val_acc": float(best_epoch["val_acc"]),
        "best_epoch": int(best_epoch["epoch"]),
    }


def classification_metric_from_logs(
    sweep_log_path: Path | None,
    run_log_path: Path | None,
    metrics_path: Path | None,
) -> float | None:
    for log_path in (sweep_log_path, run_log_path):
        if log_path is not None and log_path.is_file():
            metric = parse_classification_metric(log_path.read_text())
            if metric is not None:
                return metric
    if metrics_path is not None:
        best = parse_classification_best_metrics(metrics_path)
        if best is not None:
            return best["test_acc"]
    return None


def record_from_sweep_run_dir(
    run_dir: Path,
    task_key: str,
    *,
    group_key: str,
    variant_key: str,
    seed: int | None = None,
    source_dir: Path | None = None,
) -> dict | None:
    run_log_path = run_dir / "log.txt"
    metrics_path = run_dir / "metrics.json"
    sweep_log_path = (
        source_dir / "logs" / f"{run_dir.name}.log"
        if source_dir is not None
        else None
    )
    if not run_log_path.is_file():
        return None

    params = parse_run_log_header(run_log_path)
    parsed_seed = params.get("seed")
    if parsed_seed is not None:
        seed = int(parsed_seed)
    if seed is None:
        return None
    dynamics = params.get("dynamics")
    if dynamics is None:
        return None
    use_tanh = params.get("use_tanh", "True").lower() in ("true", "1", "yes")
    if "use_tanh" not in params and "no_tanh" in params:
        use_tanh = params.get("no_tanh", "False").lower() not in ("true", "1", "yes")
    num_hidden = int(params.get("num_hidden", 50))

    if sweep_log_path is not None and sweep_log_path.is_file():
        record_log_path = sweep_log_path
    else:
        record_log_path = run_log_path

    record: dict = {
        "task": task_key,
        group_key: variant_key,
        "dynamics": dynamics,
        "model": "horn" if dynamics == "dho" else "slon",
        "use_tanh": use_tanh,
        "num_hidden": num_hidden,
        "seed": seed,
        "output_dir": _relative_to_repo(run_dir),
        "log_path": _relative_to_repo(record_log_path),
    }

    if task_key == "mackey_glass":
        mg_metrics = parse_mackey_glass_best_metrics(metrics_path)
        if mg_metrics is None:
            return None
        record.update(mg_metrics)
        record["metric"] = mg_metrics["test_r2"]
        curve = parse_epoch_metrics(metrics_path, "test_r2")
        if curve is not None:
            record["epoch_curve"] = curve
    else:
        metric = classification_metric_from_logs(sweep_log_path, run_log_path, metrics_path)
        curve = parse_epoch_metrics(metrics_path, "test_acc") if metrics_path.is_file() else None
        if metric is None:
            return None
        record["metric"] = metric
        record["test_acc"] = metric
        if curve is not None:
            record["epoch_curve"] = curve

    record["status"] = "ok"
    return record


def find_newest_output_dir(pattern: str, since: float) -> Path | None:
    candidates = [
        p for p in REPO_ROOT.glob(pattern)
        if p.is_dir() and p.stat().st_mtime >= since - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_classification_metric(text: str) -> float | None:
    matches = BEST_TEST_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_mackey_glass_best_metrics(metrics_path: Path) -> dict[str, float] | None:
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


def parse_epoch_metrics(metrics_path: Path, metric_key: str) -> list[float] | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open() as f:
        epochs = json.load(f)
    if not epochs:
        return None
    rows = sorted(epochs, key=lambda row: row["epoch"])
    return [float(row[metric_key]) for row in rows if metric_key in row]


def load_results(results_path: Path, task_keys: list[str]) -> dict:
    if results_path.is_file():
        with results_path.open() as f:
            return json.load(f)
    return {task: [] for task in task_keys}


def save_results(results_path: Path, results: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)


def save_csv(results_path: Path, results: dict, extra_fields: list[str] | None = None) -> None:
    csv_path = results_path.with_suffix(".csv")
    rows = []
    for task_key, entries in results.items():
        for entry in entries:
            row = {
                "task": task_key,
                "seed": entry.get("seed"),
                "metric": entry.get("metric"),
                "status": entry.get("status"),
                "output_dir": entry.get("output_dir"),
            }
            if extra_fields:
                for field in extra_fields:
                    row[field] = entry.get(field)
            rows.append(row)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_scalar_results(
    entries: list[dict],
    group_key: str,
    metric_key: str = "metric",
) -> list[dict]:
    by_group: dict[float | str, list[float]] = {}
    for entry in entries:
        if entry.get("status") != "ok" or entry.get(metric_key) is None:
            continue
        key = entry[group_key]
        if isinstance(key, float):
            key = round(key, 10)
        by_group.setdefault(key, []).append(float(entry[metric_key]))

    summary = []
    for key in sorted(by_group, key=lambda v: (isinstance(v, str), v)):
        metrics = np.asarray(by_group[key], dtype=float)
        n = len(metrics)
        mean = float(np.mean(metrics))
        std = float(np.std(metrics, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * sem
        summary.append(
            {
                group_key: key,
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


def aggregate_epoch_curves(
    entries: list[dict],
    group_key: str,
    curve_key: str = "epoch_curve",
) -> dict[float | str, dict]:
    by_group: dict[float | str, list[np.ndarray]] = {}
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        curve = entry.get(curve_key)
        if not curve:
            continue
        key = entry[group_key]
        if isinstance(key, float):
            key = round(key, 10)
        by_group.setdefault(key, []).append(np.asarray(curve, dtype=float))

    summary = {}
    for key, curves in by_group.items():
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan, dtype=float)
        for i, c in enumerate(curves):
            padded[i, : len(c)] = c

        counts = np.sum(~np.isnan(padded), axis=0)
        mean = np.nanmean(padded, axis=0)
        std = np.where(counts > 1, np.nanstd(padded, axis=0, ddof=1), 0.0)
        sem = np.where(counts > 1, std / np.sqrt(np.maximum(counts, 1)), 0.0)
        ci95 = 1.96 * sem
        summary[key] = {
            "epochs": np.arange(max_len),
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci95": ci95,
            "n_runs": len(curves),
            "n_per_epoch": counts,
        }
    return summary


def save_summary(results_path: Path, results: dict, group_key: str, task_keys: list[str]) -> None:
    summary = {
        task_key: aggregate_scalar_results(results.get(task_key, []), group_key)
        for task_key in task_keys
    }
    summary_path = results_path.with_name("summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


def cmd_with_overrides(
    base_cmd: list[str],
    overrides: dict[str, str | int | float],
) -> list[str]:
    override_flags = set(overrides)
    cmd: list[str] = []
    i = 0
    while i < len(base_cmd):
        flag = base_cmd[i]
        if flag in override_flags:
            i += 2
            continue
        cmd.append(flag)
        i += 1
    for flag, value in overrides.items():
        cmd.extend([flag, str(value)])
    return cmd


def run_training_cmd(
    cmd: list[str],
    log_path: Path,
    output_glob: str,
    task_key: str,
    metric_key: str,
    dry_run: bool,
    output_dir: Path | None = None,
) -> dict:
    record: dict = {
        "status": "pending",
        "command": cmd,
        "log_path": str(log_path),
    }

    if dry_run:
        record["status"] = "dry_run"
        print("DRY RUN:", " ".join(cmd))
        return record

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(" ".join(cmd))

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_path.write_text(stdout + ("\n" + stderr if stderr else ""))

    record["returncode"] = proc.returncode
    if output_dir is None:
        output_dir = find_newest_output_dir(output_glob, start)
    if output_dir is not None:
        try:
            record["output_dir"] = str(output_dir.relative_to(REPO_ROOT))
        except ValueError:
            record["output_dir"] = str(output_dir)

    if proc.returncode != 0:
        record["status"] = "failed"
        record["error"] = stderr[-2000:] if stderr else stdout[-2000:]
        return record

    if task_key == "mackey_glass":
        metrics_path = output_dir / "metrics.json" if output_dir else None
        mg_metrics = parse_mackey_glass_best_metrics(metrics_path) if metrics_path else None
        if mg_metrics is None:
            record["status"] = "parse_failed"
            return record
        record.update(mg_metrics)
        record["metric"] = mg_metrics["test_r2"]
        curve = parse_epoch_metrics(metrics_path, "test_r2") if metrics_path else None
        if curve is not None:
            record["epoch_curve"] = curve
    else:
        metric = parse_classification_metric(stdout)
        metrics_path = output_dir / "metrics.json" if output_dir else None
        curve = parse_epoch_metrics(metrics_path, "test_acc") if metrics_path else None
        if metric is None and output_dir is not None:
            log_file = output_dir / "log.txt"
            if log_file.is_file():
                metric = parse_classification_metric(log_file.read_text())
        if metric is None and curve:
            metric = float(curve[-1])
        if metric is None:
            record["status"] = "parse_failed"
            return record
        record["metric"] = metric
        record["test_acc"] = metric
        if curve is not None:
            record["epoch_curve"] = curve

    record["status"] = "ok"
    return record


def plot_scalar_sweep(
    ax,
    summary: list[dict],
    group_key: str,
    color,
    ylabel: str,
    xlabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    legend: bool = True,
) -> None:
    if not summary:
        if title:
            ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        return

    xs = [row[group_key] for row in summary]
    means = [row["mean"] for row in summary]
    ci95 = [row["ci95"] for row in summary]

    lower_ci = [m - c for m, c in zip(means, ci95)]
    upper_ci = [m + c for m, c in zip(means, ci95)]

    ax.fill_between(
        xs, lower_ci, upper_ci, color=color, alpha=0.25, linewidth=0, label="$95\\%$ CI"
    )
    ax.plot(xs, means, "o-", color=color, linewidth=2, markersize=5, label="mean")
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left")
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="lower right", frameon=True, fontsize=14)


def plot_epoch_curves_with_stats(
    ax,
    curves_by_group: dict,
    colors: list,
    labels: list[str] | None = None,
    ylabel: str = "",
    xlabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    max_epoch: int | None = 40,
    legend_fontsize: float = 11,
    random_guess: float | None = None,
) -> None:
    for idx, (group_val, stats) in enumerate(curves_by_group.items()):
        color = colors[idx % len(colors)]
        mean = np.asarray(stats["mean"], dtype=float)
        ci95 = np.asarray(stats["ci95"], dtype=float)

        if random_guess is not None:
            mean = np.concatenate([[random_guess], mean])
            ci95 = np.concatenate([[0.0], ci95])
        epochs = np.arange(len(mean))

        if max_epoch is not None:
            keep = epochs <= max_epoch
            epochs = epochs[keep]
            mean = mean[keep]
            ci95 = ci95[keep]

        label = labels[idx] if labels and idx < len(labels) else str(group_val)

        ax.fill_between(epochs, mean - ci95, mean + ci95, color=color, alpha=0.22, linewidth=0)
        ax.plot(epochs, mean, linewidth=2.5, color=color, label=label)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=legend_fontsize, ncol=2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if max_epoch is not None:
        ax.set_xlim(0, max_epoch)


def add_stats_legend(fig, y: float = 1.08) -> None:
    handles = [
        plt.Line2D([0], [0], color="gray", linewidth=2, label="mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.25, label="$95\\%$ CI"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False)
