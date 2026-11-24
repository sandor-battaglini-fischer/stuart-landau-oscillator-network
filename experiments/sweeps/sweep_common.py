"""Shared helpers for multi-seed parameter sweeps with statistical aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
    mg_final_epoch: bool = False,
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
        mg_metrics = parse_mackey_glass_metrics(metrics_path, final_epoch=mg_final_epoch)
        if mg_metrics is None:
            return None
        record.update(mg_metrics)
        curve = parse_epoch_metrics(metrics_path, "test_r2")
        if curve is not None:
            record["epoch_curve"] = curve
        metric = scalar_metric_for_entry(record)
        if metric is None:
            return None
        record["metric"] = metric
        record["test_r2"] = metric
    else:
        curve = parse_epoch_metrics(metrics_path, "test_acc") if metrics_path.is_file() else None
        if curve is not None:
            record["epoch_curve"] = curve
        metric = scalar_metric_for_entry(record)
        if metric is None:
            metric = classification_metric_from_logs(sweep_log_path, run_log_path, metrics_path)
        if metric is None:
            return None
        record["metric"] = metric
        record["test_acc"] = metric

    if params.get("horizon") not in (None, ""):
        record["horizon"] = int(float(params["horizon"]))

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


def _mackey_glass_metrics_row(epochs: list[dict], *, final_epoch: bool) -> dict:
    if final_epoch:
        return max(epochs, key=lambda row: row["epoch"])
    return min(
        epochs,
        key=lambda row: row["val_normalized_error"]
        if row.get("val_normalized_error") is not None
        else float("inf"),
    )


def parse_mackey_glass_best_metrics(metrics_path: Path) -> dict[str, float] | None:
    return parse_mackey_glass_metrics(metrics_path, final_epoch=False)


def parse_mackey_glass_metrics(
    metrics_path: Path,
    *,
    final_epoch: bool = False,
) -> dict[str, float] | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open() as f:
        epochs = json.load(f)
    if not epochs:
        return None
    selected = _mackey_glass_metrics_row(epochs, final_epoch=final_epoch)
    return {
        "test_r2": float(selected["test_r2"]),
        "val_r2": float(selected["val_r2"]),
        "test_normalized_error": float(selected["test_normalized_error"]),
        "val_normalized_error": float(selected["val_normalized_error"]),
        "test_mse": float(selected["test_mse"]),
        "val_mse": float(selected["val_mse"]),
        "best_epoch": int(selected["epoch"]),
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


def scalar_metric_for_entry(entry: dict, metric_key: str = "metric") -> float | None:
    curve = entry.get("epoch_curve")
    if curve:
        return float(curve[-1])
    value = entry.get(metric_key)
    if value is None:
        return None
    return float(value)


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
        if entry.get("status") != "ok":
            continue
        metric = scalar_metric_for_entry(entry, metric_key)
        if metric is None:
            continue
        key = entry[group_key]
        if isinstance(key, float):
            key = round(key, 10)
        by_group.setdefault(key, []).append(metric)

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
    use_gpu: bool = False,
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
    print(" ".join(cmd), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"
    if use_gpu:
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_NUM_THREADS", "4")
        env.setdefault("NUMEXPR_NUM_THREADS", "4")
        env.setdefault("TORCH_NUM_THREADS", "4")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("TORCH_NUM_THREADS", "1")
        ld = env.get("LD_LIBRARY_PATH", "")
        if ld:
            cleaned = [
                p
                for p in ld.split(":")
                if p and "cuda" not in p.lower() and "nvidia" not in p.lower()
            ]
            env["LD_LIBRARY_PATH"] = ":".join(cleaned)

    stdout_chunks: list[str] = []
    with log_path.open("w") as log_f:
        log_f.write("CMD: " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_chunks.append(line)
            log_f.write(line)
            log_f.flush()
        returncode = proc.wait()

    stdout = "".join(stdout_chunks)
    stderr = ""

    record["returncode"] = returncode
    if output_dir is None:
        output_dir = find_newest_output_dir(output_glob, start)
    if output_dir is not None:
        try:
            record["output_dir"] = str(output_dir.relative_to(REPO_ROOT))
        except ValueError:
            record["output_dir"] = str(output_dir)

    if returncode != 0:
        record["status"] = "failed"
        record["error"] = (stderr or stdout)[-2000:]
        return record

    if task_key == "mackey_glass":
        metrics_path = output_dir / "metrics.json" if output_dir else None
        mg_metrics = parse_mackey_glass_best_metrics(metrics_path) if metrics_path else None
        if mg_metrics is None:
            record["status"] = "parse_failed"
            return record
        record.update(mg_metrics)
        curve = parse_epoch_metrics(metrics_path, "test_r2") if metrics_path else None
        if curve is not None:
            record["epoch_curve"] = curve
        metric = scalar_metric_for_entry(record)
        if metric is None:
            record["status"] = "parse_failed"
            return record
        record["metric"] = metric
        record["test_r2"] = metric
    else:
        metrics_path = output_dir / "metrics.json" if output_dir else None
        curve = parse_epoch_metrics(metrics_path, "test_acc") if metrics_path else None
        if curve is not None:
            record["epoch_curve"] = curve
        metric = scalar_metric_for_entry(record)
        if metric is None:
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
    legend_loc: str = "lower right",
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
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, ncol=2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if max_epoch is not None:
        ax.set_xlim(0, max_epoch)


def load_sweep_config(sweep_dir: Path) -> dict:
    path = sweep_dir / "sweep_config.json"
    if not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def observed_horizons(
    results: dict,
    tasks: list[str],
    horizon_key: str = "horizon",
) -> list[int]:
    found: set[int] = set()
    for task_key in tasks:
        for entry in results.get(task_key, []):
            h = entry.get(horizon_key)
            if h is not None:
                found.add(int(h))
    return sorted(found)


def union_horizons(*groups: list[int] | None) -> list[int] | None:
    merged: set[int] = set()
    for group in groups:
        if group:
            merged.update(int(h) for h in group)
    return sorted(merged) if merged else None


def select_plot_horizons(present: list[int], requested: list[int] | None) -> list[int]:
    if requested is None:
        return list(present)
    wanted = {int(h) for h in requested}
    return [h for h in present if h in wanted]


def resolve_plot_horizons(
    cli_horizons: list[int] | None,
    cfg: dict,
    results: dict,
    tasks: list[str],
    horizon_key: str = "horizon",
) -> list[int] | None:
    if cli_horizons is not None:
        return cli_horizons
    observed = observed_horizons(results, tasks, horizon_key)
    configured = [int(h) for h in cfg["horizons"]] if cfg.get("horizons") else []
    if configured:
        return union_horizons(configured, observed)
    if not observed:
        return None
    return list(range(min(observed), max(observed) + 1))


def print_finished_remaining(categories: list[tuple[str, int, int]]) -> None:
    tot_done = 0
    tot_rem = 0
    for name, done, total in categories:
        rem = max(int(total) - int(done), 0)
        print(f"{name}  {done}  {rem}")
        tot_done += int(done)
        tot_rem += rem
    if len(categories) > 1:
        print(f"total  {tot_done}  {tot_rem}")


def load_configured_horizons(sweep_dir: Path) -> list[int] | None:
    cfg = sweep_dir / "sweep_config.json"
    if not cfg.is_file():
        return None
    with cfg.open() as f:
        raw = json.load(f).get("horizons")
    if not raw:
        return None
    return [int(h) for h in raw]


def horizon_color_scale(present: list[int], sweep_dir: Path | None = None):
    configured = load_configured_horizons(sweep_dir) if sweep_dir is not None else None
    scale = union_horizons(configured, present) or [1]
    vmin = float(scale[0])
    vmax = float(scale[-1])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return plt.cm.viridis, Normalize(vmin=vmin, vmax=vmax)


def horizon_colorbar_ticks(norm: Normalize) -> list[int]:
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)
    span = vmax - vmin
    if span <= 10:
        step = 1
    elif span <= 25:
        step = 5
    else:
        step = 10
    start = int(math.ceil(vmin / step) * step)
    ticks = list(range(start, int(vmax) + 1, step))
    lo = int(round(vmin))
    hi = int(round(vmax))
    if not ticks or ticks[0] != lo:
        ticks = [lo] + [t for t in ticks if t > lo]
    if ticks[-1] != hi:
        ticks.append(hi)
    return sorted(dict.fromkeys(ticks))


def add_horizon_colorbar(fig, ax, cmap, norm, label: str = "Horizon"):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_ticks(horizon_colorbar_ticks(norm))
    label_size = ax.yaxis.label.get_size()
    tick_size = plt.rcParams.get("ytick.labelsize", label_size)
    cbar.set_label(label, size=label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    return cbar


def mackey_glass_horizons(results: dict) -> list[int]:
    return sorted({
        int(entry["horizon"])
        for entry in results.get("mackey_glass", [])
        if entry.get("status") == "ok" and entry.get("horizon") is not None
    })


def default_mackey_glass_horizon(tasks: dict) -> int | None:
    cmd = tasks["mackey_glass"].base_cmd
    try:
        return int(cmd[cmd.index("--horizon") + 1])
    except (ValueError, IndexError):
        return None


def task_panel_title(task_key: str, task, results: dict, tasks: dict) -> str:
    if task_key != "mackey_glass":
        return task.name
    hs = mackey_glass_horizons(results)
    h = hs[0] if len(hs) == 1 else (default_mackey_glass_horizon(tasks) if not hs else None)
    return task.name if h is None else rf"{task.name} $H={h}$"


def other_tasks_have_data(results: dict, tasks: dict) -> bool:
    return any(
        entry.get("status") == "ok"
        for task_key in tasks if task_key != "mackey_glass"
        for entry in results.get(task_key, [])
    )


def multi_task_panels(results: dict, tasks: dict, horizons: list[int] | None = None) -> list[tuple]:
    """(task_key, task, title, entries) per plot panel, with Mackey-Glass split by horizon."""
    panels = []
    for task_key, task in tasks.items():
        entries = [e for e in results.get(task_key, []) if e.get("status") == "ok"]
        plot_hs = select_plot_horizons(mackey_glass_horizons(results), horizons) if task_key == "mackey_glass" else []
        if not plot_hs:
            panels.append((task_key, task, task_panel_title(task_key, task, results, tasks), entries))
        for h in plot_hs:
            subset = [e for e in entries if e.get("horizon") == h]
            panels.append((task_key, task, rf"{task.name} $H={h}$", subset or entries))
    return panels
