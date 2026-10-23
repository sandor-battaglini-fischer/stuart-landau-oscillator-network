#!/usr/bin/env python3
"""Sweep hidden size N across IMDB, sMNIST, and Mackey-Glass with multi-seed statistics."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.sweep_common import (
    add_horizon_colorbar,
    aggregate_epoch_curves,
    aggregate_scalar_results,
    cmd_with_overrides,
    horizon_color_scale,
    load_results,
    load_sweep_config,
    observed_horizons,
    plot_epoch_curves_with_stats,
    resolve_plot_horizons,
    select_plot_horizons,
    union_horizons,
    plot_scalar_sweep,
    print_finished_remaining,
    record_from_sweep_run_dir,
    run_training_cmd,
    save_csv,
    save_results,
    save_summary,
)
from utils.plotting_utils.style import mycmap

GROUP_KEY = "num_hidden"
HORIZON_KEY = "horizon"
DEFAULT_HIDDEN = [1, 2, 4, 9, 50, 128]


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
        name="IMDb",
        output_glob="results/imdb/*",
        metric_label=r"Test accuracy ($\%$)",
        epoch_ylabel=r"Test accuracy ($\%$)",
        use_r2=False,
        random_guess=50.0,
        ylim=(50.0, 90.0),
        default_hidden=list(DEFAULT_HIDDEN),
        base_cmd=[
            sys.executable,
            "training/train_imdb.py",
            "--epochs", "10",
            "--early-stop-patience", "10",
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
        default_hidden=list(DEFAULT_HIDDEN),
        base_cmd=[
            sys.executable,
            "training/train_smnist.py",
            "--lambda-param", "0.1",
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
        ylim=(-0.5, 1.0),
        default_hidden=list(DEFAULT_HIDDEN),
        base_cmd=[
            sys.executable,
            "training/train_mackey_glass.py",
            "--epochs", "10",
            "--early-stop-patience", "10",
            "--input-length", "100",
            "--gamma", "0.01",
            "--lambda-param", "-0.1",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--omega", "0.15",
            "--mg-tau", "17.0",
            "--horizon", "25",
            "--series-length", "20000",
            "--val-fraction", "0.1",
            "--test-fraction", "0.1",
            "--lr", "1e-2",
            "--batch-size", "64",
            "--lr-decay-power", "1.0",
            "--min-lr-ratio", "0.0",
            "--skip-epoch-plots",
            "--sweep-mode",
        ],
    ),
}


def extra_result_fields() -> list[str]:
    return [GROUP_KEY, HORIZON_KEY]


def parse_scaling_run_name(run_name: str, task_keys: list[str]) -> tuple[str, int, int, int | None] | None:
    for task_key in sorted(task_keys, key=len, reverse=True):
        prefix = f"{task_key}_"
        if not run_name.startswith(prefix):
            continue
        rest = run_name[len(prefix) :]
        if not rest.startswith("N") or "_seed" not in rest:
            continue
        body, seed_part = rest.rsplit("_seed", 1)
        if not seed_part.isdigit():
            continue
        horizon = None
        if "_h" in body:
            n_part, h_part = body.rsplit("_h", 1)
            if h_part.isdigit():
                horizon = int(h_part)
                body = n_part
        if not body.startswith("N") or not body[1:].isdigit():
            continue
        return task_key, int(body[1:]), int(seed_part), horizon
    return None


def refresh_from_runs(sweep_dir: Path, task_keys: list[str]) -> dict:
    results = {task_key: [] for task_key in TASKS}
    runs_dir = sweep_dir / "runs"
    if not runs_dir.is_dir():
        return results
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_scaling_run_name(run_dir.name, task_keys)
        if parsed is None:
            continue
        task_key, num_hidden, seed, horizon = parsed
        record = record_from_sweep_run_dir(
            run_dir,
            task_key,
            group_key=GROUP_KEY,
            variant_key=num_hidden,
            seed=seed,
            source_dir=sweep_dir,
        )
        if record is None:
            continue
        record[GROUP_KEY] = num_hidden
        if horizon is not None:
            record[HORIZON_KEY] = horizon
        results[task_key].append(record)
    return results


def existing_completions(results: dict, task_key: str) -> set[tuple[int, int, int | None]]:
    return {
        (int(entry[GROUP_KEY]), int(entry.get("seed", 1)), entry.get(HORIZON_KEY))
        for entry in results.get(task_key, [])
        if entry.get("status") == "ok"
    }


def merge_record(
    results: dict,
    task_key: str,
    num_hidden: int,
    seed: int,
    horizon: int | None,
    record: dict,
) -> None:
    results.setdefault(task_key, [])
    results[task_key] = [
        e
        for e in results[task_key]
        if not (
            int(e.get(GROUP_KEY, -1)) == num_hidden
            and int(e.get("seed", -1)) == seed
            and e.get(HORIZON_KEY) == horizon
        )
    ]
    if record.get("status") != "dry_run":
        results[task_key].append(record)


def save_results_locked(results_path: Path, results: dict) -> None:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_result_fields())
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))


def append_record_locked(
    results_path: Path,
    task_key: str,
    num_hidden: int,
    seed: int,
    horizon: int | None,
    record: dict,
) -> dict:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        results = load_results(results_path, list(TASKS.keys()))
        merge_record(results, task_key, num_hidden, seed, horizon, record)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_result_fields())
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
        return results


def run_single(
    task_key: str,
    num_hidden: int,
    seed: int,
    sweep_dir: Path,
    dry_run: bool,
    epochs: int | None = None,
    horizon: int | None = None,
    use_gpu: bool = False,
) -> dict:
    task = TASKS[task_key]
    if horizon is not None:
        run_name = f"{task_key}_N{num_hidden}_h{horizon}_seed{seed}"
    else:
        run_name = f"{task_key}_N{num_hidden}_seed{seed}"
    run_output_dir = sweep_dir / "runs" / run_name
    overrides: dict = {"--num-hidden": num_hidden, "--seed": seed, "--output-dir": str(run_output_dir)}
    if use_gpu:
        overrides["--device"] = "cuda"
    if epochs is not None:
        overrides["--epochs"] = epochs
        if "--early-stop-patience" in task.base_cmd:
            overrides["--early-stop-patience"] = epochs
    if horizon is not None:
        overrides["--horizon"] = horizon
    cmd = cmd_with_overrides(task.base_cmd, overrides)
    log_path = sweep_dir / "logs" / f"{run_name}.log"

    record = {
        "task": task_key,
        GROUP_KEY: num_hidden,
        "seed": seed,
        **run_training_cmd(
            cmd, log_path, task.output_glob, task_key, "metric", dry_run,
            output_dir=run_output_dir, use_gpu=use_gpu,
        ),
    }
    if horizon is not None:
        record[HORIZON_KEY] = horizon
    if record.get("status") == "ok":
        print(f"  -> {task.metric_label}: {record['metric']:.4f}", flush=True)
    return record


def _worker_entry(job: tuple) -> tuple[str, int, int, int | None, dict]:
    task_key, num_hidden, seed, horizon, sweep_dir_str, dry_run, epochs, use_gpu = job
    stagger_s = (int(seed) % 8) * 1.5
    if stagger_s:
        time.sleep(stagger_s)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    record = run_single(
        task_key,
        num_hidden,
        seed,
        Path(sweep_dir_str),
        dry_run,
        epochs=epochs,
        horizon=horizon,
        use_gpu=use_gpu,
    )
    return task_key, num_hidden, seed, horizon, record


def horizons_for_task(task_key: str, horizons: list[int] | None) -> list[int | None]:
    if task_key == "mackey_glass" and horizons is not None:
        return list(horizons)
    return [None]


def build_jobs(
    tasks: list[str],
    task_grids: dict[str, list[int]],
    seeds: list[int],
    horizons: list[int] | None,
    done_by_task: dict[str, set[tuple[int, int, int | None]]],
) -> list[tuple]:
    jobs = []
    for task_key in tasks:
        done = done_by_task.get(task_key, set())
        for num_hidden in task_grids[task_key]:
            for horizon in horizons_for_task(task_key, horizons):
                for seed in seeds:
                    if (num_hidden, seed, horizon) in done:
                        continue
                    jobs.append((task_key, num_hidden, seed, horizon))
    return jobs


def report_scaling_progress(
    results: dict,
    tasks: list[str],
    task_grids: dict[str, list[int]],
    seeds: list[int],
    horizons: list[int] | None,
) -> None:
    rows: list[tuple[str, int, int]] = []
    for task_key in tasks:
        hs = horizons_for_task(task_key, horizons)
        per_cat = len(seeds) * len(hs)
        done = existing_completions(results, task_key)
        observed = {int(n) for n, _s, _h in done}
        ns = sorted(set(task_grids.get(task_key, [])) | observed)
        prefix = f"{task_key} " if len(tasks) > 1 else ""
        if task_key == "mackey_glass":
            named = [h for h in hs if h is not None]
            if len(named) == 1:
                prefix = f"{prefix}H={named[0]} "
            elif not named:
                h = default_mackey_glass_horizon()
                if h is not None:
                    prefix = f"{prefix}H={h} "
        for n in ns:
            n_done = sum(1 for nn, _s, _h in done if int(nn) == int(n))
            rows.append((f"{prefix}N={n}", n_done, per_cat))
    print_finished_remaining(rows)


def update_live_plots(
    results: dict, sweep_dir: Path, horizons: list[int] | None = None
) -> None:
    plot_summary(results, sweep_dir / "scaling_performance.png", horizons=horizons)
    if mackey_glass_horizons(results) and not other_tasks_have_data(results):
        return
    plot_epoch_results(results, sweep_dir)


def run_jobs_parallel(
    jobs: list[tuple],
    sweep_dir: Path,
    results_path: Path,
    dry_run: bool,
    epochs: int | None,
    workers: int,
    use_gpu: bool,
) -> dict:
    results = load_results(results_path, list(TASKS.keys()))
    if not jobs:
        return results

    sweep_dir_str = str(sweep_dir)
    pending = [
        (task_key, num_hidden, seed, horizon, sweep_dir_str, dry_run, epochs, use_gpu)
        for task_key, num_hidden, seed, horizon in jobs
    ]

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_entry, job): job for job in pending}
        for fut in as_completed(futures):
            job = futures[fut]
            task_key, num_hidden, seed, horizon = job[:4]
            try:
                task_key, num_hidden, seed, horizon, record = fut.result()
            except Exception as exc:
                print(
                    f"Worker failed {task_key} N={num_hidden} seed={seed} "
                    f"horizon={horizon}: {exc}",
                    flush=True,
                )
                completed += 1
                continue

            task = TASKS[task_key]
            label = f"{task.name} | N={num_hidden} | seed={seed}"
            if horizon is not None:
                label += f" | h={horizon}"
            if record.get("status") == "ok":
                print(f"Completed {label} -> {record['metric']:.4f}", flush=True)
            else:
                print(f"Completed {label} -> {record.get('status')}", flush=True)

            results = append_record_locked(
                results_path, task_key, num_hidden, seed, horizon, record
            )
            completed += 1
            print(f"Progress: {completed}/{len(pending)}", flush=True)
            update_live_plots(results, sweep_dir)

    return results


def mackey_glass_horizons(results: dict) -> list[int]:
    horizons = {
        int(entry[HORIZON_KEY])
        for entry in results.get("mackey_glass", [])
        if entry.get("status") == "ok" and entry.get(HORIZON_KEY) is not None
    }
    return sorted(horizons)


def default_mackey_glass_horizon() -> int | None:
    cmd = TASKS["mackey_glass"].base_cmd
    try:
        return int(cmd[cmd.index("--horizon") + 1])
    except (ValueError, IndexError):
        return None


def mackey_glass_display_horizon(results: dict) -> int | None:
    hs = mackey_glass_horizons(results)
    if len(hs) == 1:
        return hs[0]
    if not hs:
        return default_mackey_glass_horizon()
    return None


def task_panel_title(task_key: str, task: TaskConfig, results: dict) -> str:
    if task_key != "mackey_glass":
        return task.name
    h = mackey_glass_display_horizon(results)
    if h is None:
        return task.name
    return rf"{task.name} $H={h}$"


def other_tasks_have_data(results: dict) -> bool:
    for task_key in TASKS:
        if task_key == "mackey_glass":
            continue
        if any(entry.get("status") == "ok" for entry in results.get(task_key, [])):
            return True
    return False


def plot_horizon_scaling_performance(
    results: dict, plot_path: Path, horizons: list[int] | None = None
) -> None:
    task_key = "mackey_glass"
    task = TASKS[task_key]
    entries = [e for e in results.get(task_key, []) if e.get("status") == "ok"]
    plot_hs = select_plot_horizons(mackey_glass_horizons(results), horizons)
    if not plot_hs:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    cmap, norm = horizon_color_scale(
        horizons or plot_hs,
        None if horizons is not None else plot_path.parent,
    )

    for horizon in plot_hs:
        color = cmap(norm(horizon))
        subset = [e for e in entries if e.get(HORIZON_KEY) == horizon]
        summary = aggregate_scalar_results(subset, GROUP_KEY)
        summary = sorted(summary, key=lambda row: int(row[GROUP_KEY]))
        if not summary:
            continue
        xs = [row[GROUP_KEY] for row in summary]
        means = [row["mean"] for row in summary]
        ci95 = [row["ci95"] for row in summary]
        lower_ci = [m - c for m, c in zip(means, ci95)]
        upper_ci = [m + c for m, c in zip(means, ci95)]
        ax.fill_between(xs, lower_ci, upper_ci, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            xs,
            means,
            "o-",
            color=color,
            linewidth=1.5,
            markersize=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$N$ (hidden units)")
    ax.set_ylabel(task.metric_label)
    ax.set_ylim(*task.ylim)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"(a) {task.name}", loc="left")
    ax.grid(True, alpha=0.3)
    add_horizon_colorbar(fig, ax, cmap, norm)
    fig.suptitle("Performance vs hidden size $N$", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_summary(
    results: dict, plot_path: Path, horizons: list[int] | None = None
) -> None:
    if mackey_glass_horizons(results) and not other_tasks_have_data(results):
        plot_horizon_scaling_performance(results, plot_path, horizons=horizons)
        return

    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(7, 3.2 * n_tasks), sharex=True)
    if n_tasks == 1:
        axes = [axes]

    for i, (ax, (task_key, task)) in enumerate(zip(axes, TASKS.items())):
        summary = aggregate_scalar_results(results.get(task_key, []), GROUP_KEY)
        plot_scalar_sweep(
            ax,
            summary,
            GROUP_KEY,
            "black",
            task.metric_label,
            ylim=task.ylim,
            title=f"({chr(ord('a') + i)}) {task_panel_title(task_key, task, results)}",
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

    for i, (ax, (task_key, task)) in enumerate(zip(axes, TASKS.items())):
        ax.set_title(
            f"({chr(ord('a') + i)}) {task_panel_title(task_key, task, results)}",
            loc="left",
        )
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

        legend_loc = "center right" if task_key == "smnist" else "lower right"
        plot_epoch_curves_with_stats(
            ax,
            ordered,
            colors,
            labels=labels,
            ylim=task.ylim,
            random_guess=task.random_guess,
            max_epoch=None,
            legend_loc=legend_loc,
        )

    if not has_data:
        plt.close(fig)
        return

    max_epoch = 0
    for task_key in TASKS:
        for entry in results.get(task_key, []):
            curve = entry.get("epoch_curve") or []
            if curve:
                max_epoch = max(max_epoch, len(curve) - 1)

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Epoch")
    for ax in axes:
        ax.set_xlim(0, max(max_epoch, 1))
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
        help="Comma-separated hidden sizes (default: 1,2,4,9,50,128)",
    )
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for all tasks")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Mackey-Glass forecast horizons (default: single run with base_cmd horizon)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel training workers (use 1 with --use-gpu unless you have multiple GPUs)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Run training on CUDA (request GPU with cluster run -g 1)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--refresh-from-runs",
        action="store_true",
        help="Rebuild results from runs/*/metrics.json before plotting",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else REPO_ROOT / f"results/sweeps/scaling_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"

    results = load_results(results_path, list(TASKS.keys()))
    if args.plot_only and args.refresh_from_runs:
        results = refresh_from_runs(sweep_dir, args.tasks)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_result_fields())
        print(f"Refreshed {sum(len(v) for v in results.values())} records from {sweep_dir / 'runs'}")
    cfg = load_sweep_config(sweep_dir) if args.plot_only else {}
    if args.seeds is not None:
        seeds = args.seeds
    elif args.plot_only and cfg.get("seeds"):
        seeds = [int(s) for s in cfg["seeds"]]
    else:
        seeds = list(range(1, args.num_runs + 1))

    horizons = args.horizons
    if args.plot_only:
        horizons = resolve_plot_horizons(args.horizons, cfg, results, args.tasks, HORIZON_KEY)
    if args.horizons is not None and "mackey_glass" not in args.tasks:
        parser.error("--horizons requires mackey_glass in --tasks")

    task_grids = {}
    for task_key in args.tasks:
        if args.num_hidden_list:
            task_grids[task_key] = [int(x.strip()) for x in args.num_hidden_list.split(",")]
        else:
            task_grids[task_key] = TASKS[task_key].default_hidden
        if args.plot_only:
            cfg_ns = [int(x) for x in (cfg.get("task_grids") or {}).get(task_key, [])]
            observed = {
                int(e[GROUP_KEY])
                for e in results.get(task_key, [])
                if e.get("status") == "ok" and e.get(GROUP_KEY) is not None
            }
            task_grids[task_key] = sorted(set(task_grids[task_key]) | set(cfg_ns) | observed)

    if args.plot_only:
        report_scaling_progress(results, args.tasks, task_grids, seeds, horizons)

    if not args.plot_only:
        if args.use_gpu and args.workers > 1:
            print(
                "Warning: --use-gpu with workers>1 shares one GPU across processes; "
                "use --workers 1 with -g 1, or -g N with --workers N on multi-GPU nodes."
            )
        os.environ.setdefault("OMP_NUM_THREADS", "4" if args.use_gpu else "1")
        os.environ.setdefault("MKL_NUM_THREADS", "4" if args.use_gpu else "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "4" if args.use_gpu else "1")

        existing = load_sweep_config(sweep_dir)
        merged_horizons = union_horizons(
            horizons,
            existing.get("horizons"),
            observed_horizons(results, args.tasks, HORIZON_KEY),
        )
        with (sweep_dir / "sweep_config.json").open("w") as f:
            json.dump(
                {
                    "num_runs": args.num_runs,
                    "seeds": seeds,
                    "epochs": args.epochs,
                    "horizons": merged_horizons,
                    "workers": args.workers,
                    "use_gpu": args.use_gpu,
                    "task_grids": task_grids,
                    "tasks": args.tasks,
                },
                f,
                indent=2,
            )

        done_by_task = {
            task_key: existing_completions(results, task_key) if args.skip_existing else set()
            for task_key in args.tasks
        }
        jobs = build_jobs(args.tasks, task_grids, seeds, horizons, done_by_task)
        n_total = sum(
            len(task_grids[t]) * len(seeds) * len(horizons_for_task(t, horizons))
            for t in args.tasks
        )
        n_skipped = n_total - len(jobs)
        if n_skipped:
            print(f"Skipping {n_skipped} already-completed jobs")
        print(f"Pending jobs: {len(jobs)}")

        workers = max(1, int(args.workers))
        if workers > 1 and jobs:
            results = run_jobs_parallel(
                jobs, sweep_dir, results_path, args.dry_run, args.epochs, workers, args.use_gpu
            )
        else:
            for task_key, num_hidden, seed, horizon in jobs:
                task = TASKS[task_key]
                label = f"{task.name} | N={num_hidden} | seed={seed}"
                if horizon is not None:
                    label += f" | h={horizon}"
                print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
                record = run_single(
                    task_key,
                    num_hidden,
                    seed,
                    sweep_dir,
                    args.dry_run,
                    epochs=args.epochs,
                    horizon=horizon,
                    use_gpu=args.use_gpu,
                )
                if record["status"] != "dry_run":
                    merge_record(results, task_key, num_hidden, seed, horizon, record)
                    save_results_locked(results_path, results)
                    update_live_plots(results, sweep_dir, horizons=args.horizons)

        save_results_locked(results_path, results)
    else:
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))

    update_live_plots(results, sweep_dir, horizons=args.horizons)
    print(f"\nResults: {results_path}")
    print(f"Summary: {results_path.with_name('summary.json')}")


if __name__ == "__main__":
    main()
