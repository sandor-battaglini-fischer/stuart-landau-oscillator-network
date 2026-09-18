#!/usr/bin/env python3
"""sMNIST scaling confounds: convergence (A), sample complexity (B), capacity (C).

Does not change defaults of scaling_sweep.py. New runs go under
results/sweeps/scaling_convergence_check/ and the report is written to
results/scaling_convergence_check.md.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.sweep_common import (
    aggregate_epoch_curves,
    aggregate_scalar_results,
    cmd_with_overrides,
    load_results,
    parse_epoch_metrics,
    plot_epoch_curves_with_stats,
    plot_scalar_sweep,
    run_training_cmd,
    save_csv,
    save_results,
    save_summary,
)
from utils.plotting_utils.style import mycmap

GROUP_KEY = "num_hidden"
FRACTION_KEY = "train_fraction"
TASK_KEYS = ["A", "B"]

DEFAULT_HIDDEN = [1, 2, 4, 9, 16, 25, 50, 128]
EXP_B_HIDDEN = [9, 128]
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
LARGE_N_MIN = 50
REF_N = 128

PLATEAU_WINDOW = 5
PLATEAU_GAIN_PP = 0.2
RISING_WINDOW = 10
GAP_PASS_PP = 3.0

OPERATING_POINT = {
    "--lambda-param": "0.0",
    "--omega": "0.224",
    "--gamma-real": "-0.1",
    "--gamma-imag": "0.0",
    "--alpha": "0.04",
    "--lr": "1e-2",
    "--batch-size": "64",
    "--dynamics": "sl",
}

BASE_CMD = [
    sys.executable,
    "training/train_smnist.py",
    "--sweep-mode",
    "--log-train-acc",
    *[item for flag, value in OPERATING_POINT.items() for item in (flag, value)],
]


def ok_entries(results: dict, experiment: str) -> list[dict]:
    return [e for e in results.get(experiment, []) if e.get("status") == "ok"]


def finite_curve(curve) -> list[float]:
    if not curve:
        return []
    return [float(v) for v in curve if v is not None and v == v]


def best_test_acc(entry: dict) -> float | None:
    curve = finite_curve(entry.get("epoch_curve"))
    if curve:
        return float(max(curve))
    value = entry.get("best_test_acc", entry.get("metric"))
    return None if value is None else float(value)


def entries_best_test(entries: list[dict]) -> list[dict]:
    out = []
    for entry in entries:
        metric = best_test_acc(entry)
        if metric is None:
            continue
        rec = dict(entry)
        rec["epoch_curve"] = None
        rec["metric"] = metric
        out.append(rec)
    return out


def epochs_to_plateau(curve, window: int = PLATEAU_WINDOW, max_gain_pp: float = PLATEAU_GAIN_PP) -> int | None:
    y = np.asarray(finite_curve(curve), dtype=float)
    if len(y) < window:
        return None
    for t in range(len(y) - window + 1):
        if float(y[t + window - 1] - y[t]) < max_gain_pp:
            return int(t)
    return None


def still_rising(curve, window: int = RISING_WINDOW, min_gain_pp: float = PLATEAU_GAIN_PP) -> bool:
    y = np.asarray(finite_curve(curve), dtype=float)
    if len(y) < window:
        return True
    return float(y[-1] - y[-window]) >= min_gain_pp


def final_train_test_gap(entry: dict) -> float | None:
    test = finite_curve(entry.get("epoch_curve"))
    train = finite_curve(entry.get("train_epoch_curve"))
    if not test or not train:
        return None
    return float(train[-1] - test[-1])


def existing_completions(results: dict, experiment: str, min_epochs: int | None = None) -> set[tuple]:
    done = set()
    for entry in results.get(experiment, []):
        if entry.get("status") != "ok":
            continue
        n_trained = len(finite_curve(entry.get("epoch_curve")))
        if min_epochs is not None and n_trained < min_epochs:
            continue
        done.add(
            (
                int(entry[GROUP_KEY]),
                round(float(entry.get(FRACTION_KEY, 1.0)), 10),
                int(entry.get("seed", 1)),
            )
        )
    return done


def merge_record(results: dict, experiment: str, record: dict) -> None:
    results.setdefault(experiment, [])
    num_hidden = int(record[GROUP_KEY])
    fraction = round(float(record.get(FRACTION_KEY, 1.0)), 10)
    seed = int(record.get("seed", 1))
    results[experiment] = [
        e
        for e in results[experiment]
        if not (
            int(e.get(GROUP_KEY, -1)) == num_hidden
            and round(float(e.get(FRACTION_KEY, 1.0)), 10) == fraction
            and int(e.get("seed", -1)) == seed
        )
    ]
    if record.get("status") != "dry_run":
        results[experiment].append(record)


def extra_fields() -> list[str]:
    return [
        GROUP_KEY,
        FRACTION_KEY,
        "best_test_acc",
        "best_epoch",
        "epochs_trained",
        "participation_ratio",
        "effective_dim_pca",
    ]


def save_results_locked(results_path: Path, results: dict) -> None:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_fields())
        save_summary(results_path, results, GROUP_KEY, TASK_KEYS)


def append_record_locked(results_path: Path, experiment: str, record: dict) -> dict:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        results = load_results(results_path, TASK_KEYS)
        merge_record(results, experiment, record)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_fields())
        save_summary(results_path, results, GROUP_KEY, TASK_KEYS)
        return results


def resolve_output_dir(record: dict) -> Path | None:
    raw = record.get("output_dir")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def enrich_record(record: dict) -> dict:
    output_dir = resolve_output_dir(record)
    if output_dir is None:
        return record
    metrics_path = output_dir / "metrics.json"
    test_curve = parse_epoch_metrics(metrics_path, "test_acc")
    train_curve = parse_epoch_metrics(metrics_path, "train_acc")
    if test_curve:
        record["epoch_curve"] = test_curve
        finite = finite_curve(test_curve)
        if finite:
            best_i = int(np.argmax(np.asarray(finite, dtype=float)))
            record["best_test_acc"] = float(finite[best_i])
            record["best_epoch"] = best_i
            record["metric"] = record["best_test_acc"]
            record["epochs_trained"] = len(test_curve)
    if train_curve and finite_curve(train_curve):
        record["train_epoch_curve"] = train_curve
    manifold_path = output_dir / "manifold_dimension_results.json"
    if manifold_path.is_file():
        with manifold_path.open() as f:
            man = json.load(f)
        record["participation_ratio"] = man.get("participation_ratio")
        record["effective_dim_pca"] = man.get("effective_dim_pca")
        record["manifold"] = man
    return record


def parse_run_name(name: str) -> tuple[str, int, float, int] | None:
    if name.startswith("A_N") and "_seed" in name:
        body, seed_s = name.split("_seed", 1)
        n_s = body[len("A_N") :]
        if n_s.isdigit() and seed_s.isdigit():
            return "A", int(n_s), 1.0, int(seed_s)
    if name.startswith("B_N") and "_frac" in name and "_seed" in name:
        rest = name[len("B_N") :]
        n_s, rest = rest.split("_frac", 1)
        frac_s, seed_s = rest.split("_seed", 1)
        if n_s.isdigit() and seed_s.isdigit():
            return "B", int(n_s), float(frac_s.replace("p", ".")), int(seed_s)
    return None


def records_from_runs(sweep_dir: Path) -> dict:
    results = {key: [] for key in TASK_KEYS}
    runs_dir = sweep_dir / "runs"
    if not runs_dir.is_dir():
        return results
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        experiment, num_hidden, fraction, seed = parsed
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        record = {
            "experiment": experiment,
            "task": "smnist",
            GROUP_KEY: num_hidden,
            FRACTION_KEY: fraction,
            "seed": seed,
            "status": "ok",
            "output_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
        }
        record = enrich_record(record)
        if not finite_curve(record.get("epoch_curve")):
            continue
        merge_record(results, experiment, record)
    return results


def sync_results_from_runs(sweep_dir: Path, results_path: Path) -> dict:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        results = load_results(results_path, TASK_KEYS)
        from_disk = records_from_runs(sweep_dir)
        n_added = 0
        for experiment in TASK_KEYS:
            for record in from_disk.get(experiment, []):
                before = len(results.get(experiment, []))
                merge_record(results, experiment, record)
                if len(results.get(experiment, [])) >= before:
                    n_added += 1
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_fields())
        save_summary(results_path, results, GROUP_KEY, TASK_KEYS)
        print(
            f"Synced runs/: A={len(results.get('A', []))} B={len(results.get('B', []))} "
            f"records from {sweep_dir / 'runs'}"
        )
        return results


def run_name(experiment: str, num_hidden: int, train_fraction: float, seed: int) -> str:
    if experiment == "B":
        frac = f"{train_fraction:g}".replace(".", "p")
        return f"B_N{num_hidden}_frac{frac}_seed{seed}"
    return f"A_N{num_hidden}_seed{seed}"
    if experiment == "B":
        frac = f"{train_fraction:g}".replace(".", "p")
        return f"B_N{num_hidden}_frac{frac}_seed{seed}"
    return f"A_N{num_hidden}_seed{seed}"


def run_single(
    experiment: str,
    num_hidden: int,
    train_fraction: float,
    seed: int,
    sweep_dir: Path,
    dry_run: bool,
    epochs: int,
    analyze_manifold: bool,
    use_gpu: bool,
) -> dict:
    name = run_name(experiment, num_hidden, train_fraction, seed)
    run_output_dir = sweep_dir / "runs" / name
    metrics_path = run_output_dir / "metrics.json"
    if not dry_run and metrics_path.is_file():
        try:
            n_ep = len(json.loads(metrics_path.read_text()))
        except Exception:
            n_ep = 0
        if n_ep >= int(epochs):
            record = {
                "experiment": experiment,
                "task": "smnist",
                GROUP_KEY: num_hidden,
                FRACTION_KEY: train_fraction,
                "seed": seed,
                "epochs": epochs,
                "status": "ok",
                "output_dir": str(run_output_dir.relative_to(REPO_ROOT)),
            }
            record = enrich_record(record)
            print(f"  -> reusing completed run ({n_ep} epochs), best test: {record.get('metric')}", flush=True)
            return record
    if not dry_run and run_output_dir.exists():
        shutil.rmtree(run_output_dir)
    overrides: dict = {
        "--num-hidden": num_hidden,
        "--seed": seed,
        "--output-dir": str(run_output_dir),
        "--epochs": epochs,
        "--early-stop-patience": epochs,
        "--train-fraction": train_fraction,
    }
    if use_gpu:
        overrides["--device"] = "cuda"
    cmd = cmd_with_overrides(BASE_CMD, overrides)
    if analyze_manifold:
        cmd.append("--analyze-manifold-final")
    log_path = sweep_dir / "logs" / f"{name}.log"

    record = {
        "experiment": experiment,
        "task": "smnist",
        GROUP_KEY: num_hidden,
        FRACTION_KEY: train_fraction,
        "seed": seed,
        "epochs": epochs,
        **run_training_cmd(
            cmd,
            log_path,
            "results/smnist/*",
            "smnist",
            "metric",
            dry_run,
            output_dir=run_output_dir,
            use_gpu=use_gpu,
        ),
    }
    if record.get("status") == "ok":
        record = enrich_record(record)
        print(f"  -> best test: {record.get('metric')}", flush=True)
    return record


def _worker_entry(job: tuple) -> tuple[str, int, float, int, dict]:
    (
        experiment,
        num_hidden,
        train_fraction,
        seed,
        sweep_dir_str,
        dry_run,
        epochs,
        analyze_manifold,
        use_gpu,
    ) = job
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
        experiment,
        num_hidden,
        train_fraction,
        seed,
        Path(sweep_dir_str),
        dry_run,
        epochs,
        analyze_manifold,
        use_gpu,
    )
    return experiment, num_hidden, train_fraction, seed, record


def drop_group(results: dict, experiment: str, num_hidden: int, train_fraction: float | None = None) -> None:
    keep = []
    for entry in results.get(experiment, []):
        if int(entry.get(GROUP_KEY, -1)) != num_hidden:
            keep.append(entry)
            continue
        if train_fraction is not None and round(float(entry.get(FRACTION_KEY, 1.0)), 10) != round(train_fraction, 10):
            keep.append(entry)
            continue
    results[experiment] = keep


def drop_group_locked(
    results_path: Path,
    experiment: str,
    num_hidden: int,
    train_fraction: float | None = None,
) -> dict:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        results = load_results(results_path, TASK_KEYS)
        drop_group(results, experiment, num_hidden, train_fraction)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_fields())
        save_summary(results_path, results, GROUP_KEY, TASK_KEYS)
        return results


def rising_hidden_sizes(entries: list[dict]) -> list[int]:
    curves = aggregate_epoch_curves(entries, GROUP_KEY)
    rising = []
    for num_hidden, stats in curves.items():
        if still_rising(stats["mean"]):
            rising.append(int(num_hidden))
    return sorted(rising)


def ci_bounds(row: dict) -> tuple[float, float]:
    return float(row["mean"] - row["ci95"]), float(row["mean"] + row["ci95"])


def analyze_experiment_a(entries: list[dict]) -> dict:
    best_summary = aggregate_scalar_results(entries_best_test(entries), GROUP_KEY)
    test_curves = aggregate_epoch_curves(entries, GROUP_KEY)
    train_curves = aggregate_epoch_curves(entries, GROUP_KEY, curve_key="train_epoch_curve")

    plateau_entries = []
    plateau_by_n: dict[int, int | None] = {}
    rising_by_n: dict[int, bool] = {}
    for num_hidden, stats in test_curves.items():
        n = int(num_hidden)
        plat = epochs_to_plateau(stats["mean"])
        plateau_by_n[n] = plat
        rising_by_n[n] = still_rising(stats["mean"])
        for entry in entries:
            if int(entry.get(GROUP_KEY, -1)) != n:
                continue
            rec = dict(entry)
            rec["epoch_curve"] = None
            rec["metric"] = epochs_to_plateau(entry.get("epoch_curve"))
            if rec["metric"] is not None:
                plateau_entries.append(rec)

    plateau_summary = aggregate_scalar_results(plateau_entries, GROUP_KEY)
    by_n = {int(row[GROUP_KEY]): row for row in best_summary}
    large_rows = [row for n, row in by_n.items() if n >= LARGE_N_MIN]
    ref = by_n.get(REF_N)

    small_plateaued = True
    small_below_large = True
    large_ci_lower = None
    if large_rows:
        large_ci_lower = min(ci_bounds(row)[0] for row in large_rows)
    for n, row in sorted(by_n.items()):
        if n >= LARGE_N_MIN:
            continue
        if rising_by_n.get(n, True) or plateau_by_n.get(n) is None:
            small_plateaued = False
        if large_ci_lower is None or ci_bounds(row)[1] >= large_ci_lower:
            small_below_large = False

    largest_below_128 = None
    if ref is not None:
        ref_lo = ci_bounds(ref)[0]
        for n, row in sorted(by_n.items()):
            if ci_bounds(row)[1] < ref_lo:
                largest_below_128 = n

    any_nonconverged = [n for n, rising in rising_by_n.items() if rising]
    passed = (
        small_plateaued
        and small_below_large
        and not any_nonconverged
        and ref is not None
        and large_ci_lower is not None
    )
    return {
        "pass": passed,
        "best_summary": best_summary,
        "test_curves": test_curves,
        "train_curves": train_curves,
        "plateau_summary": plateau_summary,
        "plateau_by_n": {str(k): v for k, v in sorted(plateau_by_n.items())},
        "rising_by_n": {str(k): v for k, v in sorted(rising_by_n.items())},
        "nonconverged_n": any_nonconverged,
        "small_plateaued": small_plateaued,
        "small_below_large": small_below_large,
        "large_ci_lower": large_ci_lower,
        "largest_n_below_128": largest_below_128,
        "by_n": {str(k): v for k, v in by_n.items()},
    }


def analyze_experiment_b(entries: list[dict]) -> dict:
    by_n: dict[int, list[dict]] = {}
    for entry in entries:
        by_n.setdefault(int(entry[GROUP_KEY]), []).append(entry)

    summaries = {}
    gaps = {}
    for n, group in by_n.items():
        summaries[n] = aggregate_scalar_results(entries_best_test(group), FRACTION_KEY)
        frac1 = [e for e in group if abs(float(e.get(FRACTION_KEY, 1.0)) - 1.0) < 1e-12]
        gap_vals = [g for g in (final_train_test_gap(e) for e in frac1) if g is not None]
        if gap_vals:
            arr = np.asarray(gap_vals, dtype=float)
            n_runs = len(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n_runs > 1 else 0.0
            sem = std / np.sqrt(n_runs) if n_runs > 1 else 0.0
            gaps[n] = {
                "n_runs": n_runs,
                "mean": mean,
                "ci95": 1.96 * sem,
                "values": [float(v) for v in arr],
            }

    row128 = {round(float(r[FRACTION_KEY]), 10): r for r in summaries.get(128, [])}
    row_half = row128.get(0.5)
    row_full = row128.get(1.0)
    gain = None
    flat = False
    if row_half is not None and row_full is not None:
        gain = float(row_full["mean"] - row_half["mean"])
        overlap = ci_bounds(row_full)[0] <= ci_bounds(row_half)[1]
        within_ci = abs(gain) <= float(row_full["ci95"])
        flat = overlap or within_ci
    gap128 = gaps.get(128, {}).get("mean")
    gap_ok = gap128 is not None and gap128 < GAP_PASS_PP
    passed = flat and gap_ok and 128 in summaries and 9 in summaries
    return {
        "pass": passed,
        "summaries": {str(k): v for k, v in summaries.items()},
        "gaps": {str(k): v for k, v in gaps.items()},
        "gain_0.5_to_1.0_n128": gain,
        "data_saturated_n128": flat,
        "gap_n128": gap128,
        "gap_ok": gap_ok,
        "flag_overparam": (not flat) or (gap128 is not None and gap128 >= GAP_PASS_PP),
    }


def analyze_experiment_c(entries: list[dict]) -> dict:
    pr_entries = []
    pca_entries = []
    for entry in entries:
        rec = dict(entry)
        rec["epoch_curve"] = None
        if rec.get("participation_ratio") is not None:
            pr = dict(rec)
            pr["metric"] = float(rec["participation_ratio"])
            pr_entries.append(pr)
        if rec.get("effective_dim_pca") is not None:
            pca = dict(rec)
            pca["metric"] = float(rec["effective_dim_pca"])
            pca_entries.append(pca)
    pr_summary = aggregate_scalar_results(pr_entries, GROUP_KEY)
    pca_summary = aggregate_scalar_results(pca_entries, GROUP_KEY)
    grows = False
    if len(pr_summary) >= 2:
        xs = [row[GROUP_KEY] for row in pr_summary]
        ys = [row["mean"] for row in pr_summary]
        order = np.argsort(np.asarray(xs, dtype=float))
        y_ord = np.asarray(ys, dtype=float)[order]
        grows = bool(y_ord[-1] > y_ord[0] + 1e-6 and np.all(np.diff(y_ord) >= -0.05 * max(abs(y_ord[0]), 1.0)))
        if not grows:
            grows = bool(y_ord[-1] > y_ord[0] * 1.1 or y_ord[-1] > y_ord[0] + 0.5)
    return {
        "pr_summary": pr_summary,
        "pca_summary": pca_summary,
        "grows_with_n": grows,
    }


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def fmt_ci(row: dict | None, digits: int = 2) -> str:
    if row is None:
        return "n/a"
    return f"{row['mean']:.{digits}f} ± {row['ci95']:.{digits}f}"


def paper_sentence(verdict_a: dict | None, verdict_b: dict | None, verdict_c: dict | None) -> str:
    a_pass = bool(verdict_a and verdict_a.get("pass"))
    b_pass = bool(verdict_b and verdict_b.get("pass"))
    grows = bool(verdict_c and verdict_c.get("grows_with_n"))
    if a_pass and b_pass:
        if grows:
            return (
                "The monotonic rise of sMNIST test accuracy with hidden size $n$ survives "
                "training to plateau and a data-saturation control, and the hidden-state "
                "participation ratio continues to grow with $n$, so the trend reflects a "
                "genuine capacity requirement rather than faster optimisation or overparameterisation."
            )
        return (
            "The monotonic rise of sMNIST test accuracy with hidden size $n$ survives "
            "training to plateau and a data-saturation control, indicating a genuine "
            "capacity requirement rather than a convergence or sample-complexity artifact."
        )
    reasons = []
    if not a_pass:
        reasons.append("small-$n$ networks had not plateaued below large-$n$ accuracy")
    if not b_pass:
        reasons.append("large networks remained data- or overparameterisation-limited")
    joined = " and ".join(reasons) if reasons else "the controls did not pass"
    return (
        "We cannot yet treat the monotonic $n$-dependence as a genuine capacity requirement, "
        f"because {joined}."
    )


def relpath_from_report(plot_path: Path) -> str:
    report_dir = REPO_ROOT / "results"
    try:
        return str(plot_path.relative_to(report_dir))
    except ValueError:
        return str(plot_path)


def plot_acc_vs_n(verdict_a: dict, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_scalar_sweep(
        ax,
        verdict_a["best_summary"],
        GROUP_KEY,
        "black",
        r"Converged test accuracy ($\%$)",
        xlabel=r"$N$ (hidden units)",
        ylim=(10.0, 100.0),
        title="Experiment A: best-epoch test accuracy",
    )
    ax.set_xscale("log")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_epoch_curves(verdict_a: dict, plot_path: Path) -> None:
    test_curves = verdict_a["test_curves"]
    train_curves = verdict_a["train_curves"]
    if not test_curves:
        return
    ns = sorted(test_curves.keys(), key=lambda v: float(v))
    colors = [mycmap(i / max(1, len(ns) - 1)) for i in range(len(ns))]
    labels = [rf"$N={n}$" for n in ns]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    plot_epoch_curves_with_stats(
        axes[0],
        {n: test_curves[n] for n in ns},
        colors,
        labels=labels,
        ylabel=r"Test accuracy ($\%$)",
        ylim=(10.0, 100.0),
        title="Experiment A: test curves",
        random_guess=10.0,
        max_epoch=None,
        legend_fontsize=9,
    )
    if train_curves:
        ordered_train = {n: train_curves[n] for n in ns if n in train_curves}
        if ordered_train:
            plot_epoch_curves_with_stats(
                axes[1],
                ordered_train,
                colors,
                labels=labels,
                ylabel=r"Train accuracy ($\%$)",
                xlabel="Epoch",
                ylim=(10.0, 100.0),
                title="Experiment A: train curves",
                random_guess=None,
                max_epoch=None,
                legend_fontsize=9,
            )
        else:
            axes[1].set_ylabel(r"Train accuracy ($\%$)")
            axes[1].set_title("Experiment A: train curves (no data)", loc="left")
    else:
        axes[1].set_ylabel(r"Train accuracy ($\%$)")
        axes[1].set_title("Experiment A: train curves (no data)", loc="left")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_epochs_to_plateau(verdict_a: dict, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    summary = verdict_a["plateau_summary"]
    if summary:
        plot_scalar_sweep(
            ax,
            summary,
            GROUP_KEY,
            "black",
            "Epochs to plateau",
            xlabel=r"$N$ (hidden units)",
            title=r"Experiment A: plateau ($<0.2$ pp / 5 epochs)",
        )
        ax.set_xscale("log")
    mean_curve_pts = [
        (int(n), t)
        for n, t in verdict_a["plateau_by_n"].items()
        if t is not None
    ]
    if mean_curve_pts:
        xs, ys = zip(*sorted(mean_curve_pts))
        ax.plot(xs, ys, "s", color="0.35", markersize=7, label="mean-curve plateau")
        ax.legend(loc="best", frameon=True, fontsize=12)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_train_fraction(verdict_b: dict, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {9: mycmap(0.15), 128: mycmap(0.85)}
    for n in EXP_B_HIDDEN:
        summary = verdict_b["summaries"].get(str(n), [])
        if not summary:
            continue
        xs = [row[FRACTION_KEY] for row in summary]
        means = [row["mean"] for row in summary]
        ci95 = [row["ci95"] for row in summary]
        color = colors[n]
        ax.fill_between(
            xs,
            [m - c for m, c in zip(means, ci95)],
            [m + c for m, c in zip(means, ci95)],
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        ax.plot(xs, means, "o-", color=color, linewidth=2, markersize=5, label=rf"$N={n}$")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel(r"Test accuracy ($\%$)")
    ax.set_ylim(10.0, 100.0)
    ax.set_title("Experiment B: sample complexity", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_effective_dim(verdict_c: dict, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if verdict_c["pr_summary"]:
        plot_scalar_sweep(
            ax,
            verdict_c["pr_summary"],
            GROUP_KEY,
            "black",
            "Participation ratio",
            xlabel=r"$N$ (hidden units)",
            title="Experiment C: effective manifold dimension",
            legend=False,
        )
        ax.set_xscale("log")
    if verdict_c["pca_summary"]:
        xs = [row[GROUP_KEY] for row in verdict_c["pca_summary"]]
        ys = [row["mean"] for row in verdict_c["pca_summary"]]
        ci = [row["ci95"] for row in verdict_c["pca_summary"]]
        ax.errorbar(
            xs,
            ys,
            yerr=ci,
            fmt="s--",
            color="0.45",
            linewidth=1.5,
            markersize=5,
            label="PCA 95% dim",
        )
        ax.legend(loc="upper left", frameon=True, fontsize=12)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def write_report(
    report_path: Path,
    sweep_dir: Path,
    plots: dict[str, Path],
    verdict_a: dict | None,
    verdict_b: dict | None,
    verdict_c: dict | None,
) -> None:
    lines = [
        "# sMNIST scaling: convergence and sample-complexity controls",
        "",
        "Operating point (held fixed): "
        "`--lambda-param 0.0 --omega 0.224 --gamma-real -0.1 --gamma-imag 0.0 "
        "--alpha 0.04 --lr 1e-2 --batch-size 64 --dynamics sl` (tanh input).",
        "",
        f"Runs: `{sweep_dir}`",
        "",
    ]
    if verdict_a is not None:
        status = "PASS" if verdict_a["pass"] else "FAIL"
        lines += [
            "## Experiment A — convergence control",
            "",
            f"**Verdict: {status}** (confound 2 {'ruled out' if verdict_a['pass'] else 'not ruled out'}).",
            "",
            f"- Small-$n$ plateaued: {verdict_a['small_plateaued']}",
            f"- Small-$n$ 95% CI upper bound below large-$n$ ($n\\ge 50$) CI lower bound: {verdict_a['small_below_large']}",
            f"- Non-converged $n$ (mean test still rising over last 10 epochs): {verdict_a['nonconverged_n'] or 'none'}",
            f"- Largest $n$ still statistically below $n=128$: {verdict_a['largest_n_below_128']}",
            "",
        ]
        if "acc_vs_n" in plots:
            lines += [f"![Converged test accuracy vs $n$]({relpath_from_report(plots['acc_vs_n'])})", ""]
        if "epoch_curves" in plots:
            lines += [f"![Per-$n$ train/test curves]({relpath_from_report(plots['epoch_curves'])})", ""]
        if "epochs_to_plateau" in plots:
            lines += [f"![Epochs to plateau vs $n$]({relpath_from_report(plots['epochs_to_plateau'])})", ""]
        lines += ["Best-epoch test accuracy (mean ± 95% CI):", ""]
        for row in verdict_a["best_summary"]:
            lines.append(f"- $n={int(row[GROUP_KEY])}$: {fmt_ci(row)}% ($n_\\mathrm{{runs}}={row['n_runs']}$)")
        lines.append("")
        if not verdict_a["pass"] and verdict_a["nonconverged_n"]:
            lines += [
                f"**Do not report** non-converged $n={verdict_a['nonconverged_n']}$; raise `--epochs` and rerun those sizes.",
                "",
            ]
    if verdict_b is not None:
        status = "PASS" if verdict_b["pass"] else "FAIL"
        lines += [
            "## Experiment B — data / sample-complexity control",
            "",
            f"**Verdict: {status}** (confound 3 {'ruled out' if verdict_b['pass'] else 'not ruled out'}).",
            "",
            f"- $n=128$ data curve flat from 0.5 → 1.0: {verdict_b['data_saturated_n128']} "
            f"(gain {verdict_b['gain_0.5_to_1.0_n128'] if verdict_b['gain_0.5_to_1.0_n128'] is not None else 'n/a'} pp)",
            f"- Train–test gap at fraction=1.0, $n=128$: "
            f"{fmt_ci(verdict_b['gaps'].get('128'))} pp (pass if $<~{GAP_PASS_PP:.0f}$ pp)",
            f"- Train–test gap at fraction=1.0, $n=9$: {fmt_ci(verdict_b['gaps'].get('9'))} pp",
            "",
        ]
        if verdict_b.get("flag_overparam") and not verdict_b["pass"]:
            lines += [
                "**FLAG:** large networks are data/overparameterisation-limited, not capacity-driven "
                "(data curve still rising and/or train–test gap is large).",
                "",
            ]
        if "train_fraction" in plots:
            lines += [f"![Test accuracy vs train fraction]({relpath_from_report(plots['train_fraction'])})", ""]
    if verdict_c is not None:
        lines += [
            "## Experiment C — effective manifold dimension",
            "",
            f"Participation ratio grows with $n$: **{verdict_c['grows_with_n']}**.",
            "",
        ]
        if "effective_dim" in plots:
            lines += [f"![Effective dimension vs $n$]({relpath_from_report(plots['effective_dim'])})", ""]
        if verdict_c["pr_summary"]:
            lines.append("Participation ratio (mean ± 95% CI):")
            lines.append("")
            for row in verdict_c["pr_summary"]:
                lines.append(f"- $n={int(row[GROUP_KEY])}$: {fmt_ci(row, digits=3)}")
            lines.append("")
    lines += [
        "## Paper sentence",
        "",
        paper_sentence(verdict_a, verdict_b, verdict_c),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {report_path}")


def make_plots_and_report(results: dict, sweep_dir: Path, report_path: Path) -> dict:
    plots: dict[str, Path] = {}
    verdict_a = analyze_experiment_a(ok_entries(results, "A")) if ok_entries(results, "A") else None
    verdict_b = analyze_experiment_b(ok_entries(results, "B")) if ok_entries(results, "B") else None
    verdict_c = analyze_experiment_c(ok_entries(results, "A")) if ok_entries(results, "A") else None
    if verdict_c is not None and not verdict_c["pr_summary"] and not verdict_c["pca_summary"]:
        verdict_c = None

    if verdict_a is not None:
        plots["acc_vs_n"] = sweep_dir / "expA_acc_vs_n.png"
        plots["epoch_curves"] = sweep_dir / "expA_epoch_curves.png"
        plots["epochs_to_plateau"] = sweep_dir / "expA_epochs_to_plateau.png"
        plot_acc_vs_n(verdict_a, plots["acc_vs_n"])
        plot_epoch_curves(verdict_a, plots["epoch_curves"])
        plot_epochs_to_plateau(verdict_a, plots["epochs_to_plateau"])
    if verdict_b is not None:
        plots["train_fraction"] = sweep_dir / "expB_acc_vs_train_fraction.png"
        plot_train_fraction(verdict_b, plots["train_fraction"])
    if verdict_c is not None:
        plots["effective_dim"] = sweep_dir / "expC_effective_dim_vs_n.png"
        plot_effective_dim(verdict_c, plots["effective_dim"])

    write_report(report_path, sweep_dir, plots, verdict_a, verdict_b, verdict_c)
    verdicts_path = sweep_dir / "verdicts.json"
    with verdicts_path.open("w") as f:
        json.dump(
            json_ready(
                {
                    "A": verdict_a,
                    "B": verdict_b,
                    "C": verdict_c,
                    "paper_sentence": paper_sentence(verdict_a, verdict_b, verdict_c),
                }
            ),
            f,
            indent=2,
        )
    print(f"Wrote verdicts: {verdicts_path}")
    return {"A": verdict_a, "B": verdict_b, "C": verdict_c}


def run_jobs(
    jobs: list[tuple],
    sweep_dir: Path,
    results_path: Path,
    dry_run: bool,
    workers: int,
    use_gpu: bool,
) -> dict:
    results = load_results(results_path, TASK_KEYS)
    if not jobs:
        return results

    pending = [
        (
            experiment,
            num_hidden,
            train_fraction,
            seed,
            str(sweep_dir),
            dry_run,
            epochs,
            analyze_manifold,
            use_gpu,
        )
        for experiment, num_hidden, train_fraction, seed, epochs, analyze_manifold in jobs
    ]

    def consume(experiment: str, num_hidden: int, seed: int, record: dict, done: int, total: int) -> None:
        nonlocal results
        label = f"{experiment} | N={num_hidden} | seed={seed}"
        if record.get("status") == "ok":
            print(f"Completed {label} -> {record.get('metric')}", flush=True)
        else:
            print(f"Completed {label} -> {record.get('status')}", flush=True)
        results = append_record_locked(results_path, experiment, record)
        print(f"Progress: {done}/{total}", flush=True)

    if workers > 1:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker_entry, job): job for job in pending}
            for fut in as_completed(futures):
                job = futures[fut]
                experiment, num_hidden, train_fraction, seed = job[:4]
                try:
                    experiment, num_hidden, train_fraction, seed, record = fut.result()
                except Exception as exc:
                    print(
                        f"Worker failed {experiment} N={num_hidden} seed={seed}: {exc}",
                        flush=True,
                    )
                    completed += 1
                    continue
                completed += 1
                consume(experiment, num_hidden, seed, record, completed, len(pending))
        return results

    for i, (experiment, num_hidden, train_fraction, seed, epochs, analyze_manifold) in enumerate(jobs, start=1):
        print(
            f"\n{'=' * 60}\n{experiment} | N={num_hidden} | frac={train_fraction} | "
            f"seed={seed} | epochs={epochs}\n{'=' * 60}"
        )
        record = run_single(
            experiment,
            num_hidden,
            train_fraction,
            seed,
            sweep_dir,
            dry_run,
            epochs,
            analyze_manifold,
            use_gpu,
        )
        if record["status"] != "dry_run":
            results = append_record_locked(results_path, experiment, record)
        print(f"Progress: {i}/{len(jobs)}", flush=True)
    return results


def build_a_jobs(
    hidden: list[int],
    seeds: list[int],
    epochs: int,
    done: set[tuple],
    analyze_manifold: bool,
) -> list[tuple]:
    jobs = []
    for num_hidden in hidden:
        for seed in seeds:
            if (num_hidden, 1.0, seed) in done:
                continue
            jobs.append(("A", num_hidden, 1.0, seed, epochs, analyze_manifold))
    return jobs


def build_b_jobs(
    hidden: list[int],
    fractions: list[float],
    seeds: list[int],
    epochs: int,
    done: set[tuple],
) -> list[tuple]:
    jobs = []
    for num_hidden in hidden:
        for fraction in fractions:
            frac_key = round(float(fraction), 10)
            for seed in seeds:
                if (num_hidden, frac_key, seed) in done:
                    continue
                jobs.append(("B", num_hidden, fraction, seed, epochs, False))
    return jobs


def parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if not raw:
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sMNIST scaling convergence / sample-complexity / capacity checks."
    )
    parser.add_argument("--experiments", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--num-hidden-list", type=str, default=None, help="Comma-separated hidden sizes for A")
    parser.add_argument("--train-fractions", type=str, default=None, help="Comma-separated fractions for B")
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=8, help="Seeds for Experiment A (default: 8)")
    parser.add_argument("--num-runs-b", type=int, default=5, help="Seeds for Experiment B (default: 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds-b", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--extend-by", type=int, default=30, help="Extra epochs if a size is still rising")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--refresh-from-runs",
        action="store_true",
        help="Rebuild results.json from runs/*/metrics.json (also done automatically before plotting)",
    )
    parser.add_argument(
        "--no-manifold",
        action="store_true",
        help="Skip Experiment C manifold readout during A runs",
    )
    args = parser.parse_args()
    if args.refresh_from_runs and not args.skip_existing:
        args.plot_only = True

    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else REPO_ROOT / "results/sweeps/scaling_convergence_check"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"
    report_path = REPO_ROOT / "results" / "scaling_convergence_check.md"

    results = load_results(results_path, TASK_KEYS)
    if args.plot_only or args.refresh_from_runs or args.skip_existing:
        results = sync_results_from_runs(sweep_dir, results_path)
    seeds_a = args.seeds if args.seeds is not None else list(range(1, args.num_runs + 1))
    seeds_b = args.seeds_b if args.seeds_b is not None else list(range(1, args.num_runs_b + 1))
    hidden = parse_int_list(args.num_hidden_list, DEFAULT_HIDDEN)
    fractions = parse_float_list(args.train_fractions, TRAIN_FRACTIONS)
    analyze_manifold = "C" in args.experiments and not args.no_manifold

    if not args.plot_only:
        with (sweep_dir / "sweep_config.json").open("w") as f:
            json.dump(
                {
                    "experiments": args.experiments,
                    "hidden": hidden,
                    "train_fractions": fractions,
                    "seeds_a": seeds_a,
                    "seeds_b": seeds_b,
                    "epochs": args.epochs,
                    "extend_by": args.extend_by,
                    "max_epochs": args.max_epochs,
                    "workers": args.workers,
                    "use_gpu": args.use_gpu,
                    "operating_point": OPERATING_POINT,
                },
                f,
                indent=2,
            )

        if args.use_gpu and args.workers > 1:
            print(
                "Warning: --use-gpu with workers>1 shares one GPU across processes; "
                "prefer --workers 1 with a single GPU."
            )

        assigned_hidden = list(hidden)
        if "A" in args.experiments:
            epochs = int(args.epochs)
            while True:
                done = (
                    existing_completions(results, "A", min_epochs=epochs)
                    if args.skip_existing
                    else set()
                )
                jobs = build_a_jobs(hidden, seeds_a, epochs, done, analyze_manifold)
                n_total = len(hidden) * len(seeds_a)
                print(f"Experiment A pending: {len(jobs)} (skipped {n_total - len(jobs)}) at epochs={epochs}")
                results = run_jobs(
                    jobs, sweep_dir, results_path, args.dry_run, max(1, int(args.workers)), args.use_gpu
                )
                if args.dry_run:
                    break
                rising = [
                    n
                    for n in rising_hidden_sizes(ok_entries(results, "A"))
                    if n in assigned_hidden
                ]
                if not rising:
                    print("Experiment A: all n plateaued over the final 10 epochs.")
                    break
                if epochs >= int(args.max_epochs):
                    print(
                        f"Experiment A: still rising at n={rising} after {epochs} epochs; "
                        "not extending further. Do not report those n."
                    )
                    break
                next_epochs = min(int(args.max_epochs), epochs + int(args.extend_by))
                print(
                    f"Experiment A: n={rising} still rising over the last {RISING_WINDOW} epochs; "
                    f"rerunning those sizes with --epochs {next_epochs}"
                )
                for n in rising:
                    results = drop_group_locked(results_path, "A", n)
                hidden = rising
                epochs = next_epochs
                args.skip_existing = False

        if "B" in args.experiments:
            b_hidden = [n for n in assigned_hidden if n in EXP_B_HIDDEN]
            epochs_b = int(args.epochs)
            a_entries = [
                e for e in ok_entries(results, "A") if int(e.get(GROUP_KEY, -1)) in b_hidden
            ]
            if a_entries:
                a_epochs = [int(e.get("epochs", args.epochs)) for e in a_entries]
                if a_epochs:
                    epochs_b = max(a_epochs)
            done_b = (
                existing_completions(results, "B", min_epochs=epochs_b)
                if args.skip_existing
                else set()
            )
            jobs_b = build_b_jobs(b_hidden, fractions, seeds_b, epochs_b, done_b)
            print(f"Experiment B pending: {len(jobs_b)} at epochs={epochs_b}")
            results = run_jobs(
                jobs_b, sweep_dir, results_path, args.dry_run, max(1, int(args.workers)), args.use_gpu
            )

    results = sync_results_from_runs(sweep_dir, results_path)
    for experiment in TASK_KEYS:
        for i, entry in enumerate(results.get(experiment, [])):
            if entry.get("status") == "ok":
                results[experiment][i] = enrich_record(entry)
    if not args.dry_run:
        make_plots_and_report(results, sweep_dir, report_path)
    print(f"\nResults: {results_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
