#!/usr/bin/env python3
"""Sweep nonlinearity ablation variants with multi-seed statistics."""

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
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.sweeps.sweep_common import (
    add_horizon_colorbar,
    aggregate_epoch_curves,
    aggregate_scalar_results,
    cmd_with_overrides,
    default_mackey_glass_horizon,
    horizon_color_scale,
    load_results,
    load_sweep_config,
    mackey_glass_horizons,
    multi_task_panels,
    other_tasks_have_data,
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
    task_panel_title,
)
from utils.plotting_utils.style import thesis_red, thesis_blue, ifisc_green

GROUP_KEY = "variant"
HORIZON_KEY = "horizon"
DEFAULT_MG_HORIZONS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

ABLATION_COLORS = [
    thesis_red,
    (0.85, 0.45, 0.5),
    thesis_blue,
    (0.45, 0.65, 0.88),
    (0.6, 0.8, 0.55),
    ifisc_green,
]


@dataclass
class VariantConfig:
    label: str
    dynamics: str
    use_tanh: bool


VARIANTS: dict[str, VariantConfig] = {
    "SL": VariantConfig("SL", "sl", True),
    "SL w/o tanh": VariantConfig("SL w/o tanh", "sl", False),
    "LO": VariantConfig("LO", "lo", True),
    "LO w/o tanh": VariantConfig("LO w/o tanh", "lo", False),
    "DHO": VariantConfig("DHO", "dho", True),
    "DHO w/o tanh": VariantConfig("DHO w/o tanh", "dho", False),
}

VARIANT_LEGACY_ALIASES: dict[str, str] = {}

LEGACY_DHO_TO_LO = {
    "DHO": "LO",
    "DHO w/o tanh": "LO w/o tanh",
}

LO_VARIANTS = ("LO", "LO w/o tanh")
DHO_VARIANTS = ("DHO", "DHO w/o tanh")


def lo_variant_for_dho(variant: str) -> str:
    return LEGACY_DHO_TO_LO[variant]


def is_lo_variant(variant: str) -> bool:
    return variant in LO_VARIANTS


def is_dho_variant(variant: str) -> bool:
    return variant in DHO_VARIANTS


def _remap_dho_entry_to_lo(entry: dict) -> None:
    variant = entry[GROUP_KEY]
    lo_variant = lo_variant_for_dho(variant)
    entry[GROUP_KEY] = lo_variant
    entry["dynamics"] = "lo"
    entry["model"] = "slon"
    entry["migrated_from"] = variant


def migrate_legacy_results(results: dict, *, convert_legacy_dho: bool = False) -> int:
    """Fix metadata in-place. Does not convert standalone DHO to LO unless convert_legacy_dho."""
    updated = 0
    for entries in results.values():
        for entry in entries:
            if entry.get("status") != "ok":
                continue

            variant = entry[GROUP_KEY]
            model = entry.get("model")

            if is_lo_variant(variant) and entry.get("dynamics") == "dho":
                entry["dynamics"] = "lo"
                entry["model"] = "slon"
                entry.setdefault(
                    "migrated_from",
                    "DHO" if variant == "LO" else "DHO w/o tanh",
                )
                updated += 1
                continue

            if convert_legacy_dho and is_dho_variant(variant):
                _remap_dho_entry_to_lo(entry)
                updated += 1
                continue

            if is_dho_variant(variant) and model is None:
                entry["model"] = "horn"
                updated += 1

    return updated


def remap_orphan_dho_to_lo(results: dict) -> int:
    """If a task has DHO runs but no LO runs, treat DHO as legacy LO (pre-HORN naming)."""
    updated = 0
    for entries in results.values():
        ok = [e for e in entries if e.get("status") == "ok"]
        has_lo = any(is_lo_variant(e.get(GROUP_KEY)) for e in ok)
        has_dho = any(is_dho_variant(e.get(GROUP_KEY)) for e in ok)
        if has_lo or not has_dho:
            continue
        for entry in ok:
            if is_dho_variant(entry.get(GROUP_KEY)):
                _remap_dho_entry_to_lo(entry)
                updated += 1
    return updated


def apply_variant_migrations(results: dict, *, convert_legacy_dho: bool = False) -> int:
    n = migrate_legacy_results(results, convert_legacy_dho=convert_legacy_dho)
    if convert_legacy_dho:
        n += remap_orphan_dho_to_lo(results)
    return n


def completion_variant(entry: dict) -> str:
    variant = entry.get(GROUP_KEY)
    if variant in LEGACY_DHO_TO_LO and entry.get("model") == "slon":
        return LEGACY_DHO_TO_LO[variant]
    return variant


@dataclass
class TaskConfig:
    name: str
    output_glob: str
    metric_label: str
    epoch_ylabel: str
    use_r2: bool
    random_guess: float
    ylim: tuple[float, float]
    num_hidden: int
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
        num_hidden=50,
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
        num_hidden=50,
        base_cmd=[
            sys.executable,
            "training/train_smnist.py",
            "--lambda-param", "-0.05",
            "--gamma-real", "-0.1",
            "--gamma-imag", "0.0",
            "--epochs", "10",
            "--early-stop-patience", "10",
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
        ylim=(0.0, 1.02),
        num_hidden=50,
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

def canonical_variant(variant: str) -> str:
    return VARIANT_LEGACY_ALIASES.get(variant, variant)


def safe_variant_name(variant_key: str) -> str:
    return variant_key.replace(" ", "_").replace("/", "")


SAFE_TO_VARIANT = {safe_variant_name(key): key for key in VARIANTS}


def _build_variant_cli_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for key in VARIANTS:
        lookup[key] = key
        lookup[safe_variant_name(key)] = key
        lookup[key.replace(" ", "_")] = key
        lookup[key.replace(" ", "_").replace("/", "_")] = key
    return lookup


VARIANT_CLI_LOOKUP = _build_variant_cli_lookup()


def parse_variant_arg(value: str) -> str:
    resolved = VARIANT_CLI_LOOKUP.get(value)
    if resolved is not None:
        return resolved
    valid = sorted(set(VARIANT_CLI_LOOKUP.keys()))
    raise argparse.ArgumentTypeError(
        f"unknown variant {value!r}; use one of {valid}"
    )


def parse_ablation_run_name(
    run_name: str, task_keys: list[str]
) -> tuple[str, str, int, int | None] | None:
    for task_key in sorted(task_keys, key=len, reverse=True):
        prefix = f"{task_key}_"
        if not run_name.startswith(prefix):
            continue
        suffix = run_name[len(prefix):]
        if "_seed" not in suffix:
            continue
        body, seed_part = suffix.rsplit("_seed", 1)
        if not seed_part.isdigit():
            continue
        horizon: int | None = None
        if "_h" in body:
            variant_part, horizon_part = body.rsplit("_h", 1)
            if horizon_part.isdigit():
                horizon = int(horizon_part)
                body = variant_part
        variant_key = SAFE_TO_VARIANT.get(body)
        if variant_key is None:
            continue
        return task_key, variant_key, int(seed_part), horizon
    return None


def sync_task_from_runs(
    source_dir: Path,
    task_key: str,
    *,
    mg_final_epoch: bool = False,
) -> list[dict]:
    runs_dir = source_dir / "runs"
    if not runs_dir.is_dir():
        return []

    records: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_ablation_run_name(run_dir.name, [task_key])
        if parsed is None or parsed[0] != task_key:
            continue
        _, variant_key, seed, horizon = parsed
        record = record_from_sweep_run_dir(
            run_dir,
            task_key,
            group_key=GROUP_KEY,
            variant_key=variant_key,
            seed=seed,
            source_dir=source_dir,
            mg_final_epoch=mg_final_epoch,
        )
        if record is not None:
            if horizon is not None:
                record[HORIZON_KEY] = horizon
            records.append(record)
    return records


def load_task_results_from_logs(
    source_dir: Path,
    task_key: str,
    *,
    mg_final_epoch: bool = False,
) -> list[dict]:
    return sync_task_from_runs(source_dir, task_key, mg_final_epoch=mg_final_epoch)


@dataclass(frozen=True)
class CombineSource:
    path: Path
    horizon: int | None = None


def parse_combine_from(values: list[str]) -> dict[str, list[CombineSource]]:
    """Parse --combine-from items.

    Formats:
      task:DIR
      task@HORIZON:DIR   (Mackey-Glass only; keeps entries with that horizon)
    Multiple entries for the same task are concatenated (e.g. H=1 and H=25).
    """
    sources: dict[str, list[CombineSource]] = {}
    for item in values:
        if ":" not in item:
            raise ValueError(f"Expected task:path or task@H:path, got {item!r}")
        task_spec, raw_path = item.split(":", 1)
        horizon: int | None = None
        if "@" in task_spec:
            task_key, horizon_part = task_spec.split("@", 1)
            if not horizon_part.isdigit():
                raise ValueError(f"Expected integer horizon in {item!r}")
            horizon = int(horizon_part)
        else:
            task_key = task_spec
        if task_key not in TASKS:
            raise ValueError(f"Unknown task {task_key!r} in --combine-from {item!r}")
        if horizon is not None and task_key != "mackey_glass":
            raise ValueError(f"Horizon filter (@H) only supported for mackey_glass, got {item!r}")
        sources.setdefault(task_key, []).append(CombineSource(Path(raw_path), horizon))
    return sources


def horizons_from_combine_sources(sources: dict[str, list[CombineSource]]) -> list[int] | None:
    hs = sorted(
        {
            src.horizon
            for src_list in sources.values()
            for src in src_list
            if src.horizon is not None
        }
    )
    return hs or None


COMBINE_SOURCES_FILE = "combine_sources.json"


def combine_sources_path(sweep_dir: Path) -> Path:
    return sweep_dir / COMBINE_SOURCES_FILE


def _serialize_combine_sources(sources: dict[str, list[CombineSource]]) -> dict:
    out: dict = {}
    for task_key, src_list in sources.items():
        if len(src_list) == 1 and src_list[0].horizon is None:
            out[task_key] = str(src_list[0].path)
            continue
        out[task_key] = [
            (
                {"path": str(src.path), "horizon": src.horizon}
                if src.horizon is not None
                else {"path": str(src.path)}
            )
            for src in src_list
        ]
    return out


def _deserialize_combine_sources(raw: dict) -> dict[str, list[CombineSource]]:
    sources: dict[str, list[CombineSource]] = {}
    for task_key, value in raw.items():
        if task_key not in TASKS:
            continue
        if isinstance(value, str):
            sources[task_key] = [CombineSource(Path(value))]
        elif isinstance(value, list):
            parsed: list[CombineSource] = []
            for item in value:
                if isinstance(item, str):
                    parsed.append(CombineSource(Path(item)))
                elif isinstance(item, dict) and "path" in item:
                    h = item.get("horizon")
                    parsed.append(
                        CombineSource(Path(item["path"]), int(h) if h is not None else None)
                    )
            sources[task_key] = parsed
        elif isinstance(value, dict) and "path" in value:
            h = value.get("horizon")
            sources[task_key] = [
                CombineSource(Path(value["path"]), int(h) if h is not None else None)
            ]
    return sources


def load_combine_sources(sweep_dir: Path) -> dict[str, list[CombineSource]]:
    path = combine_sources_path(sweep_dir)
    if not path.is_file():
        return {}
    with path.open() as f:
        raw = json.load(f)
    return _deserialize_combine_sources(raw)


def save_combine_sources(sweep_dir: Path, sources: dict[str, list[CombineSource]]) -> None:
    path = combine_sources_path(sweep_dir)
    merged = _serialize_combine_sources(load_combine_sources(sweep_dir))
    merged.update(_serialize_combine_sources(sources))
    with path.open("w") as f:
        json.dump(merged, f, indent=2)


def refresh_from_combine_sources(
    sweep_dir: Path,
    *,
    mg_final_epoch: bool = False,
) -> dict[str, list[dict]] | None:
    sources = load_combine_sources(sweep_dir)
    if not sources:
        return None
    existing = load_results(sweep_dir / "results.json", list(TASKS.keys()))
    return combine_from_sources(
        sources,
        preserve_tasks=existing,
        mg_final_epoch=mg_final_epoch,
    )


def refresh_from_runs(
    sweep_dir: Path,
    task_keys: list[str],
    *,
    mg_final_epoch: bool = False,
) -> dict[str, list[dict]]:
    results = {task_key: [] for task_key in TASKS}
    for task_key in task_keys:
        entries = load_task_results_from_logs(
            sweep_dir,
            task_key,
            mg_final_epoch=mg_final_epoch,
        )
        results[task_key] = entries
        print(
            f"Refreshed {len(entries)} entries for {task_key} from "
            f"{sweep_dir}/runs/*/metrics.json"
        )
    return results


def _filter_entries_by_horizon(entries: list[dict], horizon: int | None) -> list[dict]:
    if horizon is None:
        return entries
    return [e for e in entries if e.get(HORIZON_KEY) == horizon]


def combine_from_sources(
    sources: dict[str, list[CombineSource]],
    *,
    preserve_tasks: dict[str, list[dict]] | None = None,
    mg_final_epoch: bool = False,
) -> dict[str, list[dict]]:
    combined: dict[str, list[dict]] = {
        task_key: list(preserve_tasks.get(task_key, [])) if preserve_tasks else []
        for task_key in TASKS
    }
    for task_key, src_list in sources.items():
        merged: list[dict] = []
        for src in src_list:
            entries = load_task_results_from_logs(
                src.path,
                task_key,
                mg_final_epoch=mg_final_epoch,
            )
            before = len(entries)
            entries = _filter_entries_by_horizon(entries, src.horizon)
            merged.extend(entries)
            filt = f" (horizon={src.horizon})" if src.horizon is not None else ""
            print(
                f"Combined {len(entries)}/{before} entries for {task_key}{filt} "
                f"from {src.path}/runs/*/metrics.json"
            )
        combined[task_key] = merged
    return combined


def results_for_plotting(results: dict) -> dict:
    remapped: dict[str, list[dict]] = {}
    for task_key, entries in results.items():
        remapped[task_key] = []
        for entry in entries:
            record = dict(entry)
            if GROUP_KEY in record:
                record[GROUP_KEY] = canonical_variant(record[GROUP_KEY])
            remapped[task_key].append(record)
    return remapped


def variant_completion_keys(variant_key: str) -> set[str]:
    keys = {variant_key}
    for legacy, canonical in VARIANT_LEGACY_ALIASES.items():
        if canonical == variant_key:
            keys.add(legacy)
    return keys


def variant_cmd(
    task: TaskConfig,
    variant: VariantConfig,
    seed: int,
    num_hidden: int,
    output_dir: Path | str | None = None,
    epochs: int | None = None,
    horizon: int | None = None,
    use_gpu: bool = False,
) -> list[str]:
    overrides: dict[str, str | int] = {
        "--num-hidden": num_hidden,
        "--seed": seed,
        "--dynamics": variant.dynamics,
    }
    if use_gpu:
        overrides["--device"] = "cuda"
    if output_dir is not None:
        overrides["--output-dir"] = str(output_dir)
    if epochs is not None:
        overrides["--epochs"] = epochs
        # Always disable early stopping for the full ablation schedule.
        overrides["--early-stop-patience"] = epochs
    if horizon is not None:
        overrides["--horizon"] = horizon
    cmd = cmd_with_overrides(task.base_cmd, overrides)
    if not variant.use_tanh:
        cmd.append("--no-tanh")
    return cmd


def existing_completions(results: dict, task_key: str) -> set[tuple[str, int, int | None]]:
    done: set[tuple[str, int, int | None]] = set()
    for entry in results.get(task_key, []):
        if entry.get("status") != "ok":
            continue
        variant = entry.get(GROUP_KEY)
        expected = VARIANTS.get(variant)
        if expected is not None:
            if entry.get("dynamics") != expected.dynamics:
                continue
            if entry.get("use_tanh") != expected.use_tanh:
                continue
        done.add(
            (
                completion_variant(entry),
                int(entry.get("seed", 1)),
                entry.get(HORIZON_KEY),
            )
        )
    return done


def extra_result_fields() -> list[str]:
    return [GROUP_KEY, "num_hidden", "dynamics", "use_tanh", "model", "migrated_from", HORIZON_KEY]


def merge_record(
    results: dict,
    task_key: str,
    variant_key: str,
    seed: int,
    horizon: int | None,
    record: dict,
) -> None:
    results.setdefault(task_key, [])
    results[task_key] = [
        e
        for e in results[task_key]
        if not (
            e.get(GROUP_KEY) == variant_key
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
    variant_key: str,
    seed: int,
    horizon: int | None,
    record: dict,
) -> dict:
    lock_path = results_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        results = load_results(results_path, list(TASKS.keys()))
        merge_record(results, task_key, variant_key, seed, horizon, record)
        save_results(results_path, results)
        save_csv(results_path, results, extra_fields=extra_result_fields())
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
        return results


def run_single(
    task_key: str,
    variant_key: str,
    seed: int,
    num_hidden: int,
    sweep_dir: Path,
    dry_run: bool,
    epochs: int | None = None,
    horizon: int | None = None,
    use_gpu: bool = False,
) -> dict:
    task = TASKS[task_key]
    variant = VARIANTS[variant_key]
    safe_variant = safe_variant_name(variant_key)
    if horizon is not None:
        run_name = f"{task_key}_{safe_variant}_h{horizon}_seed{seed}"
    else:
        run_name = f"{task_key}_{safe_variant}_seed{seed}"
    run_output_dir = sweep_dir / "runs" / run_name
    cmd = variant_cmd(
        task, variant, seed, num_hidden,
        output_dir=run_output_dir, epochs=epochs, horizon=horizon, use_gpu=use_gpu,
    )
    log_path = sweep_dir / "logs" / f"{run_name}.log"

    record = {
        "task": task_key,
        GROUP_KEY: variant_key,
        "dynamics": variant.dynamics,
        "model": "horn" if variant.dynamics == "dho" else "slon",
        "use_tanh": variant.use_tanh,
        "num_hidden": num_hidden,
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


def _worker_entry(job: tuple) -> tuple[str, str, int, int | None, dict]:
    (
        task_key,
        variant_key,
        seed,
        num_hidden,
        horizon,
        sweep_dir_str,
        dry_run,
        epochs,
        use_gpu,
    ) = job
    # Stagger torch/CUDA init across workers to avoid NFS/driver lock hangs.
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
        variant_key,
        seed,
        num_hidden,
        Path(sweep_dir_str),
        dry_run,
        epochs=epochs,
        horizon=horizon,
        use_gpu=use_gpu,
    )
    return task_key, variant_key, seed, horizon, record


def horizons_for_task(task_key: str, horizons: list[int] | None) -> list[int | None]:
    if task_key == "mackey_glass" and horizons is not None:
        return list(horizons)
    return [None]


def report_ablation_progress(
    results: dict,
    tasks: list[str],
    variants: list[str],
    seeds: list[int],
    horizons: list[int] | None,
) -> None:
    rows: list[tuple[str, int, int]] = []
    for task_key in tasks:
        hs = horizons_for_task(task_key, horizons)
        per_cat = len(seeds) * len(hs)
        done = existing_completions(results, task_key)
        observed = {v for v, _s, _h in done}
        ordered = [v for v in VARIANTS if v in variants or v in observed]
        for v in variants:
            if v not in VARIANTS and v not in ordered:
                ordered.append(v)
        prefix = f"{task_key} " if len(tasks) > 1 else ""
        if task_key == "mackey_glass":
            named = [h for h in hs if h is not None]
            if len(named) == 1:
                prefix = f"{prefix}H={named[0]} "
            elif not named:
                h = default_mackey_glass_horizon(TASKS)
                if h is not None:
                    prefix = f"{prefix}H={h} "
        for v in ordered:
            n_done = sum(1 for vv, _s, _h in done if vv == v)
            rows.append((f"{prefix}{v}", n_done, per_cat))
    print_finished_remaining(rows)


def build_jobs(
    tasks: list[str],
    variants: list[str],
    seeds: list[int],
    horizons: list[int] | None,
    num_hidden_by_task: dict[str, int],
    done_by_task: dict[str, set[tuple[str, int, int | None]]],
) -> list[tuple]:
    jobs = []
    for task_key in tasks:
        num_hidden = num_hidden_by_task[task_key]
        done = done_by_task.get(task_key, set())
        for variant_key in variants:
            for horizon in horizons_for_task(task_key, horizons):
                for seed in seeds:
                    if any(
                        (v, seed, horizon) in done
                        for v in variant_completion_keys(variant_key)
                    ):
                        continue
                    jobs.append(
                        (
                            task_key,
                            variant_key,
                            seed,
                            num_hidden,
                            horizon,
                        )
                    )
    return jobs


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
        (
            task_key,
            variant_key,
            seed,
            num_hidden,
            horizon,
            sweep_dir_str,
            dry_run,
            epochs,
            use_gpu,
        )
        for task_key, variant_key, seed, num_hidden, horizon in jobs
    ]

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_entry, job): job for job in pending}
        for fut in as_completed(futures):
            job = futures[fut]
            task_key, variant_key, seed, num_hidden, horizon = job[:5]
            try:
                task_key, variant_key, seed, horizon, record = fut.result()
            except Exception as exc:
                print(
                    f"Worker failed {task_key} {variant_key} seed={seed} "
                    f"horizon={horizon}: {exc}",
                    flush=True,
                )
                completed += 1
                continue

            task = TASKS[task_key]
            label = f"{task.name} | {variant_key}"
            if horizon is not None:
                label += f" | h={horizon}"
            label += f" | seed={seed}"
            if record.get("status") == "ok":
                print(f"Completed {label} -> {record['metric']:.4f}", flush=True)
            else:
                print(f"Completed {label} -> {record.get('status')}", flush=True)

            results = append_record_locked(
                results_path, task_key, variant_key, seed, horizon, record
            )
            completed += 1
            print(f"Progress: {completed}/{len(pending)}", flush=True)
            update_live_plots(results, sweep_dir)

    return results


def update_live_plots(
    results: dict,
    sweep_dir: Path,
    max_epoch: int | None = None,
    horizons: list[int] | None = None,
) -> None:
    plot_summary(
        results,
        sweep_dir / "nonlinearity_ablation_performance.png",
        horizons=horizons,
    )
    if mackey_glass_horizons(results) and not other_tasks_have_data(results, TASKS):
        return
    plot_epoch_results(
        results, sweep_dir, max_epoch=max_epoch, horizons=horizons
    )


def ablation_colors_for(variant_keys: list[str]) -> list:
    variant_order = list(VARIANTS.keys())
    return [ABLATION_COLORS[variant_order.index(k)] for k in variant_keys]


def plot_horizon_ablation_performance(
    results: dict, plot_path: Path, horizons: list[int] | None = None
) -> None:
    results = results_for_plotting(results)
    task_key = "mackey_glass"
    task = TASKS[task_key]
    entries = [e for e in results.get(task_key, []) if e.get("status") == "ok"]
    plot_hs = select_plot_horizons(mackey_glass_horizons(results), horizons)
    if not plot_hs:
        return

    variant_order = list(VARIANTS.keys())
    xs = list(range(len(variant_order)))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    cmap, norm = horizon_color_scale(
        horizons or plot_hs,
        None if horizons is not None else plot_path.parent,
    )

    for horizon in plot_hs:
        color = cmap(norm(horizon))
        subset = [e for e in entries if e.get(HORIZON_KEY) == horizon]
        summary = aggregate_scalar_results(subset, GROUP_KEY)
        by_variant = {row[GROUP_KEY]: row for row in summary}
        line_xs = []
        means = []
        lower_ci = []
        upper_ci = []
        for variant_key in variant_order:
            row = by_variant.get(variant_key)
            if row is None:
                continue
            line_xs.append(variant_order.index(variant_key))
            mean = row["mean"]
            ci95 = row["ci95"]
            means.append(mean)
            lower_ci.append(mean - ci95)
            upper_ci.append(mean + ci95)

        if not means:
            continue
        ax.fill_between(line_xs, lower_ci, upper_ci, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            line_xs,
            means,
            "o-",
            color=color,
            linewidth=1.5,
            markersize=4,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(variant_order, rotation=20, ha="right")
    ax.set_xlabel("Variant")
    ax.set_ylabel(task.metric_label)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(task.name, loc="left")
    ax.grid(True, alpha=0.3)
    add_horizon_colorbar(fig, ax, cmap, norm)
    fig.suptitle("Nonlinearity ablation (fixed $N$)", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_summary(
    results: dict, plot_path: Path, horizons: list[int] | None = None
) -> None:
    results = results_for_plotting(results)
    if mackey_glass_horizons(results) and not other_tasks_have_data(results, TASKS):
        plot_horizon_ablation_performance(results, plot_path, horizons=horizons)
        return

    panels = multi_task_panels(results, TASKS, horizons=horizons)
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    variant_order = list(VARIANTS.keys())

    for ax, (_task_key, task, title, entries) in zip(axes, panels):
        summary = aggregate_scalar_results(entries, GROUP_KEY)
        summary = sorted(summary, key=lambda row: variant_order.index(row[GROUP_KEY]))
        relabeled = [
            {**row, GROUP_KEY: variant_order.index(row[GROUP_KEY])}
            for row in summary
        ]
        plot_scalar_sweep(
            ax,
            relabeled,
            GROUP_KEY,
            "black",
            task.metric_label,
            ylim=task.ylim,
            title=title,
        )

    variant_labels = list(VARIANTS.keys())
    axes[-1].set_xticks(list(range(len(variant_labels))))
    axes[-1].set_xticklabels(variant_labels, rotation=20, ha="right")
    axes[-1].set_xlabel("Variant")
    fig.suptitle("Nonlinearity ablation (fixed $N$)", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def add_zoom_inset(ax, curves: dict, colors: list, tail_fraction: float = 0.2) -> None:
    """Inset over the last epochs, where the variants are too close to tell apart at full scale."""
    n_epochs = max(len(stats["mean"]) for stats in curves.values())
    start = int(n_epochs * (1 - tail_fraction))
    if n_epochs - start < 2:
        return
    # lower right; the IMDb legend goes to the lower left
    inset = ax.inset_axes([0.67, 0.1, 0.3, 0.38], zorder=5)
    inset.set_facecolor("white")
    lo, hi = np.inf, -np.inf
    for (label, stats), color in zip(curves.items(), colors):
        epochs, mean, ci = stats["epochs"][start:], stats["mean"][start:], stats["ci95"][start:]
        inset.plot(epochs, mean, color=color, linewidth=1.2)
        inset.fill_between(epochs, mean - ci, mean + ci, color=color, alpha=0.2, linewidth=0)
        lo, hi = min(lo, np.nanmin(mean - ci)), max(hi, np.nanmax(mean + ci))
    if np.isfinite(lo) and np.isfinite(hi):
        pad = 0.1 * max(hi - lo, 1e-3)
        inset.set_ylim(lo - pad, hi + pad)
    inset.set_xlim(start, n_epochs - 1)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)


def plot_epoch_results(
    results: dict,
    plot_dir: Path,
    max_epoch: int | None = None,
    horizons: list[int] | None = None,
) -> None:
    results = results_for_plotting(results)
    plot_dir.mkdir(parents=True, exist_ok=True)
    panels = multi_task_panels(results, TASKS, horizons=horizons)
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    variant_order = list(VARIANTS.keys())
    has_data = False

    for ax, (_task_key, task, title, entries) in zip(axes, panels):
        ax.set_title(title, loc="left")
        ax.set_ylabel(task.epoch_ylabel)
        ax.set_ylim(*task.ylim)
        ax.grid(True, alpha=0.3)

        curves = aggregate_epoch_curves(entries, GROUP_KEY)
        if not curves:
            continue

        has_data = True
        ordered = {k: curves[k] for k in variant_order if k in curves}
        colors = ablation_colors_for(list(ordered.keys()))
        labels = list(ordered.keys())

        plot_epoch_curves_with_stats(
            ax,
            ordered,
            colors,
            labels=labels,
            ylim=task.ylim,
            random_guess=task.random_guess,
            max_epoch=max_epoch,
            legend_loc="lower left" if _task_key == "imdb" else "lower right",
        )
        if _task_key == "imdb":
            add_zoom_inset(ax, ordered, colors)

    if not has_data:
        plt.close(fig)
        return

    if max_epoch is None:
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
    fig.suptitle("Nonlinearity ablation: test performance vs epoch", y=1.02)
    fig.tight_layout()
    out = plot_dir / "ablation_epochs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed nonlinearity ablation sweep.")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()), default=list(TASKS.keys()))
    parser.add_argument(
        "--variants",
        nargs="+",
        type=parse_variant_arg,
        default=list(VARIANTS.keys()),
        metavar="VARIANT",
        help=(
            "Variant name or safe alias (e.g. SL_w_o_tanh for 'SL w/o tanh'; "
            "needed when the cluster run script splits quoted names on spaces)"
        ),
    )
    parser.add_argument("--num-hidden", type=int, default=None, help="Hidden size (default: 50 per task)")
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
        "--max-epoch",
        type=int,
        default=None,
        help="Cap epoch curves / x-axis at this epoch (e.g. 200)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Override y-axis limits for the selected --tasks (e.g. 80 100)",
    )
    parser.add_argument(
        "--refresh-from-runs",
        action="store_true",
        help="Rebuild results from runs/*/metrics.json before plotting",
    )
    parser.add_argument(
        "--final-epoch",
        action="store_true",
        help="Use the last training epoch for Mackey-Glass metrics (default: best val epoch)",
    )
    parser.add_argument(
        "--combine-from",
        nargs="+",
        metavar="TASK:DIR",
        help=(
            "Merge task results from sweep dirs (rebuilt from runs/*/metrics.json). "
            "Use task@H:DIR for Mackey-Glass to keep only horizon H "
            "(repeatable, e.g. mackey_glass@1:dir_a mackey_glass@25:dir_b)."
        ),
    )
    parser.add_argument(
        "--migrate-legacy",
        action="store_true",
        help="One-time: convert old linear-SL DHO entries to LO (only when saving, not with --plot-only)",
    )
    args = parser.parse_args()

    if args.combine_from and not args.sweep_dir:
        parser.error("--combine-from requires --sweep-dir")
    if args.ylim is not None:
        ymin, ymax = args.ylim
        for task_key in args.tasks:
            TASKS[task_key].ylim = (ymin, ymax)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else REPO_ROOT / f"results/sweeps/nonlinearity_ablation_sweep_{timestamp}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"

    combine_horizon_hint: list[int] | None = None
    if args.combine_from:
        sources = parse_combine_from(args.combine_from)
        combine_horizon_hint = horizons_from_combine_sources(sources)
        preserve = load_results(results_path, list(TASKS.keys()))
        results = combine_from_sources(
            sources,
            preserve_tasks=preserve,
            mg_final_epoch=args.final_epoch,
        )
        save_combine_sources(sweep_dir, sources)
        for task_key in TASKS:
            results.setdefault(task_key, [])
        n_migrated = apply_variant_migrations(results, convert_legacy_dho=args.migrate_legacy)
        if n_migrated:
            print(f"Updated {n_migrated} legacy/mislabeled entries (LO/DHO metadata)")
        save_results(results_path, results)
        save_csv(
            results_path,
            results,
            extra_fields=extra_result_fields(),
        )
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
    elif args.plot_only and args.refresh_from_runs:
        results = refresh_from_runs(
            sweep_dir,
            args.tasks,
            mg_final_epoch=args.final_epoch,
        )
        n_migrated = apply_variant_migrations(results, convert_legacy_dho=args.migrate_legacy)
        if n_migrated:
            print(f"Applied {n_migrated} in-memory metadata fixes for plotting")
        save_results(results_path, results)
        save_csv(
            results_path,
            results,
            extra_fields=extra_result_fields(),
        )
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
    elif args.plot_only and combine_sources_path(sweep_dir).is_file():
        combine_horizon_hint = horizons_from_combine_sources(load_combine_sources(sweep_dir))
        results = refresh_from_combine_sources(sweep_dir, mg_final_epoch=args.final_epoch)
        if results is None:
            results = load_results(results_path, list(TASKS.keys()))
        else:
            print("Refreshed results from combine_sources.json")
        n_migrated = apply_variant_migrations(results, convert_legacy_dho=args.migrate_legacy)
        if n_migrated:
            print(f"Applied {n_migrated} in-memory metadata fixes for plotting")
        save_results(results_path, results)
        save_csv(
            results_path,
            results,
            extra_fields=extra_result_fields(),
        )
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
    else:
        if combine_sources_path(sweep_dir).is_file():
            combine_horizon_hint = horizons_from_combine_sources(load_combine_sources(sweep_dir))
        results = load_results(results_path, list(TASKS.keys()))
        n_migrated = apply_variant_migrations(results, convert_legacy_dho=args.migrate_legacy)
        if n_migrated and not args.plot_only and args.migrate_legacy:
            print(f"Updated {n_migrated} legacy/mislabeled entries (LO/DHO metadata)")
            save_results(results_path, results)
            save_csv(
                results_path,
                results,
                extra_fields=extra_result_fields(),
            )
            save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
        elif n_migrated and args.plot_only:
            print(f"Applied {n_migrated} in-memory metadata fixes for plotting (results.json not modified)")

    cfg = load_sweep_config(sweep_dir) if args.plot_only else {}
    if args.seeds is not None:
        seeds = args.seeds
    elif args.plot_only and cfg.get("seeds"):
        seeds = [int(s) for s in cfg["seeds"]]
    else:
        seeds = list(range(1, args.num_runs + 1))

    horizons = args.horizons
    if horizons is None and combine_horizon_hint:
        horizons = combine_horizon_hint
    elif args.plot_only and horizons is None:
        horizons = resolve_plot_horizons(None, cfg, results, args.tasks, HORIZON_KEY)
    if args.horizons is not None and "mackey_glass" not in args.tasks:
        parser.error("--horizons requires mackey_glass in --tasks")

    variants = list(args.variants)
    if args.plot_only:
        cfg_vs = [str(v) for v in (cfg.get("variants") or [])]
        observed = {
            completion_variant(e)
            for task_key in args.tasks
            for e in results.get(task_key, [])
            if e.get("status") == "ok"
        }
        extra = [v for v in cfg_vs + sorted(v for v in observed if v) if v not in variants]
        variants.extend(extra)
        report_ablation_progress(results, args.tasks, variants, seeds, horizons)

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
                    "variants": args.variants,
                    "num_hidden": args.num_hidden,
                    "epochs": args.epochs,
                    "tasks": args.tasks,
                    "horizons": merged_horizons,
                    "workers": args.workers,
                    "use_gpu": args.use_gpu,
                },
                f,
                indent=2,
            )

        num_hidden_by_task = {
            task_key: args.num_hidden if args.num_hidden is not None else TASKS[task_key].num_hidden
            for task_key in args.tasks
        }
        done_by_task = {
            task_key: existing_completions(results, task_key) if args.skip_existing else set()
            for task_key in args.tasks
        }
        jobs = build_jobs(
            args.tasks,
            args.variants,
            seeds,
            horizons,
            num_hidden_by_task,
            done_by_task,
        )
        n_skipped = sum(
            len(args.variants)
            * len(seeds)
            * len(horizons_for_task(task_key, horizons))
            for task_key in args.tasks
        ) - len(jobs)
        if n_skipped:
            print(f"Skipping {n_skipped} already-completed jobs")
        print(f"Pending jobs: {len(jobs)}")

        workers = max(1, int(args.workers))
        if workers > 1 and jobs:
            results = run_jobs_parallel(
                jobs, sweep_dir, results_path, args.dry_run, args.epochs, workers, args.use_gpu
            )
        else:
            for task_key, variant_key, seed, num_hidden, horizon in jobs:
                task = TASKS[task_key]
                label = f"{task.name} | {variant_key} | N={num_hidden} | seed={seed}"
                if horizon is not None:
                    label += f" | h={horizon}"
                print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
                record = run_single(
                    task_key,
                    variant_key,
                    seed,
                    num_hidden,
                    sweep_dir,
                    args.dry_run,
                    epochs=args.epochs,
                    horizon=horizon,
                    use_gpu=args.use_gpu,
                )
                if record["status"] != "dry_run":
                    merge_record(results, task_key, variant_key, seed, horizon, record)
                    save_results_locked(results_path, results)
                    update_live_plots(
                        results, sweep_dir, max_epoch=args.max_epoch, horizons=args.horizons
                    )

        save_results_locked(results_path, results)

    update_live_plots(
        results, sweep_dir, max_epoch=args.max_epoch, horizons=args.horizons
    )
    print(f"\nResults: {results_path}")
    print(f"Summary: {results_path.with_name('summary.json')}")


if __name__ == "__main__":
    main()
