#!/usr/bin/env python3
"""Sweep nonlinearity ablation variants with multi-seed statistics."""

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
    record_from_sweep_run_dir,
    run_training_cmd,
    save_csv,
    save_results,
    save_summary,
)
from utils.plotting_utils.style import thesis_red, thesis_blue, ifisc_green

GROUP_KEY = "variant"

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
        ylim=(0.0, 1.02),
        num_hidden=50,
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
            "--horizon", "25",
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

def canonical_variant(variant: str) -> str:
    return VARIANT_LEGACY_ALIASES.get(variant, variant)


def safe_variant_name(variant_key: str) -> str:
    return variant_key.replace(" ", "_").replace("/", "")


SAFE_TO_VARIANT = {safe_variant_name(key): key for key in VARIANTS}


def parse_ablation_run_name(run_name: str, task_keys: list[str]) -> tuple[str, str, int] | None:
    for task_key in sorted(task_keys, key=len, reverse=True):
        prefix = f"{task_key}_"
        if not run_name.startswith(prefix):
            continue
        suffix = run_name[len(prefix):]
        if "_seed" not in suffix:
            continue
        variant_part, seed_part = suffix.rsplit("_seed", 1)
        if not seed_part.isdigit():
            continue
        variant_key = SAFE_TO_VARIANT.get(variant_part)
        if variant_key is None:
            continue
        return task_key, variant_key, int(seed_part)
    return None


def sync_task_from_runs(source_dir: Path, task_key: str) -> list[dict]:
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
        _, variant_key, seed = parsed
        record = record_from_sweep_run_dir(
            run_dir,
            task_key,
            group_key=GROUP_KEY,
            variant_key=variant_key,
            seed=seed,
            source_dir=source_dir,
        )
        if record is not None:
            records.append(record)
    return records


def load_task_results_from_logs(source_dir: Path, task_key: str) -> list[dict]:
    return sync_task_from_runs(source_dir, task_key)


def parse_combine_from(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in values:
        if ":" not in item:
            raise ValueError(f"Expected task:path, got {item!r}")
        task_key, raw_path = item.split(":", 1)
        if task_key not in TASKS:
            raise ValueError(f"Unknown task {task_key!r} in --combine-from {item!r}")
        sources[task_key] = Path(raw_path)
    return sources


COMBINE_SOURCES_FILE = "combine_sources.json"


def combine_sources_path(sweep_dir: Path) -> Path:
    return sweep_dir / COMBINE_SOURCES_FILE


def load_combine_sources(sweep_dir: Path) -> dict[str, Path]:
    path = combine_sources_path(sweep_dir)
    if not path.is_file():
        return {}
    with path.open() as f:
        raw = json.load(f)
    return {task_key: Path(source_dir) for task_key, source_dir in raw.items()}


def save_combine_sources(sweep_dir: Path, sources: dict[str, Path]) -> None:
    path = combine_sources_path(sweep_dir)
    merged = {
        task_key: str(source_path)
        for task_key, source_path in load_combine_sources(sweep_dir).items()
    }
    merged.update({task_key: str(source_dir) for task_key, source_dir in sources.items()})
    with path.open("w") as f:
        json.dump(merged, f, indent=2)


def refresh_from_combine_sources(sweep_dir: Path) -> dict[str, list[dict]] | None:
    sources = load_combine_sources(sweep_dir)
    if not sources:
        return None
    existing = load_results(sweep_dir / "results.json", list(TASKS.keys()))
    return combine_from_sources(sources, preserve_tasks=existing)


def combine_from_sources(
    sources: dict[str, Path],
    *,
    preserve_tasks: dict[str, list[dict]] | None = None,
) -> dict[str, list[dict]]:
    combined: dict[str, list[dict]] = {
        task_key: list(preserve_tasks.get(task_key, [])) if preserve_tasks else []
        for task_key in TASKS
    }
    for task_key, source_dir in sources.items():
        entries = load_task_results_from_logs(source_dir, task_key)
        combined[task_key] = entries
        print(f"Combined {len(entries)} entries for {task_key} from {source_dir}/logs + runs (logs/metrics)")
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
) -> list[str]:
    overrides: dict[str, str | int] = {
        "--num-hidden": num_hidden,
        "--seed": seed,
        "--dynamics": variant.dynamics,
    }
    if output_dir is not None:
        overrides["--output-dir"] = str(output_dir)
    if epochs is not None:
        overrides["--epochs"] = epochs
    cmd = cmd_with_overrides(task.base_cmd, overrides)
    if not variant.use_tanh:
        cmd.append("--no-tanh")
    return cmd


def existing_completions(results: dict, task_key: str) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
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
        done.add((variant, int(entry.get("seed", 1))))
    return done


def run_single(
    task_key: str,
    variant_key: str,
    seed: int,
    num_hidden: int,
    sweep_dir: Path,
    dry_run: bool,
    epochs: int | None = None,
) -> dict:
    task = TASKS[task_key]
    variant = VARIANTS[variant_key]
    safe_variant = variant_key.replace(" ", "_").replace("/", "")
    run_name = f"{task_key}_{safe_variant}_seed{seed}"
    run_output_dir = sweep_dir / "runs" / run_name
    cmd = variant_cmd(task, variant, seed, num_hidden, output_dir=run_output_dir, epochs=epochs)
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
            output_dir=run_output_dir,
        ),
    }
    if record.get("status") == "ok":
        print(f"  -> {task.metric_label}: {record['metric']:.4f}")
    return record


def ablation_colors_for(variant_keys: list[str]) -> list:
    variant_order = list(VARIANTS.keys())
    return [ABLATION_COLORS[variant_order.index(k)] for k in variant_keys]


def plot_summary(results: dict, plot_path: Path) -> None:
    results = results_for_plotting(results)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(7, 3.2 * n_tasks), sharex=True)
    if n_tasks == 1:
        axes = [axes]
    variant_order = list(VARIANTS.keys())

    for ax, (task_key, task) in zip(axes, TASKS.items()):
        summary = aggregate_scalar_results(results.get(task_key, []), GROUP_KEY)
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
            title=task.name,
        )

    variant_labels = list(VARIANTS.keys())
    axes[-1].set_xticks(list(range(len(variant_labels))))
    axes[-1].set_xticklabels(variant_labels, rotation=20, ha="right")
    axes[-1].set_xlabel("variant")
    fig.suptitle("Nonlinearity ablation (fixed $N$)", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def plot_epoch_results(results: dict, plot_dir: Path) -> None:
    results = results_for_plotting(results)
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(7, 3.2 * n_tasks), sharex=True)
    if n_tasks == 1:
        axes = [axes]
    variant_order = list(VARIANTS.keys())
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
        )

    if not has_data:
        plt.close(fig)
        return

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Epoch")
    for ax in axes:
        ax.set_xlim(0, 40)
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
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
    )
    parser.add_argument("--num-hidden", type=int, default=None, help="Hidden size (default: 50 per task)")
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for all tasks")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--combine-from",
        nargs="+",
        metavar="TASK:DIR",
        help="Merge task results from sweep dirs (rebuilt from logs/*.log + runs/*/metrics.json)",
    )
    parser.add_argument(
        "--migrate-legacy",
        action="store_true",
        help="One-time: convert old linear-SL DHO entries to LO (only when saving, not with --plot-only)",
    )
    args = parser.parse_args()

    if args.combine_from and not args.sweep_dir:
        parser.error("--combine-from requires --sweep-dir")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else REPO_ROOT / f"results/sweeps/nonlinearity_ablation_sweep_{timestamp}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.json"

    if args.combine_from:
        sources = parse_combine_from(args.combine_from)
        preserve = load_results(results_path, list(TASKS.keys()))
        results = combine_from_sources(sources, preserve_tasks=preserve)
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
            extra_fields=[GROUP_KEY, "num_hidden", "dynamics", "use_tanh", "model", "migrated_from"],
        )
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
    elif args.plot_only and combine_sources_path(sweep_dir).is_file():
        results = refresh_from_combine_sources(sweep_dir)
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
            extra_fields=[GROUP_KEY, "num_hidden", "dynamics", "use_tanh", "model", "migrated_from"],
        )
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
    else:
        results = load_results(results_path, list(TASKS.keys()))
        n_migrated = apply_variant_migrations(results, convert_legacy_dho=args.migrate_legacy)
        if n_migrated and not args.plot_only and args.migrate_legacy:
            print(f"Updated {n_migrated} legacy/mislabeled entries (LO/DHO metadata)")
            save_results(results_path, results)
            save_csv(
                results_path,
                results,
                extra_fields=[GROUP_KEY, "num_hidden", "dynamics", "use_tanh", "model", "migrated_from"],
            )
            save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))
        elif n_migrated and args.plot_only:
            print(f"Applied {n_migrated} in-memory metadata fixes for plotting (results.json not modified)")

    seeds = args.seeds if args.seeds is not None else list(range(1, args.num_runs + 1))

    if not args.plot_only:
        with (sweep_dir / "sweep_config.json").open("w") as f:
            json.dump(
                {
                    "num_runs": args.num_runs,
                    "seeds": seeds,
                    "variants": args.variants,
                    "num_hidden": args.num_hidden,
                    "epochs": args.epochs,
                    "tasks": args.tasks,
                },
                f,
                indent=2,
            )

        for task_key in args.tasks:
            if task_key not in results:
                results[task_key] = []
            done = existing_completions(results, task_key) if args.skip_existing else set()
            num_hidden = args.num_hidden if args.num_hidden is not None else TASKS[task_key].num_hidden

            for variant_key in args.variants:
                for seed in seeds:
                    if any((v, seed) in done for v in variant_completion_keys(variant_key)):
                        print(f"Skipping {task_key} {variant_key} seed={seed} (already completed)")
                        continue

                    task = TASKS[task_key]
                    print(f"\n{'=' * 60}\n{task.name} | {variant_key} | N={num_hidden} | seed={seed}\n{'=' * 60}")
                    record = run_single(task_key, variant_key, seed, num_hidden, sweep_dir, args.dry_run, epochs=args.epochs)
                    if record["status"] != "dry_run":
                        results[task_key] = [
                            e for e in results[task_key]
                            if not (e.get(GROUP_KEY) == variant_key and int(e.get("seed", -1)) == seed)
                        ]
                        results[task_key].append(record)
                        save_results(results_path, results)
                        save_csv(
                            results_path,
                            results,
                            extra_fields=[GROUP_KEY, "num_hidden", "dynamics", "use_tanh", "model", "migrated_from"],
                        )
                        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))

        save_results(results_path, results)
        save_csv(
            results_path,
            results,
            extra_fields=[GROUP_KEY, "num_hidden", "dynamics", "use_tanh", "model", "migrated_from"],
        )
        save_summary(results_path, results, GROUP_KEY, list(TASKS.keys()))

    plot_summary(results, sweep_dir / "nonlinearity_ablation_performance.png")
    plot_epoch_results(results, sweep_dir)
    print(f"\nResults: {results_path}")
    print(f"Summary: {results_path.with_name('summary.json')}")


if __name__ == "__main__":
    main()
