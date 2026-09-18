"""Compute Jaeger short-term memory capacity for trained (or init) SLON/HORN models.

Does not modify training. Loads a frozen checkpoint (or builds a fresh model),
drives it with noise and/or task-like probes, fits a fresh linear readout from
hidden states to delayed probe targets, and saves JSON + plots.

Examples:
  python analysis/memory_capacity.py --task smnist \\
      --run-dir results/smnist/<run> --probes noise,task

  python analysis/memory_capacity.py --task mackey_glass \\
      --run-dir results/mackey_glass/<run> --probes noise

  python analysis/memory_capacity.py --task smnist --init-only \\
      --num-hidden 50 --lambda-param -0.04 --probes noise
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.memory_capacity import (
    drive_memory_capacity,
    effective_memory_horizon,
    make_noise_drive,
    oscillator_core,
)
from utils.model_factory import build_oscillator
from utils.plotting_utils.style import thesis_blue, thesis_red, ifisc_green
from utils.run_dirs import atomic_write_json

try:
    from utils.plotting_utils.style import apply_style
    apply_style()
except Exception:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})

PROBE_COLORS = {
    "noise": thesis_blue,
    "task": thesis_red,
    "init_noise": ifisc_green,
}


# ---------------------------------------------------------------------------
# Run-dir / checkpoint helpers
# ---------------------------------------------------------------------------

def parse_run_log(log_path: str) -> dict:
    """Parse key: value lines from a training log into a flat dict."""
    params: dict = {}
    with open(log_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("=") or ":" not in line:
                continue
            if line.startswith("epoch ") or line.startswith("Per-class"):
                break
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in params and key in ("omega", "lambda"):
                # Prefer the resolved values written after the argparse dump.
                params[key] = _parse_scalar(value)
                continue
            if key not in params:
                params[key] = _parse_scalar(value)
    return params


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def resolve_checkpoint(run_dir: str | None, checkpoint: str | None) -> str | None:
    if checkpoint is not None:
        path = checkpoint if os.path.isabs(checkpoint) else os.path.join(PROJECT_ROOT, checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if run_dir is None:
        return None
    for name in ("best_model.pt", "last_model.pt"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No best_model.pt / last_model.pt in {run_dir}")


def infer_shapes_from_state(state: dict) -> dict:
    """Infer architecture sizes from a state_dict (SLON or SLONWithEmbedding)."""
    info = {}
    if "embedding.weight" in state:
        info["vocab_size"], info["embed_dim"] = state["embedding.weight"].shape
        prefix = "slon."
    else:
        prefix = ""

    i2h = state[f"{prefix}i2h.weight"]
    h2o = state[f"{prefix}h2o.weight"]
    info["num_input"] = int(i2h.shape[1])
    info["num_nodes"] = int(i2h.shape[0])
    info["num_output"] = int(h2o.shape[0])

    # SL uses h2h: (N, 2N); HORN uses h2h: (N, N)
    h2h = state[f"{prefix}h2h.weight"]
    info["is_dho"] = h2h.shape[1] == info["num_nodes"]
    return info


def merge_model_kwargs(args, log_params: dict | None) -> dict:
    src = log_params or {}

    def pick(cli_name, *log_keys, default=None):
        cli_val = getattr(args, cli_name, None)
        # CLI defaults are always set; prefer log when --run-dir is used unless
        # the user explicitly overrode via a non-default... We keep it simple:
        # run-dir / log wins when present, else CLI.
        for key in log_keys:
            if key in src and src[key] is not None:
                return src[key]
        if cli_val is not None:
            return cli_val
        return default

    dynamics = pick("dynamics", "dynamics", default="sl")

    use_tanh = pick("use_tanh", "use_tanh", default=True)
    if "no_tanh" in src:
        use_tanh = not bool(src["no_tanh"])
    if getattr(args, "no_tanh", False):
        use_tanh = False

    return {
        "dynamics": str(dynamics),
        "num_hidden": int(pick("num_hidden", "num_hidden", default=50)),
        "h": float(pick("h", "h", default=1.0)),
        "alpha": float(pick("alpha", "alpha", default=0.5)),
        "omega": float(pick("omega", "omega", default=0.224)),
        "gamma": float(pick("gamma", "gamma", default=0.01)),
        "lambda_param": float(pick("lambda_param", "lambda", "lambda_param", default=-0.05)),
        "gamma_real": float(pick("gamma_real", "gamma_real", default=-0.1)),
        "gamma_imag": float(pick("gamma_imag", "gamma_imag", default=0.0)),
        "use_tanh": bool(use_tanh),
        "embed_dim": int(pick("embed_dim", "embed_dim", default=100)),
        "max_len": int(pick("max_len", "max_len", default=175)),
        "input_length": int(pick("input_length", "input_length", "mg_input_length", default=100)),
        "horizon": int(pick("horizon", "horizon", "mg_horizon", default=1)),
        "mg_tau": float(pick("mg_tau", "mg_tau", default=17.0)),
        "mg_delta_t": float(pick("mg_delta_t", "mg_delta_t", default=1.0)),
        "mg_beta": float(pick("mg_beta", "mg_beta", default=0.2)),
        "mg_gamma": float(pick("mg_gamma", "mg_gamma", default=0.1)),
        "mg_n": float(pick("mg_n", "mg_n", default=10.0)),
        "mg_x0": float(pick("mg_x0", "mg_x0", default=1.2)),
        "series_length": int(pick("series_length", "series_length", default=20000)),
        "seed": int(pick("seed", "seed", default=1)),
    }


def build_model_for_task(task: str, kwargs: dict, state: dict | None = None):
    shapes = infer_shapes_from_state(state) if state is not None else None

    if shapes is not None and shapes.get("is_dho"):
        kwargs = dict(kwargs)
        kwargs["dynamics"] = "dho"

    if task == "imdb":
        from training.train_imdb import SLONWithEmbedding

        if shapes is not None:
            vocab_size = shapes["vocab_size"]
            embed_dim = shapes["embed_dim"]
            num_hidden = shapes["num_nodes"]
            num_output = shapes["num_output"]
        else:
            vocab_size = 10000
            embed_dim = kwargs["embed_dim"]
            num_hidden = kwargs["num_hidden"]
            num_output = 2

        model = SLONWithEmbedding(
            vocab_size,
            embed_dim,
            num_hidden,
            num_output,
            kwargs["h"],
            kwargs["alpha"],
            kwargs["omega"],
            kwargs["gamma"],
            pad_idx=0,
            dropout=0.0,
            embedding_weights=None,
            lambda_param=kwargs["lambda_param"],
            gamma_real=kwargs["gamma_real"],
            gamma_imag=kwargs["gamma_imag"],
            dynamics=kwargs["dynamics"],
            use_tanh=kwargs["use_tanh"],
        )
    else:
        num_input = 1
        num_output = 10 if task == "smnist" else 1
        num_hidden = kwargs["num_hidden"]
        if shapes is not None:
            num_input = shapes["num_input"]
            num_hidden = shapes["num_nodes"]
            num_output = shapes["num_output"]

        model = build_oscillator(
            kwargs["dynamics"],
            num_input,
            num_hidden,
            num_output,
            kwargs["h"],
            kwargs["alpha"],
            kwargs["omega"],
            kwargs["gamma"],
            lambda_param=kwargs["lambda_param"],
            gamma_real=kwargs["gamma_real"],
            gamma_imag=kwargs["gamma_imag"],
            use_tanh=kwargs["use_tanh"],
        )

    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Task probes
# ---------------------------------------------------------------------------

def load_smnist_probe(num_steps: int, batch_size: int, seed: int, shuffle: bool = False):
    import torchvision

    torch.manual_seed(seed)
    dataset = torchvision.datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, _ = next(iter(loader))
    # (B, 1, 28, 28) -> (784, B, 1)
    seq = images.view(batch_size, 1, 784).permute(2, 0, 1).contiguous()
    if shuffle:
        perm = torch.randperm(seq.size(0))
        seq = seq[perm]
    if num_steps <= 784:
        seq = seq[:num_steps]
        target = seq[:, :, 0].detach().cpu().numpy().astype(np.float64)
        return seq, target

    # Tile digit streams to reach requested length.
    reps = int(np.ceil(num_steps / 784))
    chunks = []
    targets = []
    for r in range(reps):
        images_r, _ = next(iter(loader))
        seq_r = images_r.view(batch_size, 1, 784).permute(2, 0, 1).contiguous()
        if shuffle:
            seq_r = seq_r[torch.randperm(seq_r.size(0))]
        chunks.append(seq_r)
        targets.append(seq_r[:, :, 0].detach().cpu().numpy())
    seq = torch.cat(chunks, dim=0)[:num_steps]
    target = np.concatenate(targets, axis=0)[:num_steps].astype(np.float64)
    return seq, target


def load_mg_probe(num_steps: int, batch_size: int, kwargs: dict, seed: int):
    from training.train_mackey_glass import generate_mackey_glass

    series = generate_mackey_glass(
        length=max(num_steps + batch_size * 10, int(kwargs["series_length"])),
        tau=kwargs["mg_tau"],
        delta_t=kwargs["mg_delta_t"],
        beta=kwargs["mg_beta"],
        gamma=kwargs["mg_gamma"],
        n=kwargs["mg_n"],
        x0=kwargs["mg_x0"],
    ).astype(np.float64)
    rng = np.random.default_rng(seed)
    max_start = len(series) - num_steps
    if max_start <= 0:
        raise ValueError("Mackey-Glass series shorter than --seq-length")
    starts = rng.choice(max_start, size=batch_size, replace=False)
    windows = np.stack([series[s : s + num_steps] for s in starts], axis=1)
    inputs = torch.from_numpy(windows[:, :, None].astype(np.float32))
    return inputs, windows


def load_imdb_probe(model, num_steps: int, batch_size: int, kwargs: dict, seed: int):
    """Drive the oscillator core with real embedding streams; target = L2 norm."""
    from training.train_imdb import load_preprocessed_data

    cache_dir = os.path.join(PROJECT_ROOT, "data", "imdb_cache")
    cached = load_preprocessed_data(cache_dir, min_freq=2, max_len=kwargs["max_len"])
    if cached is None:
        raise RuntimeError(
            "IMDB cache not found. Run training/train_imdb.py once, or pass "
            "only --probes noise for a checkpoint-free noise MC."
        )
    vocab, train_tokens, _, train_labels, _ = cached
    pad_idx = vocab.word2idx["<pad>"]
    max_len = kwargs["max_len"]

    rng = np.random.default_rng(seed)
    n = min(len(train_tokens), max(batch_size * 4, batch_size))
    indices = rng.choice(len(train_tokens), size=n, replace=False)

    token_rows = []
    for idx in indices[:batch_size]:
        ids = vocab(train_tokens[int(idx)])
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [pad_idx] * (max_len - len(ids))
        token_rows.append(ids)
    token_ids = torch.tensor(token_rows, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        embedded = model.embedding(token_ids)
        embedded = embedded / np.sqrt(model.embedding.embedding_dim)
        embedded = embedded * 3.0
        # (B, L, E) -> (L, B, E)
        embedded = embedded.permute(1, 0, 2).contiguous()

    if num_steps <= embedded.size(0):
        embedded = embedded[:num_steps]
    else:
        reps = int(np.ceil(num_steps / embedded.size(0)))
        embedded = embedded.repeat(reps, 1, 1)[:num_steps]

    target = embedded.norm(dim=-1).detach().cpu().numpy().astype(np.float64)
    return embedded, target


# ---------------------------------------------------------------------------
# Plotting / IO
# ---------------------------------------------------------------------------

def resolve_output_dir(args) -> str:
    if args.output_dir:
        path = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_ROOT, args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.task
        if args.run_dir:
            tag = f"{args.task}_{os.path.basename(os.path.abspath(args.run_dir))}"
        path = os.path.join(PROJECT_ROOT, "results", "memory_capacity", f"{tag}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def plot_mc_spectra(results: dict, output_dir: str, state_dim: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, payload in results.items():
        delays = np.asarray(payload["delays"])
        mc_k = np.asarray(payload["mc_k"])
        color = PROBE_COLORS.get(name, "black")
        label = f"{name} (MC={payload['mc_total']:.2f})"
        axes[0].plot(delays, mc_k, color=color, label=label)
        axes[1].plot(delays, np.cumsum(mc_k), color=color, label=label)

    axes[0].set_xlabel(r"delay $k$")
    axes[0].set_ylabel(r"$\mathrm{MC}_k$")
    axes[0].set_title("Memory capacity spectrum")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=12)

    axes[1].set_xlabel(r"delay $k$")
    axes[1].set_ylabel(r"cumulative $\sum_{j\leq k}\mathrm{MC}_j$")
    axes[1].axhline(state_dim, color="0.4", ls="--", lw=1.0, label=rf"state dim $={state_dim}$")
    axes[1].set_title("Cumulative memory capacity")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "memory_capacity_spectrum.png"), transparent=True)
    plt.close(fig)


def parse_probes(raw: str) -> list[str]:
    probes = [p.strip().lower() for p in raw.split(",") if p.strip()]
    allowed = {"noise", "task"}
    bad = set(probes) - allowed
    if bad:
        raise ValueError(f"Unknown probes {sorted(bad)}; allowed: {sorted(allowed)}")
    if not probes:
        raise ValueError("Provide at least one probe via --probes")
    return probes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    probes = parse_probes(args.probes)
    run_dir = None
    if args.run_dir:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    log_params = None
    if run_dir is not None:
        log_path = os.path.join(run_dir, "log.txt")
        if os.path.isfile(log_path):
            log_params = parse_run_log(log_path)

    kwargs = merge_model_kwargs(args, log_params)
    ckpt_path = None if args.init_only else resolve_checkpoint(run_dir, args.checkpoint)

    state = None
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        print(f"Loaded checkpoint: {ckpt_path}")
    elif args.init_only:
        print("Using randomly initialized weights (--init-only)")
    else:
        print("No checkpoint: using randomly initialized weights")

    torch.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])

    model = build_model_for_task(args.task, kwargs, state=state)
    core = oscillator_core(model)
    num_input = int(core.num_input)
    dim = 2 * int(core.num_nodes)

    output_dir = resolve_output_dir(args)
    print(f"Output directory: {output_dir}")

    mc_results = {}
    common = dict(
        max_delay=args.max_delay,
        washout=args.washout,
        ridge=args.ridge,
        train_fraction=args.train_fraction,
        random_init=args.random_init,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    if "noise" in probes:
        print(f"Noise probe: T={args.seq_length}, B={args.batch_size}, std={args.noise_std}")
        try:
            inputs, target = make_noise_drive(
                args.seq_length,
                args.batch_size,
                num_input,
                noise_std=args.noise_std,
                seed=args.seed,
            )
            result = drive_memory_capacity(core, inputs, target, **common)
            result["probe"] = "noise"
            result["half_life_delay"] = effective_memory_horizon(result["mc_k"], threshold=0.5)
            mc_results["noise"] = result
            print(f"  MC_total={result['mc_total']:.3f}  (state_dim={result['state_dim']})")
        except Exception as exc:
            print(f"  Noise probe failed: {exc}")
            mc_results["noise"] = {"probe": "noise", "error": str(exc)}

    if "task" in probes:
        print(f"Task probe ({args.task}): T={args.seq_length}, B={args.batch_size}")
        try:
            if args.task == "smnist":
                inputs, target = load_smnist_probe(
                    args.seq_length, args.batch_size, args.seed, shuffle=args.shuffle
                )
            elif args.task == "mackey_glass":
                inputs, target = load_mg_probe(args.seq_length, args.batch_size, kwargs, args.seed)
            elif args.task == "imdb":
                inputs, target = load_imdb_probe(model, args.seq_length, args.batch_size, kwargs, args.seed)
            else:
                raise ValueError(f"Unknown task: {args.task}")

            result = drive_memory_capacity(core, inputs, target, **common)
            result["probe"] = "task"
            result["half_life_delay"] = effective_memory_horizon(result["mc_k"], threshold=0.5)
            result["note"] = (
                "Task probes may use correlated inputs; MC_total can exceed the "
                "classical i.i.d. bound of state_dim."
            )
            mc_results["task"] = result
            print(f"  MC_total={result['mc_total']:.3f}  (state_dim={result['state_dim']})")
        except Exception as exc:
            print(f"  Task probe failed: {exc}")
            mc_results["task"] = {"probe": "task", "error": str(exc)}

    if args.compare_init and state is not None:
        print("Comparing against freshly initialized model with same hyperparameters")
        try:
            init_model = build_model_for_task(args.task, kwargs, state=None)
            init_core = oscillator_core(init_model)
            inputs, target = make_noise_drive(
                args.seq_length,
                args.batch_size,
                int(init_core.num_input),
                noise_std=args.noise_std,
                seed=args.seed + 1,
            )
            init_result = drive_memory_capacity(init_core, inputs, target, **common)
            init_result["probe"] = "init_noise"
            init_result["half_life_delay"] = effective_memory_horizon(init_result["mc_k"], threshold=0.5)
            mc_results["init_noise"] = init_result
            print(f"  init MC_total={init_result['mc_total']:.3f}")
        except Exception as exc:
            print(f"  Init comparison failed: {exc}")
            mc_results["init_noise"] = {"probe": "init_noise", "error": str(exc)}

    plottable = {k: v for k, v in mc_results.items() if "mc_k" in v}
    if plottable:
        plot_mc_spectra(plottable, output_dir, state_dim=dim)
    else:
        print("No successful probes to plot.")

    summary = {
        "task": args.task,
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
        "model": {
            "dynamics": kwargs["dynamics"],
            "num_nodes": int(core.num_nodes),
            "num_input": num_input,
            "state_dim": dim,
            "h": kwargs["h"],
            "alpha": kwargs["alpha"],
            "omega": kwargs["omega"],
            "gamma": kwargs["gamma"],
            "lambda_param": kwargs["lambda_param"],
            "gamma_real": kwargs["gamma_real"],
            "gamma_imag": kwargs["gamma_imag"],
            "use_tanh": kwargs["use_tanh"],
        },
        "settings": {
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "max_delay": args.max_delay,
            "washout": args.washout,
            "ridge": args.ridge,
            "train_fraction": args.train_fraction,
            "noise_std": args.noise_std,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "probes": probes,
            "compare_init": bool(args.compare_init),
        },
        "results": mc_results,
    }
    atomic_write_json(os.path.join(output_dir, "memory_capacity.json"), summary)

    with open(os.path.join(output_dir, "log.txt"), "w") as fh:
        fh.write("Memory capacity analysis\n")
        fh.write("=" * 60 + "\n")
        for key, val in summary["model"].items():
            fh.write(f"{key}: {val}\n")
        fh.write("=" * 60 + "\n")
        for name, payload in mc_results.items():
            if "error" in payload:
                fh.write(f"{name}: ERROR {payload['error']}\n")
            else:
                fh.write(
                    f"{name}: MC_total={payload['mc_total']:.6f}, "
                    f"half_life_delay={payload.get('half_life_delay')}\n"
                )

    print(f"Saved results to {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jaeger short-term memory capacity for frozen SLON/HORN models"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["smnist", "imdb", "mackey_glass"],
        help="Task whose input dimensionality / task probe to use",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Training run directory (reads log.txt + best_model.pt)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path (overrides run-dir model file)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Ignore checkpoints and measure MC on a fresh random network",
    )
    parser.add_argument(
        "--compare-init",
        action="store_true",
        help="Also measure noise MC on an untrained twin (requires a checkpoint)",
    )
    parser.add_argument(
        "--probes",
        type=str,
        default="noise,task",
        help="Comma-separated probes: noise, task",
    )
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--seq-length", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-delay", type=int, default=100)
    parser.add_argument("--washout", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional temporal chunking for long drives (each chunk reset)",
    )
    parser.add_argument("--random-init", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true", help="sMNIST: shuffle pixel order")

    parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    parser.add_argument("--num-hidden", type=int, default=50)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.224)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--lambda-param", type=float, default=-0.05)
    parser.add_argument("--gamma-real", type=float, default=-0.1)
    parser.add_argument("--gamma-imag", type=float, default=0.0)
    parser.add_argument("--no-tanh", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=175)
    parser.add_argument("--input-length", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--mg-tau", type=float, default=17.0)
    parser.add_argument("--mg-delta-t", type=float, default=1.0)
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--mg-gamma", type=float, default=0.1)
    parser.add_argument("--mg-n", type=float, default=10.0)
    parser.add_argument("--mg-x0", type=float, default=1.2)
    parser.add_argument("--series-length", type=int, default=20000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
