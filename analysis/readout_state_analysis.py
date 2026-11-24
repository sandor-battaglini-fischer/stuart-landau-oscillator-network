"""Per-node trajectories, the final state seen by h2o, its t-SNE/PCA embedding and the linear decision.

Examples:
  python analysis/readout_state_analysis.py --task smnist --run-dir results/smnist/<run>
  python analysis/readout_state_analysis.py --task imdb --run-dir results/imdb/<run>
  python analysis/readout_state_analysis.py --task mackey_glass --run-dir results/mackey_glass/<run>
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.memory_capacity import (
    build_model_for_task,
    merge_model_kwargs,
    parse_run_log,
    resolve_checkpoint,
)
from utils.device import resolve_device, seed_all
from utils.memory_capacity import oscillator_core
from utils.plotting_utils.readout_state import plot_all_readout_analysis, trace_readout_states
from utils.plotting_utils.signal_stages import prepare_imdb_sequence
from utils.plotting_utils.class_encoding import prepare_smnist_batch
from utils.run_dirs import atomic_write_json

try:
    from utils.plotting_utils.style import apply_style

    apply_style()
except Exception:
    pass

import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200})


def resolve_output_dir(args):
    if args.output_dir:
        path = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_ROOT, args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.task
        if args.run_dir:
            tag = f"{args.task}_{os.path.basename(os.path.abspath(args.run_dir))}"
        elif args.init_only:
            tag = f"{args.task}_init"
        path = os.path.join(PROJECT_ROOT, "results", "readout_state", f"{tag}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _balanced_indices(labels, n_per_class, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == c)
        take = min(int(n_per_class), len(idx))
        if take == 0:
            continue
        chosen.append(rng.choice(idx, size=take, replace=False))
    if not chosen:
        raise RuntimeError("No examples found for any class")
    return np.concatenate(chosen)


def load_smnist_examples(n_per_class, seed, shuffle, device):
    import torchvision

    dataset = torchvision.datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    images = dataset.data.unsqueeze(1).float() / 255.0
    labels = dataset.targets.numpy()
    idx = _balanced_indices(labels, n_per_class, seed)
    images = images[idx]
    labels = labels[idx]
    inputs = prepare_smnist_batch(images)
    if shuffle:
        perm = torch.randperm(inputs.size(0))
        inputs = inputs[perm]
    return inputs.to(device), labels.astype(np.int64), None


def _imdb_test_text_lookup():
    from collections import defaultdict

    from training.train_imdb import DATA_DIR, tokenize

    root = os.path.join(DATA_DIR, "imdb", "aclImdb", "test")
    buckets = defaultdict(list)
    if not os.path.isdir(root):
        return buckets
    for lab, sub in ((0, "neg"), (1, "pos")):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for filename in os.listdir(d):
            if not filename.endswith(".txt"):
                continue
            try:
                rating = int(filename.rsplit("_", 1)[1].split(".")[0])
            except (IndexError, ValueError):
                rating = float("nan")
            path = os.path.join(d, filename)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            buckets[(tuple(tokenize(text)), lab)].append((text, rating))
    return buckets


def load_imdb_review_bundle(kwargs, n_per_class, seed, cache_dir):
    from collections import defaultdict

    from training.train_imdb import load_preprocessed_data

    cached = load_preprocessed_data(cache_dir, min_freq=2, max_len=kwargs["max_len"])
    if cached is None:
        raise RuntimeError(
            "IMDB cache not found. Run training/train_imdb.py once to build data/imdb_cache."
        )
    vocab, _, test_tokens, _, test_labels = cached
    pad_idx = vocab.word2idx["<pad>"]
    max_len = kwargs["max_len"]
    labels = np.asarray(test_labels, dtype=np.int64)
    idx = _balanced_indices(labels, n_per_class, seed)
    chosen_labels = labels[idx]

    rows = []
    token_lists = []
    for i in idx:
        toks = list(test_tokens[int(i)])
        token_lists.append(toks)
        ids = vocab(toks)
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [pad_idx] * (max_len - len(ids))
        rows.append(ids)
    token_ids = torch.tensor(rows, dtype=torch.long)

    n_tokens = np.array([len(toks) for toks in token_lists], dtype=np.int64)
    ratings = np.full(len(token_lists), np.nan, dtype=np.float64)
    texts = [" ".join(toks) for toks in token_lists]
    lookup = _imdb_test_text_lookup()
    used = defaultdict(int)
    if lookup:
        for j, (toks, lab) in enumerate(zip(token_lists, chosen_labels)):
            key = (tuple(toks), int(lab))
            opts = lookup.get(key, [])
            k = used[key]
            if k < len(opts):
                texts[j], ratings[j] = opts[k]
                used[key] += 1
    n_rated = int(np.isfinite(ratings).sum())
    print(f"IMDB reviews: {len(token_lists)} examples, ratings matched {n_rated}/{len(token_lists)}")
    extras = {
        "review_tokens": token_lists,
        "review_texts": texts,
        "review_n_tokens": n_tokens,
        "review_ratings": ratings,
        "review_max_len": int(max_len),
    }
    return token_ids, chosen_labels, extras


def load_imdb_examples(model, kwargs, n_per_class, seed, device, cache_dir):
    token_ids, chosen_labels, extras = load_imdb_review_bundle(kwargs, n_per_class, seed, cache_dir)
    model.eval()
    inputs = prepare_imdb_sequence(model, token_ids.to(device))
    return inputs, chosen_labels, None, extras


def _round_list(arr, ndigits=6):
    return np.round(np.asarray(arr, dtype=np.float64), ndigits).tolist()


def write_final_states(output_dir, task, labels, z_real_last, z_imag_last, z_mag_last, review_extras=None, logits=None):
    import csv

    from utils.plotting_utils.readout_state import class_names

    y_true = np.asarray(labels, dtype=np.int64)
    z_re = np.asarray(z_real_last, dtype=np.float64)
    z_im = np.asarray(z_imag_last, dtype=np.float64)
    z_mag = np.asarray(z_mag_last, dtype=np.float64)
    names = class_names(task, y_true)
    n_examples, n_nodes = z_re.shape
    y_pred = None
    if logits is not None:
        logits = np.asarray(logits)
        if logits.ndim == 2 and logits.shape[1] > 1:
            y_pred = np.argmax(logits, axis=1).astype(np.int64)
    payload = {
        "task": task,
        "n_examples": int(n_examples),
        "n_nodes": int(n_nodes),
        "class_names": {str(k): v for k, v in names.items()},
        "y_true": y_true.tolist(),
        "z_real": _round_list(z_re),
        "z_imag": _round_list(z_im),
        "z_mag": _round_list(z_mag),
    }
    if y_pred is not None:
        payload["y_pred"] = y_pred.tolist()
    extras = review_extras or {}
    reviews = extras.get("review_texts")
    if reviews is not None:
        payload["reviews"] = [str(t) for t in reviews]
        ratings = np.asarray(extras["review_ratings"], dtype=np.float64)
        payload["ratings"] = [None if not np.isfinite(r) else float(r) for r in ratings]
        payload["n_tokens"] = np.asarray(extras["review_n_tokens"], dtype=np.int64).tolist()

    json_path = os.path.join(output_dir, "final_states.json")
    atomic_write_json(json_path, payload, indent=None)

    csv_path = os.path.join(output_dir, "final_states.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["example", "y_true", "y_true_name", "y_pred", "y_pred_name", "node", "re", "im", "mag"]
        if reviews is not None:
            header.extend(["rating", "n_tokens"])
        writer.writerow(header)
        for i in range(n_examples):
            true_name = names.get(int(y_true[i]), str(int(y_true[i])))
            if y_pred is None:
                pred_id, pred_name = "", ""
            else:
                pred_id = int(y_pred[i])
                pred_name = names.get(pred_id, str(pred_id))
            extra = []
            if reviews is not None:
                r = extras["review_ratings"][i]
                extra = ["" if not np.isfinite(r) else float(r), int(extras["review_n_tokens"][i])]
            for n in range(n_nodes):
                writer.writerow(
                    [
                        i,
                        int(y_true[i]),
                        true_name,
                        pred_id,
                        pred_name,
                        n,
                        float(z_re[i, n]),
                        float(z_im[i, n]),
                        float(z_mag[i, n]),
                        *extra,
                    ]
                )
    return json_path, csv_path


def write_recurrent_matrix(output_dir, task, core):
    import math

    weight = core.h2h.weight.detach().cpu().numpy()
    bias = core.h2h.bias.detach().cpu().numpy() if core.h2h.bias is not None else np.zeros(weight.shape[0])
    n_nodes = int(weight.shape[0])
    gain_rec = float(core.gain_rec) if hasattr(core, "gain_rec") else 1.0 / math.sqrt(n_nodes)
    payload = {
        "task": task,
        "n_nodes": n_nodes,
        "shape": [n_nodes, int(weight.shape[1])],
        "layout": "h2h_weight is (N, 2N): rows are nodes, columns are [Re z_0..N-1, Im z_0..N-1]. Used as gain_rec * (W @ concat(re, im) + b).",
        "gain_rec": gain_rec,
        "h2h_weight": _round_list(weight),
        "h2h_bias": _round_list(bias),
    }
    if weight.shape[1] == 2 * n_nodes:
        payload["h2h_weight_re"] = _round_list(weight[:, :n_nodes])
        payload["h2h_weight_im"] = _round_list(weight[:, n_nodes:])
    path = os.path.join(output_dir, f"{task}_recurrent_matrix.json")
    atomic_write_json(path, payload, indent=None)
    return path


def load_mg_examples(kwargs, n_per_class, seed, device):
    from training.train_mackey_glass import build_dataloaders

    _, _, test_loader, _, _, _ = build_dataloaders(
        series_length=kwargs["series_length"],
        input_length=kwargs["input_length"],
        horizon=kwargs["horizon"],
        batch_size=64,
        val_fraction=0.1,
        test_fraction=0.1,
        tau=kwargs["mg_tau"],
        delta_t=kwargs["mg_delta_t"],
        beta=kwargs["mg_beta"],
        gamma_mg=kwargs["mg_gamma"],
        n=kwargs["mg_n"],
        x0=kwargs["mg_x0"],
        seed=kwargs["seed"],
    )
    windows = []
    targets = []
    for batch in test_loader:
        inputs = batch[0]
        target = batch[1]
        windows.append(inputs.squeeze(-1).numpy())
        targets.append(target.numpy().reshape(-1))
        if sum(w.shape[0] for w in windows) >= 12 * n_per_class:
            break
    windows = np.concatenate(windows, axis=0)
    targets = np.concatenate(targets, axis=0)
    q33, q66 = np.percentile(targets, [33.33, 66.67])
    groups = np.where(targets <= q33, 0, np.where(targets <= q66, 1, 2)).astype(np.int64)
    idx = _balanced_indices(groups, n_per_class, seed)
    inputs = torch.from_numpy(windows[idx, :, None].astype(np.float32)).permute(1, 0, 2)
    return inputs.to(device), groups[idx], targets[idx]


def run(args):
    device = resolve_device(args.device)
    seed_all(args.seed, device)

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
    ckpt = None if args.init_only else resolve_checkpoint(run_dir, args.checkpoint)
    state = None
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        print(f"Loaded checkpoint: {ckpt}")
    elif args.init_only:
        print("Using randomly initialized weights (--init-only)")

    model = build_model_for_task(args.task, kwargs, state=state)
    model.to(device)
    model.eval()
    core = oscillator_core(model)

    cache_dir = args.cache_dir
    if cache_dir is not None and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(PROJECT_ROOT, cache_dir)

    review_extras = None
    if args.task == "smnist":
        inputs, labels, targets = load_smnist_examples(args.examples_per_class, args.seed, args.shuffle, device)
    elif args.task == "imdb":
        inputs, labels, targets, review_extras = load_imdb_examples(
            model, kwargs, args.examples_per_class, args.seed, device, cache_dir
        )
    else:
        inputs, labels, targets = load_mg_examples(kwargs, args.examples_per_class, args.seed, device)

    print(
        f"Tracing {args.task}: {inputs.shape[1]} examples, T={inputs.shape[0]}, "
        f"N={core.num_nodes} -> {core.h2o.in_features}-D readout vector",
        flush=True,
    )
    traced = trace_readout_states(model, inputs, batch_size=args.trace_batch_size)
    weight = core.h2o.weight.detach().cpu().numpy()
    bias = core.h2o.bias.detach().cpu().numpy() if core.h2o.bias is not None else np.zeros(weight.shape[0])

    horizon = int(kwargs.get("horizon", 25))
    payload = {
        **traced,
        "labels": labels,
        "targets": None if targets is None else np.asarray(targets, dtype=np.float64),
        "h2o_weight": weight,
        "h2o_bias": bias,
        "input_windows": inputs.detach().cpu().numpy()[:, :, 0],
        "mg_delay": horizon if horizon >= 2 else 25,
    }
    if review_extras:
        payload.update(review_extras)

    output_dir = resolve_output_dir(args)
    print(f"Plotting -> {output_dir}", flush=True)
    summary = plot_all_readout_analysis(payload, output_dir, args.task, seed=args.seed)
    summary.update(
        {
            "checkpoint": ckpt,
            "run_dir": run_dir,
            "init_only": bool(args.init_only),
            "seq_length": int(inputs.shape[0]),
            "model": {
                "num_hidden": int(core.num_nodes),
                "h2o_in": int(core.h2o.in_features),
                "h2o_out": int(core.h2o.out_features),
            },
        }
    )
    atomic_write_json(os.path.join(output_dir, "summary.json"), summary)
    write_final_states(
        output_dir,
        args.task,
        labels,
        traced["z_real"][-1],
        traced["z_imag"][-1],
        traced["z_magnitude"][-1],
        review_extras,
        logits=traced["logits"],
    )
    write_recurrent_matrix(output_dir, args.task, core)
    np.savez_compressed(
        os.path.join(output_dir, "readout_states.npz"),
        z_real=traced["z_real"],
        z_imag=traced["z_imag"],
        z_magnitude=traced["z_magnitude"],
        readout_features=traced["readout_features"],
        logits=traced["logits"],
        labels=labels,
        targets=np.asarray([] if targets is None else targets, dtype=np.float64),
        input_windows=inputs.detach().cpu().numpy()[:, :, 0],
    )
    print(f"Saved results to {output_dir}")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Per-node z(t) and t-SNE of the h2o input vector")
    parser.add_argument("--task", required=True, choices=["smnist", "imdb", "mackey_glass"])
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--examples-per-class", type=int, default=80)
    parser.add_argument("--trace-batch-size", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true", help="sMNIST: shuffle pixel order")
    parser.add_argument("--cache-dir", type=str, default="data/imdb_cache")

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
