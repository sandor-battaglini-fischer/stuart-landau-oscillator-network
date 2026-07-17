import os
import sys
import math
import argparse
from datetime import datetime
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import torchvision

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()

mg_cmap = mcolors.LinearSegmentedColormap.from_list("mg_cmap", [(0.5, 0.6, 0.0), ifisc_green])

from training.train_mackey_glass import MackeyGlassDataset, generate_mackey_glass, build_dataloaders
from models import SLON, HORN
from utils.model_factory import build_oscillator
from utils.plotting_utils.mackey_glass_encoding import plot_mackey_glass_encoding_analysis_from_loader
from training.train_imdb import (
    tokenize,
    Vocabulary,
    download_imdb,
    load_imdb_reviews,
    load_preprocessed_data,
    save_preprocessed_data,
    download_glove,
    load_glove_vectors,
    SLONWithEmbedding,
)

# TASK_COLORS = {
#     "smnist": ifisc_green,
#     "imdb": thesis_blue,
#     "mackey_glass": thesis_red,
# }

TASK_COLORS = {
    "smnist": "black",
    "imdb": "black",
    "mackey_glass": "black",
}


def resolve_output_dir(prefix, output_dir=None):
    if output_dir is not None:
        path = os.path.abspath(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(PROJECT_ROOT, "results", "input_analysis", f"{prefix}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _ylim_with_margin(*arrays):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    y_min, y_max = float(values.min()), float(values.max())
    margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
    return y_min - margin, y_max + margin


def compute_pre_activations_stuart_landau(model, inputs, random_init=None):
    num_timesteps, batch_size, _ = inputs.shape

    if random_init is not None:
        z_real = torch.randn(batch_size, model.num_nodes) * random_init
        z_imag = torch.randn(batch_size, model.num_nodes) * random_init
    else:
        z_real = torch.zeros(batch_size, model.num_nodes)
        z_imag = torch.zeros(batch_size, model.num_nodes)

    lambda_omega_coeff = torch.complex(model.lambda_param, model.omega_param)
    gamma_coeff = torch.complex(model.gamma_real, model.gamma_imag)

    pre_acts = []

    for t in range(num_timesteps):
        input_t = inputs[t]
        z_state = torch.cat([z_real, z_imag], dim=1)
        pre_act = model.i2h(input_t) + model.gain_rec * model.h2h(z_state)
        if model.use_tanh:
            input_force = model.alpha * torch.tanh(pre_act)
        else:
            input_force = model.alpha * pre_act

        z = torch.complex(z_real, z_imag)
        if model.linear_dynamics:
            dz_dt = lambda_omega_coeff * z + torch.complex(input_force, torch.zeros_like(input_force))
        else:
            dz_dt = lambda_omega_coeff * z + gamma_coeff * torch.abs(z) ** 2 * z + torch.complex(
                input_force, torch.zeros_like(input_force)
            )
        z = z + model.h * dz_dt
        z_real = torch.real(z)
        z_imag = torch.imag(z)

        pre_acts.append(pre_act.detach().cpu())

    pre_acts = torch.stack(pre_acts, dim=0)
    return pre_acts


def compute_pre_activations_horn(model, inputs, random_init=None):
    num_timesteps, batch_size, _ = inputs.shape

    if random_init is not None:
        x_t = torch.randn(batch_size, model.num_nodes) * random_init
        y_t = torch.randn(batch_size, model.num_nodes) * random_init
    else:
        x_t = torch.zeros(batch_size, model.num_nodes)
        y_t = torch.zeros(batch_size, model.num_nodes)

    pre_acts = []

    for t in range(num_timesteps):
        input_t = inputs[t]
        pre_act = model.i2h(input_t) + model.gain_rec * model.h2h(y_t)
        x_t, y_t = model.dynamics_step(x_t, y_t, input_t)
        pre_acts.append(pre_act.detach().cpu())

    return torch.stack(pre_acts, dim=0)


def compute_pre_activations(model, inputs, random_init=None):
    if isinstance(model, HORN):
        return compute_pre_activations_horn(model, inputs, random_init=random_init)
    return compute_pre_activations_stuart_landau(model, inputs, random_init=random_init)


def simulate_stuart_landau_base(lambda_param, omega, gamma_real, gamma_imag, h, num_steps, z0=None):
    dtype = torch.float64
    if z0 is None:
        z = torch.complex(torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype))
    else:
        z = torch.complex(torch.tensor(z0[0], dtype=dtype), torch.tensor(z0[1], dtype=dtype))

    lambda_omega = torch.complex(torch.tensor(lambda_param, dtype=dtype), torch.tensor(omega, dtype=dtype))
    gamma = torch.complex(torch.tensor(gamma_real, dtype=dtype), torch.tensor(gamma_imag, dtype=dtype))

    traj_real = []
    for _ in range(num_steps):
        dz_dt = lambda_omega * z + gamma * torch.abs(z) ** 2 * z
        z = z + h * dz_dt
        traj_real.append(torch.real(z).item())

    return np.array(traj_real)


def plot_time_and_frequency(pre_acts, output_dir, prefix, num_units=5, sample_mean=True):
    os.makedirs(output_dir, exist_ok=True)

    if sample_mean:
        series = pre_acts.mean(dim=1)
    else:
        series = pre_acts[:, 0, :]

    series = series.numpy()
    num_timesteps, num_nodes = series.shape
    unit_indices = list(range(min(num_units, num_nodes)))

    t = np.arange(num_timesteps)

    plt.figure(figsize=(10, 6))
    for idx in unit_indices:
        plt.plot(t, series[:, idx], label=f"unit {idx}")
    plt.xlabel("time step")
    plt.ylabel("pre-activation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_time_series.png"), transparent=True)
    plt.close()

    freqs = np.fft.rfftfreq(num_timesteps, d=1.0)

    plt.figure(figsize=(10, 6))
    for idx in unit_indices:
        fft_vals = np.fft.rfft(series[:, idx])
        amp = np.abs(fft_vals)
        plt.plot(freqs, amp, label=f"unit {idx}")
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_frequency_spectrum.png"), transparent=True)
    plt.close()


def plot_single_example_time_series(pre_acts, output_dir, prefix, num_units=5, example_idx=0):
    os.makedirs(output_dir, exist_ok=True)

    series = pre_acts[:, example_idx, :].numpy()
    num_timesteps, num_nodes = series.shape
    unit_indices = list(range(min(num_units, num_nodes)))

    t = np.arange(num_timesteps)

    plt.figure(figsize=(10, 6))
    for idx in unit_indices:
        plt.plot(t, series[:, idx], label=f"unit {idx}")
    plt.xlabel("time step")
    plt.ylabel(r"pre-activation $u(t)=W_{i2h}x(t)+\mathrm{gain}_{\mathrm{rec}}W_{h2h}[\Re(z),\Im(z)]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_single_example_time_series.png"), transparent=True)
    plt.close()


def get_smnist_pre_acts(args, model=None, return_model=False):
    torch.manual_seed(args.seed)

    dim_input = 1
    dim_output = 10

    if model is None:
        model = build_oscillator(
            args.dynamics,
            dim_input,
            args.num_hidden,
            dim_output,
            args.h,
            args.alpha,
            args.omega,
            args.gamma,
            lambda_param=args.lambda_param,
            gamma_real=args.gamma_real,
            gamma_imag=args.gamma_imag,
        )

        if args.checkpoint is not None and os.path.isfile(args.checkpoint):
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state)

    dataset = torchvision.datasets.MNIST(
        root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True
    )

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    images, _ = next(iter(loader))
    images = images.reshape(-1, 1, 784)
    images = images.permute(2, 0, 1)

    input_scale = getattr(args, "input_scale", 1.0)
    if input_scale != 1.0:
        images = images * input_scale

    if args.shuffle:
        perm = torch.randperm(images.size(0))
        images = images[perm, :, :]

    pre_acts = compute_pre_activations(model, images, random_init=args.random_init)
    pixel_trace = images[:, 0, 0].detach().cpu().numpy()
    if return_model:
        return pre_acts, pixel_trace, model
    return pre_acts, pixel_trace


def get_imdb_pre_acts(args, model=None, vocab=None, return_model=False):
    torch.manual_seed(args.seed)

    target_period = args.max_len
    if not args.sweep_omega:
        args.omega = (2 * math.pi) / (target_period * args.h)

    dim_input = args.embed_dim
    dim_output = 2

    if vocab is None:
        cached_data = None
        if not args.force_reprocess:
            cached_data = load_preprocessed_data(args.cache_dir, args.min_freq, args.max_len)

        if cached_data is not None:
            vocab, train_tokens, test_tokens, train_labels, test_labels = cached_data
            vocab_size = len(vocab)
            pad_idx = vocab.word2idx["<pad>"]

            def process_tokens_to_ids(tokens_list, pad):
                token_ids_list = []
                for tokens in tokens_list:
                    token_ids = vocab(tokens)
                    if len(token_ids) > args.max_len:
                        token_ids = token_ids[: args.max_len]
                    else:
                        token_ids = token_ids + [pad] * (args.max_len - len(token_ids))
                    token_ids_list.append(token_ids)
                return token_ids_list

            train_token_ids = process_tokens_to_ids(train_tokens, pad_idx)
            train_data = [
                (torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long))
                for token_ids, label in zip(train_token_ids, train_labels)
            ]
        else:
            imdb_path = download_imdb()
            train_texts, train_labels = load_imdb_reviews(imdb_path, "train")

            vocab = Vocabulary(min_freq=args.min_freq)

            train_tokens = [tokenize(text) for text in train_texts]

            for tokens in train_tokens:
                vocab.word_counts.update(tokens)

            vocab.word2idx["<pad>"] = 0
            vocab.word2idx["<unk>"] = 1
            idx = 2
            for word, count in vocab.word_counts.items():
                if count >= args.min_freq:
                    vocab.word2idx[word] = idx
                    idx += 1

            vocab.idx2word = {idx: word for word, idx in vocab.word2idx.items()}
            vocab_size = len(vocab)
            pad_idx = vocab.word2idx["<pad>"]

            save_preprocessed_data(
                args.cache_dir, vocab, train_tokens, train_tokens, train_labels, train_labels, args.min_freq, args.max_len
            )

            def process_tokens_to_ids(tokens, pad):
                token_ids = vocab(tokens)
                if len(token_ids) > args.max_len:
                    token_ids = token_ids[: args.max_len]
                else:
                    token_ids = token_ids + [pad] * (args.max_len - len(token_ids))
                return token_ids

            train_token_ids = [process_tokens_to_ids(tokens, pad_idx) for tokens in train_tokens]
            train_data = [
                (torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long))
                for token_ids, label in zip(train_token_ids, train_labels)
            ]
    else:
        vocab_size = len(vocab)
        pad_idx = vocab.word2idx["<pad>"]
        cached_data = load_preprocessed_data(args.cache_dir, args.min_freq, args.max_len)
        if cached_data is not None:
            _, train_tokens, _, train_labels, _ = cached_data
            def process_tokens_to_ids(tokens_list, pad):
                token_ids_list = []
                for tokens in tokens_list:
                    token_ids = vocab(tokens)
                    if len(token_ids) > args.max_len:
                        token_ids = token_ids[: args.max_len]
                    else:
                        token_ids = token_ids + [pad] * (args.max_len - len(token_ids))
                    token_ids_list.append(token_ids)
                return token_ids_list
            train_token_ids = process_tokens_to_ids(train_tokens, pad_idx)
            train_data = [
                (torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long))
                for token_ids, label in zip(train_token_ids, train_labels)
            ]

    embedding_weights = None
    if args.glove:
        glove_path = args.glove
        if not os.path.isabs(glove_path):
            glove_path = os.path.join(args.glove_dir, glove_path)
        if not os.path.exists(glove_path):
            glove_path = download_glove(args.glove_dir, dim=args.embed_dim)
        embedding_weights = load_glove_vectors(glove_path, vocab, args.embed_dim)

    if model is None:
        model = SLONWithEmbedding(
            vocab_size,
            dim_input,
            args.num_hidden,
            dim_output,
            args.h,
            args.alpha,
            args.omega,
            args.gamma,
            pad_idx,
            dropout=args.dropout,
            embedding_weights=embedding_weights,
            lambda_param=args.lambda_param,
            gamma_real=args.gamma_real,
            gamma_imag=args.gamma_imag,
            dynamics=args.dynamics,
        )

        if args.checkpoint is not None and os.path.isfile(args.checkpoint):
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state)

    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    token_ids, _ = next(iter(loader))

    embedded = model.embedding(token_ids)
    embedded = embedded / math.sqrt(model.embedding.embedding_dim)
    embedded = model.dropout(embedded)

    input_scale = getattr(args, "input_scale", 1.0)
    if input_scale != 1.0:
        embedded = embedded * input_scale

    embedded = embedded.permute(1, 0, 2)

    pre_acts = compute_pre_activations(model.slon, embedded, random_init=args.random_init)
    embed_mag_trace = embedded[:, 0, :].norm(dim=-1).detach().cpu().numpy()
    if return_model:
        return pre_acts, embed_mag_trace, model, vocab
    return pre_acts, embed_mag_trace


def mg_series_indices_for_sample(dataset, last_sampled_idx):
    matches = np.where(dataset.sampled_indices == int(last_sampled_idx))[0]
    idx = int(matches[0]) if len(matches) else int(last_sampled_idx)
    start = max(0, idx - dataset.input_length + 1)
    indices = dataset.sampled_indices[start:idx + 1].astype(np.int64)
    if len(indices) < dataset.input_length:
        padding = np.full(dataset.input_length - len(indices), indices[0] if len(indices) else 0)
        indices = np.concatenate([padding, indices])
    return indices


def plot_mackey_glass_with_discretization(
    ax,
    series,
    mg_trace,
    series_indices,
    input_length,
    horizon,
    input_scale=1.0,
    sample_step=1,
    color=thesis_blue,
):
    scaled_series = series * input_scale
    window_indices = np.unique(series_indices.astype(np.int64))
    window_start = int(window_indices[0])
    window_end = int(window_indices[-1])
    view_indices = np.arange(window_start, window_end + 1)

    ax.plot(
        view_indices,
        scaled_series[view_indices],
        color=color,
        alpha=0.25,
        linewidth=0.5,
    )
    ax.scatter(
        view_indices,
        scaled_series[view_indices],
        color=color,
        s=12,
        alpha=0.5,
        label="samples",
        zorder=3,
    )
    ax.plot(
        series_indices,
        mg_trace,
        label="Input",
        color=color,
        linewidth=1.5,
        zorder=4,
    )


def get_mg_pre_acts(args, model=None, series=None, return_model=False):
    torch.manual_seed(args.seed)

    dim_input = 1
    dim_output = 1

    if model is None:
        model = build_oscillator(
            args.dynamics,
            dim_input,
            args.num_hidden,
            dim_output,
            args.h,
            args.alpha,
            args.omega,
            args.gamma,
            lambda_param=args.lambda_param,
            gamma_real=args.gamma_real,
            gamma_imag=args.gamma_imag,
        )

        if args.checkpoint is not None and os.path.isfile(args.checkpoint):
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state)

    if series is None:
        series = generate_mackey_glass(
            length=args.mg_series_length,
            tau=args.mg_tau,
            delta_t=args.mg_delta_t,
            beta=args.mg_beta,
            gamma=args.mg_gamma,
            n=args.mg_n,
            x0=args.mg_x0,
        )

    dataset = MackeyGlassDataset(series, args.mg_input_length, args.mg_horizon, sample_step=1)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    inputs, _, last_sampled_idxs, _ = next(iter(loader))
    inputs = inputs * args.mg_input_scale
    inputs = inputs.permute(1, 0, 2)

    pre_acts = compute_pre_activations(model, inputs, random_init=args.random_init)
    mg_trace = inputs[:, 0, 0].detach().cpu().numpy()
    series_indices = mg_series_indices_for_sample(dataset, last_sampled_idxs[0].item())
    if return_model:
        return pre_acts, mg_trace, series_indices, model, series
    return pre_acts, mg_trace, series_indices


def analyze_smnist(args):
    pre_acts, pixel_trace = get_smnist_pre_acts(args)
    output_dir = resolve_output_dir("smnist", getattr(args, "output_dir", None))
    plot_single_example_time_series(pre_acts, output_dir, prefix="smnist", num_units=args.num_units_plot, example_idx=0)

    t = np.arange(pixel_trace.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(t, pixel_trace)
    plt.xlabel("time step")
    plt.ylabel("pixel value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "smnist_single_example_raw_input.png"), transparent=True)
    plt.close()


def analyze_imdb(args):
    pre_acts, embed_mag_trace = get_imdb_pre_acts(args)
    output_dir = resolve_output_dir("imdb", getattr(args, "output_dir", None))
    plot_single_example_time_series(pre_acts, output_dir, prefix="imdb", num_units=args.num_units_plot, example_idx=0)

    t = np.arange(embed_mag_trace.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(t, embed_mag_trace)
    plt.xlabel("time step")
    plt.ylabel("embedding norm")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "imdb_single_example_raw_input.png"), transparent=True)
    plt.close()


def analyze_mg(args):
    pre_acts, mg_trace, series_indices, model, series = get_mg_pre_acts(args, return_model=True)
    output_dir = resolve_output_dir("mg", getattr(args, "output_dir", None))
    plot_single_example_time_series(pre_acts, output_dir, prefix="mg", num_units=args.num_units_plot, example_idx=0)

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    plot_mackey_glass_with_discretization(
        ax,
        series,
        mg_trace,
        series_indices,
        args.mg_input_length,
        args.mg_horizon,
        input_scale=args.mg_input_scale,
    )
    plt.xlabel("time step")
    plt.ylabel("MG value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mg_single_example_raw_input.png"), transparent=True)
    plt.close()

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        _, _, test_loader, _, _, _ = build_dataloaders(
            series_length=args.mg_series_length,
            input_length=args.mg_input_length,
            horizon=args.mg_horizon,
            batch_size=args.batch_size,
            val_fraction=getattr(args, "val_fraction", 0.1),
            test_fraction=getattr(args, "test_fraction", 0.1),
            tau=args.mg_tau,
            delta_t=args.mg_delta_t,
            beta=args.mg_beta,
            gamma_mg=args.mg_gamma,
            n=args.mg_n,
            x0=args.mg_x0,
            seed=args.seed,
            remove_top_n_freqs=0,
        )
        tau_steps = int(args.mg_tau / args.mg_delta_t)
        plot_mackey_glass_encoding_analysis_from_loader(
            model,
            test_loader,
            output_dir,
            epoch=None,
            grouping=getattr(args, "encoding_grouping", "all"),
            num_per_group=max(1, getattr(args, "encoding_analysis_examples", 50) // 2),
            num_units_plot=args.num_units_plot,
            tau_steps=tau_steps,
        )

def analyze_compare(args):
    use_sm = getattr(args, "use_sm", False)
    use_im = getattr(args, "use_im", False)
    use_mg = getattr(args, "use_mg", False)

    if not (use_sm or use_im or use_mg):
        raise ValueError("At least one of --use-sm, --use-im, --use-mg must be specified for 'compare'.")

    sm_lambda = args.sm_lambda
    sm_omega = args.sm_omega
    sm_gamma_real = args.sm_gamma_real
    sm_gamma_imag = args.sm_gamma_imag

    if args.im_omega is not None:
        omega_imdb = args.im_omega
    else:
        omega_imdb = (2 * math.pi) / (args.max_len * args.h)

    im_lambda = args.im_lambda
    im_gamma_real = args.im_gamma_real
    im_gamma_imag = args.im_gamma_imag

    if args.mg_lambda is None:
        mg_lambdas = [-0.04]
    else:
        mg_lambdas = args.mg_lambda if isinstance(args.mg_lambda, list) else [args.mg_lambda]
    
    if args.mg_omega is None:
        mg_omegas = [0.224]
    else:
        mg_omegas = args.mg_omega if isinstance(args.mg_omega, list) else [args.mg_omega]
    
    if args.mg_gamma_real is None:
        mg_gamma_reals = [-0.05]
    else:
        mg_gamma_reals = args.mg_gamma_real if isinstance(args.mg_gamma_real, list) else [args.mg_gamma_real]
    
    if args.mg_gamma_imag is None:
        mg_gamma_imags = [0.1]
    else:
        mg_gamma_imags = args.mg_gamma_imag if isinstance(args.mg_gamma_imag, list) else [args.mg_gamma_imag]

    sm_input_scale = args.sm_input_scale
    im_input_scale = args.im_input_scale
    
    if args.mg_input_scale is None:
        mg_input_scales = [1.0]
    else:
        mg_input_scales = args.mg_input_scale if isinstance(args.mg_input_scale, list) else [args.mg_input_scale]

    num_mg_settings = len(mg_lambdas)
    if not (
        len(mg_omegas) == num_mg_settings
        and len(mg_gamma_reals) == num_mg_settings
        and len(mg_gamma_imags) == num_mg_settings
        and len(mg_input_scales) == num_mg_settings
    ):
        raise ValueError("All MG parameter lists must have the same length.")

    pre_acts_smnist = None
    pixel_trace = None
    model_sm = None
    if use_sm:
        smnist_args = SimpleNamespace(
            dynamics=args.dynamics,
            num_hidden=args.num_hidden,
            batch_size=args.batch_size,
            seed=args.seed,
            h=args.h,
            alpha=0.04,
            omega=sm_omega,
            gamma=0.01,
            lambda_param=sm_lambda,
            gamma_real=sm_gamma_real,
            gamma_imag=sm_gamma_imag,
            input_scale=sm_input_scale,
            shuffle=args.shuffle,
            random_init=args.random_init,
            checkpoint=args.checkpoint,
        )
        pre_acts_smnist, pixel_trace, model_sm = get_smnist_pre_acts(smnist_args, return_model=True)

    pre_acts_imdb = None
    embed_mag_trace = None
    model_im = None
    vocab_im = None
    if use_im:
        imdb_args = SimpleNamespace(
            dynamics=args.dynamics,
            num_hidden=args.num_hidden,
            batch_size=args.batch_size,
            seed=args.seed,
            lr=1e-3,
            h=args.h,
            alpha=0.04,
            omega=omega_imdb,
            gamma=0.01,
            lambda_param=im_lambda,
            gamma_real=im_gamma_real,
            gamma_imag=im_gamma_imag,
            embed_dim=args.embed_dim,
            max_len=args.max_len,
            min_freq=args.min_freq,
            dropout=0.3,
            early_stop_patience=100,
            weight_decay=0.05,
            glove=args.glove,
            glove_dir=args.glove_dir,
            sweep_omega=False,
            cache_dir=args.cache_dir,
            force_reprocess=args.force_reprocess,
            random_init=args.random_init,
            input_scale=im_input_scale,
            checkpoint=args.checkpoint,
            shuffle=getattr(args, "shuffle", False),
        )
        pre_acts_imdb, embed_mag_trace, model_im, vocab_im = get_imdb_pre_acts(imdb_args, return_model=True)

    mg_pre_acts_list = []
    mg_trace_list = []
    mg_series_indices_list = []
    models_mg = []
    series_mg_list = []
    if use_mg:
        for lam, omg, g_re, g_im, inp_scale in zip(
            mg_lambdas, mg_omegas, mg_gamma_reals, mg_gamma_imags, mg_input_scales
        ):
            mg_args = SimpleNamespace(
                dynamics=args.dynamics,
                num_hidden=args.num_hidden,
                batch_size=args.batch_size,
                seed=args.seed,
                h=args.h,
                alpha=0.04,
                omega=omg,
                gamma=0.01,
                lambda_param=lam,
                gamma_real=g_re,
                gamma_imag=g_im,
                mg_series_length=args.mg_series_length,
                mg_input_length=args.mg_input_length,
                mg_horizon=args.mg_horizon,
                mg_tau=args.mg_tau,
                mg_delta_t=args.mg_delta_t,
                mg_beta=args.mg_beta,
                mg_gamma=args.mg_gamma,
                mg_n=args.mg_n,
                mg_x0=args.mg_x0,
                mg_input_scale=inp_scale,
                random_init=args.random_init,
                checkpoint=args.checkpoint,
            )
            pre_acts_mg, mg_trace, series_indices, model_mg, series_mg = get_mg_pre_acts(
                mg_args, return_model=True
            )
            mg_pre_acts_list.append(pre_acts_mg)
            mg_trace_list.append(mg_trace)
            mg_series_indices_list.append(series_indices)
            models_mg.append(model_mg)
            series_mg_list.append(series_mg)

    sm_series = pre_acts_smnist[:, 0, 0].numpy() if use_sm else None
    im_series = pre_acts_imdb[:, 0, 0].numpy() if use_im else None
    mg_series_list = [pre_acts_mg[:, 0, 0].numpy() for pre_acts_mg in mg_pre_acts_list] if use_mg else []

    output_dir = resolve_output_dir("compare", getattr(args, "output_dir", None))

    base_sm = None
    if use_sm:
        base_sm = simulate_stuart_landau_base(
            lambda_param=sm_lambda,
            omega=sm_omega,
            gamma_real=sm_gamma_real,
            gamma_imag=sm_gamma_imag,
            h=args.h,
            num_steps=sm_series.shape[0],
        )

    base_im = None
    if use_im:
        base_im = simulate_stuart_landau_base(
            lambda_param=im_lambda,
            omega=omega_imdb,
            gamma_real=im_gamma_real,
            gamma_imag=im_gamma_imag,
            h=args.h,
            num_steps=im_series.shape[0],
        )

    base_mg_list = []
    if use_mg:
        for lam, omg, g_re, g_im, mg_series in zip(
            mg_lambdas, mg_omegas, mg_gamma_reals, mg_gamma_imags, mg_series_list
        ):
            base_mg = simulate_stuart_landau_base(
                lambda_param=lam,
                omega=omg,
                gamma_real=g_re,
                gamma_imag=g_im,
                h=args.h,
                num_steps=mg_series.shape[0],
            )
            base_mg_list.append(base_mg)

    t_sm_raw = np.arange(pixel_trace.shape[0]) if use_sm else None
    t_im_raw = np.arange(embed_mag_trace.shape[0]) if use_im else None

    panels = []
    if use_im:
        panels.append("im")
    if use_sm:
        panels.append("sm")
    if use_mg:
        panels.append("mg")

    n_tasks = len(panels)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(7, 3.2 * n_tasks))
    if n_tasks == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "sm":
            color = TASK_COLORS["smnist"]
            ax.plot(t_sm_raw, pixel_trace, color=color, linewidth=1.5, label="Input")
            for x in range(28, int(t_sm_raw[-1]) + 1, 28):
                ax.axvline(x=x, color=color, alpha=0.2, linewidth=0.8)
            ax.set_ylim(_ylim_with_margin(pixel_trace, base_sm))
            ax.set_title("sMNIST", loc="left")
        elif panel == "im":
            color = TASK_COLORS["imdb"]
            ax.plot(t_im_raw, embed_mag_trace, color=color, linewidth=1.5, label="Input")
            ax.set_ylim(_ylim_with_margin(embed_mag_trace, base_im))
            ax.set_title("IMDB", loc="left")
        elif panel == "mg":
            color = TASK_COLORS["mackey_glass"]
            mg_trace = mg_trace_list[0]
            series_mg = series_mg_list[0]
            series_indices = mg_series_indices_list[0]
            base_mg = base_mg_list[0]
            inp_scale = mg_input_scales[0]
            plot_mackey_glass_with_discretization(
                ax,
                series_mg,
                mg_trace,
                series_indices,
                args.mg_input_length,
                args.mg_horizon,
                input_scale=inp_scale,
                color=color,
            )
            ax.set_ylim(_ylim_with_margin(mg_trace, base_mg))
            ax.set_title("Mackey-Glass", loc="left")
        ax.set_ylabel("Input value")
        ax.legend(loc="lower right", fontsize=14)

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Input signals", y=1.02)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "input_signals.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-nonlinearity inputs for HORN on sMNIST, IMDB, and Mackey-Glass")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    smnist_parser = subparsers.add_parser("smnist", help="Analyze sMNIST input")
    smnist_parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    smnist_parser.add_argument("--num-hidden", type=int, default=50)
    smnist_parser.add_argument("--batch-size", type=int, default=64)
    smnist_parser.add_argument("--seed", type=int, default=1)
    smnist_parser.add_argument("--h", type=float, default=1.0)
    smnist_parser.add_argument("--lambda-param", type=float, default=-0.04)
    smnist_parser.add_argument("--omega", type=float, default=0.224)
    smnist_parser.add_argument("--gamma-real", type=float, default=-0.05)
    smnist_parser.add_argument("--gamma-imag", type=float, default=0.1)
    smnist_parser.add_argument("--input-scale", type=float, default=1.0)
    smnist_parser.add_argument("--random-init", type=float, default=None)
    smnist_parser.add_argument("--checkpoint", type=str, default=None)
    smnist_parser.add_argument("--shuffle", action="store_true")
    smnist_parser.add_argument("--num-units-plot", type=int, default=5)

    imdb_parser = subparsers.add_parser("imdb", help="Analyze IMDB input")
    imdb_parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    imdb_parser.add_argument("--num-hidden", type=int, default=9)
    imdb_parser.add_argument("--batch-size", type=int, default=64)
    imdb_parser.add_argument("--seed", type=int, default=1)
    imdb_parser.add_argument("--h", type=float, default=1.0)
    imdb_parser.add_argument("--lambda-param", type=float, default=-0.05)
    imdb_parser.add_argument("--omega", type=float, default=0.224)
    imdb_parser.add_argument("--gamma-real", type=float, default=-0.1)
    imdb_parser.add_argument("--gamma-imag", type=float, default=0.1)
    imdb_parser.add_argument("--embed-dim", type=int, default=100)
    imdb_parser.add_argument("--max-len", type=int, default=175)
    imdb_parser.add_argument("--min-freq", type=int, default=2)
    imdb_parser.add_argument("--dropout", type=float, default=0.3)
    imdb_parser.add_argument("--glove", type=str, default="glove.6B.100d.txt")
    imdb_parser.add_argument("--glove-dir", type=str, default="data/glove")
    imdb_parser.add_argument("--sweep-omega", action="store_true")
    imdb_parser.add_argument("--cache-dir", type=str, default="data/imdb_cache")
    imdb_parser.add_argument("--force-reprocess", action="store_true")
    imdb_parser.add_argument("--random-init", type=float, default=None)
    imdb_parser.add_argument("--checkpoint", type=str, default=None)
    imdb_parser.add_argument("--input-scale", type=float, default=1.0)
    imdb_parser.add_argument("--num-units-plot", type=int, default=5)

    mg_parser = subparsers.add_parser("mg", help="Analyze Mackey-Glass input")
    mg_parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    mg_parser.add_argument("--num-hidden", type=int, default=50)
    mg_parser.add_argument("--batch-size", type=int, default=64)
    mg_parser.add_argument("--seed", type=int, default=1)
    mg_parser.add_argument("--h", type=float, default=1.0)
    mg_parser.add_argument("--alpha", type=float, default=0.04)
    mg_parser.add_argument("--omega", type=float, default=0.224)
    mg_parser.add_argument("--gamma", type=float, default=0.01)
    mg_parser.add_argument("--lambda-param", type=float, default=-0.04)
    mg_parser.add_argument("--gamma-real", type=float, default=-0.05)
    mg_parser.add_argument("--gamma-imag", type=float, default=0.1)
    mg_parser.add_argument("--mg-series-length", type=int, default=20000)
    mg_parser.add_argument("--mg-input-length", type=int, default=50)
    mg_parser.add_argument("--mg-horizon", type=int, default=1)
    mg_parser.add_argument("--mg-tau", type=float, default=17.0)
    mg_parser.add_argument("--mg-delta-t", type=float, default=1.0)
    mg_parser.add_argument("--mg-beta", type=float, default=0.2)
    mg_parser.add_argument("--mg-gamma", type=float, default=0.1)
    mg_parser.add_argument("--mg-n", type=float, default=10.0)
    mg_parser.add_argument("--mg-x0", type=float, default=1.2)
    mg_parser.add_argument("--mg-input-scale", type=float, default=1.0)
    mg_parser.add_argument("--random-init", type=float, default=None)
    mg_parser.add_argument("--checkpoint", type=str, default=None)
    mg_parser.add_argument("--num-units-plot", type=int, default=5)
    mg_parser.add_argument("--val-fraction", type=float, default=0.1)
    mg_parser.add_argument("--test-fraction", type=float, default=0.1)
    mg_parser.add_argument("--encoding-analysis-examples", type=int, default=50)
    mg_parser.add_argument(
        "--encoding-grouping",
        type=str,
        default="all",
        choices=["error", "target", "trend", "all"],
    )

    compare_parser = subparsers.add_parser("compare", help="Compare sMNIST and IMDB input in one plot")
    compare_parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    compare_parser.add_argument("--num-hidden", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=64)
    compare_parser.add_argument("--seed", type=int, default=1)
    compare_parser.add_argument("--h", type=float, default=1.0)
    compare_parser.add_argument("--sm-lambda", type=float, default=-0.04)
    compare_parser.add_argument("--sm-omega", type=float, default=0.224)
    compare_parser.add_argument("--sm-gamma-real", type=float, default=-0.05)
    compare_parser.add_argument("--sm-gamma-imag", type=float, default=0.1)
    compare_parser.add_argument("--sm-input-scale", type=float, default=1.0)
    compare_parser.add_argument("--im-lambda", type=float, default=-0.05)
    compare_parser.add_argument("--im-omega", type=float, default=None)
    compare_parser.add_argument("--im-gamma-real", type=float, default=-0.1)
    compare_parser.add_argument("--im-gamma-imag", type=float, default=0.1)
    compare_parser.add_argument("--im-input-scale", type=float, default=1.0)
    compare_parser.add_argument("--use-sm", action="store_true")
    compare_parser.add_argument("--use-im", action="store_true")
    compare_parser.add_argument("--use-mg", action="store_true")
    compare_parser.add_argument("--mg-lambda", type=float, action="append", default=None)
    compare_parser.add_argument("--mg-omega", type=float, action="append", default=None)
    compare_parser.add_argument("--mg-gamma-real", type=float, action="append", default=None)
    compare_parser.add_argument("--mg-gamma-imag", type=float, action="append", default=None)
    compare_parser.add_argument("--mg-input-scale", type=float, action="append", default=None)
    compare_parser.add_argument("--mg-series-length", type=int, default=20000)
    compare_parser.add_argument("--mg-input-length", type=int, default=800)
    compare_parser.add_argument("--mg-horizon", type=int, default=1)
    compare_parser.add_argument("--mg-tau", type=float, default=17.0)
    compare_parser.add_argument("--mg-delta-t", type=float, default=1.0)
    compare_parser.add_argument("--mg-beta", type=float, default=0.2)
    compare_parser.add_argument("--mg-gamma", type=float, default=0.1)
    compare_parser.add_argument("--mg-n", type=float, default=10.0)
    compare_parser.add_argument("--mg-x0", type=float, default=1.2)
    compare_parser.add_argument("--embed-dim", type=int, default=100)
    compare_parser.add_argument("--max-len", type=int, default=175)
    compare_parser.add_argument("--min-freq", type=int, default=2)
    compare_parser.add_argument("--dropout", type=float, default=0.3)
    compare_parser.add_argument("--glove", type=str, default="glove.6B.100d.txt")
    compare_parser.add_argument("--glove-dir", type=str, default="data/glove")
    compare_parser.add_argument("--sweep-omega", action="store_true")
    compare_parser.add_argument("--cache-dir", type=str, default="data/imdb_cache")
    compare_parser.add_argument("--force-reprocess", action="store_true")
    compare_parser.add_argument("--random-init", type=float, default=None)
    compare_parser.add_argument("--checkpoint", type=str, default=None)
    compare_parser.add_argument("--shuffle", action="store_true")
    compare_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/input_analysis/compare_<timestamp>)",
    )

    args = parser.parse_args()

    if args.dataset == "smnist":
        analyze_smnist(args)
    elif args.dataset == "imdb":
        analyze_imdb(args)
    elif args.dataset == "mg":
        analyze_mg(args)
    elif args.dataset == "compare":
        analyze_compare(args)


if __name__ == "__main__":
    main()


