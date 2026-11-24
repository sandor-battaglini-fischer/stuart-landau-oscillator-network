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
from matplotlib.lines import Line2D
import numpy as np
import torch
import torchvision

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()

mg_cmap = mcolors.LinearSegmentedColormap.from_list("mg_cmap", [(0.5, 0.6, 0.0), ifisc_green])

from models import HORN
from training.train_imdb import SLONWithEmbedding, download_glove, load_glove_vectors, load_imdb_data
from training.train_mackey_glass import MackeyGlassDataset, build_dataloaders, generate_mackey_glass
from utils.model_factory import build_oscillator
from utils.plotting_utils.mackey_glass_encoding import plot_mackey_glass_encoding_analysis_from_loader

TASK_COLORS = {
    "smnist": "black",
    "imdb": "black",
    "mackey_glass": "black",
}

HORIZON_COLORS = {
    1: thesis_blue,
    25: thesis_red,
}


def horizon_color(h):
    h = int(h)
    if h in HORIZON_COLORS:
        return HORIZON_COLORS[h]
    palette = [thesis_blue, thesis_red, ifisc_green]
    return palette[h % len(palette)]


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


def get_imdb_pre_acts(args, model=None, return_model=False):
    torch.manual_seed(args.seed)
    if args.omega is None:
        args.omega = 2 * math.pi / (args.max_len * args.h)

    vocab, train_data, _ = load_imdb_data(args.cache_dir, args.min_freq, args.max_len, args.force_reprocess)
    vocab_size = len(vocab)
    pad_idx = vocab.word2idx["<pad>"]
    dim_input = args.embed_dim
    dim_output = 2

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
        return pre_acts, embed_mag_trace, model
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
    view_length=None,
    marker_size=12,
    show_target=False,
    extend_to=None,
    horizons=None,
):
    scaled_series = np.asarray(series) * input_scale
    series_indices = np.asarray(series_indices, dtype=np.int64)
    mg_trace = np.asarray(mg_trace)
    last_idx = int(series_indices[-1])
    if horizons is None:
        horizons = [int(horizon)] if show_target else []
    else:
        horizons = [int(h) for h in horizons]
    extra = int(extend_to) if extend_to is not None else (max(horizons) if horizons else 0)
    window_end = min(len(scaled_series) - 1, last_idx + extra)
    if view_length is not None:
        window_start = max(0, last_idx - int(view_length) + 1)
    else:
        window_start = int(np.unique(series_indices)[0])

    past_idx = np.arange(window_start, last_idx + 1)
    ax.plot(
        past_idx,
        scaled_series[past_idx],
        color=color,
        linewidth=1.5,
        zorder=3,
        label="time series",
    )
    ax.scatter(
        past_idx,
        scaled_series[past_idx],
        color=color,
        s=marker_size,
        alpha=0.85,
        zorder=4,
    )
    visible = (series_indices >= window_start) & (series_indices <= last_idx)
    if np.any(visible):
        ax.plot(
            series_indices[visible],
            mg_trace[visible],
            color=color,
            linewidth=1.5,
            zorder=5,
        )
    if extra > 0:
        future_idx = np.arange(last_idx, window_end + 1)
        ax.plot(
            future_idx,
            scaled_series[future_idx],
            color="0.55",
            linewidth=1.5,
            zorder=3,
        )
        if len(future_idx) > 1:
            ax.scatter(
                future_idx[1:],
                scaled_series[future_idx[1:]],
                color="0.55",
                s=marker_size,
                alpha=0.9,
                zorder=4,
            )
        ax.axvline(last_idx, color="0.35", linewidth=1.4, zorder=6)
        for h in sorted(horizons, reverse=True):
            target_idx = last_idx + h
            if target_idx >= len(scaled_series):
                continue
            h_color = horizon_color(h)
            seg = np.arange(last_idx, target_idx + 1)
            ax.plot(
                seg,
                scaled_series[seg],
                color=h_color,
                linewidth=2.4,
                zorder=7,
                label=rf"$H={h}$",
            )
            ax.scatter(
                seg[1:],
                scaled_series[seg[1:]],
                color=h_color,
                s=marker_size,
                zorder=8,
            )
        ax.set_xlim(window_start - 0.5, window_end + 0.5)
    return window_start, window_end


def plot_mackey_glass_delay(
    ax,
    series,
    horizons,
    input_scale=1.0,
):
    scaled = np.asarray(series) * input_scale
    horizons = [int(h) for h in horizons if 1 <= int(h) < len(scaled)]
    if not horizons:
        return
    pad = 0.05 * (scaled.max() - scaled.min() if scaled.max() > scaled.min() else 1.0)
    for h in sorted(horizons, reverse=True):
        h_color = horizon_color(h)
        ax.plot(
            scaled[:-h],
            scaled[h:],
            color=h_color,
            lw=0.1,
            zorder=1,
        )
    ax.set_xlabel(r"$x(t-H)$")
    ax.set_ylabel(r"$x(t)$")
    ax.set_xlim(scaled.min() - pad, scaled.max() + pad)
    ax.set_ylim(scaled.min() - pad, scaled.max() + pad)
    ax.set_title("Delay plot", loc="left")
    handles = [
        Line2D([0], [0], color=horizon_color(h), lw=3.0, label=rf"$H={h}$")
        for h in horizons
    ]
    ax.legend(handles=handles, loc="best", fontsize=12, handlelength=2.4)


def plot_mg_timeseries_and_delay(
    series,
    mg_trace,
    series_indices,
    input_length,
    horizons,
    input_scale,
    output_path,
    color=None,
    view_length=80,
    zoom_length=22,
):
    if color is None:
        color = TASK_COLORS["mackey_glass"]
    horizons = [int(h) for h in horizons]
    max_h = max(horizons)
    fig, axes = plt.subplots(
        1, 3, figsize=(10.0, 3.55), gridspec_kw={"width_ratios": [1.0, 1.35, 1.05]}
    )
    ax_ts, ax_zoom, ax_delay = axes

    plot_mackey_glass_with_discretization(
        ax_ts,
        series,
        mg_trace,
        series_indices,
        input_length,
        horizons[0],
        input_scale=input_scale,
        color=color,
        view_length=view_length,
        marker_size=9,
        show_target=True,
        extend_to=2 * max_h,
        horizons=horizons,
    )
    ax_ts.set_title("Mackey–Glass", loc="left")
    ax_ts.set_ylabel("Input value")
    ax_ts.set_xlabel("Time step")
    ax_ts.legend(loc="best", fontsize=12)

    zoom_start, zoom_end = plot_mackey_glass_with_discretization(
        ax_zoom,
        series,
        mg_trace,
        series_indices,
        input_length,
        horizons[0],
        input_scale=input_scale,
        color=color,
        view_length=zoom_length,
        marker_size=30,
        show_target=True,
        extend_to=2 * max_h,
        horizons=horizons,
    )
    ax_zoom.set_xlabel("Time step")
    ax_zoom.set_title(r"Zoom ($\Delta t=1$)", loc="left")
    ax_zoom.set_xticks(
        np.unique(np.linspace(zoom_start, zoom_end, 3).astype(int))
    )

    plot_mackey_glass_delay(
        ax_delay,
        series,
        horizons,
        input_scale=input_scale,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


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

    plot_mg_timeseries_and_delay(
        series,
        mg_trace,
        series_indices,
        args.mg_input_length,
        [1, 25],
        args.mg_input_scale,
        os.path.join(output_dir, "mg_timeseries_delay.png"),
        view_length=getattr(args, "mg_view_length", 80),
        zoom_length=getattr(args, "mg_zoom_length", 22),
    )

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

    if getattr(args, "mg_horizon", None) is None:
        mg_horizons = [1, 25]
    else:
        mg_horizons = args.mg_horizon if isinstance(args.mg_horizon, list) else [args.mg_horizon]
    mg_horizons = [int(h) for h in mg_horizons]
    mg_view_length = getattr(args, "mg_view_length", 100)
    mg_zoom_length = getattr(args, "mg_zoom_length", 22)

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
            cache_dir=args.cache_dir,
            force_reprocess=args.force_reprocess,
            random_init=args.random_init,
            input_scale=im_input_scale,
            checkpoint=args.checkpoint,
            shuffle=getattr(args, "shuffle", False),
        )
        pre_acts_imdb, embed_mag_trace, model_im = get_imdb_pre_acts(imdb_args, return_model=True)

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
                mg_horizon=mg_horizons[0],
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

    output_dir = resolve_output_dir("compare", getattr(args, "output_dir", None))

    t_sm_raw = np.arange(pixel_trace.shape[0]) if use_sm else None
    t_im_raw = np.arange(embed_mag_trace.shape[0]) if use_im else None

    n_rows = int(use_im) + int(use_sm) + int(use_mg)
    fig = plt.figure(figsize=(7.2, 3.05 * max(n_rows, 1)), layout="constrained")
    gs = fig.add_gridspec(n_rows, 2)
    row = 0
    last_time_ax = None

    if use_im:
        ax = fig.add_subplot(gs[row, :])
        color = TASK_COLORS["imdb"]
        ax.plot(t_im_raw, embed_mag_trace, color=color, linewidth=1.5, label="Input")
        ax.set_ylim(_ylim_with_margin(embed_mag_trace))
        ax.set_title("IMDB", loc="left")
        ax.set_ylabel("Input value")
        ax.legend(loc="lower right", fontsize=14)
        last_time_ax = ax
        row += 1

    if use_sm:
        ax = fig.add_subplot(gs[row, :])
        color = TASK_COLORS["smnist"]
        ax.plot(t_sm_raw, pixel_trace, color=color, linewidth=1.5, label="Input")
        for x in range(28, int(t_sm_raw[-1]) + 1, 28):
            ax.axvline(x=x, color=color, alpha=0.2, linewidth=0.8)
        ax.set_ylim(_ylim_with_margin(pixel_trace))
        ax.set_title("sMNIST", loc="left")
        ax.set_ylabel("Input value")
        ax.legend(loc="lower right", fontsize=14)
        last_time_ax = ax
        row += 1

    if use_mg:
        ax_mg = fig.add_subplot(gs[row, 0])
        ax_delay = fig.add_subplot(gs[row, 1])
        color = TASK_COLORS["mackey_glass"]
        mg_trace = mg_trace_list[0]
        series_mg = series_mg_list[0]
        series_indices = mg_series_indices_list[0]
        inp_scale = mg_input_scales[0]
        max_h = max(mg_horizons)
        plot_mackey_glass_with_discretization(
            ax_mg,
            series_mg,
            mg_trace,
            series_indices,
            args.mg_input_length,
            mg_horizons[0],
            input_scale=inp_scale,
            color=color,
            view_length=mg_view_length,
            marker_size=16,
            show_target=True,
            extend_to=2 * max_h,
            horizons=mg_horizons,
        )
        ax_mg.set_title("Mackey–Glass", loc="left")
        last_idx = int(np.asarray(series_indices)[-1])
        view_slice = np.asarray(series_mg)[
            max(0, last_idx - mg_view_length + 1) : last_idx + 2 * max_h + 1
        ] * inp_scale
        ax_mg.set_ylim(_ylim_with_margin(view_slice))
        ax_mg.set_ylabel("Input value")
        ax_mg.set_xlabel("Time step")
        ax_mg.legend(loc="lower left", fontsize=12)
        plot_mackey_glass_delay(
            ax_delay,
            series_mg,
            mg_horizons,
            input_scale=inp_scale,
        )
        last_time_ax = ax_mg

    if last_time_ax is not None and not use_mg:
        last_time_ax.set_xlabel("Time step")
    fig.suptitle("Input signals")
    plot_path = os.path.join(output_dir, "input_signals.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    if use_mg:
        plot_mg_timeseries_and_delay(
            series_mg_list[0],
            mg_trace_list[0],
            mg_series_indices_list[0],
            args.mg_input_length,
            mg_horizons,
            mg_input_scales[0],
            os.path.join(output_dir, "mg_timeseries_delay.png"),
            view_length=min(mg_view_length, 80),
            zoom_length=mg_zoom_length,
        )


def main():
    parser = argparse.ArgumentParser(description="Pre-nonlinearity input statistics on sMNIST, IMDb and Mackey-Glass")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    smnist_parser = subparsers.add_parser("smnist", help="Analyze sMNIST input")
    smnist_parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    smnist_parser.add_argument("--num-hidden", type=int, default=50)
    smnist_parser.add_argument("--batch-size", type=int, default=64)
    smnist_parser.add_argument("--seed", type=int, default=1)
    smnist_parser.add_argument("--h", type=float, default=1.0)
    smnist_parser.add_argument("--lambda-param", type=float, default=-0.04)
    smnist_parser.add_argument("--alpha", type=float, default=0.05)
    smnist_parser.add_argument("--omega", type=float, default=0.224)
    smnist_parser.add_argument("--gamma", type=float, default=0.01)
    smnist_parser.add_argument("--gamma-real", type=float, default=-0.05)
    smnist_parser.add_argument("--gamma-imag", type=float, default=0.1)
    smnist_parser.add_argument("--input-scale", type=float, default=1.0)
    smnist_parser.add_argument("--random-init", type=float, default=None)
    smnist_parser.add_argument("--checkpoint", type=str, default=None)
    smnist_parser.add_argument("--shuffle", action="store_true")
    smnist_parser.add_argument("--num-units-plot", type=int, default=5)
    smnist_parser.add_argument("--output-dir", type=str, default=None)

    imdb_parser = subparsers.add_parser("imdb", help="Analyze IMDB input")
    imdb_parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    imdb_parser.add_argument("--num-hidden", type=int, default=9)
    imdb_parser.add_argument("--batch-size", type=int, default=64)
    imdb_parser.add_argument("--seed", type=int, default=1)
    imdb_parser.add_argument("--h", type=float, default=1.0)
    imdb_parser.add_argument("--lambda-param", type=float, default=-0.05)
    imdb_parser.add_argument("--alpha", type=float, default=0.5)
    imdb_parser.add_argument("--omega", type=float, default=None, help="default: one period over --max-len")
    imdb_parser.add_argument("--gamma", type=float, default=0.01)
    imdb_parser.add_argument("--gamma-real", type=float, default=-0.1)
    imdb_parser.add_argument("--gamma-imag", type=float, default=0.1)
    imdb_parser.add_argument("--embed-dim", type=int, default=100)
    imdb_parser.add_argument("--max-len", type=int, default=175)
    imdb_parser.add_argument("--min-freq", type=int, default=2)
    imdb_parser.add_argument("--dropout", type=float, default=0.3)
    imdb_parser.add_argument("--glove", type=str, default="glove.6B.100d.txt")
    imdb_parser.add_argument("--glove-dir", type=str, default="data/glove")
    imdb_parser.add_argument("--cache-dir", type=str, default="data/imdb_cache")
    imdb_parser.add_argument("--force-reprocess", action="store_true")
    imdb_parser.add_argument("--random-init", type=float, default=None)
    imdb_parser.add_argument("--checkpoint", type=str, default=None)
    imdb_parser.add_argument("--input-scale", type=float, default=1.0)
    imdb_parser.add_argument("--num-units-plot", type=int, default=5)
    imdb_parser.add_argument("--output-dir", type=str, default=None)

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
    mg_parser.add_argument("--mg-view-length", type=int, default=80)
    mg_parser.add_argument("--mg-zoom-length", type=int, default=22)
    mg_parser.add_argument("--mg-tau", type=float, default=17.0)
    mg_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/input_analysis/mg_<timestamp>)",
    )
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
    compare_parser.add_argument("--mg-horizon", type=int, action="append", default=None)
    compare_parser.add_argument("--mg-view-length", type=int, default=100)
    compare_parser.add_argument("--mg-zoom-length", type=int, default=22)
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


