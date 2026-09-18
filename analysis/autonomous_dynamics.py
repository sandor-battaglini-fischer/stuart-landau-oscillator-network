"""Compare driven vs autonomous oscillator-network dynamics.

Temporarily zeros the input projection (i2h weights and bias) while leaving
recurrent coupling intact, then compares trajectories, forces, spectra, and
λ–γ_r regime maps against the same network with drive.

Examples:
  python analysis/autonomous_dynamics.py --task smnist --run-dir results/smnist/<run>

  python analysis/autonomous_dynamics.py --task smnist --init-only \\
      --lambda-param 0.2 --gamma-real -0.1 --alpha 0.05

  python analysis/autonomous_dynamics.py --task mackey_glass --init-only \\
      --regime-grid --lambda-min -1 --lambda-max 1 --lambda-step 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.memory_capacity import (
    build_model_for_task,
    load_imdb_probe,
    load_mg_probe,
    load_smnist_probe,
    parse_run_log,
    resolve_checkpoint,
)
from utils.device import resolve_device, seed_all
from utils.manifold_dimension_analysis import participation_ratio_from_states
from utils.memory_capacity import dynamics_kind, make_noise_drive, oscillator_core
from utils.model_factory import dho_params_matching_linear_sl
from utils.plotting_utils.style import ifisc_green, thesis_blue, thesis_red
from utils.run_dirs import atomic_write_json

try:
    from utils.plotting_utils.style import apply_style

    apply_style()
except Exception:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "text.usetex": False,
        }
    )

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200})

DRIVEN_COLOR = thesis_blue
AUTO_COLOR = thesis_red
REGIME_ORDER = ("decay", "fixed_point", "limit_cycle", "growing", "blowup")
REGIME_COLORS = {
    "decay": (0.92, 0.92, 0.92),
    "fixed_point": (0.73, 0.83, 0.01),
    "limit_cycle": (0.0, 0.38, 0.68),
    "growing": (0.85, 0.55, 0.13),
    "blowup": (0.64, 0.11, 0.19),
}
TASK_SEQ_DEFAULT = {"smnist": 784, "imdb": 175, "mackey_glass": 400}


def _flag_explicit(name: str) -> bool:
    flag = "--" + name.replace("_", "-")
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


# ---------------------------------------------------------------------------
# Temporary model configuration
# ---------------------------------------------------------------------------

@contextmanager
def zero_input_path(core):
    """Zero i2h weights and bias; restore afterwards. Recurrent path unchanged."""
    weight = core.i2h.weight
    bias = core.i2h.bias
    w_saved = weight.data.clone()
    b_saved = bias.data.clone() if bias is not None else None
    weight.data.zero_()
    if bias is not None:
        bias.data.zero_()
    try:
        yield core
    finally:
        weight.data.copy_(w_saved)
        if bias is not None and b_saved is not None:
            bias.data.copy_(b_saved)


@contextmanager
def override_oscillator_params(core, *, lambda_param=None, gamma_real=None, alpha=None, omega=None):
    """Temporarily rewrite oscillator coefficients; restore afterwards."""
    kind = dynamics_kind(core)
    saved = {}

    if alpha is not None:
        saved["alpha"] = core.alpha
        core.alpha = float(alpha)

    if kind == "dho":
        saved["omega"] = core.omega
        saved["gamma"] = core.gamma
        saved["omega_factor"] = core.omega_factor
        saved["gamma_factor"] = core.gamma_factor
        sl_omega = float(omega) if omega is not None else float(np.sqrt(max(core.omega_factor - core.gamma * core.gamma, 0.0)))
        sl_lambda = float(lambda_param) if lambda_param is not None else float(-core.gamma)
        dho_omega, dho_gamma = dho_params_matching_linear_sl(sl_omega, 0.0, sl_lambda)
        core.omega = dho_omega
        core.gamma = dho_gamma
        core.omega_factor = dho_omega * dho_omega
        core.gamma_factor = 2.0 * dho_gamma
    else:
        if lambda_param is not None:
            saved["lambda_param"] = core.lambda_param.clone()
            core.lambda_param.fill_(float(lambda_param))
        if gamma_real is not None and hasattr(core, "gamma_real"):
            saved["gamma_real"] = core.gamma_real.clone()
            core.gamma_real.fill_(float(gamma_real))
        if omega is not None:
            saved["omega_param"] = core.omega_param.clone()
            core.omega_param.fill_(float(omega))

    try:
        yield core
    finally:
        if "alpha" in saved:
            core.alpha = saved["alpha"]
        if kind == "dho":
            core.omega = saved["omega"]
            core.gamma = saved["gamma"]
            core.omega_factor = saved["omega_factor"]
            core.gamma_factor = saved["gamma_factor"]
        else:
            for name in ("lambda_param", "gamma_real", "omega_param"):
                if name in saved:
                    getattr(core, name).copy_(saved[name])


def apply_runtime_params(core, kwargs):
    """Set dynamical coefficients on a loaded/built model (not restored)."""
    kind = dynamics_kind(core)
    core.h = float(kwargs["h"])
    core.alpha = float(kwargs["alpha"])
    if kind == "dho":
        dho_omega, dho_gamma = dho_params_matching_linear_sl(
            kwargs["omega"], kwargs["gamma"], kwargs["lambda_param"]
        )
        core.omega = dho_omega
        core.gamma = dho_gamma
        core.omega_factor = dho_omega * dho_omega
        core.gamma_factor = 2.0 * dho_gamma
        return
    core.lambda_param.fill_(float(kwargs["lambda_param"]))
    core.omega_param.fill_(float(kwargs["omega"]))
    if hasattr(core, "gamma_real"):
        core.gamma_real.fill_(float(kwargs["gamma_real"]))
        core.gamma_imag.fill_(float(kwargs["gamma_imag"]))


# ---------------------------------------------------------------------------
# Rollout / metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(core, inputs, init_a, init_b):
    """Step the oscillator, recording states and input vs recurrent forces.

    Args:
        inputs: (T, B, F)
        init_a, init_b: (B, N)  — (Re, Im) for SL/LO, (x, y) for DHO
    """
    core.eval()
    kind = dynamics_kind(core)
    num_steps, batch_size, _ = inputs.shape
    num_nodes = core.num_nodes
    device = inputs.device
    dtype = inputs.dtype

    a = init_a.to(device=device, dtype=dtype).clone()
    b = init_b.to(device=device, dtype=dtype).clone()

    rec_a = torch.empty(num_steps, batch_size, num_nodes, device=device, dtype=dtype)
    rec_b = torch.empty_like(rec_a)
    rec_i2h = torch.empty_like(rec_a)
    rec_h2h = torch.empty_like(rec_a)

    for t in range(num_steps):
        input_t = inputs[t]
        i2h_term = core.i2h(input_t)
        if kind == "dho":
            h2h_term = core.gain_rec * core.h2h(b)
        else:
            h2h_term = core.gain_rec * core.h2h(torch.cat([a, b], dim=1))
        rec_i2h[t] = i2h_term
        rec_h2h[t] = h2h_term
        a, b = core.dynamics_step(a, b, input_t)
        rec_a[t] = a
        rec_b[t] = b
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            rec_a[t:] = float("nan")
            rec_b[t:] = float("nan")
            rec_i2h[t:] = float("nan")
            rec_h2h[t:] = float("nan")
            break

    return {
        "a": rec_a.detach().cpu().numpy(),
        "b": rec_b.detach().cpu().numpy(),
        "i2h": rec_i2h.detach().cpu().numpy(),
        "h2h": rec_h2h.detach().cpu().numpy(),
    }


def _radius(traj):
    with np.errstate(over="ignore", invalid="ignore"):
        radius = np.sqrt(np.square(traj["a"]) + np.square(traj["b"]))
    return radius


def _rms(arr, axis=-1):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sqrt(np.mean(np.square(arr), axis=axis))


def _late_slice(num_steps, washout):
    start = int(washout) if washout >= 1 else int(round(float(washout) * num_steps))
    start = min(max(start, 0), max(num_steps - 1, 0))
    return start


def classify_regime(mean_r, r_max, finite, end_growth, motion_ratio, rest=0.05, blowup=50.0):
    if (not finite) or (not np.isfinite(r_max)) or r_max >= blowup:
        return "blowup"
    if (not np.isfinite(mean_r)) or mean_r < rest:
        return "decay"
    if end_growth > 1.4:
        return "growing"
    if motion_ratio > 0.25:
        return "limit_cycle"
    return "fixed_point"


def analytical_sl_regime(lam, gr, eps=1e-12):
    if lam > eps and gr > eps:
        return "blowup"
    if lam > eps and gr < -eps:
        return "limit_cycle"
    if lam < -eps and gr < -eps:
        return "decay"
    if lam < -eps and gr > eps:
        return "subcritical"
    return "marginal"


def summarize_traj(traj, washout=0.5, rest=0.05, blowup=50.0):
    radius = _radius(traj)
    num_steps = radius.shape[0]
    start = _late_slice(num_steps, washout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            finite = bool(np.isfinite(radius).all())
            finite_vals = radius[np.isfinite(radius)]
            r_max = float(finite_vals.max()) if finite_vals.size else float("inf")
            late = radius[start:]
            early = radius[: max(start // 4, 1)]
            mean_r = float(np.nanmean(late))
            std_r = float(np.nanstd(late))
            early_r = float(np.nanmean(early))
            growth_ratio = mean_r / (early_r + 1e-12)

            tail_n = max(num_steps // 10, 5)
            tail_r = float(np.nanmean(radius[-tail_n:]))
            prev_r = float(np.nanmean(radius[-2 * tail_n : -tail_n]))
            end_growth = tail_r / (prev_r + 1e-12)

            late_a = traj["a"][start:]
            late_b = traj["b"][start:]
            motion = float(np.sqrt(np.nanvar(late_a) + np.nanvar(late_b)))
            motion_ratio = motion / (mean_r + 1e-12)

            mean_t = np.nanmean(radius, axis=(1, 2))
            mean_re = np.nanmean(traj["a"], axis=(1, 2))
            i2h_rms = _rms(traj["i2h"], axis=-1)
            h2h_rms = _rms(traj["h2h"], axis=-1)
            late_i2h = float(np.nanmean(i2h_rms[start:]))
            late_h2h = float(np.nanmean(h2h_rms[start:]))
            i2h_rms_t = np.nanmean(i2h_rms, axis=1)
            h2h_rms_t = np.nanmean(h2h_rms, axis=1)

    spectrum = np.abs(np.fft.rfft(np.nan_to_num(mean_re, nan=0.0)))
    freqs = np.fft.rfftfreq(num_steps, d=1.0)
    if spectrum.size > 1:
        peak_idx = int(np.argmax(spectrum[1:]) + 1)
        peak_freq = float(freqs[peak_idx])
        peak_amp = float(spectrum[peak_idx])
    else:
        peak_freq, peak_amp = 0.0, 0.0

    states = np.concatenate([traj["a"], traj["b"]], axis=-1)
    late_states = states[start:].reshape(-1, states.shape[-1])
    late_states = late_states[np.isfinite(late_states).all(axis=1)]
    pr = participation_ratio_from_states(late_states) if len(late_states) > 2 else None

    return {
        "finite": finite,
        "r_max": r_max,
        "late_mean_radius": mean_r,
        "late_std_radius": std_r,
        "early_mean_radius": early_r,
        "growth_ratio": float(growth_ratio),
        "end_growth": float(end_growth),
        "motion_ratio": float(motion_ratio),
        "peak_freq": peak_freq,
        "peak_amp": peak_amp,
        "participation_ratio": None if pr is None else float(pr),
        "late_i2h_rms": late_i2h,
        "late_h2h_rms": late_h2h,
        "regime": classify_regime(
            mean_r, r_max, finite, end_growth, motion_ratio, rest=rest, blowup=blowup
        ),
        "radius_mean_t": mean_t,
        "freqs": freqs,
        "spectrum": spectrum,
        "i2h_rms_t": i2h_rms_t,
        "h2h_rms_t": h2h_rms_t,
        "late_start": start,
    }


def make_init(core, batch_size, random_init, device, dtype, generator):
    num_nodes = core.num_nodes
    if random_init is None or random_init == 0.0:
        init_a = torch.zeros(batch_size, num_nodes, dtype=dtype)
        init_b = torch.zeros(batch_size, num_nodes, dtype=dtype)
    else:
        init_a = torch.randn(batch_size, num_nodes, dtype=dtype, generator=generator) * float(
            random_init
        )
        init_b = torch.randn(batch_size, num_nodes, dtype=dtype, generator=generator) * float(
            random_init
        )
    return init_a.to(device), init_b.to(device)


def run_pair(core, inputs, init_a, init_b):
    driven = rollout(core, inputs, init_a, init_b)
    with zero_input_path(core):
        autonomous = rollout(core, torch.zeros_like(inputs), init_a, init_b)
    return driven, autonomous


# ---------------------------------------------------------------------------
# Model / drive loading
# ---------------------------------------------------------------------------

def merge_kwargs(args, log_params):
    src = log_params or {}

    def pick(cli_name, *log_keys, default=None):
        cli_val = getattr(args, cli_name, None)
        if _flag_explicit(cli_name) and cli_val is not None:
            return cli_val
        if log_params:
            for key in log_keys:
                if key in src and src[key] is not None:
                    return src[key]
        if cli_val is not None:
            return cli_val
        return default

    use_tanh = pick("use_tanh", "use_tanh", default=True)
    if "no_tanh" in src:
        use_tanh = not bool(src["no_tanh"])
    if getattr(args, "no_tanh", False):
        use_tanh = False

    return {
        "dynamics": str(pick("dynamics", "dynamics", default="sl")),
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


def load_drive(task, model, core, kwargs, args, num_steps, batch_size, seed):
    drive = args.drive
    if drive == "auto":
        drive = "task" if args.run_dir and not args.init_only else "noise"

    if drive == "task":
        try:
            if task == "smnist":
                inputs, _ = load_smnist_probe(num_steps, batch_size, seed, shuffle=args.shuffle)
            elif task == "mackey_glass":
                inputs, _ = load_mg_probe(num_steps, batch_size, kwargs, seed)
            elif task == "imdb":
                inputs, _ = load_imdb_probe(model, num_steps, batch_size, kwargs, seed)
            else:
                raise ValueError(task)
            return inputs, "task"
        except Exception as exc:
            print(f"Task drive unavailable ({exc}); falling back to noise")

    inputs, _ = make_noise_drive(
        num_steps,
        batch_size,
        int(core.num_input),
        noise_std=args.noise_std,
        seed=seed,
    )
    return inputs, "noise"


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
        path = os.path.join(PROJECT_ROOT, "results", "autonomous_dynamics", f"{tag}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _unit_indices(num_nodes, num_units):
    if num_nodes <= num_units:
        return list(range(num_nodes))
    return np.linspace(0, num_nodes - 1, num_units, dtype=int).tolist()


def _finite_prefix(*arrays):
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=1)
    if mask.all():
        return slice(None)
    if not mask.any():
        return slice(0, 1)
    last = int(np.argmax(~mask))
    return slice(0, max(last, 1))


def _nanmean(arr, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _drive_trace(inputs, example_idx=0):
    seq = inputs[:, example_idx, :]
    if seq.shape[-1] == 1:
        return seq[:, 0].detach().cpu().numpy()
    return seq.norm(dim=-1).detach().cpu().numpy()


def plot_amplitude_traces(driven, autonomous, inputs, output_dir, num_units, example_idx, late_start):
    r_d = _radius(driven)
    r_a = _radius(autonomous)
    t = np.arange(r_d.shape[0])
    units = _unit_indices(r_d.shape[-1], num_units)
    drive = _drive_trace(inputs, example_idx)
    sl = _finite_prefix(r_d[:, example_idx, :], r_a[:, example_idx, :], drive)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(t[sl], drive[sl], color="0.25", lw=1.2)
    axes[0].set_ylabel("drive")
    axes[0].set_title("input (driven condition)")

    axes[1].plot(t[sl], _nanmean(r_d[:, example_idx, :], axis=-1)[sl], color=DRIVEN_COLOR, label="driven")
    axes[1].plot(t[sl], _nanmean(r_a[:, example_idx, :], axis=-1)[sl], color=AUTO_COLOR, label="autonomous")
    axes[1].axvline(late_start, color="0.5", ls="--", lw=1.0)
    axes[1].set_ylabel(r"mean $|z|$")
    axes[1].set_title("network-mean amplitude")
    axes[1].legend()

    for idx in units:
        axes[2].plot(t[sl], r_d[sl, example_idx, idx], color=DRIVEN_COLOR, alpha=0.85, lw=1.0)
        axes[2].plot(t[sl], r_a[sl, example_idx, idx], color=AUTO_COLOR, alpha=0.85, lw=1.0, ls="--")
    axes[2].set_ylabel(r"$|z_i|$")
    axes[2].set_xlabel("time step")
    axes[2].set_title("per-unit amplitude (solid driven, dashed autonomous)")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "amplitude_traces.png"), transparent=True)
    plt.close(fig)


def plot_phase_portraits(driven, autonomous, output_dir, num_units, example_idx):
    units = _unit_indices(driven["a"].shape[-1], num_units)
    n = len(units)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    for k, idx in enumerate(units):
        ax = axes[k // cols][k % cols]
        xd, yd = driven["a"][:, example_idx, idx], driven["b"][:, example_idx, idx]
        xa, ya = autonomous["a"][:, example_idx, idx], autonomous["b"][:, example_idx, idx]
        sl_d = _finite_prefix(xd, yd)
        sl_a = _finite_prefix(xa, ya)
        ax.plot(xd[sl_d], yd[sl_d], color=DRIVEN_COLOR, lw=0.9, label="driven")
        ax.plot(xa[sl_a], ya[sl_a], color=AUTO_COLOR, lw=0.9, label="autonomous")
        ax.plot(xd[0], yd[0], "k.", ms=4)
        ax.plot(0.0, 0.0, "k+", ms=6, mew=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"unit {idx}")
        ax.set_xlabel(r"$\mathrm{Re}\,z$")
        ax.set_ylabel(r"$\mathrm{Im}\,z$")
        if k == 0:
            ax.legend(fontsize=10)

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase_portraits.png"), transparent=True)
    plt.close(fig)


def plot_mean_field_phase(driven, autonomous, output_dir, example_idx):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for traj, color, label in (
        (driven, DRIVEN_COLOR, "driven"),
        (autonomous, AUTO_COLOR, "autonomous"),
    ):
        x = _nanmean(traj["a"][:, example_idx, :], axis=-1)
        y = _nanmean(traj["b"][:, example_idx, :], axis=-1)
        sl = _finite_prefix(x, y)
        ax.plot(x[sl], y[sl], color=color, lw=1.2, label=label)
        ax.plot(x[0], y[0], color=color, marker="o", ms=4)
    ax.plot(0.0, 0.0, "k+", ms=8, mew=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"mean $\mathrm{Re}\,z$")
    ax.set_ylabel(r"mean $\mathrm{Im}\,z$")
    ax.set_title("mean-field phase portrait")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mean_field_phase.png"), transparent=True)
    plt.close(fig)


def plot_forces(driven_m, auto_m, output_dir, late_start):
    t = np.arange(len(driven_m["i2h_rms_t"]))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(t, driven_m["i2h_rms_t"], color=DRIVEN_COLOR, label=r"driven $i2h$")
    ax.plot(t, driven_m["h2h_rms_t"], color=ifisc_green, label=r"driven $h2h$")
    ax.plot(t, auto_m["h2h_rms_t"], color=AUTO_COLOR, ls="--", label=r"autonomous $h2h$")
    ax.axvline(late_start, color="0.5", ls="--", lw=1.0)
    ax.set_xlabel("time step")
    ax.set_ylabel("RMS force")
    ax.set_title("input vs recurrent contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "force_decomposition.png"), transparent=True)
    plt.close(fig)


def plot_spectra(driven_m, auto_m, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(driven_m["freqs"], driven_m["spectrum"], color=DRIVEN_COLOR, label="driven")
    ax.plot(auto_m["freqs"], auto_m["spectrum"], color=AUTO_COLOR, label="autonomous")
    ax.set_xlabel("frequency (cycles / step)")
    ax.set_ylabel(r"FFT $|\mathrm{mean}\,\mathrm{Re}\,z|$")
    ax.set_title("mean-field real-part spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "amplitude_spectrum.png"), transparent=True)
    plt.close(fig)


def plot_pca_state_space(driven, autonomous, output_dir, late_start, example_idx, max_points=4000):
    def _states(traj):
        s = np.concatenate([traj["a"][:, example_idx], traj["b"][:, example_idx]], axis=-1)
        return s

    s_d = _states(driven)
    s_a = _states(autonomous)
    both = np.concatenate([s_d[late_start:], s_a[late_start:]], axis=0)
    finite = np.isfinite(both).all(axis=1)
    both = both[finite]
    if len(both) < 4:
        return None

    mean = both.mean(axis=0)
    xc = both - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:2]

    def project(s):
        sl = _finite_prefix(s)
        pts = s[sl] - mean
        return pts @ components.T, np.arange(s.shape[0])[sl]

    p_d, t_d = project(s_d)
    p_a, t_a = project(s_a)

    def subsample(pts, times):
        if len(pts) <= max_points:
            return pts, times
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        return pts[idx], times[idx]

    p_d, t_d = subsample(p_d, t_d)
    p_a, t_a = subsample(p_a, t_a)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, pts, times, title in (
        (axes[0], p_d, t_d, "driven"),
        (axes[1], p_a, t_a, "autonomous"),
    ):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=times, s=6, cmap="viridis", alpha=0.75, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=axes, label="time step", fraction=0.03, pad=0.02)
    fig.savefig(os.path.join(output_dir, "pca_state_space.png"), transparent=True)
    plt.close(fig)

    var = np.square(np.linalg.svd(xc, full_matrices=False)[1])
    return float((var[0] + var[1]) / (var.sum() + 1e-12)) if var.size else None


def plot_metrics_bar(driven_m, auto_m, output_dir):
    keys = [
        ("late_mean_radius", r"late mean $|z|$"),
        ("late_std_radius", r"late std $|z|$"),
        ("r_max", r"max $|z|$"),
        ("participation_ratio", "PR"),
        ("late_h2h_rms", r"late $h2h$ RMS"),
    ]
    labels = [lab for _, lab in keys]
    d_vals = [driven_m.get(k) if driven_m.get(k) is not None else 0.0 for k, _ in keys]
    a_vals = [auto_m.get(k) if auto_m.get(k) is not None else 0.0 for k, _ in keys]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width / 2, d_vals, width, color=DRIVEN_COLOR, label="driven")
    ax.bar(x + width / 2, a_vals, width, color=AUTO_COLOR, label="autonomous")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title("summary metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_metrics.png"), transparent=True)
    plt.close(fig)


def _param_edges(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 1:
        step = 0.05 if vals[0] == 0 else abs(vals[0]) * 0.1
        return np.array([vals[0] - step, vals[0] + step])
    mid = 0.5 * (vals[:-1] + vals[1:])
    left = vals[0] - (mid[0] - vals[0])
    right = vals[-1] + (vals[-1] - mid[-1])
    return np.concatenate([[left], mid, [right]])


def plot_regime_heatmap(grid, lambda_vals, gamma_vals, title, path, categorical=True, cmap=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    x_edges = _param_edges(lambda_vals)
    y_edges = _param_edges(gamma_vals)
    if categorical:
        codes = np.full(grid.shape, np.nan, dtype=float)
        for i, name in enumerate(REGIME_ORDER):
            codes[grid == name] = i
        cmap_c = mcolors.ListedColormap([REGIME_COLORS[n] for n in REGIME_ORDER])
        mesh = ax.pcolormesh(x_edges, y_edges, codes, cmap=cmap_c, vmin=-0.5, vmax=len(REGIME_ORDER) - 0.5, shading="flat")
        cbar = fig.colorbar(mesh, ax=ax, ticks=list(range(len(REGIME_ORDER))))
        cbar.ax.set_yticklabels(list(REGIME_ORDER))
    else:
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.asarray(grid, dtype=float),
            cmap=cmap or "viridis",
            shading="flat",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mesh, ax=ax)
    ax.axhline(0.0, color="k", lw=1.1)
    ax.axvline(0.0, color="k", lw=1.1, ls="--")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\gamma_{\mathrm{real}}$")
    ax.set_title(title)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _json_num(val):
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer, np.bool_)):
        val = val.item()
    if isinstance(val, float) and not np.isfinite(val):
        return None
    return val


def _jsonable(metrics):
    skip = {"radius_mean_t", "freqs", "spectrum", "i2h_rms_t", "h2h_rms_t"}
    out = {}
    for key, val in metrics.items():
        if key in skip:
            continue
        out[key] = _json_num(val) if not isinstance(val, str) else val
    return out


def current_sl_params(kwargs):
    return float(kwargs["lambda_param"]), float(kwargs["gamma_real"])


def save_single_comparison(driven, autonomous, inputs, output_dir, args, kwargs, core):
    args.example_idx = int(np.clip(args.example_idx, 0, driven["a"].shape[1] - 1))
    driven_m = summarize_traj(driven, washout=args.washout, rest=args.rest, blowup=args.blowup)
    auto_m = summarize_traj(autonomous, washout=args.washout, rest=args.rest, blowup=args.blowup)
    lam, gr = current_sl_params(kwargs)

    plot_amplitude_traces(driven, autonomous, inputs, output_dir, args.num_units, args.example_idx, driven_m["late_start"])
    plot_phase_portraits(driven, autonomous, output_dir, args.num_units, args.example_idx)
    plot_mean_field_phase(driven, autonomous, output_dir, args.example_idx)
    plot_forces(driven_m, auto_m, output_dir, driven_m["late_start"])
    plot_spectra(driven_m, auto_m, output_dir)
    plot_pca_state_space(driven, autonomous, output_dir, driven_m["late_start"], args.example_idx)
    plot_metrics_bar(driven_m, auto_m, output_dir)

    payload = {
        "lambda": lam,
        "gamma_real": gr,
        "alpha": float(core.alpha),
        "analytical_regime": analytical_sl_regime(lam, gr),
        "driven": _jsonable(driven_m),
        "autonomous": _jsonable(auto_m),
    }
    atomic_write_json(os.path.join(output_dir, "comparison.json"), payload)
    return payload


def arange_inclusive(lo, hi, step):
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    n = int(np.floor((hi - lo) / step + 0.5)) + 1
    vals = [round(float(lo + i * step), 10) for i in range(max(n, 1))]
    if vals[-1] < hi - 1e-12:
        vals.append(round(float(hi), 10))
    elif vals[-1] > hi + 1e-12:
        vals[-1] = round(float(hi), 10)
    out = []
    for v in vals:
        if not out or abs(out[-1] - v) > 1e-12:
            out.append(v)
    return out


def run_regime_grid(core, inputs, init_a, init_b, output_dir, args, kwargs):
    lambda_vals = arange_inclusive(args.lambda_min, args.lambda_max, args.lambda_step)
    gamma_vals = arange_inclusive(args.gamma_real_min, args.gamma_real_max, args.gamma_real_step)
    grid_dir = os.path.join(output_dir, "regime_grid")
    os.makedirs(grid_dir, exist_ok=True)

    n_l, n_g = len(lambda_vals), len(gamma_vals)
    driven_reg = np.empty((n_g, n_l), dtype=object)
    auto_reg = np.empty((n_g, n_l), dtype=object)
    driven_r = np.full((n_g, n_l), np.nan)
    auto_r = np.full((n_g, n_l), np.nan)
    driven_pr = np.full((n_g, n_l), np.nan)
    auto_pr = np.full((n_g, n_l), np.nan)
    expected = np.empty((n_g, n_l), dtype=object)
    rows = []

    total = n_l * n_g
    done = 0
    for j, gr in enumerate(gamma_vals):
        for i, lam in enumerate(lambda_vals):
            with override_oscillator_params(core, lambda_param=lam, gamma_real=gr):
                driven, autonomous = run_pair(core, inputs, init_a, init_b)
            d_m = summarize_traj(driven, washout=args.washout, rest=args.rest, blowup=args.blowup)
            a_m = summarize_traj(autonomous, washout=args.washout, rest=args.rest, blowup=args.blowup)
            driven_reg[j, i] = d_m["regime"]
            auto_reg[j, i] = a_m["regime"]
            driven_r[j, i] = d_m["late_mean_radius"]
            auto_r[j, i] = a_m["late_mean_radius"]
            driven_pr[j, i] = d_m["participation_ratio"] if d_m["participation_ratio"] is not None else np.nan
            auto_pr[j, i] = a_m["participation_ratio"] if a_m["participation_ratio"] is not None else np.nan
            expected[j, i] = analytical_sl_regime(lam, gr)
            rows.append(
                {
                    "lambda": lam,
                    "gamma_real": gr,
                    "analytical": expected[j, i],
                    "driven_regime": d_m["regime"],
                    "autonomous_regime": a_m["regime"],
                    "driven_late_mean_radius": _json_num(d_m["late_mean_radius"]),
                    "autonomous_late_mean_radius": _json_num(a_m["late_mean_radius"]),
                    "driven_pr": _json_num(d_m["participation_ratio"]),
                    "autonomous_pr": _json_num(a_m["participation_ratio"]),
                    "driven_finite": d_m["finite"],
                    "autonomous_finite": a_m["finite"],
                }
            )
            done += 1
            if done == 1 or done == total or done % max(total // 10, 1) == 0:
                print(f"  regime grid {done}/{total}", flush=True)

    plot_regime_heatmap(auto_reg, lambda_vals, gamma_vals, "autonomous regimes", os.path.join(grid_dir, "regime_autonomous.png"))
    plot_regime_heatmap(driven_reg, lambda_vals, gamma_vals, "driven regimes", os.path.join(grid_dir, "regime_driven.png"))

    def _log_radius(arr):
        out = np.array(arr, dtype=float)
        out[~np.isfinite(out)] = args.blowup
        return np.log10(np.clip(out, 1e-6, args.blowup))

    plot_regime_heatmap(
        _log_radius(auto_r),
        lambda_vals,
        gamma_vals,
        r"autonomous $\log_{10}$ late mean $|z|$",
        os.path.join(grid_dir, "late_radius_autonomous.png"),
        categorical=False,
        cmap="magma",
    )
    plot_regime_heatmap(
        _log_radius(driven_r),
        lambda_vals,
        gamma_vals,
        r"driven $\log_{10}$ late mean $|z|$",
        os.path.join(grid_dir, "late_radius_driven.png"),
        categorical=False,
        cmap="magma",
    )
    plot_regime_heatmap(
        auto_pr,
        lambda_vals,
        gamma_vals,
        "autonomous participation ratio",
        os.path.join(grid_dir, "pr_autonomous.png"),
        categorical=False,
        cmap="viridis",
    )
    plot_regime_heatmap(
        driven_pr,
        lambda_vals,
        gamma_vals,
        "driven participation ratio",
        os.path.join(grid_dir, "pr_driven.png"),
        categorical=False,
        cmap="viridis",
    )
    ratio = np.array(driven_r, dtype=float) / (np.array(auto_r, dtype=float) + 1e-8)
    ratio[~np.isfinite(ratio)] = np.nan
    plot_regime_heatmap(
        np.log10(np.clip(ratio, 1e-3, 1e3)),
        lambda_vals,
        gamma_vals,
        r"$\log_{10}(r_{\mathrm{driven}} / r_{\mathrm{auto}})$",
        os.path.join(grid_dir, "radius_ratio_driven_over_auto.png"),
        categorical=False,
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
    )

    agree_auto = sum(
        1
        for r in rows
        if r["analytical"] in ("decay", "limit_cycle", "blowup")
        and (
            (r["analytical"] == "decay" and r["autonomous_regime"] == "decay")
            or (r["analytical"] == "limit_cycle" and r["autonomous_regime"] in ("limit_cycle", "fixed_point"))
            or (r["analytical"] == "blowup" and r["autonomous_regime"] == "blowup")
        )
    )
    comparable = sum(1 for r in rows if r["analytical"] in ("decay", "limit_cycle", "blowup"))
    summary = {
        "lambda_values": lambda_vals,
        "gamma_real_values": gamma_vals,
        "n_cells": total,
        "analytical_agreement_autonomous": (agree_auto / comparable) if comparable else None,
        "cells": rows,
    }
    atomic_write_json(os.path.join(grid_dir, "regime_grid.json"), summary)
    return summary


def write_log(path, payload, args, kwargs, drive_name, output_dir):
    with open(path, "w") as fh:
        fh.write("autonomous vs driven oscillator dynamics\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"task: {args.task}\n")
        fh.write(f"drive: {drive_name}\n")
        fh.write(f"output: {output_dir}\n")
        for key in (
            "dynamics",
            "num_hidden",
            "h",
            "alpha",
            "omega",
            "gamma",
            "lambda_param",
            "gamma_real",
            "gamma_imag",
            "use_tanh",
            "seed",
        ):
            fh.write(f"{key}: {kwargs[key]}\n")
        fh.write(f"random_init: {args.random_init}\n")
        fh.write(f"seq_length: {args.seq_length}\n")
        fh.write("=" * 60 + "\n")
        if "driven" in payload:
            def _fmt(metrics):
                radius = metrics.get("late_mean_radius")
                radius_s = "nan" if radius is None else f"{radius:.4g}"
                return (
                    f"{metrics['regime']}  late |z|={radius_s}  "
                    f"PR={metrics.get('participation_ratio')}"
                )

            fh.write(
                f"analytical: {payload['analytical_regime']}\n"
                f"driven regime: {_fmt(payload['driven'])}\n"
                f"autonomous regime: {_fmt(payload['autonomous'])}\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = resolve_device(args.device)
    seed_all(args.seed, device)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

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

    kwargs = merge_kwargs(args, log_params)
    ckpt = None if args.init_only else resolve_checkpoint(run_dir, args.checkpoint)
    state = None
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    model = build_model_for_task(args.task, kwargs, state=state)
    model.to(device)
    model.eval()
    core = oscillator_core(model)
    apply_runtime_params(core, kwargs)

    if args.seq_length is None:
        args.seq_length = TASK_SEQ_DEFAULT.get(args.task, 400)

    inputs, drive_name = load_drive(
        args.task, model, core, kwargs, args, args.seq_length, args.batch_size, args.seed
    )
    inputs = inputs.to(device=device, dtype=torch.float32)
    init_a, init_b = make_init(core, inputs.shape[1], args.random_init, device, inputs.dtype, generator)

    output_dir = resolve_output_dir(args)
    print(f"Running {args.task}  drive={drive_name}  autonomous=zero i2h  -> {output_dir}", flush=True)

    driven, autonomous = run_pair(core, inputs, init_a, init_b)
    payload = save_single_comparison(driven, autonomous, inputs, output_dir, args, kwargs, core)

    grid_summary = None
    if args.regime_grid:
        if dynamics_kind(core) == "dho":
            print("Note: DHO has no cubic γ_r; grid remaps λ onto damping and still varies γ_r as a no-op on the ODE.")
        print("Sweeping λ × γ_r with frozen weights...", flush=True)
        grid_summary = run_regime_grid(core, inputs, init_a, init_b, output_dir, args, kwargs)

    write_log(os.path.join(output_dir, "log.txt"), payload, args, kwargs, drive_name, output_dir)
    meta = {
        "task": args.task,
        "drive": drive_name,
        "checkpoint": ckpt,
        "run_dir": run_dir,
        "init_only": bool(args.init_only),
        "model": kwargs,
        "comparison": payload,
        "regime_grid": None
        if grid_summary is None
        else {
            "lambda_values": grid_summary["lambda_values"],
            "gamma_real_values": grid_summary["gamma_real_values"],
            "n_cells": grid_summary["n_cells"],
            "analytical_agreement_autonomous": grid_summary["analytical_agreement_autonomous"],
        },
    }
    atomic_write_json(os.path.join(output_dir, "summary.json"), meta)
    print(
        f"driven={payload['driven']['regime']}  "
        f"autonomous={payload['autonomous']['regime']}  "
        f"analytical={payload['analytical_regime']}"
    )
    print(f"Saved results to {output_dir}")
    return meta


def build_parser():
    parser = argparse.ArgumentParser(
        description="Driven vs autonomous (zero i2h, recurrent h2h kept) oscillator dynamics"
    )
    parser.add_argument("--task", required=True, choices=["smnist", "imdb", "mackey_glass"])
    parser.add_argument("--run-dir", type=str, default=None, help="Training run directory (log.txt + checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--init-only", action="store_true", help="Fresh random weights; ignore checkpoints")
    parser.add_argument("--drive", type=str, default="auto", choices=["auto", "task", "noise"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--random-init", type=float, default=0.1, help="Gaussian state init scale (needed off origin)")
    parser.add_argument("--washout", type=float, default=0.5, help="Late-window start as fraction (<1) or steps")
    parser.add_argument("--rest", type=float, default=0.05, help="Radius below this counts as decay to origin")
    parser.add_argument("--blowup", type=float, default=50.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true", help="sMNIST: shuffle pixel order")

    parser.add_argument("--regime-grid", action="store_true", help="Sweep λ × γ_r with frozen weights")
    parser.add_argument("--lambda-min", type=float, default=-1.0)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--lambda-step", type=float, default=0.2)
    parser.add_argument("--gamma-real-min", type=float, default=-0.2)
    parser.add_argument("--gamma-real-max", type=float, default=0.2)
    parser.add_argument("--gamma-real-step", type=float, default=0.04)

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
