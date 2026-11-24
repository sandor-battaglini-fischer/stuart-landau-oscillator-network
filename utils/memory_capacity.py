"""Jaeger short-term memory capacity of a frozen network, using a fresh ridge readout (not h2o)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def oscillator_core(model: torch.nn.Module) -> torch.nn.Module:
    return model.slon if hasattr(model, "slon") else model


def dynamics_kind(core: torch.nn.Module) -> str:
    if hasattr(core, "linear_dynamics") or hasattr(core, "lambda_param"):
        return "sl"
    return "dho"


def state_dim(core: torch.nn.Module) -> int:
    return 2 * int(core.num_nodes)


def record_states(
    core: torch.nn.Module,
    inputs: torch.Tensor,
    random_init: float | None = None,
) -> np.ndarray:
    """(T, B, F) inputs -> (T, B, 2N) float64 hidden states."""
    core.eval()
    with torch.no_grad():
        out = core(inputs, random_init=random_init, record=True)

    kind = dynamics_kind(core)
    if kind == "dho":
        a = out["rec_x_t"].detach().cpu().numpy()
        b = out["rec_y_t"].detach().cpu().numpy()
    else:
        a = out["rec_z_real"].detach().cpu().numpy()
        b = out["rec_z_imag"].detach().cpu().numpy()
    # Model records (B, T, N); MC code uses (T, B, D).
    states = np.concatenate([a, b], axis=-1).astype(np.float64)
    return np.transpose(states, (1, 0, 2))


def make_noise_drive(
    num_steps: int,
    batch_size: int,
    num_input: int,
    noise_std: float = 1.0,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray]:
    """i.i.d. Gaussian scalar probe projected onto a fixed random input direction; returns (inputs, target)."""
    rng = np.random.default_rng(seed)
    target = rng.normal(0.0, noise_std, size=(num_steps, batch_size)).astype(np.float64)
    direction = rng.normal(0.0, 1.0, size=(num_input,)).astype(np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    inputs = target[:, :, None] * direction[None, None, :]
    return torch.from_numpy(inputs.astype(np.float32)), target


def _flatten_aligned(
    states: np.ndarray,
    target: np.ndarray,
    delay: int,
    washout: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``states[t]`` with ``target[t - delay]`` after washout."""
    t_len, batch, dim = states.shape
    t0 = max(washout, delay)
    if t0 >= t_len:
        raise ValueError(
            f"Not enough timesteps for delay={delay}, washout={washout} "
            f"(T={t_len}). Increase --seq-length."
        )
    x = states[t0:].reshape(-1, dim)
    y = target[t0 - delay : t_len - delay].reshape(-1)
    return x, y


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-18:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Closed-form ridge with intercept; returns test predictions."""
    n, d = x_train.shape
    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean()
    xc = x_train - x_mean
    yc = y_train - y_mean
    xtx = xc.T @ xc
    if ridge > 0:
        xtx = xtx + ridge * np.eye(d)
    try:
        w = np.linalg.solve(xtx, xc.T @ yc)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(xtx, xc.T @ yc, rcond=None)[0]
    return (x_test - x_mean) @ w + y_mean


def memory_capacity_spectrum(
    states: np.ndarray,
    target: np.ndarray,
    delays: Sequence[int],
    washout: int = 100,
    ridge: float = 1e-6,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> dict:
    """MC_k for each delay from (T, B, D) states and a (T, B) target."""
    del seed
    if states.ndim != 3:
        raise ValueError(f"states must be (T, B, D), got {states.shape}")
    if target.shape != states.shape[:2]:
        raise ValueError(
            f"target shape {target.shape} incompatible with states {states.shape}"
        )
    if not np.isfinite(states).all():
        raise ValueError(
            "Hidden states contain NaN/Inf; dynamics likely diverged. "
            "Try smaller --noise-std, negative lambda, or a shorter sequence."
        )
    if not np.isfinite(target).all():
        raise ValueError("Probe target contains NaN/Inf")

    mc_k = []
    n_train_list = []
    n_test_list = []

    for delay in delays:
        if delay < 1:
            raise ValueError(f"delays must be >= 1, got {delay}")
        x, y = _flatten_aligned(states, target, delay=delay, washout=washout)
        n = x.shape[0]
        n_train = max(1, int(n * train_fraction))
        n_test = n - n_train
        if n_test < 1:
            raise ValueError(
                f"Need more samples for delay={delay}: n={n}, "
                f"train_fraction={train_fraction}"
            )
        # Temporal split avoids leakage from autocorrelated states.
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n)
        y_hat = _ridge_fit_predict(
            x[train_idx], y[train_idx], x[test_idx], ridge=ridge
        )
        score = _r2_score(y[test_idx], y_hat)
        mc_k.append(max(0.0, min(1.0, score)))
        n_train_list.append(int(n_train))
        n_test_list.append(int(n_test))

    mc_k_arr = np.asarray(mc_k, dtype=np.float64)
    return {
        "delays": [int(d) for d in delays],
        "mc_k": mc_k_arr.tolist(),
        "mc_total": float(mc_k_arr.sum()),
        "state_dim": int(states.shape[-1]),
        "washout": int(washout),
        "ridge": float(ridge),
        "train_fraction": float(train_fraction),
        "n_train": n_train_list,
        "n_test": n_test_list,
    }


def drive_memory_capacity(
    core: torch.nn.Module,
    inputs: torch.Tensor,
    target: np.ndarray,
    max_delay: int,
    washout: int = 100,
    ridge: float = 1e-6,
    train_fraction: float = 0.7,
    random_init: float | None = None,
    seed: int = 0,
    chunk_size: int | None = None,
) -> dict:
    """MC spectrum under `inputs`. With chunk_size, time is split into blocks driven separately and pooled."""
    delays = list(range(1, max_delay + 1))
    t_len = inputs.shape[0]

    if chunk_size is None or chunk_size >= t_len:
        states = record_states(core, inputs, random_init=random_init)
        result = memory_capacity_spectrum(
            states,
            target,
            delays=delays,
            washout=washout,
            ridge=ridge,
            train_fraction=train_fraction,
            seed=seed,
        )
        result["num_steps"] = int(t_len)
        result["batch_size"] = int(inputs.shape[1])
        return result

    state_blocks = []
    target_blocks = []
    for start in range(0, t_len, chunk_size):
        end = min(start + chunk_size, t_len)
        if end - start <= washout + max_delay + 2:
            continue
        block_in = inputs[start:end]
        block_tgt = target[start:end]
        block_states = record_states(core, block_in, random_init=random_init)
        state_blocks.append(block_states)
        target_blocks.append(block_tgt)

    if not state_blocks:
        raise ValueError(
            f"chunk_size={chunk_size} too small for washout={washout} "
            f"and max_delay={max_delay}"
        )

    # Chunks may differ in length if seq_length is not divisible by chunk_size.
    min_t = min(s.shape[0] for s in state_blocks)
    states = np.concatenate([s[:min_t] for s in state_blocks], axis=1)
    target_cat = np.concatenate([t[:min_t] for t in target_blocks], axis=1)

    result = memory_capacity_spectrum(
        states,
        target_cat,
        delays=delays,
        washout=washout,
        ridge=ridge,
        train_fraction=train_fraction,
        seed=seed,
    )
    result["num_steps"] = int(min_t)
    result["batch_size"] = int(states.shape[1])
    result["chunk_size"] = int(chunk_size)
    return result


def effective_memory_horizon(mc_k: Sequence[float], threshold: float = 0.5) -> int | None:
    """Smallest delay where MC_k falls below ``threshold * MC_1`` (if MC_1 > 0)."""
    vals = list(mc_k)
    if not vals or vals[0] <= 0:
        return None
    cutoff = threshold * vals[0]
    for k, v in enumerate(vals, start=1):
        if v < cutoff:
            return k
    return len(vals)
