import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .signal_stages import (
    _epoch_filename,
    _unit_indices,
    get_slon_core,
    prepare_imdb_sequence,
)
from .style import thesis_blue, thesis_red, ifisc_green


def find_class_examples_batch(data_loader, labels=(0, 1), num_per_class=30, max_batches=100):
    found = {label: [] for label in labels}
    for batch_idx, (token_ids, batch_labels) in enumerate(data_loader):
        for label in labels:
            if len(found[label]) >= num_per_class:
                continue
            matches = (batch_labels == label).nonzero(as_tuple=True)[0]
            for idx in matches:
                if len(found[label]) >= num_per_class:
                    break
                found[label].append(token_ids[idx : idx + 1].clone())
        if all(len(found[label]) >= num_per_class for label in labels):
            break
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    missing = [label for label in labels if len(found[label]) < num_per_class]
    if missing:
        raise ValueError(
            f"Could not collect {num_per_class} examples for labels {missing}; "
            f"got {{{', '.join(f'{l}: {len(found[l])}' for l in labels)}}}"
        )
    return {label: torch.cat(found[label], dim=0) for label in labels}


@torch.no_grad()
def trace_model_signal_stages_batch(model, inputs, input_mode="norm"):
    slon = get_slon_core(model)
    num_timesteps, batch_size = inputs.size(0), inputs.size(1)

    z_real = torch.zeros(batch_size, slon.num_nodes, device=inputs.device, dtype=inputs.dtype)
    z_imag = torch.zeros(batch_size, slon.num_nodes, device=inputs.device, dtype=inputs.dtype)

    projected_inputs = []
    pre_activations = []
    z_real_trace = []
    z_mag_trace = []

    for t in range(num_timesteps):
        input_t = inputs[t]
        projected = slon.i2h(input_t)
        pre_act = projected + slon.gain_rec * slon.h2h(z_real)
        z_real, z_imag = slon.dynamics_step(z_real, z_imag, input_t)

        projected_inputs.append(projected.detach().cpu())
        pre_activations.append(pre_act.detach().cpu())
        z_real_trace.append(z_real.detach().cpu())
        z_mag_trace.append(torch.sqrt(z_real ** 2 + z_imag ** 2).detach().cpu())

    logits = slon.h2o(z_real).detach().cpu()
    if input_mode == "norm":
        raw_input = inputs.norm(dim=-1).detach().cpu().numpy()
    else:
        raw_input = inputs[:, :, 0].detach().cpu().numpy()

    return {
        "raw_input": raw_input,
        "projected_input": torch.stack(projected_inputs).numpy(),
        "pre_activation": torch.stack(pre_activations).numpy(),
        "z_real": torch.stack(z_real_trace).numpy(),
        "z_magnitude": torch.stack(z_mag_trace).numpy(),
        "logits": logits.numpy(),
    }


def _mean_over_units(series, unit_indices):
    return series[:, :, unit_indices].mean(axis=-1)


def _rfft_amplitude(trace):
    return np.abs(np.fft.rfft(trace, axis=0))


def _spectral_centroid(amps, freqs):
    power = amps ** 2
    total = power.sum(axis=0, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return (freqs[:, None] * power).sum(axis=0) / total.sum(axis=0)


def _cohens_d(group_a, group_b):
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) < 2 or len(b) < 2:
        return float(np.mean(a) - np.mean(b))
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _logit_margin(logits):
    if logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    return logits.max(axis=-1)


def _example_features(stages, unit_indices):
    raw = stages["raw_input"]
    pre = _mean_over_units(stages["pre_activation"], unit_indices)
    z_real = _mean_over_units(stages["z_real"], unit_indices)
    z_mag = _mean_over_units(stages["z_magnitude"], unit_indices)
    freqs = np.fft.rfftfreq(raw.shape[0], d=1.0)

    features = {
        "mean_input_norm": raw.mean(axis=0),
        "std_input_norm": raw.std(axis=0),
        "mean_pre_activation": pre.mean(axis=0),
        "std_pre_activation": pre.std(axis=0),
        "mean_z_real": z_real.mean(axis=0),
        "std_z_real": z_real.std(axis=0),
        "mean_z_magnitude": z_mag.mean(axis=0),
        "std_z_magnitude": z_mag.std(axis=0),
        "final_z_real": z_real[-1],
        "final_z_magnitude": z_mag[-1],
        "max_z_magnitude": z_mag.max(axis=0),
        "logit_margin": _logit_margin(stages["logits"]),
        "positive_prob": torch.softmax(torch.tensor(stages["logits"]), dim=-1).numpy()[:, 1],
    }

    for name, trace in [
        ("input", raw),
        ("pre_activation", pre),
        ("z_real", z_real),
        ("z_magnitude", z_mag),
    ]:
        amps = _rfft_amplitude(trace)
        features[f"spectral_centroid_{name}"] = _spectral_centroid(amps, freqs)
        low = amps[freqs <= 0.1].sum(axis=0)
        high = amps[freqs > 0.1].sum(axis=0)
        features[f"low_high_power_ratio_{name}"] = low / np.maximum(high, 1e-8)

    return features


def _mean_std_spectrum(trace, unit_indices):
    if trace.ndim == 2:
        amps = _rfft_amplitude(trace)
    else:
        mean_trace = trace[:, :, unit_indices].mean(axis=-1)
        amps = _rfft_amplitude(mean_trace)
    return amps.mean(axis=1), amps.std(axis=1)


def _plot_mean_std_band(ax, freqs, mean_amp, std_amp, color, label):
    ax.plot(freqs, mean_amp, color=color, linewidth=1.8, label=label)
    ax.fill_between(freqs, mean_amp - std_amp, mean_amp + std_amp, color=color, alpha=0.2)


def plot_sentiment_encoding_analysis(
    neg_stages,
    pos_stages,
    output_dir,
    epoch,
    num_units_plot=5,
):
    os.makedirs(output_dir, exist_ok=True)

    num_timesteps = neg_stages["raw_input"].shape[0]
    num_nodes = neg_stages["pre_activation"].shape[-1]
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    freqs = np.fft.rfftfreq(num_timesteps, d=1.0)

    neg_features = _example_features(neg_stages, unit_indices)
    pos_features = _example_features(pos_stages, unit_indices)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    stage_specs = [
        ("raw_input", "embedding norm", "input"),
        ("pre_activation", r"pre-activation $u(t)$", "pre_activation"),
        ("z_real", r"$\Re(z(t))$", "z_real"),
        ("z_magnitude", r"$|z(t)|$", "z_magnitude"),
    ]
    for row, (key, ylabel, prefix) in enumerate(stage_specs):
        neg_mean, neg_std = _mean_std_spectrum(neg_stages[key], unit_indices)
        pos_mean, pos_std = _mean_std_spectrum(pos_stages[key], unit_indices)
        _plot_mean_std_band(axes[row, 0], freqs, neg_mean, neg_std, thesis_red, "negative")
        _plot_mean_std_band(axes[row, 0], freqs, pos_mean, pos_std, ifisc_green, "positive")
        axes[row, 0].set_ylabel("amplitude")
        axes[row, 0].set_title(f"{ylabel}: frequency spectrum")
        axes[row, 0].legend(loc="upper right", fontsize=10)

        diff = pos_mean - neg_mean
        axes[row, 1].plot(freqs, diff, color=thesis_blue, linewidth=1.8)
        axes[row, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axes[row, 1].fill_between(freqs, 0.0, diff, where=diff >= 0, color=ifisc_green, alpha=0.2)
        axes[row, 1].fill_between(freqs, 0.0, diff, where=diff < 0, color=thesis_red, alpha=0.2)
        axes[row, 1].set_title(f"{ylabel}: spectral difference (pos $-$ neg)")
        axes[row, 1].set_ylabel("amplitude diff")

    axes[-1, 0].set_xlabel("frequency")
    axes[-1, 1].set_xlabel("frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename("sentiment_encoding_spectra", epoch)), transparent=True)
    plt.close(fig)

    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    t = np.arange(num_timesteps)
    neg_raw_mean = neg_stages["raw_input"].mean(axis=1)
    pos_raw_mean = pos_stages["raw_input"].mean(axis=1)
    neg_raw_std = neg_stages["raw_input"].std(axis=1)
    pos_raw_std = pos_stages["raw_input"].std(axis=1)

    axes2[0, 0].plot(t, neg_raw_mean, color=thesis_red, linewidth=2.0, label="mean")
    axes2[0, 0].fill_between(t, neg_raw_mean - neg_raw_std, neg_raw_mean + neg_raw_std, color=thesis_red, alpha=0.2)
    axes2[0, 0].set_title("negative: input norm")
    axes2[0, 0].set_ylabel("embedding norm")
    axes2[0, 1].plot(t, pos_raw_mean, color=ifisc_green, linewidth=2.0)
    axes2[0, 1].fill_between(t, pos_raw_mean - pos_raw_std, pos_raw_mean + pos_raw_std, color=ifisc_green, alpha=0.2)
    axes2[0, 1].set_title("positive: input norm")

    neg_zm = _mean_over_units(neg_stages["z_magnitude"], unit_indices)
    pos_zm = _mean_over_units(pos_stages["z_magnitude"], unit_indices)
    axes2[1, 0].plot(t, neg_zm.mean(axis=1), color=thesis_red, linewidth=2.0)
    axes2[1, 0].fill_between(
        t,
        (neg_zm.mean(axis=1) - neg_zm.std(axis=1)),
        (neg_zm.mean(axis=1) + neg_zm.std(axis=1)),
        color=thesis_red,
        alpha=0.2,
    )
    axes2[1, 0].set_title("negative: $|z(t)|$")
    axes2[1, 0].set_ylabel(r"$|z|$")
    axes2[1, 1].plot(t, pos_zm.mean(axis=1), color=ifisc_green, linewidth=2.0)
    axes2[1, 1].fill_between(
        t,
        (pos_zm.mean(axis=1) - pos_zm.std(axis=1)),
        (pos_zm.mean(axis=1) + pos_zm.std(axis=1)),
        color=ifisc_green,
        alpha=0.2,
    )
    axes2[1, 1].set_title("positive: $|z(t)|$")

    neg_zr = _mean_over_units(neg_stages["z_real"], unit_indices)
    pos_zr = _mean_over_units(pos_stages["z_real"], unit_indices)
    axes2[2, 0].plot(t, neg_zr.mean(axis=1), color=thesis_red, linewidth=2.0)
    axes2[2, 0].fill_between(
        t,
        (neg_zr.mean(axis=1) - neg_zr.std(axis=1)),
        (neg_zr.mean(axis=1) + neg_zr.std(axis=1)),
        color=thesis_red,
        alpha=0.2,
    )
    axes2[2, 0].set_title("negative: $\Re(z)$ before readout")
    axes2[2, 0].set_ylabel(r"$\Re(z)$")
    axes2[2, 0].set_xlabel("time step")
    axes2[2, 1].plot(t, pos_zr.mean(axis=1), color=ifisc_green, linewidth=2.0)
    axes2[2, 1].fill_between(
        t,
        (pos_zr.mean(axis=1) - pos_zr.std(axis=1)),
        (pos_zr.mean(axis=1) + pos_zr.std(axis=1)),
        color=ifisc_green,
        alpha=0.2,
    )
    axes2[2, 1].set_title("positive: $\Re(z)$ before readout")
    axes2[2, 1].set_xlabel("time step")
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, _epoch_filename("sentiment_encoding_magnitude", epoch)), transparent=True)
    plt.close(fig2)

    n_neg = neg_stages["logits"].shape[0]
    n_pos = pos_stages["logits"].shape[0]
    n_examples = min(n_neg, n_pos)
    fig3, axes3 = plt.subplots(4, n_examples, figsize=(2.8 * n_examples, 10), sharex=True, squeeze=False)
    row_specs = [
        (neg_stages["raw_input"], thesis_red, "negative: input norm"),
        (_mean_over_units(neg_stages["z_magnitude"], unit_indices), thesis_red, "negative: $|z(t)|$"),
        (pos_stages["raw_input"], ifisc_green, "positive: input norm"),
        (_mean_over_units(pos_stages["z_magnitude"], unit_indices), ifisc_green, "positive: $|z(t)|$"),
    ]
    for row, (data, color, row_title) in enumerate(row_specs):
        for col in range(n_examples):
            ax = axes3[row, col]
            ax.plot(data[:, col], color=color, linewidth=1.3)
            if col == 0:
                ax.set_ylabel(row_title)
            ax.set_title(f"example {col}")
        axes3[row, -1].set_xlabel("time step")
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, _epoch_filename("sentiment_encoding_examples", epoch)), transparent=True)
    plt.close(fig3)

    fig3b, axes3b = plt.subplots(2, n_examples, figsize=(2.8 * n_examples, 6), sharex=True, squeeze=False)
    for col in range(n_examples):
        axes3b[0, col].plot(_mean_over_units(neg_stages["z_real"], unit_indices)[:, col], color=thesis_red, linewidth=1.3)
        axes3b[1, col].plot(_mean_over_units(pos_stages["z_real"], unit_indices)[:, col], color=ifisc_green, linewidth=1.3)
        axes3b[0, col].set_title(f"negative ex. {col}")
        axes3b[1, col].set_title(f"positive ex. {col}")
        axes3b[1, col].set_xlabel("time step")
    axes3b[0, 0].set_ylabel(r"$\Re(z)$ before readout")
    axes3b[1, 0].set_ylabel(r"$\Re(z)$ before readout")
    plt.tight_layout()
    fig3b.savefig(os.path.join(output_dir, _epoch_filename("sentiment_encoding_zreal_examples", epoch)), transparent=True)
    plt.close(fig3b)

    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
    scatter_specs = [
        ("final_z_magnitude", "final $|z|$", axes4[0, 0]),
        ("mean_z_magnitude", "mean $|z|$", axes4[0, 1]),
        ("final_z_real", "final $\Re(z)$", axes4[0, 2]),
        ("spectral_centroid_z_magnitude", "spectral centroid of $|z|$", axes4[1, 0]),
        ("spectral_centroid_z_real", "spectral centroid of $\Re(z)$", axes4[1, 1]),
        ("low_high_power_ratio_z_magnitude", "low/high power ratio of $|z|$", axes4[1, 2]),
    ]
    for key, xlabel, ax in scatter_specs:
        ax.scatter(neg_features[key], neg_features["logit_margin"], color=thesis_red, alpha=0.8, label="negative")
        ax.scatter(pos_features[key], pos_features["logit_margin"], color=ifisc_green, alpha=0.8, label="positive")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("logit margin (pos $-$ neg)")
        ax.set_title(f"decision vs {xlabel}")
        ax.legend(loc="best", fontsize=9)
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, _epoch_filename("sentiment_encoding_decision_drivers", epoch)), transparent=True)
    plt.close(fig4)

    feature_names = list(neg_features.keys())
    effect_sizes = {}
    for name in feature_names:
        effect_sizes[name] = _cohens_d(pos_features[name], neg_features[name])

    ranked = sorted(effect_sizes.items(), key=lambda item: abs(item[1]), reverse=True)
    top_names = [name for name, _ in ranked[:10]]
    top_vals = [effect_sizes[name] for name in top_names]

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    colors = [ifisc_green if v > 0 else thesis_red for v in top_vals]
    ax5.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax5.axvline(0.0, color="0.4", linewidth=0.8)
    ax5.set_xlabel("Cohen's $d$ (positive $-$ negative)")
    ax5.set_title("which signal properties separate sentiment?")
    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, _epoch_filename("sentiment_encoding_separation", epoch)), transparent=True)
    plt.close(fig5)

    magnitude_keys = [k for k in effect_sizes if "magnitude" in k or "z_real" in k or "input_norm" in k]
    frequency_keys = [k for k in effect_sizes if "spectral" in k or "power_ratio" in k]
    magnitude_score = float(np.mean([abs(effect_sizes[k]) for k in magnitude_keys])) if magnitude_keys else 0.0
    frequency_score = float(np.mean([abs(effect_sizes[k]) for k in frequency_keys])) if frequency_keys else 0.0

    summary = {
        "num_negative_examples": int(neg_stages["logits"].shape[0]),
        "num_positive_examples": int(pos_stages["logits"].shape[0]),
        "effect_sizes": {k: float(v) for k, v in effect_sizes.items()},
        "top_separating_features": [{"feature": n, "cohens_d": float(effect_sizes[n])} for n, _ in ranked[:5]],
        "magnitude_separation_score": magnitude_score,
        "frequency_separation_score": frequency_score,
        "dominant_encoding": (
            "magnitude" if magnitude_score >= frequency_score else "frequency"
        ),
        "negative_mean_logit_margin": float(np.mean(neg_features["logit_margin"])),
        "positive_mean_logit_margin": float(np.mean(pos_features["logit_margin"])),
        "negative_mean_final_z_magnitude": float(np.mean(neg_features["final_z_magnitude"])),
        "positive_mean_final_z_magnitude": float(np.mean(pos_features["final_z_magnitude"])),
        "negative_mean_spectral_centroid_z": float(np.mean(neg_features["spectral_centroid_z_magnitude"])),
        "positive_mean_spectral_centroid_z": float(np.mean(pos_features["spectral_centroid_z_magnitude"])),
    }
    with open(os.path.join(output_dir, _epoch_filename("sentiment_encoding_summary", epoch, suffix="json")), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def plot_imdb_sentiment_encoding_analysis(
    model,
    neg_token_ids,
    pos_token_ids,
    output_dir,
    epoch,
    num_units_plot=5,
):
    model.eval()
    neg_inputs = prepare_imdb_sequence(model, neg_token_ids)
    pos_inputs = prepare_imdb_sequence(model, pos_token_ids)
    neg_stages = trace_model_signal_stages_batch(model, neg_inputs, input_mode="norm")
    pos_stages = trace_model_signal_stages_batch(model, pos_inputs, input_mode="norm")
    return plot_sentiment_encoding_analysis(
        neg_stages,
        pos_stages,
        output_dir,
        epoch,
        num_units_plot=num_units_plot,
    )
