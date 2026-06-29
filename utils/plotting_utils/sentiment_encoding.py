import os

import numpy as np
import torch

from .signal_stages import (
    _epoch_filename,
    _unit_indices,
    get_slon_core,
    prepare_imdb_sequence,
)
from .style import thesis_red, ifisc_green


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
    z_imag_trace = []
    z_mag_trace = []

    for t in range(num_timesteps):
        input_t = inputs[t]
        projected = slon.i2h(input_t)
        z_state = torch.cat([z_real, z_imag], dim=1)
        pre_act = projected + slon.gain_rec * slon.h2h(z_state)
        z_real, z_imag = slon.dynamics_step(z_real, z_imag, input_t)

        projected_inputs.append(projected.detach().cpu())
        pre_activations.append(pre_act.detach().cpu())
        z_real_trace.append(z_real.detach().cpu())
        z_imag_trace.append(z_imag.detach().cpu())
        z_mag_trace.append(torch.sqrt(z_real ** 2 + z_imag ** 2).detach().cpu())

    z_features = torch.cat([z_real, z_imag], dim=1)
    logits = slon.h2o(z_features).detach().cpu()
    if input_mode == "norm":
        raw_input = inputs.norm(dim=-1).detach().cpu().numpy()
    else:
        raw_input = inputs[:, :, 0].detach().cpu().numpy()

    return {
        "raw_input": raw_input,
        "projected_input": torch.stack(projected_inputs).numpy(),
        "pre_activation": torch.stack(pre_activations).numpy(),
        "z_real": torch.stack(z_real_trace).numpy(),
        "z_imag": torch.stack(z_imag_trace).numpy(),
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
    z_imag = _mean_over_units(stages["z_imag"], unit_indices)
    z_mag = _mean_over_units(stages["z_magnitude"], unit_indices)
    z_phase = np.unwrap(np.arctan2(z_imag, z_real), axis=0)
    freqs = np.fft.rfftfreq(raw.shape[0], d=1.0)

    features = {
        "mean_input_norm": raw.mean(axis=0),
        "std_input_norm": raw.std(axis=0),
        "mean_pre_activation": pre.mean(axis=0),
        "std_pre_activation": pre.std(axis=0),
        "mean_z_real": z_real.mean(axis=0),
        "std_z_real": z_real.std(axis=0),
        "mean_z_imag": z_imag.mean(axis=0),
        "std_z_imag": z_imag.std(axis=0),
        "mean_z_phase": z_phase.mean(axis=0),
        "std_z_phase": z_phase.std(axis=0),
        "mean_z_magnitude": z_mag.mean(axis=0),
        "std_z_magnitude": z_mag.std(axis=0),
        "final_z_real": z_real[-1],
        "final_z_imag": z_imag[-1],
        "final_z_phase": z_phase[-1],
        "final_z_magnitude": z_mag[-1],
        "max_z_magnitude": z_mag.max(axis=0),
        "logit_margin": _logit_margin(stages["logits"]),
        "positive_prob": torch.softmax(torch.tensor(stages["logits"]), dim=-1).numpy()[:, 1],
    }

    for name, trace in [
        ("input", raw),
        ("pre_activation", pre),
        ("z_real", z_real),
        ("z_imag", z_imag),
        ("z_phase", z_phase),
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
    from .signal_analysis import (
        MAX_EXAMPLE_TIME_SERIES,
        plot_autocorrelation_analysis,
        plot_decision_drivers,
        plot_granular_stage_spectra,
        plot_phase_vs_magnitude_separation,
        plot_real_imag_spectrum_comparison,
        save_signal_analysis_summary,
        _plot_example_time_series,
    )
    from .signal_stages import _unit_indices

    os.makedirs(output_dir, exist_ok=True)

    num_nodes = neg_stages["pre_activation"].shape[-1]
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    groups = {0: neg_stages, 1: pos_stages}
    colors = {0: thesis_red, 1: ifisc_green}

    neg_features = _example_features(neg_stages, unit_indices)
    pos_features = _example_features(pos_stages, unit_indices)
    features_by_group = {0: neg_features, 1: pos_features}

    plot_granular_stage_spectra(
        groups,
        output_dir,
        epoch,
        prefix="sentiment",
        unit_indices=unit_indices,
        colors=colors,
        title_suffix="sentiment encoding: spectra at each processing stage",
    )
    plot_real_imag_spectrum_comparison(groups, output_dir, epoch, "sentiment", unit_indices, colors)
    plot_autocorrelation_analysis(
        groups,
        output_dir,
        epoch,
        prefix="sentiment",
        unit_indices=unit_indices,
        colors=colors,
        title_suffix="sentiment encoding: temporal autocorrelation",
    )

    n_examples = min(neg_stages["logits"].shape[0], pos_stages["logits"].shape[0])
    if n_examples <= MAX_EXAMPLE_TIME_SERIES:
        from .signal_stages import _epoch_filename
        for stage_key, ylabel in [("z_real", r"$\Re(z)$"), ("z_imag", r"$\Im(z)$"), ("z_magnitude", r"$|z|$")]:
            _plot_example_time_series(
                groups,
                stage_key,
                ylabel,
                os.path.join(output_dir, _epoch_filename(f"sentiment_examples_{stage_key.replace('z_', '')}", epoch)),
                colors,
                unit_indices,
            )

    separation_summary = plot_phase_vs_magnitude_separation(
        features_by_group,
        output_dir,
        epoch,
        prefix="sentiment",
        compare_mode="binary",
        reference_groups=(0, 1),
    )

    plot_decision_drivers(
        features_by_group,
        output_dir,
        epoch,
        prefix="sentiment",
        colors=colors,
        ylabel="logit margin (pos $-$ neg)",
    )

    summary = {
        "num_negative_examples": int(neg_stages["logits"].shape[0]),
        "num_positive_examples": int(pos_stages["logits"].shape[0]),
        "negative_mean_logit_margin": float(np.mean(neg_features["logit_margin"])),
        "positive_mean_logit_margin": float(np.mean(pos_features["logit_margin"])),
        "negative_mean_final_z_magnitude": float(np.mean(neg_features["final_z_magnitude"])),
        "positive_mean_final_z_magnitude": float(np.mean(pos_features["final_z_magnitude"])),
        "negative_mean_final_z_imag": float(np.mean(neg_features["final_z_imag"])),
        "positive_mean_final_z_imag": float(np.mean(pos_features["final_z_imag"])),
        "negative_mean_spectral_centroid_z_magnitude": float(np.mean(neg_features["spectral_centroid_z_magnitude"])),
        "positive_mean_spectral_centroid_z_magnitude": float(np.mean(pos_features["spectral_centroid_z_magnitude"])),
        "negative_mean_spectral_centroid_z_imag": float(np.mean(neg_features["spectral_centroid_z_imag"])),
        "positive_mean_spectral_centroid_z_imag": float(np.mean(pos_features["spectral_centroid_z_imag"])),
        **separation_summary,
    }
    return save_signal_analysis_summary(summary, output_dir, epoch, "sentiment")


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
