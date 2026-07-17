import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .sentiment_encoding import (
    _mean_over_units,
    _rfft_amplitude,
    _spectral_centroid,
    trace_model_signal_stages_batch,
)
from .signal_stages import _epoch_filename, _unit_indices
from .style import ifisc_green, thesis_blue, thesis_red


GROUP_LABELS = {
    "error": ("easy", "hard"),
    "target": ("low", "mid", "high"),
    "trend": ("rising", "falling"),
}

GROUP_COLORS = {
    "error": {"easy": ifisc_green, "hard": thesis_red},
    "target": {"low": thesis_blue, "mid": "0.45", "high": thesis_red},
    "trend": {"rising": ifisc_green, "falling": thesis_red},
}

FORECAST_DRIVER_KEYS = [
    ("final_z_magnitude", "final $|z|$"),
    ("mean_z_magnitude", "mean $|z|$"),
    ("final_z_real", r"final $\Re(z)$"),
    ("final_z_imag", r"final $\Im(z)$"),
    ("final_z_phase", r"final $\arg(z)$"),
    ("spectral_centroid_z_magnitude", "spectral centroid of $|z|$"),
    ("spectral_centroid_z_imag", r"spectral centroid of $\Im(z)$"),
    ("low_high_power_ratio_z_magnitude", "low/high power ratio of $|z|$"),
    ("mean_input_norm", "mean input"),
    ("std_input_norm", "input variability"),
]


def _json_float(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def _normalized_error(pred, target, prev_value, min_denominator=1e-4):
    denom = max(abs(target - prev_value), min_denominator)
    return abs(pred - target) / denom


def _window_slope(window):
    window = np.asarray(window, dtype=np.float64)
    t = np.arange(len(window), dtype=np.float64)
    return float(np.polyfit(t, window, 1)[0])


def _assign_error_group(normalized_errors, error):
    median = float(np.median(normalized_errors))
    return "easy" if error <= median else "hard"


def _assign_target_group(target, q33, q66):
    if target <= q33:
        return "low"
    if target <= q66:
        return "mid"
    return "high"


def _assign_trend_group(window):
    return "rising" if _window_slope(window) >= 0.0 else "falling"


def _takens_coords(window, tau_steps):
    window = np.asarray(window, dtype=np.float64)
    length = len(window)
    idx0 = length - 1
    idx1 = max(0, length - 1 - tau_steps)
    idx2 = max(0, length - 1 - 2 * tau_steps)
    return window[idx0], window[idx1], window[idx2]


@torch.no_grad()
def collect_mg_forecast_batches(
    model,
    data_loader,
    num_per_group=25,
    grouping="error",
    max_batches=200,
):
    model.eval()
    if grouping not in GROUP_LABELS:
        raise ValueError(f"Unknown grouping '{grouping}'. Choose from {tuple(GROUP_LABELS)}")

    labels = GROUP_LABELS[grouping]
    candidates = []

    for batch_idx, batch in enumerate(data_loader):
        if len(batch) == 4:
            inputs, targets, _, prev_values = batch
        elif len(batch) == 3:
            inputs, targets, _ = batch
            prev_values = inputs[:, -1, :]
        else:
            inputs, targets = batch
            prev_values = inputs[:, -1, :]

        inputs_perm = inputs.permute(1, 0, 2)
        outputs = model(inputs_perm)["output"]
        preds = outputs.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        prev_np = prev_values.detach().cpu().numpy().reshape(-1)
        windows = inputs.squeeze(-1).detach().cpu().numpy()

        for i in range(inputs.size(0)):
            norm_err = _normalized_error(preds[i], targets_np[i], prev_np[i])
            candidates.append(
                {
                    "input": inputs[i : i + 1],
                    "target": float(targets_np[i]),
                    "pred": float(preds[i]),
                    "prev_value": float(prev_np[i]),
                    "normalized_error": float(norm_err),
                    "window": windows[i],
                }
            )

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    if not candidates:
        raise ValueError("No Mackey-Glass forecast examples collected from data loader.")

    norm_errors = [c["normalized_error"] for c in candidates]
    targets_all = [c["target"] for c in candidates]
    q33, q66 = np.percentile(targets_all, [33.33, 66.67])

    for candidate in candidates:
        if grouping == "error":
            candidate["group"] = _assign_error_group(norm_errors, candidate["normalized_error"])
        elif grouping == "target":
            candidate["group"] = _assign_target_group(candidate["target"], q33, q66)
        else:
            candidate["group"] = _assign_trend_group(candidate["window"])

    buckets = {label: [] for label in labels}
    for candidate in candidates:
        buckets[candidate["group"]].append(candidate)

    for label in labels:
        bucket = buckets[label]
        if grouping == "error":
            bucket.sort(key=lambda c: c["normalized_error"], reverse=(label == "hard"))
        elif grouping == "target":
            bucket.sort(key=lambda c: c["target"], reverse=(label == "high"))
        else:
            bucket.sort(key=lambda c: _window_slope(c["window"]), reverse=(label == "rising"))
        buckets[label] = bucket[:num_per_group]

    groups = {}
    metadata = {}
    for label in labels:
        bucket = buckets[label]
        if not bucket:
            continue
        batch_inputs = torch.cat([c["input"] for c in bucket], dim=0)
        inputs_perm = batch_inputs.permute(1, 0, 2)
        groups[label] = trace_model_signal_stages_batch(model, inputs_perm, input_mode="scalar")
        metadata[label] = {
            "targets": np.array([c["target"] for c in bucket], dtype=np.float64),
            "preds": np.array([c["pred"] for c in bucket], dtype=np.float64),
            "prev_values": np.array([c["prev_value"] for c in bucket], dtype=np.float64),
            "normalized_errors": np.array([c["normalized_error"] for c in bucket], dtype=np.float64),
            "delay_windows": np.stack([c["window"] for c in bucket], axis=0),
        }

    if not groups:
        raise ValueError(f"Could not form any groups for grouping='{grouping}'.")

    return groups, metadata


def _example_features_regression(stages, unit_indices, metadata):
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
        "prediction_raw": stages["logits"].reshape(-1),
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

    features["target"] = metadata["targets"]
    features["prediction"] = metadata["preds"]
    features["abs_error"] = np.abs(features["prediction"] - features["target"])
    features["normalized_error"] = metadata["normalized_errors"]
    features["signed_error"] = features["prediction"] - features["target"]
    return features


def _features_by_group(groups, metadata_by_group, unit_indices):
    return {
        label: _example_features_regression(stages, unit_indices, metadata_by_group[label])
        for label, stages in groups.items()
    }


def _features_for_separation(features_by_group):
    exclude = {
        "target",
        "prediction",
        "abs_error",
        "normalized_error",
        "signed_error",
        "logit_margin",
        "positive_prob",
        "predicted_class",
        "correct",
    }
    return {
        label: {key: value for key, value in features.items() if key not in exclude}
        for label, features in features_by_group.items()
    }


def _stages_dynamics_finite(stages):
    for key in ("pre_activation", "z_real", "z_imag", "z_magnitude", "logits"):
        if key in stages and not np.isfinite(stages[key]).all():
            return False
    return True


def plot_forecast_drivers(features_by_group, output_dir, epoch, prefix, colors):
    os.makedirs(output_dir, exist_ok=True)
    group_labels = sorted(features_by_group.keys())
    available = [
        (key, label)
        for key, label in FORECAST_DRIVER_KEYS
        if key in features_by_group[group_labels[0]]
    ]

    n = len(available)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (key, xlabel) in zip(axes_flat, available):
        for label in group_labels:
            color = colors[label] if isinstance(colors, dict) else colors[group_labels.index(label)]
            ax.scatter(
                features_by_group[label][key],
                features_by_group[label]["normalized_error"],
                color=color,
                alpha=0.8,
                s=18,
                label=str(label),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|normalized error|")
        ax.set_title(f"forecast error vs {xlabel}")
    axes_flat[0].legend(title="group", fontsize=8, ncol=2, loc="best")
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_forecast_drivers", epoch)), transparent=True)
    plt.close(fig)


def plot_delay_embedding_3d(metadata_by_group, output_dir, epoch, prefix, tau_steps, colors):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label, metadata in metadata_by_group.items():
        windows = metadata["delay_windows"]
        coords = np.array([_takens_coords(window, tau_steps) for window in windows])
        color = colors[label] if isinstance(colors, dict) else colors
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            color=color,
            alpha=0.75,
            s=24,
            label=str(label),
        )

    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(fr"$x(t-{tau_steps})$")
    ax.set_zlabel(fr"$x(t-{2 * tau_steps})$")
    ax.set_title(f"Takens delay embedding ($\\tau={tau_steps}$ steps)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_takens_3d", epoch)), transparent=True)
    plt.close(fig)


def _pca_2d(data):
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(data)


def plot_delay_pca(metadata_by_group, output_dir, epoch, prefix, colors):
    os.makedirs(output_dir, exist_ok=True)
    all_windows = []
    all_labels = []
    all_errors = []
    for label, metadata in metadata_by_group.items():
        windows = metadata["delay_windows"]
        all_windows.append(windows)
        all_labels.extend([label] * windows.shape[0])
        all_errors.append(metadata["normalized_errors"])
    windows = np.concatenate(all_windows, axis=0)
    errors = np.concatenate(all_errors, axis=0)

    if windows.shape[0] < 3:
        return

    coords = _pca_2d(windows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for label in metadata_by_group:
        mask = np.array(all_labels) == label
        color = colors[label] if isinstance(colors, dict) else colors
        axes[0].scatter(coords[mask, 0], coords[mask, 1], color=color, alpha=0.75, s=22, label=str(label))
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("Delay-window PCA by forecast group")
    axes[0].legend(loc="best")

    scatter = axes[1].scatter(coords[:, 0], coords[:, 1], c=errors, cmap="magma", alpha=0.8, s=22)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].set_title("Delay-window PCA colored by |normalized error|")
    fig.colorbar(scatter, ax=axes[1], shrink=0.9, label="|normalized error|")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_delay_pca", epoch)), transparent=True)
    plt.close(fig)


def plot_hidden_vs_delay_geometry(groups, metadata_by_group, output_dir, epoch, prefix, unit_indices):
    os.makedirs(output_dir, exist_ok=True)
    all_windows = []
    hidden_states = []
    errors = []
    labels = []

    for label, metadata in metadata_by_group.items():
        stages = groups[label]
        windows = metadata["delay_windows"]
        z_real = stages["z_real"][:, :, unit_indices].mean(axis=-1)
        z_imag = stages["z_imag"][:, :, unit_indices].mean(axis=-1)
        final_real = z_real[-1]
        final_imag = z_imag[-1]
        if final_real.ndim == 1:
            final_real = final_real[:, None]
            final_imag = final_imag[:, None]
        hidden = np.concatenate([final_real, final_imag], axis=1)
        all_windows.append(windows)
        hidden_states.append(hidden)
        errors.append(metadata["normalized_errors"])
        labels.extend([label] * windows.shape[0])

    windows = np.concatenate(all_windows, axis=0)
    hidden = np.concatenate(hidden_states, axis=0)
    errors = np.concatenate(errors, axis=0)
    if windows.shape[0] < 3:
        return

    delay_pca = _pca_2d(windows)
    hidden_pca = _pca_2d(hidden)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    scatter0 = axes[0].scatter(delay_pca[:, 0], delay_pca[:, 1], c=errors, cmap="magma", alpha=0.8, s=22)
    axes[0].set_xlabel("delay PC 1")
    axes[0].set_ylabel("delay PC 2")
    axes[0].set_title("Input delay geometry")
    fig.colorbar(scatter0, ax=axes[0], shrink=0.9, label="|normalized error|")

    scatter1 = axes[1].scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=errors, cmap="magma", alpha=0.8, s=22)
    axes[1].set_xlabel("hidden PC 1")
    axes[1].set_ylabel("hidden PC 2")
    axes[1].set_title("Final hidden-state geometry")
    fig.colorbar(scatter1, ax=axes[1], shrink=0.9, label="|normalized error|")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_geometry_comparison", epoch)), transparent=True)
    plt.close(fig)


def plot_cross_correlation_input_hidden(groups, metadata_by_group, output_dir, epoch, prefix, unit_indices, tau_steps):
    from .signal_analysis import _mean_over_units

    os.makedirs(output_dir, exist_ok=True)
    max_lag = min(next(iter(groups.values()))["raw_input"].shape[0] - 1, max(50, tau_steps * 2))
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, stages in groups.items():
        raw = stages["raw_input"]
        z_mag = _mean_over_units(stages["z_magnitude"], unit_indices)
        ccfs = []
        for example_idx in range(raw.shape[1]):
            input_trace = raw[:, example_idx]
            hidden_trace = z_mag[:, example_idx]
            centered_input = input_trace - input_trace.mean()
            centered_hidden = hidden_trace - hidden_trace.mean()
            denom = np.sqrt(np.sum(centered_input ** 2) * np.sum(centered_hidden ** 2))
            if denom <= 0:
                continue
            ccf = np.array(
                [
                    np.sum(centered_input[lag:] * centered_hidden[: input_trace.shape[0] - lag]) / denom
                    for lag in lags
                ]
            )
            ccfs.append(ccf)
        if not ccfs:
            continue
        mean_ccf = np.mean(ccfs, axis=0)
        ax.plot(lags, mean_ccf, linewidth=1.8, label=str(label))
    ax.axvline(tau_steps, color="0.4", linestyle="--", linewidth=1.0, label=fr"MG $\tau={tau_steps}$")
    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8)
    ax.set_xlabel("input lag (steps)")
    ax.set_ylabel("corr(lagged input, $|z|$ trace)")
    ax.set_title("Cross-correlation: input history vs final hidden magnitude")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_input_hidden_ccf", epoch)), transparent=True)
    plt.close(fig)


def plot_mackey_glass_encoding_analysis(
    groups,
    metadata_by_group,
    output_dir,
    epoch,
    grouping="error",
    num_units_plot=5,
    tau_steps=17,
):
    from .signal_analysis import (
        MAX_EXAMPLE_TIME_SERIES,
        plot_autocorrelation_analysis,
        plot_granular_stage_spectra,
        plot_node_activity_analysis,
        plot_phase_vs_magnitude_separation,
        plot_real_imag_spectrum_comparison,
        save_signal_analysis_summary,
        _plot_example_time_series,
    )

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"mg_{grouping}"
    group_labels = sorted(groups.keys())
    colors = {label: GROUP_COLORS[grouping][label] for label in group_labels if label in GROUP_COLORS[grouping]}

    num_nodes = next(iter(groups.values()))["pre_activation"].shape[-1]
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    features_by_group = _features_by_group(groups, metadata_by_group, unit_indices)
    dynamics_finite = all(_stages_dynamics_finite(groups[label]) for label in group_labels)

    title_suffix = f"Mackey-Glass {grouping} encoding"
    plot_granular_stage_spectra(
        groups,
        output_dir,
        epoch,
        prefix=prefix,
        unit_indices=unit_indices,
        colors=colors,
        title_suffix=f"{title_suffix}: spectra at each processing stage",
    )
    plot_real_imag_spectrum_comparison(groups, output_dir, epoch, prefix, unit_indices, colors)
    plot_autocorrelation_analysis(
        groups,
        output_dir,
        epoch,
        prefix=prefix,
        unit_indices=unit_indices,
        colors=colors,
        title_suffix=f"{title_suffix}: temporal autocorrelation",
    )
    node_summary = plot_node_activity_analysis(
        groups,
        output_dir,
        epoch,
        prefix=prefix,
        title_suffix=title_suffix,
    )

    examples_per_group = groups[group_labels[0]]["logits"].shape[0]
    if examples_per_group <= MAX_EXAMPLE_TIME_SERIES:
        for stage_key, ylabel in [("z_real", r"$\Re(z)$"), ("z_imag", r"$\Im(z)$"), ("z_magnitude", r"$|z|$")]:
            _plot_example_time_series(
                groups,
                stage_key,
                ylabel,
                os.path.join(output_dir, _epoch_filename(f"{prefix}_examples_{stage_key.replace('z_', '')}", epoch)),
                colors,
                unit_indices,
            )

    separation_features = _features_for_separation(features_by_group)
    if grouping in ("error", "trend") and len(group_labels) == 2:
        separation_summary = plot_phase_vs_magnitude_separation(
            separation_features,
            output_dir,
            epoch,
            prefix=prefix,
            compare_mode="binary",
            reference_groups=tuple(group_labels),
        )
    else:
        separation_summary = plot_phase_vs_magnitude_separation(
            separation_features,
            output_dir,
            epoch,
            prefix=prefix,
            compare_mode="multiclass",
        )

    plot_forecast_drivers(features_by_group, output_dir, epoch, prefix, colors)
    plot_delay_embedding_3d(metadata_by_group, output_dir, epoch, prefix, tau_steps, colors)
    plot_delay_pca(metadata_by_group, output_dir, epoch, prefix, colors)
    plot_hidden_vs_delay_geometry(groups, metadata_by_group, output_dir, epoch, prefix, unit_indices)
    plot_cross_correlation_input_hidden(
        groups, metadata_by_group, output_dir, epoch, prefix, unit_indices, tau_steps
    )

    summary = {
        "grouping": grouping,
        "dynamics_finite": dynamics_finite,
        "tau_steps": int(tau_steps),
        "examples_per_group": {label: int(groups[label]["logits"].shape[0]) for label in group_labels},
        "mean_normalized_error": {
            label: _json_float(np.mean(metadata_by_group[label]["normalized_errors"])) for label in group_labels
        },
        "mean_abs_error": {
            label: _json_float(np.mean(features_by_group[label]["abs_error"])) for label in group_labels
        },
        "mean_final_z_magnitude": {
            label: _json_float(np.mean(features_by_group[label]["final_z_magnitude"])) for label in group_labels
        },
        "node_activity": node_summary,
        **separation_summary,
    }
    if not dynamics_finite:
        summary["warning"] = (
            "Oscillator states or outputs are non-finite. This usually means unstable dynamics "
            "(e.g. lambda_param > 0 or diverged weights)."
        )
    return save_signal_analysis_summary(summary, output_dir, epoch, prefix)


@torch.no_grad()
def plot_mackey_glass_encoding_analysis_from_loader(
    model,
    data_loader,
    output_dir,
    epoch,
    grouping="error",
    num_per_group=25,
    num_units_plot=5,
    tau_steps=17,
    max_batches=200,
):
    groupings = list(GROUP_LABELS) if grouping == "all" else [grouping]
    summaries = {}
    for group_mode in groupings:
        groups, metadata = collect_mg_forecast_batches(
            model,
            data_loader,
            num_per_group=num_per_group,
            grouping=group_mode,
            max_batches=max_batches,
        )
        summaries[group_mode] = plot_mackey_glass_encoding_analysis(
            groups,
            metadata,
            output_dir,
            epoch,
            grouping=group_mode,
            num_units_plot=num_units_plot,
            tau_steps=tau_steps,
        )
    return summaries if grouping == "all" else summaries[groupings[0]]
