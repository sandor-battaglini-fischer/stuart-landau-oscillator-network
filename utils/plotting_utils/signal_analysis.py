import json
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .signal_stages import _epoch_filename
from .sentiment_encoding import (
    _cohens_d,
    _mean_over_units,
    _mean_std_spectrum,
    _plot_mean_std_band,
    _rfft_amplitude,
)
from .style import thesis_blue, thesis_red, ifisc_green


MAX_EXAMPLE_TIME_SERIES = 10

SPECTRAL_STAGE_SPECS = [
    ("raw_input", "input"),
    ("pre_activation", r"pre-activation $u(t)$"),
    ("z_real", r"$\Re(z)$"),
    ("z_imag", r"$\Im(z)$"),
    ("z_magnitude", r"$|z|$"),
    ("z_phase", r"$\arg(z)$"),
]

NODE_ACTIVITY_STAGES = [
    ("pre_activation", r"pre-activation $u(t)$"),
    ("z_real", r"$\Re(z)$"),
    ("z_imag", r"$\Im(z)$"),
    ("z_magnitude", r"$|z|$"),
]


def _json_float(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def _ticks_for_freqs(freqs, n_ticks=8):
    step = max(1, len(freqs) // n_ticks)
    idx = np.arange(0, len(freqs), step)
    return idx, [f"{freqs[i]:.3f}" for i in idx]


def plot_class_frequency_heatmaps(spectra, freqs, group_labels, title_prefix, output_path):
    spectra = np.asarray(spectra, dtype=np.float64)
    spectra = np.where(np.isfinite(spectra), spectra, 0.0)
    n_classes = spectra.shape[0]

    cross_class_mean = spectra.mean(axis=0, keepdims=True)
    deviation = spectra - cross_class_mean
    col_mean = spectra.mean(axis=0, keepdims=True)
    col_std = spectra.std(axis=0, keepdims=True)
    zscore = (spectra - col_mean) / np.where(col_std > 0, col_std, 1.0)

    tick_idx, tick_labels = _ticks_for_freqs(freqs)

    fig, axes = plt.subplots(3, 1, figsize=(13, 3.0 + 0.55 * n_classes * 3))

    positive = spectra[spectra > 0]
    vmin = positive.min() if positive.size else 1e-6
    vmax = spectra.max() if spectra.max() > 0 else 1.0
    im0 = axes[0].imshow(
        np.maximum(spectra, vmin),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    axes[0].set_title(f"{title_prefix}: amplitude (log color scale, low-freq dominance visible)")
    fig.colorbar(im0, ax=axes[0], shrink=0.9, label="amplitude (log)")

    dev_max = np.max(np.abs(deviation)) if deviation.size and np.max(np.abs(deviation)) > 0 else 1.0
    im1 = axes[1].imshow(
        deviation,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-dev_max,
        vmax=dev_max,
    )
    axes[1].set_title(f"{title_prefix}: deviation from cross-class mean (where classes differ)")
    fig.colorbar(im1, ax=axes[1], shrink=0.9, label="amp $-$ mean")

    z_max = np.max(np.abs(zscore)) if zscore.size and np.max(np.abs(zscore)) > 0 else 1.0
    im2 = axes[2].imshow(
        zscore,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-z_max,
        vmax=z_max,
    )
    axes[2].set_title(f"{title_prefix}: per-frequency z-score across classes (discriminative bands)")
    fig.colorbar(im2, ax=axes[2], shrink=0.9, label="z-score")

    for ax in axes:
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels([str(l) for l in group_labels])
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylabel("class")
    axes[-1].set_xlabel("frequency")

    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def _normalized_autocorr(trace, max_lag=None):
    if trace.ndim == 1:
        trace = trace[:, None]
    num_timesteps = trace.shape[0]
    if max_lag is None:
        max_lag = min(num_timesteps // 2, 100)
    max_lag = max(1, min(max_lag, num_timesteps - 1))

    centered = trace - trace.mean(axis=0, keepdims=True)
    var = np.mean(centered ** 2, axis=0)
    var = np.where(var > 0, var, 1.0)

    acfs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acfs.append(np.ones(trace.shape[1]))
        else:
            acfs.append(np.mean(centered[:-lag] * centered[lag:], axis=0) / var)
    return np.arange(max_lag + 1), np.stack(acfs, axis=0)


def _mean_std_autocorr(trace, unit_indices):
    if trace.ndim == 2:
        lags, acf = _normalized_autocorr(trace)
    else:
        mean_trace = _mean_over_units(trace, unit_indices)
        lags, acf = _normalized_autocorr(mean_trace)
    return lags, acf.mean(axis=1), acf.std(axis=1)


def _stage_trace(stages, key, unit_indices):
    if key == "z_phase":
        z_real = _mean_over_units(stages["z_real"], unit_indices)
        z_imag = _mean_over_units(stages["z_imag"], unit_indices)
        return np.unwrap(np.arctan2(z_imag, z_real), axis=0)
    if key == "raw_input":
        return stages[key]
    return _mean_over_units(stages[key], unit_indices)


def _plot_example_time_series(groups, stage_key, ylabel, output_path, colors, unit_indices, max_examples=MAX_EXAMPLE_TIME_SERIES):
    labels = list(groups.keys())
    actual_n = max(_stage_trace(stages, stage_key, unit_indices).shape[1] for stages in groups.values())
    if actual_n > max_examples:
        return False
    n_examples = actual_n

    fig, axes = plt.subplots(len(labels), n_examples, figsize=(2.4 * n_examples, 2.2 * len(labels)), sharex=True, squeeze=False)
    for row, label in enumerate(labels):
        data = _stage_trace(groups[label], stage_key, unit_indices)
        color = colors[label] if isinstance(colors, dict) else colors[row]
        for col in range(n_examples):
            ax = axes[row, col]
            ax.plot(data[:, col], color=color, linewidth=1.2)
            if row == 0:
                ax.set_title(f"ex. {col}")
            if col == 0:
                ax.set_ylabel(f"{label}\n{ylabel}")
        axes[row, -1].set_xlabel("time step")
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    return True


def plot_granular_stage_spectra(
    groups,
    output_dir,
    epoch,
    prefix,
    unit_indices,
    colors,
    title_suffix="",
):
    os.makedirs(output_dir, exist_ok=True)
    num_timesteps = next(iter(groups.values()))["raw_input"].shape[0]
    freqs = np.fft.rfftfreq(num_timesteps, d=1.0)

    n_stages = len(SPECTRAL_STAGE_SPECS)
    fig, axes = plt.subplots(n_stages, 2, figsize=(16, 3.2 * n_stages))
    if n_stages == 1:
        axes = np.array([axes])

    group_labels = list(groups.keys())
    ref_mean_by_stage = {}
    for stage_key, stage_label in SPECTRAL_STAGE_SPECS:
        spectra = []
        for label in group_labels:
            trace = _stage_trace(groups[label], stage_key, unit_indices)
            mean_amp, _ = _mean_std_spectrum(trace, unit_indices)
            spectra.append(mean_amp)
        ref_mean_by_stage[stage_key] = np.mean(spectra, axis=0)

    for row, (stage_key, stage_label) in enumerate(SPECTRAL_STAGE_SPECS):
        ax_amp, ax_diff = axes[row]
        ref_mean = ref_mean_by_stage[stage_key]
        for label in group_labels:
            trace = _stage_trace(groups[label], stage_key, unit_indices)
            mean_amp, std_amp = _mean_std_spectrum(trace, unit_indices)
            color = colors[label] if isinstance(colors, dict) else colors[group_labels.index(label)]
            _plot_mean_std_band(ax_amp, freqs, mean_amp, std_amp, color, str(label))
        ax_amp.set_ylabel("amplitude")
        ax_amp.set_title(f"{stage_label}: mean spectrum")
        ax_amp.legend(loc="upper right", fontsize=8, ncol=2)

        for label in group_labels:
            trace = _stage_trace(groups[label], stage_key, unit_indices)
            mean_amp, _ = _mean_std_spectrum(trace, unit_indices)
            diff = mean_amp - ref_mean
            color = colors[label] if isinstance(colors, dict) else colors[group_labels.index(label)]
            ax_diff.plot(freqs, diff, color=color, linewidth=1.2, alpha=0.85, label=str(label))
        ax_diff.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        ax_diff.set_ylabel("amp. diff")
        ax_diff.set_title(f"{stage_label}: spectral diff vs class mean")
        ax_diff.legend(loc="upper right", fontsize=7, ncol=2)

    axes[-1, 0].set_xlabel("frequency")
    axes[-1, 1].set_xlabel("frequency")
    if title_suffix:
        fig.suptitle(title_suffix, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_spectra_all_stages", epoch)), transparent=True)
    plt.close(fig)

    for stage_key, stage_label in SPECTRAL_STAGE_SPECS:
        group_labels = list(groups.keys())

        fig_s, ax_line = plt.subplots(figsize=(11, 4.5))
        spectra = []
        for label in group_labels:
            trace = _stage_trace(groups[label], stage_key, unit_indices)
            mean_amp, std_amp = _mean_std_spectrum(trace, unit_indices)
            spectra.append(mean_amp)
            color = colors[label] if isinstance(colors, dict) else colors[group_labels.index(label)]
            _plot_mean_std_band(ax_line, freqs, mean_amp, std_amp, color, str(label))
        ax_line.set_xlabel("frequency")
        ax_line.set_ylabel("amplitude")
        ax_line.set_yscale("log")
        ax_line.set_title(f"{stage_label}: per-class spectrum (log amplitude)")
        ax_line.legend(loc="upper right", fontsize=8, ncol=2)
        plt.tight_layout()
        safe_key = stage_key.replace("z_", "")
        fig_s.savefig(
            os.path.join(output_dir, _epoch_filename(f"{prefix}_spectra_{safe_key}", epoch)),
            transparent=True,
        )
        plt.close(fig_s)

        plot_class_frequency_heatmaps(
            spectra,
            freqs,
            group_labels,
            f"{stage_label}",
            os.path.join(output_dir, _epoch_filename(f"{prefix}_spectra_{safe_key}_heatmap", epoch)),
        )


def plot_autocorrelation_analysis(
    groups,
    output_dir,
    epoch,
    prefix,
    unit_indices,
    colors,
    title_suffix="",
):
    os.makedirs(output_dir, exist_ok=True)
    autocorr_stages = [
        ("raw_input", "input"),
        ("z_real", r"$\Re(z)$"),
        ("z_imag", r"$\Im(z)$"),
        ("z_magnitude", r"$|z|$"),
    ]

    n_stages = len(autocorr_stages)
    fig, axes = plt.subplots(n_stages, 1, figsize=(12, 3.2 * n_stages), sharex=True)
    if n_stages == 1:
        axes = [axes]

    for ax, (stage_key, stage_label) in zip(axes, autocorr_stages):
        for label, stages in groups.items():
            trace = _stage_trace(stages, stage_key, unit_indices)
            lags, mean_acf, std_acf = _mean_std_autocorr(trace, unit_indices)
            color = colors[label] if isinstance(colors, dict) else colors[list(groups.keys()).index(label)]
            ax.plot(lags, mean_acf, color=color, linewidth=1.8, label=str(label))
            ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color=color, alpha=0.15)
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_ylabel("autocorr")
        ax.set_title(f"{stage_label}: normalized autocorrelation")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("lag")
    if title_suffix:
        fig.suptitle(title_suffix, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_autocorr_all_stages", epoch)), transparent=True)
    plt.close(fig)

    for stage_key, stage_label in autocorr_stages:
        fig_a, ax_a = plt.subplots(figsize=(10, 4.5))
        for label, stages in groups.items():
            trace = _stage_trace(stages, stage_key, unit_indices)
            lags, mean_acf, std_acf = _mean_std_autocorr(trace, unit_indices)
            color = colors[label] if isinstance(colors, dict) else colors[list(groups.keys()).index(label)]
            ax_a.plot(lags, mean_acf, color=color, linewidth=1.8, label=str(label))
            ax_a.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color=color, alpha=0.15)
        ax_a.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        ax_a.set_xlabel("lag")
        ax_a.set_ylabel("autocorr")
        ax_a.set_title(f"{stage_label}: normalized autocorrelation")
        ax_a.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        safe_key = stage_key.replace("z_", "")
        fig_a.savefig(
            os.path.join(output_dir, _epoch_filename(f"{prefix}_autocorr_{safe_key}", epoch)),
            transparent=True,
        )
        plt.close(fig_a)


def _feature_category(feature_name):
    if "spectral" in feature_name or "power_ratio" in feature_name:
        return "frequency"
    if "imag" in feature_name or "phase" in feature_name:
        return "phase"
    if "magnitude" in feature_name or "z_real" in feature_name:
        return "magnitude"
    if "input" in feature_name or "pre_activation" in feature_name:
        return "input"
    return "other"


def plot_phase_vs_magnitude_separation(
    features_by_group,
    output_dir,
    epoch,
    prefix,
    compare_mode="multiclass",
    reference_groups=None,
):
    os.makedirs(output_dir, exist_ok=True)
    group_labels = sorted(features_by_group.keys())
    feature_names = [k for k in features_by_group[group_labels[0]].keys() if k not in ("predicted_class", "correct", "positive_prob")]

    if compare_mode == "binary" and reference_groups is not None:
        neg_label, pos_label = reference_groups
        effect_sizes = {}
        for name in feature_names:
            effect_sizes[name] = _cohens_d(features_by_group[pos_label][name], features_by_group[neg_label][name])
        ranked = sorted(effect_sizes.items(), key=lambda item: abs(item[1]), reverse=True)
        top_names = [name for name, _ in ranked[:12]]
        top_vals = [effect_sizes[name] for name in top_names]

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_colors = [ifisc_green if v > 0 else thesis_red for v in top_vals]
        ax.barh(top_names[::-1], top_vals[::-1], color=bar_colors[::-1], alpha=0.85)
        ax.axvline(0.0, color="0.4", linewidth=0.8)
        ax.set_xlabel("Cohen's $d$ (positive $-$ negative)")
        ax.set_title("signal properties separating sentiment (magnitude vs phase vs frequency)")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_phase_magnitude_separation", epoch)), transparent=True)
        plt.close(fig)

        by_category = {"magnitude": [], "phase": [], "frequency": [], "input": [], "other": []}
        for name, val in effect_sizes.items():
            by_category[_feature_category(name)].append(abs(val))
        category_scores = {k: float(np.mean(v)) if v else 0.0 for k, v in by_category.items()}
        magnitude_score = category_scores["magnitude"]
        phase_score = category_scores["phase"]
        frequency_score = category_scores["frequency"]
        dominant = max(
            [("magnitude", magnitude_score), ("phase", phase_score), ("frequency", frequency_score)],
            key=lambda x: x[1],
        )[0]
        return {
            "effect_sizes": {k: float(v) for k, v in effect_sizes.items()},
            "category_separation_scores": category_scores,
            "magnitude_separation_score": magnitude_score,
            "phase_separation_score": phase_score,
            "frequency_separation_score": frequency_score,
            "dominant_encoding": dominant,
            "top_separating_features": [{"feature": n, "cohens_d": float(effect_sizes[n])} for n, _ in ranked[:5]],
        }

    separation = {}
    for feat in feature_names:
        if compare_mode == "multiclass":
            separation[feat] = [
                _cohens_d(
                    features_by_group[d][feat],
                    np.concatenate([features_by_group[g][feat] for g in group_labels if g != d]),
                )
                for d in group_labels
            ]
        else:
            separation[feat] = [_cohens_d(features_by_group[group_labels[0]][feat], features_by_group[group_labels[1]][feat])]

    ranked_features = sorted(
        separation.keys(),
        key=lambda f: max(abs(v) for v in separation[f]) if compare_mode == "multiclass" else abs(separation[f][0]),
        reverse=True,
    )
    top_features = ranked_features[:10]
    heatmap = np.array([separation[f] for f in top_features])

    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = np.max(np.abs(heatmap)) if heatmap.size else 1.0
    im = ax.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels([str(l) for l in group_labels])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("class (one-vs-rest Cohen's $d$)" if compare_mode == "multiclass" else "comparison")
    ax.set_title("which signal properties separate classes? (magnitude, phase, frequency)")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Cohen's $d$")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_phase_magnitude_separation", epoch)), transparent=True)
    plt.close(fig)

    magnitude_keys = [k for k in ranked_features if _feature_category(k) == "magnitude"]
    phase_keys = [k for k in ranked_features if _feature_category(k) == "phase"]
    frequency_keys = [k for k in ranked_features if _feature_category(k) == "frequency"]

    def _mean_sep(keys):
        if not keys:
            return 0.0
        if compare_mode == "multiclass":
            return float(np.mean([max(abs(separation[k][i]) for i in range(len(group_labels))) for k in keys]))
        return float(np.mean([abs(separation[k][0]) for k in keys]))

    magnitude_score = _mean_sep(magnitude_keys)
    phase_score = _mean_sep(phase_keys)
    frequency_score = _mean_sep(frequency_keys)
    dominant = max(
        [("magnitude", magnitude_score), ("phase", phase_score), ("frequency", frequency_score)],
        key=lambda x: x[1],
    )[0]

    per_group_dominant = {}
    for label in group_labels:
        idx = group_labels.index(label)
        mag = np.mean([abs(separation[k][idx]) for k in magnitude_keys]) if magnitude_keys else 0.0
        ph = np.mean([abs(separation[k][idx]) for k in phase_keys]) if phase_keys else 0.0
        freq = np.mean([abs(separation[k][idx]) for k in frequency_keys]) if frequency_keys else 0.0
        per_group_dominant[str(label)] = max([("magnitude", mag), ("phase", ph), ("frequency", freq)], key=lambda x: x[1])[0]

    return {
        "one_vs_rest_cohens_d": {
            f: {str(d): _json_float(separation[f][i]) for i, d in enumerate(group_labels)} for f in top_features
        },
        "category_separation_scores": {
            "magnitude": magnitude_score,
            "phase": phase_score,
            "frequency": frequency_score,
        },
        "magnitude_separation_score": magnitude_score,
        "phase_separation_score": phase_score,
        "frequency_separation_score": frequency_score,
        "dominant_encoding": dominant,
        "per_group_dominant_encoding": per_group_dominant,
        "top_separating_features": [
            {
                "feature": f,
                "max_cohens_d": _json_float(
                    max(abs(v) for v in separation[f]) if compare_mode == "multiclass" else abs(separation[f][0])
                ),
            }
            for f in top_features[:5]
        ],
    }


def plot_real_imag_spectrum_comparison(groups, output_dir, epoch, prefix, unit_indices, colors):
    os.makedirs(output_dir, exist_ok=True)
    num_timesteps = next(iter(groups.values()))["raw_input"].shape[0]
    freqs = np.fft.rfftfreq(num_timesteps, d=1.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for label, stages in groups.items():
        z_real = _mean_over_units(stages["z_real"], unit_indices)
        z_imag = _mean_over_units(stages["z_imag"], unit_indices)
        real_mean, real_std = _mean_std_spectrum(z_real, unit_indices)
        imag_mean, imag_std = _mean_std_spectrum(z_imag, unit_indices)
        color = colors[label] if isinstance(colors, dict) else colors[list(groups.keys()).index(label)]
        _plot_mean_std_band(axes[0], freqs, real_mean, real_std, color, str(label))
        _plot_mean_std_band(axes[1], freqs, imag_mean, imag_std, color, str(label))

    axes[0].set_ylabel("amplitude")
    axes[0].set_title(r"$\Re(z)$ frequency spectrum per class")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].set_ylabel("amplitude")
    axes[1].set_title(r"$\Im(z)$ frequency spectrum per class")
    axes[1].set_xlabel("frequency")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_real_imag_spectra", epoch)), transparent=True)
    plt.close(fig)

    group_labels = list(groups.keys())
    for component, key, comp_label in [("real", "z_real", r"$\Re(z)$"), ("imag", "z_imag", r"$\Im(z)$")]:
        spectra = []
        for label in group_labels:
            trace = _mean_over_units(groups[label][key], unit_indices)
            mean_amp, _ = _mean_std_spectrum(trace, unit_indices)
            spectra.append(mean_amp)
        plot_class_frequency_heatmaps(
            spectra,
            freqs,
            group_labels,
            comp_label,
            os.path.join(output_dir, _epoch_filename(f"{prefix}_{component}_spectrum_heatmap", epoch)),
        )


DECISION_DRIVER_KEYS = [
    ("final_z_magnitude", "final $|z|$"),
    ("mean_z_magnitude", "mean $|z|$"),
    ("final_z_real", r"final $\Re(z)$"),
    ("final_z_imag", r"final $\Im(z)$"),
    ("final_z_phase", r"final $\arg(z)$"),
    ("spectral_centroid_z_magnitude", "spectral centroid of $|z|$"),
    ("spectral_centroid_z_imag", r"spectral centroid of $\Im(z)$"),
    ("low_high_power_ratio_z_magnitude", "low/high power ratio of $|z|$"),
]


def plot_decision_drivers(features_by_group, output_dir, epoch, prefix, colors, ylabel="logit margin"):
    os.makedirs(output_dir, exist_ok=True)
    group_labels = sorted(features_by_group.keys())
    available = [(k, lbl) for k, lbl in DECISION_DRIVER_KEYS if k in features_by_group[group_labels[0]] for lbl in [lbl]]

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
                features_by_group[label]["logit_margin"],
                color=color,
                alpha=0.8,
                s=18,
                label=str(label),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"decision vs {xlabel}")
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axes_flat[0].legend(title="class", fontsize=8, ncol=2, loc="best")
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename(f"{prefix}_decision_drivers", epoch)), transparent=True)
    plt.close(fig)


def _node_activity_trace(stages, stage_key):
    trace = stages[stage_key]
    if stage_key == "raw_input":
        return trace
    if stage_key in ("z_real", "z_imag", "pre_activation"):
        return np.abs(trace)
    return trace


def _class_mean_node_traces(groups, stage_key):
    return {
        label: _node_activity_trace(stages, stage_key).mean(axis=1)
        for label, stages in groups.items()
    }


def _pooled_node_trace(groups, stage_key):
    traces = [_node_activity_trace(stages, stage_key) for stages in groups.values()]
    return np.concatenate(traces, axis=1).mean(axis=1)


def _node_spectra_over_nodes(time_series):
    return _rfft_amplitude(time_series).T


def _plot_node_matrix_heatmap(matrix, x_tick_idx, x_tick_labels, xlabel, title, output_path, cbar_label, log_scale=False):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    num_nodes = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(13, max(4.0, 0.18 * num_nodes + 2.0)))
    if log_scale:
        positive = matrix[matrix > 0]
        vmin = positive.min() if positive.size else 1e-6
        vmax = matrix.max() if matrix.max() > 0 else 1.0
        im = ax.imshow(
            np.maximum(matrix, vmin),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        vmax = matrix.max() if matrix.size and matrix.max() > 0 else 1.0
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_ylabel("node")
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(num_nodes))
    ax.set_yticklabels([str(i) for i in range(num_nodes)])
    if x_tick_labels is not None:
        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.9, label=cbar_label)
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def _per_node_class_separation(groups, stage_key):
    traces = []
    labels = []
    group_labels = sorted(groups.keys())
    label_to_idx = {label: idx for idx, label in enumerate(group_labels)}
    for label, stages in zip(group_labels, [groups[l] for l in group_labels]):
        trace = _node_activity_trace(stages, stage_key)
        traces.append(trace)
        labels.append(np.full(trace.shape[1], label_to_idx[label], dtype=np.int64))
    data = np.concatenate(traces, axis=1)
    labels = np.concatenate(labels)
    num_nodes = data.shape[-1]
    scores = np.zeros(num_nodes, dtype=np.float64)
    for node in range(num_nodes):
        node_values = data[:, :, node].mean(axis=0)
        per_group = []
        for label in group_labels:
            label_idx = label_to_idx[label]
            other = np.concatenate(
                [node_values[labels == label_to_idx[other_label]] for other_label in group_labels if other_label != label]
            )
            own = node_values[labels == label_idx]
            per_group.append(abs(_cohens_d(own, other)))
        scores[node] = np.nanmean(per_group)
    return scores


def _top_node_time_series(groups, stage_key, num_examples=3):
    digit = sorted(groups.keys())[0]
    trace = _node_activity_trace(groups[digit], stage_key)
    class_means = _class_mean_node_traces(groups, stage_key)
    n_examples = min(num_examples, trace.shape[1])
    t = np.arange(trace.shape[0])
    return t, trace[:, :n_examples, :], class_means[digit], digit


def plot_node_activity_analysis(groups, output_dir, epoch, prefix, title_suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    num_timesteps = next(iter(groups.values()))["z_magnitude"].shape[0]
    num_nodes = next(iter(groups.values()))["z_magnitude"].shape[-1]
    freqs = np.fft.rfftfreq(num_timesteps, d=1.0)
    freq_tick_idx, freq_tick_labels = _ticks_for_freqs(freqs)
    time_tick_idx = np.linspace(0, num_timesteps - 1, min(8, num_timesteps), dtype=int)
    time_tick_labels = [str(i) for i in time_tick_idx]

    node_summary = {}
    for stage_key, stage_label in NODE_ACTIVITY_STAGES:
        safe_key = stage_key.replace("z_", "")
        pooled = _pooled_node_trace(groups, stage_key)
        class_means = _class_mean_node_traces(groups, stage_key)
        class_stack = np.stack([class_means[label] for label in sorted(class_means.keys())], axis=0)

        activity_map = pooled.T
        class_diff_map = class_stack.std(axis=0).T
        spectrum_map = _node_spectra_over_nodes(pooled)
        class_spectra = np.stack([_rfft_amplitude(class_means[label]) for label in sorted(class_means.keys())], axis=0)
        spectrum_class_diff = class_spectra.std(axis=0).T

        suffix = f" ({title_suffix})" if title_suffix else ""
        _plot_node_matrix_heatmap(
            activity_map,
            time_tick_idx,
            time_tick_labels,
            "time step",
            f"{stage_label}: mean activity per node (all digits){suffix}",
            os.path.join(output_dir, _epoch_filename(f"{prefix}_node_activity_{safe_key}", epoch)),
            "mean activity",
            log_scale=True,
        )
        _plot_node_matrix_heatmap(
            class_diff_map,
            time_tick_idx,
            time_tick_labels,
            "time step",
            f"{stage_label}: cross-digit spread per node (where classes diverge){suffix}",
            os.path.join(output_dir, _epoch_filename(f"{prefix}_node_class_diff_{safe_key}", epoch)),
            "std across digit means",
        )
        _plot_node_matrix_heatmap(
            spectrum_map,
            freq_tick_idx,
            freq_tick_labels,
            "frequency",
            f"{stage_label}: frequency content per node (pooled){suffix}",
            os.path.join(output_dir, _epoch_filename(f"{prefix}_node_spectrum_{safe_key}", epoch)),
            "FFT amplitude",
            log_scale=True,
        )
        _plot_node_matrix_heatmap(
            spectrum_class_diff,
            freq_tick_idx,
            freq_tick_labels,
            "frequency",
            f"{stage_label}: cross-digit spectral spread per node{suffix}",
            os.path.join(output_dir, _epoch_filename(f"{prefix}_node_spectrum_class_diff_{safe_key}", epoch)),
            "std across digit spectra",
        )

        separation = _per_node_class_separation(groups, stage_key)
        activity_score = pooled.mean(axis=0)
        node_summary[stage_key] = {
            "top_active_nodes": [int(i) for i in np.argsort(activity_score)[::-1][:5]],
            "top_discriminative_nodes": [int(i) for i in np.argsort(separation)[::-1][:5]],
            "mean_class_separation": _json_float(np.nanmean(separation)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.15 * num_nodes + 2.0)))
        order = np.argsort(activity_score)[::-1]
        axes[0].barh([str(i) for i in order[::-1]], activity_score[order[::-1]], color=thesis_blue, alpha=0.85)
        axes[0].set_xlabel("mean activity")
        axes[0].set_title(f"{stage_label}: node activity ranking")
        order = np.argsort(separation)[::-1]
        axes[1].barh([str(i) for i in order[::-1]], separation[order[::-1]], color=ifisc_green, alpha=0.85)
        axes[1].set_xlabel("mean |one-vs-rest Cohen's $d$|")
        axes[1].set_title(f"{stage_label}: node class-separation ranking")
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, _epoch_filename(f"{prefix}_node_ranking_{safe_key}", epoch)),
            transparent=True,
        )
        plt.close(fig)

        top_nodes = node_summary[stage_key]["top_active_nodes"][: min(8, num_nodes)]
        t, examples, class_mean, digit = _top_node_time_series(groups, stage_key)
        fig, axes = plt.subplots(len(top_nodes), 1, figsize=(12, 2.2 * len(top_nodes)), sharex=True, squeeze=False)
        for row, node in enumerate(top_nodes):
            ax = axes[row, 0]
            for col in range(examples.shape[1]):
                ax.plot(t, examples[:, col, node], color=thesis_blue, alpha=0.35, linewidth=1.0)
            ax.plot(t, class_mean[:, node], color=thesis_red, linewidth=2.0, label="digit mean")
            ax.set_ylabel(f"node {node}")
            values = np.concatenate([examples[:, :, node].reshape(-1), class_mean[:, node]])
            margin = (values.max() - values.min()) * 0.05 if values.max() > values.min() else 0.05
            ax.set_ylim(values.min() - margin, values.max() + margin)
        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("time step")
        fig.suptitle(f"{stage_label}: top active nodes (digit {digit} examples){suffix}", y=1.01)
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, _epoch_filename(f"{prefix}_node_timeseries_top_{safe_key}", epoch)),
            transparent=True,
        )
        plt.close(fig)

    return node_summary


def save_signal_analysis_summary(summary, output_dir, epoch, prefix):
    with open(os.path.join(output_dir, _epoch_filename(f"{prefix}_summary", epoch, suffix="json")), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
