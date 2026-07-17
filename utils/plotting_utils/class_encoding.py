import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .signal_stages import _epoch_filename, _unit_indices
from .sentiment_encoding import (
    _example_features,
    find_class_examples_batch,
    trace_model_signal_stages_batch,
)


DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def prepare_smnist_batch(images):
    if images.dim() == 4:
        images = images.reshape(images.size(0), 1, 784)
    elif images.dim() == 2:
        images = images.unsqueeze(1)
    return images.permute(2, 0, 1)


def _multiclass_logit_margin(logits, labels):
    labels = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(logits).any():
        return np.full(len(labels), np.nan)
    correct = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    best_wrong = masked.max(axis=1)
    margin = correct - best_wrong
    margin[~np.isfinite(logits).any(axis=1)] = np.nan
    return margin


def _predicted_classes(logits):
    preds = np.full(logits.shape[0], -1, dtype=np.int64)
    finite_rows = np.isfinite(logits).any(axis=1)
    if finite_rows.any():
        preds[finite_rows] = logits[finite_rows].argmax(axis=1)
    return preds


def _stages_dynamics_finite(stages):
    for key in ("pre_activation", "z_real", "z_imag", "z_magnitude", "logits"):
        if key in stages and not np.isfinite(stages[key]).all():
            return False
    return True


def _json_float(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def _example_features_multiclass(stages, unit_indices, labels):
    features = _example_features(stages, unit_indices)
    labels = np.asarray(labels, dtype=np.int64)
    logits = stages["logits"]
    features.pop("positive_prob", None)
    if np.isfinite(logits).any():
        features["correct_logit"] = logits[np.arange(len(labels)), labels]
    else:
        features["correct_logit"] = np.full(len(labels), np.nan)
    features["logit_margin"] = _multiclass_logit_margin(logits, labels)
    features["predicted_class"] = _predicted_classes(logits)
    features["correct"] = (features["predicted_class"] == labels).astype(np.float64)
    features["correct"][features["predicted_class"] < 0] = np.nan
    return features


def _collect_class_stages(model, class_batches, input_mode="scalar"):
    stages_by_class = {}
    labels_by_class = {}
    for digit, images in class_batches.items():
        inputs = prepare_smnist_batch(images)
        stages_by_class[digit] = trace_model_signal_stages_batch(model, inputs, input_mode=input_mode)
        labels_by_class[digit] = np.full(images.size(0), digit, dtype=np.int64)
    return stages_by_class, labels_by_class


def _features_by_class(stages_by_class, labels_by_class, unit_indices):
    return {
        digit: _example_features_multiclass(stages, unit_indices, labels_by_class[digit])
        for digit, stages in stages_by_class.items()
    }


def plot_digit_encoding_analysis(
    stages_by_class,
    labels_by_class,
    output_dir,
    epoch,
    num_units_plot=5,
    num_classes=10,
):
    from .signal_analysis import (
        MAX_EXAMPLE_TIME_SERIES,
        plot_autocorrelation_analysis,
        plot_decision_drivers,
        plot_granular_stage_spectra,
        plot_node_activity_analysis,
        plot_phase_vs_magnitude_separation,
        plot_real_imag_spectrum_comparison,
        save_signal_analysis_summary,
        _plot_example_time_series,
    )

    os.makedirs(output_dir, exist_ok=True)
    all_digits = sorted(stages_by_class.keys())
    num_nodes = next(iter(stages_by_class.values()))["pre_activation"].shape[-1]
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    colors = {d: DIGIT_COLORS[d] for d in all_digits}

    features_by_class = _features_by_class(stages_by_class, labels_by_class, unit_indices)
    dynamics_finite = all(_stages_dynamics_finite(stages_by_class[d]) for d in all_digits)
    examples_per_digit = stages_by_class[all_digits[0]]["logits"].shape[0]

    plot_granular_stage_spectra(
        stages_by_class,
        output_dir,
        epoch,
        prefix="digit",
        unit_indices=unit_indices,
        colors=colors,
        title_suffix="sMNIST digit encoding: spectra at each processing stage",
    )
    plot_real_imag_spectrum_comparison(stages_by_class, output_dir, epoch, "digit", unit_indices, colors)
    node_summary = plot_node_activity_analysis(
        stages_by_class,
        output_dir,
        epoch,
        prefix="digit",
        title_suffix="sMNIST digit encoding",
    )
    plot_autocorrelation_analysis(
        stages_by_class,
        output_dir,
        epoch,
        prefix="digit",
        unit_indices=unit_indices,
        colors=colors,
        title_suffix="sMNIST digit encoding: temporal autocorrelation",
    )

    if examples_per_digit <= MAX_EXAMPLE_TIME_SERIES:
        for stage_key, ylabel in [("z_real", r"$\Re(z)$"), ("z_imag", r"$\Im(z)$"), ("z_magnitude", r"$|z|$")]:
            _plot_example_time_series(
                stages_by_class,
                stage_key,
                ylabel,
                os.path.join(output_dir, _epoch_filename(f"digit_examples_{stage_key.replace('z_', '')}", epoch)),
                colors,
                unit_indices,
            )

    separation_summary = plot_phase_vs_magnitude_separation(
        features_by_class,
        output_dir,
        epoch,
        prefix="digit",
        compare_mode="multiclass",
    )

    plot_decision_drivers(features_by_class, output_dir, epoch, prefix="digit", colors=colors)

    summary = {
        "dynamics_finite": dynamics_finite,
        "num_digits": len(all_digits),
        "examples_per_digit": {str(d): int(stages_by_class[d]["logits"].shape[0]) for d in all_digits},
        "per_digit_accuracy": {
            str(d): _json_float(np.nanmean(features_by_class[d]["correct"])) for d in all_digits
        },
        "per_digit_mean_logit_margin": {
            str(d): _json_float(np.nanmean(features_by_class[d]["logit_margin"])) for d in all_digits
        },
        "per_digit_mean_final_z_magnitude": {
            str(d): _json_float(np.nanmean(features_by_class[d]["final_z_magnitude"])) for d in all_digits
        },
        "per_digit_mean_final_z_imag": {
            str(d): _json_float(np.nanmean(features_by_class[d]["final_z_imag"])) for d in all_digits
        },
        "node_activity": node_summary,
        **separation_summary,
    }
    if not dynamics_finite:
        summary["warning"] = (
            "Oscillator states or logits are non-finite. This usually means unstable dynamics "
            "(e.g. lambda_param > 0 or diverged weights)."
        )
    return save_signal_analysis_summary(summary, output_dir, epoch, "digit")


@torch.no_grad()
def plot_smnist_digit_encoding_analysis(
    model,
    class_batches,
    output_dir,
    epoch,
    num_units_plot=5,
    num_classes=10,
):
    model.eval()
    stages_by_class, labels_by_class = _collect_class_stages(model, class_batches, input_mode="scalar")
    return plot_digit_encoding_analysis(
        stages_by_class,
        labels_by_class,
        output_dir,
        epoch,
        num_units_plot=num_units_plot,
        num_classes=num_classes,
    )
