import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .signal_stages import _epoch_filename, _unit_indices
from .sentiment_encoding import (
    _cohens_d,
    _example_features,
    _mean_over_units,
    _mean_std_spectrum,
    _plot_mean_std_band,
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
    for key in ("pre_activation", "z_real", "z_magnitude", "logits"):
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


def _one_vs_rest_cohens_d(features_by_class, feature_name, target_digit, all_digits):
    target = features_by_class[target_digit][feature_name]
    other = np.concatenate(
        [features_by_class[d][feature_name] for d in all_digits if d != target_digit]
    )
    return _cohens_d(target, other)


def _mean_logits_matrix(stages_by_class, all_digits, num_classes):
    matrix = np.zeros((len(all_digits), num_classes))
    for i, digit in enumerate(all_digits):
        matrix[i] = stages_by_class[digit]["logits"].mean(axis=0)
    return matrix


def plot_digit_encoding_analysis(
    stages_by_class,
    labels_by_class,
    output_dir,
    epoch,
    num_units_plot=5,
    num_classes=10,
):
    os.makedirs(output_dir, exist_ok=True)
    all_digits = sorted(stages_by_class.keys())
    num_timesteps = next(iter(stages_by_class.values()))["raw_input"].shape[0]
    num_nodes = next(iter(stages_by_class.values()))["pre_activation"].shape[-1]
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    freqs = np.fft.rfftfreq(num_timesteps, d=1.0)

    features_by_class = _features_by_class(stages_by_class, labels_by_class, unit_indices)
    dynamics_finite = all(_stages_dynamics_finite(stages_by_class[d]) for d in all_digits)

    stat_features = [
        "final_z_magnitude",
        "mean_z_magnitude",
        "final_z_real",
        "mean_z_real",
        "mean_pre_activation",
        "spectral_centroid_z_magnitude",
        "low_high_power_ratio_z_magnitude",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, stat_features):
        means = [np.nanmean(features_by_class[d][feat]) for d in all_digits]
        stds = [np.nanstd(features_by_class[d][feat]) for d in all_digits]
        ax.bar(all_digits, means, yerr=stds, color=[DIGIT_COLORS[d] for d in all_digits], alpha=0.85, capsize=3)
        ax.set_title(feat.replace("_", " "))
        ax.set_xlabel("digit")
        ax.set_xticks(all_digits)
    fig.suptitle(
        "pre-readout statistics per digit (mean $\\pm$ std over examples)"
        + ("" if dynamics_finite else " [dynamics non-finite]")
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename("digit_pre_readout_stats", epoch)), transparent=True)
    plt.close(fig)

    separation = {}
    for feat in features_by_class[all_digits[0]].keys():
        if feat in ("predicted_class", "correct"):
            continue
        separation[feat] = [_one_vs_rest_cohens_d(features_by_class, feat, d, all_digits) for d in all_digits]

    ranked_features = sorted(
        separation.keys(),
        key=lambda f: max(abs(v) for v in separation[f]),
        reverse=True,
    )
    top_features = ranked_features[:8]
    heatmap = np.array([separation[f] for f in top_features])

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(heatmap)), vmax=np.max(np.abs(heatmap)))
    ax2.set_xticks(range(len(all_digits)))
    ax2.set_xticklabels([str(d) for d in all_digits])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features)
    ax2.set_xlabel("digit (one-vs-rest Cohen's $d$)")
    ax2.set_title("which signal properties separate each digit from all others?")
    fig2.colorbar(im, ax=ax2, shrink=0.9, label="Cohen's $d$")
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, _epoch_filename("digit_encoding_separation", epoch)), transparent=True)
    plt.close(fig2)

    n_digits = len(all_digits)
    n_cols = min(5, n_digits)
    n_rows = int(np.ceil(n_digits / n_cols))
    fig3, axes3 = plt.subplots(2 * n_rows, n_cols, figsize=(2.8 * n_cols, 4 * n_rows), sharex=True, squeeze=False)
    stage_keys = [("raw_input", "pixel value"), ("z_magnitude", r"$|z(t)|$")]
    t = np.arange(num_timesteps)
    for idx, digit in enumerate(all_digits):
        col = idx % n_cols
        block = idx // n_cols
        color = DIGIT_COLORS[digit]
        stages = stages_by_class[digit]
        for row, (key, ylabel) in enumerate(stage_keys):
            ax = axes3[block * 2 + row, col]
            if key == "raw_input":
                data = stages[key]
            else:
                data = _mean_over_units(stages[key], unit_indices)
            for ex in range(data.shape[1]):
                ax.plot(data[:, ex], color=color, alpha=0.35, linewidth=0.9)
            ax.plot(data.mean(axis=1), color=color, linewidth=2.0)
            ax.set_title(f"digit {digit}")
            if col == 0:
                ax.set_ylabel(ylabel)
            if block == n_rows - 1:
                ax.set_xlabel("time step")
    for idx in range(n_digits, n_rows * n_cols):
        col = idx % n_cols
        block = idx // n_cols
        for row in range(2):
            axes3[block * 2 + row, col].axis("off")
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, _epoch_filename("digit_encoding_examples", epoch)), transparent=True)
    plt.close(fig3)

    fig3b, axes3b = plt.subplots(1, len(all_digits), figsize=(2.2 * len(all_digits), 3.5), sharex=True, sharey=True)
    if len(all_digits) == 1:
        axes3b = [axes3b]
    for col, digit in enumerate(all_digits):
        zr = _mean_over_units(stages_by_class[digit]["z_real"], unit_indices)
        for ex in range(zr.shape[1]):
            axes3b[col].plot(zr[:, ex], color=DIGIT_COLORS[digit], alpha=0.35, linewidth=0.9)
        axes3b[col].plot(zr.mean(axis=1), color=DIGIT_COLORS[digit], linewidth=2.0)
        axes3b[col].set_title(f"digit {digit}")
        axes3b[col].set_xlabel("time step")
    axes3b[0].set_ylabel(r"$\Re(z)$ before readout")
    plt.tight_layout()
    fig3b.savefig(os.path.join(output_dir, _epoch_filename("digit_encoding_zreal_examples", epoch)), transparent=True)
    plt.close(fig3b)

    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    spectra_specs = [
        ("raw_input", "pixel"),
        ("z_magnitude", r"$|z|$"),
    ]
    for ax, (key, label) in zip(axes4.flat[:2], spectra_specs):
        for digit in all_digits:
            if key == "raw_input":
                neg_mean, neg_std = _mean_std_spectrum(stages_by_class[digit][key], unit_indices)
            else:
                neg_mean, neg_std = _mean_std_spectrum(stages_by_class[digit][key], unit_indices)
            _plot_mean_std_band(ax, freqs, neg_mean, neg_std, DIGIT_COLORS[digit], f"digit {digit}")
        ax.set_title(f"{label}: mean spectrum per digit")
        ax.set_xlabel("frequency")
        ax.set_ylabel("amplitude")

    zm_by_digit = {d: np.mean(features_by_class[d]["final_z_magnitude"]) for d in all_digits}
    margin_by_digit = {d: np.mean(features_by_class[d]["logit_margin"]) for d in all_digits}
    axes4[1, 0].bar(all_digits, [zm_by_digit[d] for d in all_digits], color=[DIGIT_COLORS[d] for d in all_digits], alpha=0.85)
    axes4[1, 0].set_xlabel("digit")
    axes4[1, 0].set_ylabel("final $|z|$")
    axes4[1, 0].set_title("final magnitude before readout")
    axes4[1, 1].bar(all_digits, [margin_by_digit[d] for d in all_digits], color=[DIGIT_COLORS[d] for d in all_digits], alpha=0.85)
    axes4[1, 1].set_xlabel("digit")
    axes4[1, 1].set_ylabel("logit margin")
    axes4[1, 1].set_title("decision confidence (correct $-$ best wrong logit)")
    axes4[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, _epoch_filename("digit_encoding_spectra", epoch)), transparent=True)
    plt.close(fig4)

    logit_matrix = _mean_logits_matrix(stages_by_class, all_digits, num_classes)
    if not dynamics_finite:
        logit_matrix = np.where(np.isfinite(logit_matrix), logit_matrix, 0.0)

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    im5 = ax5.imshow(logit_matrix, aspect="auto", cmap="coolwarm")
    ax5.set_xticks(range(num_classes))
    ax5.set_xticklabels([str(i) for i in range(num_classes)])
    ax5.set_yticks(range(len(all_digits)))
    ax5.set_yticklabels([str(d) for d in all_digits])
    ax5.set_xlabel("readout logit (class)")
    ax5.set_ylabel("true digit")
    ax5.set_title(
        "mean readout logits: how each digit's signal maps to the decision"
        + ("" if dynamics_finite else " [NaN logits replaced for display]")
    )
    for i in range(len(all_digits)):
        for j in range(num_classes):
            ax5.text(j, i, f"{logit_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="0.1")
    fig5.colorbar(im5, ax=ax5, shrink=0.85, label="logit")
    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, _epoch_filename("digit_readout_mapping", epoch)), transparent=True)
    plt.close(fig5)

    probs_matrix = torch.softmax(torch.tensor(np.where(np.isfinite(logit_matrix), logit_matrix, -1e9)), dim=1).numpy()
    fig5b, ax5b = plt.subplots(figsize=(10, 8))
    im5b = ax5b.imshow(probs_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax5b.set_xticks(range(num_classes))
    ax5b.set_xticklabels([str(i) for i in range(num_classes)])
    ax5b.set_yticks(range(len(all_digits)))
    ax5b.set_yticklabels([str(d) for d in all_digits])
    ax5b.set_xlabel("predicted class probability")
    ax5b.set_ylabel("true digit")
    ax5b.set_title("mean softmax probabilities per true digit")
    fig5b.colorbar(im5b, ax=ax5b, shrink=0.85, label="probability")
    plt.tight_layout()
    fig5b.savefig(os.path.join(output_dir, _epoch_filename("digit_readout_probs", epoch)), transparent=True)
    plt.close(fig5b)

    fig6, axes6 = plt.subplots(2, 3, figsize=(15, 8))
    scatter_keys = [
        ("final_z_magnitude", "final $|z|$"),
        ("mean_z_magnitude", "mean $|z|$"),
        ("final_z_real", "final $\Re(z)$"),
        ("spectral_centroid_z_magnitude", "spectral centroid of $|z|$"),
        ("low_high_power_ratio_z_magnitude", "low/high power ratio"),
        ("mean_pre_activation", "mean pre-activation"),
    ]
    for ax, (key, xlabel) in zip(axes6.flat, scatter_keys):
        for digit in all_digits:
            ax.scatter(
                features_by_class[digit][key],
                features_by_class[digit]["logit_margin"],
                color=DIGIT_COLORS[digit],
                alpha=0.8,
                label=str(digit),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("logit margin")
        ax.set_title(f"decision vs {xlabel}")
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axes6[0, 0].legend(title="digit", fontsize=8, ncol=2, loc="best")
    plt.tight_layout()
    fig6.savefig(os.path.join(output_dir, _epoch_filename("digit_encoding_decision_drivers", epoch)), transparent=True)
    plt.close(fig6)

    magnitude_keys = [k for k in ranked_features if "magnitude" in k or "z_real" in k or "input" in k]
    frequency_keys = [k for k in ranked_features if "spectral" in k or "power_ratio" in k]
    per_digit_dominant = {}
    for digit in all_digits:
        mag = np.mean([abs(separation[k][all_digits.index(digit)]) for k in magnitude_keys]) if magnitude_keys else 0.0
        freq = np.mean([abs(separation[k][all_digits.index(digit)]) for k in frequency_keys]) if frequency_keys else 0.0
        per_digit_dominant[str(digit)] = "magnitude" if mag >= freq else "frequency"

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
        "one_vs_rest_cohens_d": {
            f: {str(d): _json_float(separation[f][i]) for i, d in enumerate(all_digits)} for f in top_features
        },
        "top_separating_features_per_digit": {
            str(d): {
                "feature": max(
                    ranked_features,
                    key=lambda f: abs(separation[f][all_digits.index(d)]) if np.isfinite(separation[f][all_digits.index(d)]) else -1,
                ),
                "cohens_d": _json_float(
                    separation[max(
                        ranked_features,
                        key=lambda f: abs(separation[f][all_digits.index(d)]) if np.isfinite(separation[f][all_digits.index(d)]) else -1,
                    )][all_digits.index(d)]
                ),
            }
            for d in all_digits
        },
        "per_digit_dominant_encoding": per_digit_dominant,
        "mean_logits_matrix": {
            str(d): [_json_float(v) for v in logit_matrix[i]] for i, d in enumerate(all_digits)
        },
    }
    if not dynamics_finite:
        summary["warning"] = (
            "Oscillator states or logits are non-finite. This usually means unstable dynamics "
            "(e.g. lambda_param > 0 or diverged weights). Input-only features may still be valid."
        )
    with open(os.path.join(output_dir, _epoch_filename("digit_encoding_summary", epoch, suffix="json")), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


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
