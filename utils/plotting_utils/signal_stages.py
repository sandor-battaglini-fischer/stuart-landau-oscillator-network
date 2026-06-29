import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .helpers import set_epoch_xlim, set_value_ylim
from .style import thesis_blue, thesis_red, ifisc_green


def get_slon_core(model):
    if hasattr(model, "slon"):
        return model.slon
    return model


def prepare_imdb_sequence(model, token_ids):
    embedded = model.embedding(token_ids)
    embedded = embedded / math.sqrt(model.embedding.embedding_dim)
    embedded = model.dropout(embedded)
    embedded = embedded * 3.0
    return embedded.permute(1, 0, 2)


def compute_raw_input_trace(inputs, example_idx=0, input_mode="scalar"):
    if input_mode == "norm":
        return inputs[:, example_idx, :].norm(dim=-1).detach().cpu().numpy()
    return inputs[:, example_idx, 0].detach().cpu().numpy()


@torch.no_grad()
def compute_signal_stages(slon, inputs, example_idx=0, input_mode="scalar"):
    num_timesteps = inputs.size(0)
    batch_size = inputs.size(1)

    z_real = torch.zeros(batch_size, slon.num_nodes, device=inputs.device, dtype=inputs.dtype)
    z_imag = torch.zeros(batch_size, slon.num_nodes, device=inputs.device, dtype=inputs.dtype)

    projected_inputs = []
    pre_activations = []
    input_forces = []
    z_real_trace = []
    z_imag_trace = []
    z_mag_trace = []

    for t in range(num_timesteps):
        input_t = inputs[t]
        projected = slon.i2h(input_t)
        z_state = torch.cat([z_real, z_imag], dim=1)
        pre_act = projected + slon.gain_rec * slon.h2h(z_state)
        input_force = slon.alpha * torch.tanh(pre_act)

        z_real, z_imag = slon.dynamics_step(z_real, z_imag, input_t)

        projected_inputs.append(projected[example_idx].detach().cpu())
        pre_activations.append(pre_act[example_idx].detach().cpu())
        input_forces.append(input_force[example_idx].detach().cpu())
        z_real_trace.append(z_real[example_idx].detach().cpu())
        z_imag_trace.append(z_imag[example_idx].detach().cpu())
        z_mag_trace.append(torch.sqrt(z_real[example_idx] ** 2 + z_imag[example_idx] ** 2).detach().cpu())

    z_features = torch.cat([z_real, z_imag], dim=1)
    logits = slon.h2o(z_features[example_idx : example_idx + 1]).squeeze(0).detach().cpu()
    raw_input = compute_raw_input_trace(inputs, example_idx=example_idx, input_mode=input_mode)

    return {
        "raw_input": raw_input,
        "projected_input": torch.stack(projected_inputs).numpy(),
        "pre_activation": torch.stack(pre_activations).numpy(),
        "input_force": torch.stack(input_forces).numpy(),
        "z_real": torch.stack(z_real_trace).numpy(),
        "z_imag": torch.stack(z_imag_trace).numpy(),
        "z_magnitude": torch.stack(z_mag_trace).numpy(),
        "logits": logits.numpy(),
    }


@torch.no_grad()
def trace_model_signal_stages(model, inputs, example_idx=0, input_mode="scalar"):
    slon = get_slon_core(model)
    return compute_signal_stages(slon, inputs, example_idx=example_idx, input_mode=input_mode)


def _unit_indices(num_nodes, num_units_plot):
    return list(range(min(num_units_plot, num_nodes)))


def _plot_unit_traces(ax, t, series, unit_indices, ylabel, title):
    for idx in unit_indices:
        ax.plot(t, series[:, idx], label=f"unit {idx}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(unit_indices) > 1:
        ax.legend(loc="upper right", fontsize=12)
    set_epoch_xlim(ax, len(t))
    set_value_ylim(ax, series[:, unit_indices].reshape(-1))


def _plot_output_panel(ax, logits, task_type, class_labels=None, true_label=None, target_value=None):
    if task_type == "regression":
        pred = float(logits.reshape(-1)[0])
        ax.bar(["prediction"], [pred], color=thesis_blue, alpha=0.85)
        if target_value is not None:
            ax.axhline(target_value, color=thesis_red, linestyle="--", linewidth=1.5, label=f"target = {target_value:.4f}")
            ax.legend(loc="upper right", fontsize=12)
        ax.set_ylabel("output")
        ax.set_title("final prediction")
        set_value_ylim(ax, [pred, target_value] if target_value is not None else [pred])
        return

    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    labels = class_labels if class_labels is not None else [f"class {i}" for i in range(len(probs))]
    colors = [thesis_blue if (true_label is None or i != true_label) else ifisc_green for i in range(len(probs))]
    ax.bar(labels, probs, color=colors, alpha=0.85)
    ax.set_ylabel("softmax probability")
    ax.set_title("final logits / probabilities")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30)

    for i, (logit, prob) in enumerate(zip(logits, probs)):
        ax.text(i, prob + 0.02, f"logit={logit:.2f}", ha="center", va="bottom", fontsize=11)


def plot_signal_stages(
    stages,
    output_dir,
    epoch,
    task_type="classification",
    class_labels=None,
    true_label=None,
    target_value=None,
    num_units_plot=5,
    raw_input_label="input",
    title_suffix=None,
):
    os.makedirs(output_dir, exist_ok=True)

    num_timesteps, num_nodes = stages["pre_activation"].shape
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    t = np.arange(num_timesteps)

    fig, axes = plt.subplots(5, 1, figsize=(12, 17), sharex=True)

    axes[0].plot(t, stages["raw_input"], color=thesis_blue, linewidth=1.5)
    axes[0].set_ylabel(raw_input_label)
    axes[0].set_title("raw input")
    set_epoch_xlim(axes[0], num_timesteps)
    set_value_ylim(axes[0], stages["raw_input"])

    _plot_unit_traces(
        axes[1],
        t,
        stages["pre_activation"],
        unit_indices,
        r"pre-activation $u(t)$",
        "pre-activation (before tanh)",
    )

    _plot_unit_traces(
        axes[2],
        t,
        stages["z_real"],
        unit_indices,
        r"$\Re(z(t))$",
        "node state real part (before readout)",
    )

    _plot_unit_traces(
        axes[3],
        t,
        stages["z_imag"],
        unit_indices,
        r"$\Im(z(t))$",
        "node state imaginary part (before readout)",
    )

    _plot_output_panel(
        axes[4],
        stages["logits"],
        task_type=task_type,
        class_labels=class_labels,
        true_label=true_label,
        target_value=target_value,
    )
    axes[4].set_xlabel("class")

    if title_suffix:
        fig.suptitle(title_suffix, y=1.01)

    plt.tight_layout()
    filename = f"signal_stages_epoch{epoch:02d}.png" if epoch is not None else "signal_stages.png"
    fig.savefig(os.path.join(output_dir, filename), transparent=True)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    _plot_unit_traces(
        axes2[0],
        t,
        stages["projected_input"],
        unit_indices,
        r"$W_{i2h}x(t)$",
        "projected input (no recurrence)",
    )
    _plot_unit_traces(
        axes2[1],
        t,
        stages["input_force"],
        unit_indices,
        r"$\alpha\tanh(u(t))$",
        "driving force into oscillator",
    )
    axes2[1].set_xlabel("time step")
    plt.tight_layout()
    detail_name = f"signal_stages_input_force_epoch{epoch:02d}.png" if epoch is not None else "signal_stages_input_force.png"
    fig2.savefig(os.path.join(output_dir, detail_name), transparent=True)
    plt.close(fig2)


def plot_signal_stages_for_example(
    model,
    inputs,
    output_dir,
    epoch,
    example_idx=0,
    input_mode="scalar",
    task_type="classification",
    class_labels=None,
    true_label=None,
    target_value=None,
    num_units_plot=5,
    raw_input_label="input",
    title_suffix=None,
):
    model.eval()
    stages = trace_model_signal_stages(model, inputs, example_idx=example_idx, input_mode=input_mode)
    plot_signal_stages(
        stages,
        output_dir,
        epoch,
        task_type=task_type,
        class_labels=class_labels,
        true_label=true_label,
        target_value=target_value,
        num_units_plot=num_units_plot,
        raw_input_label=raw_input_label,
        title_suffix=title_suffix,
    )


def find_class_examples(data_loader, labels=(0, 1), max_batches=None):
    found = {}
    for batch_idx, (token_ids, batch_labels) in enumerate(data_loader):
        for label in labels:
            if label in found:
                continue
            matches = (batch_labels == label).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                found[label] = token_ids[matches[0] : matches[0] + 1].clone()
        if len(found) == len(labels):
            break
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    missing = [label for label in labels if label not in found]
    if missing:
        raise ValueError(f"Could not find examples for labels: {missing}")
    return found


def _epoch_filename(prefix, epoch, suffix="png"):
    if epoch is None:
        return f"{prefix}.{suffix}"
    return f"{prefix}_epoch{epoch:02d}.{suffix}"


def _plot_sentiment_overlay(ax, t, neg_trace, pos_trace, ylabel, title, neg_label="negative", pos_label="positive"):
    ax.plot(t, neg_trace, color=thesis_red, linewidth=1.8, label=neg_label)
    ax.plot(t, pos_trace, color=ifisc_green, linewidth=1.8, label=pos_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=12)
    set_epoch_xlim(ax, len(t))
    set_value_ylim(ax, np.concatenate([neg_trace, pos_trace]))


def _plot_sentiment_difference(ax, t, diff_trace, ylabel, title):
    ax.plot(t, diff_trace, color=thesis_blue, linewidth=1.8, label="positive $-$ negative")
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.fill_between(t, 0.0, diff_trace, where=diff_trace >= 0, color=ifisc_green, alpha=0.2)
    ax.fill_between(t, 0.0, diff_trace, where=diff_trace < 0, color=thesis_red, alpha=0.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=12)
    set_epoch_xlim(ax, len(t))
    set_value_ylim(ax, diff_trace)


def _plot_unit_sentiment_overlay(ax, t, neg_series, pos_series, unit_indices, ylabel, title):
    for idx in unit_indices:
        ax.plot(t, neg_series[:, idx], color=thesis_red, linewidth=1.2, alpha=0.85, label=f"neg unit {idx}")
        ax.plot(t, pos_series[:, idx], color=ifisc_green, linewidth=1.2, alpha=0.85, label=f"pos unit {idx}")
    if len(unit_indices) > 1:
        ax.plot(
            t,
            neg_series[:, unit_indices].mean(axis=1),
            color=thesis_red,
            linewidth=2.4,
            linestyle="--",
            label="neg mean",
        )
        ax.plot(
            t,
            pos_series[:, unit_indices].mean(axis=1),
            color=ifisc_green,
            linewidth=2.4,
            linestyle="--",
            label="pos mean",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    set_epoch_xlim(ax, len(t))
    set_value_ylim(ax, np.concatenate([neg_series[:, unit_indices], pos_series[:, unit_indices]], axis=1).reshape(-1))


def _plot_unit_sentiment_difference(ax, t, diff_series, unit_indices, ylabel, title):
    for idx in unit_indices:
        ax.plot(t, diff_series[:, idx], linewidth=1.2, alpha=0.85, label=f"unit {idx}")
    if len(unit_indices) > 1:
        ax.plot(t, diff_series[:, unit_indices].mean(axis=1), color=thesis_blue, linewidth=2.4, label="mean")
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    set_epoch_xlim(ax, len(t))
    set_value_ylim(ax, diff_series[:, unit_indices].reshape(-1))


def _plot_readout_comparison(ax, neg_logits, pos_logits, class_labels):
    x = np.arange(len(class_labels))
    width = 0.35
    ax.bar(x - width / 2, neg_logits, width, color=thesis_red, alpha=0.85, label="negative review")
    ax.bar(x + width / 2, pos_logits, width, color=ifisc_green, alpha=0.85, label="positive review")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=20)
    ax.set_ylabel("logit (readout input)")
    ax.set_title("readout logits before softmax")
    ax.legend(loc="upper right", fontsize=12)
    set_value_ylim(ax, np.concatenate([neg_logits, pos_logits]))


def _sentiment_comparison_summary(neg_stages, pos_stages):
    input_diff = pos_stages["raw_input"] - neg_stages["raw_input"]
    pre_diff = pos_stages["pre_activation"] - neg_stages["pre_activation"]
    z_diff = pos_stages["z_real"] - neg_stages["z_real"]
    neg_probs = torch.softmax(torch.tensor(neg_stages["logits"]), dim=0).numpy()
    pos_probs = torch.softmax(torch.tensor(pos_stages["logits"]), dim=0).numpy()
    return {
        "mean_abs_input_diff": float(np.mean(np.abs(input_diff))),
        "mean_input_diff": float(np.mean(input_diff)),
        "final_mean_pre_activation_diff": float(np.mean(pre_diff[-1])),
        "final_mean_z_real_diff": float(np.mean(z_diff[-1])),
        "mean_abs_z_real_diff": float(np.mean(np.abs(z_diff))),
        "negative_logits": [float(x) for x in neg_stages["logits"]],
        "positive_logits": [float(x) for x in pos_stages["logits"]],
        "negative_probs": [float(x) for x in neg_probs],
        "positive_probs": [float(x) for x in pos_probs],
        "negative_predicted_class": int(np.argmax(neg_stages["logits"])),
        "positive_predicted_class": int(np.argmax(pos_stages["logits"])),
    }


def plot_sentiment_comparison(
    neg_stages,
    pos_stages,
    output_dir,
    epoch,
    num_units_plot=5,
    class_labels=None,
):
    os.makedirs(output_dir, exist_ok=True)
    class_labels = class_labels or ["Negative", "Positive"]

    num_timesteps, num_nodes = neg_stages["pre_activation"].shape
    unit_indices = _unit_indices(num_nodes, num_units_plot)
    t = np.arange(num_timesteps)

    pre_diff = pos_stages["pre_activation"] - neg_stages["pre_activation"]
    z_diff = pos_stages["z_real"] - neg_stages["z_real"]

    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    _plot_sentiment_overlay(
        axes[0],
        t,
        neg_stages["raw_input"],
        pos_stages["raw_input"],
        "embedding norm",
        "input: embedding norm over time",
    )
    _plot_sentiment_difference(
        axes[1],
        t,
        pos_stages["raw_input"] - neg_stages["raw_input"],
        "embedding norm diff",
        "input difference (positive $-$ negative)",
    )
    _plot_unit_sentiment_overlay(
        axes[2],
        t,
        neg_stages["pre_activation"],
        pos_stages["pre_activation"],
        unit_indices,
        r"pre-activation $u(t)$",
        "pre-activation before tanh",
    )
    _plot_unit_sentiment_difference(
        axes[3],
        t,
        pre_diff,
        unit_indices,
        r"$\Delta u(t)$",
        "pre-activation difference (positive $-$ negative)",
    )
    _plot_unit_sentiment_overlay(
        axes[4],
        t,
        neg_stages["z_real"],
        pos_stages["z_real"],
        unit_indices,
        r"$\Re(z(t))$",
        "node state before readout",
    )
    _plot_unit_sentiment_difference(
        axes[5],
        t,
        z_diff,
        unit_indices,
        r"$\Delta \Re(z(t))$",
        "node-state difference before readout",
    )
    axes[5].set_xlabel("time step")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, _epoch_filename("sentiment_comparison", epoch)), transparent=True)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_readout_comparison(axes2[0], neg_stages["logits"], pos_stages["logits"], class_labels)
    neg_probs = torch.softmax(torch.tensor(neg_stages["logits"]), dim=0).numpy()
    pos_probs = torch.softmax(torch.tensor(pos_stages["logits"]), dim=0).numpy()
    x = np.arange(len(class_labels))
    width = 0.35
    axes2[1].bar(x - width / 2, neg_probs, width, color=thesis_red, alpha=0.85, label="negative review")
    axes2[1].bar(x + width / 2, pos_probs, width, color=ifisc_green, alpha=0.85, label="positive review")
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(class_labels, rotation=20)
    axes2[1].set_ylabel("softmax probability")
    axes2[1].set_title("readout after softmax")
    axes2[1].set_ylim(0.0, 1.0)
    axes2[1].legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, _epoch_filename("sentiment_readout_comparison", epoch)), transparent=True)
    plt.close(fig2)

    if num_nodes > 1:
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        vmin_pre = min(neg_stages["pre_activation"].min(), pos_stages["pre_activation"].min())
        vmax_pre = max(neg_stages["pre_activation"].max(), pos_stages["pre_activation"].max())
        vmin_z = min(neg_stages["z_real"].min(), pos_stages["z_real"].min())
        vmax_z = max(neg_stages["z_real"].max(), pos_stages["z_real"].max())

        im0 = axes3[0, 0].imshow(
            neg_stages["pre_activation"].T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin_pre,
            vmax=vmax_pre,
        )
        axes3[0, 0].set_title("negative: pre-activation")
        axes3[0, 0].set_ylabel("unit")
        im1 = axes3[0, 1].imshow(
            pos_stages["pre_activation"].T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin_pre,
            vmax=vmax_pre,
        )
        axes3[0, 1].set_title("positive: pre-activation")
        fig3.colorbar(im1, ax=axes3[0, :].tolist(), shrink=0.85, label=r"$u(t)$")

        im2 = axes3[1, 0].imshow(
            neg_stages["z_real"].T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin_z,
            vmax=vmax_z,
        )
        axes3[1, 0].set_title("negative: $\Re(z)$ before readout")
        axes3[1, 0].set_ylabel("unit")
        axes3[1, 0].set_xlabel("time step")
        im3 = axes3[1, 1].imshow(
            pos_stages["z_real"].T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin_z,
            vmax=vmax_z,
        )
        axes3[1, 1].set_title("positive: $\Re(z)$ before readout")
        axes3[1, 1].set_xlabel("time step")
        fig3.colorbar(im3, ax=axes3[1, :].tolist(), shrink=0.85, label=r"$\Re(z)$")
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, _epoch_filename("sentiment_comparison_heatmap", epoch)), transparent=True)
        plt.close(fig3)

        fig4, ax4 = plt.subplots(1, 1, figsize=(12, 4))
        im4 = ax4.imshow(
            z_diff.T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
        )
        ax4.set_title("node-state difference before readout (positive $-$ negative)")
        ax4.set_xlabel("time step")
        ax4.set_ylabel("unit")
        fig4.colorbar(im4, ax=ax4, shrink=0.9, label=r"$\Delta \Re(z)$")
        plt.tight_layout()
        fig4.savefig(os.path.join(output_dir, _epoch_filename("sentiment_z_diff_heatmap", epoch)), transparent=True)
        plt.close(fig4)

    fig5, axes5 = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    panels = [
        (neg_stages["raw_input"], pos_stages["raw_input"], "embedding norm", "input"),
        (neg_stages["pre_activation"][:, unit_indices[0]], pos_stages["pre_activation"][:, unit_indices[0]], r"$u(t)$", f"pre-activation (unit {unit_indices[0]})"),
        (neg_stages["z_real"][:, unit_indices[0]], pos_stages["z_real"][:, unit_indices[0]], r"$\Re(z)$", f"node state (unit {unit_indices[0]})"),
    ]
    for row, (neg_trace, pos_trace, ylabel, title) in enumerate(panels):
        axes5[row, 0].plot(t, neg_trace, color=thesis_red, linewidth=1.8)
        axes5[row, 0].set_ylabel(ylabel)
        axes5[row, 0].set_title(f"negative: {title}")
        set_epoch_xlim(axes5[row, 0], num_timesteps)
        set_value_ylim(axes5[row, 0], neg_trace)
        axes5[row, 1].plot(t, pos_trace, color=ifisc_green, linewidth=1.8)
        axes5[row, 1].set_title(f"positive: {title}")
        set_epoch_xlim(axes5[row, 1], num_timesteps)
        set_value_ylim(axes5[row, 1], pos_trace)
    axes5[-1, 0].set_xlabel("time step")
    axes5[-1, 1].set_xlabel("time step")
    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, _epoch_filename("sentiment_comparison_side_by_side", epoch)), transparent=True)
    plt.close(fig5)

    summary = _sentiment_comparison_summary(neg_stages, pos_stages)
    with open(os.path.join(output_dir, _epoch_filename("sentiment_comparison_summary", epoch, suffix="json")), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def plot_imdb_sentiment_comparison(
    model,
    neg_token_ids,
    pos_token_ids,
    output_dir,
    epoch,
    num_units_plot=5,
    class_labels=None,
):
    model.eval()
    neg_inputs = prepare_imdb_sequence(model, neg_token_ids)
    pos_inputs = prepare_imdb_sequence(model, pos_token_ids)
    neg_stages = trace_model_signal_stages(model, neg_inputs, example_idx=0, input_mode="norm")
    pos_stages = trace_model_signal_stages(model, pos_inputs, example_idx=0, input_mode="norm")
    return plot_sentiment_comparison(
        neg_stages,
        pos_stages,
        output_dir,
        epoch,
        num_units_plot=num_units_plot,
        class_labels=class_labels,
    )


def plot_signal_stages_for_example(
    model,
    inputs,
    output_dir,
    epoch,
    example_idx=0,
    input_mode="scalar",
    task_type="classification",
    class_labels=None,
    true_label=None,
    target_value=None,
    num_units_plot=5,
    raw_input_label="input",
    title_suffix=None,
):
    model.eval()
    stages = trace_model_signal_stages(model, inputs, example_idx=example_idx, input_mode=input_mode)
    plot_signal_stages(
        stages,
        output_dir,
        epoch,
        task_type=task_type,
        class_labels=class_labels,
        true_label=true_label,
        target_value=target_value,
        num_units_plot=num_units_plot,
        raw_input_label=raw_input_label,
        title_suffix=title_suffix,
    )
