import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from utils.memory_capacity import dynamics_kind, oscillator_core
from .helpers import set_value_ylim
from .style import ifisc_green, mycmap, thesis_blue, thesis_red


DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
IMDB_COLORS = {0: thesis_red, 1: ifisc_green}
IMDB_NAMES = {0: "Negative", 1: "Positive"}
MG_COLORS = {0: thesis_blue, 1: (0.45, 0.45, 0.45), 2: thesis_red}
MG_NAMES = {0: "low", 1: "mid", 2: "high"}
SCATTER_ALPHA = 0.45


def class_colors(task, labels):
    uniq = sorted(np.unique(labels).tolist())
    if task == "imdb":
        return {c: IMDB_COLORS.get(int(c), thesis_blue) for c in uniq}
    if task == "mackey_glass":
        return {c: MG_COLORS.get(int(c), thesis_blue) for c in uniq}
    return {c: DIGIT_COLORS[int(c) % 10] for c in uniq}


def class_names(task, labels):
    uniq = sorted(np.unique(labels).tolist())
    if task == "imdb":
        return {c: IMDB_NAMES.get(int(c), str(c)) for c in uniq}
    if task == "mackey_glass":
        return {c: MG_NAMES.get(int(c), str(c)) for c in uniq}
    return {c: str(int(c)) for c in uniq}


def readout_feature_names(num_nodes, feature_dim):
    real_names = [rf"$\mathrm{{Re}}\,z_{{{i}}}$" for i in range(num_nodes)]
    imag_names = [rf"$\mathrm{{Im}}\,z_{{{i}}}$" for i in range(num_nodes)]
    if feature_dim == num_nodes:
        return [rf"$x_{{{i}}}$" for i in range(num_nodes)]
    return real_names + imag_names


@torch.no_grad()
def trace_readout_states(model, inputs, batch_size=32):
    core = oscillator_core(model)
    core.eval()
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype
    kind = dynamics_kind(core)
    num_steps, n_examples, _ = inputs.shape
    num_nodes = core.num_nodes

    z_real_parts = []
    z_imag_parts = []
    feature_parts = []
    logit_parts = []

    for start in range(0, n_examples, batch_size):
        chunk = inputs[:, start : start + batch_size].to(device=device, dtype=dtype)
        bsz = chunk.size(1)
        a = torch.zeros(bsz, num_nodes, device=device, dtype=dtype)
        b = torch.zeros(bsz, num_nodes, device=device, dtype=dtype)
        rec_a = torch.empty(num_steps, bsz, num_nodes, device="cpu")
        rec_b = torch.empty_like(rec_a)
        for t in range(num_steps):
            a, b = core.dynamics_step(a, b, chunk[t])
            rec_a[t] = a.detach().cpu()
            rec_b[t] = b.detach().cpu()
        if kind == "dho":
            features = a
        else:
            features = torch.cat([a, b], dim=1)
        logits = core.h2o(features)
        z_real_parts.append(rec_a.numpy())
        z_imag_parts.append(rec_b.numpy())
        feature_parts.append(features.detach().cpu().numpy())
        logit_parts.append(logits.detach().cpu().numpy())

    z_real = np.concatenate(z_real_parts, axis=1)
    z_imag = np.concatenate(z_imag_parts, axis=1)
    return {
        "z_real": z_real,
        "z_imag": z_imag,
        "z_magnitude": np.sqrt(np.square(z_real) + np.square(z_imag)),
        "readout_features": np.concatenate(feature_parts, axis=0),
        "logits": np.concatenate(logit_parts, axis=0),
    }


def _grid_shape(n_nodes):
    if n_nodes <= 4:
        cols = n_nodes
    elif n_nodes <= 9:
        cols = 3
    elif n_nodes <= 16:
        cols = 4
    elif n_nodes <= 25:
        cols = 5
    else:
        cols = 10
    rows = int(np.ceil(n_nodes / cols))
    return rows, cols


def _mean_std(arr, axis):
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=axis), np.nanstd(arr, axis=axis)


def plot_per_node_class_timeseries(z, labels, names, colors, ylabel, title, path, max_faint=12, t_offset=0):
    num_steps, _, num_nodes = z.shape
    t = np.arange(num_steps) + int(t_offset)
    rows, cols = _grid_shape(num_nodes)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.4 * rows), sharex=True, squeeze=False)
    classes = sorted(np.unique(labels).tolist())

    for i in range(num_nodes):
        ax = axes[i // cols][i % cols]
        for c in classes:
            mask = labels == c
            series = z[:, mask, i]
            n_c = series.shape[1]
            if n_c == 0:
                continue
            color = colors[c]
            if n_c <= max_faint:
                for k in range(n_c):
                    ax.plot(t, series[:, k], color=color, lw=0.6, alpha=0.18)
            mean, std = _mean_std(series, axis=1)
            ax.plot(t, mean, color=color, lw=1.8, label=names[c])
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.set_title(f"node {i}")
        if num_steps <= 1:
            ax.set_xlim(float(t[0]) - 0.5, float(t[0]) + 0.5)
        else:
            ax.set_xlim(float(t[0]), float(t[-1]))
        set_value_ylim(ax, z[:, :, i].reshape(-1))
        if i % cols == 0:
            ax.set_ylabel(ylabel)
        if i // cols == rows - 1:
            ax.set_xlabel("time step")

    for k in range(num_nodes, rows * cols):
        axes[k // cols][k % cols].axis("off")

    handles, labels_legend = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, loc="upper center", ncol=min(len(handles), 5), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06 if rows > 1 else 1.12)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_final_state_heatmaps(z_real, z_imag, z_mag, labels, names, path):
    classes = sorted(np.unique(labels).tolist())
    num_nodes = z_real.shape[-1]
    mats = []
    titles = [r"final $\mathrm{Re}\,z$", r"final $\mathrm{Im}\,z$", r"final $|z|$"]
    for arr in (z_real, z_imag, z_mag):
        last = arr[-1]
        mat = np.stack([np.nanmean(last[labels == c], axis=0) for c in classes], axis=0)
        mats.append(mat)

    fig, axes = plt.subplots(1, 3, figsize=(4.4 * 3, max(3.6, 0.28 * len(classes) + 2.4)))
    for ax, mat, title in zip(axes, mats, titles):
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        cmap = "coolwarm" if title != r"final $|z|$" else "viridis"
        kw = {"vmin": -vmax, "vmax": vmax} if cmap == "coolwarm" else {}
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, **kw)
        ax.set_title(title)
        ax.set_xlabel("node")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([names[c] for c in classes])
        ax.set_xticks(range(num_nodes) if num_nodes <= 16 else np.linspace(0, num_nodes - 1, 8, dtype=int))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("class")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return {
        str(names[c]): {
            "mean_final_re": [float(x) for x in mats[0][i]],
            "mean_final_im": [float(x) for x in mats[1][i]],
            "mean_final_mag": [float(x) for x in mats[2][i]],
        }
        for i, c in enumerate(classes)
    }


def plot_final_amplitude_vs_node(z_real, z_imag, z_mag, labels, names, colors, path):
    labels = np.asarray(labels)
    last = [
        (np.asarray(z_real[-1]), r"$\mathrm{Re}\,z_i$"),
        (np.asarray(z_imag[-1]), r"$\mathrm{Im}\,z_i$"),
        (np.asarray(z_mag[-1]), r"$|z_i|$"),
    ]
    num_nodes = last[0][0].shape[-1]
    nodes = np.arange(num_nodes)
    classes = sorted(np.unique(labels).tolist())
    offsets = np.linspace(-0.2, 0.2, num=len(classes)) if len(classes) > 1 else np.array([0.0])
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(4.4 * 3, 4.8), sharex=True)
    for ax, (vals, ylabel) in zip(axes, last):
        for c, off in zip(classes, offsets):
            mask = labels == c
            color = colors.get(c, thesis_blue)
            jitter = rng.uniform(-0.08, 0.08, size=vals[mask].shape)
            ax.scatter(
                nodes[None, :] + off + jitter, vals[mask],
                s=7, alpha=SCATTER_ALPHA, color=color, edgecolors="none", rasterized=True, zorder=2,
            )
            mean = np.nanmean(vals[mask], axis=0)
            ax.hlines(mean, nodes + off - 0.16, nodes + off + 0.16, colors="white", lw=4.0, zorder=4)
            ax.hlines(mean, nodes + off - 0.16, nodes + off + 0.16, colors=[color], lw=2.2, zorder=5)
        ax.axhline(0.0, color="0.35", lw=0.8, zorder=1)
        ax.set_xlabel("node index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(nodes if num_nodes <= 16 else np.linspace(0, num_nodes - 1, 8, dtype=int))
        ax.set_xlim(-0.7, num_nodes - 0.3)
    fig.legend(
        handles=[
            Line2D([0], [0], color=colors.get(c, thesis_blue), lw=2.4, label=str(names.get(c, c)))
            for c in classes
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_readout_vector_by_class(features, labels, names, colors, feature_names, path):
    classes = sorted(np.unique(labels).tolist())
    d = features.shape[1]
    x = np.arange(d)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.18 * d + 4.0), 4.8))
    for c in classes:
        vals = features[labels == c]
        mean, std = _mean_std(vals, axis=0)
        ax.plot(x, mean, color=colors[c], lw=1.8, label=names[c])
        ax.fill_between(x, mean - std, mean + std, color=colors[c], alpha=0.18, linewidth=0)
    if d == 2 * (d // 2) and d > 2:
        ax.axvline(d // 2 - 0.5, color="0.5", ls="--", lw=0.9)
    ax.set_xlabel("readout coordinate")
    ax.set_ylabel("value")
    ax.set_title(r"vector that $h2o$ receives (last $z$)")
    if d <= 24:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=60, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_readout_weights(weight, bias, feature_names, class_label_list, path):
    fig, ax = plt.subplots(figsize=(max(8.0, 0.22 * weight.shape[1] + 3.0), max(3.6, 0.45 * weight.shape[0] + 2.2)))
    vmax = np.nanmax(np.abs(weight)) if np.isfinite(weight).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(weight, aspect="auto", origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(class_label_list)))
    ax.set_yticklabels(class_label_list)
    if weight.shape[1] <= 24:
        ax.set_xticks(range(weight.shape[1]))
        ax.set_xticklabels(feature_names, rotation=60, ha="right")
    else:
        ax.set_xlabel("readout coordinate")
    ax.set_title(r"$W_{h2o}$ (rows = logits)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return [float(x) for x in np.asarray(bias).reshape(-1)]


def _finite_rows(*arrays):
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        flat = np.asarray(arr).reshape(arr.shape[0], -1)
        mask &= np.isfinite(flat).all(axis=1)
    return mask


def _fit_tsne(features, n_components, seed, perplexity=30):
    from sklearn.manifold import TSNE

    n = features.shape[0]
    if n < 5:
        return None
    perp = float(min(perplexity, max(5, (n - 1) // 3)))
    try:
        reducer = TSNE(
            n_components=n_components,
            perplexity=perp,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(features)
    except Exception:
        return None


def _fit_pca(features, n_components):
    from sklearn.decomposition import PCA

    n = min(n_components, features.shape[0], features.shape[1])
    if n < 1:
        return None, None
    pca = PCA(n_components=n, random_state=0)
    return pca.fit_transform(features), pca.explained_variance_ratio_


def _mg_abs_error(logits, targets):
    pred = np.asarray(logits, dtype=float)
    if pred.ndim == 2:
        pred = pred[:, 0]
    else:
        pred = pred.reshape(-1)
    targ = np.asarray(targets, dtype=float).reshape(-1)
    return np.abs(pred - targ)


def _marker_sizes_from_error(errors, s_min=8.0, s_max=90.0):
    err = np.abs(np.asarray(errors, dtype=float).reshape(-1))
    err = np.where(np.isfinite(err), err, 0.0)
    lo, hi = float(np.min(err)), float(np.max(err))
    if hi <= lo:
        return np.full(err.shape, 0.5 * (s_min + s_max))
    t = (err - lo) / (hi - lo)
    return s_min + (s_max - s_min) * t


def _error_size_legend_handles(errors, sizes):
    from matplotlib.lines import Line2D

    err = np.abs(np.asarray(errors, dtype=float).reshape(-1))
    sz = np.asarray(sizes, dtype=float).reshape(-1)
    finite = np.isfinite(err) & np.isfinite(sz)
    err, sz = err[finite], sz[finite]
    if err.size == 0:
        return [], []
    queries = [float(np.min(err)), float(np.median(err)), float(np.max(err))]
    handles, labels, seen = [], [], set()
    for q in queries:
        key = round(q, 8)
        if key in seen:
            continue
        seen.add(key)
        i = int(np.argmin(np.abs(err - q)))
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=float(np.sqrt(sz[i])),
                markerfacecolor="0.35",
                markeredgecolor="none",
            )
        )
        labels.append(f"{q:.3g}")
    return handles, labels


def _maybe_size_legend(ax, sizes, size_values, loc="lower right"):
    if sizes is None or size_values is None:
        return
    handles, labels = _error_size_legend_handles(size_values, sizes)
    if handles:
        ax.legend(handles, labels, title=r"$|\hat{y}-y|$", loc=loc, frameon=False, fontsize=12, title_fontsize=13)


def _scatter_sizes(n, sizes, order=None):
    if sizes is None:
        s = np.full(n, 14.0)
    else:
        s = np.asarray(sizes, dtype=float).reshape(-1)
    if order is None:
        return s
    return s[order]


def plot_embedding_2d_gradient(
    embedding, values, path, title, xlabel, ylabel, cbar_label="target", sizes=None, size_values=None
):
    values = np.asarray(values, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    order = np.argsort(values)
    sc = ax.scatter(
        embedding[order, 0],
        embedding[order, 1],
        c=values[order],
        s=_scatter_sizes(values.shape[0], sizes, order),
        alpha=SCATTER_ALPHA,
        cmap=mycmap,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _maybe_size_legend(ax, sizes, size_values)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_tsne_1d_gradient(
    embedding, values, path, title, cbar_label="target", sizes=None, size_values=None
):
    xs = embedding.reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    order = np.argsort(values)
    s = _scatter_sizes(values.shape[0], sizes, order)
    sc0 = axes[0].scatter(
        xs[order],
        np.zeros_like(xs)[order],
        c=values[order],
        s=s,
        alpha=SCATTER_ALPHA,
        cmap=mycmap,
        edgecolors="none",
    )
    axes[0].set_yticks([])
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_title("1-D embedding")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, label=cbar_label)

    sc1 = axes[1].scatter(
        xs[order],
        values[order],
        c=values[order],
        s=s,
        alpha=SCATTER_ALPHA,
        cmap=mycmap,
        edgecolors="none",
    )
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel(cbar_label)
    axes[1].set_title("embedding vs target")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04, label=cbar_label)
    _maybe_size_legend(axes[1], sizes, size_values, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_embedding_3d_gradient(
    embedding, values, path, title, cbar_label="target", sizes=None, size_values=None
):
    values = np.asarray(values, dtype=float).reshape(-1)
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    order = np.argsort(values)
    sc = ax.scatter(
        embedding[order, 0],
        embedding[order, 1],
        embedding[order, 2],
        c=values[order],
        s=_scatter_sizes(values.shape[0], sizes, order),
        alpha=SCATTER_ALPHA,
        cmap=mycmap,
        edgecolors="none",
        depthshade=False,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label(cbar_label)
    ax.set_xlabel("t-SNE 1", labelpad=8)
    ax.set_ylabel("t-SNE 2", labelpad=8)
    ax.set_zlabel("t-SNE 3", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=18, azim=42)
    _maybe_size_legend(ax, sizes, size_values, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_tsne_3d(embedding, labels, names, colors, path, title, markers=None):
    fig = plt.figure(figsize=(9.4, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    classes = sorted(np.unique(labels).tolist())
    for c in classes:
        mask = labels == c
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=14,
            alpha=SCATTER_ALPHA,
            color=colors[c],
            label=names[c],
            edgecolors="none",
            depthshade=False,
        )
    if markers is not None:
        wrong = ~markers
        if wrong.any():
            ax.scatter(
                embedding[wrong, 0],
                embedding[wrong, 1],
                embedding[wrong, 2],
                s=21,
                facecolors="none",
                edgecolors="k",
                linewidths=0.7,
                label="miscl.",
                depthshade=False,
            )
    ax.set_xlabel("t-SNE 1", labelpad=8)
    ax.set_ylabel("t-SNE 2", labelpad=8)
    ax.set_zlabel("t-SNE 3", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=18, azim=42)
    _legend_outside_right(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_mg_delay_embedding(
    windows,
    delay,
    path,
    targets=None,
    sizes=None,
    size_values=None,
    seed=1,
    max_trajectories=150,
):
    windows = np.asarray(windows, dtype=float)
    if windows.ndim == 3:
        windows = windows[..., 0]
    delay = int(delay)
    t_len, n = windows.shape
    if delay < 1 or delay >= t_len:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    delayed = windows[-1 - delay]
    current = windows[-1]
    if targets is not None:
        values = np.asarray(targets, dtype=float).reshape(-1)
        order = np.argsort(values)
        sc = ax.scatter(
            delayed[order],
            current[order],
            c=values[order],
            s=_scatter_sizes(values.shape[0], sizes, order),
            alpha=SCATTER_ALPHA,
            cmap=mycmap,
            edgecolors="none",
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("target")
        _maybe_size_legend(ax, sizes, size_values, loc="upper left")
    else:
        ax.scatter(delayed, current, s=11, alpha=SCATTER_ALPHA, color=thesis_red, edgecolors="none", zorder=3)

    n_lines = min(int(n), int(max_trajectories))
    rng = np.random.default_rng(seed)
    line_idx = np.arange(n) if n_lines == n else rng.choice(n, size=n_lines, replace=False)
    for i in line_idx:
        ax.plot(
            windows[:-delay, i],
            windows[delay:, i],
            color="k",
            lw=0.45,
            alpha=0.35,
            zorder=4,
        )

    ax.set_xlabel(rf"$x(t-{delay})$")
    ax.set_ylabel(r"$x(t)$")
    ax.set_title(rf"Mackey–Glass delay plot ($h={delay}$)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return delay


def _legend_outside_right(ax, fontsize=12):
    return ax.legend(
        markerscale=1.2,
        fontsize=fontsize,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        borderaxespad=0.0,
        frameon=False,
        handletextpad=0.45,
        labelspacing=0.35,
        scatterpoints=1,
    )


def plot_tsne_2d(embedding, labels, names, colors, path, title, markers=None):
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    classes = sorted(np.unique(labels).tolist())
    for c in classes:
        mask = labels == c
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=SCATTER_ALPHA,
            color=colors[c],
            label=names[c],
            edgecolors="none",
        )
    if markers is not None:
        wrong = ~markers
        if wrong.any():
            ax.scatter(
                embedding[wrong, 0],
                embedding[wrong, 1],
                s=21,
                facecolors="none",
                edgecolors="k",
                linewidths=0.7,
                label="miscl.",
            )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    _legend_outside_right(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_tsne_1d(embedding, labels, names, colors, path, title):
    values = embedding.reshape(-1)
    classes = sorted(np.unique(labels).tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    rng = np.random.default_rng(0)
    for i, c in enumerate(classes):
        mask = labels == c
        vals = values[mask]
        jitter = rng.normal(0.0, 0.08, size=vals.shape)
        axes[0].scatter(vals, np.full_like(vals, i, dtype=float) + jitter, s=11, alpha=SCATTER_ALPHA, color=colors[c])
        axes[0].plot([np.nanmean(vals)], [i], marker="|", markersize=18, markeredgewidth=2.4, color="k")
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels([names[c] for c in classes])
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_title("1-D embedding (strip)")

    bins = np.linspace(np.nanmin(values), np.nanmax(values), 28) if np.isfinite(values).any() else 20
    for c in classes:
        axes[1].hist(values[labels == c], bins=bins, color=colors[c], alpha=0.45, label=names[c], density=True)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("density")
    axes[1].set_title("1-D embedding (density)")
    fig.suptitle(title)
    _legend_outside_right(axes[1], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_pca_2d(embedding, var_ratio, labels, names, colors, path, weight_dir=None):
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    classes = sorted(np.unique(labels).tolist())
    for c in classes:
        mask = labels == c
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=SCATTER_ALPHA,
            color=colors[c],
            label=names[c],
            edgecolors="none",
        )
    if weight_dir is not None and embedding.shape[1] >= 2:
        span = max(np.ptp(embedding[:, 0]), np.ptp(embedding[:, 1]), 1e-6)
        vec = np.asarray(weight_dir, dtype=float)
        if vec.shape[0] >= 2 and np.linalg.norm(vec[:2]) > 0:
            vec = vec[:2]
            vec = vec / (np.linalg.norm(vec) + 1e-12) * 0.35 * span
            ax.annotate(
                "",
                xy=(vec[0], vec[1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4),
            )
            ax.text(vec[0], vec[1], r"$W$ dir.", fontsize=12, ha="left", va="bottom")
    var0 = 100.0 * float(var_ratio[0]) if var_ratio is not None and len(var_ratio) > 0 else 0.0
    var1 = 100.0 * float(var_ratio[1]) if var_ratio is not None and len(var_ratio) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({var0:.1f}\\%)")
    ax.set_ylabel(f"PC2 ({var1:.1f}\\%)")
    ax.set_title(r"PCA of last $z$ (linear view of $h2o$ input)")
    _legend_outside_right(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def plot_linear_decision(logits, labels, names, colors, path, targets=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    classes = sorted(np.unique(labels).tolist())
    logits = np.asarray(logits)

    if logits.ndim == 1 or logits.shape[1] == 1:
        score = logits.reshape(-1)
        title_l = r"linear readout $y=W h+b$"
        if targets is not None:
            axes[1].scatter(targets, score, s=11, alpha=SCATTER_ALPHA, color=thesis_blue, edgecolors="none")
            lo = min(np.nanmin(targets), np.nanmin(score))
            hi = max(np.nanmax(targets), np.nanmax(score))
            axes[1].plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1.0)
            axes[1].set_xlabel("target")
            axes[1].set_ylabel("prediction")
            axes[1].set_title("readout vs target")
        else:
            axes[1].axis("off")
    elif logits.shape[1] == 2:
        score = logits[:, 1] - logits[:, 0]
        title_l = r"decision score $y_+-y_-$"
        preds = logits.argmax(axis=1)
        acc = float(np.mean(preds == labels))
        axes[1].axvline(0.0, color="0.5", ls="--", lw=1.0)
        bins = np.linspace(np.nanmin(score), np.nanmax(score), 30) if np.isfinite(score).any() else 20
        for c in classes:
            axes[1].hist(score[labels == c], bins=bins, color=colors[c], alpha=0.45, label=names[c], density=True)
        axes[1].set_xlabel(r"$y_+-y_-$")
        axes[1].set_ylabel("density")
        axes[1].set_title(f"trained linear separator (acc={acc:.3f})")
        axes[1].legend(fontsize=11)
    else:
        correct = logits[np.arange(len(labels)), labels.astype(int)]
        masked = logits.copy()
        masked[np.arange(len(labels)), labels.astype(int)] = -np.inf
        best_wrong = masked.max(axis=1)
        score = correct - best_wrong
        title_l = "logit margin (true $-$ best other)"
        preds = logits.argmax(axis=1)
        acc = float(np.mean(preds == labels))
        axes[1].axvline(0.0, color="0.5", ls="--", lw=1.0)
        bins = np.linspace(np.nanmin(score), np.nanmax(score), 30) if np.isfinite(score).any() else 20
        for c in classes:
            axes[1].hist(score[labels == c], bins=bins, color=colors[c], alpha=0.45, label=names[c], density=True)
        axes[1].set_xlabel("margin")
        axes[1].set_ylabel("density")
        axes[1].set_title(f"trained linear margin (acc={acc:.3f})")
        axes[1].legend(fontsize=10, ncol=2)

    rng = np.random.default_rng(1)
    for i, c in enumerate(classes):
        vals = score[labels == c]
        jitter = rng.normal(0.0, 0.08, size=vals.shape)
        axes[0].scatter(vals, np.full_like(vals, i, dtype=float) + jitter, s=11, alpha=SCATTER_ALPHA, color=colors[c])
        axes[0].plot([np.nanmean(vals)], [i], marker="|", markersize=18, markeredgewidth=2.4, color="k")
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels([names[c] for c in classes])
    axes[0].set_xlabel("score")
    axes[0].set_title(title_l)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return score


def _cohens_d(group_a, group_b):
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _ovr_mean_abs_d(values, labels):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels)
    ds = []
    for c in np.unique(labels):
        d = _cohens_d(values[labels == c], values[labels != c])
        if np.isfinite(d):
            ds.append(abs(d))
    return float(np.mean(ds)) if ds else float("nan")


def _logit_margin(logits, labels):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim == 1 or logits.shape[1] == 1:
        return logits.reshape(-1)
    if logits.shape[1] == 2:
        return logits[:, 1] - logits[:, 0]
    labels = labels.astype(int)
    correct = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    return correct - masked.max(axis=1)


def _leave_one_out_class_alignment(vectors, labels):
    vectors = np.asarray(vectors, dtype=np.float64)
    labels = np.asarray(labels)
    align = np.full(len(labels), np.nan)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        zc = vectors[idx]
        n_c = zc.shape[0]
        if n_c < 2:
            continue
        total = zc.sum(axis=0)
        for k, i in enumerate(idx):
            mu = (total - zc[k]) / (n_c - 1)
            den = np.linalg.norm(zc[k]) * np.linalg.norm(mu)
            align[i] = float(np.dot(zc[k], mu) / den) if den > 1e-12 else np.nan
    return align


def _finite_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    var = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    if var <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(var))


def _tex_escape(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _plain_review(text, limit=700):
    import re

    text = re.sub(r"<br\s*/?>", "\n", str(text), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _log_odds_z(docs_a, docs_b, min_count=20):
    from collections import Counter

    ca = Counter(w for doc in docs_a for w in doc)
    cb = Counter(w for doc in docs_b for w in doc)
    vocab = [
        w
        for w in set(ca) | set(cb)
        if ca[w] + cb[w] >= min_count and len(w) > 1 and w != "br"
    ]
    n_a = sum(ca[w] for w in vocab)
    n_b = sum(cb[w] for w in vocab)
    v = max(len(vocab), 1)
    alpha = 1.0
    out = {}
    for w in vocab:
        y1 = ca[w] + alpha
        y2 = cb[w] + alpha
        n1 = n_a + alpha * v
        n2 = n_b + alpha * v
        lod = np.log(y1) - np.log(n1 - y1) - np.log(y2) + np.log(n2 - y2)
        var = 1.0 / y1 + 1.0 / y2
        out[w] = float(lod / np.sqrt(var))
    return out


def _stratified_log_odds(token_lists, lobe, labels, max_len, min_count=20):
    labels = np.asarray(labels)
    lobe = np.asarray(lobe)
    zs = []
    for c in np.unique(labels):
        a = [toks[:max_len] for toks, lab, k in zip(token_lists, labels, lobe) if lab == c and k == 1]
        b = [toks[:max_len] for toks, lab, k in zip(token_lists, labels, lobe) if lab == c and k == 0]
        if a and b:
            zs.append(_log_odds_z(a, b, min_count=min_count))
    if not zs:
        a = [toks[:max_len] for toks, k in zip(token_lists, lobe) if k == 1]
        b = [toks[:max_len] for toks, k in zip(token_lists, lobe) if k == 0]
        return _log_odds_z(a, b, min_count=min_count)
    keys = set(zs[0])
    for z in zs[1:]:
        keys &= set(z)
    if not keys:
        a = [toks[:max_len] for toks, k in zip(token_lists, lobe) if k == 1]
        b = [toks[:max_len] for toks, k in zip(token_lists, lobe) if k == 0]
        return _log_odds_z(a, b, min_count=min_count)
    return {w: float(np.mean([z[w] for z in zs])) for w in keys}


def _lexical_rate(token_lists, vocab, max_len):
    rates = np.zeros(len(token_lists), dtype=float)
    vocab = set(vocab)
    for i, toks in enumerate(token_lists):
        seen = toks[:max_len]
        if not seen:
            continue
        rates[i] = sum(w in vocab for w in seen) / len(seen)
    return rates


def plot_imdb_tsne_lobes(
    embedding,
    labels,
    names,
    colors,
    token_lists,
    texts,
    n_tokens,
    ratings,
    logits,
    z_mag_last,
    max_len,
    path,
    reviews_path,
    seed=1,
):
    embedding = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels)
    n_tokens = np.asarray(n_tokens, dtype=float)
    ratings = np.asarray(ratings, dtype=float)
    z_mag = np.nanmean(np.asarray(z_mag_last, dtype=float), axis=1)
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 2 and logits.shape[1] >= 2:
        conf = np.abs(logits[:, 1] - logits[:, 0])
        pred = logits.argmax(axis=1)
        correct = pred == labels
    else:
        conf = np.full(len(labels), np.nan)
        correct = np.zeros(len(labels), dtype=bool)
    truncated = n_tokens >= float(max_len)
    lobe = truncated.astype(int)
    lobe_names = {0: "Short", 1: "Truncated"}
    lobe_colors = {0: (0.45, 0.45, 0.45), 1: thesis_blue}

    lex = {
        "First person": {"i", "me", "my", "myself", "we", "our", "ours", "us"},
        "Film": {"film", "films", "movie", "movies", "cinema"},
        "TV": {"show", "shows", "episode", "episodes", "season", "series", "tv"},
        "Plot": {"plot", "story", "character", "characters", "scene", "scenes"},
        "Craft": {"acting", "actor", "actress", "director", "script", "cinematography"},
    }
    lex_rates = {k: _lexical_rate(token_lists, v, max_len) for k, v in lex.items()}
    attr = [
        ("Rating", ratings),
        ("Confidence", conf),
        (r"$|z|$", z_mag),
        ("Accuracy", correct.astype(float)),
        ("Positive", (labels == 1).astype(float)),
    ]
    attr.extend((k, v) for k, v in lex_rates.items())
    d_vals = []
    d_labels = []
    for name, values in attr:
        d = _cohens_d(values[lobe == 1], values[lobe == 0])
        d_vals.append(d)
        d_labels.append(name)

    word_z = _stratified_log_odds(token_lists, lobe, labels, max_len)
    top_words = sorted(word_z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.6))
    sent_markers = {0: "o", 1: "s"}
    for lab in sorted(np.unique(labels).tolist()):
        for k in (0, 1):
            mask = (labels == lab) & (lobe == k)
            axes[0, 0].scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=11,
                alpha=SCATTER_ALPHA,
                color=lobe_colors[k],
                marker=sent_markers.get(int(lab), "o"),
                edgecolors="none",
                label=f"{lobe_names[k]}, {names[lab]}",
            )
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")
    axes[0, 0].set_title("t-SNE by truncation", pad=14)
    axes[0, 0].legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="0.8",
        markerscale=1.6,
        fontsize=14,
        labelspacing=0.2,
        handletextpad=0.35,
    )

    sc = axes[0, 1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=11,
        c=n_tokens,
        cmap="cividis",
        alpha=SCATTER_ALPHA,
        edgecolors="none",
    )
    cb = fig.colorbar(sc, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb.set_label("Tokens")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")
    axes[0, 1].set_title("t-SNE by length", pad=14)

    sc_r = axes[0, 2].scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=11,
        c=ratings,
        cmap="cividis",
        alpha=SCATTER_ALPHA,
        edgecolors="none",
        vmin=1,
        vmax=10,
    )
    cb_r = fig.colorbar(sc_r, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cb_r.set_label("Star rating")
    axes[0, 2].set_xlabel("t-SNE 1")
    axes[0, 2].set_ylabel("t-SNE 2")
    axes[0, 2].set_title("t-SNE by rating", pad=14)

    bins = np.linspace(np.nanmin(n_tokens), np.nanmax(n_tokens), 24)
    for k in (0, 1):
        axes[1, 0].hist(
            n_tokens[lobe == k],
            bins=bins,
            color=lobe_colors[k],
            alpha=0.55,
            label=lobe_names[k],
            density=True,
        )
    axes[1, 0].axvline(max_len, color="k", ls="--", lw=1.1, label="Max length")
    axes[1, 0].set_xlabel("Tokens")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Review length", pad=14)
    axes[1, 0].legend(loc="best", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")

    order = np.argsort(np.nan_to_num(d_vals, nan=0.0))
    y = np.arange(len(order))
    bar_colors = [thesis_blue if d_vals[i] >= 0 else (0.45, 0.45, 0.45) for i in order]
    axes[1, 1].barh(y, [d_vals[i] for i in order], color=bar_colors, height=0.7)
    axes[1, 1].axvline(0.0, color="k", lw=0.8)
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels([d_labels[i] for i in order])
    axes[1, 1].set_xlabel(r"Cohen's $d$")
    axes[1, 1].set_title("Truncated minus short", pad=14)
    axes[1, 1].plot([], [], color=thesis_blue, lw=8, label="Truncated higher")
    axes[1, 1].plot([], [], color=(0.45, 0.45, 0.45), lw=8, label="Short higher")
    axes[1, 1].legend(loc="best", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")

    if top_words:
        words, zscores = zip(*sorted(top_words, key=lambda kv: kv[1]))
        y = np.arange(len(words))
        wcolors = [thesis_blue if z >= 0 else (0.45, 0.45, 0.45) for z in zscores]
        axes[1, 2].barh(y, zscores, color=wcolors, height=0.7)
        axes[1, 2].axvline(0.0, color="k", lw=0.8)
        axes[1, 2].set_yticks(y)
        axes[1, 2].set_yticklabels([_tex_escape(w) for w in words])
    axes[1, 2].set_xlabel(r"Log-odds $z$")
    axes[1, 2].set_title("Distinctive words", pad=14)
    axes[1, 2].plot([], [], color=thesis_blue, lw=8, label="Truncated")
    axes[1, 2].plot([], [], color=(0.45, 0.45, 0.45), lw=8, label="Short")
    axes[1, 2].legend(loc="best", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")

    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    fig.savefig(path, transparent=True)
    plt.close(fig)

    centroids = {k: np.mean(embedding[lobe == k], axis=0) for k in (0, 1)}
    lines = []
    for k in (1, 0):
        mask_k = lobe == k
        lines.append(f"=== {lobe_names[k]} (n={int(np.sum(mask_k))}) ===")
        lines.append(
            "mean length={:.1f}, mean rating={:.2f}, acc={:.3f}, "
            "mean |z|={:.3f}, mean |logit|={:.3f}, pos frac={:.3f}".format(
                float(np.nanmean(n_tokens[mask_k])),
                float(np.nanmean(ratings[mask_k])),
                float(np.mean(correct[mask_k])),
                float(np.nanmean(z_mag[mask_k])),
                float(np.nanmean(conf[mask_k])),
                float(np.mean(labels[mask_k] == 1)),
            )
        )
        for lab in sorted(np.unique(labels).tolist()):
            sel = np.flatnonzero(mask_k & (labels == lab))
            if len(sel) == 0:
                continue
            dist = np.linalg.norm(embedding[sel] - centroids[k], axis=1)
            take = sel[np.argsort(dist)[:4]]
            lines.append(f"-- {names[lab]} examples nearest the {lobe_names[k].lower()} centroid --")
            for i in take:
                rating = ratings[i]
                rating_s = "NA" if not np.isfinite(rating) else str(int(rating))
                lines.append(
                    f"[{names[lab]}, rating {rating_s}, {int(n_tokens[i])} tokens, "
                    f"{'truncated' if truncated[i] else 'short'}]"
                )
                lines.append(_plain_review(texts[i]))
                lines.append("")
        lines.append("")
    with open(reviews_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")

    def _lobe_stats(k):
        mask_k = lobe == k
        return {
            "n": int(np.sum(mask_k)),
            "mean_length": float(np.nanmean(n_tokens[mask_k])),
            "mean_rating": float(np.nanmean(ratings[mask_k])),
            "accuracy": float(np.mean(correct[mask_k])),
            "mean_abs_z": float(np.nanmean(z_mag[mask_k])),
            "mean_abs_logit": float(np.nanmean(conf[mask_k])),
            "positive_frac": float(np.mean(labels[mask_k] == 1)),
        }

    return {
        "tsne1_vs_truncated_corr": _finite_corr(embedding[:, 0], truncated.astype(float)),
        "tsne2_vs_truncated_corr": _finite_corr(embedding[:, 1], truncated.astype(float)),
        "tsne2_vs_n_tokens_corr": _finite_corr(embedding[:, 1], n_tokens),
        "tsne2_vs_rating_corr": _finite_corr(embedding[:, 1], ratings),
        "tsne1_vs_rating_corr": _finite_corr(embedding[:, 0], ratings),
        "tsne2_vs_confidence_corr": _finite_corr(embedding[:, 1], conf),
        "truncated": _lobe_stats(1),
        "short": _lobe_stats(0),
        "cohens_d_truncated_minus_short": {
            lab: float(val) for lab, val in zip(d_labels, d_vals) if np.isfinite(val)
        },
        "distinctive_words": [
            {"word": w, "log_odds_z": float(z)}
            for w, z in sorted(word_z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
        ],
    }


def plot_mg_tsne_structure(embedding, windows, delay, targets, logits, path, seed=1):
    embedding = np.asarray(embedding, dtype=float)
    windows = np.asarray(windows, dtype=float)
    if windows.ndim == 3:
        windows = windows[..., 0]
    targets = np.asarray(targets, dtype=float).reshape(-1)
    pred = np.asarray(logits, dtype=float)
    if pred.ndim == 2:
        pred = pred[:, 0]
    else:
        pred = pred.reshape(-1)
    err = np.abs(pred - targets)
    t_len = windows.shape[0]
    delay = int(delay)
    x_t = windows[-1]
    x_delay = windows[-1 - delay] if 1 <= delay < t_len else np.full_like(x_t, np.nan)
    mean = np.nanmean(windows, axis=0)
    std = np.nanstd(windows, axis=0)
    slope = x_t - windows[-2] if t_len >= 2 else np.full_like(x_t, np.nan)
    covariates = [
        (r"$x(T)$", x_t),
        (rf"$x(T-{delay})$", x_delay),
        ("Mean", mean),
        ("Std", std),
        ("Slope", slope),
        ("Target", targets),
        (r"$|\hat{y}-y|$", err),
    ]
    r1 = [_finite_corr(embedding[:, 0], vals) for _, vals in covariates]
    r2 = [_finite_corr(embedding[:, 1], vals) for _, vals in covariates]
    fold_ax = 0 if abs(_finite_corr(embedding[:, 0], targets)) <= abs(_finite_corr(embedding[:, 1], targets)) else 1
    fold = embedding[:, fold_ax]
    fold_label = f"t-SNE {fold_ax + 1}"

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.6))

    def _color_scatter(ax, values, cbar_label, title):
        values = np.asarray(values, dtype=float).reshape(-1)
        order = np.argsort(values)
        sc = ax.scatter(
            embedding[order, 0],
            embedding[order, 1],
            c=values[order],
            s=11,
            cmap=mycmap,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title, pad=14)
        return sc

    _color_scatter(axes[0, 0], x_t, r"$x(T)$", rf"t-SNE by last value ($r_1={r1[0]:.2f}$, $r_2={r2[0]:.2f}$)")
    _color_scatter(
        axes[0, 1],
        x_delay,
        rf"$x(T-{delay})$",
        rf"t-SNE by delayed value ($r_1={r1[1]:.2f}$, $r_2={r2[1]:.2f}$)",
    )
    _color_scatter(axes[0, 2], slope, "Slope", rf"t-SNE by slope ($r_1={r1[4]:.2f}$, $r_2={r2[4]:.2f}$)")

    if np.isfinite(x_delay).any():
        order = np.argsort(fold)
        sc_d = axes[1, 0].scatter(
            x_delay[order],
            x_t[order],
            c=fold[order],
            s=11,
            cmap=mycmap,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
            zorder=3,
        )
        cb_d = fig.colorbar(sc_d, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cb_d.set_label(fold_label)
        axes[1, 0].set_xlabel(rf"$x(t-{delay})$")
        axes[1, 0].set_ylabel(r"$x(t)$")
        axes[1, 0].set_title("Delay plot by t-SNE fold", pad=14)
        axes[1, 0].set_aspect("equal", adjustable="box")
    else:
        axes[1, 0].axis("off")

    y = np.arange(len(covariates))
    h = 0.38
    axes[1, 1].barh(y + h / 2, r1, height=h, color="k", label="t-SNE 1")
    axes[1, 1].barh(y - h / 2, r2, height=h, color="0.55", label="t-SNE 2")
    axes[1, 1].axvline(0.0, color="k", lw=0.8)
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels([name for name, _ in covariates])
    axes[1, 1].set_xlabel(r"Pearson $r$")
    axes[1, 1].set_title("Covariates vs t-SNE", pad=14)
    axes[1, 1].legend(loc="best", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")

    lo, hi = np.percentile(targets, [40.0, 60.0])
    mid = np.flatnonzero((targets >= lo) & (targets <= hi))
    example_ids = []
    example_labels = []
    if len(mid) >= 2:
        order = mid[np.argsort(fold[mid])]
        example_ids.extend([int(order[0]), int(order[-1])])
        example_labels.extend(["Mid, fold A", "Mid, fold B"])
    example_ids.append(int(np.nanargmin(targets)))
    example_labels.append("Low target")
    example_ids.append(int(np.nanargmax(targets)))
    example_labels.append("High target")
    kept = []
    kept_labels = []
    seen = set()
    for i, lab in zip(example_ids, example_labels):
        if i in seen:
            continue
        seen.add(i)
        kept.append(i)
        kept_labels.append(lab)
    example_ids, example_labels = kept, kept_labels
    trace_colors = [thesis_blue, (0.45, 0.45, 0.45), thesis_red, ifisc_green]
    t = np.arange(t_len)
    for k, (idx, lab) in enumerate(zip(example_ids, example_labels)):
        color = trace_colors[k % len(trace_colors)]
        axes[1, 2].plot(t, windows[:, idx], color=color, lw=1.7, label=lab)
        if np.isfinite(x_delay).any() and 1 <= delay < t_len:
            axes[1, 0].plot(
                windows[:-delay, idx],
                windows[delay:, idx],
                color=color,
                lw=1.3,
                alpha=0.9,
                zorder=5,
            )
            axes[1, 0].scatter(
                [x_delay[idx]],
                [x_t[idx]],
                s=42,
                color=color,
                edgecolors="k",
                linewidths=0.7,
                zorder=6,
            )
    if 1 <= delay < t_len:
        axes[1, 2].axvline(t_len - 1 - delay, color="k", ls="--", lw=1.0, label=rf"$t-{delay}$")
    axes[1, 2].set_xlabel("Time step")
    axes[1, 2].set_ylabel(r"$x(t)$")
    axes[1, 2].set_title("Example windows", pad=14)
    axes[1, 2].legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="0.8",
        labelspacing=0.2,
        handletextpad=0.35,
        fontsize=14,
    )

    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    fig.savefig(path, transparent=True)
    plt.close(fig)

    return {
        "delay": int(delay),
        "fold_axis": fold_label,
        "tsne1_corr": {
            "x(T)": r1[0],
            f"x(T-{delay})": r1[1],
            "mean": r1[2],
            "std": r1[3],
            "slope": r1[4],
            "target": r1[5],
            "abs_error": r1[6],
        },
        "tsne2_corr": {
            "x(T)": r2[0],
            f"x(T-{delay})": r2[1],
            "mean": r2[2],
            "std": r2[3],
            "slope": r2[4],
            "target": r2[5],
            "abs_error": r2[6],
        },
        "example_indices": example_ids,
        "example_labels": example_labels,
        "example_targets": [float(targets[i]) for i in example_ids],
        "example_x_t": [float(x_t[i]) for i in example_ids],
    }


def plot_phase_vs_magnitude_decision(z_real_last, z_imag_last, z_mag_last, logits, labels, names, colors, path):
    labels = np.asarray(labels)
    z_re = np.asarray(z_real_last, dtype=np.float64)
    z_im = np.asarray(z_imag_last, dtype=np.float64)
    z_mag = np.asarray(z_mag_last, dtype=np.float64)
    margin = _logit_margin(logits, labels)
    mean_mag = np.nanmean(z_mag, axis=1)
    mean_re = np.nanmean(z_re, axis=1)
    mean_im = np.nanmean(z_im, axis=1)
    vectors = np.concatenate([z_re, z_im], axis=1)
    align = _leave_one_out_class_alignment(vectors, labels)
    classes = sorted(np.unique(labels).tolist())
    num_nodes = z_mag.shape[1]
    mag_d = np.array([_ovr_mean_abs_d(z_mag[:, i], labels) for i in range(num_nodes)])
    re_d = np.array([_ovr_mean_abs_d(z_re[:, i], labels) for i in range(num_nodes)])
    im_d = np.array([_ovr_mean_abs_d(z_im[:, i], labels) for i in range(num_nodes)])
    phase_d = 0.5 * (re_d + im_d)
    r_mag = _finite_corr(mean_mag, margin)
    r_re = _finite_corr(mean_re, margin)
    r_im = _finite_corr(mean_im, margin)
    r_align = _finite_corr(align, margin)

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.6))
    scatter_specs = [
        (axes[0, 0], mean_mag, r"Mean final $|z|$", rf"Logit vs amplitude ($r={r_mag:.2f}$)"),
        (axes[0, 1], mean_re, r"Mean final $\mathrm{Re}\,z$", rf"Logit vs $\mathrm{{Re}}\,z$ ($r={r_re:.2f}$)"),
        (axes[0, 2], mean_im, r"Mean final $\mathrm{Im}\,z$", rf"Logit vs $\mathrm{{Im}}\,z$ ($r={r_im:.2f}$)"),
        (axes[1, 0], align, r"$\cos\angle(z(T), \mu_{\mathrm{class}})$", rf"Logit vs phase alignment ($r={r_align:.2f}$)"),
    ]
    n_legend = len(classes)
    for ax, xs, xlabel, title in scatter_specs:
        for c in classes:
            mask = labels == c
            ax.scatter(xs[mask], margin[mask], s=9, alpha=SCATTER_ALPHA, color=colors[c], edgecolors="none", label=names[c])
            ax.scatter(
                [np.nanmean(xs[mask])],
                [np.nanmean(margin[mask])],
                s=64,
                color=colors[c],
                edgecolors="k",
                linewidths=1.1,
                zorder=5,
                marker="o",
            )
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Logit margin")
        ax.set_title(title, pad=14)
        ax.legend(
            loc="best",
            ncol=2 if n_legend > 4 else 1,
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            edgecolor="0.8",
            borderpad=0.25,
            labelspacing=0.2,
            handletextpad=0.35,
            markerscale=1.6,
        )

    x = np.arange(num_nodes)
    w = 0.25
    axes[1, 1].bar(x - w, mag_d, w, color="k", label=r"$|z|$")
    axes[1, 1].bar(x, re_d, w, color="0.45", label=r"$\mathrm{Re}\,z$")
    axes[1, 1].bar(x + w, im_d, w, color="0.75", label=r"$\mathrm{Im}\,z$")
    axes[1, 1].set_xlabel("Node")
    axes[1, 1].set_ylabel(r"Mean one-vs-rest $|d|$")
    axes[1, 1].set_title("Per-node class discriminability", pad=14)
    axes[1, 1].set_xticks(x if num_nodes <= 16 else np.linspace(0, num_nodes - 1, 8, dtype=int))
    axes[1, 1].legend(loc="best", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")

    mag_score = float(np.nanmean(mag_d))
    re_score = float(np.nanmean(re_d))
    im_score = float(np.nanmean(im_d))
    phase_score = float(np.nanmean(phase_d))
    axes[1, 2].bar([0], [mag_score], color="k", width=0.6, label=r"$|z|$")
    axes[1, 2].bar([1], [re_score], color="0.45", width=0.6, label=r"$\mathrm{Re}\,z$")
    axes[1, 2].bar([2], [im_score], color="0.75", width=0.6, label=r"$\mathrm{Im}\,z$")
    axes[1, 2].set_xticks([0, 1, 2])
    axes[1, 2].set_xticklabels([r"$|z|$", r"$\mathrm{Re}\,z$", r"$\mathrm{Im}\,z$"])
    axes[1, 2].set_ylabel(r"Mean one-vs-rest $|d|$")
    axes[1, 2].set_title("Amplitude vs Re / Im", pad=14)
    axes[1, 2].legend(loc="best", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")
    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return {
        "margin_vs_mean_abs_z_corr": r_mag,
        "margin_vs_mean_re_z_corr": r_re,
        "margin_vs_mean_im_z_corr": r_im,
        "margin_vs_phase_alignment_corr": r_align,
        "magnitude_ovr_abs_d": mag_score,
        "real_ovr_abs_d": re_score,
        "imag_ovr_abs_d": im_score,
        "phase_ovr_abs_d": phase_score,
        "per_node_magnitude_ovr_abs_d": [float(x) for x in mag_d],
        "per_node_real_ovr_abs_d": [float(x) for x in re_d],
        "per_node_imag_ovr_abs_d": [float(x) for x in im_d],
        "per_node_phase_ovr_abs_d": [float(x) for x in phase_d],
        "dominant_encoding": "phase" if max(re_score, im_score) >= mag_score else "magnitude",
    }


def plot_readout_confusion_matrix(logits, labels, names, colors, path):
    labels = np.asarray(labels).astype(int)
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] < 2:
        return None
    preds = logits.argmax(axis=1)
    classes = sorted(np.unique(np.concatenate([labels, preds])).tolist())
    n = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[index[int(t)], index[int(p)]] += 1
    acc = float(np.mean(preds == labels))
    row_sum = cm.sum(axis=1).astype(np.float64)
    per_class = np.divide(np.diag(cm), row_sum, out=np.full(n, np.nan), where=row_sum > 0)
    tick_labels = [names.get(c, str(c)) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1.2, 0.9]})
    vmax = float(cm.max()) if cm.max() > 0 else 1.0
    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(tick_labels, rotation=40, ha="right")
    axes[0].set_yticklabels(tick_labels)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion matrix", pad=14)
    thresh = vmax / 2.0
    cell_fs = 16 if n <= 2 else 14
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > thresh else "black"
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=cell_fs)

    bar_colors = [colors.get(c, thesis_blue) for c in classes]
    axes[1].bar(range(n), per_class, color=bar_colors, width=0.7)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(tick_labels, rotation=40, ha="right")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-class accuracy", pad=14)
    axes[1].axhline(acc, color="0.35", ls="--", lw=1.2, label="Overall")
    axes[1].legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.92, edgecolor="0.8")
    for i, val in enumerate(per_class):
        if np.isfinite(val):
            axes[1].text(i, min(val + 0.03, 1.02), f"{val:.2f}", ha="center", va="bottom")
    fig.suptitle(rf"Accuracy ${acc:.3f}$")
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return {
        "accuracy": acc,
        "per_class_accuracy": {str(tick_labels[i]): None if not np.isfinite(per_class[i]) else float(per_class[i]) for i in range(n)},
        "confusion_matrix": cm.tolist(),
        "class_order": tick_labels,
    }


def _task_plot_path(output_dir, task, name):
    return os.path.join(output_dir, f"{task}_{name}")


def plot_all_readout_analysis(payload, output_dir, task, seed=1):
    os.makedirs(output_dir, exist_ok=True)
    labels = np.asarray(payload["labels"])
    names = class_names(task, labels)
    colors = class_colors(task, labels)
    z_real = payload["z_real"]
    z_imag = payload["z_imag"]
    z_mag = payload["z_magnitude"]
    features = payload["readout_features"]
    logits = payload["logits"]
    num_nodes = z_real.shape[-1]
    feat_names = readout_feature_names(num_nodes, features.shape[1])

    def p(name):
        return _task_plot_path(output_dir, task, name)

    plot_per_node_class_timeseries(
        z_real, labels, names, colors, r"$\mathrm{Re}\,z_i(t)$",
        r"per-node $\mathrm{Re}\,z$ (mean $\pm$ std)",
        p("per_node_timeseries_real.png"),
    )
    plot_per_node_class_timeseries(
        z_imag, labels, names, colors, r"$\mathrm{Im}\,z_i(t)$",
        r"per-node $\mathrm{Im}\,z$ (mean $\pm$ std)",
        p("per_node_timeseries_imag.png"),
    )
    plot_per_node_class_timeseries(
        z_mag, labels, names, colors, r"$|z_i(t)|$",
        r"per-node $|z|$ (mean $\pm$ std)",
        p("per_node_timeseries_magnitude.png"),
    )
    if task == "smnist" and z_real.shape[0] > 100:
        t0 = int(z_real.shape[0] - 100)
        plot_per_node_class_timeseries(
            z_real[-100:], labels, names, colors, r"$\mathrm{Re}\,z_i(t)$",
            r"per-node $\mathrm{Re}\,z$ (mean $\pm$ std), last 100 steps",
            p("per_node_timeseries_real_last100.png"),
            t_offset=t0,
        )
        plot_per_node_class_timeseries(
            z_imag[-100:], labels, names, colors, r"$\mathrm{Im}\,z_i(t)$",
            r"per-node $\mathrm{Im}\,z$ (mean $\pm$ std), last 100 steps",
            p("per_node_timeseries_imag_last100.png"),
            t_offset=t0,
        )
        plot_per_node_class_timeseries(
            z_mag[-100:], labels, names, colors, r"$|z_i(t)|$",
            r"per-node $|z|$ (mean $\pm$ std), last 100 steps",
            p("per_node_timeseries_magnitude_last100.png"),
            t_offset=t0,
        )

    class_means = plot_final_state_heatmaps(
        z_real, z_imag, z_mag, labels, names,
        p("final_state_mean_heatmap.png"),
    )
    amp_colors = {0: thesis_red, 1: thesis_blue} if task == "imdb" else colors
    plot_final_amplitude_vs_node(
        z_real, z_imag, z_mag, labels, names, amp_colors, p("final_amplitude_vs_node.png"),
    )
    plot_readout_vector_by_class(
        features, labels, names, colors, feat_names,
        p("readout_input_by_class.png"),
    )

    weight = payload["h2o_weight"]
    class_label_list = [names[c] for c in sorted(np.unique(labels).tolist())]
    if task != "mackey_glass":
        if weight.shape[0] == len(class_label_list):
            weight_labels = class_label_list
        else:
            weight_labels = [f"logit {i}" for i in range(weight.shape[0])]
    else:
        weight_labels = ["prediction"]
    bias_list = plot_readout_weights(
        weight, payload["h2o_bias"], feat_names, weight_labels,
        p("readout_weights.png"),
    )

    preds = None
    correct = None
    if logits.ndim == 2 and logits.shape[1] > 1:
        preds = logits.argmax(axis=1)
        correct = preds == labels

    embed_mask = _finite_rows(features)
    feat_e = features[embed_mask]
    lab_e = labels[embed_mask]
    correct_e = None if correct is None else correct[embed_mask]
    targets = payload.get("targets")
    if targets is not None:
        targets = np.asarray(targets, dtype=float).reshape(-1)
        targ_e = targets[embed_mask]
    else:
        targ_e = None
    use_gradient = task == "mackey_glass" and targ_e is not None
    sizes_e = None
    size_values_e = None
    if use_gradient:
        size_values_e = _mg_abs_error(logits[embed_mask], targ_e)
        sizes_e = _marker_sizes_from_error(size_values_e)

    tsne2 = _fit_tsne(feat_e, 2, seed)
    if tsne2 is not None:
        if use_gradient:
            plot_embedding_2d_gradient(
                tsne2,
                targ_e,
                p("readout_tsne_2d.png"),
                r"t-SNE of last $z$ (the $h2o$ input)",
                "t-SNE 1",
                "t-SNE 2",
                cbar_label="target",
                sizes=sizes_e,
                size_values=size_values_e,
            )
        else:
            plot_tsne_2d(
                tsne2, lab_e, names, colors,
                p("readout_tsne_2d.png"),
                r"t-SNE of last $z$ (the $h2o$ input)",
                markers=correct_e,
            )
    lobe_summary = None
    if task == "imdb" and tsne2 is not None and payload.get("review_tokens") is not None:
        idx_e = np.flatnonzero(embed_mask)
        tokens_e = [payload["review_tokens"][i] for i in idx_e]
        texts_e = [payload["review_texts"][i] for i in idx_e]
        n_tok_e = np.asarray(payload["review_n_tokens"])[embed_mask]
        rate_e = np.asarray(payload["review_ratings"])[embed_mask]
        mag_e = z_mag[-1][embed_mask]
        lobe_summary = plot_imdb_tsne_lobes(
            tsne2,
            lab_e,
            names,
            colors,
            tokens_e,
            texts_e,
            n_tok_e,
            rate_e,
            logits[embed_mask],
            mag_e,
            int(payload.get("review_max_len", 175)),
            p("tsne_lobes.png"),
            p("tsne_lobe_reviews.txt"),
            seed=seed,
        )
    mg_tsne_summary = None
    if task == "mackey_glass" and tsne2 is not None and payload.get("input_windows") is not None and targ_e is not None:
        windows = np.asarray(payload["input_windows"])
        if windows.ndim == 3:
            windows = windows[..., 0]
        win_e = windows[:, embed_mask]
        mg_tsne_summary = plot_mg_tsne_structure(
            tsne2,
            win_e,
            int(payload.get("mg_delay", 25)),
            targ_e,
            logits[embed_mask],
            p("tsne_structure.png"),
            seed=seed,
        )
    tsne1 = _fit_tsne(feat_e, 1, seed)
    if tsne1 is not None:
        if use_gradient:
            plot_tsne_1d_gradient(
                tsne1,
                targ_e,
                p("readout_tsne_1d.png"),
                r"t-SNE of last $z$ reduced to one coordinate",
                cbar_label="target",
                sizes=sizes_e,
                size_values=size_values_e,
            )
        else:
            plot_tsne_1d(
                tsne1, lab_e, names, colors,
                p("readout_tsne_1d.png"),
                r"t-SNE of last $z$ reduced to one coordinate",
            )
    tsne3 = _fit_tsne(feat_e, 3, seed)
    if tsne3 is not None:
        if use_gradient:
            plot_embedding_3d_gradient(
                tsne3,
                targ_e,
                p("readout_tsne_3d.png"),
                r"t-SNE of last $z$ in three coordinates",
                cbar_label="target",
                sizes=sizes_e,
                size_values=size_values_e,
            )
        else:
            plot_tsne_3d(
                tsne3, lab_e, names, colors,
                p("readout_tsne_3d.png"),
                r"t-SNE of last $z$ in three coordinates",
                markers=correct_e,
            )

    mg_delay = None
    if task == "mackey_glass" and payload.get("input_windows") is not None:
        windows = np.asarray(payload["input_windows"])
        if windows.ndim == 3:
            windows = windows[..., 0]
        delay = int(payload.get("mg_delay", 25))
        win_e = windows[:, embed_mask]
        mg_delay = plot_mg_delay_embedding(
            win_e,
            delay,
            p("delay.png"),
            targets=targ_e,
            sizes=sizes_e,
            size_values=size_values_e,
            seed=seed,
        )

    pca_emb, var_ratio = _fit_pca(feat_e, 2)
    weight_dir = None
    if pca_emb is not None and task == "imdb" and weight.shape[0] == 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, feat_e.shape[1]), random_state=0)
        pca.fit(feat_e)
        delta_w = weight[1] - weight[0]
        weight_dir = pca.components_ @ delta_w
    if pca_emb is not None:
        if use_gradient:
            var0 = 100.0 * float(var_ratio[0]) if var_ratio is not None and len(var_ratio) > 0 else 0.0
            var1 = 100.0 * float(var_ratio[1]) if var_ratio is not None and len(var_ratio) > 1 else 0.0
            plot_embedding_2d_gradient(
                pca_emb,
                targ_e,
                p("readout_pca_2d.png"),
                r"PCA of last $z$ (linear view of $h2o$ input)",
                f"PC1 ({var0:.1f}%)",
                f"PC2 ({var1:.1f}%)",
                cbar_label="target",
            )
        else:
            plot_pca_2d(
                pca_emb, var_ratio, lab_e, names, colors,
                p("readout_pca_2d.png"),
                weight_dir=weight_dir,
            )

    score = plot_linear_decision(
        logits, labels, names, colors,
        p("linear_decision.png"),
        targets=payload.get("targets"),
    )

    phase_summary = None
    confusion_summary = None
    if task in ("smnist", "imdb"):
        phase_summary = plot_phase_vs_magnitude_decision(
            z_real[-1],
            z_imag[-1],
            z_mag[-1],
            logits,
            labels,
            names,
            colors,
            p("phase_vs_magnitude_decision.png"),
        )
        confusion_summary = plot_readout_confusion_matrix(
            logits,
            labels,
            names,
            colors,
            p("confusion_matrix.png"),
        )

    summary = {
        "task": task,
        "n_examples": int(len(labels)),
        "n_nodes": int(num_nodes),
        "readout_dim": int(features.shape[1]),
        "examples_per_class": {str(names[c]): int(np.sum(labels == c)) for c in sorted(np.unique(labels).tolist())},
        "class_final_means": class_means,
        "h2o_bias": bias_list,
        "pca_explained_variance": None if var_ratio is None else [float(x) for x in var_ratio],
        "mg_delay": None if mg_delay is None else int(mg_delay),
    }
    if preds is not None:
        summary["accuracy"] = float(np.mean(correct))
        summary["per_class_accuracy"] = {
            str(names[c]): float(np.mean(preds[labels == c] == c)) for c in sorted(np.unique(labels).tolist())
        }
    if score is not None:
        summary["decision_score_mean"] = {
            str(names[c]): float(np.nanmean(score[labels == c])) for c in sorted(np.unique(labels).tolist())
        }
    if phase_summary is not None:
        summary["phase_vs_magnitude"] = phase_summary
    if confusion_summary is not None:
        summary["confusion"] = confusion_summary
    if lobe_summary is not None:
        summary["tsne_lobes"] = lobe_summary
    if mg_tsne_summary is not None:
        summary["tsne_structure"] = mg_tsne_summary
    return summary
