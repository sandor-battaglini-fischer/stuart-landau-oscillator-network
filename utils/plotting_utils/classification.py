import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .helpers import as_float, set_epoch_xlim, set_value_ylim
from .style import ifisc_green, thesis_blue, thesis_red


def plot_confusion_matrix(y_true, y_pred, output_dir, num_classes=10, class_labels=None, epoch=None, normalize=True, title=None):
    if class_labels is None:
        if num_classes == 2:
            class_labels = ["Negative", "Positive"]
        else:
            class_labels = [str(i) for i in range(num_classes)]

    if not HAS_SKLEARN:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1
    else:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    if normalize:
        cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        cm_to_plot = cm_normalized
        fmt = ".2f"
        vmin, vmax = 0, 1
    else:
        cm_to_plot = cm
        fmt = "d"
        vmin, vmax = None, None

    figsize = (8, 6) if num_classes == 2 else (10, 8)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_to_plot, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title=(title or "Confusion Matrix")
        + (" (Normalized)" if normalize else "")
        + (f" - Epoch {epoch}" if epoch is not None else ""),
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm_to_plot.max() / 2.0
    fontsize = 12 if num_classes == 2 else 9
    for i in range(num_classes):
        for j in range(num_classes):
            text_color = "white" if cm_to_plot[i, j] > thresh else "black"
            if normalize:
                ax.text(
                    j,
                    i,
                    f"{cm_to_plot[i, j]:.2f}\n({cm[i, j]})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fontsize,
                )
            else:
                ax.text(
                    j,
                    i,
                    format(cm_to_plot[i, j], fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fontsize,
                )

    fig.tight_layout()

    if epoch is not None:
        filename = f"{output_dir}/confusion_matrix_epoch{epoch:02d}.png"
    else:
        filename = f"{output_dir}/confusion_matrix.png"

    fig.savefig(filename)
    plt.close(fig)

    return cm


def plot_classification_metrics(train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, output_dir):
    train_accs = [as_float(v) for v in train_accs]
    val_accs = [as_float(v) for v in val_accs]
    test_accs = [as_float(v) for v in test_accs]
    train_losses = [as_float(v) for v in train_losses]
    val_losses = [as_float(v) for v in val_losses]
    test_losses = [as_float(v) for v in test_losses]

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    epochs_so_far = np.arange(len(test_accs))
    ax_acc.plot(epochs_so_far, train_accs, label="train accuracy", color=ifisc_green, linestyle=":")
    ax_acc.plot(epochs_so_far, val_accs, label="val accuracy", color=thesis_blue, linestyle="--")
    ax_acc.plot(epochs_so_far, test_accs, label="test accuracy", color=thesis_red, linewidth=2)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title("Accuracy Over Time")
    if len(test_accs) > 0:
        set_value_ylim(ax_acc, train_accs + val_accs + test_accs, default=(0.0, 100.0), clamp_min=0.0, clamp_max=100.0)
        set_epoch_xlim(ax_acc, len(test_accs))
    else:
        ax_acc.set_ylim(0, 100)
        ax_acc.set_xlim(0, 0)
    ax_acc.legend(loc="best")
    fig_acc.tight_layout()
    fig_acc.savefig(f"{output_dir}/accuracy_over_time.png")
    plt.close(fig_acc)

    fig_loss, axes_loss = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax1_loss, ax2_loss = axes_loss

    epochs_so_far = np.arange(len(test_losses))
    ax1_loss.plot(epochs_so_far, train_losses, label="train loss", color=ifisc_green, linestyle=":")
    ax1_loss.plot(epochs_so_far, val_losses, label="val loss", color=thesis_blue, linestyle="--")
    ax1_loss.plot(epochs_so_far, test_losses, label="test loss", color=thesis_red, linewidth=2)
    ax1_loss.set_ylabel("Cross-Entropy Loss")
    ax1_loss.legend(loc="best")
    if len(test_losses) > 0:
        set_value_ylim(ax1_loss, train_losses + val_losses + test_losses, default=(0.0, 1.0), clamp_min=0.0)
        set_epoch_xlim(ax1_loss, len(test_losses))

    ax2_loss.plot(epochs_so_far, val_accs, label="val accuracy", color="tab:purple", linestyle="--")
    ax2_loss.plot(epochs_so_far, test_accs, label="test accuracy", color="tab:orange", linewidth=2)
    ax2_loss.set_ylabel("Accuracy (%)")
    ax2_loss.set_xlabel("epoch")
    if len(test_accs) > 0 and len(val_accs) > 0:
        set_value_ylim(ax2_loss, val_accs + test_accs, default=(0.0, 100.0), clamp_min=0.0, clamp_max=100.0)
    else:
        ax2_loss.set_ylim(0, 100)
    ax2_loss.legend(loc="best")
    if len(epochs_so_far) > 0:
        set_epoch_xlim(ax2_loss, len(epochs_so_far))

    fig_loss.tight_layout()
    fig_loss.savefig(f"{output_dir}/loss_and_accuracy_over_time.png")
    plt.close(fig_loss)
