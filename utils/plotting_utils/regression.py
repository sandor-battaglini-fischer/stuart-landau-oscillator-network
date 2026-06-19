import numpy as np
import matplotlib.pyplot as plt

from .style import ifisc_green, thesis_blue, thesis_red


def plot_regression_metrics(
    train_losses,
    val_losses,
    test_losses,
    test_r2_scores=None,
    test_normalized_errors=None,
    val_r2_scores=None,
    val_normalized_errors=None,
    output_dir=None,
):
    if test_r2_scores is not None:
        fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
        epochs_so_far = np.arange(len(test_r2_scores))
        ax_acc.plot(epochs_so_far, test_r2_scores, label="test $R^2$", color=thesis_red, linewidth=2)
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("test $R^2$")
        ax_acc.set_title("Test Accuracy Over Time")
        if len(test_r2_scores) > 0:
            y_min = min(test_r2_scores)
            y_max = max(test_r2_scores)
            y_range = y_max - y_min if y_max > y_min else 0.1
            y_padding = y_range * 0.1 if y_range > 0 else 0.05
            ax_acc.set_ylim(y_min - y_padding, y_max + y_padding)
            ax_acc.set_xlim(0, max(epochs_so_far) if len(epochs_so_far) > 0 else 0)
        else:
            ax_acc.set_ylim(-0.1, 1.1)
            ax_acc.set_xlim(0, 0)
        ax_acc.legend(loc="best")
        fig_acc.tight_layout()
        if output_dir:
            fig_acc.savefig(f"{output_dir}/test_accuracy_over_time.png")
        plt.close(fig_acc)

    fig_loss, axes_loss = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    ax1_loss, ax2_loss, ax3_loss = axes_loss

    epochs_so_far = np.arange(len(test_losses))

    ax1_loss.plot(epochs_so_far, train_losses, label="train loss", color=ifisc_green, linestyle=":")
    ax1_loss.plot(epochs_so_far, val_losses, label="val loss", color=thesis_blue, linestyle="--")
    ax1_loss.plot(epochs_so_far, test_losses, label="test loss", color=thesis_red)
    ax1_loss.set_ylabel("MSE")
    ax1_loss.legend(loc="best")
    if len(test_losses) > 0:
        all_losses = train_losses + val_losses + test_losses
        loss_min = min(all_losses)
        loss_max = max(all_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else loss_max * 0.1 if loss_max > 0 else 0.1
        loss_padding = loss_range * 0.1 if loss_range > 0 else 0.05
        ax1_loss.set_ylim(max(0, loss_min - loss_padding), loss_max + loss_padding)
        ax1_loss.set_xlim(0, max(epochs_so_far) if len(epochs_so_far) > 0 else 0)

    if test_normalized_errors is not None or val_normalized_errors is not None:
        if val_normalized_errors is not None:
            valid_val_normalized = [x for x in val_normalized_errors if np.isfinite(x)]
            if valid_val_normalized:
                ax2_loss.plot(
                    epochs_so_far,
                    val_normalized_errors,
                    label="val normalized error",
                    color="tab:purple",
                    linestyle="--",
                )
        if test_normalized_errors is not None:
            valid_test_normalized = [x for x in test_normalized_errors if np.isfinite(x)]
            if valid_test_normalized:
                ax2_loss.plot(
                    epochs_so_far,
                    test_normalized_errors,
                    label="test normalized error",
                    color="tab:brown",
                    linestyle="-.",
                )
                norm_min = min(valid_test_normalized)
                norm_max = max(valid_test_normalized)
                if val_normalized_errors is not None and valid_val_normalized:
                    norm_min = min(norm_min, min(valid_val_normalized))
                    norm_max = max(norm_max, max(valid_val_normalized))
                norm_range = norm_max - norm_min if norm_max > norm_min else norm_max * 0.1 if norm_max > 0 else 0.1
                norm_padding = norm_range * 0.1 if norm_range > 0 else 0.05
                ax2_loss.set_ylim(max(0, norm_min - norm_padding), norm_max + norm_padding)
            else:
                ax2_loss.set_ylim(0.0, 1.0)
        else:
            ax2_loss.set_ylim(0.0, 1.0)
    else:
        ax2_loss.set_ylim(0.0, 1.0)
    ax2_loss.set_ylabel("Normalized Error")
    ax2_loss.legend(loc="best")
    if len(epochs_so_far) > 0:
        ax2_loss.set_xlim(0, max(epochs_so_far))

    if test_r2_scores is not None:
        if val_r2_scores is not None:
            ax3_loss.plot(epochs_so_far, val_r2_scores, label="val $R^2$", color="tab:purple", linestyle="--")
        ax3_loss.plot(epochs_so_far, test_r2_scores, label="test $R^2$", color="tab:orange")
        ax3_loss.set_ylabel("$R^2$")
        ax3_loss.set_xlabel("epoch")
        if len(test_r2_scores) > 0:
            all_r2 = test_r2_scores
            if val_r2_scores is not None:
                all_r2 = val_r2_scores + test_r2_scores
            r2_min = min(all_r2)
            r2_max = max(all_r2)
            r2_range = r2_max - r2_min if r2_max > r2_min else 0.1
            r2_padding = r2_range * 0.1 if r2_range > 0 else 0.05
            ax3_loss.set_ylim(r2_min - r2_padding, r2_max + r2_padding)
        else:
            ax3_loss.set_ylim(-0.1, 1.1)
        ax3_loss.legend(loc="best")
        if len(epochs_so_far) > 0:
            ax3_loss.set_xlim(0, max(epochs_so_far))
    else:
        ax3_loss.set_xlabel("epoch")

    fig_loss.tight_layout()
    if output_dir:
        fig_loss.savefig(f"{output_dir}/loss_and_accuracy_over_time.png")
    plt.close(fig_loss)
