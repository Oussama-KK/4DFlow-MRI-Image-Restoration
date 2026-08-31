import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.image_utils import tensor_to_numpy


def save_comparison(
    input_img,
    pred_img,
    target_img,
    save_path,
    epoch=None,
    metrics=None,
    titles=(
        "Input",
        "Prediction",
        "Ground Truth",
        "Absolute Error",
    ),
):
    """
    Save a 2x2 comparison figure with metrics and epoch number.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_np = tensor_to_numpy(input_img)
    pred_np = tensor_to_numpy(pred_img)
    target_np = tensor_to_numpy(target_img)

    error_np = np.abs(pred_np - target_np)

    images = [
        input_np,
        pred_np,
        target_np,
        error_np,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (ax, img, title) in enumerate(
        zip(axes, images, titles)
    ):
        if i < 3:
            im = ax.imshow(img, cmap="gray")
        else:
            im = ax.imshow(img, cmap="hot")

            fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                label="Absolute Error",
            )

        ax.set_title(title)
        ax.axis("off")

    # Epoch title
    if epoch is not None:
        fig.suptitle(
            f"Epoch {epoch}",
            fontsize=16,
            fontweight="bold",
        )

    # Metrics caption
    if metrics is not None:
        metrics_text = (
            f"PSNR: {metrics['PSNR']:.4f} dB   |   "
            f"SSIM: {metrics['SSIM']:.4f}   |   "
            f"MSE: {metrics['MSE']:.6f}   |   "
            f"MAE: {metrics['MAE']:.6f}"
        )

        fig.text(
            0.5,
            0.02,
            metrics_text,
            ha="center",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=0.8,
            ),
        )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

def update_loss_curve(
    train_losses: list,
    val_losses: list,
    epoch: int,
    save_path: str = "outputs/figures/loss_curve.png",
) -> None:
    """
    Update and overwrite the loss curve figure during training.

    Args:
        train_losses : training losses accumulated so far
        val_losses   : validation losses accumulated so far
        epoch        : current epoch
        save_path    : path to save/update figure
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    epochs = range(
        1,
        len(train_losses) + 1,
    )

    ax.plot(
        epochs,
        train_losses,
        color="tab:blue",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Training Loss",
    )

    ax.plot(
        epochs,
        val_losses,
        color="tab:orange",
        marker="s",
        markersize=4,
        linewidth=2,
        label="Validation Loss",
    )

    # Mark latest point
    ax.scatter(
        len(train_losses),
        train_losses[-1],
        color="tab:blue",
        s=80,
        zorder=5,
    )

    ax.scatter(
        len(val_losses),
        val_losses[-1],
        color="tab:orange",
        s=80,
        zorder=5,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Training Progress (Epoch {epoch})"
    )

    ax.legend()
    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
