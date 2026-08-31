import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from restormer.model import Restormer
from datasets.preprocessing import preprocess
from utils.checkpoint import load_checkpoint

from metrics.psnr import psnr
from metrics.ssim import ssim
from metrics.mse import mse
from metrics.mae import mae


MODEL_CFG = {
    "in_channels": 1,
    "out_channels": 1,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "heads": [1, 2, 4, 8],
}

DEFAULT_CHECKPOINT = "outputs/checkpoints/best.pth"
DEFAULT_OUTPUT = "outputs/predictions/inference_prediction.png"
CROP_SIZE = 256


def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device):
    model = Restormer(
        in_channels=MODEL_CFG["in_channels"],
        out_channels=MODEL_CFG["out_channels"],
        dim=MODEL_CFG["dim"],
        num_blocks=MODEL_CFG["num_blocks"],
        heads=MODEL_CFG["heads"],
    )

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        map_location=device,
    )

    model.to(device)
    model.eval()

    return model


def prepare_input(image_path: str):
    """Load and preprocess one 4D Flow image."""

    image = preprocess(
        image_path,
        split="test",
        crop_size=CROP_SIZE,
    )

    if image is None:
        raise ValueError(
            f"Image is too small for {CROP_SIZE}x{CROP_SIZE} crop:\n{image_path}"
        )

    # Convert (C, H, W) -> (1, C, H, W)
    return torch.from_numpy(image).float().unsqueeze(0)


def run_inference(model, input_tensor, device):
    """Run Restormer inference on one image."""

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction


def save_prediction(input_tensor, prediction, save_path):
    """Save input and prediction visualization."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_np = input_tensor[0].detach().cpu().numpy()
    prediction_np = prediction[0].detach().cpu().numpy()

    # Remove channel dimension
    input_np = np.squeeze(input_np)
    prediction_np = np.squeeze(prediction_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(input_np, cmap="gray")
    axes[0].set_title("4D Flow Input")
    axes[0].axis("off")

    axes[1].imshow(prediction_np, cmap="gray")
    axes[1].set_title("Restormer Prediction")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"\nPrediction saved to:\n{save_path}")


def calculate_metrics(prediction, target):
    """Calculate PSNR, SSIM, MSE and MAE."""

    with torch.no_grad():
        psnr_value = psnr(prediction, target).item()
        ssim_value = ssim(prediction, target).item()
        mse_value = mse(prediction, target).item()
        mae_value = mae(prediction, target).item()

    print("\n" + "=" * 50)
    print("INFERENCE METRICS")
    print("=" * 50)
    print(f"PSNR : {psnr_value:.4f} dB")
    print(f"SSIM : {ssim_value:.4f}")
    print(f"MSE  : {mse_value:.6f}")
    print(f"MAE  : {mae_value:.6f}")
    print("=" * 50)

    return {
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "MSE": mse_value,
        "MAE": mae_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Restormer inference on 4D Flow MRI"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to 4D Flow .nii or .nii.gz image",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained Restormer checkpoint",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Optional Cine ground-truth image. "
            "If provided, PSNR/SSIM/MSE/MAE are calculated."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save prediction visualization",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Verify required files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found:\n{args.input}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found:\n{args.checkpoint}")

    if args.target is not None and not os.path.exists(args.target):
        raise FileNotFoundError(f"Target image not found:\n{args.target}")

    print("\nLoading Restormer...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully.")

    print("\nPreprocessing input:")
    print(args.input)

    input_tensor = prepare_input(args.input)

    print(f"Input tensor shape: {input_tensor.shape}")

    print("\nRunning inference...")
    prediction = run_inference(model, input_tensor, device)

    print(f"Prediction shape: {prediction.shape}")

    save_prediction(
        input_tensor,
        prediction,
        args.output,
    )

    if args.target is not None:
        print("\nLoading target:")
        print(args.target)

        target_tensor = prepare_input(args.target)
        target_tensor = target_tensor.to(device)

        calculate_metrics(
            prediction,
            target_tensor,
        )

    print("\nInference completed successfully.")


if __name__ == "__main__": main()