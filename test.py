import os
import torch

from restormer.model import Restormer
from datasets.dataloader import get_dataloaders

from utils.checkpoint import load_checkpoint
from utils.visualization import save_comparison

from metrics.psnr import psnr
from metrics.ssim import ssim
from metrics.mse import mse
from metrics.mae import mae


# Configuration

MODEL_CFG = {
    "in_channels": 1,
    "out_channels": 1,
    "dim": 48,
    "num_blocks": (2, 4, 4, 4),
    "num_refinement_blocks": 4,
    "heads": (1, 2, 4, 8),
    "ffn_expansion": 2.66,
    "bias": False,
    "norm_type": "WithBias",
}

DATASET_CFG = {
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "test_csv": "data/test.csv",
    "crop_size": 256,
    "seed": 42,
    "augment": False,
}

TRAIN_CFG = {
    "batch_size": 1,
    "num_workers": 4,
    "pin_memory": True,
}

# Checkpoint to test
CHECKPOINT_PATH = "outputs_restormer_2/checkpoints/best.pth"

# Where test figures will be saved
OUTPUT_DIR = "outputs_restormer_2/test_results"


# Build model

def build_model(model_cfg, device):

    model = Restormer(**model_cfg)

    return model.to(device)


# Main testing function

def test():


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")


    #  Build Restormer

    model = build_model(
        MODEL_CFG,
        device
    )


    # Load best.pth

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        model
    )

    model.eval()

    print("Checkpoint loaded successfully.")


    # Load test dataset

    loaders = get_dataloaders(
        DATASET_CFG,
        TRAIN_CFG
    )

    test_loader = loaders["test"]

    print(f"Number of test batches: {len(test_loader)}")


    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # 5. Test every sample

    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_mae = 0.0

    n_samples = 0


    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            if batch is None:
                continue


            inputs = batch["Input"].to(device)
            targets = batch["Target"].to(device)


            # Model prediction

            predictions = model(inputs)


            # Process every image in the batch

            for i in range(inputs.shape[0]):

                input_img = inputs[i:i+1]
                pred_img = predictions[i:i+1]
                target_img = targets[i:i+1]


                # Calculate metrics

                sample_psnr = psnr(
                    pred_img,
                    target_img
                ).item()

                sample_ssim = ssim(
                    pred_img,
                    target_img
                ).item()

                sample_mse = mse(
                    pred_img,
                    target_img
                ).item()

                sample_mae = mae(
                    pred_img,
                    target_img
                ).item()


                # Accumulate metrics

                total_psnr += sample_psnr
                total_ssim += sample_ssim
                total_mse += sample_mse
                total_mae += sample_mae

                n_samples += 1


                # Print metrics

                print(
                    f"Sample {n_samples:04d} | "
                    f"PSNR: {sample_psnr:.4f} dB | "
                    f"SSIM: {sample_ssim:.4f} | "
                    f"MSE: {sample_mse:.6f} | "
                    f"MAE: {sample_mae:.6f}"
                )


                # Save comparison figure

                sample_metrics = {
                    "PSNR": sample_psnr,
                    "SSIM": sample_ssim,
                    "MSE": sample_mse,
                    "MAE": sample_mae,
                }


                save_comparison(
                    input_img[0].cpu(),
                    pred_img[0].cpu(),
                    target_img[0].cpu(),

                    save_path=(
                        f"{OUTPUT_DIR}/"
                        f"sample_{n_samples:04d}.png"
                    ),

                    metrics=sample_metrics,
                )


    # Average test metrics

    if n_samples == 0:

        print("No valid samples were found in the test set.")
        return


    avg_psnr = total_psnr / n_samples
    avg_ssim = total_ssim / n_samples
    avg_mse = total_mse / n_samples
    avg_mae = total_mae / n_samples


    print("\n")
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Number of samples : {n_samples}")
    print(f"Average PSNR      : {avg_psnr:.4f} dB")
    print(f"Average SSIM      : {avg_ssim:.4f}")
    print(f"Average MSE       : {avg_mse:.6f}")
    print(f"Average MAE       : {avg_mae:.6f}")

    print("=" * 60)

    print(
        f"\nComparison figures saved in:\n"
        f"{OUTPUT_DIR}"
    )


# Run

if __name__ == "__main__":
    test()