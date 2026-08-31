import argparse
import os

import torch

from restormer.model import Restormer

from datasets.dataloader import get_dataloaders

from losses.combined_loss import CombinedLoss

from metrics.psnr import psnr
from metrics.ssim import ssim
from metrics.mse import mse
from metrics.mae import mae

from utils.seed import set_seed
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visualization import save_comparison, update_loss_curve

'''

'''

MODEL_CFG = {
    "in_channels": 1,            
    "out_channels": 1,
    "dim": 48,
    "num_blocks": (2, 4, 4, 4),
    "num_refinement_blocks": 4,
    "heads": (1, 2, 4, 8),
    "ffn_expansion": 2.66,
    "bias": False,
    "norm_type": "WithBias",     # "WithBias" or "BiasFree"
}

DATASET_CFG = {
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "test_csv": "data/test.csv",
    "crop_size": 256,  # must be divisible by 8, 3 down stages
    "seed": 42,
    "augment": True,
}

TRAIN_CFG = {
    "batch_size": 2,
    "num_workers": 4,
    "pin_memory": True,
    "epochs": 100,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "log_every": 50,
    "visualize_every": 5,
    "loss": {
        "pixel_type": "l1",
        "pixel_weight": 0.8,
        "ssim_weight": 0.2,
        "data_range": 1.0,
    },
}


# Making different outputs dir that will used by the code.

OUTPUT_DIR = "outputs_restormer_2" # folder name 

os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)




def build_model(model_cfg: dict, device: torch.device) -> torch.nn.Module:
    """Builds the Restormer model from MODEL_CFG and moves it to the device."""
    model = Restormer(**model_cfg)
    return model.to(device)


def build_loss(train_cfg: dict) -> torch.nn.Module:
    """Builds the combined pixel + SSIM loss from TRAIN_CFG['loss']."""
    loss_cfg = train_cfg.get("loss", {})
    return CombinedLoss(
        pixel_loss_type=loss_cfg.get("pixel_type"),
        pixel_weight=loss_cfg.get("pixel_weight"),
        ssim_weight=loss_cfg.get("ssim_weight"),
        data_range=loss_cfg.get("data_range"),
    )


def evaluate(model, val_loader, device, loss_fn) -> dict:
    """
    Runs validation and returns averaged
    Loss / PSNR / SSIM / MSE / MAE.
    """

    model.eval()

    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "mse": 0.0, "mae": 0.0,}

    n_batches = 0

    with torch.no_grad():

        for batch in val_loader:

            if batch is None:
                continue

            inputs = batch["Input"].to(device)
            targets = batch["Target"].to(device)
            mask = batch["Mask"].to(device)

            preds = model(inputs)

            # Masked Validation loss
            val_loss, _ = loss_fn(preds, targets, mask)

            totals["loss"] += val_loss.item()
            totals["psnr"] += psnr(preds, targets).item()
            totals["ssim"] += ssim(preds, targets).item()
            totals["mse"] += mse(preds, targets).item()
            totals["mae"] += mae(preds, targets).item()

            n_batches += 1

    return {
        k: v / max(n_batches, 1)
        for k, v in totals.items()
    }


def train(resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(TRAIN_CFG.get("seed", 42))

    logger = Logger(log_dir=f"{OUTPUT_DIR}/logs")
    logger.log(f"Using device: {device}")

    loaders = get_dataloaders(DATASET_CFG, TRAIN_CFG)
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = build_model(MODEL_CFG, device)
    loss_fn = build_loss(TRAIN_CFG)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG.get("weight_decay", 1e-4),
    )


    start_epoch, best_psnr = 0, 0.0

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_metric", 0.0)
        logger.log(f"Resumed from {resume_path} at epoch {start_epoch}")

    train_losses, val_losses = [], []

    for epoch in range(start_epoch, TRAIN_CFG["epochs"]):
        model.train()
        running_loss, n_valid_steps = 0.0, 0

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            inputs = batch["Input"].to(device)
            targets = batch["Target"].to(device)
            mask = batch["Mask"].to(device)

            optimizer.zero_grad()

            # Masked Loss
            preds = model(inputs)
            total_loss, loss_dict = loss_fn(preds, targets, mask)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            n_valid_steps += 1

            if step % TRAIN_CFG.get("log_every", 1000) == 0:
                logger.log_metrics(epoch, step, loss_dict)

    

        avg_train_loss = running_loss / max(n_valid_steps, 1)
        train_losses.append(avg_train_loss)

        val_metrics = evaluate(model, val_loader, device, loss_fn)
        val_losses.append(val_metrics["loss"])

        if epoch % TRAIN_CFG.get("visualize_every", 5) == 0:
            update_loss_curve(
                train_losses,
                val_losses,
                epoch,
                save_path=f"{OUTPUT_DIR}/figures/loss_curve.png",
            )

        logger.log(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_psnr={val_metrics['psnr']:.2f} "
            f"val_ssim={val_metrics['ssim']:.4f} "
            f"val_mse={val_metrics['mse']:.4f} "
            f"val_mae={val_metrics['mae']:.4f}"
        )

        is_best = val_metrics["psnr"] > best_psnr
        best_psnr = max(best_psnr, val_metrics["psnr"])

        save_checkpoint(
            state={
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_psnr,
            },
            save_path=f"{OUTPUT_DIR}/figures/epoch_{epoch}.png",
            is_best=is_best,
        )

        if epoch % TRAIN_CFG.get("visualize_every", 5) == 0:

            sample = next(iter(val_loader))

            if sample is not None:

                sample_input = sample["Input"].to(device)
                sample_target = sample["Target"].to(device)

                with torch.no_grad():


                    sample_pred = model(sample_input)

                    sample_metrics = {
                        "PSNR": psnr(
                            sample_pred[0].unsqueeze(0),
                            sample_target[0].unsqueeze(0),
                        ).item(),

                        "SSIM": ssim(
                            sample_pred[0].unsqueeze(0),
                            sample_target[0].unsqueeze(0),
                        ).item(),

                        "MSE": mse(
                            sample_pred[0].unsqueeze(0),
                            sample_target[0].unsqueeze(0),
                        ).item(),

                        "MAE": mae(
                            sample_pred[0].unsqueeze(0),
                            sample_target[0].unsqueeze(0),
                        ).item(),
                    }

                save_comparison(
                    sample["Input"][0],
                    sample_pred[0].cpu(),
                    sample["Target"][0],
                    save_path=f"outputs/figures/epoch_{epoch}.png",
                    epoch=epoch,
                    metrics=sample_metrics,
                )



    logger.log("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Restormer model.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(resume_path=args.resume)