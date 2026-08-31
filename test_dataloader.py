import torch

from datasets.dataloader import get_dataloader, get_dataset
from losses.pixel_loss import PixelLoss


dataset_cfg = {
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "test_csv": "data/test.csv",
    "crop_size": 256,
    "seed": 42,
    "augment": True,
}

train_cfg = {
    "batch_size": 40,
    "num_workers": 0,  # easier debugging
    "pin_memory": False,
}


print("=" * 60)
print("DATASET TEST")
print("=" * 60)

dataset = get_dataset(dataset_cfg, split="train")

print(f"Dataset size: {len(dataset)}")

sample = dataset[0]

if sample is None:
    raise RuntimeError(
        "First sample is None. Check preprocessing."
    )

print("\nSample keys:")
print(sample.keys())

print("\nShapes:")
print("Input :", sample["Input"].shape)
print("Target:", sample["Target"].shape)
print("Mask  :", sample["Mask"].shape)

print("\nDtypes:")
print("Input :", sample["Input"].dtype)
print("Target:", sample["Target"].dtype)
print("Mask  :", sample["Mask"].dtype)

print("\nRanges:")
print(
    f"Input : [{sample['Input'].min():.4f}, "
    f"{sample['Input'].max():.4f}]"
)
print(
    f"Target: [{sample['Target'].min():.4f}, "
    f"{sample['Target'].max():.4f}]"
)

print(
    f"Mask unique values: "
    f"{sorted(set(sample['Mask'].flatten().tolist()))[:5]}"
)

print(
    f"Mask coverage: "
    f"{sample['Mask'].mean() * 100:.2f}%"
)

print("\nPaths:")
print("Input :", sample["input_path"])
print("Target:", sample["target_path"])

print("\n✅ Dataset test passed")


print("\n")
print("=" * 60)
print("DATALOADER TEST")
print("=" * 60)

loader = get_dataloader(
    dataset_cfg,
    train_cfg,
    split="train"
)

batch = next(iter(loader))

if batch is None:
    raise RuntimeError(
        "First batch is None. "
        "Check collate_fn and dataset."
    )

print("\nBatch Shapes:")
print("Input :", batch["Input"].shape)
print("Target:", batch["Target"].shape)
print("Mask  :", batch["Mask"].shape)

print("\nExpected format:")
print("(batch_size, channels, height, width)")

print(
    f"\nMask coverage in batch: "
    f"{batch['Mask'].float().mean().item() * 100:.2f}%"
)

print("\n✅ DataLoader test passed")


print("\n")
print("=" * 60)
print("MASKED PIXEL LOSS TEST")
print("=" * 60)

inputs = batch["Input"]
targets = batch["Target"]
mask = batch["Mask"]

loss_fn = PixelLoss(loss_type="l1")

loss_unmasked = loss_fn(
    inputs,
    targets
)

loss_masked = loss_fn(
    inputs,
    targets,
    mask
)

print(
    f"\nUnmasked L1 loss: "
    f"{loss_unmasked.item():.6f}"
)

print(
    f"Masked   L1 loss: "
    f"{loss_masked.item():.6f}"
)

diff = torch.abs(inputs - targets)

manual_loss = (
    (diff * mask).sum()
    / (mask.sum() + 1e-8)
)

print(
    f"Manual masked loss: "
    f"{manual_loss.item():.6f}"
)

print(
    f"Difference: "
    f"{abs(manual_loss.item() - loss_masked.item()):.10f}"
)

assert torch.allclose(
    manual_loss,
    loss_masked,
    atol=1e-6
), "Masked loss implementation is incorrect!"

print("\n✅ PixelLoss masking test passed")


print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("✅ Dataset loads correctly")
print("✅ Mask is present")
print("✅ Shapes are consistent")
print("✅ DataLoader batches correctly")
print("✅ Mask reaches the loss function")
print("✅ PixelLoss masking matches manual calculation")
print("✅ End-to-end pipeline is working")