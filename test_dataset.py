"""
Quick manual test for datasets/dataset.py

Loads one sample through the full pipeline:
    preprocess() [load -> size check -> normalize -> crop -> channel dim]
    -> transforms.py augmentations (if --augment and split == train)

and shows/saves a side-by-side visualization so you can eyeball whether
everything lines up correctly (input, target, and their difference).

Usage (run from project root):

    python test_dataset.py --csv data/train.csv --split train --index 0
    python test_dataset.py --csv data/val.csv   --split val   --index 5 --augment
    python test_dataset.py --csv data/test.csv  --split test  --index 2 --save outputs/figures/sample.png

If --save is not given, the plot is shown in a window (plt.show()).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset import MRIRestorationDataset
from datasets.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    RandomIntensityShift,
)


def build_transform():
    return Compose([
        RandomHorizontalFlip(probability=0.5),
        RandomVerticalFlip(probability=0.5),
        RandomRotation90(),
        RandomIntensityShift(max_shift=0.05),
    ])


def describe(name, arr):
    print(f"{name:>12s} | shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                         help="path to train.csv / val.csv / test.csv")
    parser.add_argument("--split", type=str, default="train",
                         choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0,
                         help="which row of the csv to load")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--augment", action="store_true",
                         help="apply transforms.py augmentations "
                              "(only meaningful if --split train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                         help="if given, save the figure here instead of "
                              "showing it interactively")
    args = parser.parse_args()

    transform = build_transform() if (args.augment and args.split == "train") else None

    dataset = MRIRestorationDataset(
        csv_path=args.csv,
        split=args.split,
        crop_size=args.crop_size,
        transform=transform,
        seed=args.seed,
    )

    print(f"Dataset loaded: {len(dataset)} samples ({args.split} split)")
    if args.index >= len(dataset):
        raise IndexError(f"--index {args.index} out of range (0..{len(dataset)-1})")

    sample = dataset[args.index]
    input_img = sample["Input"][0]   # (1, H, W) -> (H, W) for plotting
    target_img = sample["Target"][0]

    print(f"\nSample #{args.index}")
    print(f"  input_path : {sample['input_path']}")
    print(f"  target_path: {sample['target_path']}")
    describe("input", input_img)
    describe("target", target_img)

    diff = np.abs(input_img - target_img)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(target_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Target")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="inferno")
    axes[2].set_title(f"|Input - Target|  (max={diff.max():.3f})")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{args.split} split | index {args.index} | "
                 f"augment={transform is not None}")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()