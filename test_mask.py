import sys

import numpy as np
import matplotlib.pyplot as plt

from datasets.preprocessing import (
    load_nii,
    normalize,
    random_crop,
    create_signal_mask,
    is_large_enough,
    CROP_SIZE,
)

# Parameters

SIGMA = 2.2
THRESHOLD = 0.08


# Input Argument

if len(sys.argv) < 2:
    print("Usage:")
    print(r'python test_mask.py "C:\data\image.nii.gz"')
    sys.exit(1)

path = sys.argv[1]

# Load Image

original = load_nii(path)

print("File:", path)
print("Original shape:", original.shape)


# Size Check

if not is_large_enough(original, CROP_SIZE):
    raise ValueError(
        f"Image is too small for {CROP_SIZE}x{CROP_SIZE} crop. "
        f"Image shape: {original.shape}"
    )


# Normalize

normalized = normalize(original)

# Center or random crop

cropped = random_crop(
    normalized,
    crop_size=CROP_SIZE,
)

print("Cropped shape:", cropped.shape)


# Create Signal Mask

smoothed, mask = create_signal_mask(
    cropped,
    sigma=SIGMA,
    threshold=THRESHOLD,
)

masked_image = cropped * mask




print("\nMask Parameters")
print("-----------------------------")
print(f"Gaussian sigma : {SIGMA}")
print(f"Threshold      : {THRESHOLD}")




# Visualization

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original
axes[0, 0].imshow(original, cmap="gray")
axes[0, 0].set_title(
    f"Original\nShape: {original.shape}"
)
axes[0, 0].axis("off")

# Normalized + Crop
axes[0, 1].imshow(
    cropped,
    cmap="gray",
    vmin=0,
    vmax=1,
)
axes[0, 1].set_title(
    f"Normalized + Crop\nShape: {cropped.shape}"
)
axes[0, 1].axis("off")

# Smoothed
axes[0, 2].imshow(
    smoothed,
    cmap="gray",
    vmin=0,
    vmax=1,
)
axes[0, 2].set_title(
    f"Gaussian Smoothed\nσ = {SIGMA}"
)
axes[0, 2].axis("off")

# Binary Mask
axes[1, 0].imshow(
    mask,
    cmap="gray",
    vmin=0,
    vmax=1,
)
axes[1, 0].set_title(
    f"Binary Mask\nT = {THRESHOLD}"
)
axes[1, 0].axis("off")

# Masked Image
axes[1, 1].imshow(
    masked_image,
    cmap="gray",
    vmin=0,
    vmax=1,
)
axes[1, 1].set_title(
    "Mask * Cropped"
)
axes[1, 1].axis("off")

# Statistics Panel
axes[1, 2].text(
    0.05,
    0.9,
    (
        f"Sigma               : {SIGMA}\n\n"
        f"Threshold           : {THRESHOLD}\n\n"
    ),
    fontsize=11,
    verticalalignment="top",
)

axes[1, 2].set_title("Mask Statistics")
axes[1, 2].axis("off")

plt.suptitle(
    "Signal Mask Generation Pipeline",
    fontsize=14,
)

plt.tight_layout()


# Show Figure Directly

plt.figure()