import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Constants
CROP_SIZE = 256
LOW_PERC = 1
HIGH_PERC = 99
DTYPE = np.float32


def load_nii(path: str | Path) -> np.ndarray:
    """
    Load a NIfTI file and return a 2D numpy array.
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    # (H, W, 1) -> (H, W)
    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D image after squeezing, "
            f"got shape {data.shape} for file {path}"
        )

    return data


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Percentile-based normalization to [0, 1].
    """
    low = np.percentile(image, LOW_PERC)
    high = np.percentile(image, HIGH_PERC)

    image = np.clip(image, low, high)

    denom = high - low
    if denom == 0:
        return np.zeros_like(image, dtype=DTYPE)

    image = (image - low) / denom
    return image.astype(DTYPE)


def is_large_enough(
    image: np.ndarray,
    crop_size: int = CROP_SIZE
) -> bool:
    """
    Check whether the image is large enough
    for cropping.
    """
    h, w = image.shape
    return h >= crop_size and w >= crop_size


def center_crop(
    image: np.ndarray,
    crop_size: int = CROP_SIZE
) -> np.ndarray:
    """
    Extract a center crop.
    """
    h, w = image.shape

    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2

    return image[
        start_h:start_h + crop_size,
        start_w:start_w + crop_size
    ]


def random_crop(
    image: np.ndarray,
    crop_size: int = CROP_SIZE,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Extract a random crop.
    """
    h, w = image.shape

    if rng is None:
        rng = np.random.default_rng()

    max_h = h - crop_size
    max_w = w - crop_size

    start_h = rng.integers(0, max_h + 1)
    start_w = rng.integers(0, max_w + 1)

    return image[
        start_h:start_h + crop_size,
        start_w:start_w + crop_size
    ]


def add_channel_dim(image: np.ndarray) -> np.ndarray:
    """
    Convert (H, W) -> (1, H, W)
    """
    return image[np.newaxis, :, :]


def preprocess_with_mask(
    path: str | Path,
    split: str = "train",
    crop_size: int = CROP_SIZE,
    rng: np.random.Generator | None = None,
    sigma: float = 2.0,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Preprocess an image and create a corresponding signal mask.

    Returns:
        image: (1, H, W)
        mask:  (1, H, W)
    """

    # 1. Load
    image = load_nii(path)

    # 2. Check size
    if not is_large_enough(image, crop_size):
        return None

    # 3. Normalize
    image = normalize(image)

    # 4. Crop
    if split == "train":
        image = random_crop(
            image,
            crop_size=crop_size,
            rng=rng
        )
    else:
        image = center_crop(
            image,
            crop_size=crop_size
        )

    # 5. Create mask on the SAME crop
    _, mask = create_signal_mask(
        image,
        sigma=sigma,
        threshold=threshold
    )

    # 6. Add channel dimension
    image = add_channel_dim(image)
    mask = add_channel_dim(mask)

    return (
        image.astype(DTYPE),
        mask.astype(DTYPE)
    )

def preprocess(
    path: str | Path,
    split: str = "train",
    crop_size: int = CROP_SIZE,
    rng: np.random.Generator | None = None
) -> np.ndarray | None:
    """
    Full preprocessing pipeline.
    """

    # 1. Load image
    image = load_nii(path)

    # 2. Size check
    if not is_large_enough(image, crop_size):
        return None

    # 3. Normalize
    image = normalize(image)

    # 4. Crop
    if split == "train":
        image = random_crop(
            image,
            crop_size=crop_size,
            rng=rng
        )
    else:
        image = center_crop(
            image,
            crop_size=crop_size
        )

    # 5. Add channel dimension
    image = add_channel_dim(image)

    return image.astype(DTYPE)



def create_signal_mask(
    image: np.ndarray,
    sigma: float = 2.2,
    threshold: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a binary signal mask from a normalized MRI image.

    """

    # Gaussian smoothing
    smoothed = gaussian_filter(image, sigma=sigma)

    # Binary threshold mask
    mask = (smoothed > threshold).astype(np.float32)

    return smoothed.astype(np.float32), mask




if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage:")
        print(r'python preprocessing.py "C:\data\image.nii.gz"')
        sys.exit(1)

    path = sys.argv[1]

    original = load_nii(path)
    normalized = normalize(original)
    cropped = center_crop(normalized)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title(f"Original\n{original.shape}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(normalized, cmap="gray")
    plt.title("After Normalization")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cropped, cmap="gray")
    plt.title(f"After Crop\n{cropped.shape}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("File:", path)
    print("Original shape:", original.shape)
    print("Normalized range:", normalized.min(), normalized.max())
    print("Cropped shape:", cropped.shape)