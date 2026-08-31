import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a single image tensor (C, H, W) into a displayable NumPy
    array (H, W, C) with values clamped to [0, 1].

    Detaches and moves the tensor to CPU first, so it's safe to call
    directly on model outputs that still require grad.
    """
    image = tensor.detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

    if image.shape[-1] == 1:
        image = image.squeeze(-1)  # (H, W, 1) -> (H, W) for grayscale

    return image


def normalize(image: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    """
    Min-max normalizes an image to [0, data_range].
    Useful for raw MRI slices that aren't already in [0, 1].
    """
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-8:
        return np.zeros_like(image)

    return (image - img_min) / (img_max - img_min) * data_range


def denormalize(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Reverses a mean/std normalization applied during preprocessing,
    e.g. to bring a model output back to its original intensity range
    before saving or computing metrics against raw ground truth.
    """
    return tensor * std + mean


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts a float image in [0, 1] to uint8 in [0, 255],
    ready for saving with PIL/OpenCV.
    """
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)