import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error between prediction and target.

    Args:
        pred, target : tensors of the same shape, e.g. (B, C, H, W)

    Returns:
        Scalar tensor.
    """
    return torch.mean((pred - target) ** 2)