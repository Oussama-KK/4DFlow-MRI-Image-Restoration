import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error between prediction and target.

    Args:
        pred, target : tensors of the same shape, e.g. (B, C, H, W)

    Returns:
        Scalar tensor.
    """
    return torch.mean(torch.abs(pred - target))