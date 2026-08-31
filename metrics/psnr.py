import torch

from metrics.mse import mse


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio between prediction and target.

        PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        pred, target : tensors of the same shape, e.g. (B, C, H, W)
        data_range   : max intensity value (1.0 if images are normalized to [0, 1])

    Returns:
        Scalar tensor, in dB. Returns a large finite value instead of inf
        when pred == target exactly (perfect reconstruction).
    """
    error = mse(pred, target)

    if error == 0:
        return torch.tensor(100.0, device=pred.device, dtype=pred.dtype)

    return 10.0 * torch.log10((data_range ** 2) / error)