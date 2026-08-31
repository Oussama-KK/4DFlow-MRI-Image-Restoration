import torch
import torch.nn.functional as F


def _gaussian_kernel1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_window(window_size: int, channels: int, sigma: float) -> torch.Tensor:
    g1d = _gaussian_kernel1d(window_size, sigma)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    window = g2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) between prediction and target,
    computed with a Gaussian sliding window (standard formulation).

    Higher is better (1.0 = identical images).

    Args:
        pred, target : (B, C, H, W) tensors
        data_range   : max intensity value (1.0 if images are normalized to [0, 1])
        window_size  : size of the Gaussian window (paper-standard: 11)
        sigma        : std of the Gaussian window (paper-standard: 1.5)

    Returns:
        Scalar tensor: mean SSIM over the batch.
    """
    channels = pred.size(1)
    window = _create_window(window_size, channels, sigma).to(pred.device).type_as(pred)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8

    ssim_map = numerator / denominator
    return ssim_map.mean()