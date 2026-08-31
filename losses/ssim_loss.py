# losses/ssim_loss.py

import torch
import torch.nn as nn

from metrics.ssim import ssim


class SSIMLoss(nn.Module):
    """
    SSIM-based training loss: 1 - SSIM(pred, target).

    Reuses metrics/ssim.py's ssim() function directly rather than
    reimplementing the windowed-SSIM math a second time 
    
    Args:
        data_range  : max intensity value (1.0, matching [0, 1] normalization)
        window_size : Gaussian window size 
        sigma       : Gaussian window std 
    """

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_val = ssim(
            pred, target,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
        )
        return 1.0 - ssim_val