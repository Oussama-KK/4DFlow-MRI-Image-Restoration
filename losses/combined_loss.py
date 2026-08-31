import torch
import torch.nn as nn

from losses.pixel_loss import PixelLoss
from losses.ssim_loss import SSIMLoss


class CombinedLoss(nn.Module):


    def __init__(
        self,
        pixel_loss_type: str = "l1",
        pixel_weight: float = 0.6,
        ssim_weight: float = 0.4,
        data_range: float = 1.0,
    ):
        super().__init__()

        self.pixel_loss = PixelLoss(loss_type=pixel_loss_type)
        self.ssim_loss = SSIMLoss(data_range=data_range)

        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor| None = None,):
        """
        Returns:
            total_loss : scalar tensor, used for backward()
            loss_dict  : dict of individual (unweighted) loss values,
                         useful for logging each component separately
                         via utils/logger.py without affecting training
        """
        pixel = self.pixel_loss(pred, target, mask)
        ssim_l = self.ssim_loss(pred, target)

        total = self.pixel_weight * pixel + self.ssim_weight * ssim_l

        loss_dict = {
            "pixel_loss": pixel.item(),
            "ssim_loss": ssim_l.item(),
            "total_loss": total.item(),
        }

        return total, loss_dict