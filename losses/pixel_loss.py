import torch
import torch.nn as nn


class PixelLoss(nn.Module):
    """
    Pixel-wise reconstruction loss.

    Supports three variants:
        - "l1"          : mean absolute error
        - "l2"          : mean squared error
        - "charbonnier" : smooth L1 variant,
                          sqrt((pred-target)^2 + eps^2)

    Args:
        loss_type : "l1", "l2", or "charbonnier"
        eps       : smoothing constant for Charbonnier
    """

    def __init__(self, loss_type: str = "l1", eps: float = 1e-3):
        super().__init__()

        assert loss_type in ("l1", "l2", "charbonnier"), (
            f"Invalid loss_type: {loss_type}"
        )

        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        diff = pred - target

        # Compute element-wise loss
        if self.loss_type == "l1":
            loss = torch.abs(diff)

        elif self.loss_type == "l2":
            loss = diff ** 2

        else:  # charbonnier
            loss = torch.sqrt(diff ** 2 + self.eps ** 2)

        # Apply mask
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss