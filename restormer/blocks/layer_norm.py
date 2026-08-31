# restormer/blocks/layer_norm.py

import torch
import torch.nn as nn
from einops import rearrange


def to_3d(x):
    """
    (B, C, H, W) -> (B, H*W, C)

    Restormer normalizes across the channel dimension.
    """
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    """
    (B, H*W, C) -> (B, C, H, W)
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    """
    LayerNorm without bias and without mean subtraction.
    """

    def __init__(self, num_channels):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.num_channels = num_channels

    def forward(self, x):
        """
        x: (B, N, C)
        """
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)

        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """
    Standard LayerNorm with learnable weight and bias.
    """

    def __init__(self, num_channels):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.num_channels = num_channels

    def forward(self, x):
        """
        x: (B, N, C)
        """
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)

        return (
            (x - mu)
            / torch.sqrt(sigma + 1e-5)
            * self.weight
            + self.bias
        )


class LayerNorm(nn.Module):
    """
    Channel-wise LayerNorm for feature maps of shape:
    (B, C, H, W)

    Args:
        num_channels: number of channels
        norm_type: "BiasFree" or "WithBias"
    """

    def __init__(self, num_channels, norm_type="WithBias"):
        super().__init__()

        assert norm_type in ("BiasFree", "WithBias"), (
            f"norm_type must be 'BiasFree' or 'WithBias', "
            f"got {norm_type}"
        )

        if norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(num_channels)
        else:
            self.body = WithBiasLayerNorm(num_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        x = to_3d(x)
        x = self.body(x)
        x = to_4d(x, h, w)

        return x