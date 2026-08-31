# restormer/blocks/mdta.py

import torch
import torch.nn as nn
from einops import rearrange


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (Restormer's attention block).

    Instead of computing attention over spatial positions (H*W x H*W),
    MDTA computes attention over channels, making it much more efficient
    for high-resolution image restoration.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.num_heads = num_heads

        # Learnable temperature parameter
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )

        # Q, K, V projection
        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=1,
            bias=bias
        )

        # Depthwise convolution
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        # Output projection
        self.project_out = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):

        b, c, h, w = x.shape

        # Generate Q, K, V
        qkv = self.qkv_dwconv(
            self.qkv(x)
        )

        q, k, v = qkv.chunk(3, dim=1)

        # (B,C,H,W) -> (B,heads,C_per_head,H*W)
        q = rearrange(
            q,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads
        )

        k = rearrange(
            k,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads
        )

        v = rearrange(
            v,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads
        )

        # L2 normalization
        q = torch.nn.functional.normalize(
            q,
            dim=-1
        )

        k = torch.nn.functional.normalize(
            k,
            dim=-1
        )

        # Channel attention
        attn = (
            q @ k.transpose(-2, -1)
        ) * self.temperature

        attn = attn.softmax(dim=-1)

        # Weighted aggregation
        out = attn @ v

        # Restore image layout
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=h,
            w=w
        )

        return self.project_out(out)