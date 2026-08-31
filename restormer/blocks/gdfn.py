# restormer/blocks/gdfn.py

import torch
import torch.nn as nn


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (Restormer's feedforward block).

        1. 1x1 conv: expand C -> 2 * hidden_dim (two parallel paths at once)
        2. 3x3 depthwise conv: local spatial mixing on both paths
           (the "Dconv" part of the name)
        3. split into two halves along channels: x1, x2
        4. gating: GELU(x1) * x2
           - x1 acts as a learned "gate" deciding how much of x2 passes
           - x2 carries the actual content being gated
        5. 1x1 conv: project hidden_dim -> C (back to original size)

    Args:
        dim               : number of input/output channels C
        ffn_expansion     : expansion factor for the hidden dimension
                             (paper default: 2.66)
        bias              : whether conv layers use a bias term
    """

    def __init__(self, dim, ffn_expansion=2.66, bias=False):
        super().__init__()

        hidden_dim = int(dim * ffn_expansion)


        self.project_in = nn.Conv2d(
            dim,
            hidden_dim * 2,
            kernel_size=1,
            bias=bias
        )

        # Depthwise 3x3 conv applied to both paths together
        self.dwconv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim * 2,  # depthwise: independent per channel
            bias=bias,
        )


        self.project_out = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)      # (B, 2*hidden_dim, H, W)
        x = self.dwconv(x)          # local spatial mixing

        x1, x2 = x.chunk(2, dim=1)  # each (B, hidden_dim, H, W)

        # Gating: x1 passed through GELU acts as a soft on/off gate
        # controlling how much of x2's content survives.
        x = torch.nn.functional.gelu(x1) * x2

        return self.project_out(x)  # (B, C, H, W)