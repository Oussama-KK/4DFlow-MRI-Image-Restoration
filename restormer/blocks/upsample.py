# restormer/blocks/upsample.py

import torch.nn as nn


class Upsample(nn.Module):
    """
    Upsampling block used between decoder stages.

    Doubles spatial resolution and halves channel count, using
    PixelShuffle - the exact inverse operation of Downsample's
    PixelUnshuffle.

    Flow:
        (B, C, H, W)
          -> 3x3 conv, C -> C*2       (B, 2C, H, W)
          -> PixelShuffle(2)          (B, 2C / 4, H*2, W*2)
                                      = (B, C/2, 2H, 2W)

    Net effect: channels halve, spatial size doubles.

    Args:
        num_channels : input channel count C
    """

    def __init__(self, num_channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)