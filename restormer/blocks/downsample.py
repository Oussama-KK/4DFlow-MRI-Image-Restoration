import torch.nn as nn


class Downsample(nn.Module):
    """
    Downsampling block used between encoder stages.

    Halves spatial resolution and doubles channel count, using
    PixelUnshuffle instead of strided convolution or max-pooling.

    Flow:
        (B, C, H, W)
          -> 3x3 conv, C -> C/2      (B, C/2, H, W)
          -> PixelUnshuffle(2)       (B, C/2 * 4, H/2, W/2)
                                     = (B, 2C, H/2, W/2)

    Net effect: channels double, spatial size halves.

    Args:
        num_channels : input channel count C
    """

    def __init__(self, num_channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)