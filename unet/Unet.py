import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions.

    Conv -> ReLU -> Conv -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Encoder block:

    MaxPool 2x2 -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder block:

    Upsample -> concatenate skip connection -> DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels // 2 + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Handle possible spatial-size differences.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net baseline for 256x256 single-channel image-to-image translation.

    Input:
        [B, 1, H, W]

    Output:
        [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(
            in_channels,
            64,
        )

        self.down1 = DownBlock(
            64,
            128,
        )

        self.down2 = DownBlock(
            128,
            256,
        )

        self.down3 = DownBlock(
            256,
            512,
        )

        self.down4 = DownBlock(
            512,
            1024,
        )

        # Decoder
        self.up1 = UpBlock(
            in_channels=1024,
            skip_channels=512,
            out_channels=512,
        )

        self.up2 = UpBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.up3 = UpBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.up4 = UpBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        # Final prediction layer
        self.out_conv = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.down4(x4)

        # Decoder + skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        x = self.out_conv(x)

        return x