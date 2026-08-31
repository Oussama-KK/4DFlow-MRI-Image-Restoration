import torch
from restormer.blocks.downsample import Downsample

x = torch.randn(2, 48, 128, 128)

down = Downsample(48)

y = down(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)