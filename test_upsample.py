import torch
from restormer.blocks.upsample import Upsample

x = torch.randn(2, 96, 64, 64)

up = Upsample(96)

y = up(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)