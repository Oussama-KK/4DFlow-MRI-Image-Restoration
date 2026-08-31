import torch
from restormer.blocks.layer_norm import LayerNorm

x = torch.randn(2, 48, 64, 64)

ln = LayerNorm(48, norm_type="WithBias")

y = ln(x)

print("Input :", x.shape)
print("Output:", y.shape)