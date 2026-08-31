"""

This test verifies that the MDTA (Multi-DConv Head Transposed Attention)
block can process an input tensor and return an output with the expected shape.

"""
import torch
from restormer.blocks.mdta import MDTA

x = torch.randn(2, 48, 64, 64)

mdta = MDTA(
    dim=48,
    num_heads=4
)

y = mdta(x)

print("Input :", x.shape)
print("Output:", y.shape)