import torch

from restormer.blocks.transformer_block import TransformerBlock

x = torch.randn(2, 48, 64, 64)

block = TransformerBlock(
    dim=48,
    num_heads=4,
    ffn_expansion=2.66,
    bias=False,
    norm_type="WithBias"
)

y = block(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)

assert y.shape == x.shape

print("✅ TransformerBlock works correctly!")