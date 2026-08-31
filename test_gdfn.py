# test_gdfn.py

import torch

from restormer.blocks.gdfn import GDFN


def main():
    # Dummy input
    x = torch.randn(
        2,      # batch size
        48,     # channels
        64,     # height
        64      # width
    )

    # Create GDFN block
    gdfn = GDFN(
        dim=48,
        ffn_expansion=2.66,
        bias=False
    )

    # Forward pass
    y = gdfn(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    # Check shape preservation
    assert y.shape == x.shape, (
        f"Expected {x.shape}, got {y.shape}"
    )

    # Check for invalid values
    assert not torch.isnan(y).any(), "Output contains NaNs"
    assert not torch.isinf(y).any(), "Output contains Infs"

    print("\n✅ GDFN test passed!")



x = torch.randn(
    2, 48, 64, 64,
    requires_grad=True
)

model = GDFN(
    dim=48,
    ffn_expansion=2.66
)

y = model(x)

loss = y.mean()
loss.backward()

print("Output shape:", y.shape)
print("Gradient shape:", x.grad.shape)

assert x.grad is not None

print("✅ Forward and backward pass successful!")



if __name__ == "__main__":
    main()
