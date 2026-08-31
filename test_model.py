"""

Quick check that the full Restormer model runs correctly.
Verifies the forward pass and that input/output shapes match.
"""

import torch

from restormer.model import Restormer


def main():

    x = torch.randn(2, 1, 256, 256)

    model = Restormer()

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    assert y.shape == x.shape, (
        f"Expected {x.shape}, got {y.shape}"
    )

    print("✅ Restormer test passed!")


if __name__ == "__main__":
    main()