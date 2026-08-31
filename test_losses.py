"""
Quick manual test for losses/pixel_loss.py, losses/ssim_loss.py,
and losses/combined_loss.py.

Checks, for each loss:
    1. Identical images  -> loss should be ~0 (or SSIM loss ~0)
    2. Different images  -> loss should be > 0
    3. Gradients flow    -> pred.grad is not None after backward()

"""

import torch

from losses.pixel_loss import PixelLoss
from losses.ssim_loss import SSIMLoss
from losses.combined_loss import CombinedLoss


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    return condition


def test_pixel_loss():
    print("\n--- PixelLoss ---")
    all_ok = True

    for loss_type in ["l1", "l2", "charbonnier"]:
        criterion = PixelLoss(loss_type=loss_type)

        target = torch.rand(2, 1, 64, 64)
        identical = target.clone()
        different = torch.rand(2, 1, 64, 64)

        loss_same = criterion(identical, target)
        loss_diff = criterion(different, target)

        print(f"  {loss_type:>11s} | identical={loss_same.item():.6f} "
              f"| different={loss_diff.item():.6f}")

        all_ok &= check(f"{loss_type} identical ~0",
                         loss_same.item() < 1e-5, f"(got {loss_same.item():.2e})")
        all_ok &= check(f"{loss_type} different > identical",
                         loss_diff.item() > loss_same.item())

    # Gradient flow check
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    criterion = PixelLoss(loss_type="charbonnier")
    loss = criterion(pred, target)
    loss.backward()
    all_ok &= check("gradients flow to pred", pred.grad is not None)

    return all_ok


def test_ssim_loss():
    print("\n--- SSIMLoss ---")
    all_ok = True

    criterion = SSIMLoss(data_range=1.0)

    target = torch.rand(2, 1, 64, 64)
    identical = target.clone()
    different = torch.rand(2, 1, 64, 64)

    loss_same = criterion(identical, target)
    loss_diff = criterion(different, target)

    print(f"  identical={loss_same.item():.6f} | different={loss_diff.item():.6f}")

    all_ok &= check("identical images -> loss ~0",
                     loss_same.item() < 1e-4, f"(got {loss_same.item():.2e})")
    all_ok &= check("different images -> loss > identical",
                     loss_diff.item() > loss_same.item())

    # Gradient flow check
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    loss = criterion(pred, target)
    loss.backward()
    all_ok &= check("gradients flow to pred", pred.grad is not None)

    return all_ok


def test_combined_loss():
    print("\n--- CombinedLoss ---")
    all_ok = True

    criterion = CombinedLoss(
        pixel_loss_type="charbonnier",
        pixel_weight=1.0,
        ssim_weight=0.2,
        data_range=1.0,
    )

    target = torch.rand(2, 1, 64, 64)
    identical = target.clone()
    different = torch.rand(2, 1, 64, 64)

    total_same, dict_same = criterion(identical, target)
    total_diff, dict_diff = criterion(different, target)

    print(f"  identical -> {dict_same}")
    print(f"  different -> {dict_diff}")

    all_ok &= check("identical images -> total loss ~0",
                     total_same.item() < 1e-4, f"(got {total_same.item():.2e})")
    all_ok &= check("different images -> total loss > identical",
                     total_diff.item() > total_same.item())
    all_ok &= check("loss_dict has expected keys",
                     set(dict_same.keys()) == {"pixel_loss", "ssim_loss", "total_loss"})

    # Gradient flow check
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    total, _ = criterion(pred, target)
    total.backward()
    all_ok &= check("gradients flow to pred", pred.grad is not None)

    return all_ok


def main():
    torch.manual_seed(42)

    results = {
        "PixelLoss": test_pixel_loss(),
        "SSIMLoss": test_ssim_loss(),
        "CombinedLoss": test_combined_loss(),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name:>13s}: {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\nAll losses look correct.")
    else:
        print("\nSome checks failed - see details above.")


if __name__ == "__main__":
    main()