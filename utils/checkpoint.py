import os
import torch


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str = "last.pth",
    is_best: bool = False,
    best_filename: str = "best.pth",
) -> str:
    """
    Saves a training checkpoint to disk.

    Args:
        state          : dict containing everything needed to resume training,
                         e.g. {"epoch": ..., "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(),
                               "best_metric": ...}
        checkpoint_dir : directory where checkpoints are stored
        filename       : filename for the regular/latest checkpoint
        is_best        : if True, also saves a copy as best_filename
        best_filename  : filename used for the best checkpoint

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    map_location: str = "cpu",
) -> dict:
    """
    Loads a checkpoint from disk into a model (and optionally an optimizer).

    Args:
        checkpoint_path : path to the .pth checkpoint file
        model           : model to load the weights into (modified in-place)
        optimizer       : optimizer to restore state into (optional)
        map_location    : device to map the checkpoint tensors to

    Returns:
        The full checkpoint dict, so the caller can read extra fields
        such as "epoch" or "best_metric" to resume training.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint