import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets the random seed across all libraries used in the pipeline
    (Python, NumPy, PyTorch CPU/GPU) so experiments are reproducible.

    Args:
        seed          : the seed value to use everywhere
        deterministic : if True, forces PyTorch's CuDNN backend to use
                         deterministic algorithms. This can slightly slow
                         down training but guarantees the same results
                         across runs on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster, but introduces run-to-run non-determinism
        torch.backends.cudnn.benchmark = True