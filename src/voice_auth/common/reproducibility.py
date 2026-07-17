"""Reproducibility helpers (seeds and deterministic flags)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and (if available) PyTorch RNGs.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducible augmentations."""
    worker_seed = (np.random.get_state()[1][0] + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
