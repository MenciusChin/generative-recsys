"""Seed every random number generator a run touches."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU, CUDA and MPS).

    With ``deterministic=True``, also ask torch for deterministic kernels. Ops
    without a deterministic implementation warn rather than fail, since some
    MPS kernels have none. Call this before any CUDA work so the cuBLAS
    workspace setting takes effect.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # inherited by DataLoader worker processes
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG used by third-party code
    torch.manual_seed(seed)  # also seeds CUDA and MPS

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
