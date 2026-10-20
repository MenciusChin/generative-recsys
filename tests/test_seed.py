import os
import random

import numpy as np
import pytest
import torch

from generative_recsys.utils import seed_everything


@pytest.fixture(autouse=True)
def restore_global_state(monkeypatch):
    """seed_everything changes process-wide settings; undo them after each test."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def draw(device: str = "cpu") -> tuple:
    return (
        random.random(),
        np.random.rand(3).tolist(),  # noqa: NPY002 - the global RNG is what's being seeded
        torch.randn(4, device=device).cpu().tolist(),
    )


def test_same_seed_gives_identical_draws():
    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert draw() == first


def test_different_seeds_give_different_draws():
    seed_everything(1)
    first = draw()
    seed_everything(2)
    assert draw() != first


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_seeding_covers_mps():
    seed_everything(7)
    first = draw("mps")
    seed_everything(7)
    assert draw("mps") == first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_seeding_covers_cuda():
    seed_everything(7)
    first = draw("cuda")
    seed_everything(7)
    assert draw("cuda") == first


def test_deterministic_flags_are_set():
    seed_everything(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_false_leaves_flags_alone():
    torch.use_deterministic_algorithms(False)
    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
