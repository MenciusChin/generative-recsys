import pytest
import torch

from generative_recsys.utils import get_device


@pytest.fixture
def available(monkeypatch):
    """Fake which accelerators exist: available(cuda=..., mps=...)."""

    def set_available(*, cuda: bool, mps: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    return set_available


@pytest.mark.parametrize("request_", [None, "auto"])
@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_prefers_cuda_then_mps_then_cpu(available, request_, cuda, mps, expected):
    available(cuda=cuda, mps=mps)
    assert get_device(request_) == torch.device(expected)


@pytest.mark.parametrize("request_", ["cpu", torch.device("cpu")])
def test_cpu_override_always_works(available, request_):
    available(cuda=True, mps=True)
    assert get_device(request_) == torch.device("cpu")


def test_override_beats_auto_order(available):
    available(cuda=True, mps=True)
    assert get_device("mps") == torch.device("mps")


def test_cuda_index_is_kept(available):
    available(cuda=True, mps=False)
    assert get_device("cuda:1") == torch.device("cuda", 1)


@pytest.mark.parametrize(
    ("request_", "backend"),
    [("cuda", "CUDA"), ("cuda:0", "CUDA"), ("mps", "MPS")],
)
def test_unavailable_override_raises(available, request_, backend):
    available(cuda=False, mps=False)
    with pytest.raises(RuntimeError, match=f"{backend} is not available"):
        get_device(request_)


def test_invalid_device_string_raises():
    with pytest.raises(RuntimeError):
        get_device("not-a-device")
