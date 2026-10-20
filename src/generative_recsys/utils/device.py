"""Pick the torch device for a run."""

import torch

AUTO = "auto"


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return the requested device, or the best available one (CUDA, then MPS, then CPU).

    ``device`` may be ``None`` or ``"auto"`` for automatic selection, or anything
    ``torch.device`` accepts (``"cpu"``, ``"mps"``, ``"cuda"``, ``"cuda:1"``).
    Explicitly requesting a backend that isn't available raises instead of
    silently falling back, so a config asking for CUDA never runs on CPU by accident.
    """
    if device is None or device == AUTO:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Device {str(resolved)!r} requested but CUDA is not available.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"Device {str(resolved)!r} requested but MPS is not available.")
    return resolved
