"""Inference-device selection."""


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` as NVIDIA CUDA, Apple MPS, then CPU."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
