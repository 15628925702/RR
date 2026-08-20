"""Device and dtype selection with an explicit CPU fallback."""

from __future__ import annotations


def select_device(requested: str = "auto") -> str:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return "cuda" if available else "cpu"

