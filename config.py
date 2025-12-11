"""Shared configuration: device selection, dtypes."""

from __future__ import annotations

import os
import torch

# Ensure Apple GPUs fall back cleanly to CPU for unsupported ops.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Environment toggles mirror the original script.
ALLOW_TF32 = os.getenv("IKKT_ALLOW_TF32", "1") == "1"
ENABLE_TORCH_COMPILE = os.getenv("IKKT_COMPILE", "0") == "1"

# Default to single-precision complex for speed unless overridden.
dtype = torch.complex64
real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32


def _enable_tf32() -> None:
    if not ALLOW_TF32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


def configure_device(gpuFlag: bool | None) -> torch.device:
    """Select device from CLI preference (cpu, cuda, auto)."""
    
    requested = "cuda" if gpuFlag else "cpu"
    dev: torch.device
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            dev = torch.device(requested)
            _enable_tf32()
            print(f"Using CUDA device: {torch.cuda.get_device_name(dev)}")
        else:
            print("CUDA requested but not available; falling back to CPU.")
            dev = torch.device("cpu")
    elif requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            _enable_tf32()
            print(f"Using CUDA device: {torch.cuda.get_device_name(dev)}")
        else:
            dev = torch.device("cpu")
            print("GPU not available, using CPU.")
    else:
        dev = torch.device("cpu")
        print("Using CPU.")

    global device
    device = dev
    return dev


# Default to CPU until configure_device is called.
device = torch.device("cpu")


__all__ = [
    "ALLOW_TF32",
    "ENABLE_TORCH_COMPILE",
    "configure_device",
    "device",
    "dtype",
    "real_dtype",
]
