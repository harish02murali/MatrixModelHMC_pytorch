"""Shared configuration: device selection, dtypes."""

from __future__ import annotations

import os
import torch

# Ensure Apple GPUs fall back cleanly to CPU for unsupported ops.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Environment toggles mirror the original script.
ALLOW_TF32 = os.getenv("IKKT_ALLOW_TF32", "1") == "1"
ENABLE_TORCH_COMPILE = os.getenv("IKKT_COMPILE", "0") == "1"

def _real_dtype_for(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if complex_dtype == torch.complex64 else torch.float64


# Default to double-precision complex unless overridden via CLI.
dtype = torch.complex128
real_dtype = _real_dtype_for(dtype)


def _enable_tf32() -> None:
    if not ALLOW_TF32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


def configure_device(noGPUFlag: bool | None) -> torch.device:
    """Select device from CLI preference."""
    
    requested = "cpu" if noGPUFlag else "auto"
    dev: torch.device
    if requested == "auto":
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


def configure_dtype(use_complex64: bool) -> torch.dtype:
    """Set the global complex dtype used for matrices."""
    global dtype, real_dtype
    dtype = torch.complex64 if use_complex64 else torch.complex128
    real_dtype = _real_dtype_for(dtype)
    return dtype


# Default to CPU until configure_device is called.
device = torch.device("cpu")


__all__ = [
    "ALLOW_TF32",
    "ENABLE_TORCH_COMPILE",
    "configure_device",
    "configure_dtype",
    "device",
    "dtype",
    "real_dtype",
]
