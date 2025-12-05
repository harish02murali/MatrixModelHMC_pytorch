"""Shared configuration: device selection, dtypes, and gamma matrices."""

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


def select_device() -> torch.device:
    """Pick CUDA if available, otherwise CPU. Enable TF32 when requested."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if ALLOW_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
        print("Using CUDA device:", torch.cuda.get_device_name(dev))
    else:
        dev = torch.device("cpu")
        print("GPU not available, using CPU.")
    return dev


device = select_device()

__all__ = [
    "ALLOW_TF32",
    "ENABLE_TORCH_COMPILE",
    "device",
    "dtype",
    "real_dtype",
]
