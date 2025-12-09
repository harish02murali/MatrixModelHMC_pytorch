"""Command Line Interface parsing helpers for the D=4 pIKKT HMC driver."""

import argparse
import os
from typing import Sequence

import numpy as np

root_path = ("/mnt/beegfs/hmurali/ML" if os.path.isdir('/mnt/beegfs/hmurali/ML') else "../")
DEFAULT_DATA_PATH = os.path.join(root_path, "data")
DEFAULT_PROFILE = False


def _parse_source(expr: str):
    """Evaluate a numpy-based expression like np.linspace(-1,1,20) for --source."""
    try:
        return eval(expr, {"np": np}, {})  # noqa: S307 - controlled namespace for numpy expressions
    except Exception as exc:  # pragma: no cover - arg parsing error path
        raise argparse.ArgumentTypeError(f"Invalid --source expression {expr!r}: {exc}") from exc


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """User-facing CLI with flags."""
    parser = argparse.ArgumentParser(
        description="Choosing pIKKT model type and simulation parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pIKKT-type", type=int, choices=[1, 2], default=1, help="Choose pIKKT model type (1 or 2)")
    parser.add_argument("--resume", action="store_true", help="Load a checkpoint if present")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints and start from zero fields")
    parser.add_argument("--save", action="store_true", help="Save configurations every --save-every trajectories")
    parser.add_argument("--ncol", type=int, default=4, help="Matrix size N")
    parser.add_argument("--niters", type=int, default=300, help="Number of trajectories to run")
    parser.add_argument("--coupling", type=float, default=100, help="Coupling g")
    parser.add_argument("--omega", type=float, default=1.0, help="Ratio of Omega2/Omega1")
    parser.add_argument("--name", type=str, default="run", help="Prefix for outputs")
    parser.add_argument("--step-size", type=float, dest="step_size", default=2, help="Leapfrog step size Δt")
    parser.add_argument("--nsteps", type=int, default=180, help="Leapfrog steps per trajectory")
    parser.add_argument("--save-every", type=int, dest="save_every", default=10, help="Write observables every K trajectories")
    parser.add_argument("--data-path", type=str, dest="data_path", default=DEFAULT_DATA_PATH, help="Directory for outputs and checkpoints")
    parser.add_argument("--profile", dest="profile", action="store_true", default=DEFAULT_PROFILE, help="Enable cProfile sampling")
    parser.add_argument("--no-profile", dest="profile", action="store_false", help="Disable cProfile sampling")
    parser.add_argument("--seed", type=int, default=None, help="Set RNG seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files and checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration and exit")
    parser.add_argument("--spin", type=float, default=0.0, help="Spin for the fuzzy sphere. Should be less than (N-1)/2.")
    parser.add_argument("--source", type=_parse_source, default=None, help="Numpy expression for source, e.g., np.linspace(-1,1,20)")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Clamp obviously invalid inputs early to avoid cryptic runtime failures."""
    if args.ncol < 1:
        raise ValueError("--ncol must be positive")
    if args.niters < 1:
        raise ValueError("--niters must be positive")
    if args.coupling <= 0:
        raise ValueError("--coupling must be positive")
    if args.nsteps < 1:
        raise ValueError("--nsteps must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")
    if args.source is not None and args.source.shape != (args.ncol,):
        raise ValueError(f"--source expression must evaluate to shape ({args.ncol},), got {args.source.shape}") 


__all__ = ["parse_args", "DEFAULT_DATA_PATH", "DEFAULT_PROFILE"]
