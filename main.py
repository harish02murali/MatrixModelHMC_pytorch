#!/usr/bin/env python
"""
Entry point for running Hybrid Monte Carlo on the D=4 polarized IKKT matrix model.

Responsibilities:
- configure model and HMC parameters from the CLI,
- manage checkpoint/observable I/O,
- launch HMC trajectories and optional profiling.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# Support both package and script execution
if __package__:
    from . import config
    from .hmc import HMCParams, update, thermalize
    from .models import MatrixModel, build_model
    from .cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE
else:
    # When executed as "python pIKKT4D/main.py"
    print(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pIKKT4D import config  # type: ignore
    from pIKKT4D.hmc import HMCParams, update, thermalize  # type: ignore
    from pIKKT4D.models import MatrixModel, build_model  # type: ignore
    from pIKKT4D.cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE  # type: ignore


DATA_PATH = DEFAULT_DATA_PATH
PROFILE_DEFAULT = DEFAULT_PROFILE


def ensure_output_slots(paths: Iterable[str], force: bool, allow_existing: bool = False) -> None:
    """Validate writable targets for outputs and optionally clear existing files."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not (force or allow_existing):
        existing_str = "\n".join(existing)
        raise FileExistsError(
            f"Output files already exist:\n{existing_str}\nUse --force to overwrite, --resume to append, or change --name/--data-path."
        )
    if force:
        for path in existing:
            os.remove(path)


def maybe_profile(enabled: bool):
    if not enabled:
        return None
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def stop_and_report_profile(profiler: cProfile.Profile | None):
    if profiler is None:
        return
    profiler.disable()
    ps = pstats.Stats(profiler)
    ps.strip_dirs().sort_stats(pstats.SortKey.TIME)
    ps.print_stats(10)


def save_buffers(ev_buf: list[np.ndarray], corr_buf: list[np.ndarray], paths: dict[str, str]) -> None:
    with open(paths["eigs"], "a") as f:
        np.array(ev_buf).astype("complex128").tofile(f)
        ev_buf.clear()
    with open(paths["corrs"], "a") as f:
        np.array(corr_buf).astype("complex128").tofile(f)
        corr_buf.clear()


def seed_everything(seed: int | None) -> None:
    """Seed numpy and torch generators for reproducibility when a seed is provided."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_simulation(args: argparse.Namespace) -> torch.Tensor:
    """Configure model/HMC parameters and execute the requested number of trajectories."""
    dt = args.step_size / args.nsteps
    model = build_model(args)
    hmc_params = HMCParams(
        dt=dt,
        nsteps=args.nsteps,
    )

    paths = model.build_paths(args.name, args.data_path)
    os.makedirs(paths["dir"], exist_ok=True)
    allow_existing = args.resume and not args.fresh
    ensure_output_slots([paths["eigs"], paths["corrs"]], force=args.force, allow_existing=allow_existing)

    print("\n------------------------------------------------")
    print("Configuration:")
    print(f"  Model                    = {model.name}")
    print(f"  Matrix size N            = {model.ncol}")
    print(f"  Number of Trajectories   = {args.niters}")
    for line in model.extra_config_lines():
        print(line)
    print(f"  Step size, Nsteps        = {args.step_size}, {hmc_params.nsteps} (dt = {hmc_params.dt})")
    print(f"  Save                     = {args.save}")
    print(f"  outputs                  = {paths['eigs']}")
    print(f"  device/dtype             = {config.device}/{config.dtype}")
    source = getattr(model, "source", None)
    if source is not None:
        print(f"  Source                   = {args.source}")
    print("------------------------------------------------\n")

    if args.dry_run:
        print("Dry run; resolved configuration:")
        return

    seed_everything(args.seed)
    profiler = maybe_profile(args.profile)

    # model.initialize_fields(args.spin)
    resumed = model.initialize_configuration(args, paths["ckpt"])

    if not resumed:
        ensure_output_slots([paths["eigs"], paths["corrs"]], force=True)
        thermalize(model, hmc_params)

    acc_count = 0
    ev_X_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []

    for MDTU in range(1, args.niters + 1):
        acc_count = update(acc_count, hmc_params, model)

        eigs, corrs = model.measure_observables()
        ev_X_buf.append(np.concatenate(eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if MDTU % args.save_every == 0:
            save_buffers(ev_X_buf, corr_buf, paths)
            status_string = model.status_string()
            print(f"Iteration {MDTU}, Acceptance rate so far = {acc_count/MDTU:.3f}, " + status_string)
            if args.save:
                print(f"Saving configuration to {paths['ckpt']}")
                model.save_state(paths["ckpt"])

    if acc_count / max(args.niters, 1) < 0.5:
        print("WARNING: Acceptance rate is below 50%")

    stop_and_report_profile(profiler)
    return model.get_state()


def main(argv: Sequence[str]) -> torch.Tensor:
    start_time = time.time()
    print("STARTED:", datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))

    args = parse_args(argv)
    config.configure_device(args.gpu)
    Xfin = run_simulation(args)

    print("Runtime =", time.time() - start_time, "s")

    return Xfin


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    Xfin = main(sys.argv[1:])
