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
from dataclasses import replace

import numpy as np
import torch

# Support both package and script execution
if __package__:
    from .config import dtype, device
    from .hmc import HMCParams, update
    from .model import ModelParams, measure_observables, potential, spinJMatrices
    from .cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE
else:
    # When executed as "python pIKKT4D/main.py"
    print(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pIKKT4D.config import dtype, device  # type: ignore
    from pIKKT4D.hmc import HMCParams, update  # type: ignore
    from pIKKT4D.model import ModelParams, measure_observables, potential, spinJMatrices  # type: ignore
    from pIKKT4D.cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE  # type: ignore


NMAT_DEFAULT = 4
DATA_PATH = DEFAULT_DATA_PATH
PROFILE_DEFAULT = DEFAULT_PROFILE


def build_paths(name_prefix: str, omega: float, coupling: float, ncol: int, data_path: str, typ: int) -> dict[str, str]:
    if typ == 2:
        ev = os.path.join(data_path, f"eigenvalues_{name_prefix}_type2_Omega{round(omega, 2)}_g{round(coupling, 4)}_N{ncol}.dat")
        comms = os.path.join(data_path, f"comm2_{name_prefix}_type2_Omega{round(omega, 2)}_g{round(coupling, 4)}_N{ncol}.dat")
        ckpt = os.path.join(data_path, f"config_{name_prefix}_type2_Omega{round(omega, 2)}_g{round(coupling, 4)}_N{ncol}.pt")
    else:
        ev = os.path.join(data_path, f"eigenvalues_{name_prefix}_type1_g{round(coupling, 4)}_N{ncol}.dat")
        comms = os.path.join(data_path, f"comm2_{name_prefix}_type1_g{round(coupling, 4)}_N{ncol}.dat")
        ckpt = os.path.join(data_path, f"config_{name_prefix}_type1_g{round(coupling, 4)}_N{ncol}.pt")
    return {"eigs": ev, "comms": comms, "ckpt": ckpt}


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


def initialize_fields(params: ModelParams, spin: float) -> torch.Tensor:
    """Construct the starting field configuration, optionally embedding a spin-j background."""
    X = torch.zeros((params.nmat, params.ncol, params.ncol), dtype=dtype, device=device)
    if spin == 0.0:
        return X
    else:
        J_matrices = torch.from_numpy(spinJMatrices(spin)).to(dtype=dtype, device=device)
        ntimes = params.ncol // J_matrices.shape[1]
        for i in range(3):
            X[i] = (2/3 + params.omega) * torch.kron(torch.eye(ntimes, dtype=dtype, device=device), J_matrices[i])

    return X


def load_configuration(resume: bool, ckpt_path: str, X: torch.Tensor) -> tuple[bool, torch.Tensor]:
    """Restore a saved configuration when requested, otherwise return the input."""
    if resume and os.path.isfile(ckpt_path):
        print("Reading old configuration file:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        return True, ckpt["X"]
    if resume:
        print("Configuration not found, loading fresh")
    return False, X


def thermalize(X: torch.Tensor, hmc_params: HMCParams, model_params: ModelParams, steps: int = 5) -> torch.Tensor:
    """Run short, mostly-accepting trajectories to move the system toward equilibrium."""
    print("Thermalization steps, accept most jumps")
    therm_params = replace(hmc_params, nsteps=int(hmc_params.nsteps * 1.5), dt=hmc_params.dt / 10)
    acc_count = 0
    for _ in range(steps):
        X, acc_count = update(X, acc_count, therm_params, model_params, potential, reject_prob=0.0)
    print("End of thermalization")
    return X


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
    with open(paths["eigs"], "a") as f2:
        np.array(ev_buf).astype("complex128").tofile(f2)
        ev_buf.clear()
    with open(paths["comms"], "a") as egf:
        np.array(corr_buf).astype("complex128").tofile(egf)
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
    model_params = ModelParams(
        nmat=NMAT_DEFAULT,
        ncol=args.ncol,
        coupling=args.coupling,
        omega=args.omega,
        pIKKT_type=args.pIKKT_type,
        source=args.source,
    )
    hmc_params = HMCParams(
        dt=dt,
        nsteps=args.nsteps,
    )

    os.makedirs(args.data_path, exist_ok=True)
    paths = build_paths(args.name, args.omega, args.coupling, args.ncol, args.data_path, args.pIKKT_type)
    allow_existing = args.resume and not args.fresh
    ensure_output_slots([paths["eigs"], paths["comms"]], force=args.force, allow_existing=allow_existing)

    print("\n------------------------------------------------")
    print("Configuration:")
    print(f"  4D Polarized IKKT type   = {model_params.pIKKT_type}")
    print(f"  Matrix size N            = {model_params.ncol}")
    print(f"  Number of Trajectories   = {args.niters}")
    print(f"  Coupling g               = {model_params.coupling}")
    if model_params.pIKKT_type == 2:
        print(f"  Omega2/Omega1            = {model_params.omega}")
    print(f"  Step size, Nsteps        = {args.step_size}, {hmc_params.nsteps} (dt = {hmc_params.dt})")
    print(f"  Save                     = {args.save}")
    print(f"  outputs                  = {paths['eigs']}")
    print(f"  device/dtype             = {device}/{dtype}")
    if model_params.source is not None:
        print(f"  Source                   = {args.source}")
    print("------------------------------------------------\n")

    if args.dry_run:
        print("Dry run; resolved configuration:")
        return

    seed_everything(args.seed)
    profiler = maybe_profile(args.profile)

    X = initialize_fields(model_params, args.spin)
    X.zero_()
    resumed, X = load_configuration(args.resume and not args.fresh, paths["ckpt"], X)

    if not resumed:
        ensure_output_slots([paths["eigs"], paths["comms"]], force=True)
        X = thermalize(X, hmc_params, model_params)

    acc_count = 0
    ev_X_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []

    for MDTU in range(1, args.niters + 1):
        X, acc_count = update(X, acc_count, hmc_params, model_params, potential)

        eigs, corrs = measure_observables(X, model_params)
        ev_X_buf.append(np.concatenate(eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if MDTU % args.save_every == 0:
            save_buffers(ev_X_buf, corr_buf, paths)
            if model_params.pIKKT_type == 1:
                status_string = f"trX_1^2 = {1 / model_params.ncol * torch.trace(X[0] @ X[0]).item().real:.5f}, " +\
                  f"trX_4^2 = {1 / model_params.ncol * torch.trace(X[3] @ X[3]).item().real:.5f}. "
            else:
                status_string = f"casimir = {1 / model_params.ncol * torch.trace(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]).item().real:.5f}, " +\
                  f"trX_4^2 = {1 / model_params.ncol * torch.trace(X[3] @ X[3]).item().real:.5f}. "
                
            print(f"Iteration {MDTU}, Acceptance rate so far = {acc_count/MDTU:.3f}, " + status_string)
            if args.save:
                print(f"Saving configuration to {paths['ckpt']}")
                torch.save({"X": X}, paths["ckpt"])

    if acc_count / max(args.niters, 1) < 0.5:
        print("WARNING: Acceptance rate is below 50%")

    stop_and_report_profile(profiler)
    return X


def main(argv: Sequence[str]) -> torch.Tensor:
    start_time = time.time()
    print("STARTED:", datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))

    args = parse_args(argv)
    Xfin = run_simulation(args)

    print("Runtime =", time.time() - start_time, "s")

    return Xfin


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    Xfin = main(sys.argv[1:])
