#!/usr/bin/env python
"""
Torch/CUDA HMC driver for the D=4 pIKKT matrix model.

Key pieces:
- run_simulation: orchestrates seeding, I/O setup, HMC trajectories.
- helper utilities: checkpoint loading, thermalization, profiling.
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
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import torch

# Support both package and script execution
if __package__:
    from .config import dtype, device
    from .hmc import SimulationParams, measure_observables, update
    from .algebra import random_hermitian
    from .cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE
else:
    # When executed as "python pIKKT4D/main.py"
    print(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pIKKT4D.config import dtype, device  # type: ignore
    from pIKKT4D.hmc import SimulationParams, measure_observables, update  # type: ignore
    from pIKKT4D.algebra import random_hermitian  # type: ignore
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
    """Clear or allow existing outputs depending on the user's intent."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not (force or allow_existing):
        existing_str = "\n".join(existing)
        raise FileExistsError(
            f"Output files already exist:\n{existing_str}\nUse --force to overwrite, --resume to append, or change --name/--data-path."
        )
    if force:
        for path in existing:
            os.remove(path)


def spinJMatrices(j_val: float):
    "Generate spin-j angular momentum matrices Jx, Jy, Jz on CPU with NumPy."
    dim = int(round(2 * j_val + 1))

    Jp = np.zeros((dim, dim), dtype=np.complex128)

    # Physical m-values in descending order: j, j-1, ..., -j
    m_vals = np.arange(j_val, -j_val - 1, -1, dtype=np.float64)

    # Ladder operator: J+ |m> = sqrt(j(j+1) - m(m+1)) |m+1>
    # In descending order, raising moves one index up (row = col-1).
    for col in range(1, dim):
        m = m_vals[col]
        Jp[col - 1, col] = np.sqrt(j_val * (j_val + 1) - m * (m + 1))

    Jm = Jp.conj().T

    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    Jz = np.diag(m_vals)

    assert np.allclose(Jx @ Jy - Jy @ Jx, 1j * Jz, atol=1e-7)
    assert np.allclose(Jy @ Jz - Jz @ Jy, 1j * Jx, atol=1e-7)
    assert np.allclose(Jz @ Jx - Jx @ Jz, 1j * Jy, atol=1e-7)

    return np.stack([Jx, Jy, Jz], axis=0)


def initialize_fields(params: SimulationParams) -> torch.Tensor:
    """Initialize configuration matrices to random hermitian or a spin-j representation."""
    X = torch.zeros((params.nmat, params.ncol, params.ncol), dtype=dtype, device=device)
    if params.spin == 0.0:
        return X
        # for i in range(nmat):
        #     X[i] = random_hermitian(ncol, dtype=dtype, device=device)
    else:
        J_matrices = torch.from_numpy(spinJMatrices(params.spin)).to(dtype=dtype, device=device)
        ntimes = params.ncol // J_matrices.shape[1]
        for i in range(3):
            X[i] = (2/3 + params.omega) * torch.kron(torch.eye(ntimes, dtype=dtype, device=device), J_matrices[i])

    return X


def load_configuration(resume: bool, ckpt_path: str, X: torch.Tensor) -> tuple[bool, torch.Tensor]:
    """Try loading a checkpoint; return (did_resume, config)."""
    if resume and os.path.isfile(ckpt_path):
        print("Reading old configuration file:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        return True, ckpt["X"]
    if resume:
        print("Configuration not found, loading fresh")
    return False, X


def thermalize(X: torch.Tensor, params: SimulationParams, steps: int = 5) -> torch.Tensor:
    """Warm-up trajectories that always accept to reach equilibrium quickly. For these steps, we reject rarely and also increase nsteps and reduce dt."""
    print("Thermalization steps, accept most jumps")
    therm_params = replace(params, nsteps=int(params.nsteps * 1.5), dt=params.dt / 10)
    acc_count = 0
    for _ in range(steps):
        X, acc_count = update(X, acc_count, therm_params, reject_prob=0.0)
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
    """Set RNG seeds for reproducibility when requested."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_simulation(args: argparse.Namespace) -> torch.Tensor:
    """Entry point: configure and launch the HMC loop."""
    dt = args.step_size / args.nsteps
    params = SimulationParams(
        nmat=NMAT_DEFAULT,
        ncol=args.ncol,
        coupling=args.coupling,
        dt=dt,
        nsteps=args.nsteps,
        omega=args.omega,
        pIKKT_type=args.pIKKT_type,
        spin=args.spin,
        source=args.source,
    )

    os.makedirs(args.data_path, exist_ok=True)
    paths = build_paths(args.name, args.omega, args.coupling, args.ncol, args.data_path, args.pIKKT_type)
    allow_existing = args.resume and not args.fresh
    ensure_output_slots([paths["eigs"], paths["comms"]], force=args.force, allow_existing=allow_existing)

    print("\n------------------------------------------------")
    print("Configuration:")
    print(f"  4D Polarized IKKT type   = {params.pIKKT_type}")
    print(f"  Matrix size N            = {params.ncol}")
    print(f"  Number of Trajectories   = {args.niters}")
    print(f"  Coupling g               = {params.coupling}")
    if params.pIKKT_type == "type2":
        print(f"  Omega2/Omega1            = {params.omega}")
    print(f"  Step size, Nsteps        = {args.step_size}, {params.nsteps} (dt = {params.dt})")
    print(f"  Save                     = {args.save}")
    print(f"  outputs                  = {paths['eigs']}")
    print(f"  device/dtype             = {device}/{dtype}")
    if params.source is not None:
        print(f"  Source                   = {args.source}")
    print("------------------------------------------------\n")

    if args.dry_run:
        print("Dry run; resolved configuration:")
        return

    seed_everything(args.seed)
    profiler = maybe_profile(args.profile)

    X = initialize_fields(params)
    X.zero_()
    resumed, X = load_configuration(args.resume and not args.fresh, paths["ckpt"], X)

    if not resumed:
        # X.zero_()
        ensure_output_slots([paths["eigs"], paths["comms"]], force=True)
        X = thermalize(X, params)

    acc_count = 0
    ev_X_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []

    for MDTU in range(1, args.niters + 1):
        X, acc_count = update(X, acc_count, params)

        eigs, corrs = measure_observables(X, params)
        ev_X_buf.append(np.concatenate(eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if MDTU % args.save_every == 0:
            save_buffers(ev_X_buf, corr_buf, paths)
            if args.save:
                print(
                    f"Iteration {MDTU}, Acceptance rate so far = {acc_count/MDTU:.3f}, "
                    f"trX_1^2 = {1 / params.ncol * torch.trace(X[0] @ X[0]).item().real:.5f}, "
                    f"trX_4^2 = {1 / params.ncol * torch.trace(X[3] @ X[3]).item().real:.5f}. "
                    f"Saving configuration to {paths['ckpt']}"
                )
                torch.save({"X": X}, paths["ckpt"])
            else:
                print(
                    f"Iteration {MDTU}, Acceptance rate so far = {acc_count/MDTU:.3f}, "
                    f"trX_1^2 = {1 / params.ncol * torch.trace(X[0] @ X[0]).item().real:.5f}, "
                    f"trX_4^2 = {1 / params.ncol * torch.trace(X[3] @ X[3]).item().real:.5f}"
                )

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
