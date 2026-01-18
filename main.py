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
import json
import os
import pstats
import sys
import time
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# Support both package and script execution
if not __package__:
    # When executed as "python MatrixModelHMC_pytorch/main.py"
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from MatrixModelHMC_pytorch import config
    from MatrixModelHMC_pytorch.hmc import HMCParams, update, thermalize
    from MatrixModelHMC_pytorch.models import (
        MatrixModel,
        build_model,
        PIKKTTypeIModel,
        PIKKTTypeIIModel,
        gammaMajorana,
        gammaWeyl,
    )
    from MatrixModelHMC_pytorch.cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from hmc import HMCParams, update, thermalize  # type: ignore
    from models import (  # type: ignore
        MatrixModel,
        build_model,
        PIKKTTypeIModel,
        PIKKTTypeIIModel,
        gammaMajorana,
        gammaWeyl,
    )
    from cli import parse_args, DEFAULT_DATA_PATH, DEFAULT_PROFILE  # type: ignore


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


def prepare_matrix_snapshot_dir(run_dir: str, *, force: bool, allow_existing: bool) -> tuple[str, int]:
    """Return the directory and current max index for raw matrix snapshots."""
    target = os.path.join(run_dir, "all_mats")
    if os.path.exists(target):
        if force:
            shutil.rmtree(target)
        elif not allow_existing:
            raise FileExistsError(
                f"Matrix snapshot directory {target} already exists. Use --force or --resume appropriately."
            )
    os.makedirs(target, exist_ok=True)
    max_end = 0
    if allow_existing:
        for name in os.listdir(target):
            if not name.endswith(".npy"):
                continue
            base = name[:-4]
            if "_" not in base:
                continue
            start_str, end_str = base.split("_", 1)
            if not (start_str.isdigit() and end_str.isdigit()):
                continue
            max_end = max(max_end, int(end_str))
    return target, max_end


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


def _append_npz(path: str, values: np.ndarray, *, key: str, dtype: np.dtype) -> None:
    """Append new measurements to an .npz file (reloads existing data if present)."""
    if values.size == 0:
        return
    new_values = np.asarray(values, dtype=dtype)
    if os.path.exists(path):
        with np.load(path) as existing:
            existing_values = existing[key]
        new_values = np.concatenate((existing_values, new_values), axis=0)
    np.savez(path, **{key: new_values})


def save_buffers(ev_buf: list[np.ndarray], corr_buf: list[np.ndarray], paths: dict[str, str]) -> None:
    if ev_buf:
        stacked = np.stack(ev_buf).astype(np.complex128)
        _append_npz(paths["eigs"], stacked, key="values", dtype=np.complex128)
        ev_buf.clear()
    if corr_buf:
        stacked = np.stack(corr_buf).astype(np.complex128)
        _append_npz(paths["corrs"], stacked, key="values", dtype=np.complex128)
        corr_buf.clear()


def create_matrix_snapshot_chunk(
    directory: str,
    *,
    start: int,
    count: int,
    dtype: np.dtype,
    shape_tail: tuple[int, ...],
) -> np.memmap:
    """Create a memmap file for a chunk of snapshots."""
    end = start + count - 1
    filename = os.path.join(directory, f"{start:08d}_{end:08d}.npy")
    return np.lib.format.open_memmap(filename, mode="w+", dtype=dtype, shape=(count, *shape_tail))


def seed_everything(seed: int | None) -> None:
    """Seed numpy and torch generators for reproducibility when a seed is provided."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def write_run_metadata(path: str, model: MatrixModel, args: argparse.Namespace) -> None:
    summary = {
        "model": model.run_metadata(),
        "cli": {
            "name": args.name,
            "data_path": args.data_path,
            "niters": args.niters,
            "step_size": args.step_size,
            "nsteps": args.nsteps,
            "save_every": args.save_every,
            "save_all_mats": args.saveAllMats,
            "fresh": args.fresh,
            "resume": args.resume,
            "seed": args.seed,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_simulation(args: argparse.Namespace) -> MatrixModel:
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
    write_run_metadata(paths["meta"], model, args)

    print("\n------------------------------------------------")
    print("Configuration:")
    print(f"  Model                    = {model.model_name}")
    print(f"  Matrix size N            = {model.ncol}")
    print(f"  Number of Trajectories   = {args.niters}")
    for line in model.extra_config_lines():
        print(line)
    print(f"  Step size, Nsteps        = {args.step_size}, {hmc_params.nsteps} (dt = {hmc_params.dt})")
    print(f"  Save                     = {args.save}")
    print(f"  outputs                  = {paths['dir']}")
    print(f"  device/dtype             = {config.device}/{config.dtype}")
    source = getattr(model, "source", None)
    if source is not None:
        print(f"  Source                   = {args.source}")
    print("------------------------------------------------\n")

    if args.dry_run:
        print("Dry run; resolved configuration:")
        return model

    seed_everything(args.seed)
    profiler = maybe_profile(args.profile)

    resumed = model.initialize_configuration(args, paths["ckpt"])

    if not resumed:
        ensure_output_slots([paths["eigs"], paths["corrs"]], force=True)
        # thermalize(model, hmc_params)

    acc_count = 0
    ev_X_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []
    matrix_snapshot_dir = None
    matrix_snapshot_offset = 0
    chunk: np.memmap | None = None
    chunk_start = 0
    chunk_size = args.save_every
    if args.saveAllMats:
        matrix_snapshot_dir, matrix_snapshot_offset = prepare_matrix_snapshot_dir(
            paths["dir"], force=args.force, allow_existing=allow_existing
        )
        state_shape = model.get_state().shape
        state_dtype = np.dtype(model.get_state().detach().cpu().numpy().dtype)

    for iter in range(1, args.niters + 1):
        acc_count = update(acc_count, hmc_params, model)
        if matrix_snapshot_dir is not None:
            global_iter = matrix_snapshot_offset + iter
            if chunk is None or (global_iter - 1) % chunk_size == 0:
                if chunk is not None:
                    chunk.flush()
                chunk_start = global_iter
                remaining = args.niters - iter + 1
                chunk = create_matrix_snapshot_chunk(
                    matrix_snapshot_dir,
                    start=chunk_start,
                    count=min(chunk_size, remaining),
                    dtype=state_dtype,
                    shape_tail=state_shape,
                )
            state = model.get_state().detach()
            if state.device.type != "cpu":
                state = state.to("cpu")
            chunk[global_iter - chunk_start] = state.numpy()

        eigs, corrs = model.measure_observables()
        ev_X_buf.append(np.stack(eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if iter % args.save_every == 0:
            save_buffers(ev_X_buf, corr_buf, paths)
            if chunk is not None:
                chunk.flush()
            status_string = model.status_string()
            print(f"Iteration {iter}, Acceptance rate so far = {acc_count/iter:.3f}, " + status_string)
            if args.save:
                print(f"Saving configuration to {paths['ckpt']}")
                model.save_state(paths["ckpt"])

    if acc_count / max(args.niters, 1) < 0.5:
        print("WARNING: Acceptance rate is below 50%")

    if chunk is not None:
        chunk.flush()

    stop_and_report_profile(profiler)
    return model


# def main(argv: Sequence[str]):
#     start_time = time.time()
#     print("STARTED:", datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))

#     args = parse_args(argv)
#     config.configure_device(args.noGPU)
#     config.configure_dtype(args.complex64)
#     model = run_simulation(args)

#     print("Runtime =", time.time() - start_time, "s")

#     return model

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    start_time = time.time()
    print("STARTED:", datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))

    config.configure_device(args.noGPU)
    config.configure_dtype(args.complex64)
    model = run_simulation(args)
    
    print("Runtime =", time.time() - start_time, "s")
