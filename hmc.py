"""Model-agnostic Hybrid Monte Carlo kernels: force, leapfrog, and Metropolis step."""

import math
import random
from dataclasses import dataclass
from typing import Any, Callable
import torch

from .algebra import makeH, random_hermitian

PotentialFn = Callable[[torch.Tensor, Any], torch.Tensor]


@dataclass
class HMCParams:
    """Integrator controls for an HMC trajectory."""
    dt: float
    nsteps: int


def force(X: torch.Tensor, model_params: Any, potential_fn: PotentialFn) -> torch.Tensor:
    """Compute the traceless Hermitian force dV/dX for a given potential."""
    Y = X.detach().requires_grad_(True)
    pot = potential_fn(Y, model_params)
    pot.backward()
    return makeH(Y.grad)


def hamil(X: torch.Tensor, mom_X: torch.Tensor, model_params: Any, potential_fn: PotentialFn) -> float:
    """Total Hamiltonian = potential(X) + kinetic(momentum)."""
    ham = potential_fn(X, model_params).item()
    kin = 0.0
    for j in range(model_params.nmat):
        Pj = mom_X[j]
        kin = kin + 0.5 * torch.trace(Pj @ Pj).real
    return ham + kin.item()


def leapfrog(X: torch.Tensor, hmc_params: HMCParams, model_params: Any, potential_fn: PotentialFn) -> tuple[torch.Tensor, float, float]:
    """Symplectic leapfrog integrator returning the proposal and initial/final energies."""
    dt_local = hmc_params.dt

    mom_X = torch.stack([random_hermitian(model_params.ncol) for _ in range(model_params.nmat)], dim=0)
    ham_init = hamil(X, mom_X, model_params, potential_fn)

    X = X + 0.5 * dt_local * mom_X

    for _ in range(1, hmc_params.nsteps):
        f_X = force(X, model_params, potential_fn)
        mom_X = mom_X - dt_local * f_X
        X = X + dt_local * mom_X

    f_X = force(X, model_params, potential_fn)
    mom_X = mom_X - dt_local * f_X
    X = X + 0.5 * dt_local * mom_X

    ham_final = hamil(X, mom_X, model_params, potential_fn)
    return X, ham_init, ham_final


def update(X: torch.Tensor, acc_count: int, hmc_params: HMCParams, model_params: Any, potential_fn: PotentialFn, reject_prob: float = 1.0):
    """Run one HMC trajectory and Metropolis accept/reject step."""
    X_bak = X.clone()
    X_new, H0, H1 = leapfrog(X, hmc_params, model_params, potential_fn)
    dH = H1 - H0

    accept = True
    if reject_prob > 0.0:
        r = random.uniform(0.0, reject_prob)
        if math.exp(-dH) < r:
            accept = False

    if accept:
        X = X_new
        acc_count += 1
        print(f"ACCEPT: dH={dH: 8.3f}, expDH={math.exp(-dH): 8.3f}, H0={H0: 8.4f}")
    else:
        X = X_bak
        print(f"REJECT: dH={dH: 8.3f}, expDH={math.exp(-dH): 8.3f}, H0={H0: 8.4f}")

    return X, acc_count


__all__ = [
    "HMCParams",
    "force",
    "hamil",
    "leapfrog",
    "update",
]
