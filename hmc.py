"""Model-agnostic Hybrid Monte Carlo kernels: leapfrog and Metropolis step."""

import math
import random
from dataclasses import dataclass, replace
from typing import Any
import torch

try:
    from pIKKT4D.algebra import random_hermitian
except ImportError:  # pragma: no cover
    from algebra import random_hermitian  # type: ignore

@dataclass
class HMCParams:
    """Integrator controls for an HMC trajectory."""
    dt: float
    nsteps: int
    

def hamil(X: torch.Tensor, mom_X: torch.Tensor, model: Any) -> float:
    """Total Hamiltonian = potential(X) + kinetic(momentum)."""
    ham = model.potential(X).item()
    kin = 0.0
    for j in range(model.nmat):
        Pj = mom_X[j]
        kin = kin + 0.5 * torch.trace(Pj @ Pj).real
    return ham + kin.item()


def leapfrog(X: torch.Tensor, hmc_params: HMCParams, model: Any) -> tuple[torch.Tensor, float, float]:
    """Symplectic leapfrog integrator returning the proposal and initial/final energies."""
    dt_local = hmc_params.dt

    mom_X = torch.stack([random_hermitian(model.ncol) for _ in range(model.nmat)], dim=0)
    ham_init = hamil(X, mom_X, model)

    X = X + 0.5 * dt_local * mom_X

    for _ in range(1, hmc_params.nsteps):
        f_X = model.force(X)
        mom_X = mom_X - dt_local * f_X
        X = X + dt_local * mom_X

    f_X = model.force(X)
    mom_X = mom_X - dt_local * f_X
    X = X + 0.5 * dt_local * mom_X

    ham_final = hamil(X, mom_X, model)
    return X, ham_init, ham_final


def update(acc_count: int, hmc_params: HMCParams, model: Any, reject_prob: float = 1.0):
    """Run one HMC trajectory and Metropolis accept/reject step, mutating model.X."""
    model.refresh_aux_fields()
    X = model.get_state()
    X_bak = X.clone()
    X_new, H0, H1 = leapfrog(X, hmc_params, model)
    dH = H1 - H0

    accept = True
    if reject_prob > 0.0:
        r = random.uniform(0.0, reject_prob)
        if math.exp(-dH) < r:
            accept = False

    if accept:
        model.set_state(X_new)
        acc_count += 1
        print(f"ACCEPT: dH={dH: 8.3f}, expDH={math.exp(-dH): 8.3f}, H0={H0: 8.4f}")
    else:
        model.set_state(X_bak)
        print(f"REJECT: dH={dH: 8.3f}, expDH={math.exp(-dH): 8.3f}, H0={H0: 8.4f}")

    return acc_count


def thermalize(model: Any, hmc_params: HMCParams, steps: int = 10) -> None:
    """Run short, mostly-accepting trajectories to move the system toward equilibrium."""
    print("Thermalization steps, accept most jumps")
    therm_params = replace(hmc_params, nsteps=int(hmc_params.nsteps * 1.5), dt=hmc_params.dt / 10.0)
    acc_count = 0
    for _ in range(steps):
        acc_count = update(acc_count, therm_params, model, reject_prob=0.1)
    print("End of thermalization ", model.status_string())

__all__ = [
    "HMCParams",
    "hamil",
    "leapfrog",
    "update",
    "thermalize",
]
