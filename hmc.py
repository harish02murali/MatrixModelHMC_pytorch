"""Core HMC kernels: potential, force, leapfrog integrator, and measurements."""

import math
import random
from dataclasses import dataclass
import torch
import numpy as np

from .algebra import makeH, random_hermitian
from .fermions import logDetM_type1, logDetM_type2


@dataclass
class SimulationParams:
    nmat: int
    ncol: int
    coupling: float
    dt: float
    nsteps: int
    omega: float
    pIKKT_type: int
    spin: float


def potential(X: torch.Tensor, params: SimulationParams) -> torch.Tensor:
    """
    X: (nmat, ncol, ncol) complex Hermitian matrices.
    For type 1:
    V = (ncol/g) * Re[sum_i Tr(X_i^2) - 0.5 sum_{i<j} Tr([X_i, X_j]^2) ] - 0.5 log det Mmaj.
    For type 2:
    V = (ncol/g) * Re[sum_(i=1..3) (2/9 + omega/3) Tr(X_i^2) + omega/3 Tr(X_4^2) + 2i (1+omega) Tr(X_1 X_2 X_3 - X_1 X_3 X_2) - 0.5 sum_{i<j} Tr([X_i, X_j]^2) ] - log det Mweyl.
    """

    s1 = 0.0 + 0.0j
    det = 0.0 + 0.0j

    for i in range(params.nmat):
        for j in range(i + 1, params.nmat):
            C = X[i] @ X[j] - X[j] @ X[i]
            s1 = s1 - 0.5 * torch.trace(C @ C)

    if params.pIKKT_type == 1:
        for i in range(params.nmat):
            s1 = s1 + torch.trace(X[i] @ X[i])
        det = -0.5 * logDetM_type1(X)[1].real

    if params.pIKKT_type == 2:
        s1 = s1 + 2j * (1 + params.omega) * (torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1]))
        for i in range(params.nmat):
            s1 = s1 + ( (2/9 if i < 3 else 0) + params.omega / 3) * torch.trace(X[i] @ X[i])
        det = -logDetM_type2(X)[1].real

    val = (s1.real * (params.ncol / params.coupling)) + det

    return val


def force(X: torch.Tensor, params: SimulationParams) -> torch.Tensor:
    Y = X.detach().requires_grad_(True)
    pot = potential(Y, params)
    pot.backward()
    return makeH(Y.grad)


def hamil(X: torch.Tensor, mom_X: torch.Tensor, params: SimulationParams) -> float:
    ham = potential(X, params).item()
    kin = 0.0
    for j in range(params.nmat):
        Pj = mom_X[j]
        kin = kin + 0.5 * torch.trace(Pj @ Pj).real
    return ham + kin.item()


def leapfrog(X: torch.Tensor, params: SimulationParams, dt_override: float | None = None) -> tuple[torch.Tensor, float, float]:
    """
    Standard leapfrog integrator for HMC.

    Returns:
        X_new, H_initial, H_final
    """
    dt_local = params.dt if dt_override is None else dt_override

    mom_X = torch.stack([random_hermitian(params.ncol) for _ in range(params.nmat)], dim=0)
    ham_init = hamil(X, mom_X, params)

    X = X + 0.5 * dt_local * mom_X

    for _ in range(1, params.nsteps):
        f_X = force(X, params)
        mom_X = mom_X - dt_local * f_X
        X = X + dt_local * mom_X

    f_X = force(X, params)
    mom_X = mom_X - dt_local * f_X
    X = X + 0.5 * dt_local * mom_X

    ham_final = hamil(X, mom_X, params)
    return X, ham_init, ham_final


def update(X: torch.Tensor, acc_count: int, params: SimulationParams, reject_prob: float = 1.0):
    """
    Single HMC trajectory (one leapfrog integration + Metropolis step).
    """
    X_bak = X.clone()
    X_new, H0, H1 = leapfrog(X, params)
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


def measure_observables(X: torch.Tensor, params: SimulationParams):
    """
    Rough analogue of the original eigenvalue/correlator measurements.

    Returns:
        eigs: list of numpy arrays of eigenvalues (all matrices and (X0 + i X1) and (X2 + i X3))
        corrs: numpy array of two correlators (or None if nmat < 2)
    """
    with torch.no_grad():
        eigs = []
        for i in range(params.nmat):
            e = torch.linalg.eigvalsh(X[i]).cpu().numpy()
            eigs.append(e)

        if params.nmat >= 4:
            e_complex = torch.linalg.eigvals((X[0] + 1j * X[1])).cpu().numpy()
            eigs.append(e_complex)
            e_complex = torch.linalg.eigvals((X[2] + 1j * X[3])).cpu().numpy()
            eigs.append(e_complex)
        
        eigs.append(torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]).cpu().numpy())

        if params.nmat >= 4:
            C = X[0] @ X[1] - X[1] @ X[0]
            A = X[0] @ X[1] + X[1] @ X[0]
            c1 = torch.trace(C @ C).real
            c2 = torch.trace(A @ A).real
            C = X[2] @ X[3] - X[3] @ X[2]
            A = X[2] @ X[3] + X[3] @ X[2]
            c3 = torch.trace(C @ C).real
            c4 = torch.trace(A @ A).real
            corrs = torch.stack([c1, c2, c3, c4]).cpu().numpy()
        else:
            corrs = None

    return eigs, corrs


__all__ = [
    "SimulationParams",
    "force",
    "hamil",
    "leapfrog",
    "measure_observables",
    "potential",
    "update",
]
