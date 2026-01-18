"""Shared helper routines for matrix models."""

from __future__ import annotations

import torch

from MatrixModelHMC_pytorch import config
from MatrixModelHMC_pytorch.algebra import kron_2d


def _commutator_action_sum(X: torch.Tensor) -> torch.Tensor:
    nmat = X.shape[0]
    if nmat < 2:
        return X.new_zeros((), dtype=X.dtype)

    total = torch.tensor(0.0, dtype=config.real_dtype, device=X.device)
    for i in range(nmat - 1):
        for j in range(i + 1, nmat):
            comm = X[i] @ X[j] - X[j] @ X[i]
            total = total + torch.trace(comm @ comm).real
    return total.to(dtype=X.dtype)


def _anticommutator_action_sum(X: torch.Tensor) -> torch.Tensor:
    nmat = X.shape[0]
    if nmat < 2:
        return X.new_zeros((), dtype=X.dtype)

    total = torch.tensor(0.0, dtype=config.real_dtype, device=X.device)
    for i in range(nmat - 1):
        for j in range(i + 1, nmat):
            anti = X[i] @ X[j] + X[j] @ X[i]
            total = total + torch.trace(anti @ anti).real
    return total.to(dtype=X.dtype)


def _fermion_det_log_identity_plus_sum_adX(X: torch.Tensor) -> torch.Tensor:
    """Return log det(1 + \sum_i ad_{X_i}) using the eigenvalue formula."""
    sum_X2 = (X @ X).sum(dim=0)
    eigvals = torch.sqrt(torch.linalg.eigvalsh(sum_X2).real.to(dtype=config.real_dtype))
    diffs = eigvals.unsqueeze(0) - eigvals.unsqueeze(1)
    factors = diffs + 1.0
    logabs = torch.log(factors.abs())
    return logabs.sum()


def gammaMajorana() -> torch.Tensor:
    """Construct Majorana gamma matrices and their conjugate in 4D."""
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=config.dtype, device=config.device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=config.dtype, device=config.device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=config.dtype, device=config.device)
    Id2 = torch.eye(2, dtype=config.dtype, device=config.device)
    gam0 = 1j * kron_2d(sigma2, Id2)
    gam1 = kron_2d(sigma3, Id2)
    gam2 = kron_2d(sigma1, sigma3)
    gam3 = kron_2d(sigma1, sigma1)
    gam4 = 1j * gam0
    conj = gam4
    gammas = torch.stack([gam1, gam2, gam3, gam4], dim=0)
    return gammas, conj


def gammaWeyl() -> torch.Tensor:
    """Construct the Weyl-basis Dirac matrices Gamma_1..Gamma_4."""
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=config.dtype, device=config.device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=config.dtype, device=config.device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=config.dtype, device=config.device)
    Id2 = torch.eye(2, dtype=config.dtype, device=config.device)

    gamma0 = -Id2
    gammas = (sigma1, sigma2, sigma3, 1j * gamma0)
    gamma_bars = gammas[:3] + (1j * (-gamma0),)

    zero2 = torch.zeros_like(Id2)

    def block(g, gb):
        return torch.cat((torch.cat((zero2, g), dim=1), torch.cat((gb, zero2), dim=1)), dim=0)

    return torch.stack([block(g, gb) for g, gb in zip(gammas, gamma_bars)], dim=0)

