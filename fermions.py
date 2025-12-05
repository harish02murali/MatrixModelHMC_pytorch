"""Fermionic determinant helpers."""

import torch
from pIKKT4D.config import ENABLE_TORCH_COMPILE, device, dtype  # type: ignore
from pIKKT4D.algebra import ad_matrix_traceless, get_eye_cached, kron_2d  # type: ignore


def gammaMajorana() -> torch.Tensor:
    """Construct the 4D gamma^5 matrix in the chosen dtype and device."""
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    Id2 = torch.eye(2, dtype=dtype, device=device)
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
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    Id2 = torch.eye(2, dtype=dtype, device=device)

    gamma0 = -Id2
    gammas = (sigma1, sigma2, sigma3, 1j * gamma0)
    gamma_bars = gammas[:3] + (1j * (-gamma0),)

    zero2 = torch.zeros_like(Id2)

    def block(g, gb):
        return torch.cat(
            (torch.cat((zero2, g), dim=1), torch.cat((gb, zero2), dim=1)), dim=0
        )

    return torch.stack([block(g, gb) for g, gb in zip(gammas, gamma_bars)], dim=0)

gammasM, conj = gammaMajorana()
gammasW = gammaWeyl()

def logDetM_type1(X: torch.Tensor) -> torch.Tensor:
    """
    Compute log det of the 4(n^2-1)x4(n^2-1) operator S using the
    2(n^2-1)x2(n^2-1) matrix.
    """
    if X.shape[0] != 4:
        raise ValueError("X must have shape (4, N, N): four matrices X_1..X_4")

    _, N, _ = X.shape
    dev = X.device
    dtp = X.dtype
    omega = 1.0

    dim_tr = N * N - 1

    adX1 = 1j * ad_matrix_traceless(X[0])
    adX2 = 1j * ad_matrix_traceless(X[1])
    adX3 = 1j * ad_matrix_traceless(X[2])
    adX4 = 1j * ad_matrix_traceless(X[3])

    eye_tr = get_eye_cached(dim_tr, device=dev, dtype=dtp)
    i = 1j

    two_i_omega_eye = (2 * i * omega) * eye_tr

    A = torch.zeros((2 * dim_tr, 2 * dim_tr), device=dev, dtype=dtp)
    A[:dim_tr, dim_tr:] = two_i_omega_eye
    A[dim_tr:, :dim_tr] = -two_i_omega_eye

    B = torch.empty_like(A)
    B[:dim_tr, :dim_tr] = -(adX3 + i * adX4)
    B[:dim_tr, dim_tr:] = -(adX1 - i * adX2)
    B[dim_tr:, :dim_tr] = -(adX1 + i * adX2)
    B[dim_tr:, dim_tr:] = adX3 - i * adX4

    C = torch.empty_like(A)
    C[:dim_tr, :dim_tr] = -(adX3 + i * adX4)
    C[:dim_tr, dim_tr:] = -(adX1 + i * adX2)
    C[dim_tr:, :dim_tr] = -(adX1 - i * adX2)
    C[dim_tr:, dim_tr:] = adX3 - i * adX4

    Ainv = A / (4 * omega**2)
    K = -A - C @ Ainv @ B

    det = torch.linalg.slogdet(K)
    if (det[0].abs() - 1) > 1e-4:
        raise ValueError("Fermion matrix determinant is non-positive")
    return det

def logDetM_type2(X: torch.Tensor) -> torch.Tensor:
    """
    Compute log det of the 4(n^2-1)x4(n^2-1) operator S using the
    2(n^2-1)x2(n^2-1) matrix.
    """
    if X.shape[0] != 4:
        raise ValueError("X must have shape (4, N, N): four matrices X_1..X_4")

    _, N, _ = X.shape
    dev = X.device
    dtp = X.dtype
    omega = 1.0

    dim_tr = N * N - 1

    adX1 = 1j * ad_matrix_traceless(X[0])
    adX2 = 1j * ad_matrix_traceless(X[1])
    adX3 = 1j * ad_matrix_traceless(X[2])
    adX4 = 1j * ad_matrix_traceless(X[3])

    eye_2tr = get_eye_cached(2 * dim_tr, device=dev, dtype=dtp)
    i = 1j

    A = torch.zeros((2 * dim_tr, 2 * dim_tr), device=dev, dtype=dtp)
    A[:dim_tr, :dim_tr] = -adX4 + i * adX3
    A[:dim_tr, dim_tr:] = adX2 + i * adX1
    A[dim_tr:, :dim_tr] = -adX2 + i * adX1
    A[dim_tr:, dim_tr:] = -adX4 - i * adX3

    K = A - 2 / 3 * omega * eye_2tr 
    det = torch.linalg.slogdet(K)
    if (det[0].abs() - 1) > 1e-4:
        raise ValueError("Fermion matrix determinant is non-positive")
    return det


if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
    logDetM_type1 = torch.compile(logDetM_type1, dynamic=False)
    logDetM_type2 = torch.compile(logDetM_type2, dynamic=False)


__all__ = ["logDetM_type1", "logDetM_type2"]