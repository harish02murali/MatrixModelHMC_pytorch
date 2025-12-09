"""Model-specific components: fermion operators, bosonic potential, and observables."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .config import ENABLE_TORCH_COMPILE, device, dtype
from .algebra import ad_matrix_traceless, get_eye_cached, kron_2d


@dataclass
class ModelParams:
    """Physical parameters for the polarized IKKT model."""
    nmat: int
    ncol: int
    coupling: float
    omega: float
    pIKKT_type: int
    source: np.ndarray | None = None


def gammaMajorana() -> torch.Tensor:
    """Construct Majorana gamma matrices and their conjugate in 4D."""
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


def logDetM_type1(X: torch.Tensor) -> torch.Tensor:
    """Log-determinant of the Majorana fermion matrix for the type-1 model."""
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
    """Log-determinant of the Weyl fermion matrix for the type-2 model."""
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


def potential(X: torch.Tensor, params: ModelParams) -> torch.Tensor:
    """
    Combined bosonic and fermionic action for the polarized IKKT model.

    Args:
        X: Tensor of shape (nmat, ncol, ncol) containing Hermitian matrices.
        params: Model parameters (coupling, omega, model type, sources).

    Returns:
        Scalar torch tensor equal to the potential energy used by HMC.
    """

    s1 = 0.0 + 0.0j
    det = 0.0 + 0.0j
    src = 0.0 + 0.0j

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
            s1 = s1 + ((2 / 9 if i < 3 else 0) + params.omega / 3) * torch.trace(X[i] @ X[i])
        det = -logDetM_type2(X)[1].real

    if params.source is not None:
        src = -(params.ncol / np.sqrt(params.coupling)) * torch.trace(
            torch.tensor(np.diag(params.source), device=X.device, dtype=X.dtype) @ X[0]
        )

    val = (s1.real * (params.ncol / params.coupling)) + det + src.real

    return val


def measure_observables(X: torch.Tensor, params: ModelParams):
    """
    Compute eigenvalue spectra and basic correlators for monitoring simulations.

    Returns:
        eigs: list of numpy arrays for each matrix and selected complex combinations.
        corrs: numpy array of commutator/anticommutator traces and the bosonic action.
    """
    with torch.no_grad():
        eigs = []
        for i in range(params.nmat):
            e = torch.linalg.eigvalsh(X[i]).cpu().numpy()
            eigs.append(e)

        e_complex = torch.linalg.eigvals((X[0] + 1j * X[1])).cpu().numpy()
        eigs.append(e_complex)
        e_complex = torch.linalg.eigvals((X[2] + 1j * X[3])).cpu().numpy()
        eigs.append(e_complex)

        eigs.append(torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]).cpu().numpy())

        C = X[0] @ X[1] - X[1] @ X[0]
        A = X[0] @ X[1] + X[1] @ X[0]
        c1 = torch.trace(C @ C).real
        c2 = torch.trace(A @ A).real
        C = X[2] @ X[3] - X[3] @ X[2]
        A = X[2] @ X[3] + X[3] @ X[2]
        c3 = torch.trace(C @ C).real
        c4 = torch.trace(A @ A).real

        s1 = 0.0 + 0.0j

        for i in range(params.nmat):
            for j in range(i + 1, params.nmat):
                C = X[i] @ X[j] - X[j] @ X[i]
                s1 = s1 - 0.5 * torch.trace(C @ C)

        if params.pIKKT_type == 1:
            for i in range(params.nmat):
                s1 = s1 + torch.trace(X[i] @ X[i])

        if params.pIKKT_type == 2:
            s1 = s1 + 2j * (1 + params.omega) * (torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1]))
            for i in range(params.nmat):
                s1 = s1 + ((2 / 9 if i < 3 else 0) + params.omega / 3) * torch.trace(X[i] @ X[i])

        action = s1.real * (params.ncol / params.coupling)

        corrs = torch.stack([c1, c2, c3, c4, action]).cpu().numpy()

    return eigs, corrs


def spinJMatrices(j_val: float):
    """Generate spin-j angular momentum matrices Jx, Jy, Jz on CPU with NumPy."""
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


__all__ = [
    "ModelParams",
    "gammaMajorana",
    "gammaWeyl",
    "logDetM_type1",
    "logDetM_type2",
    "measure_observables",
    "potential",
    "spinJMatrices",
]
