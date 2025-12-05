"""Algebra utilities: Hermitian projections, commutators, and cached maps."""

import torch
from .config import device, dtype, real_dtype
from typing import Optional

# Caches keyed by (size, device, dtype) to avoid repeated allocations.
_traceless_cache: dict[tuple[int, str, Optional[int], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
_eye_cache: dict[tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}


def dagger(a: torch.Tensor) -> torch.Tensor:
    """Hermitian conjugate."""
    return a.transpose(-1, -2).conj()


def comm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix commutator [A, B]."""
    return A @ B - B @ A


def random_hermitian(n: int) -> torch.Tensor:
    """
    Draw a random traceless Hermitian n x n matrix.

    Off-diagonal entries are complex Gaussian, diagonals are real Gaussian,
    matching the spirit of the original box-muller implementation.
    """
    re = torch.randn(n, n, device=device, dtype=real_dtype)
    im = torch.randn(n, n, device=device, dtype=real_dtype)

    mat = torch.zeros(n, n, dtype=dtype, device=device)

    iu, ju = torch.triu_indices(n, n, offset=1)
    vals = (re[iu, ju] + 1j * im[iu, ju]) / (2.0**0.5)
    mat[iu, ju] = vals
    mat[ju, iu] = vals.conj()

    diag_re = torch.randn(n, device=device, dtype=real_dtype)
    idx = torch.arange(n, device=device)
    mat[idx, idx] = diag_re.to(dtype)

    mat = mat - (torch.trace(mat) / n) * torch.eye(n, dtype=dtype, device=device)
    return mat


def kron_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Kronecker product of 2D tensors A (m*n) and B (p*q):

        kron(A, B) has shape (m*p, n*q)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"kron_2d expects 2D tensors, got {A.shape}, {B.shape}")

    A = A.contiguous()
    B = B.contiguous()

    m, n = A.shape
    p, q = B.shape

    A_exp = A.unsqueeze(1).unsqueeze(3)
    B_exp = B.unsqueeze(0).unsqueeze(2)

    return (A_exp * B_exp).reshape(m * p, n * q)


def make_traceless_maps(N: int, device=None, dtype=None):
    """
    Build linear maps Q and S for the traceless subspace:

      vec(A) = Q v,   A traceless, v ∈ C^{N^2-1}
      v      = S vec(A), for traceless A.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.complex64

    N2 = N * N
    Dtr = N2 - 1

    Q = torch.zeros(N2, Dtr, device=device, dtype=dtype)
    S = torch.zeros(Dtr, N2, device=device, dtype=dtype)

    def idx(i, j):
        return i + j * N

    coords = []
    for j in range(N):
        for i in range(N):
            if not (i == N - 1 and j == N - 1):
                coords.append((i, j))
    last_diag_row = idx(N - 1, N - 1)

    for col, (i, j) in enumerate(coords):
        row = idx(i, j)

        Q[row, col] = 1.0
        S[col, row] = 1.0

        if i == j and i < N - 1:
            Q[last_diag_row, col] -= 1.0

    return Q, S


def get_traceless_maps_cached(N: int, device: torch.device, dtype: torch.dtype):
    """Cache Q,S per (N, device, dtype) to avoid rebuilding every call."""
    key = (N, device.type, device.index, dtype)
    if key not in _traceless_cache:
        _traceless_cache[key] = make_traceless_maps(N, device=device, dtype=dtype)
    return _traceless_cache[key]


def get_eye_cached(n: int, device: torch.device, dtype: torch.dtype):
    """Cache identity matrices per (n, device, dtype)."""
    key = (n, device.type, device.index, dtype)
    eye = _eye_cache.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=dtype)
        _eye_cache[key] = eye
    return eye


def ad_matrix_traceless(X: torch.Tensor) -> torch.Tensor:
    """
    Adjoint action ad_X on the traceless subspace su(N).

    X: (N, N)
    Returns:
        A_tr: (N^2 - 1, N^2 - 1) such that
              in coordinates v for traceless matrices,
              v' = A_tr v corresponds to [X, A].
    """
    if X.ndim != 2:
        raise ValueError(f"ad_matrix_traceless expects X with shape (N, N), got {X.shape}")

    N = X.shape[0]
    dev = X.device
    dtp = X.dtype

    I = get_eye_cached(N, device=dev, dtype=dtp)
    Xt = X.transpose(-1, -2)

    ad_full = kron_2d(I, X) - kron_2d(Xt, I)

    Q, S = get_traceless_maps_cached(N, device=dev, dtype=dtp)

    tmp = ad_full @ Q
    return S @ tmp


def makeH(mat: torch.Tensor) -> torch.Tensor:
    """Project a matrix (or batch of matrices) to its traceless Hermitian part."""
    tmp = 0.5 * (mat + dagger(mat))

    n = tmp.shape[-1]
    trace = tmp.diagonal(dim1=-2, dim2=-1).sum(-1).real / n

    eye = torch.eye(n, dtype=tmp.dtype, device=tmp.device)
    tmp = tmp - trace[..., None, None] * eye

    tmp = 0.5 * (tmp + dagger(tmp))
    return tmp


__all__ = [
    "ad_matrix_traceless",
    "comm",
    "dagger",
    "get_eye_cached",
    "kron_2d",
    "makeH",
    "random_hermitian",
]
