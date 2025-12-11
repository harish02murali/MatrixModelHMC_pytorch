"""Model-specific components: fermion operators, bosonic potential, and observables."""

from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import torch

from . import config
from .algebra import ad_matrix, get_eye_cached, get_trace_projector_cached, kron_2d, spinJMatrices, makeH, random_hermitian

ENABLE_TORCH_COMPILE = config.ENABLE_TORCH_COMPILE
dtype = config.dtype


def build_model(args: Namespace) -> MatrixModel:
    """Model picker that maps CLI arguments to a fully-initialized model."""
    model_name = getattr(args, "model").lower()
    if model_name == "pikkt4d_type1":
        return PIKKTTypeIModel(
            ncol=args.ncol,
            g=args.coupling[0],
            source=args.source,
        )
    if model_name == "pikkt4d_type2":
        return PIKKTTypeIIModel(
            ncol=args.ncol,
            g=args.coupling[0],
            omega=args.coupling[1],
            source=args.source,
            no_myers=args.no_myers,
        )
    if model_name == "yangmills":
        return YangMillsModel(
            dim=args.nmat,
            ncol=args.ncol,
            g=args.coupling[0],
            source=args.source,
        )

    raise ValueError(f"Unknown model '{args.model}'")


class MatrixModel:
    """Base class for matrix models used with the HMC driver."""

    def __init__(self, name: str, nmat: int, ncol: int) -> None:
        self.name = name
        self.nmat = nmat
        self.ncol = ncol
        self._X: torch.Tensor | None = None

    def _resolve_X(self, X: torch.Tensor | None = None) -> torch.Tensor:
        if X is not None:
            return X
        if self._X is None:
            raise ValueError("Model configuration has not been initialized")
        return self._X

    def set_state(self, X: torch.Tensor) -> None:
        self._X = X

    def get_state(self) -> torch.Tensor:
        return self._resolve_X()
    
    def load_fresh(self, args: Namespace) -> None:
        """Load a fresh configuration (zero matrices)."""
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=dtype, device=config.device)
        self.set_state(X)

    def initialize_configuration(self, args: Namespace, ckpt_path: str) -> bool:
        if args.resume:
            if os.path.isfile(ckpt_path):
                print("Reading old configuration file:", ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=config.device)
                self.set_state(ckpt["X"].to(dtype=dtype, device=config.device))
                return True
            else:
                print("Configuration not found, loading fresh")
        else:
            print("Loading fresh configuration")
        
        self.load_fresh(args)

        return False

    def save_state(self, ckpt_path: str) -> None:
        torch.save({"X": self.get_state()}, ckpt_path)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def measure_observables(self, X: torch.Tensor | None = None):
        raise NotImplementedError

    def status_string(self, X: torch.Tensor | None = None) -> str:
        raise NotImplementedError

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        raise NotImplementedError

    def extra_config_lines(self) -> list[str]:
        """Optional human-readable configuration lines for logging."""
        return []


def _type1_logdet_impl(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    adX = 1j * ad_matrix(X[:4])
    adX1, adX2, adX3, adX4 = adX
    i = torch.tensor(1j, dtype=X.dtype, device=X.device)
    twoi = torch.tensor(2j, dtype=X.dtype, device=X.device)

    upper_left = -(adX3 + i * adX4)
    upper_right = -(adX1 - i * adX2)
    lower_left = -(adX1 + i * adX2)
    lower_right = adX3 - i * adX4

    C = torch.cat(
        (torch.cat((upper_left, lower_left), dim=1), torch.cat((upper_right, lower_right), dim=1)),
        dim=0,
    )

    AB = torch.cat(
        (
            torch.cat((twoi * lower_left, twoi * lower_right), dim=1),
            torch.cat((-twoi * upper_left, -twoi * upper_right), dim=1),
        ),
        dim=0,
    )

    K = -A - 0.25 * (C @ AB)
    
    # Lift zero modes from the trace sector (identity direction)
    # This adds a constant mass term to the trace mode, ensuring invertibility
    # without affecting the physics (since it's a constant factor in det).
    N = X.shape[-1]
    dim = N * N
    P = get_trace_projector_cached(N, K.device, K.dtype)
    K[:dim, :dim] += P
    K[dim:, dim:] += P

    det = torch.linalg.slogdet(K)
    if (det[0].abs() - 1) > 1e-4:
        raise ValueError("Fermion matrix determinant is non-positive")
    return det


def _type2_logdet_impl(X: torch.Tensor, omega_eye: torch.Tensor) -> torch.Tensor:
    adX = 1j * ad_matrix(X[:4])
    adX1, adX2, adX3, adX4 = adX
    i = 1j

    upper_left = -adX4 + i * adX3
    upper_right = adX2 + i * adX1
    lower_left = -adX2 + i * adX1
    lower_right = -adX4 - i * adX3

    top = torch.cat((upper_left, upper_right), dim=1)
    bottom = torch.cat((lower_left, lower_right), dim=1)
    A = torch.cat((top, bottom), dim=0)

    K = A - omega_eye

    # Lift zero modes from the trace sector (identity direction)
    # Essential when omega=0 to avoid singular matrix.
    N = X.shape[-1]
    dim = N * N
    P = get_trace_projector_cached(N, K.device, K.dtype)
    K[:dim, :dim] += P
    K[dim:, dim:] += P

    det = torch.linalg.slogdet(K)
    if (det[0].abs() - 1) > 1e-4:
        raise ValueError("Fermion matrix determinant is non-positive")
    return det


def _commutator_action_sum(X: torch.Tensor) -> torch.Tensor:
    nmat = X.shape[0]
    if nmat < 2:
        return X.new_zeros((), dtype=X.dtype)
    comms = []
    for i in range(nmat - 1):
        Xi = X[i]
        for j in range(i + 1, nmat):
            comms.append(Xi @ X[j] - X[j] @ Xi)
    comm_stack = torch.stack(comms)
    comm_sq = comm_stack @ comm_stack
    traces = torch.real(torch.einsum("bii->b", comm_sq))
    return (traces.sum()).to(dtype=X.dtype)


class PIKKTTypeIModel(MatrixModel):
    """Type I polarized IKKT model definition."""

    model_name = "pikkt4d_type1"

    def __init__(self, ncol: int, g: float, omega: float = 0.0, source: np.ndarray | None = None, no_myers: bool = False) -> None:
        super().__init__(name="pIKKT Type I", nmat=4, ncol=ncol)
        self.g = g
        self.omega = omega
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=dtype)) if source is not None else None
        self.no_myers = no_myers
        self.model_key = "type1"

        # Caching for performance optimization
        dim_tr = self.ncol * self.ncol
        eye_tr = get_eye_cached(dim_tr, device=config.device, dtype=dtype)
        i = 1j
        two_i_I = (2 * i) * eye_tr
        A = torch.zeros((2 * dim_tr, 2 * dim_tr), device=config.device, dtype=dtype)
        A[:dim_tr, dim_tr:] = two_i_I
        A[dim_tr:, :dim_tr] = -two_i_I
        self._type1_A = A.clone()
        # self._type1_Ainv = self._type1_A / 4.0

        def base_fn(X: torch.Tensor, *, model=self) -> torch.Tensor:
            return _type1_logdet_impl(X, model._type1_A)
        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False)
        else:
            self._log_det_fn = base_fn
    
    def load_fresh(self, args):
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=dtype, device=config.device)
        for i in range(self.nmat):
            X[i] = random_hermitian(self.ncol)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        bos = -0.5 * _commutator_action_sum(X)
        trace_sq = torch.einsum("bij,bji->", X, X)
        bos = bos + trace_sq

        det = -0.5 * self._log_det_fn(X)[1].real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])

        return (bos.real * (self.ncol / self.g)) + det + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        return _measure_pikkt_observables(self._resolve_X(X), self)

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        omega_suffix = f"_omega{round(self.omega, 4)}" if hasattr(self, "omega") else ""
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}{omega_suffix}_N{self.ncol}"
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.dat"),
            "corrs": os.path.join(run_dir, "corrs.dat"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return []

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        trX1 = (torch.trace(X[0] @ X[0]) / self.ncol).item().real
        trX4 = (torch.trace(X[3] @ X[3]) / self.ncol).item().real
        return f"trX_1^2 = {trX1:.5f}, trX_4^2 = {trX4:.5f}. "
    
    def extra_config_lines(self) -> list[str]:
        return [f"Coupling g               = {self.g}"]


class PIKKTTypeIIModel(MatrixModel):
    """Type II polarized IKKT model definition."""

    model_name = "pikkt4d_type2"

    def __init__(self, ncol: int, g: float, omega: float, source: np.ndarray | None = None, no_myers: bool = False) -> None:
        super().__init__(name="pIKKT Type II", nmat=4, ncol=ncol)
        self.g = g
        self.omega = omega
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=dtype)) if source is not None else None
        self.no_myers = no_myers
        self.model_key = "type2"

        # Caching for performance optimization
        dim_tr = self.ncol * self.ncol
        omega_eye = (2 / 3 * self.omega) * get_eye_cached(2 * dim_tr, device=config.device, dtype=dtype)
        base_fn = lambda X: _type2_logdet_impl(X, omega_eye)
        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False)
        else:
            self._log_det_fn = base_fn

    def load_fresh(self, args):
        mats = [makeH(random_hermitian(self.ncol)) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=dtype, device=config.device)

        if args.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(args.spin)).to(dtype=dtype, device=config.device)
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=dtype, device=config.device)
            for i in range(3):
                X[i] = (2 / 3 + self.omega) * torch.kron(eye_nt, J_matrices[i])
        
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        bos = -0.5 * _commutator_action_sum(X)
        if not self.no_myers:
            bos = bos + 2j * (1 + self.omega) * (torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1]))
        trace_sq = torch.einsum("bij,bji->b", X, X)
        coeffs = torch.full((self.nmat,), self.omega / 3, dtype=X.dtype, device=X.device)
        extra = torch.tensor(2 / 9, dtype=X.dtype, device=X.device)
        upto = min(3, self.nmat)
        coeffs[:upto] = coeffs[:upto] + extra
        bos = bos + torch.dot(coeffs, trace_sq)

        det = -self._log_det_fn(X)[1].real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])
        return (bos.real * (self.ncol / self.g)) + det + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        return _measure_pikkt_observables(self._resolve_X(X), self)

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}_omega{round(self.omega, 4)}_N{self.ncol}"
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.dat"),
            "corrs": os.path.join(run_dir, "corrs.dat"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return [f"Coupling g               = {self.g}", f"Coupling Omega2/Omega1      = {self.omega}"]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        casimir = (torch.trace(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]) / self.ncol).item().real
        trX4 = (torch.trace(X[3] @ X[3]) / self.ncol).item().real
        return f"casimir = {casimir:.5f}, trX_4^2 = {trX4:.5f}. "


class YangMillsModel(MatrixModel):
    """D-dimensional Yang-Mills matrix model."""

    model_name = "yangmills"

    def __init__(self, dim: int, ncol: int, g: float, source: np.ndarray | None = None) -> None:
        super().__init__(name=f"{dim}D Yang-Mills", nmat=dim, ncol=ncol)
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=dtype)) if source is not None else None
        self.g = g

    def load_fresh(self, args: Namespace) -> None:  # type: ignore[override]
        mats = [makeH(random_hermitian(self.ncol)) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=dtype, device=config.device)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        trace_sq = torch.einsum("bij,bji->", X, X).real
        mass_term = trace_sq
        comm_term = -0.5 * _commutator_action_sum(X).real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])
        return (self.ncol / self.g) * (mass_term + comm_term) + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        X = self._resolve_X(X)
        eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
        trace_sq = (torch.einsum("bij,bji->", X, X).real * self.ncol).item()
        comm_raw = _commutator_action_sum(X).real.item()
        corrs = np.array([trace_sq, comm_raw], dtype=np.float64)
        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_D{self.nmat}_g{round(self.g, 4)}_N{self.ncol}"
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.dat"),
            "corrs": os.path.join(run_dir, "corrs.dat"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        avg_tr = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"trX_i^2 = {avg_tr:.5f}. "

    def extra_config_lines(self) -> list[str]:
        return [f"Coupling g               = {self.g}", f"Dimension D             = {self.nmat}"]

if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
    PIKKTTypeIModel._log_det_m = staticmethod(
        torch.compile(PIKKTTypeIModel._log_det_m.__func__, dynamic=False)
    )
    PIKKTTypeIIModel._log_det_m = staticmethod(
        torch.compile(PIKKTTypeIIModel._log_det_m.__func__, dynamic=False)
    )


################################################# HELPER FUNCTIONS ####################################################


def _measure_pikkt_observables(X: torch.Tensor, model: MatrixModel):
    with torch.no_grad():
        eigs = []
        for i in range(model.nmat):
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

        corrs = torch.stack([c1, c2, c3, c4]).cpu().numpy()

    return eigs, corrs


def gammaMajorana() -> torch.Tensor:
    """Construct Majorana gamma matrices and their conjugate in 4D."""
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=config.device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=config.device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=config.device)
    Id2 = torch.eye(2, dtype=dtype, device=config.device)
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
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=config.device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=config.device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=config.device)
    Id2 = torch.eye(2, dtype=dtype, device=config.device)

    gamma0 = -Id2
    gammas = (sigma1, sigma2, sigma3, 1j * gamma0)
    gamma_bars = gammas[:3] + (1j * (-gamma0),)

    zero2 = torch.zeros_like(Id2)

    def block(g, gb):
        return torch.cat(
            (torch.cat((zero2, g), dim=1), torch.cat((gb, zero2), dim=1)), dim=0
        )

    return torch.stack([block(g, gb) for g, gb in zip(gammas, gamma_bars)], dim=0)


__all__ = [
    "MatrixModel",
    "PIKKTTypeIModel",
    "PIKKTTypeIIModel",
    "YangMillsModel",
    "build_model",
    "gammaMajorana",
    "gammaWeyl",
]
