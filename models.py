"""Model-specific components: fermion operators, bosonic potential, and observables."""

from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import torch

from . import config
from .algebra import ad_matrix, get_eye_cached, get_trace_projector_cached, kron_2d, spinJMatrices, makeH, random_hermitian

ENABLE_TORCH_COMPILE = config.ENABLE_TORCH_COMPILE


def build_model(args: Namespace) -> MatrixModel:
    """Model picker that maps CLI arguments to a fully-initialized model."""
    model_name = getattr(args, "model").lower()
    if model_name in ("1mm", "one_matrix"):
        return OneMatrixPolynomialModel(
            ncol=args.ncol,
            couplings=args.coupling,
        )
    if model_name == "pikkt4d_type1":
        return PIKKTTypeIModel(
            ncol=args.ncol,
            couplings=args.coupling,
            source=args.source,
        )
    if model_name == "pikkt4d_type2":
        return PIKKTTypeIIModel(
            ncol=args.ncol,
            couplings=args.coupling,
            source=args.source,
            no_myers=args.no_myers,
        )
    if model_name == "yangmills":
        return YangMillsModel(
            dim=args.nmat,
            ncol=args.ncol,
            couplings=args.coupling,
            source=args.source,
        )
    if model_name == "adjoint_det":
        if args.nmat is None:
            raise ValueError("--nmat must be provided for adjoint_det model")
        return AdjointDetModel(
            dim=args.nmat,
            ncol=args.ncol,
            couplings=args.coupling,
            source=args.source,
        )

    raise ValueError(f"Unknown model '{args.model}'")


class MatrixModel:
    """Base class for matrix models used with the HMC driver."""

    def __init__(self, name: str, nmat: int, ncol: int) -> None:
        self.name = name
        self.nmat = nmat
        self.ncol = ncol
        self.couplings = None
        self.is_hermitian = None
        self.is_traceless = None
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
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def initialize_configuration(self, args: Namespace, ckpt_path: str) -> bool:
        if args.resume:
            if os.path.isfile(ckpt_path):
                print("Reading old configuration file:", ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
                self.set_state(ckpt["X"].to(dtype=config.dtype, device=config.device))
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

    def run_metadata(self) -> dict[str, object]:
        return {
            "model_key": getattr(self, "model_name", self.__class__.__name__.lower()),
            "display_name": self.name,
            "nmat": self.nmat,
            "ncol": self.ncol,
            "couplings": self.couplings,
            "dtype": str(config.dtype),
        }


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


class OneMatrixPolynomialModel(MatrixModel):
    """Single-matrix polynomial model V(X) = sum_n t_n Tr(X^n)."""

    model_name = "1mm"

    def __init__(self, ncol: int, couplings: list) -> None:
        if len(couplings) == 0:
            raise ValueError("1MM model requires at least one coupling via --coupling t1 [t2 ...]")
        super().__init__(name="1MM Polynomial", nmat=1, ncol=ncol)
        self.couplings = couplings
        self.model_key = "1mm"
        self.is_hermitian = True
        self.is_traceless = False
        self._coupling_tensor = torch.tensor(
            couplings, dtype=config.real_dtype, device=config.device
        )

    def load_fresh(self, args: Namespace) -> None:  # type: ignore[override]
        # X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        X = 2*torch.eye(self.ncol, dtype=config.dtype, device=config.device).unsqueeze(0)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        mat = X[0]
        total = torch.zeros((), dtype=config.real_dtype, device=mat.device)
        for power, coeff in enumerate(self._coupling_tensor, start=1):
            if coeff == 0:
                continue
            trace_power = torch.trace(torch.linalg.matrix_power(mat, power)).real
            total = total + coeff.type_as(trace_power) * trace_power
        return self.ncol * total

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            mat = self._resolve_X(X)[0]
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy()]
            trace_powers = []
            for power in range(1, len(self.couplings) + 1):
                trace_val = torch.trace(torch.linalg.matrix_power(mat, power)).cpu().numpy()
                trace_powers.append(trace_val)
            corrs = np.array(trace_powers, dtype=np.complex128)
        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        coupling_suffix = "_".join(f"t{idx + 1}-{float(c):g}" for idx, c in enumerate(self.couplings))
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_N{self.ncol}_{coupling_suffix}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        couplings_str = ", ".join(f"t_{i+1}={c}" for i, c in enumerate(self.couplings))
        return [f"  Couplings t_n            = {couplings_str}"]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        mat = self._resolve_X(X)[0]
        trX2 = (torch.trace(mat @ mat).real / self.ncol).item()
        return f"trX^2 = {trX2:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "polynomial_degree": len(self.couplings),
                "model_variant": "1mm_polynomial",
            }
        )
        return meta


class PIKKTTypeIModel(MatrixModel):
    """Type I polarized IKKT model definition."""

    model_name = "pikkt4d_type1"

    def __init__(self, ncol: int, couplings: list, source: np.ndarray | None = None, no_myers: bool = False) -> None:
        super().__init__(name="pIKKT Type I", nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype)) if source is not None else None
        self.no_myers = no_myers
        self.is_hermitian = True
        self.is_traceless = True
        self.model_key = "type1"

        # Caching for performance optimization
        dim_tr = self.ncol * self.ncol
        eye_tr = get_eye_cached(dim_tr, device=config.device, dtype=config.dtype)
        i = 1j
        two_i_I = (2 * i) * eye_tr
        A = torch.zeros((2 * dim_tr, 2 * dim_tr), device=config.device, dtype=config.dtype)
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
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        # for i in range(self.nmat):
        #     X[i] = random_hermitian(self.ncol)
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
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
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
        return [f"  Coupling g               = {self.g}"]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "type1",
            }
        )
        return meta


class PIKKTTypeIIModel(MatrixModel):
    """Type II polarized IKKT model definition."""

    model_name = "pikkt4d_type2"

    def __init__(self, ncol: int, couplings: list, source: np.ndarray | None = None, no_myers: bool = False) -> None:
        super().__init__(name="pIKKT Type II", nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = self.couplings[1]
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype)) if source is not None else None
        self.no_myers = no_myers
        self.is_hermitian = True
        self.is_traceless = True
        self.model_key = "type2"

        # Caching for performance optimization
        dim_tr = self.ncol * self.ncol
        omega_eye = (2 / 3 * self.omega) * get_eye_cached(2 * dim_tr, device=config.device, dtype=config.dtype)
        base_fn = lambda X: _type2_logdet_impl(X, omega_eye)
        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False)
        else:
            self._log_det_fn = base_fn

    def load_fresh(self, args):
        mats = [makeH(random_hermitian(self.ncol)) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)

        if args.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(args.spin)).to(dtype=config.dtype, device=config.device)
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=config.dtype, device=config.device)
            for i in range(3):
                X[i][:ntimes * J_matrices.shape[1], :ntimes * J_matrices.shape[1]] = (2 / 3 + self.omega) * torch.kron(eye_nt, J_matrices[i])
            X[3] = torch.zeros_like(X[3])
        
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
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return [f"  Coupling g               = {self.g}", f"  Coupling Omega2/Omega1      = {self.omega}"]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        casimir = (torch.trace(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]) / self.ncol).item().real
        trX4 = (torch.trace(X[3] @ X[3]) / self.ncol).item().real
        return f"casimir = {casimir:.5f}, trX_4^2 = {trX4:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "no_myers": self.no_myers,
                "has_source": self.source is not None,
                "model_variant": "type2",
            }
        )
        return meta


class AdjointDetModel(MatrixModel):
    """Matrix model with product fermion determinant det(1 + \sum_i ad X_i)."""

    model_name = "adjoint_det"

    def __init__(self, dim: int, ncol: int, couplings: list, source: np.ndarray | None = None) -> None:
        super().__init__(name=f"{dim}D AdjointDet", nmat=dim, ncol=ncol)
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype)) if source is not None else None
        self.couplings = couplings
        self.g = self.couplings[0]
        self.is_hermitian = True
        self.is_traceless = True

        def base_fn(X: torch.Tensor, *, model=self) -> torch.Tensor:
            return _fermion_det_log_identity_plus_sum_adX(X)
        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False)
        else:
            self._log_det_fn = base_fn

    def load_fresh(self, args: Namespace) -> None:  # type: ignore[override]
        # mats = [makeH(random_hermitian(self.ncol)) for _ in range(self.nmat)]
        # X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        trace_sq = torch.einsum("bij,bji->", X, X).real
        comm_term = -0.5 * _commutator_action_sum(X).real
        bos = trace_sq + comm_term

        det_coeff = torch.tensor((self.nmat - 2), dtype=config.real_dtype, device=X.device)
        det = -det_coeff * _fermion_det_log_identity_plus_sum_adX(X)

        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])

        return (self.ncol / self.g) * bos + det + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X] + [torch.linalg.eigvals(X[0] + 1j * X[1]).cpu().numpy()]
            comm_raw = _commutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
            anticomm_raw = _anticommutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
            
            # Moments tr(X_i X_j) to diagnose emergent rotational symmetry.
            moments = torch.einsum("aij,bji->ab", X, X).real
            corrs = np.concatenate(
                (
                    np.array([anticomm_raw, comm_raw], dtype=np.float64),
                    moments.detach().cpu().numpy().astype(np.float64).reshape(-1),
                )
            )
        return eigs, corrs

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_D{self.nmat}_g{round(self.g, 4)}_N{self.ncol}"
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return [f"  Coupling g               = {self.g}", f"  Dimension D             = {self.nmat}"]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        avg_tr = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"trX_i^2 = {avg_tr:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "adjoint_det",
            }
        )
        return meta


class YangMillsModel(MatrixModel):
    """D-dimensional Yang-Mills matrix model."""

    model_name = "yangmills"

    def __init__(self, dim: int, ncol: int, couplings: list, source: np.ndarray | None = None) -> None:
        super().__init__(name=f"{dim}D Yang-Mills", nmat=dim, ncol=ncol)
        self.source = torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype)) if source is not None else None
        self.couplings = couplings
        self.is_hermitian = True
        self.is_traceless = True
        self.g = self.couplings[0]

    def load_fresh(self, args: Namespace) -> None:  # type: ignore[override]
        mats = [makeH(random_hermitian(self.ncol)) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)
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
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        avg_tr = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"trX_i^2 = {avg_tr:.5f}. "

    def extra_config_lines(self) -> list[str]:
        return [f"Coupling g               = {self.g}", f"Dimension D             = {self.nmat}"]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "yangmills",
            }
        )
        return meta

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
        return torch.cat(
            (torch.cat((zero2, g), dim=1), torch.cat((gb, zero2), dim=1)), dim=0
        )

    return torch.stack([block(g, gb) for g, gb in zip(gammas, gamma_bars)], dim=0)


__all__ = [
    "MatrixModel",
    "OneMatrixPolynomialModel",
    "PIKKTTypeIModel",
    "PIKKTTypeIIModel",
    "YangMillsModel",
    "build_model",
    "gammaMajorana",
    "gammaWeyl",
]
