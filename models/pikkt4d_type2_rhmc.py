"""Type II polarized IKKT model with RHMC pseudofermions."""

from __future__ import annotations

import os

import numpy as np
import torch

from pIKKT4D import config
from pIKKT4D.algebra import get_eye_cached, random_hermitian, spinJMatrices
from pIKKT4D.models.base import MatrixModel
from pIKKT4D.models.utils import _commutator_action_sum

ENABLE_TORCH_COMPILE = config.ENABLE_TORCH_COMPILE


def _rhmc_shifts(lam_min: float, lam_max: float, degree: int) -> np.ndarray:
    ratio = lam_max / lam_min
    exponents = (np.arange(degree, dtype=np.float64) + 0.5) / degree
    return lam_min * ratio ** exponents


def _fit_rational_coeffs(
    power: float,
    lam_min: float,
    lam_max: float,
    shifts: np.ndarray,
    samples: int,
) -> tuple[float, np.ndarray]:
    xs = np.logspace(np.log10(lam_min), np.log10(lam_max), samples)
    A = np.empty((samples, shifts.size + 1), dtype=np.float64)
    A[:, 0] = 1.0
    for idx, shift in enumerate(shifts):
        A[:, idx + 1] = 1.0 / (xs + shift)
    b = xs**power
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coeffs[0]), coeffs[1:]


def _complex_gaussian(shape: tuple[int, ...]) -> torch.Tensor:
    real = torch.randn(shape, device=config.device, dtype=config.real_dtype)
    imag = torch.randn(shape, device=config.device, dtype=config.real_dtype)
    return ((real + 1j * imag) / (2**0.5)).to(dtype=config.dtype)


def _inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.conj() * b).sum()


class PIKKTTypeIIRHMCModel(MatrixModel):
    """Type II polarized IKKT model with RHMC pseudofermions."""

    model_name = "pikkt4d_type2_rhmc"

    def __init__(
        self,
        ncol: int,
        couplings: list,
        source: np.ndarray | None = None,
        no_myers: bool = False,
        rhmc_lambda_min: float | None = None,
        rhmc_lambda_max: float | None = None,
        rhmc_degree: int = 8,
        rhmc_samples: int = 200,
    ) -> None:
        super().__init__(name="pIKKT Type II (RHMC)", nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = self.couplings[1]
        self.source = (
            torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype))
            if source is not None
            else None
        )
        self.no_myers = no_myers
        self.is_hermitian = True
        self.is_traceless = True
        self.model_key = "type2_rhmc"

        if rhmc_lambda_min is None or rhmc_lambda_max is None:
            raise ValueError("RHMC bounds must be provided via --rhmc-lambda-min/--rhmc-lambda-max")
        self.rhmc_lambda_min = rhmc_lambda_min
        self.rhmc_lambda_max = rhmc_lambda_max
        self.rhmc_degree = rhmc_degree
        self.rhmc_samples = rhmc_samples

        self._omega_coeff = torch.tensor(2.0 / 3.0 * self.omega, device=config.device, dtype=config.real_dtype)
        self._eye_n = get_eye_cached(self.ncol, device=config.device, dtype=config.dtype)

        shifts = _rhmc_shifts(self.rhmc_lambda_min, self.rhmc_lambda_max, self.rhmc_degree)
        c0_inv, coeff_inv = _fit_rational_coeffs(
            -0.5, self.rhmc_lambda_min, self.rhmc_lambda_max, shifts, self.rhmc_samples
        )
        c0_quarter, coeff_quarter = _fit_rational_coeffs(
            0.25, self.rhmc_lambda_min, self.rhmc_lambda_max, shifts, self.rhmc_samples
        )
        self._rhmc_shifts = torch.tensor(shifts, device=config.device, dtype=config.real_dtype)
        self._rhmc_c0_inv = torch.tensor(c0_inv, device=config.device, dtype=config.real_dtype)
        self._rhmc_c_inv = torch.tensor(coeff_inv, device=config.device, dtype=config.real_dtype)
        self._rhmc_c0_quarter = torch.tensor(c0_quarter, device=config.device, dtype=config.real_dtype)
        self._rhmc_c_quarter = torch.tensor(coeff_quarter, device=config.device, dtype=config.real_dtype)

        self._phi: torch.Tensor | None = None
        self._cg_tol = 1e-8
        self._cg_maxiter = 500

        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._apply_ktk_compiled = torch.compile(self._apply_ktk, dynamic=False)
        else:
            self._apply_ktk_compiled = self._apply_ktk

    def load_fresh(self, args):
        mats = [random_hermitian(self.ncol) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)

        if args.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(args.spin)).to(
                dtype=config.dtype, device=config.device
            )
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=config.dtype, device=config.device)
            for i in range(3):
                X[i][: ntimes * J_matrices.shape[1], : ntimes * J_matrices.shape[1]] = (
                    (2 / 3 + self.omega) * torch.kron(eye_nt, J_matrices[i])
                )
            X[3] = torch.zeros_like(X[3])

        self.set_state(X)

    def refresh_aux_fields(self) -> None:
        with torch.no_grad():
            X = self.get_state()
            eta = _complex_gaussian((2, self.ncol, self.ncol))
            self._phi = self._apply_rational(X, eta, self._rhmc_c0_quarter, self._rhmc_c_quarter)

    def _apply_k(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Vectorized commutator calculation
        # X: (4, N, N), v: (2, N, N)
        # C[mu, a] = [X_mu, v_a]
        X_exp = X.unsqueeze(1)  # (4, 1, N, N)
        v_exp = v.unsqueeze(0)  # (1, 2, N, N)
        C = X_exp @ v_exp - v_exp @ X_exp  # (4, 2, N, N)

        # C[0,:] -> X1, C[1,:] -> X2, C[2,:] -> X3, C[3,:] -> X4
        # v[0] -> v1, v[1] -> v2

        # top = -ad(X4, v1) + 1j * ad(X3, v1) + ad(X2, v2) + 1j * ad(X1, v2)
        top = -C[3, 0] + 1j * C[2, 0] + C[1, 1] + 1j * C[0, 1]

        # bottom = -ad(X2, v1) + 1j * ad(X1, v1) - ad(X4, v2) - 1j * ad(X3, v2)
        bottom = -C[1, 0] + 1j * C[0, 0] - C[3, 1] - 1j * C[2, 1]

        # Trace projection
        traces = torch.diagonal(v, dim1=-2, dim2=-1).sum(-1)  # (2,)
        # traces[0] is tr(v1), traces[1] is tr(v2)
        
        top = top - self._omega_coeff * v[0] + (traces[0] / self.ncol) * self._eye_n
        bottom = bottom - self._omega_coeff * v[1] + (traces[1] / self.ncol) * self._eye_n

        return torch.stack((top, bottom), dim=0)

    def _apply_k_dag(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        X_exp = X.unsqueeze(1)
        v_exp = v.unsqueeze(0)
        C = X_exp @ v_exp - v_exp @ X_exp

        # K_11^dag = -D4 - iD3
        # K_21^dag = -D2 - iD1
        # top = K_11^dag v1 + K_21^dag v2
        # top = (-D4 - iD3) v1 + (-D2 - iD1) v2
        top = -C[3, 0] - 1j * C[2, 0] - C[1, 1] - 1j * C[0, 1]

        # K_12^dag = D2 - iD1
        # K_22^dag = -D4 + iD3
        # bottom = K_12^dag v1 + K_22^dag v2
        # bottom = (D2 - iD1) v1 + (-D4 + iD3) v2
        bottom = C[1, 0] - 1j * C[0, 0] - C[3, 1] + 1j * C[2, 1]

        traces = torch.diagonal(v, dim1=-2, dim2=-1).sum(-1)
        
        top = top - self._omega_coeff * v[0] + (traces[0] / self.ncol) * self._eye_n
        bottom = bottom - self._omega_coeff * v[1] + (traces[1] / self.ncol) * self._eye_n

        return torch.stack((top, bottom), dim=0)

    def _apply_ktk(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._apply_k_dag(X, self._apply_k(X, v))

    def _multi_shift_cg_solve(
        self, X: torch.Tensor, b: torch.Tensor, shifts: torch.Tensor
    ) -> list[torch.Tensor]:
        # Solves (K^dag K + shift) x = b for multiple shifts
        # Assumes shifts are sorted ascending (shifts[0] is smallest)
        sigma0 = shifts[0]
        
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_sq = _inner(r, r).real
        
        xs = [torch.zeros_like(b) for _ in shifts]
        ps = [b.clone() for _ in shifts]
        zetas = [torch.tensor(1.0, device=b.device, dtype=b.dtype) for _ in shifts]
        zetas_prev = [torch.tensor(1.0, device=b.device, dtype=b.dtype) for _ in shifts]
        
        alpha_prev = torch.tensor(1.0, device=b.device, dtype=b.dtype)
        beta_prev = torch.tensor(0.0, device=b.device, dtype=b.dtype)
        
        # Use compiled operator if available
        apply_op = self._apply_ktk_compiled
        
        for k in range(self._cg_maxiter):
            # Base system step
            Ap = apply_op(X, p) + sigma0 * p
            pAp = _inner(p, Ap).real
            
            if pAp.item() == 0.0:
                break
            
            alpha = r_sq / pAp
            
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            r_sq_new = _inner(r_new, r_new).real
            
            if r_sq.item() == 0.0:
                beta = torch.tensor(0.0, device=b.device, dtype=b.dtype)
            else:
                beta = r_sq_new / r_sq
            
            # Shifted systems update
            for i, s in enumerate(shifts):
                if i == 0:
                    xs[i] = x_new
                    continue
                
                du = s - sigma0
                
                if k == 0:
                    zeta_new = zetas[i] / (1.0 + alpha * du)
                else:
                    num = zetas[i] * zetas_prev[i] * alpha_prev
                    denom = (alpha * beta_prev * (zetas_prev[i] - zetas[i]) + 
                             zetas_prev[i] * alpha_prev * (1.0 + du * alpha))
                    if denom.abs().item() < 1e-20:
                        zeta_new = zetas[i]
                    else:
                        zeta_new = num / denom
                
                if zetas[i].abs().item() < 1e-20:
                    beta_i = beta
                    alpha_i = alpha
                else:
                    ratio = zeta_new / zetas[i]
                    beta_i = (ratio**2) * beta
                    alpha_i = ratio * alpha
                
                xs[i] = xs[i] + alpha_i * ps[i]
                ps[i] = zeta_new * r_new + beta_i * ps[i]
                
                zetas_prev[i] = zetas[i]
                zetas[i] = zeta_new
            
            x = x_new
            r = r_new
            r_sq = r_sq_new
            p = r + beta * p
            ps[0] = p
            
            alpha_prev = alpha
            beta_prev = beta
            
            if r_sq.item() < self._cg_tol**2:
                break
                
        return xs

    def _apply_rational(
        self,
        X: torch.Tensor,
        vec: torch.Tensor,
        c0: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> torch.Tensor:
        out = c0 * vec
        # Use multi-shift CG
        sols = self._multi_shift_cg_solve(X, vec, self._rhmc_shifts)
        for coeff, sol in zip(coeffs, sols):
            out = out + coeff * sol
        return out

    def _pseudofermion_action(self, X: torch.Tensor) -> torch.Tensor:
        if self._phi is None:
            self.refresh_aux_fields()
        phi = self._phi
        if phi is None:
            raise ValueError("Pseudofermion field was not initialized")
        y = self._apply_rational(X, phi, self._rhmc_c0_inv, self._rhmc_c_inv)
        return _inner(phi, y).real

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        bos = -0.5 * _commutator_action_sum(X)
        if not self.no_myers:
            bos = bos + 2j * (1 + self.omega) * (
                torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1])
            )
        trace_sq = torch.einsum("bij,bji->b", X, X)
        coeffs = torch.full((self.nmat,), self.omega / 3, dtype=X.dtype, device=X.device)
        extra = torch.tensor(2 / 9, dtype=X.dtype, device=X.device)
        upto = min(3, self.nmat)
        coeffs[:upto] = coeffs[:upto] + extra
        bos = bos + torch.dot(coeffs, trace_sq)

        pf_action = self._pseudofermion_action(X)

        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])

        return (bos.real * (self.ncol / self.g)) + pf_action + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = []
            for i in range(self.nmat):
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

    def build_paths(self, name_prefix: str, data_path: str) -> dict[str, str]:
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self.model_name}_g{round(self.g, 4)}_omega{round(self.omega, 4)}_N{self.ncol}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        return [
            f"  Coupling g               = {self.g}",
            f"  Coupling Omega2/Omega1   = {self.omega}",
            f"  RHMC degree              = {self.rhmc_degree}",
            f"  RHMC lambda min/max      = {self.rhmc_lambda_min}, {self.rhmc_lambda_max}",
        ]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        casimir = (
            torch.trace(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]) / self.ncol
        ).item().real
        trX4 = (torch.trace(X[3] @ X[3]) / self.ncol).item().real
        return f"casimir = {casimir:.5f}, trX_4^2 = {trX4:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "no_myers": self.no_myers,
                "has_source": self.source is not None,
                "model_variant": "type2_rhmc",
                "rhmc_degree": self.rhmc_degree,
                "rhmc_lambda_min": self.rhmc_lambda_min,
                "rhmc_lambda_max": self.rhmc_lambda_max,
                "rhmc_samples": self.rhmc_samples,
            }
        )
        return meta
