"""Type II polarized IKKT model."""

from __future__ import annotations

import os

import numpy as np
import torch

from pIKKT4D import config
from pIKKT4D.algebra import ad_matrix, get_eye_cached, get_trace_projector_cached, random_hermitian, spinJMatrices
from pIKKT4D.models.base import MatrixModel
from pIKKT4D.models.utils import _commutator_action_sum


ENABLE_TORCH_COMPILE = config.ENABLE_TORCH_COMPILE


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

    det = torch.slogdet(K)
    return det


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

        dim_tr = self.ncol * self.ncol
        omega_eye = (2 / 3 * self.omega) * get_eye_cached(2 * dim_tr, device=config.device, dtype=config.dtype)
        base_fn = lambda X: _type2_logdet_impl(X, omega_eye)
        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._log_det_fn = torch.compile(base_fn, dynamic=False)
        else:
            self._log_det_fn = base_fn

    def load_fresh(self, args):
        mats = [random_hermitian(self.ncol) for _ in range(self.nmat)]
        X = torch.stack(mats, dim=0).to(dtype=config.dtype, device=config.device)

        if args.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(args.spin)).to(dtype=config.dtype, device=config.device)
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=config.dtype, device=config.device)
            for i in range(3):
                X[i][:ntimes * J_matrices.shape[1], :ntimes * J_matrices.shape[1]] = (
                    (2 / 3 + self.omega) * torch.kron(eye_nt, J_matrices[i])
                )
            X[3] = torch.zeros_like(X[3])

        self.set_state(X)

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

        det = -self._log_det_fn(X)[1].real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / np.sqrt(self.g)) * torch.trace(self.source @ X[0])
        return (bos.real * (self.ncol / self.g)) + det + src.real

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
        return [f"  Coupling g               = {self.g}", f"  Coupling Omega2/Omega1   = {self.omega}"]

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
                "model_variant": "type2",
            }
        )
        return meta
