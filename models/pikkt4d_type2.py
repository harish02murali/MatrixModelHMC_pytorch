"""Type II polarized IKKT model."""

from __future__ import annotations

import os

import numpy as np
import torch

from pIKKT4D import config
from pIKKT4D.algebra import (
    ad_matrix,
    get_eye_cached,
    get_trace_projector_cached,
    makeH,
    random_hermitian,
    spinJMatrices,
)
from pIKKT4D.models.base import MatrixModel
from pIKKT4D.models.utils import _commutator_action_sum


def _adjoint_grad_from_matrix(M: torch.Tensor, ncol: int) -> torch.Tensor:
    """Return grad of Tr(M^T ad_X) with column-major vec ordering."""
    M4 = M.reshape(ncol, ncol, ncol, ncol).permute(1, 0, 3, 2)
    diag_jl = M4.diagonal(dim1=1, dim2=3)
    grad_left = diag_jl.sum(dim=-1)
    diag_ik = M4.diagonal(dim1=0, dim2=2)
    grad_right = diag_ik.sum(dim=-1).transpose(0, 1)
    return (grad_left - grad_right).conj()


class PIKKTTypeIIModel(MatrixModel):
    """Type II polarized IKKT model definition."""

    model_name = "pikkt4d_type2"

    def __init__(self, ncol: int, couplings: list, source: np.ndarray | None = None) -> None:
        super().__init__(name="pIKKT Type II", nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = self.couplings[1]
        self.source = (
            torch.diag(torch.tensor(source, device=config.device, dtype=config.dtype))
            if source is not None
            else None
        )
        self.is_hermitian = True
        self.is_traceless = True
        self.model_key = "type2"

        dim_tr = self.ncol * self.ncol
        self._omega_eye = (2 / 3 * self.omega) * get_eye_cached(
            2 * dim_tr, device=config.device, dtype=config.dtype
        )
        if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            self._force_impl = torch.compile(self._force_impl, dynamic=False)

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

    def fermionMat(self, X: torch.Tensor) -> torch.Tensor:
        adX = 1j * ad_matrix(X[:4])
        adX1, adX2, adX3, adX4 = adX
        i = 1j

        upper_left = -adX4 + i * adX3
        upper_right = adX2 + i * adX1
        lower_left = -adX2 + i * adX1
        lower_right = -adX4 - i * adX3

        top = torch.cat((upper_left, upper_right), dim=1)
        bottom = torch.cat((lower_left, lower_right), dim=1)
        K = torch.cat((top, bottom), dim=0)

        omega_eye = self._omega_eye.to(dtype=K.dtype)
        K = K - omega_eye

        N = X.shape[-1]
        dim = N * N
        P = get_trace_projector_cached(N, K.device, K.dtype)
        K[:dim, :dim] += P
        K[dim:, dim:] += P
        return K

    def _fermion_force(self, X: torch.Tensor) -> torch.Tensor:
        K = self.fermionMat(X)
        dim = self.ncol * self.ncol
        eye = get_eye_cached(2 * dim, device=K.device, dtype=K.dtype)
        K_inv = torch.linalg.solve(K, eye)

        G11 = K_inv[:dim, :dim].t()
        G12 = K_inv[:dim, dim:].t()
        G21 = K_inv[dim:, :dim].t()
        G22 = K_inv[dim:, dim:].t()

        M1 = -(G21 + G12)
        M2 = 1j * (G21 - G12)
        M3 = -G11 + G22
        M4 = -1j * (G11 + G22)

        grads = []
        for M in (M1, M2, M3, M4):
            grads.append(-_adjoint_grad_from_matrix(M, self.ncol))
        return torch.stack(grads, dim=0)

    def _force_impl(self, X: torch.Tensor) -> torch.Tensor:
        X = self._resolve_X(X)
        grad = torch.zeros_like(X)

        for i in range(self.nmat):
            acc = torch.zeros_like(X[i])
            for j in range(self.nmat):
                if i == j:
                    continue
                comm = X[i] @ X[j] - X[j] @ X[i]
                acc = acc + (X[j] @ comm - comm @ X[j])
            grad[i] = -acc

        coeff = 2j * (1 + self.omega)
        grad[0] += coeff * (X[1] @ X[2] - X[2] @ X[1])
        grad[1] += coeff * (X[2] @ X[0] - X[0] @ X[2])
        grad[2] += coeff * (X[0] @ X[1] - X[1] @ X[0])

        coeffs = torch.full((self.nmat,), self.omega / 3, dtype=config.real_dtype, device=X.device)
        extra = torch.tensor(2 / 9, dtype=config.real_dtype, device=X.device)
        upto = min(3, self.nmat)
        coeffs[:upto] = coeffs[:upto] + extra
        grad = grad + 2 * coeffs[:, None, None] * X

        grad = grad * (self.ncol / self.g)
        grad = grad + self._fermion_force(X)

        if self.source is not None:
            grad[0] += -(self.ncol / np.sqrt(self.g)) * self.source

        return grad

    def force(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        grad = self._force_impl(X)
        if self.is_hermitian:
            grad = makeH(grad)
        if self.is_traceless:
            trs = torch.diagonal(grad, dim1=-2, dim2=-1).sum(-1).real / self.ncol
            eye = get_eye_cached(self.ncol, device=grad.device, dtype=grad.dtype)
            grad = grad - trs[..., None, None] * eye
        return grad

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        bos = -0.5 * _commutator_action_sum(X)
        bos = bos + 2j * (1 + self.omega) * (
            torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1])
        )
        trace_sq = torch.einsum("bij,bji->b", X, X).real
        coeffs = torch.full((self.nmat,), self.omega / 3, dtype=config.real_dtype, device=X.device)
        extra = torch.tensor(2 / 9, dtype=config.real_dtype, device=X.device)
        upto = min(3, self.nmat)
        coeffs[:upto] = coeffs[:upto] + extra
        bos = bos + torch.dot(coeffs, trace_sq)

        K = self.fermionMat(X)
        det = -torch.slogdet(K)[1].real

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

            eigs.append(
                torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2])
                .cpu()
                .numpy()
            )

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
                "has_source": self.source is not None,
                "model_variant": "type2",
            }
        )
        return meta
