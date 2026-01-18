"""Single-matrix polynomial model."""

from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import torch

from pIKKT4D import config
from pIKKT4D.models.base import MatrixModel


class OneMatrixPolynomialModel(MatrixModel):
    """Single-matrix polynomial model V(X) = sum_n t_n Tr(X^n)."""

    model_name = "1mm"

    def __init__(self, ncol: int, couplings: list) -> None:
        if len(couplings) == 0:
            raise ValueError("1MM model requires at least one coupling via --coupling t1 [t2 ...]")
        super().__init__(nmat=1, ncol=ncol)
        self.couplings = couplings
        self.is_hermitian = True
        self.is_traceless = False
        self._coupling_tensor = torch.tensor(couplings, dtype=config.real_dtype, device=config.device)

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

    def measure_observables(self, X: torch.Tensor | None = None) -> tuple:
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

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "polynomial_degree": len(self.couplings),
                "model_variant": "1mm_polynomial",
            }
        )
        return meta
