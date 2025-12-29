"""Matrix models and helpers."""

from __future__ import annotations

from argparse import Namespace

from pIKKT4D.models.base import MatrixModel
from pIKKT4D.models.one_matrix import OneMatrixPolynomialModel
from pIKKT4D.models.pikkt4d_type1 import PIKKTTypeIModel
from pIKKT4D.models.pikkt4d_type2 import PIKKTTypeIIModel
from pIKKT4D.models.adjoint_det import AdjointDetModel
from pIKKT4D.models.yang_mills import YangMillsModel
from pIKKT4D.models.utils import gammaMajorana, gammaWeyl


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


__all__ = [
    "MatrixModel",
    "OneMatrixPolynomialModel",
    "PIKKTTypeIModel",
    "PIKKTTypeIIModel",
    "AdjointDetModel",
    "YangMillsModel",
    "build_model",
    "gammaMajorana",
    "gammaWeyl",
]
