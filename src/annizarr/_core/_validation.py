from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annizarr.errors import ValidationError

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["require_matrix"]


def require_matrix(adata: AnnData, *, x: Any = None) -> None:
    """Reject an AnnData with no cells, no genes, or no X; anything else converts.

    ``x`` is the matrix that will be written when it is not ``adata.X`` (a source's own reader).
    """
    if adata.n_obs < 1:
        raise ValidationError(f"AnnData has too few observations: {adata.n_obs} < 1")
    if adata.n_vars < 1:
        raise ValidationError(f"AnnData has too few variables: {adata.n_vars} < 1")
    if (adata.X if x is None else x) is None:
        raise ValidationError("AnnData has no expression matrix in X.")
