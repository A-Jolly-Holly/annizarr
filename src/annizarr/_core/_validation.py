from __future__ import annotations

from typing import TYPE_CHECKING

from annizarr.errors import ValidationError

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["require_matrix"]


def require_matrix(adata: AnnData) -> None:
    """Reject an AnnData with no cells, no genes, or no X; anything else converts."""
    if adata.n_obs < 1:
        raise ValidationError(f"AnnData has too few observations: {adata.n_obs} < 1")
    if adata.n_vars < 1:
        raise ValidationError(f"AnnData has too few variables: {adata.n_vars} < 1")
    if adata.X is None:
        raise ValidationError("AnnData has no expression matrix in X.")
