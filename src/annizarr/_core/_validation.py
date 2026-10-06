from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from annizarr.errors import ValidationError

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["ValidationResult", "validate_single_cell_anndata"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    warnings: list[str]


def _has_spatial_markers(adata: AnnData) -> bool:
    if "spatial" in adata.uns:
        return True

    # Common conventions in AnnData objects carrying spatial coordinates.
    for key in adata.obsm:
        if "spatial" in key.lower() or key.lower().startswith("x_spatial"):
            return True

    return False


def validate_single_cell_anndata(adata: AnnData) -> ValidationResult:
    # non-empty, non-spatial single-cell AnnData with an X; duplicate names only warn
    warnings: list[str] = []

    if adata.n_obs < 1:
        raise ValidationError(f"AnnData has too few observations: {adata.n_obs} < 1")
    if adata.n_vars < 1:
        raise ValidationError(f"AnnData has too few variables: {adata.n_vars} < 1")

    if adata.X is None:
        raise ValidationError("AnnData has no expression matrix in X.")

    if _has_spatial_markers(adata):
        raise ValidationError(
            "Input appears spatial (detected spatial markers in uns/obsm); "
            "annizarr handles non-spatial single-cell AnnData only."
        )

    if adata.obs_names.has_duplicates:
        warnings.append("obs_names contain duplicates; downstream tools may require uniqueness.")

    if adata.var_names.has_duplicates:
        warnings.append("var_names contain duplicates; downstream tools may require uniqueness.")

    return ValidationResult(ok=True, warnings=warnings)
