import numpy as np
import pytest
from anndata import AnnData

from annizarr._core._validation import validate_single_cell_anndata
from annizarr.errors import ValidationError


def test_validate_non_spatial_ok() -> None:
    assert validate_single_cell_anndata(AnnData(X=np.ones((3, 4)))).ok


def test_validate_reject_spatial_from_uns() -> None:
    adata = AnnData(X=np.ones((3, 4)))
    adata.uns["spatial"] = {"library": {}}
    with pytest.raises(ValidationError):
        validate_single_cell_anndata(adata)


def test_validate_reject_empty() -> None:
    with pytest.raises(ValidationError):
        validate_single_cell_anndata(AnnData(X=np.ones((0, 4))))
