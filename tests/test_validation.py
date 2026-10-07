import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from annizarr._core._validation import require_matrix
from annizarr.errors import ValidationError


def test_spatial_input_is_accepted() -> None:
    adata = AnnData(X=np.ones((3, 4)))
    adata.uns["spatial"] = {"library": {}}
    adata.obsm["spatial"] = np.zeros((3, 2))
    require_matrix(adata)  # no raise: spatial AnnData is welcome


def test_reject_empty() -> None:
    with pytest.raises(ValidationError, match="too few observations"):
        require_matrix(AnnData(X=np.ones((0, 4))))
    with pytest.raises(ValidationError, match="too few variables"):
        require_matrix(AnnData(X=np.ones((3, 0))))


def test_reject_missing_x() -> None:
    adata = AnnData(obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g1", "g2", "g3"]))
    with pytest.raises(ValidationError, match="no expression matrix"):
        require_matrix(adata)
