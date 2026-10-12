from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from annizarr._core._config import DENSE_CHUNK, resolve_backend_cfg
from annizarr._core._validation import require_matrix
from annizarr.config import AppConfig, ChunkConfig, IOConfig, apply_cli_overrides
from annizarr.errors import ValidationError


def test_defaults() -> None:
    cfg = AppConfig()
    assert cfg.chunks.row_chunk is None and cfg.chunks.col_chunk is None  # dense falls back to DENSE_CHUNK
    assert DENSE_CHUNK == 2048 and cfg.chunks.nnz_chunk == 9_000_000
    assert cfg.io.layout == "csr" and cfg.io.lazy is True and cfg.chunks.auto_shard is False
    assert cfg.chunks.cpus == (os.cpu_count() or 1)


def test_cli_overrides_apply_and_validate() -> None:
    cfg = apply_cli_overrides(AppConfig(), row_chunk=128, overwrite=True, layout="csr", auto_shard=True)
    assert (cfg.chunks.row_chunk, cfg.io.overwrite, cfg.io.layout, cfg.chunks.auto_shard) == (128, True, "csr", True)
    assert apply_cli_overrides(cfg, auto_shard=None).chunks.auto_shard is True  # None leaves a setting alone
    with pytest.raises(ValueError, match="layout"):
        apply_cli_overrides(AppConfig(), layout="not-a-mode")
    with pytest.raises(ValueError, match="shard_factor"):
        apply_cli_overrides(AppConfig(), shard_factor=0)


def test_resolve_backend_cfg_keeps_lazy_and_warns_on_sparse_sharding(caplog: pytest.LogCaptureFixture) -> None:
    for backend in ("zarr", "icechunk"):
        for lazy in (True, False):
            assert resolve_backend_cfg(AppConfig(io=IOConfig(backend=backend, lazy=lazy))).io.lazy == lazy
    resolve_backend_cfg(AppConfig(chunks=ChunkConfig(shard_factor=2), io=IOConfig(layout="csr")))
    assert any("only applies to dense X" in r.message for r in caplog.records)


def test_require_matrix() -> None:
    spatial = AnnData(X=np.ones((3, 4)))
    spatial.uns["spatial"] = {"library": {}}
    spatial.obsm["spatial"] = np.zeros((3, 2))
    require_matrix(spatial)
    with pytest.raises(ValidationError, match="too few observations"):
        require_matrix(AnnData(X=np.ones((0, 4))))
    with pytest.raises(ValidationError, match="too few variables"):
        require_matrix(AnnData(X=np.ones((3, 0))))
    no_x = AnnData(obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g1", "g2", "g3"]))
    with pytest.raises(ValidationError, match="no expression matrix"):
        require_matrix(no_x)
    require_matrix(no_x, x=np.ones((2, 3)))  # a source's own reader stands in for X
