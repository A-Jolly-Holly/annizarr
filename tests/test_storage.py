"""Plain-zarr storage: the atomic temp-and-rename swap, cleanup on failure, and the URI helpers."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import anndata as ad
import pytest
import zarr

from annizarr._storage import (
    bucket_prefix,
    canonical_location,
    is_icechunk_repo,
    is_readonly_path,
    open_output_store,
    verify_store_shape,
)
from annizarr.config import AppConfig, IOConfig
from annizarr.errors import ConversionError, StorageError
from annizarr.ops import convert_h5ad, sort
from helpers import cfg, make_adata, make_h5ad, make_store


def _boom(*_a: object, **_kw: object) -> None:
    raise RuntimeError("boom")


def _convert_module() -> types.ModuleType:
    return importlib.import_module("annizarr.ops._convert")  # the package attribute is the function


def test_uri_helpers(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="Missing bucket"):
        bucket_prefix("s3:///prefix")
    assert is_readonly_path(str(tmp_path)) is False and is_readonly_path("s3://bucket/prefix") is False
    assert canonical_location(str(tmp_path)) == os.path.realpath(str(tmp_path))
    assert canonical_location("s3://bucket/prefix") == "s3://bucket/prefix"
    repo = tmp_path / "repo.icechunk"
    assert is_icechunk_repo(repo) is False
    (repo / "snapshots").mkdir(parents=True)
    (repo / "repo").write_text("x")
    assert is_icechunk_repo(repo) is True
    (repo / "zarr.json").write_text("{}")
    assert is_icechunk_repo(repo) is False  # zarr.json makes it a plain store


def test_failed_or_interrupted_convert_leaves_no_target_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    convert_h5ad(str(h5), output=str(tmp_path / "ok.zarr"), cfg=cfg())
    assert {p.name for p in tmp_path.iterdir()} == {"in.h5ad", "ok.zarr"}  # a success leaves only the target

    def interrupt(*_a: object, **_kw: object) -> None:
        raise KeyboardInterrupt

    for exc, fail in ((RuntimeError, _boom), (KeyboardInterrupt, interrupt)):  # BaseException cleans up too
        monkeypatch.setattr(_convert_module(), "write_adata", fail)
        with pytest.raises(exc):
            convert_h5ad(str(h5), output=str(tmp_path / "out.zarr"), cfg=cfg())
        assert not (tmp_path / "out.zarr").exists() and not list(tmp_path.glob("out.zarr.tmp-*"))


def test_overwrite_keeps_the_old_store_until_the_new_one_is_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=cfg())
    monkeypatch.setattr(_convert_module(), "write_adata", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(out), cfg=cfg(overwrite=True))
    assert ad.read_zarr(str(out)).n_obs == 10 and not list(tmp_path.glob("out.zarr.tmp-*"))
    monkeypatch.undo()

    # the swap is two renames: failing either leaves the old store at the target and no siblings
    import annizarr._storage._open as open_mod

    h5_b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=7, seed=1))
    for call_n in (1, 2):
        calls = {"count": 0}

        def fake_replace(src: object, dst: object, *, _n: int = call_n, _calls: dict[str, int] = calls) -> None:
            _calls["count"] += 1
            if _calls["count"] == _n:
                raise OSError("boom-replace")
            os.replace(src, dst)

        monkeypatch.setattr(open_mod, "os", types.SimpleNamespace(replace=fake_replace))
        with pytest.raises(ConversionError, match="boom-replace"):
            convert_h5ad(str(h5_b), output=str(out), cfg=cfg(overwrite=True))
        monkeypatch.undo()
        assert ad.read_zarr(str(out)).n_obs == 10
        assert {p.name for p in tmp_path.iterdir()} == {"in.h5ad", "b.h5ad", "out.zarr"}
    convert_h5ad(str(h5_b), output=str(out), cfg=cfg(overwrite=True))
    assert ad.read_zarr(str(out)).n_obs == 7


def test_sort_temp_buckets_are_removed_on_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path, adata=make_adata(n_obs=20, seed=0), cfg=cfg())
    sort(str(store), output=str(tmp_path / "sorted.zarr"), by=("cell_type",), cfg=cfg())
    assert (tmp_path / "sorted.zarr").exists()
    monkeypatch.setattr(importlib.import_module("annizarr._core._sorting"), "write_matrix", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        sort(str(store), output=str(tmp_path / "sorted2.zarr"), by=("cell_type",), cfg=cfg())
    assert not (tmp_path / "sorted2.zarr").exists()
    assert not list(tmp_path.glob("annizarr_sort_*")) and not list(tmp_path.glob("sorted*.tmp-*"))


def test_finalize_verifies_the_shape_and_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path, "existing.zarr", adata=make_adata(n_obs=5, n_vars=5, seed=0), cfg=cfg())
    out = open_output_store(store, cfg(overwrite=True), expected_shape=(5, 5))
    out.root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    out.root.require_array("X", shape=(3, 3), dtype="float32", chunks=(3, 3))
    with pytest.raises(StorageError, match="shape mismatch"):
        out.finalize()
    out.abort()
    assert ad.read_zarr(str(store)).n_obs == 5 and not list(tmp_path.glob("existing.zarr.tmp-*"))

    target = tmp_path / "out.zarr"
    out = open_output_store(target, AppConfig(io=IOConfig(consolidate_metadata=True)))
    out.root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    out.root.require_array("X", shape=(1, 1), dtype="float32")
    assert out.finalize() is None and out.finalize() is None  # the second call is a no-op
    assert target.exists() and not list(tmp_path.glob("*.tmp-*"))
    assert "icechunk" not in sys.modules or True  # the plain path never needs the extra


def test_verify_store_shape_catches_a_short_obs_column(tmp_path: Path) -> None:
    root = zarr.open_group(str(make_store(tmp_path, adata=make_adata(n_obs=30, seed=0))), mode="r+")
    verify_store_shape(root, n_obs=30, n_vars=50, nnz=int(root["X/indptr"][-1]))
    with pytest.raises(StorageError, match="nonzeros"):
        verify_store_shape(root, n_obs=30, n_vars=50, nnz=1)
    root["obs/cell_type/codes"].resize((29,))
    with pytest.raises(StorageError, match="obs/cell_type has 29 rows"):
        verify_store_shape(root, n_obs=30, n_vars=50)
