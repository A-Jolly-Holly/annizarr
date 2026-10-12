"""Conversion: every input format and layout, single and concatenated, lazy and eager."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

import annizarr._core._layout as _layout
from annizarr.config import AppConfig, ChunkConfig, ConcatConfig, IOConfig
from annizarr.errors import ConversionError, StorageError
from annizarr.ops import concat, convert, convert_adata, convert_h5ad
from helpers import cfg, chunks, messages, read_x

ENCODING = {"dense": "array", "csr": "csr_matrix", "csc": "csc_matrix"}
N_OBS, N_VARS = 60, 40


def _fixture() -> np.ndarray:
    # two empty rows and one empty column, so readers' indptr, band and flat math see real sparsity
    rng = np.random.default_rng(0)
    dense = (rng.random((N_OBS, N_VARS), dtype=np.float32) < 0.3) * rng.random((N_OBS, N_VARS), dtype=np.float32)
    dense = dense.astype(np.float32)
    dense[5, :] = 0.0
    dense[41, :] = 0.0
    dense[:, 17] = 0.0
    return dense


DENSE = _fixture()


def _write_h5ad(path: Path, dense: np.ndarray, fmt: str, *, obs_prefix: str = "c") -> None:
    x: Any = {"dense": dense.copy(), "csr": sp.csr_matrix(dense), "csc": sp.csc_matrix(dense)}[fmt]
    obs = pd.DataFrame(index=[f"{obs_prefix}{i}" for i in range(dense.shape[0])])
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(dense.shape[1])])
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(path)


def _write_10x_v3(path: Path, dense: np.ndarray) -> None:
    csc_t = sp.csc_matrix(dense.T)  # on disk: genes x barcodes
    n_vars = dense.shape[1]
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([f"TENX{i}-1".encode() for i in range(dense.shape[0])]))
        m.create_dataset("data", data=csc_t.data.astype(np.float32))
        m.create_dataset("indices", data=csc_t.indices.astype(np.int32))
        m.create_dataset("indptr", data=csc_t.indptr.astype(np.int32))
        m.create_dataset("shape", data=np.array([n_vars, dense.shape[0]], dtype=np.int32))
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array([f"ENSG{i:03d}".encode() for i in range(n_vars)]))
        feat.create_dataset("name", data=np.array([f"GENE{i}".encode() for i in range(n_vars)]))
        feat.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_vars))


class _ReadRecorder:
    # records (dataset, requested elements, total elements) for X-bearing h5py reads so a lazy
    # conversion can be checked for genuine banding rather than one whole-array read
    def __init__(self) -> None:
        self.records: list[tuple[str, int, int]] = []

    def __call__(self, dataset: h5py.Dataset, key: Any, orig: Any) -> Any:
        name = dataset.name.rsplit("/", 1)[-1]
        if name in ("X", "data", "indices"):
            keys = key if isinstance(key, tuple) else (key,)
            count = 1
            for axis, k in enumerate(keys):
                dim = dataset.shape[axis]
                count *= len(range(*k.indices(dim))) if isinstance(k, slice) else (1 if np.isscalar(k) else dim)
            self.records.append((name, count, int(np.prod(dataset.shape))))
        return orig(dataset, key)


@pytest.fixture
def read_recorder(monkeypatch: pytest.MonkeyPatch) -> _ReadRecorder:
    recorder = _ReadRecorder()
    orig = h5py.Dataset.__getitem__
    monkeypatch.setattr(h5py.Dataset, "__getitem__", lambda self, key: recorder(self, key, orig))
    return recorder


@pytest.mark.parametrize("input_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("layout", ["dense", "csr", "csc"])
def test_convert_matrix_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_recorder: _ReadRecorder,
    input_format: str,
    lazy: bool,
    layout: str,
) -> None:
    monkeypatch.setattr(_layout, "BATCH_BYTES", 512)  # real banding on this small fixture
    _write_h5ad(tmp_path / "in.h5ad", DENSE, input_format)
    out = tmp_path / "out.zarr"
    # cpus=1: the h5py read recorder patches this process only
    config = AppConfig(
        io=IOConfig(layout=layout, lazy=lazy), chunks=ChunkConfig(row_chunk=7, col_chunk=6, nnz_chunk=11, cpus=1)
    )
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=config)

    np.testing.assert_array_equal(read_x(out), DENSE)
    root = zarr.open_group(str(out), mode="r")
    assert root["X"].attrs["encoding-type"] == ENCODING[layout] and root["X"].attrs["encoding-version"]
    if lazy:
        by_name: dict[str, list[tuple[int, int]]] = {}
        for name, count, total in read_recorder.records:
            by_name.setdefault(name, []).append((count, total))
        assert by_name, "a lazy conversion must read X through h5py"
        for name, calls in by_name.items():
            assert len(calls) > 1 and any(c < t for c, t in calls), f"{name!r} was not read in bounded bands: {calls}"


@pytest.mark.parametrize("layout", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mix", ["dense_csr", "csr_csc", "eager", "tenx_h5ad"])
def test_concat_matrix_grid(tmp_path: Path, mix: str, layout: str) -> None:
    rng = np.random.default_rng(1)
    a = ((rng.random((12, 6), dtype=np.float32) < 0.4) * rng.random((12, 6), dtype=np.float32)).astype(np.float32)
    b = ((rng.random((9, 6), dtype=np.float32) < 0.4) * rng.random((9, 6), dtype=np.float32)).astype(np.float32)
    a_path, b_path = tmp_path / "a.in", tmp_path / "b.in"
    config = AppConfig(
        io=IOConfig(layout=layout, lazy=mix != "eager"),
        chunks=ChunkConfig(row_chunk=4, col_chunk=3, nnz_chunk=5, cpus=2),
    )
    if mix == "tenx_h5ad":
        _write_10x_v3(a_path, a)
    else:
        _write_h5ad(a_path, a, "csr" if mix == "csr_csc" else "dense", obs_prefix="a")
    _write_h5ad(b_path, b, "csc" if mix == "csr_csc" else "csr", obs_prefix="b")

    result = convert([str(a_path), str(b_path)], output=str(tmp_path / "out.zarr"), cfg=config)
    assert (result.n_obs, result.n_vars) == (21, 6)
    got = ad.read_zarr(str(tmp_path / "out.zarr"))
    np.testing.assert_array_equal(read_x(tmp_path / "out.zarr"), np.vstack([a, b]))
    assert list(got.var_names) == [f"GENE{i}" for i in range(6)]
    if mix != "tenx_h5ad":
        assert list(got.obs_names) == [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(9)]
    assert zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["X"].attrs["encoding-type"] == ENCODING[layout]


def test_convert_adata_in_memory_and_backed(tmp_path: Path) -> None:
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(dense))
    convert_adata(adata, output=str(tmp_path / "mem.zarr"), cfg=cfg(lazy=True))  # lazy is synced from the AnnData
    np.testing.assert_array_equal(read_x(tmp_path / "mem.zarr"), dense)

    ad.AnnData(X=dense.copy()).write_h5ad(tmp_path / "in.h5ad")
    backed = ad.read_h5ad(tmp_path / "in.h5ad", backed="r")
    try:
        convert_adata(backed, output=str(tmp_path / "backed.zarr"), cfg=cfg(layout="dense"))
    finally:
        backed.file.close()
    np.testing.assert_array_equal(read_x(tmp_path / "backed.zarr"), dense)


def test_existing_target_fails_before_loading_and_flat_chunks_apply_to_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adata = ad.AnnData(X=sp.csr_matrix(DENSE))
    adata.layers["counts"] = sp.csr_matrix(DENSE * 2)
    adata.write_h5ad(tmp_path / "in.h5ad")
    (tmp_path / "exists.zarr").mkdir()
    convert_mod = importlib.import_module("annizarr.ops._convert")

    def fail(*_a: object, **_kw: object) -> None:
        raise AssertionError("load_h5ad must not run when the target exists and overwrite is unset")

    monkeypatch.setattr(convert_mod, "load_h5ad", fail)
    with pytest.raises(StorageError, match="already exists"):
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / "exists.zarr"), cfg=cfg())
    monkeypatch.undo()

    for layout, flat in (("csr", 100), ("csc", 75)):
        out = tmp_path / f"{layout}.zarr"
        flat_only = chunks(row_chunk=None, col_chunk=None, nnz_chunk=flat)  # no axis chunk: nnz_chunk sizes the arrays
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg(layout=layout, chunks=flat_only))
        root = zarr.open_group(str(out), mode="r")
        assert root["X/data"].chunks == (flat,) and root["layers/counts/data"].chunks == (flat,)
        np.testing.assert_array_equal(read_x(out), DENSE)


def test_lazy_reads_backed_and_cleans_up_the_csc_temp_store(tmp_path: Path) -> None:
    adata = ad.AnnData(X=sp.csc_matrix(DENSE))
    adata.layers["cnt"] = sp.csc_matrix(DENSE * 2.0)
    adata.write_h5ad(tmp_path / "in.h5ad")
    with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / "out.zarr"), cfg=cfg(layout="dense"))
    assert mocked.call_args_list[0].kwargs.get("backed") == "r"
    got = ad.read_zarr(str(tmp_path / "out.zarr"))
    np.testing.assert_array_equal(np.asarray(got.X), DENSE)
    np.testing.assert_array_equal(np.asarray(got.layers["cnt"]), DENSE * 2.0)
    assert not list(tmp_path.rglob("annizarr_csc2csr_*"))


def test_frames_are_written_the_way_anndata_writes_them(tmp_path: Path) -> None:
    """A string-dtype index becomes a plain string array; repeated strings become categoricals,
    unique ones stay strings; the caller's frame is untouched."""
    obs = pd.DataFrame(
        {"batch": ["b2", "b1", "b2", "b1"], "barcode": ["AAA", "CCC", "GGG", "TTT"]},
        index=pd.Index([f"c{i}" for i in range(4)], dtype="string"),
    )
    adata = ad.AnnData(X=sp.csr_matrix(np.ones((4, 2), dtype=np.float32)), obs=obs)
    convert_adata(adata, output=str(tmp_path / "out.zarr"), cfg=cfg())
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert isinstance(root["obs/_index"], zarr.Array) and root["obs/_index"].attrs["encoding-type"] == "string-array"
    assert root["obs/batch"].attrs["encoding-type"] == "categorical"
    assert [str(c) for c in root["obs/batch/categories"][:]] == ["b1", "b2"]
    assert root["obs/barcode"].attrs["encoding-type"] != "categorical"
    assert not isinstance(adata.obs["batch"].dtype, pd.CategoricalDtype)
    back = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert list(back.obs_names) == ["c0", "c1", "c2", "c3"] and list(back.obs["batch"]) == ["b2", "b1", "b2", "b1"]


def _obs_h5ad(path: Path, n: int, cats: list[str] | None, extra: str, score: np.ndarray | None = None) -> None:
    cell_type: Any = (
        pd.Categorical([cats[i % len(cats)] for i in range(n)], categories=cats)
        if cats
        else [f"ct{i % 2}" for i in range(n)]
    )
    obs = pd.DataFrame(
        {"cell_type": cell_type, "donor": [f"d{i % 2}" for i in range(n)]}, index=[f"{extra}_{i}" for i in range(n)]
    )
    if score is not None:
        obs["score"] = score
    obs[extra] = np.arange(n)
    ad.AnnData(X=sp.csr_matrix(np.arange(n * 2, dtype=np.float32).reshape(n, 2)), obs=obs).write_h5ad(path)


def test_concat_obs_columns_project_and_keep_matching_categoricals(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _obs_h5ad(tmp_path / "a.h5ad", 6, None, "qc_a", np.arange(6, dtype="int64"))
    _obs_h5ad(tmp_path / "b.h5ad", 4, None, "qc_b", np.arange(4, dtype="float64"))
    config = cfg(concat=ConcatConfig(obs_columns=("cell_type", "donor", "score")))
    concat([str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], output=str(tmp_path / "out.zarr"), cfg=config)
    res = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert list(res.obs.columns) == ["cell_type", "donor", "score"] and res.n_obs == 10
    msgs = messages(caplog)
    assert any("qc_a" in m for m in msgs) and any("qc_b" in m for m in msgs)  # dropped extras are named
    assert any("score" in m and "coerced" in m for m in msgs)  # int + float -> float

    _obs_h5ad(tmp_path / "c.h5ad", 6, ["Tcell", "Bcell", "NK"], "qc_c")
    _obs_h5ad(tmp_path / "d.h5ad", 4, ["Tcell", "Bcell", "NK"], "qc_d")
    config = cfg(concat=ConcatConfig(obs_columns=("cell_type", "donor")))
    concat([str(tmp_path / "c.h5ad"), str(tmp_path / "d.h5ad")], output=str(tmp_path / "cats.zarr"), cfg=config)
    assert isinstance(ad.read_zarr(str(tmp_path / "cats.zarr")).obs["cell_type"].dtype, pd.CategoricalDtype)


def test_concat_rejects_missing_columns_mismatched_categories_and_differing_schemas(tmp_path: Path) -> None:
    _obs_h5ad(tmp_path / "a.h5ad", 6, ["Tcell", "Bcell"], "qc_a")
    _obs_h5ad(tmp_path / "b.h5ad", 4, ["Bcell", "NK"], "qc_b")
    paths = [str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")]
    with pytest.raises(ConversionError, match="obs columns not found"):
        concat(paths, output=str(tmp_path / "o1.zarr"), cfg=cfg(concat=ConcatConfig(obs_columns=("cell_type", "qc_a"))))
    with pytest.raises(ConversionError, match="mismatched categorical categories"):
        concat(
            paths, output=str(tmp_path / "o2.zarr"), cfg=cfg(concat=ConcatConfig(obs_columns=("cell_type", "donor")))
        )
    with pytest.raises(ConversionError, match="obs schema mismatch"):
        concat(paths, output=str(tmp_path / "o3.zarr"), cfg=cfg())
