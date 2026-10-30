"""The edit ops on plain zarr: add-layer, rechunk, sort, append, and their lifecycle together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, sparse_dataset, write_elem

from annizarr._storage import open_input_group
from annizarr.config import AppConfig, ChunkConfig, GroupingConfig
from annizarr.errors import ConversionError, StorageError
from annizarr.ops import add_layer, append, convert_h5ad, plan_append, rechunk, sort
from helpers import assert_anndata_readable, cfg, chunks, make_labelled_h5ad, messages, read_x


def _layer_cfg(nnz_chunk: int) -> AppConfig:
    return AppConfig(chunks=ChunkConfig(nnz_chunk=nnz_chunk, cpus=2))


def _adata(n: int = 40, v: int = 6, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = sp.random(n, v, density=0.5, format="csr", dtype=np.float32, random_state=seed)
    x.data = np.abs(x.data) + 1.0
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["b", "a", "c"], n), categories=["a", "b", "c"])},
        index=[f"c{seed}_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(v)])
    return ad.AnnData(X=x, obs=obs, var=var, obsm={"X_umap": rng.random((n, 2)).astype(np.float32)})


def _store(tmp_path: Path, adata: ad.AnnData, name: str = "store.zarr", config: AppConfig | None = None) -> Path:
    adata.write_h5ad(tmp_path / f"{name}.h5ad")
    out = tmp_path / name
    convert_h5ad(str(tmp_path / f"{name}.h5ad"), output=str(out), cfg=config or cfg())
    assert_anndata_readable(out)
    return out


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _layer(store: Path, name: str) -> np.ndarray:
    return _dense(ad.read_zarr(str(store)).layers[name])


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_add_layer_copies_x_value_for_value(tmp_path: Path, layout: str) -> None:
    adata = _adata()
    out = _store(tmp_path, adata)
    add_layer(str(out), layer="copy", layout=layout, cfg=_layer_cfg(80))
    assert_anndata_readable(out)
    node = zarr.open_group(str(out), mode="r")["layers/copy"]
    if layout == "dense":
        assert node.chunks == (adata.n_obs, adata.n_vars)  # row_chunk x col_chunk tiles, clipped to the shape
    else:
        assert node.attrs["encoding-type"] == {"csr": "csr_matrix", "csc": "csc_matrix"}[layout]
    assert node.attrs["annizarr-source"] == "X"
    np.testing.assert_array_equal(_layer(out, "copy"), adata.X.toarray())
    assert ad.read_zarr(str(out)).layers["copy"].dtype == adata.X.dtype
    with pytest.raises(ConversionError, match="already exists"):
        add_layer(str(out), layer="copy", layout=layout, cfg=_layer_cfg(80))
    add_layer(str(out), layer="copy", layout="csr", overwrite=True, cfg=_layer_cfg(80))
    np.testing.assert_array_equal(_layer(out, "copy"), adata.X.toarray())


def test_add_layer_sources_empty_rows_and_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.zeros((10, 5), dtype=np.float32)
    x[1:9, [0, 1, 3, 4]] = np.arange(1, 33, dtype=np.float32).reshape(8, 4)  # empty first and last rows, empty gene 2
    empty = ad.AnnData(
        X=sp.csr_matrix(x),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        layers={"counts": sp.csr_matrix(x * 2)},
    )
    empty.raw = empty.copy()
    out = _store(tmp_path, empty, "empty.zarr")
    for layer, matrix, layout in (
        ("from_x", "X", "csc"),
        ("from_layer", "layers/counts", "dense"),
        ("from_raw", "raw/X", "csc"),
    ):
        add_layer(str(out), layer=layer, matrix=matrix, layout=layout, cfg=_layer_cfg(16))
        expected = x * 2 if matrix == "layers/counts" else x
        np.testing.assert_array_equal(_layer(out, layer), expected)
        assert zarr.open_group(str(out), mode="r")[f"layers/{layer}"].attrs["annizarr-source"] == matrix

    with pytest.raises(ConversionError, match="onto itself"):
        add_layer(str(out), layer="counts", matrix="layers/counts", overwrite=True, cfg=cfg())
    with pytest.raises(ConversionError, match="not a matrix element"):
        add_layer(str(out), layer="new", matrix="obsm/X_pca", cfg=cfg())
    with pytest.raises(ConversionError, match="layout"):
        add_layer(str(out), layer="new", layout="coo", cfg=cfg())  # type: ignore[arg-type]
    narrow = empty.copy()
    narrow.raw = empty[:, :3].copy()
    with pytest.raises(ConversionError, match="not cells x var"):
        add_layer(str(_store(tmp_path, narrow, "narrow.zarr")), layer="new", matrix="raw/X", cfg=cfg())

    monkeypatch.setattr("annizarr._core._layout.BATCH_BYTES", 4000)  # dozens of source and output bands
    adata = _adata(n=300, v=12, seed=4)
    for layout in ("csc", "dense"):
        out = _store(tmp_path, adata, f"bands-{layout}.zarr")
        add_layer(str(out), layer="copy", layout=layout, cfg=AppConfig(chunks=ChunkConfig(nnz_chunk=80, cpus=1)))
        np.testing.assert_array_equal(_layer(out, "copy"), adata.X.toarray())


def test_rechunk_targets_one_matrix_and_copies_the_rest(tmp_path: Path) -> None:
    adata = _adata()
    out = _store(tmp_path, adata)
    add_layer(str(out), layer="csc", cfg=_layer_cfg(32))
    rechunk(str(out), output=str(tmp_path / "sparse.zarr"), cfg=AppConfig(chunks=ChunkConfig(nnz_chunk=16, cpus=2)))
    assert_anndata_readable(tmp_path / "sparse.zarr")
    root = zarr.open_group(str(tmp_path / "sparse.zarr"), mode="r")
    assert root["X/data"].chunks == (16,) and root["layers/csc/data"].chunks == (32,)  # the layer is copied as is
    got = ad.read_zarr(str(tmp_path / "sparse.zarr"))
    np.testing.assert_allclose(got.X.toarray(), adata.X.toarray())
    assert list(got.obs["cell_type"]) == list(adata.obs["cell_type"])
    np.testing.assert_allclose(got.obsm["X_umap"], adata.obsm["X_umap"])
    np.testing.assert_array_equal(got.layers["csc"].toarray(), adata.X.toarray())

    # the rechunked matrix keeps its source's compression unless clevel says otherwise; copies always keep theirs
    rechunk(
        str(out), output=str(tmp_path / "c9.zarr"), cfg=AppConfig(chunks=ChunkConfig(nnz_chunk=16, clevel=9, cpus=2))
    )
    root9 = zarr.open_group(str(tmp_path / "c9.zarr"), mode="r")
    assert [c.clevel for c in root["X/data"].compressors] == [3]
    assert [c.clevel for c in root9["X/data"].compressors] == [9]
    assert [c.clevel for c in root9["layers/csc/data"].compressors] == [3]

    dense_src = _store(tmp_path, _adata(n=64, v=8), "dense_src.zarr", cfg(layout="dense"))
    for shard_factor in (1, 2):
        out2 = tmp_path / f"dense_sf{shard_factor}.zarr"
        rechunk(
            str(dense_src),
            output=str(out2),
            cfg=AppConfig(chunks=ChunkConfig(row_chunk=8, col_chunk=4, shard_factor=shard_factor, cpus=2)),
        )
        x = zarr.open_group(str(out2), mode="r")["X"]
        assert x.chunks == (8, 4) and (x.shards is None) == (shard_factor == 1)
        np.testing.assert_allclose(read_x(out2), _adata(n=64, v=8).X.toarray())


def _runs(obs: pd.DataFrame, keys: list[str]) -> list[tuple[int, int]]:
    # contiguous runs of equal key tuples in a sorted obs: what a self-serve reader slices by
    tuples = list(zip(*(obs[k].astype(str) for k in keys), strict=True))
    starts = [i for i in range(len(tuples)) if i == 0 or tuples[i] != tuples[i - 1]]
    return list(zip(starts, [*starts[1:], len(tuples)], strict=True))


def test_sort_eager_lazy_and_dense_agree_and_read_by_contiguous_runs(tmp_path: Path) -> None:
    """Convert-time sort by two keys gives the permutation A/x(2,5) A/y(3) B/x(4) B/y(1,6) for the
    labelled fixture, the same for the eager csr, lazy csr and dense paths; obsm follows the rows;
    each key tuple is one contiguous run readable with stock zarr."""
    make_labelled_h5ad(tmp_path / "in.h5ad")
    grouping = GroupingConfig(enabled=True, sort_by=("cell_type", "demographic"))
    variants = {
        "eager": cfg(lazy=False, grouping=grouping),
        "lazy": cfg(lazy=True, grouping=grouping),
        "dense": cfg(layout="dense", lazy=False, grouping=grouping, chunks=chunks(row_chunk=2, col_chunk=2)),
    }
    for name, config in variants.items():
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / f"{name}.zarr"), cfg=config)
        got = ad.read_zarr(str(tmp_path / f"{name}.zarr"))
        assert list(read_x(tmp_path / f"{name}.zarr")[:, 0].astype(int)) == [2, 5, 3, 4, 1, 6]
        assert list(got.obsm["coords"][:, 0].astype(int)) == [2, 8, 4, 6, 0, 10]
        assert list(got.obs["cell_type"]) == sorted(got.obs["cell_type"])
        root = open_input_group(str(tmp_path / f"{name}.zarr"))
        obs = read_elem(root["obs"])
        for s, e in _runs(obs, ["cell_type", "demographic"]):
            block = root["X"][s:e] if name == "dense" else sparse_dataset(root["X"])[s:e].toarray()
            assert block.shape == (e - s, 3) and len(set(obs["cell_type"].iloc[s:e])) == 1
    for key in ("eager.zarr", "lazy.zarr"):
        np.testing.assert_array_equal(read_x(tmp_path / key), read_x(tmp_path / "dense.zarr"))

    with pytest.raises(ConversionError, match="requires layout='csr' or 'dense'"):
        convert_h5ad(
            str(tmp_path / "in.h5ad"),
            output=str(tmp_path / "o1.zarr"),
            cfg=cfg(layout="csc", lazy=False, grouping=grouping),
        )
    with pytest.raises(ConversionError, match="csr"):  # the streamed sort is csr-only
        convert_h5ad(
            str(tmp_path / "in.h5ad"),
            output=str(tmp_path / "o2.zarr"),
            cfg=cfg(layout="dense", lazy=True, grouping=grouping),
        )


def _with_derived(adata: ad.AnnData) -> ad.AnnData:
    # a counts layer, raw, and a cells x cells graph whose entries name their (row, column) pair
    adata = adata.copy()
    n = adata.n_obs
    adata.layers["counts"] = (adata.X * 3).astype(np.float64)
    adata.raw = adata.copy()
    rows = np.repeat(np.arange(n), 3)
    cols = (rows * 7 + np.tile([1, 2, 5], n)) % n
    graph = sp.csr_matrix((rows * 1000.0 + cols, (rows, cols)), shape=(n, n))
    graph.sum_duplicates()
    adata.obsp["distances"] = graph
    return adata


def _assert_sorted_like(got: ad.AnnData, adata: ad.AnnData, layers: tuple[str, ...]) -> None:
    perm = adata.obs_names.get_indexer(got.obs_names)
    want = adata[perm]
    np.testing.assert_array_equal(_dense(got.X), _dense(want.X))
    np.testing.assert_allclose(got.obsm["X_umap"], want.obsm["X_umap"])
    for k in layers:
        np.testing.assert_array_equal(_dense(got.layers[k]), _dense(want.layers[k]), err_msg=k)
    np.testing.assert_array_equal(_dense(got.raw.X), _dense(want.raw.X))
    assert list(got.raw.var_names) == list(adata.raw.var_names)
    np.testing.assert_array_equal(_dense(got.obsp["distances"]), _dense(want.obsp["distances"]))


def test_sort_reorders_every_layer_raw_and_obsp(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    adata = _with_derived(_adata())
    out = _store(tmp_path, adata)
    add_layer(str(out), layer="counts_csc", matrix="layers/counts", layout="csc", cfg=_layer_cfg(32))
    dense_cfg = AppConfig(chunks=ChunkConfig(row_chunk=16, col_chunk=4, cpus=2))
    add_layer(str(out), layer="counts_dense", matrix="layers/counts", layout="dense", cfg=dense_cfg)
    adata.layers["counts_csc"] = adata.layers["counts_dense"] = adata.layers["counts"]
    sorted_out = tmp_path / "sorted.zarr"
    with caplog.at_level("INFO", logger="annizarr"):
        sort(str(out), output=str(sorted_out), by=("cell_type",), cfg=cfg())
    assert any("streamed bucketing" in m for m in messages(caplog))
    assert_anndata_readable(sorted_out)
    got = ad.read_zarr(str(sorted_out))
    assert (np.diff(got.obs["cell_type"].cat.codes.to_numpy()) >= 0).all()
    _assert_sorted_like(got, adata, ("counts", "counts_csc", "counts_dense"))
    root = zarr.open_group(str(sorted_out), mode="r")
    assert root["layers/counts_csc"].attrs["encoding-type"] == "csc_matrix"  # each element keeps its own layout
    assert root["layers/counts_dense"].chunks == (16, 4) and root["layers/counts_csc/data"].chunks == (32,)

    with pytest.raises(ConversionError, match="by="):
        sort(str(out), output=str(tmp_path / "s.zarr"), by=(), cfg=cfg())
    with pytest.raises(StorageError, match="already exists"):
        sort(str(out), output=str(sorted_out), by=("cell_type",), cfg=cfg())


def test_convert_sort_by_carries_layers_raw_and_obsp(tmp_path: Path) -> None:
    adata = _with_derived(_adata(seed=6))
    adata.write_h5ad(tmp_path / "in.h5ad")
    grouping = GroupingConfig(enabled=True, sort_by=("cell_type",))
    for lazy in (True, False):
        out = tmp_path / f"lazy{lazy}.zarr"
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg(lazy=lazy, grouping=grouping))
        assert_anndata_readable(out)
        _assert_sorted_like(ad.read_zarr(str(out)), adata, ("counts",))


def test_append_extends_every_obs_encoding_and_drops_invalidated_obsm(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def build(n: int, seed: int) -> ad.AnnData:
        adata = _adata(n=n, seed=seed)
        rng = np.random.default_rng(seed + 100)
        adata.obs["n_genes"] = rng.integers(0, 1000, n)
        adata.obs["doublet_score"] = rng.random(n)
        adata.obs["sample"] = [f"s{seed}_{i}" for i in range(n)]  # unique per row: stays a string array
        qc = pd.array(rng.integers(0, 5, n), dtype="Int64")
        qc[0] = pd.NA
        adata.obs["qc_flag"] = qc
        return adata

    a, b = build(40, 0), build(15, 1)
    sa, sb = _store(tmp_path, a, "a.zarr"), _store(tmp_path, b, "b.zarr")
    append(str(sa), cells=str(sb), drop_derived=True, cfg=cfg())
    assert any("obsm" in m and "dropped" in m for m in messages(caplog))
    assert_anndata_readable(sa)
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 55 and len(got.obsm) == 0
    np.testing.assert_allclose(got.X.toarray(), sp.vstack([a.X, b.X]).toarray())
    assert list(got.obs_names) == list(a.obs_names) + list(b.obs_names)
    expected_obs = pd.concat([a.obs, b.obs], axis=0)
    # anndata 0.12 reads the nullable index back as pandas 2's pd.NA string dtype, >=0.13 as pandas 3's str
    pd.testing.assert_frame_equal(got.obs, expected_obs, check_dtype=False, check_index_type=False)


def test_append_smaller_than_the_partial_chunk_still_lands(tmp_path: Path) -> None:
    """970 nonzeros at chunk 50 leave room for 30 in the last chunk: an append of exactly 30 is one cut, not none."""
    config = AppConfig(chunks=ChunkConfig(nnz_chunk=50, cpus=1))
    target, cells = _adata(n=97, v=50, seed=0), _adata(n=3, v=50, seed=1)
    t, s = _store(tmp_path, target, "t.zarr", config), _store(tmp_path, cells, "s.zarr", config)
    append(str(t), cells=str(s), drop_derived=True, cfg=config)
    np.testing.assert_array_equal(read_x(t), sp.vstack([target.X, cells.X]).toarray())


def test_append_guards_var_schema_dtype_and_duplicate_names(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    a = _adata(n=20, seed=0)
    a.obsp["conn"] = sp.eye(20, format="csr")
    sa = _store(tmp_path, a, "a.zarr")
    with pytest.raises(ConversionError, match="var mismatch"):  # checked in full before any write
        append(
            str(sa), cells=str(_store(tmp_path, _adata(n=10, v=5, seed=1), "bad.zarr")), drop_derived=True, cfg=cfg()
        )
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 20 and "conn" in got.obsp

    a2, b2 = _adata(n=20, seed=2), _adata(n=10, seed=3)
    a2.obs["cell_type"] = pd.Categorical(a2.obs["cell_type"], categories=["a", "b", "c"], ordered=True)
    with pytest.raises(ConversionError, match="categorical dtype mismatch"):  # ordered categoricals match exactly
        append(str(_store(tmp_path, a2, "a2.zarr")), cells=str(_store(tmp_path, b2, "b2.zarr")), cfg=cfg())
    a3, b3 = _adata(n=20, seed=4), _adata(n=10, seed=5)
    a3.obs["n_genes"], b3.obs["n_genes"] = np.arange(20, dtype=np.int64), np.arange(10, dtype=np.float64)
    with pytest.raises(ConversionError, match="dtype mismatch"):  # columns extend in place, so dtypes must match
        append(str(_store(tmp_path, a3, "a3.zarr")), cells=str(_store(tmp_path, b3, "b3.zarr")), cfg=cfg())

    # every appended name already present (a re-run) is an error, counted once per row even when
    # the cells also repeat each other; a partial overlap is only a warning (barcodes collide across samples)
    with pytest.raises(ConversionError, match="already"):
        append(
            str(_store(tmp_path, _adata(n=20, seed=0), "d.zarr")),
            cells=str(_store(tmp_path, _adata(n=20, seed=0), "e.zarr")),
            drop_derived=True,
            cfg=cfg(),
        )
    a4 = _adata(n=20, seed=1)
    b4 = _adata(n=2, seed=2)
    b4.obs_names = [a4.obs_names[0], a4.obs_names[0]]
    with pytest.raises(ConversionError, match="already"):
        append(
            str(_store(tmp_path, a4, "a4.zarr")),
            cells=str(_store(tmp_path, b4, "b4.zarr")),
            drop_derived=True,
            cfg=cfg(),
        )
    a5, b5 = _adata(n=20, seed=3), _adata(n=10, seed=4)
    b5.obs_names = [a5.obs_names[0], a5.obs_names[1], *b5.obs_names[2:]]
    append(
        str(_store(tmp_path, a5, "a5.zarr")), cells=str(_store(tmp_path, b5, "b5.zarr")), drop_derived=True, cfg=cfg()
    )
    assert any("2 duplicate" in m for m in messages(caplog))


def _layered(n: int, seed: int, *, wide: type) -> ad.AnnData:
    adata = _adata(n=n, seed=seed)
    adata.layers["counts"] = adata.X * 2
    adata.layers["wide"] = (adata.X * 3).astype(wide)
    return adata


def test_append_appends_shared_layers_in_the_store_encoding(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    a, b = _layered(30, 0, wide=np.float64), _layered(12, 1, wide=np.float32)
    b.layers["cells_only"] = b.X
    sa, sb = _store(tmp_path, a, "a.zarr"), _store(tmp_path, b, "b.zarr")
    tiles = AppConfig(chunks=ChunkConfig(row_chunk=8, col_chunk=4, cpus=2))
    level7 = AppConfig(chunks=ChunkConfig(nnz_chunk=32, clevel=7, cpus=2))
    one_chunk = AppConfig(chunks=ChunkConfig(cpus=2))  # 9M-nonzero chunks: one chunk, capped at the layer's size
    by_store = {
        sa: {"by_gene": "csc", "tiles": "dense", "only_store": "csc", "whole": "csc"},
        sb: {"by_gene": "csr", "tiles": "csc", "whole": "csr"},
    }
    for store, layouts in by_store.items():
        for layer, layout in layouts.items():
            layer_cfg = {"tiles": tiles, "whole": one_chunk}.get(layer, level7 if store == sa else _layer_cfg(32))
            add_layer(str(store), layer=layer, matrix="layers/counts", layout=layout, cfg=layer_cfg)
    whole_before = zarr.open_group(str(sa), mode="r")["layers/whole/data"]
    assert whole_before.chunks[0] == whole_before.shape[0]  # one chunk, capped at the layer's own size

    plan = plan_append(str(sa), cells=str(sb))
    assert set(plan.append_layers) == {"counts", "wide", "by_gene", "tiles", "whole"}
    assert plan.drop_layers == ("only_store",)
    assert plan.drops() == ("obsm/X_umap", "layers/only_store")
    assert any("layers ['cells_only']" in n for n in plan.notes)
    with pytest.raises(ConversionError, match="layers/only_store"):  # a store-only layer never drops without consent
        append(str(sa), cells=str(sb), cfg=cfg())
    append(str(sa), cells=str(sb), drop_derived=True, cfg=cfg())
    assert any("add-layer --layer only_store --matrix layers/counts" in m for m in messages(caplog))
    assert any("by_gene is csc: rewriting it whole" in m for m in messages(caplog))
    assert_anndata_readable(sa)

    got = ad.read_zarr(str(sa))
    root = zarr.open_group(str(sa), mode="r")
    assert got.n_obs == 42 and "only_store" not in got.layers and "cells_only" not in got.layers
    counts = sp.vstack([a.layers["counts"], b.layers["counts"]]).toarray()
    for k in ("counts", "by_gene", "tiles", "whole"):
        np.testing.assert_array_equal(_layer(sa, k), counts, err_msg=k)
    np.testing.assert_array_equal(_layer(sa, "wide"), sp.vstack([a.layers["wide"], b.layers["wide"]]).toarray())
    assert got.layers["wide"].dtype == np.float64  # the cells' float32 widened to the store's dtype
    assert root["layers/by_gene"].attrs["encoding-type"] == "csc_matrix" and root["layers/by_gene/data"].chunks == (32,)
    assert root["layers/by_gene"].attrs["annizarr-source"] == "layers/counts"  # kept through the rewrite
    assert [c.clevel for c in root["layers/by_gene/data"].compressors] == [7]  # and so is its codec
    whole = root["layers/whole/data"]  # a chunk only capped by the old size is not kept: the op's nnz_chunk applies
    assert whole.shape[0] > whole_before.shape[0] and whole.chunks[0] == cfg().chunks.nnz_chunk
    assert root["layers/tiles"].chunks == (8, 4)

    c, d = _layered(20, 2, wide=np.float32), _layered(5, 3, wide=np.float64)
    sc, sd = _store(tmp_path, c, "c.zarr"), _store(tmp_path, d, "d.zarr")
    with pytest.raises(ConversionError, match="layers/wide: the cells hold float64"):  # lossy: refused up front
        append(str(sc), cells=str(sd), drop_derived=True, cfg=cfg())
    assert ad.read_zarr(str(sc)).n_obs == 20
    e, f = _layered(20, 4, wide=np.float64), _layered(5, 5, wide=np.int64)
    se, sf = _store(tmp_path, e, "e.zarr"), _store(tmp_path, f, "f.zarr")
    with pytest.raises(ConversionError, match="layers/wide: the cells hold int64"):  # numpy says safe; > 2**53 rounds
        append(str(se), cells=str(sf), drop_derived=True, cfg=cfg())


def test_append_reads_anndata_written_stores_and_keeps_the_store_column_encoding(tmp_path: Path) -> None:
    """The store column's encoding wins: a categorical gains categories for unseen values whether
    the cells arrive categorical or plain, a string column takes categorical cells as strings, an
    ordered categorical must match exactly, and anndata's nullable-string index is handled."""
    a, b, c = _adata(n=6, seed=0), _adata(n=4, seed=1), _adata(n=3, seed=2)
    for x in (a, b, c):
        x.obsm.clear()
    a.obs["batch"] = pd.Categorical(["b1", "b2", "b1", "b2", "b1", "b2"])
    a.obs["sample"] = [f"s{i}" for i in range(6)]
    b.obs["batch"] = pd.Categorical(["b3", "b2", "b3", "b2"])
    b.obs["sample"] = pd.Categorical(["s9", "s9", "s8", "s8"])
    c.obs["batch"] = ["b4", "b5", "b6"]
    c.obs["sample"] = [f"t{i}" for i in range(3)]
    sa, sb, sc = (_store(tmp_path, x, f"{n}.zarr") for x, n in ((a, "a"), (b, "b"), (c, "c")))
    assert any("'batch': 1 new category appended (b3)" in n for n in plan_append(str(sa), cells=str(sb)).notes)
    append(str(sa), cells=str(sb), cfg=cfg())
    append(str(sa), cells=str(sc), cfg=cfg())
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 13 and list(got.obs["batch"].cat.categories) == ["b1", "b2", "b3", "b4", "b5", "b6"]
    assert list(got.obs["batch"]) == [*a.obs["batch"], *b.obs["batch"], *c.obs["batch"]]
    assert got.obs["sample"].dtype != "category" and list(got.obs["sample"]) == [
        *a.obs["sample"],
        *b.obs["sample"],
        *c.obs["sample"],
    ]

    d, e = _adata(n=4, seed=3), _adata(n=2, seed=4)
    d.obs["grade"] = pd.Categorical(["lo", "hi", "lo", "hi"], categories=["lo", "hi"], ordered=True)
    e.obs["grade"] = pd.Categorical(["mid", "hi"], categories=["lo", "mid", "hi"], ordered=True)
    with pytest.raises(ConversionError, match="ordered categorical"):
        plan_append(str(_store(tmp_path, d, "d.zarr")), cells=str(_store(tmp_path, e, "e.zarr")))

    f, g = _adata(n=8, seed=0), _adata(n=3, seed=1)
    for x in (f, g):
        x.obs.index = x.obs.index.astype("string")
        x.obsm.clear()
    previous = ad.settings.allow_write_nullable_strings
    ad.settings.allow_write_nullable_strings = True
    try:
        f.write_zarr(tmp_path / "f.zarr")
        g.write_zarr(tmp_path / "g.zarr")
    finally:
        ad.settings.allow_write_nullable_strings = previous
    assert not isinstance(zarr.open_group(str(tmp_path / "f.zarr"), mode="r")["obs/_index"], zarr.Array)
    append(str(tmp_path / "f.zarr"), cells=str(tmp_path / "g.zarr"), cfg=cfg())
    assert list(ad.read_zarr(str(tmp_path / "f.zarr")).obs_names) == [*f.obs_names, *g.obs_names]


def test_append_widens_categorical_codes_that_outgrow_the_stored_width(tmp_path: Path) -> None:
    """A categorical written with int8 codes keeps growing: the codes array is replaced at the next
    signed width, the plan says so first, and the column reads back whole."""
    a, b = _adata(n=250, seed=0), _adata(n=10, seed=1)
    for x in (a, b):
        x.obsm.clear()
    a.obs["sample"] = pd.Categorical([f"s{i // 2}" for i in range(250)])  # 125 categories -> int8 codes
    b.obs["sample"] = pd.Categorical([f"t{i // 2}" for i in range(10)])
    sa, sb = _store(tmp_path, a, "a.zarr"), _store(tmp_path, b, "b.zarr")
    assert zarr.open_group(str(sa), mode="r")["obs/sample/codes"].dtype == np.int8

    assert any("codes widen int8 → int16" in n for n in plan_append(str(sa), cells=str(sb)).notes)
    append(str(sa), cells=str(sb), cfg=cfg())
    assert zarr.open_group(str(sa), mode="r")["obs/sample/codes"].dtype == np.int16
    assert_anndata_readable(sa)
    got = ad.read_zarr(str(sa))
    assert len(got.obs["sample"].cat.categories) == 130
    assert list(got.obs["sample"]) == [*a.obs["sample"], *b.obs["sample"]]


def test_lifecycle_on_plain_and_consolidated_stores(tmp_path: Path) -> None:
    a, b = _adata(n=40, seed=0), _adata(n=15, seed=1)
    sa = _store(tmp_path, a, "a.zarr", cfg(consolidate_metadata=True))
    sb = _store(tmp_path, b, "b.zarr")
    sorted1 = tmp_path / "sorted1.zarr"
    sort(str(sa), output=str(sorted1), by=("cell_type",), cfg=cfg())
    append(str(sorted1), cells=str(sb), drop_derived=True, cfg=cfg())
    add_layer(str(sorted1), layer="csc", cfg=_layer_cfg(32))
    sort(str(sorted1), output=str(tmp_path / "sorted2.zarr"), by=("cell_type",), cfg=cfg())
    got = ad.read_zarr(str(tmp_path / "sorted2.zarr"))
    assert got.n_obs == 55 and "csc" in got.layers
    assert (np.diff(got.obs["cell_type"].cat.codes.to_numpy()) >= 0).all()
    by_name = {n: r for src in (a, b) for n, r in zip(src.obs_names, src.X.toarray(), strict=True)}
    for i, n in enumerate(got.obs_names):
        np.testing.assert_allclose(got.X[i].toarray().ravel(), by_name[n])
    np.testing.assert_array_equal(got.layers["csc"].toarray(), got.X.toarray())
    # in-place ops on the consolidated store, whose workers reopen it without the stale metadata
    add_layer(str(sa), layer="csc", cfg=_layer_cfg(32))
    append(str(sa), cells=str(sb), drop_derived=True, cfg=cfg())
    assert ad.read_zarr(str(sa)).n_obs == 55


def test_append_reconciles_obs_column_sets(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A store column the cells lack gets missing values; a cells-only column is left out unless
    include_new_obs adds it full height; a plain string column the cells lack is refused."""
    a, b = _adata(n=8, seed=0), _adata(n=4, seed=1)
    for x in (a, b):
        x.obsm.clear()
    a.obs["score"] = np.linspace(0, 1, 8)
    a.obs["donor"] = pd.Categorical(["d1", "d2"] * 4)
    b.obs["batch"] = pd.Categorical(["x", "y", "x", "y"])
    plan = plan_append(str(_store(tmp_path, a, "p.zarr")), cells=str(_store(tmp_path, b, "b.zarr")))
    assert plan.obs_extra == ("batch",) and plan.obs_filled == ("score", "donor") and plan.obs_promoted == ()

    sa = _store(tmp_path, a, "a.zarr")
    append(str(sa), cells=str(tmp_path / "b.zarr"), cfg=cfg())
    assert any("not in the cells: ['score', 'donor']" in m for m in messages(caplog))
    assert any("['batch'] not appended" in m for m in messages(caplog))
    assert_anndata_readable(sa)
    got = ad.read_zarr(str(sa))
    assert "batch" not in got.obs and got.obs["score"].iloc[8:].isna().all() and got.obs["donor"].iloc[8:].isna().all()
    np.testing.assert_allclose(got.obs["score"].iloc[:8], a.obs["score"])

    sn = _store(tmp_path, a, "n.zarr")
    append(str(sn), cells=str(tmp_path / "b.zarr"), include_new_obs=True, cfg=cfg())
    assert zarr.open_group(str(sn), mode="r")["obs"].attrs["column-order"][-1] == "batch"
    got = ad.read_zarr(str(sn))
    assert got.obs["batch"].iloc[:8].isna().all() and list(got.obs["batch"].iloc[8:]) == list(b.obs["batch"])

    ss = _store(tmp_path, _adata(n=8, seed=2), "s.zarr")
    obs = zarr.open_group(str(ss), mode="r+")["obs"]
    write_elem(obs, "uuid", np.array([f"u{i}" for i in range(8)], dtype=object))
    obs.attrs["column-order"] = [*obs.attrs["column-order"], "uuid"]
    assert obs["uuid"].attrs["encoding-type"] == "string-array"
    with pytest.raises(ConversionError, match="'uuid': the cells do not carry it, and a plain string-array"):
        plan_append(str(ss), cells=str(tmp_path / "b.zarr"))


def test_append_reconciles_nullable_obs_encodings(tmp_path: Path) -> None:
    """Plain int/bool and their nullable encodings append onto each other; a plain store column that
    must hold a missing value is promoted to its nullable form, with its values intact."""
    a, b = _adata(n=6, seed=0), _adata(n=3, seed=1)
    for x in (a, b):
        x.obsm.clear()
    a.obs["n"] = np.arange(6, dtype=np.int32)
    b.obs["n"] = pd.array([7, None, 9], dtype="Int32")
    a.obs["ok"] = pd.array([True, False] * 3, dtype="boolean")
    b.obs["ok"] = np.array([True, True, False])
    a.obs["keep"] = np.array([True] * 6)
    b.obs["keep"] = pd.array([False, None, True], dtype="boolean")
    a.obs["m"] = np.arange(6, dtype=np.int64)
    b.obs["m"] = pd.array([1, 2, 3], dtype="Int64")
    sa, sb = _store(tmp_path, a, "a.zarr"), _store(tmp_path, b, "b.zarr")
    assert plan_append(str(sa), cells=str(sb)).obs_promoted == ("n", "keep")

    append(str(sa), cells=str(sb), cfg=cfg())
    assert_anndata_readable(sa)
    obs = zarr.open_group(str(sa), mode="r")["obs"]
    encodings = {k: obs[k].attrs["encoding-type"] for k in ("n", "keep", "m")}
    assert encodings == {"n": "nullable-integer", "keep": "nullable-boolean", "m": "array"}
    got = ad.read_zarr(str(sa)).obs
    assert got["n"].dtype == "Int32" and list(got["n"]) == [0, 1, 2, 3, 4, 5, 7, pd.NA, 9]
    assert list(got["ok"]) == [True, False] * 3 + [True, True, False]
    assert list(got["keep"]) == [True] * 6 + [False, pd.NA, True]
    assert list(got["m"]) == [0, 1, 2, 3, 4, 5, 1, 2, 3]
