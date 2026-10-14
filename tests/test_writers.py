"""Parallel writers: cpus=1 and cpus=4 write byte-identical stores, shards are respected, and the
write-unit guard speaks up."""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

import annizarr._core._layout as _layout
from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.ops import append, concat, convert_adata, convert_h5ad, rechunk
from helpers import assert_anndata_readable, assert_byte_identical, make_labelled_h5ad, random_dense, read_x

N_OBS, N_VARS = 300, 200  # row_chunk 64 and col_chunk 48 put several blocks, with a ragged one, on each axis
SHARD_ROW_CHUNK, SHARD_COL_CHUNK, SHARD_FACTOR = 32, 48, 2
AUTOSHARD_FLAT_CHUNK, AUTOSHARD_CHUNKS_PER_SHARD = 500, 4


def _adata(dense: np.ndarray, *, sparse: bool, seed: int) -> ad.AnnData:
    obs = pd.DataFrame({"batch": ["b"] * dense.shape[0]}, index=[f"{seed}_{i}" for i in range(dense.shape[0])])
    var = pd.DataFrame(index=[f"g{i}" for i in range(dense.shape[1])])
    return ad.AnnData(X=sp.csr_matrix(dense) if sparse else dense.copy(), obs=obs, var=var)


def _cfg(
    layout: str, *, cpus: int, row_chunk: int = 64, col_chunk: int = 48, flat_chunk: int = 500, shard_factor: int = 1
) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, layout=layout),
        chunks=ChunkConfig(
            row_chunk=row_chunk, col_chunk=col_chunk, nnz_chunk=flat_chunk, cpus=cpus, shard_factor=shard_factor
        ),
    )


def _autoshard_cfg(*, cpus: int) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, layout="csr"),
        chunks=ChunkConfig(nnz_chunk=AUTOSHARD_FLAT_CHUNK, cpus=cpus, auto_shard=True),
    )


def _forced_autoshard():
    # pins zarr's shards="auto" heuristic to exactly AUTOSHARD_CHUNKS_PER_SHARD chunks per shard
    return zarr.config.set({"array.target_shard_size_bytes": AUTOSHARD_FLAT_CHUNK * 4 * AUTOSHARD_CHUNKS_PER_SHARD})


def _n_objects(store: Path, array: str) -> int:
    return sum(1 for p in (store / array / "c").rglob("*") if p.is_file())


def _concat_inputs(tmp_path: Path, n_vars: int, seed0: int) -> tuple[list[str], np.ndarray]:
    # 70 + 130 + 100 rows seam at 70 and 200, neither a multiple of row_chunk 64 nor of the 64-row shard
    parts = [random_dense(n, n_vars, seed=seed0 + i) for i, n in enumerate((70, 130, 100))]
    paths = []
    for i, part in enumerate(parts):
        _adata(part, sparse=True, seed=seed0 + i).write_h5ad(tmp_path / f"in{i}.h5ad")
        paths.append(str(tmp_path / f"in{i}.h5ad"))
    return paths, np.vstack(parts)


@pytest.mark.parametrize(
    ("path", "layout", "sparse_source"),
    [
        pytest.param("direct", "dense", False, id="dense_from_dense"),
        pytest.param("direct", "dense", True, id="sparse_to_dense"),
        pytest.param("direct", "csr", True, id="sparse_to_csr"),
        pytest.param("direct", "csc", True, id="sparse_to_csc"),
        pytest.param("concat", "dense", True, id="dense_concat"),
        pytest.param("concat", "csr", True, id="csr_concat"),
    ],
)
def test_cpus1_matches_cpus4_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, layout: str, sparse_source: bool
) -> None:
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1)  # as many bands as there are write units
    out1, out4 = tmp_path / "cpus1.zarr", tmp_path / "cpus4.zarr"
    if path == "direct":
        dense = random_dense(N_OBS, N_VARS, seed=3, density=0.25)
        adata = _adata(dense, sparse=sparse_source, seed=3)
        convert_adata(adata, output=out1, cfg=_cfg(layout, cpus=1))
        convert_adata(adata, output=out4, cfg=_cfg(layout, cpus=4))
    else:
        paths, dense = _concat_inputs(tmp_path, 96, 10)
        concat(paths, output=out1, cfg=_cfg(layout, cpus=1, col_chunk=96))
        concat(paths, output=out4, cfg=_cfg(layout, cpus=4, col_chunk=96))
    assert_byte_identical(out1, out4)
    np.testing.assert_array_equal(read_x(out1), dense)
    assert_anndata_readable(out1)


def test_dense_shards_pack_chunks_and_survive_concat_seams_and_lazy_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1)
    paths, expected = _concat_inputs(tmp_path, N_VARS, 30)
    expected_shards = (SHARD_ROW_CHUNK * SHARD_FACTOR, SHARD_COL_CHUNK * SHARD_FACTOR)
    outs: dict[int, Path] = {}
    for cpus in (1, 4):
        out = tmp_path / f"out_cpus{cpus}.zarr"
        concat(
            paths,
            output=out,
            cfg=_cfg(
                "dense", cpus=cpus, row_chunk=SHARD_ROW_CHUNK, col_chunk=SHARD_COL_CHUNK, shard_factor=SHARD_FACTOR
            ),
        )
        np.testing.assert_array_equal(read_x(out), expected)
        x = zarr.open_group(str(out), mode="r")["X"]
        assert x.chunks == (SHARD_ROW_CHUNK, SHARD_COL_CHUNK) and x.shards == expected_shards
        assert _n_objects(out, "X") == math.ceil(300 / expected_shards[0]) * math.ceil(N_VARS / expected_shards[1])
        outs[cpus] = out
    assert_byte_identical(outs[1], outs[4])

    # a lazy h5 input fans out across worker processes, each writing whole shards
    make_labelled_h5ad(tmp_path / "small.h5ad")
    small_cfg = AppConfig(
        io=IOConfig(layout="dense", lazy=True), chunks=ChunkConfig(row_chunk=2, col_chunk=2, shard_factor=2, cpus=2)
    )
    convert_h5ad(str(tmp_path / "small.h5ad"), output=str(tmp_path / "small.zarr"), cfg=small_cfg)
    x = zarr.open_group(str(tmp_path / "small.zarr"), mode="r")["X"]
    assert x.chunks == (2, 2) and x.shards == (4, 4)
    assert np.array_equal(read_x(tmp_path / "small.zarr")[:, 0], np.arange(1, 7))
    # sharding is a dense setting: sparse arrays stay unsharded and the config says so
    convert_h5ad(
        str(tmp_path / "small.h5ad"),
        output=str(tmp_path / "sparse.zarr"),
        cfg=AppConfig(chunks=ChunkConfig(shard_factor=4, cpus=2)),
    )
    assert all(
        zarr.open_group(str(tmp_path / "sparse.zarr"), mode="r")["X"][n].shards is None
        for n in ("data", "indices", "indptr")
    )


def test_autosharded_sparse_arrays_are_written_extended_and_rechunked_on_the_shard_grid(tmp_path: Path) -> None:
    """With zarr's shard inference pinned to 4 chunks per shard, convert, append and rechunk all
    partition their writes on the shard, so cpus=1 and cpus=4 stay byte-identical with one object per shard."""
    dense = random_dense(N_OBS, N_VARS, seed=11, density=0.25)
    adata = _adata(dense, sparse=True, seed=11)
    more = _adata(random_dense(40, N_VARS, seed=22, density=0.25), sparse=True, seed=22)
    shard = AUTOSHARD_FLAT_CHUNK * AUTOSHARD_CHUNKS_PER_SHARD
    previous_setting = ad.settings.auto_shard_zarr_v3
    outs: dict[str, dict[int, Path]] = {"convert": {}, "append": {}, "rechunk": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # neither anndata's nor zarr's sharding warnings may escape
        with _forced_autoshard():
            convert_adata(more, output=tmp_path / "cells.zarr", cfg=_autoshard_cfg(cpus=1))
            convert_adata(adata, output=tmp_path / "plain.zarr", cfg=_cfg("csr", cpus=1, flat_chunk=64))
            for cpus in (1, 4):
                convert_adata(adata, output=tmp_path / f"c{cpus}.zarr", cfg=_autoshard_cfg(cpus=cpus))
                outs["convert"][cpus] = tmp_path / f"c{cpus}.zarr"
                convert_adata(adata, output=tmp_path / f"a{cpus}.zarr", cfg=_autoshard_cfg(cpus=cpus))
                append(
                    tmp_path / f"a{cpus}.zarr",
                    cells=tmp_path / "cells.zarr",
                    cfg=AppConfig(chunks=ChunkConfig(cpus=cpus)),
                )
                outs["append"][cpus] = tmp_path / f"a{cpus}.zarr"
                rechunk(tmp_path / "plain.zarr", output=tmp_path / f"r{cpus}.zarr", cfg=_autoshard_cfg(cpus=cpus))
                outs["rechunk"][cpus] = tmp_path / f"r{cpus}.zarr"
    assert ad.settings.auto_shard_zarr_v3 is previous_setting

    for op, by_cpus in outs.items():
        expected = np.vstack([dense, read_x(tmp_path / "cells.zarr")]) if op == "append" else dense
        for out in by_cpus.values():
            np.testing.assert_array_equal(read_x(out), expected)
            assert_anndata_readable(out)
            root = zarr.open_group(str(out), mode="r")
            assert root["X/data"].shards == (shard,) and root["X/indices"].shards == (shard,)
            if op != "append":
                assert _n_objects(out, "X/data") == math.ceil(adata.X.nnz / shard)
        assert_byte_identical(by_cpus[1], by_cpus[4])


def test_write_unit_above_the_band_budget_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="annizarr")
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1024)
    adata = _adata(random_dense(64, 64, seed=3, density=0.5), sparse=True, seed=3)
    convert_adata(adata, output=tmp_path / "big_unit.zarr", cfg=_cfg("csr", cpus=1, flat_chunk=2000))
    assert any("above the 0MB band budget" in r.message for r in caplog.records)


def test_string_columns_are_written_as_nullable_string_arrays(tmp_path: Path) -> None:
    # the encoding must not follow whichever dtype the installed pandas infers for text
    obs = pd.DataFrame(
        {
            "barcode": [f"bc{i}" for i in range(8)],  # all distinct: stays a string
            "batch": ["a", "a", "b", "b", "c", "c", "d", "d"],  # repeats: becomes a categorical
            "sparse_label": pd.Series([None, *[f"s{i}" for i in range(7)]], index=[str(i) for i in range(8)]),
        },
        index=[str(i) for i in range(8)],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(random_dense(8, 6, seed=11)), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(6)])
    )
    out = tmp_path / "strings.zarr"
    convert_adata(adata, output=out, cfg=_cfg("csr", cpus=2))

    obs_group = zarr.open_group(str(out), mode="r")["obs"]
    assert obs_group["_index"].attrs["encoding-type"] == "nullable-string-array"
    assert obs_group["barcode"].attrs["encoding-type"] == "nullable-string-array"
    assert obs_group["batch"].attrs["encoding-type"] == "categorical"
    back = ad.read_zarr(str(out)).obs
    assert isinstance(back["barcode"].dtype, pd.StringDtype)
    assert back["barcode"].tolist() == obs["barcode"].tolist()
    assert back.index.tolist() == obs.index.tolist()
    assert back["sparse_label"].isna().tolist() == obs["sparse_label"].isna().tolist()
    assert_anndata_readable(out)
