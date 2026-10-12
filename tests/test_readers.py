"""Every reader presents the same contract over the same 6 x 5 fixture, which has an empty row (2)
and an empty column (3) so indptr, band and flat math see real sparsity."""

from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

import annizarr._core._layout as _layout
from annizarr._sources._readers import (
    ConcatReader,
    CSRFileReader,
    CSRH5Reader,
    CSRMemoryReader,
    CSRZarrReader,
    DenseH5Reader,
    DenseMemoryReader,
    DenseZarrReader,
    ScaledReader,
    TenxCSRReader,
    as_reader,
)
from annizarr._writers._buckets import Buckets
from annizarr.config import AppConfig
from annizarr.errors import ConversionError
from helpers import expected_lognorm, make_10x_v2_h5, make_10x_v3_h5

DENSE = np.array(
    [[1, 0, 2, 0, 3], [0, 4, 0, 0, 5], [0, 0, 0, 0, 0], [6, 0, 7, 0, 0], [0, 8, 0, 0, 9], [10, 0, 0, 0, 11]],
    dtype=np.float32,
)
CSR = sp.csr_matrix(DENSE)


def _csr_group(group: zarr.Group, mat: sp.csr_matrix) -> zarr.Group:
    g = group.require_group("X")
    g.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": list(mat.shape)})
    for name, arr, dtype in (("data", mat.data, mat.data.dtype), ("indices", mat.indices, mat.indices.dtype)):
        g.require_array(name, shape=arr.shape, dtype=dtype, chunks=arr.shape or (1,))[:] = arr
    g.require_array("indptr", shape=mat.indptr.shape, dtype=np.int64, chunks=mat.indptr.shape)[:] = mat.indptr
    return g


def _dense_h5(tmp_path: Path) -> DenseH5Reader:
    with h5py.File(tmp_path / "dense.h5", "w") as f:
        f.create_dataset("X", data=DENSE, chunks=(2, DENSE.shape[1]))
    return DenseH5Reader(str(tmp_path / "dense.h5"), "X")


def _csr_h5(tmp_path: Path) -> CSRH5Reader:
    ad.AnnData(X=CSR.copy()).write_h5ad(tmp_path / "in.h5ad")
    return CSRH5Reader(str(tmp_path / "in.h5ad"), "/X")


def _dense_zarr(tmp_path: Path) -> DenseZarrReader:
    group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
    arr = group.require_array("X", shape=DENSE.shape, dtype=DENSE.dtype, chunks=(2, 5))
    arr[:] = DENSE
    return DenseZarrReader(arr)


def _csr_zarr(tmp_path: Path) -> CSRZarrReader:
    return CSRZarrReader(_csr_group(zarr.open_group(str(tmp_path / "csr.zarr"), mode="w"), CSR))


BUILDERS = {
    "dense_memory": lambda tmp_path: DenseMemoryReader(DENSE.copy()),
    "dense_h5": _dense_h5,
    "dense_zarr": _dense_zarr,
    "csr_memory": lambda tmp_path: CSRMemoryReader(CSR.copy()),
    "csr_h5": _csr_h5,
    "csr_zarr": _csr_zarr,
}


@pytest.fixture(autouse=True)
def _tiny_batch_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_layout, "BATCH_BYTES", 24)  # several bands even on this tiny fixture


def _check_contract(reader: object) -> None:
    assert reader.shape == DENSE.shape and reader.dtype == DENSE.dtype  # type: ignore[attr-defined]
    np.testing.assert_array_equal(reader.dense_block(1, 4, 1, 4), DENSE[1:4, 1:4])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(reader.csr_rows(1, 4).toarray(), DENSE[1:4])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(np.diff(reader.indptr), np.count_nonzero(DENSE, axis=1))  # type: ignore[attr-defined]
    data, idx = reader.flat(3, 8)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(data, CSR.data[3:8])
    np.testing.assert_array_equal(idx, CSR.indices[3:8])
    np.testing.assert_array_equal(reader.flat_indices(0, CSR.nnz), CSR.indices)  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_reader_contract(tmp_path: Path, kind: str) -> None:
    reader = BUILDERS[kind](tmp_path)
    assert reader.lazy == ("memory" not in kind)
    _check_contract(reader)
    if reader.lazy:  # worker processes get a pickled copy that reopens its own handle
        restored = pickle.loads(pickle.dumps(reader))
        _check_contract(restored)
        restored.close()
    reader.close()


def test_h5_readers_report_source_chunking_as_grain(tmp_path: Path) -> None:
    assert _dense_h5(tmp_path).grain == 2
    with h5py.File(tmp_path / "flat.h5", "w") as f:
        f.create_dataset("X", data=DENSE)  # contiguous: no grain
    assert DenseH5Reader(str(tmp_path / "flat.h5"), "X").grain is None
    with h5py.File(tmp_path / "csr.h5", "w") as f:
        g = f.create_group("X")
        g.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": list(CSR.shape)})
        g.create_dataset("data", data=CSR.data, chunks=(3,))
        g.create_dataset("indices", data=CSR.indices, chunks=(3,))
        g.create_dataset("indptr", data=CSR.indptr)
    assert CSRH5Reader(str(tmp_path / "csr.h5"), "X").grain == 3


def test_concat_reader_stitches_across_a_seam_inside_a_band_and_a_flat_segment(tmp_path: Path) -> None:
    bottom = _csr_group(zarr.open_group(str(tmp_path / "bottom.zarr"), mode="w"), sp.csr_matrix(DENSE[3:]))
    reader = ConcatReader([CSRMemoryReader(sp.csr_matrix(DENSE[:3])), CSRZarrReader(bottom)])
    assert reader.shape == DENSE.shape
    np.testing.assert_array_equal(np.diff(reader.indptr), np.count_nonzero(DENSE, axis=1))
    np.testing.assert_array_equal(reader.dense_block(1, 5, 0, 5), DENSE[1:5])  # the row seam is inside this band
    np.testing.assert_array_equal(reader.csr_rows(1, 5).toarray(), DENSE[1:5])
    data, idx = reader.flat(3, 8)  # the seam's nnz offset (5) is inside this segment
    np.testing.assert_array_equal(data, CSR.data[3:8])
    np.testing.assert_array_equal(idx, CSR.indices[3:8])
    np.testing.assert_array_equal(reader.flat_indices(3, 8), CSR.indices[3:8])
    reader.close()


def test_csr_band_tiles_slice_columns_before_densifying(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, int]] = []
    real = sp.csr_matrix.toarray

    def recording(self: sp.csr_matrix, *args: object, **kwargs: object) -> np.ndarray:
        seen.append(self.shape)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", recording)
    band = CSRMemoryReader(CSR.copy()).dense_band(0, 6)
    np.testing.assert_array_equal(np.hstack([band.tile(0, 2), band.tile(2, 5)]), DENSE)
    assert seen == [(6, 2), (6, 3)]


def test_as_reader_rejects_non_matrix_inputs() -> None:
    cfg = AppConfig()
    with pytest.raises(ConversionError, match="1-D"):
        as_reader(np.zeros(5, dtype=np.float32), cfg=cfg)
    with pytest.raises(ConversionError, match="coo"):
        as_reader(sp.coo_matrix(DENSE), cfg=cfg)
    with pytest.raises(ConversionError, match="masked"):
        as_reader(np.ma.masked_array(DENSE), cfg=cfg)
    assert as_reader(DENSE, cfg=cfg).cpus == cfg.chunks.cpus


@pytest.mark.parametrize("kind", ["csr_memory", "csr_h5", "csr_zarr"])
def test_scaled_reader_applies_per_row_factors_on_every_access_path(tmp_path: Path, kind: str) -> None:
    """Flat segments that start and end mid-row, csr rows and dense tiles all match the formula,
    including the empty row (factor 0) and the empty column."""
    from annizarr.ops._expr import lognorm_factors

    inner = BUILDERS[kind](tmp_path)
    inner.cpus = 2
    reader = ScaledReader(inner, lognorm_factors(inner, 100.0))
    expected = expected_lognorm(DENSE, 100.0)
    assert reader.shape == CSR.shape and reader.dtype == np.float32 and reader.lazy == inner.lazy
    assert reader.band_bytes_per_nnz == _layout.LOGNORM_BYTES_PER_NNZ
    np.testing.assert_allclose(reader.csr_rows(0, 6).toarray(), expected, rtol=1e-6)
    np.testing.assert_allclose(reader.dense_band(0, 6).tile(1, 4), expected[:, 1:4], rtol=1e-6)
    flat_expected = sp.csr_matrix(expected).data
    for s0, s1 in [(0, CSR.nnz), (1, 4), (3, 9), (8, 11), (5, 5)]:
        data, idx = reader.flat(s0, s1)
        np.testing.assert_allclose(data, flat_expected[s0:s1], rtol=1e-6)
        np.testing.assert_array_equal(idx, CSR.indices[s0:s1])
        np.testing.assert_allclose(reader.flat_data(s0, s1), flat_expected[s0:s1], rtol=1e-6)
    reader.close()


def test_tenx_reader_matches_the_eager_loader_with_and_without_a_feature_mask(tmp_path: Path) -> None:
    from annizarr._sources._tenx import load_10x_h5, load_10x_lazy

    for path in (
        make_10x_v3_h5(tmp_path / "v3.h5", include_non_gex=True, int32_data=True),
        make_10x_v2_h5(tmp_path / "v2.h5"),
    ):
        adata, reader = load_10x_lazy(path)
        eager = load_10x_h5(path)
        assert isinstance(reader, TenxCSRReader) and adata.X is None
        assert adata.shape == eager.shape == reader.shape == (2, 3)
        assert list(adata.var_names) == list(eager.var_names) and list(adata.obs_names) == list(eager.obs_names)
        assert reader.dtype == np.float32  # int32 counts widen like the eager loader's
        np.testing.assert_array_equal(reader.indptr, eager.X.indptr)
        np.testing.assert_array_equal(reader.csr_rows(0, 2).toarray(), eager.X.toarray())
        for s0, s1 in [(0, 3), (1, 2), (2, 3), (0, 1)]:
            data, idx = reader.flat(s0, s1)
            np.testing.assert_array_equal(data, eager.X.data[s0:s1])
            np.testing.assert_array_equal(idx, eager.X.indices[s0:s1])
        clone = pickle.loads(pickle.dumps(reader))  # a worker's copy keeps the mask it needs
        np.testing.assert_array_equal(clone.flat(0, 3)[0], eager.X.data)
        clone.close()
        reader.close()


def test_buckets_and_file_reader(tmp_path: Path) -> None:
    """Pieces land at offsets or append in order; a CSRFileReader reads rows and ranges back from the files."""
    fields = {"data": np.dtype(np.float32), "indices": np.dtype(np.int32)}
    buckets = Buckets.create(2, fields, dir=str(tmp_path), prefix="bk_", sizes=[4, 0])
    buckets.write_at(0, 2, data=np.array([3, 4], dtype=np.float32), indices=np.array([0, 1], dtype=np.int32))
    buckets.write_at(0, 0, data=np.array([1, 2], dtype=np.float64), indices=np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(buckets.read(0, "data"), [1, 2, 3, 4])
    assert buckets.read(1, "data").size == 0 and buckets.read(0, "indices").dtype == np.int32
    appended = Buckets.create(1, fields, dir=str(tmp_path), prefix="ap_")
    appended.append(0, data=np.array([1], dtype=np.float32), indices=np.array([2], dtype=np.int32))
    appended.append(0, data=np.array([2], dtype=np.float32), indices=np.array([0], dtype=np.int32))
    assert appended.counts == [2]
    np.testing.assert_array_equal(appended.read(0, "indices"), [2, 0])
    # rows [a . b], [. . .], [c d .] from the first bucket's files
    reader = CSRFileReader(
        buckets.path(0, "data"), buckets.path(0, "indices"), np.array([0, 2, 2, 4]), (3, 3), np.dtype(np.float32)
    )
    assert reader.thread_safe and reader.lazy and reader.nnz == 4
    np.testing.assert_array_equal(reader.csr_rows(0, 3).toarray(), [[1, 0, 2], [0, 0, 0], [3, 4, 0]])
    np.testing.assert_array_equal(reader.flat(1, 3)[0], [2, 3])
    np.testing.assert_array_equal(reader.dense_band(0, 3).tile(1, 3), [[0, 2], [0, 0], [4, 0]])
    buckets.remove()
    appended.remove()
    assert not buckets.root.exists()
