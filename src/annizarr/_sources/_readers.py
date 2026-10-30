from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from annizarr._core import _layout
from annizarr._core._runtime import map_parallel
from annizarr._core._zarr import get_array, shape_attr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Literal

    import scipy.sparse as sp
    import zarr
    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CSRFileReader",
    "CSRH5Reader",
    "CSRMemoryReader",
    "CSRZarrReader",
    "CastReader",
    "ConcatReader",
    "DenseBand",
    "DenseH5Reader",
    "DenseMemoryReader",
    "DenseZarrReader",
    "Reader",
    "RemapColumnsReader",
    "TenxCSRReader",
    "as_reader",
]


class DenseBand(Protocol):
    def tile(self, c0: int, c1: int) -> NDArray[Any]: ...


class _ArrayBand:
    __slots__ = ("_rows",)

    def __init__(self, rows: Any) -> None:
        self._rows = rows

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._rows[:, c0:c1])


class _TiledBand:
    # each tile read straight from the lazy array: used when a whole row band would exceed BATCH_BYTES
    __slots__ = ("_arr", "_r0", "_r1")

    def __init__(self, arr: Any, r0: int, r1: int) -> None:
        self._arr = arr
        self._r0 = r0
        self._r1 = r1

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._arr[self._r0 : self._r1, c0:c1])


class _CSRBand:
    __slots__ = ("_rows",)

    def __init__(self, rows: sp.csr_matrix) -> None:
        self._rows = rows

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        sub = self._rows if (c0 == 0 and c1 == self._rows.shape[1]) else self._rows[:, c0:c1]
        return np.asarray(sub.toarray())


class _StitchedBand:
    __slots__ = ("_dtype", "_n_rows", "_parts")

    def __init__(self, parts: list[tuple[int, int, DenseBand]], n_rows: int, dtype: np.dtype[Any]) -> None:
        self._parts = parts
        self._n_rows = n_rows
        self._dtype = dtype

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        out = np.empty((self._n_rows, c1 - c0), dtype=self._dtype)
        for lo, hi, band in self._parts:
            out[lo:hi] = band.tile(c0, c1)
        return out


class Reader(ABC):
    def __init__(self) -> None:
        self._indptr_cache: NDArray[np.int64] | None = None
        self.cpus = 1

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]: ...

    @property
    @abstractmethod
    def lazy(self) -> bool: ...

    @property
    @abstractmethod
    def thread_safe(self) -> bool: ...

    @property
    def grain(self) -> int | None:
        # the source's own chunk size along the read axis: rows of one source chunk for a
        # dense reader, flat elements of one source chunk of `data` for a CSR-like reader.
        # None when the source isn't chunked (memory readers, a concat of mixed sources).
        return None

    @property
    def indptr(self) -> NDArray[np.int64]:
        if self._indptr_cache is None:
            self._indptr_cache = self._compute_indptr()
        return self._indptr_cache

    @abstractmethod
    def _compute_indptr(self) -> NDArray[np.int64]: ...

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    @property
    def band_bytes_per_nnz(self) -> int:
        # bytes this reader holds per nonzero while producing one flat band: the band itself
        # unless a transform adds temporaries
        return self.dtype.itemsize + 4

    @abstractmethod
    def _row_bytes(self) -> int: ...

    @property
    def rows_per_batch(self) -> int:
        step = max(1, _layout.BATCH_BYTES // max(1, self._row_bytes()))
        grain = self.grain
        if not grain:
            return step
        if step <= grain:
            return grain
        return (step // grain) * grain

    @abstractmethod
    def dense_band(self, r0: int, r1: int) -> DenseBand: ...

    def dense_block(self, r0: int, r1: int, c0: int, c1: int) -> NDArray[Any]:
        return self.dense_band(r0, r1).tile(c0, c1)

    @abstractmethod
    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix: ...

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        # default path (dense readers): band the overlapping rows by rows_per_batch, convert
        # each sub-band to CSR, and copy out only the part inside [s0, s1) — never densifies
        # more than one native row band at a time.
        indptr = self.indptr
        n_rows = self.shape[0]
        r0 = max(int(np.searchsorted(indptr, s0, side="right")) - 1, 0)
        r1 = min(max(int(np.searchsorted(indptr, s1, side="left")), r0 + 1), n_rows)
        step = self.rows_per_batch

        data_parts: list[NDArray[Any]] = []
        idx_parts: list[NDArray[np.int32]] = []
        row = r0
        while row < r1:
            row_end = min(row + step, r1)
            band = self.csr_rows(row, row_end)
            band_s0 = int(indptr[row])
            lo = max(s0, band_s0) - band_s0
            hi = min(s1, int(indptr[row_end])) - band_s0
            data_parts.append(np.asarray(band.data[lo:hi]))
            idx_parts.append(np.asarray(band.indices[lo:hi], dtype=np.int32))
            row = row_end

        if not data_parts:
            return np.empty(0, dtype=self.dtype), np.empty(0, dtype=np.int32)
        return np.concatenate(data_parts), np.concatenate(idx_parts)

    def flat_data(self, s0: int, s1: int) -> NDArray[Any]:
        return self.flat(s0, s1)[0]

    def flat_indices(self, s0: int, s1: int) -> NDArray[np.int32]:
        return self.flat(s0, s1)[1]

    def close(self) -> None:  # noqa: B027  # intentional default no-op; not every reader holds a handle
        pass


class _DenseReaderMixin(Reader):
    def _row_bytes(self) -> int:
        return self.shape[1] * self.dtype.itemsize

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        return sp.csr_matrix(self.dense_block(r0, r1, 0, self.shape[1]))

    def _count_band(self, r0: int, r1: int) -> NDArray[np.int64]:
        band = self.dense_block(r0, r1, 0, self.shape[1])
        return np.asarray(np.count_nonzero(band, axis=1), dtype=np.int64)

    def _compute_indptr(self) -> NDArray[np.int64]:
        n_rows, n_cols = self.shape
        step = max(1, _layout.BATCH_BYTES // max(1, n_cols * self.dtype.itemsize))
        bands = _layout.band_plan(n_rows, step)
        if not bands:
            return np.zeros(1, dtype=np.int64)
        jobs = [(r0, r1) for r0, r1 in bands]
        mode: Literal["threads", "processes"] = "threads" if self.thread_safe else "processes"
        counts = map_parallel(self._count_band, jobs, self.cpus, mode=mode)
        return np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.int64)

    def _lazy_band(self, arr: Any, r0: int, r1: int) -> DenseBand:
        if (r1 - r0) * self._row_bytes() <= _layout.BATCH_BYTES:
            return _ArrayBand(np.asarray(arr[r0:r1]))
        return _TiledBand(arr, r0, r1)


class _CSRReaderMixin(Reader):
    def _row_bytes(self) -> int:
        n_rows = self.shape[0]
        return max(1, self.nnz // max(1, n_rows)) * (self.dtype.itemsize + 4)

    @abstractmethod
    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix: ...

    @abstractmethod
    def _flat_data(self) -> Any: ...

    @abstractmethod
    def _flat_indices(self) -> Any: ...

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return _CSRBand(self._fetch_csr_band(r0, r1))

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._fetch_csr_band(r0, r1)

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        return self.flat_data(s0, s1), self.flat_indices(s0, s1)

    def flat_data(self, s0: int, s1: int) -> NDArray[Any]:
        return np.asarray(self._flat_data()[s0:s1])

    def flat_indices(self, s0: int, s1: int) -> NDArray[np.int32]:
        return np.asarray(self._flat_indices()[s0:s1], dtype=np.int32)


class DenseMemoryReader(_DenseReaderMixin):
    def __init__(self, arr: NDArray[Any]) -> None:
        super().__init__()
        self._arr = arr

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._arr.shape[0]), int(self._arr.shape[1]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._arr.dtype

    @property
    def lazy(self) -> bool:
        return False

    @property
    def thread_safe(self) -> bool:
        return True

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return _ArrayBand(self._arr[r0:r1])


class CSRMemoryReader(_CSRReaderMixin):
    def __init__(self, mat: sp.csr_matrix) -> None:
        super().__init__()
        self._mat = mat

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._mat.shape[0]), int(self._mat.shape[1]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._mat.dtype)  # type: ignore[no-any-return]  # scipy-stubs: csr_matrix.dtype is Any

    @property
    def lazy(self) -> bool:
        return False

    @property
    def thread_safe(self) -> bool:
        return True

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._mat.indptr, dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._mat[r0:r1]

    def _flat_data(self) -> NDArray[Any]:
        return np.asarray(self._mat.data)

    def _flat_indices(self) -> NDArray[Any]:
        return np.asarray(self._mat.indices)


def _clear_h5_state(state: dict[str, Any]) -> dict[str, Any]:
    for key in ("_file", "_dataset_obj"):
        if key in state:
            state[key] = None
    return state


class DenseH5Reader(_DenseReaderMixin):
    def __init__(self, path: str, dataset: str) -> None:
        super().__init__()
        self._path = path
        self._dataset_name = dataset
        self._file: Any = None
        self._shape_val: tuple[int, int] | None = None
        self._dtype_val: np.dtype[Any] | None = None

    def _ds(self) -> Any:
        if self._file is None:
            import h5py

            self._file = h5py.File(self._path, "r")
        return self._file[self._dataset_name]

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape_val is None:
            self._shape_val = (int(self._ds().shape[0]), int(self._ds().shape[1]))
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._dtype_val is None:
            self._dtype_val = self._ds().dtype
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return False  # h5py is not thread-safe

    @property
    def grain(self) -> int | None:
        chunks = self._ds().chunks
        return int(chunks[0]) if chunks else None

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return self._lazy_band(self._ds(), r0, r1)

    def dense_block(self, r0: int, r1: int, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._ds()[r0:r1, c0:c1])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict[str, Any]:
        return _clear_h5_state(self.__dict__.copy())

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class CSRH5Reader(_CSRReaderMixin):
    def __init__(self, path: str, group: str) -> None:
        super().__init__()
        self._path = path
        self._group_name = group
        self._file: Any = None
        self._dataset_obj: Any = None
        self._shape_val: tuple[int, int] | None = None
        self._dtype_val: np.dtype[Any] | None = None

    def _dataset(self) -> Any:
        if self._dataset_obj is None:
            import h5py
            from anndata.io import sparse_dataset

            self._file = h5py.File(self._path, "r")
            self._dataset_obj = sparse_dataset(self._file[self._group_name])
        return self._dataset_obj

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape_val is None:
            shape = self._dataset().shape
            self._shape_val = (int(shape[0]), int(shape[1]))
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._dtype_val is None:
            self._dtype_val = self._dataset().dtype
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return False  # h5py is not thread-safe

    @property
    def grain(self) -> int | None:
        chunks = self._flat_data().chunks
        return int(chunks[0]) if chunks else None

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._dataset().group["indptr"][:], dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._dataset()[r0:r1]

    def _flat_data(self) -> Any:
        return self._dataset().group["data"]

    def _flat_indices(self) -> Any:
        return self._dataset().group["indices"]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset_obj = None

    def __getstate__(self) -> dict[str, Any]:
        # a worker's flat reads never need indptr; pickling it per job would copy 8 bytes per row
        state = _clear_h5_state(self.__dict__.copy())
        state["_indptr_cache"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class TenxCSRReader(_CSRReaderMixin):
    # a 10x h5 matrix group read as cells-by-genes CSR (its indptr runs over barcodes), with an
    # optional feature mask applied band by band: kept features are renumbered densely and
    # int32 counts widen to float32 like the eager loader's
    def __init__(self, path: str, group: str, *, n_obs: int, n_features: int, keep: Any = None) -> None:
        super().__init__()
        self._path = path
        self._group_name = group
        self._file: Any = None
        self._group: Any = None
        self._n_obs = n_obs
        self._keep = None if keep is None else np.asarray(keep, dtype=bool)
        if self._keep is None:
            self._n_vars = n_features
            self._remap = None
        else:
            self._n_vars = int(self._keep.sum())
            self._remap = np.cumsum(self._keep, dtype=np.int32) - 1
        self._dtype_val: np.dtype[Any] | None = None

    def _node(self) -> Any:
        if self._group is None:
            import h5py

            self._file = h5py.File(self._path, "r")
            self._group = self._file[self._group_name]
        return self._group

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_obs, self._n_vars)

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._dtype_val is None:
            on_disk = self._node()["data"].dtype
            self._dtype_val = np.dtype(np.float32) if on_disk == np.dtype(np.int32) else np.dtype(on_disk)
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return False  # h5py is not thread-safe

    @property
    def grain(self) -> int | None:
        if self._keep is not None:
            return None  # source chunks do not map onto masked flat positions
        chunks = self._node()["data"].chunks
        return int(chunks[0]) if chunks else None

    def _source_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._node()["indptr"][:], dtype=np.int64)

    def _compute_indptr(self) -> NDArray[np.int64]:
        src = self._source_indptr()
        if self._keep is None:
            return src
        # one pass over indices counts the kept features per cell
        kept = np.zeros(self._n_obs, dtype=np.int64)
        max_nnz = min(_layout.BATCH_BYTES, _layout.PIPELINE_BATCH_BYTES) // _layout.COUNT_BYTES_PER_NNZ
        for b0, b1 in _layout.major_bands(src, max_nnz=max_nnz, cpus=1):
            idx = np.asarray(self._node()["indices"][int(src[b0]) : int(src[b1])])
            rows = np.repeat(np.arange(b0, b1), np.diff(src[b0 : b1 + 1]))
            kept[b0:b1] = np.bincount(rows[self._keep[idx]] - b0, minlength=b1 - b0)
        return np.concatenate([[0], np.cumsum(kept)]).astype(np.int64)

    def _read_rows(self, r0: int, r1: int) -> tuple[NDArray[Any], NDArray[np.int32], NDArray[np.int64]]:
        # kept entries of source rows [r0, r1) with their local indptr
        src = self._source_indptr() if self._keep is not None else None
        node = self._node()
        if src is None:
            indptr = self.indptr
            s0, s1 = int(indptr[r0]), int(indptr[r1])
            data = np.asarray(node["data"][s0:s1])
            idx = np.asarray(node["indices"][s0:s1], dtype=np.int32)
            return data.astype(self.dtype, copy=False), idx, (indptr[r0 : r1 + 1] - s0).astype(np.int64)
        s0, s1 = int(src[r0]), int(src[r1])
        idx = np.asarray(node["indices"][s0:s1])
        mask = self._keep[idx]  # type: ignore[index]  # keep is set on this branch
        data = np.asarray(node["data"][s0:s1])[mask].astype(self.dtype, copy=False)
        rows = np.repeat(np.arange(r0, r1), np.diff(src[r0 : r1 + 1]))[mask]
        local = np.concatenate([[0], np.cumsum(np.bincount(rows - r0, minlength=r1 - r0))]).astype(np.int64)
        remap = self._remap
        assert remap is not None
        return data, remap[idx[mask]], local

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        data, idx, local = self._read_rows(r0, r1)
        return sp.csr_matrix((data, idx, local), shape=(r1 - r0, self._n_vars))

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        if self._keep is None:
            node = self._node()
            data = np.asarray(node["data"][s0:s1]).astype(self.dtype, copy=False)
            return data, np.asarray(node["indices"][s0:s1], dtype=np.int32)
        if s1 <= s0:
            return np.empty(0, dtype=self.dtype), np.empty(0, dtype=np.int32)
        # masked flat positions are output positions: find the rows that cover them
        indptr = self.indptr
        r0 = max(int(np.searchsorted(indptr, s0, side="right")) - 1, 0)
        r1 = min(max(int(np.searchsorted(indptr, s1, side="left")), r0 + 1), self._n_obs)
        data, idx, _ = self._read_rows(r0, r1)
        lo = s0 - int(indptr[r0])
        return data[lo : lo + (s1 - s0)], idx[lo : lo + (s1 - s0)]

    def flat_data(self, s0: int, s1: int) -> NDArray[Any]:
        return self.flat(s0, s1)[0]

    def flat_indices(self, s0: int, s1: int) -> NDArray[np.int32]:
        return self.flat(s0, s1)[1]

    def _flat_data(self) -> Any:
        return self._node()["data"]

    def _flat_indices(self) -> Any:
        return self._node()["indices"]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._group = None

    def __getstate__(self) -> dict[str, Any]:
        # a masked reader's workers need indptr to place flat reads, so it travels; without a
        # mask the flat reads never touch it
        state = self.__dict__.copy()
        state["_file"] = None
        state["_group"] = None
        if self._keep is None:
            state["_indptr_cache"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class DenseZarrReader(_DenseReaderMixin):
    def __init__(self, arr: zarr.Array[Any]) -> None:
        super().__init__()
        self._arr = arr

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._arr.shape[0]), int(self._arr.shape[1]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._arr.dtype

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return True  # zarr is thread-safe

    @property
    def grain(self) -> int | None:
        return int(self._arr.chunks[0])

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return self._lazy_band(self._arr, r0, r1)

    def dense_block(self, r0: int, r1: int, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._arr[r0:r1, c0:c1])


class CSRZarrReader(_CSRReaderMixin):
    def __init__(self, group: zarr.Group, *, _tmp_dir: Any = None) -> None:
        super().__init__()
        self._group = group
        self._shape_val = shape_attr(group)
        self._dtype_val = get_array(group, "data").dtype
        self._tmp_dir = _tmp_dir

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return True  # zarr is thread-safe

    @property
    def grain(self) -> int | None:
        return int(get_array(self._group, "data").chunks[0])

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(get_array(self._group, "indptr")[:], dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        indptr = self.indptr
        s0, s1 = int(indptr[r0]), int(indptr[r1])
        data = np.asarray(get_array(self._group, "data")[s0:s1])
        indices = np.asarray(get_array(self._group, "indices")[s0:s1])
        sub_indptr = (indptr[r0 : r1 + 1] - s0).astype(np.int64)
        return sp.csr_matrix((data, indices, sub_indptr), shape=(r1 - r0, self._shape_val[1]))

    def _flat_data(self) -> Any:
        return get_array(self._group, "data")

    def _flat_indices(self) -> Any:
        return get_array(self._group, "indices")

    def close(self) -> None:
        if self._tmp_dir is not None:
            import shutil

            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def __getstate__(self) -> dict[str, Any]:
        # a pickled (worker-process) copy doesn't own the temp dir: only the original,
        # parent-process reader deletes it (once, after every job has finished) — closing a
        # worker's own copy must not race-delete a directory sibling jobs still need.
        state = self.__dict__.copy()
        state["_tmp_dir"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class _SwappedCSRView(_CSRReaderMixin):
    # a raw CSC group's own (data, indices, indptr) reinterpreted as CSR over the transposed
    # shape (scipy's own `.T` trick, applied lazily): feeds a CSC source into
    # write_transposed_sparse without ever densifying or copying it. Internal to as_reader's
    # CSC normalisation; never part of the public reader surface. An h5py group is held by
    # file path and reopened per process, since h5py objects cannot be pickled.
    def __init__(self, group: Any, true_shape: tuple[int, int], dtype: np.dtype[Any]) -> None:
        import zarr

        super().__init__()
        self._shape_val = (true_shape[1], true_shape[0])
        self._dtype_val = dtype
        self._thread_safe = isinstance(group, zarr.Group)
        self._group: Any = group if self._thread_safe else None
        self._path = None if self._thread_safe else group.file.filename
        self._group_name = None if self._thread_safe else group.name
        self._file: Any = None

    def _node(self) -> Any:
        if self._group is None:
            import h5py

            self._file = h5py.File(self._path, "r")
            self._group = self._file[self._group_name]
        return self._group

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return self._thread_safe

    @property
    def grain(self) -> int | None:
        chunks = self._node()["data"].chunks
        return int(chunks[0]) if chunks else None

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._node()["indptr"][:], dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        indptr = self.indptr
        s0, s1 = int(indptr[r0]), int(indptr[r1])
        node = self._node()
        data = np.asarray(node["data"][s0:s1])
        indices = np.asarray(node["indices"][s0:s1])
        sub_indptr = (indptr[r0 : r1 + 1] - s0).astype(np.int64)
        return sp.csr_matrix((data, indices, sub_indptr), shape=(r1 - r0, self._shape_val[1]))

    def _flat_data(self) -> Any:
        return self._node()["data"]

    def _flat_indices(self) -> Any:
        return self._node()["indices"]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._group = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if not self._thread_safe:
            state["_group"] = None
            state["_file"] = None
            state["_indptr_cache"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class ConcatReader(Reader):
    def __init__(self, readers: Sequence[Reader]) -> None:
        super().__init__()
        if not readers:
            raise ConversionError("ConcatReader requires at least one reader.")
        n_cols = readers[0].shape[1]
        dtype = readers[0].dtype
        for r in readers[1:]:
            if r.shape[1] != n_cols:
                raise ConversionError(f"ConcatReader: column count mismatch ({r.shape[1]} vs {n_cols}).")
            if r.dtype != dtype:
                raise ConversionError(f"ConcatReader: dtype mismatch ({r.dtype} vs {dtype}).")
        self._readers = tuple(readers)
        self._n_cols = n_cols
        self._dtype_val = dtype
        self._row_offsets = tuple(int(v) for v in np.cumsum([0, *(r.shape[0] for r in readers)]))

    @property
    def shape(self) -> tuple[int, int]:
        return (self._row_offsets[-1], self._n_cols)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return any(r.lazy for r in self._readers)

    @property
    def thread_safe(self) -> bool:
        return all(r.thread_safe for r in self._readers)

    def _row_bytes(self) -> int:
        return max(r._row_bytes() for r in self._readers)

    def _compute_indptr(self) -> NDArray[np.int64]:
        chunks: list[NDArray[np.int64]] = []
        nnz_offset = 0
        for r in self._readers:
            ip = r.indptr.astype(np.int64)
            chunks.append(ip[:-1] + nnz_offset)
            nnz_offset += int(ip[-1])
        chunks.append(np.array([nnz_offset], dtype=np.int64))
        return np.concatenate(chunks)

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        parts: list[tuple[int, int, DenseBand]] = []
        for reader, off_lo, off_hi in zip(self._readers, self._row_offsets[:-1], self._row_offsets[1:], strict=True):
            lo, hi = max(r0, off_lo), min(r1, off_hi)
            if lo >= hi:
                continue
            parts.append((lo - r0, hi - r0, reader.dense_band(lo - off_lo, hi - off_lo)))
        return _StitchedBand(parts, r1 - r0, self._dtype_val)

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        parts = []
        for reader, off_lo, off_hi in zip(self._readers, self._row_offsets[:-1], self._row_offsets[1:], strict=True):
            lo, hi = max(r0, off_lo), min(r1, off_hi)
            if lo >= hi:
                continue
            parts.append(reader.csr_rows(lo - off_lo, hi - off_lo))
        if len(parts) == 1:
            return parts[0]
        return sp.vstack(parts, format="csr")

    def _flat_parts(self, s0: int, s1: int) -> list[tuple[int, int, Reader, int, int]]:
        # (offset into the result, length, reader, its local s0, its local s1) for the overlapping readers
        parts = []
        nnz_offset = 0
        for reader in self._readers:
            off_lo, off_hi = nnz_offset, nnz_offset + reader.nnz
            lo, hi = max(s0, off_lo), min(s1, off_hi)
            if lo < hi:
                parts.append((lo - s0, hi - lo, reader, lo - off_lo, hi - off_lo))
            nnz_offset = off_hi
        return parts

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        parts = self._flat_parts(s0, s1)
        if len(parts) == 1:  # the common case needs no second buffer
            _, _, reader, lo, hi = parts[0]
            return reader.flat(lo, hi)
        data_buf = np.empty(s1 - s0, dtype=self._dtype_val)
        idx_buf = np.empty(s1 - s0, dtype=np.int32)
        for at, n, reader, lo, hi in parts:
            data_buf[at : at + n], idx_buf[at : at + n] = reader.flat(lo, hi)
        return data_buf, idx_buf

    def flat_indices(self, s0: int, s1: int) -> NDArray[np.int32]:
        parts = self._flat_parts(s0, s1)
        if len(parts) == 1:
            _, _, reader, lo, hi = parts[0]
            return reader.flat_indices(lo, hi)
        idx_buf = np.empty(s1 - s0, dtype=np.int32)
        for at, n, reader, lo, hi in parts:
            idx_buf[at : at + n] = reader.flat_indices(lo, hi)
        return idx_buf

    def close(self) -> None:
        for r in self._readers:
            r.close()


class _FileSlice:
    # 1-D array stored raw in a plain file: slicing reads just that range, no memmap pages
    __slots__ = ("_dtype", "_path")

    def __init__(self, path: Path, dtype: np.dtype[Any]) -> None:
        self._path = path
        self._dtype = dtype

    def __getitem__(self, item: slice) -> NDArray[Any]:
        s0, s1 = item.start or 0, item.stop
        count = max(0, s1 - s0)
        if count == 0:
            return np.empty(0, dtype=self._dtype)
        return np.fromfile(self._path, dtype=self._dtype, count=count, offset=s0 * self._dtype.itemsize)


class CSRFileReader(_CSRReaderMixin):
    # CSR rows whose data and indices sit raw in two plain files (sort's per-group buckets)
    def __init__(
        self,
        data_path: Path,
        indices_path: Path,
        indptr: NDArray[np.int64],
        shape: tuple[int, int],
        dtype: np.dtype[Any],
    ) -> None:
        super().__init__()
        self._data = _FileSlice(data_path, np.dtype(dtype))
        self._indices = _FileSlice(indices_path, np.dtype(np.int32))
        self._indptr_val = np.asarray(indptr, dtype=np.int64)
        self._shape_val = shape
        self._dtype_val = np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return True  # every read opens its own handle

    def _compute_indptr(self) -> NDArray[np.int64]:
        return self._indptr_val

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        indptr = self.indptr
        s0, s1 = int(indptr[r0]), int(indptr[r1])
        sub_indptr = (indptr[r0 : r1 + 1] - s0).astype(np.int64)
        return sp.csr_matrix((self._data[s0:s1], self._indices[s0:s1], sub_indptr), shape=(r1 - r0, self._shape_val[1]))

    def _flat_data(self) -> Any:
        return self._data

    def _flat_indices(self) -> Any:
        return self._indices


class _WrappedReader(Reader):
    # a per-band transform over another reader: geometry, parallel mode and grain are the inner's
    def __init__(self, inner: Reader) -> None:
        super().__init__()
        self._inner = inner
        self.cpus = inner.cpus

    @property
    def shape(self) -> tuple[int, int]:
        return self._inner.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._inner.dtype

    @property
    def lazy(self) -> bool:
        return self._inner.lazy

    @property
    def thread_safe(self) -> bool:
        return self._inner.thread_safe

    @property
    def grain(self) -> int | None:
        return self._inner.grain

    def _compute_indptr(self) -> NDArray[np.int64]:
        return self._inner.indptr

    def _row_bytes(self) -> int:
        return self._inner._row_bytes()

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return _CSRBand(self.csr_rows(r0, r1))

    def close(self) -> None:
        self._inner.close()


class CastReader(_WrappedReader):
    """Another reader's values cast to ``dtype`` band by band (append's lossless widening)."""

    def __init__(self, inner: Reader, dtype: np.dtype[Any]) -> None:
        super().__init__(inner)
        self._dtype_val = np.dtype(dtype)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def band_bytes_per_nnz(self) -> int:
        return self._inner.band_bytes_per_nnz + self._dtype_val.itemsize  # the cast copy

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        data, indices = self._inner.flat(s0, s1)
        return data.astype(self._dtype_val, copy=False), indices

    def flat_data(self, s0: int, s1: int) -> NDArray[Any]:
        return self._inner.flat_data(s0, s1).astype(self._dtype_val, copy=False)

    def flat_indices(self, s0: int, s1: int) -> NDArray[np.int32]:
        return self._inner.flat_indices(s0, s1)

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._inner.csr_rows(r0, r1).astype(self._dtype_val, copy=False)

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return _CastBand(self._inner.dense_band(r0, r1), self._dtype_val)


class _CastBand:
    __slots__ = ("_band", "_dtype")

    def __init__(self, band: DenseBand, dtype: np.dtype[Any]) -> None:
        self._band = band
        self._dtype = dtype

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        return self._band.tile(c0, c1).astype(self._dtype, copy=False)


class RemapColumnsReader(_WrappedReader):
    """Another reader with column ``j`` moved to ``new_index[j]``, each row's entries re-sorted.

    A row permutation of a cells x cells graph (sort's ``obsp``) moves its columns too; the rows go
    through the sort buckets and this renumbers the columns as each band is read. A flat band is
    read as the whole rows it touches, so a row cut by the band edge is still sorted as a whole.
    """

    def __init__(self, inner: Reader, new_index: NDArray[np.int64]) -> None:
        super().__init__(inner)
        self._new_index = np.asarray(new_index, dtype=np.int32)

    @property
    def band_bytes_per_nnz(self) -> int:
        return self._inner.band_bytes_per_nnz + 12  # remapped int32 indices + int64 sort order

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        rows = self._inner.csr_rows(r0, r1)
        out = sp.csr_matrix((rows.data, self._new_index[rows.indices], rows.indptr), shape=rows.shape)
        out.has_sorted_indices = False
        out.sort_indices()
        return out

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        if s1 <= s0:
            return np.empty(0, dtype=self.dtype), np.empty(0, dtype=np.int32)
        indptr = self.indptr
        r0 = int(np.searchsorted(indptr, s0, side="right")) - 1
        r1 = int(np.searchsorted(indptr, s1, side="left"))
        rows = self.csr_rows(r0, r1)
        lo, hi = s0 - int(indptr[r0]), s1 - int(indptr[r0])
        return np.asarray(rows.data[lo:hi]), np.asarray(rows.indices[lo:hi], dtype=np.int32)


def _is_anndata_sparse_dataset(matrix: Any) -> bool:
    fmt = getattr(matrix, "format", None)
    return fmt in ("csr", "csc") and hasattr(matrix, "group")


def _transpose_csc_to_csr(
    group: Any, true_shape: tuple[int, int], dtype: np.dtype[Any], cfg: AppConfig, tmp_dir: str | None
) -> Reader:
    import shutil
    import tempfile
    from pathlib import Path

    import zarr

    from annizarr._writers._sparse import write_transposed_sparse

    scratch = Path(tempfile.mkdtemp(prefix="annizarr_csc2csr_", dir=tmp_dir))
    try:
        tmp_root = zarr.open_group(str(scratch), mode="w")
        csr_group = write_transposed_sparse(tmp_root, "X", _SwappedCSRView(group, true_shape, dtype), cfg, target="csr")
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    logger.info("lazy CSC input streamed to a temporary CSR store (never materialised in memory).")
    return CSRZarrReader(csr_group, _tmp_dir=scratch)


def as_reader(matrix: Any, *, cfg: AppConfig, tmp_dir: str | None = None) -> Reader:
    # CSC never reaches a writer: in-memory CSC converts to CSR here, a lazy CSC streams through
    # write_transposed_sparse into a temp CSR store under tmp_dir (the output's directory).
    reader = _reader_for(matrix, cfg, tmp_dir)
    reader.cpus = cfg.chunks.cpus
    return reader


def _reader_for(matrix: Any, cfg: AppConfig, tmp_dir: str | None) -> Reader:
    import h5py
    import numpy as np
    import scipy.sparse as sp
    import zarr

    if isinstance(matrix, Reader):
        return matrix

    if isinstance(matrix, np.ma.MaskedArray):
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {type(matrix).__name__} (masked)")

    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {matrix.ndim}-D ndarray")
        return DenseMemoryReader(matrix)

    if sp.issparse(matrix):
        fmt = matrix.format
        if fmt == "csr":
            return CSRMemoryReader(matrix)
        if fmt == "csc":
            logger.info("adata.X was CSC and converted to CSR in memory.")
            return CSRMemoryReader(matrix.tocsr())
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {fmt}")

    if isinstance(matrix, h5py.Dataset):
        return DenseH5Reader(matrix.file.filename, matrix.name)

    if isinstance(matrix, zarr.Array):
        return DenseZarrReader(matrix)

    if _is_anndata_sparse_dataset(matrix):
        if matrix.format == "csr":
            if isinstance(matrix.group, zarr.Group):
                return CSRZarrReader(matrix.group)
            return CSRH5Reader(matrix.group.file.filename, matrix.group.name)
        return _transpose_csc_to_csr(matrix.group, matrix.shape, matrix.dtype, cfg, tmp_dir)

    if isinstance(matrix, zarr.Group):
        enc = matrix.attrs.get("encoding-type")
        if enc == "csr_matrix":
            return CSRZarrReader(matrix)
        if enc == "csc_matrix":
            return _transpose_csc_to_csr(matrix, shape_attr(matrix), get_array(matrix, "data").dtype, cfg, tmp_dir)
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: zarr group ({enc!r})")

    got = getattr(matrix, "format", None) or type(matrix).__name__
    raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {got}")
