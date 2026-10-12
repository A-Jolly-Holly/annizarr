from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from annizarr._core import _layout
from annizarr._core._layout import (
    check_write_unit,
    copy_bytes_per_item,
    major_bands,
    plan_bands,
    write_grid,
    x_compressors,
)
from annizarr._core._runtime import (
    ArrayTarget,
    map_parallel,
    pipeline_parallel,
    progress,
    run_parallel,
    stage,
    writer_parallel_mode,
)
from annizarr._writers._buckets import Buckets
from annizarr._writers._encoding import make_sparse_group, set_array_attrs, sparse_shards, suppress_autoshard_warning

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader

type _CsrPayload = tuple[int, int, NDArray[Any], NDArray[Any]]
type _Placed = tuple[int, int, NDArray[Any], NDArray[Any]]

# target cells for the per-band histograms the scatter offsets come from; output bands are whole cells
_TARGET_GRID = 8192


def _no_tick() -> None:
    pass


def _csr_job(
    data_target: ArrayTarget,
    indices_target: ArrayTarget,
    reader: Reader,
    s0: int,
    s1: int,
    indices_dtype: Any,
    tick: Callable[[], None],
) -> None:
    try:
        data_arr = data_target.resolve()
        indices_arr = indices_target.resolve()
        data, indices = reader.flat(s0, s1)
        data_arr[s0:s1] = data
        indices_arr[s0:s1] = np.asarray(indices, dtype=indices_dtype)
        tick()
    finally:
        if data_target.in_worker:
            reader.close()


def _csr_pipeline_produce(reader: Reader, s0: int, s1: int) -> _CsrPayload:
    try:
        data, indices = reader.flat(s0, s1)
        return s0, s1, data, indices
    finally:
        reader.close()


def _local_tmp_dir(group: zarr.Group) -> str | None:
    from zarr.storage import LocalStore

    store = group.store_path.store
    return str(store.root) if isinstance(store, LocalStore) else None


def write_csr(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    _write_compressed(group, key, reader, cfg, csr=True, shape=reader.shape)


def write_csc(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    if reader.lazy:
        write_transposed_sparse(group, key, reader, cfg, target="csc")
        return
    from annizarr._sources._readers import CSRMemoryReader

    # in memory: scipy's tocsc, then the parallel flat copy over the column-major view (.T is zero-copy)
    n_rows = reader.shape[0]
    columns = CSRMemoryReader(reader.csr_rows(0, n_rows).tocsc().T)
    columns.cpus = reader.cpus
    _write_compressed(group, key, columns, cfg, csr=False, shape=reader.shape)


def _write_compressed(
    group: zarr.Group, key: str, reader: Reader, cfg: AppConfig, *, csr: bool, shape: tuple[int, int]
) -> None:
    # reader is major-axis ordered: rows of a CSR output, columns of a CSC output
    n_major = reader.shape[0]
    indptr_full = reader.indptr
    nnz_total = int(indptr_full[-1])
    indices_dtype = np.int32  # matches scipy's default; values fit unless > 2^31 along the minor axis

    sp_group = make_sparse_group(group, key, csr=csr, shape=shape)
    flat_chunk = _layout.sparse_flat_chunk(cfg.chunks, csr=csr, nnz=nnz_total, n_major=n_major)
    shards = sparse_shards(cfg.chunks.auto_shard)
    with suppress_autoshard_warning(cfg.chunks.auto_shard):
        data_arr = sp_group.require_array(
            "data",
            shape=(nnz_total,),
            dtype=reader.dtype,
            chunks=(flat_chunk,),
            shards=shards,
            compressors=x_compressors(),
            overwrite=True,
        )
        indices_arr = sp_group.require_array(
            "indices",
            shape=(nnz_total,),
            dtype=indices_dtype,
            chunks=(flat_chunk,),
            shards=shards,
            compressors=x_compressors(),
            overwrite=True,
        )
    indptr_dtype = np.int64 if nnz_total > np.iinfo(np.int32).max else np.int32
    indptr_arr = sp_group.require_array(
        "indptr", shape=(n_major + 1,), dtype=indptr_dtype, chunks=(n_major + 1,), overwrite=True
    )
    for a in (data_arr, indices_arr, indptr_arr):
        set_array_attrs(a)
    indptr_arr[:] = indptr_full.astype(indptr_dtype)

    if nnz_total == 0:
        return

    check_write_unit(data_arr)
    # the reader's own band cost plus zarr's write-side copy of what gets written
    out_bytes = np.dtype(reader.dtype).itemsize + np.dtype(indices_dtype).itemsize
    bytes_per_nnz = reader.band_bytes_per_nnz + copy_bytes_per_item(out_bytes) - out_bytes
    step = write_grid(data_arr)[0]
    cpus = cfg.chunks.cpus
    segments = plan_bands(nnz_total, unit=step, bytes_per_item=bytes_per_nnz, cpus=cpus, grain=reader.grain)
    mode = writer_parallel_mode(reader, data_arr.store_path.store)

    if mode == "processes":
        data_target = ArrayTarget.detached(data_arr)
        indices_target = ArrayTarget.detached(indices_arr)
        jobs = [(data_target, indices_target, reader, s0, s1, indices_dtype, _no_tick) for s0, s1 in segments]
        run_parallel(_csr_job, jobs, cfg.chunks.cpus, mode="processes")
        return

    if mode == "pipeline":
        budget = min(_layout.BATCH_BYTES, _layout.PIPELINE_BATCH_BYTES)
        pipeline_segments = plan_bands(
            nnz_total, unit=step, bytes_per_item=bytes_per_nnz, cpus=cpus, grain=reader.grain, budget=budget
        )
        pipeline_jobs = [(reader, s0, s1) for s0, s1 in pipeline_segments]

        def consume(payload: _CsrPayload) -> None:
            ps0, ps1, data, indices = payload
            data_arr[ps0:ps1] = data
            indices_arr[ps0:ps1] = np.asarray(indices, dtype=indices_dtype)

        pipeline_parallel(_csr_pipeline_produce, consume, pipeline_jobs, cfg.chunks.cpus)
        return

    tick = progress(len(segments), f"Writing {key}")
    data_target = ArrayTarget.attached(data_arr)
    indices_target = ArrayTarget.attached(indices_arr)
    jobs = [(data_target, indices_target, reader, s0, s1, indices_dtype, tick) for s0, s1 in segments]
    run_parallel(_csr_job, jobs, cpus, mode="threads")


def _count_job(
    reader: Reader, bands: list[tuple[int, int, int]], n_target: int, grid: int, close: bool
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    # nonzeros per target over several source bands, plus each band's histogram over a fixed grid
    # of target cells: enough to place every band's pieces at exact offsets without a bands by
    # targets table (8192 cells is 64KB per band whatever the feature count)
    total = np.zeros(n_target, dtype=np.int64)
    hist = np.zeros((len(bands), grid), dtype=np.int64)
    try:
        for i, (_, s0, s1) in enumerate(bands):
            tgt = reader.flat_indices(s0, s1)
            total += np.bincount(tgt, minlength=n_target)
            hist[i] = np.bincount(tgt.astype(np.int64) * grid // n_target, minlength=grid)
    finally:
        if close:
            reader.close()
    return total, hist


def _scatter_band(
    reader: Reader,
    s0: int,
    s1: int,
    b0: int,
    row_nnz: NDArray[np.int64],
    band_edges: NDArray[np.int64],
    offsets: NDArray[np.int64],
    buckets: Buckets,
    close: bool,
) -> None:
    # one source band split by output band, each run written at its precomputed offset in source
    # order; three large sorted arrays rather than one small array per band, so no allocator churn
    try:
        vals, tgt = reader.flat(s0, s1)
    finally:
        if close:
            reader.close()
    src = np.repeat(np.arange(b0, b0 + len(row_nnz), dtype=np.int32), row_nnz)
    band_ids = np.searchsorted(band_edges, tgt, side="right")
    order = np.argsort(band_ids, kind="stable")
    bounds = np.searchsorted(band_ids[order], np.arange(len(band_edges) + 1))
    src, tgt, vals = src[order], tgt[order], vals[order]
    for bi in range(len(band_edges)):
        lo, hi = int(bounds[bi]), int(bounds[bi + 1])
        if lo < hi:
            buckets.write_at(bi, int(offsets[bi]), src=src[lo:hi], tgt=tgt[lo:hi], val=vals[lo:hi])


def _sorted_band(buckets: Buckets, band: int) -> tuple[NDArray[Any], NDArray[Any]]:
    # entries of one output band back in source order; a stable sort by target gives each
    # target's entries in ascending source position
    order = np.argsort(buckets.read(band, "tgt"), kind="stable")
    return buckets.read(band, "val")[order], buckets.read(band, "src")[order]


def _place_job(
    buckets: Buckets,
    band: int,
    o0: int,
    o1: int,
    data_target: ArrayTarget,
    indices_target: ArrayTarget,
    unit: int,
    shared: Buckets,
    shared_units: NDArray[np.int64],
    indices_dtype: Any,
) -> None:
    # writes the write units this band alone covers straight to the store; a unit shared with a
    # neighbouring band gets this band's part in a per-unit temp file, written once by _seam_job
    vals, src = _sorted_band(buckets, band)
    src = np.asarray(src, dtype=indices_dtype)
    data_arr, indices_arr = data_target.resolve(), indices_target.resolve()
    k_first, k_last = o0 // unit, (o1 - 1) // unit
    owned: list[tuple[int, int]] = []
    for k in range(k_first, k_last + 1):
        lo, hi = max(o0, k * unit), min(o1, (k + 1) * unit)
        if np.searchsorted(shared_units, k) < len(shared_units) and shared_units[np.searchsorted(shared_units, k)] == k:
            shared.write_at(
                int(np.searchsorted(shared_units, k)),
                lo - k * unit,
                val=vals[lo - o0 : hi - o0],
                src=src[lo - o0 : hi - o0],
            )
        else:
            owned.append((lo, hi))
    if owned:  # consecutive, so one call lets the codec pool encode the units concurrently
        lo, hi = owned[0][0], owned[-1][1]
        data_arr[lo:hi] = vals[lo - o0 : hi - o0]
        indices_arr[lo:hi] = src[lo - o0 : hi - o0]


def _seam_job(
    shared: Buckets,
    i: int,
    k: int,
    unit: int,
    data_target: ArrayTarget,
    indices_target: ArrayTarget,
    indices_dtype: Any,
) -> None:
    lo = k * unit
    hi = lo + shared.counts[i]
    data_target.resolve()[lo:hi] = shared.read(i, "val")
    indices_target.resolve()[lo:hi] = np.asarray(shared.read(i, "src"), dtype=indices_dtype)


def _place_band(buckets: Buckets, band: int, o0: int, o1: int) -> _Placed:
    vals, src = _sorted_band(buckets, band)
    return o0, o1, vals, src


class _Placer:
    # the non-local (Icechunk session) fallback: this process is the only writer, so it takes the
    # sorted bands in order, carries the ragged tail into the next band so no chunk is written
    # twice, and writes several units per call so the codec pool encodes them concurrently
    def __init__(self, data_arr: Any, indices_arr: Any, unit: int, indices_dtype: Any) -> None:
        self._data = data_arr
        self._indices = indices_arr
        self._unit = unit
        self._indices_dtype = indices_dtype
        self._pos = 0
        self._vals: NDArray[Any] = np.empty(0, dtype=data_arr.dtype)
        self._src: NDArray[Any] = np.empty(0, dtype=np.int32)
        unit_bytes = unit * (data_arr.dtype.itemsize + np.dtype(indices_dtype).itemsize)
        self._group = max(1, min(_codec_threads(), (_layout.BATCH_BYTES // 2) // max(1, unit_bytes)))

    def consume(self, payload: _Placed) -> None:
        o0, _, vals, src = payload
        assert o0 == self._pos + len(self._vals), "output bands must arrive in order"
        if len(self._vals):
            vals = np.concatenate([self._vals, vals])
            src = np.concatenate([self._src, src])
        whole = ((self._pos + len(vals)) // self._unit) * self._unit - self._pos
        grouped = (whole // (self._unit * self._group)) * (self._unit * self._group)
        if grouped > 0:
            self._write(vals[:grouped], src[:grouped])
        self._vals, self._src = vals[grouped:].copy(), src[grouped:].copy()  # a view would pin the payload

    def flush(self) -> None:
        if len(self._vals):
            self._write(self._vals, self._src)
            self._vals, self._src = self._vals[:0], self._src[:0]

    def _write(self, vals: NDArray[Any], src: NDArray[Any]) -> None:
        end = self._pos + len(vals)
        self._data[self._pos : end] = vals
        self._indices[self._pos : end] = np.asarray(src, dtype=self._indices_dtype)
        self._pos = end


def _codec_threads() -> int:
    import os

    configured = zarr.config.get("threading.max_workers")
    return int(configured) if configured else (os.cpu_count() or 1)


def _output_bands(cell_nnz: NDArray[np.int64], max_nnz: int, cpus: int) -> list[tuple[int, int]]:
    # consecutive grid cells per output band, at most max_nnz nonzeros unless one cell alone is
    # larger, and at least cpus bands when the cells allow
    target = max(1, min(max_nnz, int(cell_nnz.sum()) // max(cpus, 1) + 1))
    bands: list[tuple[int, int]] = []
    c0, acc = 0, 0
    for c, n in enumerate(cell_nnz.tolist()):
        if c > c0 and acc + n > target:
            bands.append((c0, c))
            c0, acc = c, 0
        acc += n
    bands.append((c0, len(cell_nnz)))
    return bands


def write_transposed_sparse(
    group: zarr.Group, key: str, reader: Reader, cfg: AppConfig, *, target: Literal["csc", "csr"]
) -> zarr.Group:
    # four band-bounded parallel passes with nothing in this process but small tables: count
    # nonzeros per target and per (source band, target grid cell); scatter every source band's
    # entries into per-output-band files at offsets known from those counts; place each output
    # band sorted, writing the write units it alone covers; write the units shared by two bands
    # once from per-unit files. `reader` is source-major: reader.shape[0] is the source major-axis
    # count, reader.indptr is over it, reader.flat(...) gives (data, target-axis index) in
    # source-major flat order. The true output shape is reader.shape when target=="csc"; a CSC
    # source arrives pre-swapped (as_reader's _SwappedCSRView), so for target=="csr" it is reversed.
    from zarr.storage import LocalStore

    n_source_major, n_target_major = reader.shape
    true_shape = reader.shape if target == "csc" else (n_target_major, n_source_major)
    indptr_src = reader.indptr
    nnz = reader.nnz
    row_nnz = np.diff(indptr_src)
    cpus = cfg.chunks.cpus
    mode: Literal["threads", "processes"] = "threads" if reader.thread_safe else "processes"
    close_in_worker = mode == "processes"
    local = isinstance(group.store_path.store, LocalStore)

    value_dtype = np.dtype(reader.dtype)
    indices_dtype = np.int32  # source-major positions in the output; matches scipy's default
    indptr_dtype = np.int64 if nnz > np.iinfo(np.int32).max else np.int32

    scatter_nnz = max(1, _layout.BATCH_BYTES // _layout.SCATTER_BYTES_PER_NNZ)
    source_bands = major_bands(indptr_src, max_nnz=scatter_nnz, cpus=cpus)
    band_ranges = [(b0, int(indptr_src[b0]), int(indptr_src[b1])) for b0, b1 in source_bands]
    grid = min(n_target_major, _TARGET_GRID)
    cell_edges = -((-np.arange(grid + 1, dtype=np.int64) * n_target_major) // grid)  # ceil(c * n / grid)

    with stage(f"Counting nonzeros per {'column' if target == 'csc' else 'row'} of {key}"):
        n_jobs = min(len(band_ranges), max(1, 2 * cpus))
        job_bands = [band_ranges[i::n_jobs] for i in range(n_jobs)] if nnz else []
        results = map_parallel(
            _count_job,
            [(reader, bands, n_target_major, grid, close_in_worker) for bands in job_bands],
            cpus,
            mode=mode,
        )
        target_nnz = np.zeros(n_target_major, dtype=np.int64)
        hist = np.zeros((len(band_ranges), grid), dtype=np.int64)
        for i, (total, h) in enumerate(results):
            target_nnz += total
            hist[i::n_jobs] = h
    target_indptr = np.concatenate([[0], np.cumsum(target_nnz)]).astype(np.int64)

    sp_group = make_sparse_group(group, key, csr=(target == "csr"), shape=true_shape)
    flat_chunk = _layout.sparse_flat_chunk(cfg.chunks, csr=(target == "csr"), nnz=nnz, n_major=n_target_major)
    shards = sparse_shards(cfg.chunks.auto_shard)
    with suppress_autoshard_warning(cfg.chunks.auto_shard):
        data_arr = sp_group.require_array(
            "data",
            shape=(nnz,),
            dtype=value_dtype,
            chunks=(flat_chunk,),
            shards=shards,
            compressors=x_compressors(),
            overwrite=True,
        )
        indices_arr = sp_group.require_array(
            "indices",
            shape=(nnz,),
            dtype=indices_dtype,
            chunks=(flat_chunk,),
            shards=shards,
            compressors=x_compressors(),
            overwrite=True,
        )
    indptr_arr = sp_group.require_array(
        "indptr", shape=(n_target_major + 1,), dtype=indptr_dtype, chunks=(n_target_major + 1,), overwrite=True
    )
    for a in (data_arr, indices_arr, indptr_arr):
        set_array_attrs(a)
    indptr_arr[:] = target_indptr.astype(indptr_dtype)
    if nnz == 0:
        return sp_group
    check_write_unit(data_arr)

    # output bands are whole grid cells, so each (source band, output band) count is a sum of
    # histogram cells and every piece has an exact offset in its band's files
    cell_nnz = np.diff(target_indptr[cell_edges])
    place_nnz = max(1, _layout.BATCH_BYTES // _layout.PLACE_BYTES_PER_NNZ)
    cells = _output_bands(cell_nnz, place_nnz, cpus)
    band_edges = np.asarray([int(cell_edges[c1]) for _, c1 in cells], dtype=np.int64)  # target-index edges
    flat_edges = target_indptr[np.concatenate([[0], band_edges])]
    counts = np.stack([hist[:, c0:c1].sum(axis=1) for c0, c1 in cells], axis=1)  # (source band, output band)
    offsets = np.cumsum(counts, axis=0) - counts
    assert (counts.sum(axis=0) == np.diff(flat_edges)).all(), "grid counts must tile every output band"

    unit = write_grid(data_arr)[0]
    tmp_dir = _local_tmp_dir(group)
    fields = {"src": np.dtype(np.int32), "tgt": np.dtype(np.int32), "val": value_dtype}
    buckets = Buckets.create(len(cells), fields, dir=tmp_dir, prefix="annizarr_transpose_", sizes=np.diff(flat_edges))
    shared_units = np.unique(flat_edges[1:-1] // unit)  # units containing a band boundary
    shared_sizes = [min(unit, nnz - int(k) * unit) for k in shared_units]
    shared = Buckets.create(
        len(shared_units),
        {"val": value_dtype, "src": np.dtype(np.int32)},
        dir=tmp_dir,
        prefix="annizarr_seams_",
        sizes=shared_sizes,
    )
    try:
        with stage(f"Transposing {key} to {target} ({len(source_bands)} source bands, {len(cells)} output bands)"):
            scatter_jobs = [
                (
                    reader,
                    s0,
                    s1,
                    b0,
                    row_nnz[b0 : b0 + (source_bands[j][1] - b0)],
                    band_edges,
                    offsets[j],
                    buckets,
                    close_in_worker,
                )
                for j, (b0, s0, s1) in enumerate(band_ranges)
            ]
            run_parallel(_scatter_band, scatter_jobs, cpus, mode=mode)

        with stage(f"Writing {key} ({target}, nnz={nnz})"):
            if local:
                data_target, indices_target = ArrayTarget.detached(data_arr), ArrayTarget.detached(indices_arr)
                place_jobs = [
                    (
                        buckets,
                        bi,
                        int(flat_edges[bi]),
                        int(flat_edges[bi + 1]),
                        data_target,
                        indices_target,
                        unit,
                        shared,
                        shared_units,
                        indices_dtype,
                    )
                    for bi in range(len(cells))
                ]
                run_parallel(_place_job, place_jobs, cpus, mode="processes")
                seam_jobs = [
                    (shared, i, int(k), unit, data_target, indices_target, indices_dtype)
                    for i, k in enumerate(shared_units)
                ]
                run_parallel(_seam_job, seam_jobs, cpus, mode="processes")
            else:
                placer = _Placer(data_arr, indices_arr, unit, indices_dtype)
                place_jobs_ic = [
                    (buckets, bi, int(flat_edges[bi]), int(flat_edges[bi + 1])) for bi in range(len(cells))
                ]
                pipeline_parallel(_place_band, placer.consume, place_jobs_ic, cpus, mode="processes", ordered=True)
                placer.flush()
    finally:
        buckets.remove()
        shared.remove()

    return sp_group
