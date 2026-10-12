from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from annizarr._core._config import DENSE_CHUNK

if TYPE_CHECKING:
    from collections.abc import Sequence

    import zarr
    from numpy.typing import NDArray

    from annizarr._core._config import ChunkConfig

logger = logging.getLogger(__name__)

# read as `_layout.BATCH_BYTES` (module attribute, not a `from`-import) so tests can monkeypatch it.
# working-set budget per band worker: wall time was flat from 64 to 256MiB at 600k cells and the
# larger budget is for very large stores and large chunks, where per-band overhead matters more
BATCH_BYTES = 512 * 1024 * 1024

# per-job payload budget for the read-ahead pipeline: a pickled payload crossing the process
# boundary costs a copy, so it stays well below BATCH_BYTES. Never below one write unit.
PIPELINE_BATCH_BYTES = 64 * 1024 * 1024

# measured on Linux arm64 with numpy, scipy, pandas, h5py, zarr and anndata imported
WORKER_BASELINE_BYTES = 119 * 1024 * 1024

# zarr's write-side buffers for a whole-chunk write, as a fraction of the band: 99MB on a 205MB
# band in a worker on Linux, up to 113MB on a 137MB band in-process on macOS (2 or 12 codec threads)
WRITE_COPY_FRACTION = 0.85

# working set per nonzero of a band for each pass, itemized (measured 2026-10-08, zarr 3.4.1):
# 8 band (float32 data + int32 indices) + 4 source positions + 8 band ids + 12 argsort
# + 8 permuted + 4 slice
SCATTER_BYTES_PER_NNZ = 44
# 12 read back (value, target, source) + 8 argsort + 8 permuted copies
PLACE_BYTES_PER_NNZ = 28
# 8 band + 8 float64 copy scaled in place + 8 repeated factors + 4 float32 result
LOGNORM_BYTES_PER_NNZ = 28
# 8 CSR batch + 8 per-group copy
BUCKET_BYTES_PER_NNZ = 16
# int32 indices only, plus the bincount input
COUNT_BYTES_PER_NNZ = 8
# 4 data + 8 float64 copy + 8 cumulative sum
ROW_SUM_BYTES_PER_NNZ = 20

__all__ = [
    "BATCH_BYTES",
    "BUCKET_BYTES_PER_NNZ",
    "COUNT_BYTES_PER_NNZ",
    "LOGNORM_BYTES_PER_NNZ",
    "PIPELINE_BATCH_BYTES",
    "PLACE_BYTES_PER_NNZ",
    "ROW_SUM_BYTES_PER_NNZ",
    "SCATTER_BYTES_PER_NNZ",
    "WORKER_BASELINE_BYTES",
    "WRITE_COPY_FRACTION",
    "DenseLayout",
    "band_plan",
    "check_chunk_axis",
    "check_write_unit",
    "copy_bytes_per_item",
    "dense_chunks",
    "dense_shards",
    "major_bands",
    "plan_bands",
    "sparse_flat_chunk",
    "write_grid",
    "x_compressors",
]


def write_grid(arr: zarr.Array[Any]) -> tuple[int, ...]:
    # a partial-shard write read-modify-writes the whole shard, so writers must partition here.
    return arr.shards if arr.shards is not None else arr.chunks


def check_write_unit(arr: zarr.Array[Any]) -> None:
    # a band is never smaller than one write unit, so no budget can hold below it
    unit_bytes = math.prod(write_grid(arr)) * arr.dtype.itemsize
    if unit_bytes > BATCH_BYTES:
        logger.warning(
            f"one write unit of '{arr.path}' is {unit_bytes / 1024**2:.0f}MB, above the {BATCH_BYTES / 1024**2:.0f}MB "
            "band budget; each worker holds at least one unit, so pass smaller chunks to bound memory."
        )


def copy_bytes_per_item(*itemsizes: int) -> int:
    # a copy pass holds the band plus zarr's write-side copy of it
    band = sum(itemsizes)
    return band + math.ceil(band * WRITE_COPY_FRACTION)


def plan_bands(
    total: int, *, unit: int, bytes_per_item: int, cpus: int, grain: int | None = None, budget: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Split ``[0, total)`` into bands of whole write units sized to the band budget.

    As many bands as the budget needs, never fewer than ``cpus`` so every worker has one, and
    never below one unit. A band is also a multiple of the source ``grain`` when that fits
    the budget and still leaves a band per worker, so no source chunk is read by two bands.
    """
    if total <= 0:
        return ()
    budget = BATCH_BYTES if budget is None else budget
    cpus = max(cpus, 1)
    step = max(1, unit)
    if grain is not None and grain > 0:
        aligned = math.lcm(step, grain)
        if aligned * bytes_per_item <= budget and math.ceil(total / aligned) >= min(cpus, math.ceil(total / step)):
            step = aligned
    n_bands = max(math.ceil(total * bytes_per_item / budget), cpus)
    band = (total // n_bands) // step * step
    cap = max(1, budget // max(1, step * bytes_per_item)) * step
    return band_plan(total, max(step, min(band, cap)))


def major_bands(indptr: Sequence[int] | NDArray[Any], *, max_nnz: int, cpus: int) -> tuple[tuple[int, int], ...]:
    """Split the major axis of a compressed matrix into bands of at most ``max_nnz`` nonzeros.

    Edges follow ``indptr`` exactly rather than the average nonzeros per row, so a dense block
    of rows never overflows a band. A band holds at least one row, and there are never fewer
    than ``cpus`` bands while the nonzeros can be split that far.
    """
    n_major = len(indptr) - 1
    if n_major <= 0:
        return ()
    nnz = int(indptr[-1])
    target = max(1, min(max_nnz, math.ceil(nnz / max(cpus, 1))))
    edges = [0]
    while edges[-1] < n_major:
        start = edges[-1]
        nxt = bisect.bisect_right(indptr, int(indptr[start]) + target) - 1
        edges.append(min(max(nxt, start + 1), n_major))
    return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))


def x_compressors() -> tuple[object, ...]:
    from zarr.codecs import BloscCodec

    return (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)  # zarr's default has no shuffle


@dataclass(frozen=True, slots=True)
class DenseLayout:
    chunks: tuple[int, int]
    shards: tuple[int, int] | None
    block: tuple[int, int]


def dense_shards(row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int) -> DenseLayout:
    if factor <= 1:
        return DenseLayout(chunks=(row_chunk, col_chunk), shards=None, block=(row_chunk, col_chunk))
    # zarr requires shard shape to be an integer multiple of the chunk shape
    rf = min(factor, math.ceil(n_rows / row_chunk))
    cf = min(factor, math.ceil(n_cols / col_chunk))
    shard_row = row_chunk * rf
    shard_col = col_chunk * cf
    return DenseLayout(chunks=(row_chunk, col_chunk), shards=(shard_row, shard_col), block=(shard_row, shard_col))


def band_plan(n_rows: int, band_rows: int) -> tuple[tuple[int, int], ...]:
    if n_rows <= 0 or band_rows <= 0:
        return ()
    return tuple((r0, min(r0 + band_rows, n_rows)) for r0 in range(0, n_rows, band_rows))


def dense_chunks(chunks: ChunkConfig, n_rows: int, n_cols: int) -> tuple[int, int]:
    """Exact (row, col) chunk shape for a dense matrix: the axis chunks, DENSE_CHUNK when unset."""
    return min(chunks.row_chunk or DENSE_CHUNK, n_rows), min(chunks.col_chunk or DENSE_CHUNK, n_cols)


def check_chunk_axis(chunks: ChunkConfig, layout: str) -> None:
    # sparse chunks follow the major axis only; the other axis's chunk setting has no meaning here
    if layout == "csr" and chunks.col_chunk is not None:
        logger.warning("col_chunk (--col-chunk) is ignored for csr output, whose chunks follow rows; use row_chunk.")
    if layout == "csc" and chunks.row_chunk is not None:
        logger.warning("row_chunk (--row-chunk) is ignored for csc output, whose chunks follow columns; use col_chunk.")


def sparse_flat_chunk(chunks: ChunkConfig, *, csr: bool, nnz: int, n_major: int) -> int:
    """Chunk length for a sparse matrix's flat ``data``/``indices`` arrays.

    With the major-axis chunk set (``row_chunk`` for CSR, ``col_chunk`` for CSC) it is about
    that many rows' or columns' worth of nonzeros, sized from the average; otherwise
    ``nnz_chunk``. Always within ``[1, nnz]``.
    """
    check_chunk_axis(chunks, "csr" if csr else "csc")
    per_major = chunks.row_chunk if csr else chunks.col_chunk
    if per_major is None or n_major <= 0:
        target = chunks.nnz_chunk
    else:
        target = math.ceil(per_major * nnz / n_major)
    return max(1, min(target, max(1, nnz)))
