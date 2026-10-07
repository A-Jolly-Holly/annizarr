from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from annizarr._core._config import DENSE_CHUNK

if TYPE_CHECKING:
    import zarr

    from annizarr._core._config import ChunkConfig

logger = logging.getLogger(__name__)

# read as `_layout.BATCH_BYTES` (module attribute, not a `from`-import) so tests can monkeypatch it.
BATCH_BYTES = 256 * 1024 * 1024

# per-job payload budget for the read-ahead pipeline (item 4): smaller than BATCH_BYTES so a
# pickled payload crossing the process/thread boundary stays modest even when BATCH_BYTES is large.
PIPELINE_BATCH_BYTES = 64 * 1024 * 1024

__all__ = [
    "BATCH_BYTES",
    "PIPELINE_BATCH_BYTES",
    "DenseLayout",
    "band_plan",
    "check_chunk_axis",
    "dense_chunks",
    "dense_shards",
    "sparse_flat_chunk",
    "write_grid",
    "x_compressors",
]


def write_grid(arr: zarr.Array[Any]) -> tuple[int, ...]:
    # a partial-shard write read-modify-writes the whole shard, so writers must partition here.
    return arr.shards if arr.shards is not None else arr.chunks


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
