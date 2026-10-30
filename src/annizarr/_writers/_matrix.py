from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from annizarr._writers._dense import write_dense
from annizarr._writers._sparse import write_csc, write_csr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    import zarr

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader
    from annizarr.typing import Layout

__all__ = ["MatrixLayout", "write_matrix"]


def write_matrix(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    # dispatches purely on cfg.io.layout; the reader's backing kind never matters here
    # (only Reader.thread_safe, inside each writer, for the parallel-mode choice).
    if cfg.io.layout == "dense":
        write_dense(group, key, reader, cfg)
    elif cfg.io.layout == "csr":
        write_csr(group, key, reader, cfg)
    else:
        write_csc(group, key, reader, cfg)


@dataclass(frozen=True, slots=True)
class MatrixLayout:
    # how an existing matrix element is chunked, so a rewrite (sort, append) reproduces it
    layout: Layout
    row_chunk: int | None
    col_chunk: int | None
    nnz_chunk: int | None
    shard_factor: int
    auto_shard: bool

    @classmethod
    def of(cls, node: Any) -> MatrixLayout:
        # a chunk as long as the element itself was capped by its size, not chosen: None lets the
        # default apply, so an element that grows is not written in chunks of its old size
        import zarr

        if isinstance(node, zarr.Array):
            rows, cols = (int(c) for c in node.chunks)
            n_rows, n_cols = (int(n) for n in node.shape)
            shard_factor = int(node.shards[0]) // rows if node.shards is not None else 1
            row_chunk, col_chunk = (rows if rows < n_rows else None), (cols if cols < n_cols else None)
            return cls("dense", row_chunk, col_chunk, None, shard_factor, False)
        layouts: dict[Any, Layout] = {"csr_matrix": "csr", "csc_matrix": "csc"}
        layout = layouts.get(node.attrs.get("encoding-type"))
        if layout is None:
            raise ConversionError(
                f"'{node.path}' is not a matrix element (encoding {node.attrs.get('encoding-type')!r})."
            )
        data = node["data"]
        chunk = int(data.chunks[0])
        return cls(layout, None, None, chunk if chunk < data.shape[0] else None, 1, data.shards is not None)

    def cfg(self, cfg: AppConfig) -> AppConfig:
        chunks = replace(
            cfg.chunks,
            row_chunk=self.row_chunk,
            col_chunk=self.col_chunk,
            nnz_chunk=self.nnz_chunk or cfg.chunks.nnz_chunk,
            shard_factor=self.shard_factor,
            auto_shard=self.auto_shard,
        )
        return replace(cfg, io=replace(cfg.io, layout=self.layout), chunks=chunks)
