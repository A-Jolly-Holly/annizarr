from __future__ import annotations

from typing import TYPE_CHECKING, Any

import zarr

from annizarr._core import _layout
from annizarr._core._layout import check_write_unit, dense_shards, write_grid, x_compressors
from annizarr._core._runtime import run_parallel
from annizarr._core._zarr import matrix_shape
from annizarr._writers._encoding import sparse_shards, suppress_autoshard_warning
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from annizarr._core._config import AppConfig

__all__ = ["copy_matrix"]


def copy_matrix(
    node: Any, dst_root: Any, key: str, cfg: AppConfig, *, rechunk: bool, codecs: Mapping[str, Any] | None = None
) -> None:
    """Copy one matrix element (dense array or CSR/CSC group) to ``dst_root[key]`` in parallel bands.

    ``rechunk=False`` keeps the source's chunks, shards and codecs exactly; ``rechunk=True`` lays
    it out by ``cfg.chunks`` and re-encodes at ``cfg.chunks.clevel`` when that is set. ``codecs``
    (compressors by array name: ``""`` for a dense matrix, ``"data"`` and ``"indices"`` for sparse)
    overrides both, for a copy that must keep another element's codecs.
    """

    def compressors(arr: Any, name: str = "") -> Any:
        if codecs is not None:
            return codecs[name]
        return x_compressors(cfg.chunks.clevel) if rechunk and cfg.chunks.clevel is not None else arr.compressors

    parent_path, _, name = key.rpartition("/")
    parent = dst_root[parent_path] if parent_path else dst_root

    if isinstance(node, zarr.Array):
        n_rows, n_cols = node.shape
        if rechunk:
            row_chunk, col_chunk = _layout.dense_chunks(cfg.chunks, n_rows, n_cols)
            layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.shard_factor)
            out_chunks, shards = layout.chunks, layout.shards
        else:
            out_chunks = (node.chunks[0], node.chunks[1])
            node_shards = node.shards
            shards = (node_shards[0], node_shards[1]) if node_shards is not None else None
        out = parent.require_array(
            name,
            shape=node.shape,
            dtype=node.dtype,
            chunks=out_chunks,
            shards=shards,
            compressors=compressors(node),
            overwrite=True,
        )
        out.attrs.update(dict(node.attrs))
        check_write_unit(out)
        block_row, block_col = write_grid(out)
        jobs = [
            (node, out, r0, min(r0 + block_row, n_rows), c0, min(c0 + block_col, n_cols))
            for r0 in range(0, n_rows, block_row)
            for c0 in range(0, n_cols, block_col)
        ]
        run_parallel(_copy_block, jobs, cfg.chunks.cpus)
        return

    enc = node.attrs.get("encoding-type")
    if enc not in ("csr_matrix", "csc_matrix"):
        raise ConversionError(f"cannot copy '{key}': unsupported encoding {enc!r}.")
    g = parent.require_group(name)
    g.attrs.update(dict(node.attrs))
    nnz = int(node["data"].shape[0])
    # rechunk=False (copy-as-is) preserves the source's exact chunk+shard shape, so a copy
    # never silently drops sharding.
    if rechunk:
        n_rows, n_cols = matrix_shape(node)
        csr = enc == "csr_matrix"
        flat = _layout.sparse_flat_chunk(cfg.chunks, csr=csr, nnz=nnz, n_major=n_rows if csr else n_cols)
        out_shards = sparse_shards(cfg.chunks.auto_shard)
    else:
        flat = node["data"].chunks[0]
        out_shards = node["data"].shards
    for arr_name in ("data", "indices"):
        src_a = node[arr_name]
        with suppress_autoshard_warning(rechunk and cfg.chunks.auto_shard):
            out = g.require_array(
                arr_name,
                shape=src_a.shape,
                dtype=src_a.dtype,
                chunks=(flat,),
                shards=out_shards,
                compressors=compressors(src_a, arr_name),
                overwrite=True,
            )
        out.attrs.update(dict(src_a.attrs))
        check_write_unit(out)
        step = write_grid(out)[0]
        bytes_per_item = _layout.copy_bytes_per_item(src_a.dtype.itemsize)
        segments = _layout.plan_bands(
            nnz, unit=step, bytes_per_item=bytes_per_item, cpus=cfg.chunks.cpus, grain=src_a.chunks[0]
        )
        flat_jobs = [(src_a, out, s0, s1) for s0, s1 in segments]
        run_parallel(_copy_flat, flat_jobs, cfg.chunks.cpus)
    ip = node["indptr"]
    out = g.require_array("indptr", shape=ip.shape, dtype=ip.dtype, chunks=ip.shape, overwrite=True)
    out.attrs.update(dict(ip.attrs))
    out[:] = ip[:]


def _copy_block(src: Any, dst: Any, r0: int, r1: int, c0: int, c1: int) -> None:
    dst[r0:r1, c0:c1] = src[r0:r1, c0:c1]


def _copy_flat(src: Any, dst: Any, s0: int, s1: int) -> None:
    dst[s0:s1] = src[s0:s1]
