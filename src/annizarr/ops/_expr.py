from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from annizarr._core import _layout
from annizarr._core._config import AppConfig
from annizarr._core._runtime import configure_runtime, map_parallel, stage
from annizarr._core._zarr import get_group, shape_attr
from annizarr._sources._readers import ScaledReader, as_reader
from annizarr._storage import open_store_rw
from annizarr._writers._matrix import write_matrix
from annizarr._writers._sparse import _local_tmp_dir
from annizarr.errors import ConversionError
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    from typing import Literal

    import zarr
    from numpy.typing import NDArray

    from annizarr._core._config import ChunkConfig
    from annizarr._sources._readers import Reader
    from annizarr.typing import Layout, PathLike

logger = logging.getLogger(__name__)

_TARGET_SUM_ATTR = "annizarr_target_sum"
_TARGET_SUM_ATTR_LEGACY = "zarrsmith_target_sum"  # written by pre-merge zarrsmith; read-compat only


def target_sum_attr(attrs: Any) -> float | None:
    value = attrs.get(_TARGET_SUM_ATTR, attrs.get(_TARGET_SUM_ATTR_LEGACY))
    return None if value is None else float(value)


def add_expr(
    store: PathLike,
    *,
    layout: Layout = "csc",
    layer: str = "gexp",
    target_sum: float = 1e4,
    overwrite: bool = False,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Add a log-normalized expression layer (``layers/<layer>``) derived from CSR X.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store to update, in place.
    layout
        On-disk layout of the new layer: ``"csr"``, ``"csc"``, or ``"dense"``. Chunking
        follows ``cfg.chunks`` the way X's does on convert: ``row_chunk`` (csr), ``col_chunk``
        (csc), else ``nnz_chunk``; ``row_chunk`` by ``col_chunk`` tiles for dense.
    layer
        Layer name.
    target_sum
        Library-size normalization target (per-row sum after normalization).
    overwrite
        Replace an existing ``layers/<layer>`` instead of erroring.
    cfg
        Resolved configuration; ``None`` uses the :class:`~annizarr.config.AppConfig` defaults.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet.
        Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op, ``layout``, and the layer.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``store`` has no CSR X, ``layers/<layer>`` already exists and ``overwrite`` is
        not set, or ``layout`` is not one of ``"csr"``, ``"csc"``, ``"dense"``.
    """
    if cfg is None:
        cfg = AppConfig()
    configure_runtime(cfg.chunks.cpus)
    commit_message = message or f"annizarr add-expr {layout} → layers/{layer}"
    root, finalize = open_store_rw(store, cfg, commit_message=commit_message, branch=branch)
    write_expr_layer(root, cfg, layout=layout, layer=layer, target_sum=target_sum, overwrite=overwrite)
    n_obs, n_vars = shape_attr(get_group(root, "X"))
    snapshot_id = finalize()
    return OpResult(path=str(store), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)


def write_expr_layer(
    root: zarr.Group,
    cfg: AppConfig,
    *,
    layout: Layout = "csc",
    layer: str = "gexp",
    target_sum: float = 1e4,
    overwrite: bool = False,
) -> None:
    # does not finalize the store — the caller owns that, so sort() can re-derive a lone
    # gexp layer in the same commit as the sort itself.
    if layout not in ("csc", "dense", "csr"):
        raise ConversionError(f"add-expr layout must be csc, dense, or csr; got '{layout}'.")
    if "X" not in root:
        raise ConversionError("no X in store — not an AnnData zarr store?")
    x = get_group(root, "X")
    if x.attrs.get("encoding-type") != "csr_matrix":
        raise ConversionError(f"add-expr requires CSR X; got encoding {x.attrs.get('encoding-type')!r}.")

    n_obs, n_vars = shape_attr(x)
    if max(n_obs, n_vars) > 2**31 - 1:
        raise ConversionError("add-expr supports up to 2^31-1 cells/genes.")

    layers = root.require_group("layers")
    if "encoding-type" not in dict(layers.attrs):
        layers.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    if layer in layers:
        if not overwrite:
            raise ConversionError(f"layers/{layer} already exists; pass overwrite=True to replace it.")
        del layers[layer]

    # the layer is X read through a scaling reader and written by the ordinary writer for its
    # layout, so csr, csc and dense all inherit the writers' memory bound and parallelism
    x_reader = as_reader(x, cfg=cfg, tmp_dir=_local_tmp_dir(layers))
    try:
        with stage(f"Computing per-cell normalisation factors (target_sum={target_sum:g})"):
            factors = lognorm_factors(x_reader, target_sum)
        scaled = ScaledReader(x_reader, factors)
        layer_cfg = replace(cfg, io=replace(cfg.io, layout=layout))
        with stage(f"Writing layers/{layer} ({layout}, nnz={x_reader.nnz})"):
            write_matrix(layers, layer, scaled, layer_cfg)
    finally:
        x_reader.close()
    layers[layer].attrs[_TARGET_SUM_ATTR] = float(target_sum)


def _row_sums(reader: Reader, s0: int, s1: int, local_indptr: NDArray[np.int64], close: bool) -> NDArray[np.float64]:
    # cumulative sums in float64 so an empty row at a band edge never truncates its neighbour
    try:
        seg = reader.flat_data(s0, s1).astype(np.float64)
    finally:
        if close:
            reader.close()
    cs = np.concatenate(([0.0], np.cumsum(seg)))
    return np.asarray(cs[local_indptr[1:]] - cs[local_indptr[:-1]])


def lognorm_factors(reader: Reader, target_sum: float) -> NDArray[np.float64]:
    """Per-row scale factors ``target_sum / row_sum`` (0 for empty rows), one parallel pass over bands."""
    indptr = reader.indptr
    # a cheap pass: the small pipeline budget keeps it from holding a full band per worker
    budget = min(_layout.BATCH_BYTES, _layout.PIPELINE_BATCH_BYTES)
    bands = _layout.major_bands(indptr, max_nnz=budget // _layout.ROW_SUM_BYTES_PER_NNZ, cpus=reader.cpus)
    mode: Literal["threads", "processes"] = "threads" if reader.thread_safe else "processes"
    jobs = [
        (reader, int(indptr[b0]), int(indptr[b1]), indptr[b0 : b1 + 1] - indptr[b0], mode == "processes")
        for b0, b1 in bands
    ]
    parts = map_parallel(_row_sums, jobs, reader.cpus, mode=mode)
    sums = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    factors = np.zeros(len(indptr) - 1, dtype=np.float64)
    nz = sums > 0
    factors[nz] = target_sum / sums[nz]
    return factors


@dataclass(frozen=True, slots=True)
class GexpLayout:
    layout: Layout
    row_chunk: int | None
    col_chunk: int | None
    nnz_chunk: int | None
    target_sum: float | None

    def chunks(self, chunks: ChunkConfig) -> ChunkConfig:
        # pins the existing layer's chunking so a re-derived layer is laid out like the original
        if self.layout == "dense":
            return replace(chunks, row_chunk=self.row_chunk, col_chunk=self.col_chunk)
        return replace(chunks, row_chunk=None, col_chunk=None, nnz_chunk=self.nnz_chunk or chunks.nnz_chunk)


def introspect_gexp(node: Any) -> GexpLayout:
    import zarr

    target_sum = target_sum_attr(node.attrs)
    if isinstance(node, zarr.Array):
        return GexpLayout("dense", int(node.chunks[0]), int(node.chunks[1]), None, target_sum)
    enc = node.attrs.get("encoding-type")
    layout_map: dict[str, Layout] = {"csr_matrix": "csr", "csc_matrix": "csc"}
    layout = layout_map.get(enc)
    if layout is None:
        raise ConversionError(f"cannot re-derive layers/gexp: unsupported encoding {enc!r}.")
    return GexpLayout(layout, None, None, int(node["data"].chunks[0]), target_sum)
