from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from annizarr._core._config import AppConfig
from annizarr._core._runtime import configure_runtime, stage
from annizarr._core._zarr import matrix_keys, matrix_shape
from annizarr._sources._readers import as_reader
from annizarr._storage import op_metadata, open_store_rw, store_path
from annizarr._writers._matrix import write_matrix
from annizarr._writers._sparse import _local_tmp_dir
from annizarr.errors import ConversionError
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    import pandas as pd
    import zarr

    from annizarr.typing import Layout, StoreLike

logger = logging.getLogger(__name__)

SOURCE_ATTR = "annizarr-source"  # the matrix a layer was copied from; informational, nothing depends on it
_LAYOUTS = ("csc", "csr", "dense")


def add_layer(
    store: StoreLike,
    *,
    layer: str,
    matrix: str = "X",
    layout: Layout = "csc",
    overwrite: bool = False,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
    commit: bool = True,
) -> OpResult:
    """Copy a matrix into ``layers/<layer>`` in another layout, values untouched.

    The usual use is a CSC copy of X for fast per-gene reads. The copy streams band by band with
    the same memory bound and parallelism as every write; nothing is normalized or transformed
    (do that in scanpy or rapids-singlecell, then copy the result with ``matrix="layers/<k>"``).

    Parameters
    ----------
    store
        Existing AnnData zarr store, or an Icechunk repo (path or :class:`~annizarr.ic.Repo`) to update, in place.
    layer
        Name of the new layer.
    matrix
        Matrix to copy: ``"X"``, ``"layers/<name>"``, or ``"raw/X"`` when ``raw/var`` matches ``var``.
    layout
        On-disk layout of the new layer: ``"csc"``, ``"csr"``, or ``"dense"``. Chunking
        follows ``cfg.chunks`` the way X's does on convert: ``row_chunk`` (csr), ``col_chunk``
        (csc), else ``nnz_chunk``; ``row_chunk`` by ``col_chunk`` tiles for dense.
    overwrite
        Replace an existing ``layers/<layer>`` instead of erroring.
    cfg
        Resolved configuration; ``None`` uses the :class:`~annizarr.config.AppConfig` defaults.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet.
        Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op, ``matrix`` and the layer.
    commit
        ``False`` stages the layer without committing, so several ops land in one commit made
        with ``store.commit()``. Requires ``store`` to be a :class:`~annizarr.ic.Repo`.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``matrix`` is not a matrix element of the store or not cells x var, ``matrix`` is the
        layer being written, ``layers/<layer>`` exists and ``overwrite`` is not set, or
        ``layout`` is not one of ``"csc"``, ``"csr"``, ``"dense"``.
    """
    if cfg is None:
        cfg = AppConfig()
    if layout not in _LAYOUTS:
        raise ConversionError(f"add-layer layout must be one of {', '.join(_LAYOUTS)}; got '{layout}'.")
    if matrix == f"layers/{layer}":
        raise ConversionError(f"cannot copy {matrix} onto itself; name the new layer something else.")
    configure_runtime(cfg.chunks.cpus)
    root, finalize = open_store_rw(
        store,
        cfg,
        commit_message=message or f"annizarr add-layer {matrix} → layers/{layer} ({layout})",
        commit_metadata=op_metadata("add-layer", matrix=matrix, layout=layout),
        branch=branch,
        commit=commit,
    )
    _write_layer(root, cfg, layer=layer, matrix=matrix, layout=layout, overwrite=overwrite)
    n_obs, n_vars = matrix_shape(root["X"])
    snapshot_id = finalize()
    return OpResult(path=store_path(store), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)


def _write_layer(root: zarr.Group, cfg: AppConfig, *, layer: str, matrix: str, layout: Layout, overwrite: bool) -> None:
    keys = matrix_keys(root)
    if "X" not in keys:
        raise ConversionError("no X in store — not an AnnData zarr store?")
    if matrix not in keys:
        raise ConversionError(f"matrix '{matrix}' is not a matrix element ({keys}).")
    expected = matrix_shape(root["X"])
    source = root[matrix]
    if matrix_shape(source) != expected:
        raise ConversionError(f"{matrix} is {matrix_shape(source)}, not cells x var {expected}; a layer must match X.")
    if matrix == "raw/X":
        from anndata.io import read_elem

        raw_var: pd.DataFrame = read_elem(root["raw/var"])
        var: pd.DataFrame = read_elem(root["var"])
        if not raw_var.index.equals(var.index):
            raise ConversionError("raw/var differs from var (names or order), so raw/X cannot become a layer.")

    exists = f"layers/{layer}" in keys
    if exists and not overwrite:  # refused before anything is written, so the store is left as it was
        raise ConversionError(f"layers/{layer} already exists; pass overwrite=True to replace it.")
    layers = root.require_group("layers")
    if "encoding-type" not in dict(layers.attrs):
        layers.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    if exists:
        del layers[layer]

    reader = as_reader(source, cfg=cfg, tmp_dir=_local_tmp_dir(layers))
    try:
        with stage(f"Copying {matrix} → layers/{layer} ({layout})"):  # no nnz: a dense source would count it
            write_matrix(layers, layer, reader, replace(cfg, io=replace(cfg.io, layout=layout)))
    finally:
        reader.close()
    layers[layer].attrs[SOURCE_ATTR] = matrix
