from __future__ import annotations

from typing import TYPE_CHECKING

import zarr

from annizarr._core._config import AppConfig, resolve_backend_cfg
from annizarr._core._runtime import configure_runtime, stage
from annizarr._core._zarr import get_group, has_element, matrix_keys, matrix_shape
from annizarr._storage import (
    check_output_target,
    op_metadata,
    open_input_group,
    open_output_store,
    store_name,
    store_path,
)
from annizarr._writers._copy import copy_matrix
from annizarr.errors import ConversionError
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    from annizarr.typing import StoreLike

_SMALL_ELEMS = ("obs", "var", "uns", "varm", "varp")


def rechunk(
    store: StoreLike,
    *,
    output: StoreLike,
    matrix: str = "X",
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Rewrite one matrix with the configured chunking; stream-copy everything else as-is.

    Parameters
    ----------
    store
        Existing AnnData zarr store, or an Icechunk repo (path or :class:`~annizarr.ic.Repo`) to read from.
    output
        Destination store path or URI, or a :class:`~annizarr.ic.Repo`.
    matrix
        Which matrix to rechunk: ``"X"``, ``"layers/<name>"``, or ``"raw/X"``.
    cfg
        Resolved configuration; ``None`` uses the :class:`~annizarr.config.AppConfig` defaults.
    branch
        Icechunk branch to write ``output`` to; created off the current tip if it
        doesn't exist yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and ``matrix``.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``matrix`` is not a matrix element present on ``store``.
    """
    import anndata as ad
    from anndata.io import read_elem

    from annizarr._writers._encoding import write_elem, write_frame, write_settings

    if cfg is None:
        cfg = AppConfig()
    cfg = resolve_backend_cfg(cfg)
    check_output_target(output, cfg)
    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(store)

    keys = matrix_keys(src)
    if matrix not in keys:
        raise ConversionError(f"matrix '{matrix}' is not a matrix element ({keys}).")

    ad.settings.zarr_write_format = 3
    commit_message = message or f"annizarr rechunk {matrix} → {store_name(output)}"
    out = open_output_store(
        output,
        cfg,
        commit_message=commit_message,
        commit_metadata=op_metadata("rechunk", source=store_name(store)),
        branch=branch,
        expected_shape=matrix_shape(src["X"]),
    )
    try:
        dst = out.root
        dst.attrs.update(dict(src.attrs))

        with write_settings(cfg.chunks.auto_shard):
            with stage("Copying metadata elements"):
                for key in _SMALL_ELEMS:
                    if key not in src:
                        continue
                    if key in ("obs", "var"):  # re-laid out: a 0.12-written index reads back as pd.NA strings
                        write_frame(dst, key, read_elem(src[key]))
                    else:
                        write_elem(dst, key, read_elem(src[key]))
                for key in ("obsm", "obsp"):
                    if key not in src:
                        continue
                    src_group = get_group(src, key)
                    g = dst.require_group(key)
                    g.attrs.update(dict(src_group.attrs))
                    for child in src_group:
                        node = src_group[child]
                        if isinstance(node, zarr.Array) or node.attrs.get("encoding-type") in (
                            "csr_matrix",
                            "csc_matrix",
                        ):
                            copy_matrix(node, dst, f"{key}/{child}", cfg, rechunk=False)
                        else:
                            write_elem(g, child, read_elem(node))
                if has_element(src, "raw"):
                    src_raw = get_group(src, "raw")
                    raw = dst.require_group("raw")
                    raw.attrs.update(dict(src_raw.attrs))
                    if "var" in src_raw:
                        write_frame(raw, "var", read_elem(src_raw["var"]))
                    if "varm" in src_raw:
                        write_elem(raw, "varm", read_elem(src_raw["varm"]))

            if "layers" in src:
                src_layers = get_group(src, "layers")
                layers = dst.require_group("layers")
                layers.attrs.update(dict(src_layers.attrs))

            for key in keys:
                rechunked = key == matrix
                with stage(f"{'Rechunking' if rechunked else 'Copying'} {key}"):
                    copy_matrix(src[key], dst, key, cfg, rechunk=rechunked)

        n_obs, n_vars = matrix_shape(src["X"])
        snapshot_id = out.finalize()
    except BaseException:
        out.abort()
        raise
    return OpResult(path=store_path(output), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)
