from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from annizarr._core._config import AppConfig, resolve_backend_cfg
from annizarr._core._sorting import stream_sorted_store
from annizarr._core._zarr import get_group, matrix_keys
from annizarr._storage import check_output_target, is_remote, op_metadata, open_input_group, store_name, store_path
from annizarr._writers._matrix import MatrixLayout
from annizarr.errors import ConversionError
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from annizarr.typing import StoreLike

logger = logging.getLogger(__name__)


def sort(
    store: StoreLike,
    *,
    output: StoreLike,
    by: Sequence[str],
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Physically sort an existing zarr store by obs column(s) into a new store.

    Parameters
    ----------
    store
        Existing AnnData zarr store, or an Icechunk repo (path or :class:`~annizarr.ic.Repo`), with CSR X, to read from.
    output
        Destination store path or URI, or a :class:`~annizarr.ic.Repo`.
    by
        Obs column name(s) to sort by, primary key first.
    cfg
        Resolved configuration; ``None`` uses the :class:`~annizarr.config.AppConfig` defaults.
    branch
        Icechunk branch to write ``output`` to; created off the current tip if it
        doesn't exist yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the sort keys.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``by`` is empty, ``output`` is a remote (s3://, gs://) URI, or ``store`` has no CSR X.

    Notes
    -----
    Every cells-aligned element moves with the rows: obs, obsm, every layer and ``raw/X`` (each
    in its own encoding and chunking), and every ``obsp`` graph on both axes, so a neighbour
    graph still links the same cells.
    """
    from anndata.io import read_elem, sparse_dataset

    if cfg is None:
        cfg = AppConfig()
    cfg = resolve_backend_cfg(cfg)
    by = tuple(by)
    if is_remote(store_path(output)):
        raise ConversionError("sort streams through local temp stores; a remote output is not supported yet.")
    if not by:
        raise ConversionError("sort requires by=[OBS_COLUMN, ...].")
    if cfg.io.layout != "csr":
        raise ConversionError(f"sort supports layout='csr' only (got '{cfg.io.layout}').")
    check_output_target(output, cfg)

    src = open_input_group(store)
    if "X" not in src:
        raise ConversionError(f"no X in {store_path(store)} — not an AnnData zarr store?")
    if src["X"].attrs.get("encoding-type") != "csr_matrix":
        raise ConversionError(f"sort requires CSR X; got encoding {src['X'].attrs.get('encoding-type')!r}.")

    def _read(key: str) -> Any:
        return read_elem(src[key]) if key in src else {}

    # every other matrix keeps its own layout and chunking; its codec follows cfg, like every array sort writes
    keys = [k for k in matrix_keys(src) if k != "X"]
    if "obsp" in src:
        keys += [f"obsp/{k}" for k in get_group(src, "obsp")]
    matrices = {k: (src[k], MatrixLayout.of(src[k]).cfg(cfg)) for k in keys}
    raw: tuple[Any, dict[str, Any]] | None = None
    if "raw/X" in matrices:
        raw_group = get_group(src, "raw")
        raw = (read_elem(raw_group["var"]), _read("raw/varm"))

    x = sparse_dataset(src["X"])
    n_obs, n_vars = x.shape
    snapshot_id = stream_sorted_store(
        x,
        read_elem(src["obs"]),
        read_elem(src["var"]),
        _read("uns"),
        _read("obsm"),
        _read("varm"),
        _read("varp"),
        output,
        cfg,
        sort_by=by,
        matrices=matrices,
        raw=raw,
        commit_message=message or f"annizarr sort by {','.join(by)} → {store_name(output)}",
        commit_metadata=op_metadata("sort", source=store_name(store)),
        branch=branch,
    )
    return OpResult(path=store_path(output), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)
