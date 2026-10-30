from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr

from annizarr._core import _layout
from annizarr._core._runtime import configure_runtime, run_parallel, stage
from annizarr._core._validation import require_matrix
from annizarr._core._zarr import get_group
from annizarr._sources._readers import ConcatReader, CSRFileReader, RemapColumnsReader, as_reader
from annizarr._storage import open_output_store, store_name, store_path
from annizarr._writers._buckets import Buckets
from annizarr._writers._encoding import (
    set_anndata_root_attrs,
    set_raw_group_attrs,
    write_elem,
    write_frame,
    write_settings,
)
from annizarr._writers._matrix import write_matrix
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader
    from annizarr.typing import Layout, StoreLike

logger = logging.getLogger(__name__)


def compute_sort(obs: pd.DataFrame, sort_by: tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    missing = [c for c in sort_by if c not in obs.columns]
    if missing:
        raise ConversionError(f"grouping sort_by columns not found in obs: {missing}. Available: {list(obs.columns)}")

    keys = []
    for col in sort_by:
        codes, _ = pd.factorize(obs[col], sort=True)  # codes follow sorted value order
        if (np.asarray(codes) < 0).any():
            raise ConversionError(f"obs column '{col}' has missing (NaN) values; cannot sort by it.")
        keys.append(np.asarray(codes))

    # np.lexsort treats the LAST key as primary, so reverse to make sort_by[0] primary.
    perm = np.lexsort(keys[::-1]).astype(np.int64)

    obs_sorted = obs.iloc[perm]
    sizes = obs_sorted.groupby(list(sort_by), sort=False, observed=True).size()
    ends = np.cumsum(sizes.to_numpy())
    starts = ends - sizes.to_numpy()
    ranges = sizes.index.to_frame(index=False)
    ranges["start"] = starts.astype(np.int64)
    ranges["end"] = ends.astype(np.int64)
    return perm, ranges


def maybe_sort_adata(adata: ad.AnnData, cfg: AppConfig) -> ad.AnnData:
    if not cfg.grouping.enabled:
        return adata

    sort_by = cfg.grouping.sort_by
    if cfg.io.layout not in ("csr", "dense"):
        raise ConversionError(f"grouping (sort_by) requires layout='csr' or 'dense'; got '{cfg.io.layout}'.")
    if cfg.io.lazy:
        raise ConversionError(
            "grouping (sort_by) requires an eager (in-memory) load; not supported with --lazy yet. Omit --lazy to sort."
        )

    perm, ranges_df = compute_sort(adata.obs, sort_by)
    with stage(f"Sorting {adata.n_obs} cells by {list(sort_by)} ({len(ranges_df)} groups)"):
        adata = adata[perm].copy()  # reorders X/obs/obsm/obsp/layers/raw consistently
    logger.warning(
        f"Rows sorted by {list(sort_by)} into {len(ranges_df)} contiguous groups; "
        "obs/obsm/obsp/layers/raw reordered to match. Store is a plain sorted AnnData "
        "(no tool index); derive ranges from the sorted obs column(s) if needed."
    )
    return adata


def _write_sorted_lazy(
    adata: ad.AnnData,
    output_path: StoreLike,
    cfg: AppConfig,
    *,
    branch: str | None = None,
    message: str | None = None,
    commit_metadata: dict[str, Any] | None = None,
) -> str | None:
    # streams X, every layer, raw/X and obsp into temp per-group buckets (peak RAM one row-batch),
    # unlike maybe_sort_adata's adata[perm].copy() (~2x X in RAM).
    from annizarr._writers._adata import named_layers

    if cfg.io.layout != "csr":
        raise ConversionError(
            f"lazy --sort-by supports layout='csr' only (got '{cfg.io.layout}'). "
            "Pass --eager to sort dense/CSC in memory."
        )
    x = adata.X
    if sp.issparse(x) or getattr(x, "format", None) != "csr":
        got = "in-memory " + type(x).__name__ if sp.issparse(x) else (getattr(x, "format", None) or type(x).__name__)
        raise ConversionError(
            f"lazy --sort-by requires the lazy input's X to be CSR on disk; got {got}. Pass --eager to sort in memory."
        )

    require_matrix(adata)
    # layers and raw/X are written as convert writes them (X's layout and chunking); a graph keeps its own encoding
    matrices: dict[str, tuple[Any, AppConfig]] = {f"layers/{k}": (v, cfg) for k, v in named_layers(adata).items()}
    raw = None
    if adata.raw is not None:
        adata_raw: Any = adata.raw  # anndata 0.13 types Raw.varm against AnnData rather than Raw
        raw = (adata_raw.var, dict(adata_raw.varm))
        matrices["raw/X"] = (adata_raw.X, cfg)
    for k, v in adata.obsp.items():
        matrices[f"obsp/{k}"] = (v, replace(cfg, io=replace(cfg.io, layout=_object_layout(v))))
    sort_by = cfg.grouping.sort_by
    return stream_sorted_store(
        x,
        adata.obs,
        adata.var,
        dict(adata.uns),
        dict(adata.obsm),
        dict(adata.varm),
        dict(adata.varp),
        output_path,
        cfg,
        sort_by=sort_by,
        matrices=matrices,
        raw=raw,
        commit_message=message or f"annizarr convert (sorted by {','.join(sort_by)}) → {store_name(output_path)}",
        commit_metadata=commit_metadata,
        branch=branch,
    )


def _object_layout(matrix: Any) -> Layout:
    fmt = getattr(matrix, "format", None)
    return "csc" if fmt == "csc" else "csr" if fmt == "csr" else "dense"


@dataclass(frozen=True, slots=True)
class _Groups:
    # which sorted group each source row lands in, and each group's source rows in output order
    of_row: NDArray[np.int64]
    rows: tuple[NDArray[np.int64], ...]


def _groups(perm: NDArray[np.int64], ranges: pd.DataFrame, n_obs: int) -> _Groups:
    of_row = np.empty(n_obs, dtype=np.int64)
    rows = []
    for gi, (start, end) in enumerate(zip(ranges["start"].to_numpy(), ranges["end"].to_numpy(), strict=True)):
        group_rows = perm[start:end]
        of_row[group_rows] = gi
        rows.append(group_rows)
    return _Groups(of_row, tuple(rows))


def _bucket_batch(
    reader: Reader,
    s0: int,
    s1: int,
    local_indptr: np.ndarray,
    groups: np.ndarray,
    n_vars: int,
    offsets: np.ndarray,
    buckets: Buckets,
    close: bool,
) -> None:
    # one contiguous row batch split into per-group runs, each run's rows in source order, written
    # at the run's precomputed offset in its group's files
    try:
        data, indices = reader.flat(s0, s1)
    finally:
        if close:
            reader.close()
    batch = sp.csr_matrix((data, indices, local_indptr), shape=(len(groups), n_vars))
    order = np.argsort(groups, kind="stable")
    sorted_g = groups[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_g[1:] != sorted_g[:-1])))
    ends = np.concatenate((starts[1:], [sorted_g.size]))
    for lo, hi in zip(starts, ends, strict=True):
        sub = batch[order[lo:hi]]
        if sub.nnz:
            gi = int(sorted_g[lo])
            buckets.write_at(gi, int(offsets[gi]), data=sub.data, indices=sub.indices.astype(np.int32, copy=False))


def bucket_rows(reader: Reader, groups: _Groups, cfg: AppConfig, *, tmp_dir: str) -> tuple[ConcatReader, Buckets]:
    # the reader's rows in sorted order, as per-group CSR buckets on disk: workers split contiguous row
    # batches by group and write each run at an offset known up front, so nothing passes through this
    # process. The caller removes the buckets once the returned reader is written.
    n_cols = reader.shape[1]
    n_groups = len(groups.rows)
    indptr = reader.indptr
    row_nnz = np.diff(indptr).astype(np.int64)
    indptr_each = [np.concatenate([[0], np.cumsum(row_nnz[r])]).astype(np.int64) for r in groups.rows]
    buckets = Buckets.create(
        n_groups,
        {"data": np.dtype(reader.dtype), "indices": np.dtype(np.int32)},
        dir=tmp_dir,
        prefix="annizarr_sort_",
        sizes=[int(ip[-1]) for ip in indptr_each],
    )
    try:
        batch_nnz = max(1, _layout.BATCH_BYTES // _layout.BUCKET_BYTES_PER_NNZ)
        batches = _layout.major_bands(indptr, max_nnz=batch_nnz, cpus=cfg.chunks.cpus)
        counts = np.stack(
            [np.bincount(groups.of_row[b0:b1], weights=row_nnz[b0:b1], minlength=n_groups) for b0, b1 in batches]
        ).astype(np.int64)
        offsets = np.cumsum(counts, axis=0) - counts
        mode: Literal["threads", "processes"] = "threads" if reader.thread_safe else "processes"
        jobs = [
            (
                reader,
                int(indptr[b0]),
                int(indptr[b1]),
                indptr[b0 : b1 + 1] - indptr[b0],
                groups.of_row[b0:b1],
                n_cols,
                offsets[i],
                buckets,
                mode == "processes",
            )
            for i, (b0, b1) in enumerate(batches)
        ]
        run_parallel(_bucket_batch, jobs, cfg.chunks.cpus, mode=mode)
    except BaseException:
        buckets.remove()
        raise
    sorted_reader = ConcatReader(
        [
            CSRFileReader(
                buckets.path(gi, "data"),
                buckets.path(gi, "indices"),
                indptr_each[gi],
                (len(groups.rows[gi]), n_cols),
                np.dtype(reader.dtype),
            )
            for gi in range(n_groups)
        ]
    )
    return sorted_reader, buckets


def stream_sorted_store(
    x: Any,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict[str, Any],
    obsm: dict[str, Any],
    varm: dict[str, Any],
    varp: dict[str, Any],
    output_path: StoreLike,
    cfg: AppConfig,
    *,
    sort_by: tuple[str, ...],
    matrices: Mapping[str, tuple[Any, AppConfig]] | None = None,
    raw: tuple[pd.DataFrame, dict[str, Any]] | None = None,
    commit_message: str | None = None,
    commit_metadata: dict[str, Any] | None = None,
    branch: str | None = None,
) -> str | None:
    # matrices: "layers/<k>", "raw/X", "obsp/<k>" -> (source, the config to write it with); every one is
    # reordered with X, an obsp graph on both axes. raw: (raw.var, raw.varm), copied as they are.
    configure_runtime(cfg.chunks.cpus)
    matrices = dict(matrices or {})
    n_obs, n_vars = x.shape
    perm, ranges = compute_sort(obs, sort_by)
    groups = _groups(perm, ranges, n_obs)
    new_index = np.empty(n_obs, dtype=np.int64)
    new_index[perm] = np.arange(n_obs, dtype=np.int64)  # an obsp column's position after the sort
    tmp_dir = str(Path(store_path(output_path)).parent)

    ad.settings.zarr_write_format = 3
    logger.info(
        f"Sorting (streamed) → {output_path} (n_obs={n_obs}, n_vars={n_vars}, csr, "
        f"{len(groups.rows)} groups, {len(matrices)} more matrices, backend={cfg.io.backend})"
    )
    t0 = time.perf_counter()

    out = open_output_store(
        output_path,
        cfg,
        commit_message=commit_message or f"annizarr sort → {store_name(output_path)}",
        commit_metadata=commit_metadata,
        branch=branch,
        expected_shape=(n_obs, n_vars),
    )
    try:
        store = out.root
        set_anndata_root_attrs(store)
        with write_settings(cfg.chunks.auto_shard):
            with stage("Writing metadata (sorted obs/obsm; var/varm/varp/uns as-is)"):
                write_frame(store, "obs", obs.iloc[perm])
                write_frame(store, "var", var)
                write_elem(store, "uns", dict(uns))
                write_elem(
                    store, "obsm", {k: (v.iloc[perm] if hasattr(v, "iloc") else v[perm]) for k, v in obsm.items()}
                )
                write_elem(store, "varm", dict(varm))
                write_elem(store, "obsp", {})
                write_elem(store, "varp", dict(varp))
                if any(k.startswith("layers/") for k in matrices):
                    write_elem(store, "layers", {})
                if raw is not None:
                    raw_group = store.require_group("raw")
                    set_raw_group_attrs(raw_group)
                    write_frame(raw_group, "var", raw[0])
                    write_elem(raw_group, "varm", dict(raw[1]))

        for key, (source, key_cfg) in {"X": (x, cfg), **matrices}.items():
            _write_sorted(store, key, source, key_cfg, groups, new_index if key.startswith("obsp/") else None, tmp_dir)
        snapshot_id = out.finalize()
    except BaseException:
        out.abort()
        raise

    logger.info(f"Done in {time.perf_counter() - t0:.1f}s")
    logger.info(
        f"Rows sorted by {list(sort_by)} into {len(groups.rows)} contiguous groups via lazy streamed "
        "bucketing (no matrix ever fully materialised); obs, obsm, layers, raw and obsp reordered to match. "
        "Store is a plain sorted AnnData (no tool index)."
    )
    return snapshot_id


def _write_sorted(
    store: zarr.Group,
    key: str,
    source: Any,
    cfg: AppConfig,
    groups: _Groups,
    new_index: NDArray[np.int64] | None,
    tmp_dir: str,
) -> None:
    # one element at a time, so temp disk holds one element's buckets
    reader = as_reader(source, cfg=cfg, tmp_dir=tmp_dir)
    try:
        with stage(f"Bucketing {key} rows into {len(groups.rows)} groups"):
            sorted_reader, buckets = bucket_rows(reader, groups, cfg, tmp_dir=tmp_dir)
    finally:
        reader.close()
    try:
        out: Reader = sorted_reader if new_index is None else RemapColumnsReader(sorted_reader, new_index)
        out.cpus = cfg.chunks.cpus
        parent_key, _, name = key.rpartition("/")
        with stage(f"Writing {key} (shape={out.shape}, {cfg.io.layout})"):
            write_matrix(get_group(store, parent_key) if parent_key else store, name, out, cfg)
    finally:
        buckets.remove()
