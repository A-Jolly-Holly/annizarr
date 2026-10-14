from __future__ import annotations

import logging
import time
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
from annizarr._sources._readers import ConcatReader, CSRFileReader, as_reader
from annizarr._storage import open_output_store
from annizarr._writers._buckets import Buckets
from annizarr._writers._encoding import write_elem, write_frame, write_settings
from annizarr._writers._matrix import write_matrix
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader

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
    adata: ad.AnnData, output_path: Path, cfg: AppConfig, *, branch: str | None = None, message: str | None = None
) -> str | None:
    # streams X into temp per-group CSR stores (peak RAM one row-batch), unlike
    # maybe_sort_adata's adata[perm].copy() (~2x X in RAM).
    if cfg.io.layout != "csr":
        raise ConversionError(
            f"lazy --sort-by supports layout='csr' only (got '{cfg.io.layout}'). "
            "Pass --eager to sort dense/CSC in memory."
        )
    has_layers = any(k is not None for k in adata.layers.keys())  # anndata>=0.13 lists X under None
    if has_layers or adata.raw is not None or len(adata.obsp) > 0:
        raise ConversionError(
            "lazy --sort-by does not reorder layers/raw/obsp yet (they are obs-aligned and "
            "would need their own streamed reorder). Pass --eager to sort in memory, or drop them."
        )
    x = adata.X
    if sp.issparse(x) or getattr(x, "format", None) != "csr":
        got = "in-memory " + type(x).__name__ if sp.issparse(x) else (getattr(x, "format", None) or type(x).__name__)
        raise ConversionError(
            f"lazy --sort-by requires the lazy input's X to be CSR on disk; got {got}. Pass --eager to sort in memory."
        )

    require_matrix(adata)
    sort_by = cfg.grouping.sort_by
    snapshot_id = stream_sorted_store(
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
        commit_message=message or f"annizarr convert (sorted by {','.join(sort_by)}) → {output_path.name}",
        branch=branch,
    )
    return snapshot_id


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


def stream_sorted_store(
    x: Any,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict[str, Any],
    obsm: dict[str, Any],
    varm: dict[str, Any],
    varp: dict[str, Any],
    output_path: Path,
    cfg: AppConfig,
    *,
    sort_by: tuple[str, ...],
    commit_message: str | None = None,
    branch: str | None = None,
    after_write: Callable[[zarr.Group], None] | None = None,
) -> str | None:
    configure_runtime(cfg.chunks.cpus)

    src_reader = as_reader(x, cfg=cfg)
    n_obs, n_vars = x.shape
    x_dtype = x.dtype
    indices_dtype = np.int32  # matches the rest of the writers (fits unless > 2^31 cols)

    perm, ranges = compute_sort(obs, sort_by)
    n_groups = len(ranges)
    starts = ranges["start"].to_numpy()
    ends = ranges["end"].to_numpy()

    group_of_source = np.empty(n_obs, dtype=np.int64)
    group_rows = []
    for gi in range(n_groups):
        rows = perm[starts[gi] : ends[gi]]
        group_of_source[rows] = gi
        group_rows.append(rows)

    indptr = src_reader.indptr
    row_nnz = np.diff(indptr).astype(np.int64)
    n_rows_each = [int(r.size) for r in group_rows]
    indptr_each = [np.concatenate([[0], np.cumsum(row_nnz[r])]).astype(np.int64) for r in group_rows]
    nnz_each = [int(ip[-1]) for ip in indptr_each]

    ad.settings.zarr_write_format = 3
    logger.info(
        f"Sorting (streamed) → {output_path} (n_obs={n_obs}, n_vars={n_vars}, csr, "
        f"{n_groups} groups, backend={cfg.io.backend})"
    )
    t0 = time.perf_counter()

    # workers split contiguous row batches by group and write each run at an offset known up front
    # from the rows' groups and nonzeros, so rows within a group keep source order (== perm's)
    # and nothing passes through this process
    buckets = Buckets.create(
        n_groups,
        {"data": np.dtype(x_dtype), "indices": np.dtype(indices_dtype)},
        dir=str(output_path.parent),
        prefix="annizarr_sort_",
        sizes=nnz_each,
    )
    try:
        batch_nnz = max(1, _layout.BATCH_BYTES // _layout.BUCKET_BYTES_PER_NNZ)
        batches = _layout.major_bands(indptr, max_nnz=batch_nnz, cpus=cfg.chunks.cpus)
        counts = np.stack(
            [np.bincount(group_of_source[b0:b1], weights=row_nnz[b0:b1], minlength=n_groups) for b0, b1 in batches]
        ).astype(np.int64)
        offsets = np.cumsum(counts, axis=0) - counts
        mode: Literal["threads", "processes"] = "threads" if src_reader.thread_safe else "processes"
        jobs = [
            (
                src_reader,
                int(indptr[b0]),
                int(indptr[b1]),
                indptr[b0 : b1 + 1] - indptr[b0],
                group_of_source[b0:b1],
                n_vars,
                offsets[i],
                buckets,
                mode == "processes",
            )
            for i, (b0, b1) in enumerate(batches)
        ]
        with stage(f"Bucketing {n_obs} rows into {n_groups} groups ({len(batches)} batches)"):
            run_parallel(_bucket_batch, jobs, cfg.chunks.cpus, mode=mode)

        out = open_output_store(
            output_path,
            cfg,
            commit_message=commit_message or f"annizarr sort → {output_path.name}",
            branch=branch,
            expected_shape=(n_obs, n_vars),
        )
        try:
            store = out.root
            store.attrs["encoding-type"] = "anndata"
            store.attrs["encoding-version"] = "0.1.0"
            with write_settings(cfg.chunks.auto_shard):
                with stage("Writing metadata (sorted obs/obsm; var/varm/varp/uns as-is)"):
                    write_frame(store, "obs", obs.iloc[perm])
                    write_frame(store, "var", var)
                    write_elem(store, "uns", dict(uns))
                    write_elem(
                        store, "obsm", {k: (v.iloc[perm] if hasattr(v, "iloc") else v[perm]) for k, v in obsm.items()}
                    )
                    write_elem(store, "varm", dict(varm))
                    write_elem(store, "obsp", {})  # empty (non-empty obsp is rejected by callers)
                    write_elem(store, "varp", dict(varp))

                concat_reader = ConcatReader(
                    [
                        CSRFileReader(
                            buckets.path(gi, "data"),
                            buckets.path(gi, "indices"),
                            indptr_each[gi],
                            (n_rows_each[gi], n_vars),
                            np.dtype(x_dtype),
                        )
                        for gi in range(n_groups)
                    ]
                )
                with stage(f"Writing X (n_obs={n_obs}, n_vars={n_vars}, csr, concat {n_groups} groups)"):
                    write_matrix(store, "X", concat_reader, cfg)
                if after_write is not None:
                    after_write(store)
            snapshot_id = out.finalize()
        except BaseException:
            out.abort()
            raise
    finally:
        src_reader.close()
        buckets.remove()

    logger.info(f"Done in {time.perf_counter() - t0:.1f}s")
    logger.info(
        f"Rows sorted by {list(sort_by)} into {n_groups} contiguous groups via lazy streamed "
        "bucketing (X never fully materialised); obs/obsm reordered to match. Store is a plain "
        "sorted AnnData (no tool index)."
    )
    return snapshot_id
