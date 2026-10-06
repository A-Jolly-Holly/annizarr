from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from annizarr.errors import ValidationError

if TYPE_CHECKING:
    from annizarr.typing import Layout

logger = logging.getLogger(__name__)

BackendMode = Literal["zarr", "icechunk"]

DENSE_CHUNK = 2048  # rows and columns per chunk for dense X when row_chunk/col_chunk are unset

__all__ = [
    "DENSE_CHUNK",
    "AppConfig",
    "ChunkConfig",
    "ConcatConfig",
    "GroupingConfig",
    "IOConfig",
    "apply_cli_overrides",
]


@dataclass(frozen=True, slots=True)
class IOConfig:
    """Input/output toggles shared by every op.

    Parameters
    ----------
    overwrite
        Replace an existing output path/store instead of erroring.
    consolidate_metadata
        Consolidate zarr metadata into one object after writing (plain zarr only).
    layout
        On-disk layout for X (and layers): ``"csr"``, ``"csc"``, or ``"dense"``.
    lazy
        Load h5ad input in lazy (HDF5-streamed) mode instead of eagerly. ``True`` (the
        default) streams the input band by band; ``False`` loads it whole into memory
        first. Ignored for in-memory/10x input (always eager).
    backend
        ``"zarr"`` writes a plain on-disk store; ``"icechunk"`` writes through a
        transactional, versioned Icechunk repository (one commit per op). Icechunk
        targets are a local path or an ``s3://bucket/prefix`` URL (env credentials).
    """

    overwrite: bool = False
    consolidate_metadata: bool = False
    layout: Layout = "csr"
    lazy: bool = True
    backend: BackendMode = "zarr"


def _all_cpus() -> int:
    return os.cpu_count() or 1


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunk/shard sizing and worker count for matrix writes.

    Parameters
    ----------
    row_chunk
        Rows per chunk. Exact for dense X (``DENSE_CHUNK`` when unset); for CSR output,
        about this many cells' worth of nonzeros per ``data``/``indices`` chunk, sized
        from the average nonzeros per row. ``None`` (the default) leaves CSR chunking to
        ``nnz_chunk``. Ignored for CSC.
    col_chunk
        Columns per chunk. Exact for dense X (``DENSE_CHUNK`` when unset); for CSC output,
        about this many genes' worth of nonzeros per chunk. ``None`` (the default) leaves
        CSC chunking to ``nnz_chunk``. Ignored for CSR.
    nnz_chunk
        Flat ``data``/``indices`` chunk length (nonzeros) for sparse output when the
        matching axis chunk above is unset.
    cpus
        Workers for parallel matrix chunk writes: threads for a thread-safe reader
        (in-memory or zarr-backed), processes for a lazy h5py-backed one (not thread-safe).
        Defaults to every available CPU core (``os.cpu_count()``); pass a lower number to
        leave headroom on a shared/HPC host.
    shard_factor
        Pack this many chunks per shard along each axis of dense X (``1`` = no
        sharding; sparse output ignores it). See :func:`annizarr._core._layout.dense_shards`.
    auto_shard
        Shard our own 1-D sparse arrays (``data``/``indices`` of X, layers, raw.X, and the
        add-expr/rechunk/sort-created ones) with zarr's ``shards="auto"``, and set
        ``ad.settings.auto_shard_zarr_v3`` around every ``write_elem`` call this op makes
        (obs/var columns, obsm, uns, …), so anndata's own writes are auto-sharded too.
        Cuts object/file count on remote or many-small-chunk stores at a small write-time
        cost (see :func:`annizarr._writers._encoding.sparse_shards`). Dense X is unaffected
        — it keeps the explicit ``shard_factor`` above, never ``shards="auto"``. Default
        chosen from a read-latency benchmark (see
        ``benchmarking_results/autoshard/README.md``); ``False`` reproduces the pre-autoshard
        on-disk layout exactly.
    """

    row_chunk: int | None = None
    col_chunk: int | None = None
    nnz_chunk: int = 1_000_000
    cpus: int = field(default_factory=_all_cpus)
    shard_factor: int = 1
    auto_shard: bool = False


@dataclass(frozen=True, slots=True)
class GroupingConfig:
    """Sort + partition X by one or more obs columns, for ``convert --sort-by``."""

    enabled: bool = False
    sort_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConcatConfig:
    """obs-column policy for multi-file ``convert``/``concat``."""

    obs_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """The resolved configuration passed to every op.

    Parameters
    ----------
    io
        Input/output toggles.
    chunks
        Chunk/shard sizing and worker count.
    grouping
        ``convert --sort-by`` sort/partition settings.
    concat
        Multi-file concat obs-column policy.
    """

    io: IOConfig = IOConfig()
    chunks: ChunkConfig = ChunkConfig()
    grouping: GroupingConfig = GroupingConfig()
    concat: ConcatConfig = ConcatConfig()


def _normalize_layout(value: str) -> Layout:
    mode = value.lower().strip()
    allowed = {"csr", "csc", "dense"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationError(f"Invalid io.layout '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]  # mode is in `allowed`, a subset of Layout's literals


def _normalize_backend(value: str) -> BackendMode:
    mode = value.lower().strip()
    allowed = {"zarr", "icechunk"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationError(f"Invalid io.backend '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]  # mode is in `allowed`, a subset of BackendMode's literals


def _validate_config(config: AppConfig) -> AppConfig:
    io = replace(
        config.io,
        layout=_normalize_layout(config.io.layout),
        backend=_normalize_backend(config.io.backend),
    )
    sort_by = config.grouping.sort_by  # may arrive as a list or bare string; freeze to a tuple
    if isinstance(sort_by, str):
        sort_by = (sort_by,)
    grouping = replace(config.grouping, sort_by=tuple(sort_by))
    if grouping.enabled and not grouping.sort_by:
        raise ValidationError("grouping.enabled is true but grouping.sort_by is empty.")
    obs_columns = config.concat.obs_columns  # same list-or-string freeze as sort_by
    if isinstance(obs_columns, str):
        obs_columns = (obs_columns,)
    concat = replace(config.concat, obs_columns=tuple(obs_columns))
    for name in ("row_chunk", "col_chunk"):
        axis_chunk = getattr(config.chunks, name)
        if axis_chunk is not None and axis_chunk < 1:
            raise ValidationError(f"chunks.{name} must be >= 1; got {axis_chunk}.")
    if config.chunks.nnz_chunk < 1:
        raise ValidationError(f"chunks.nnz_chunk must be >= 1; got {config.chunks.nnz_chunk}.")
    if config.chunks.shard_factor < 1:
        raise ValidationError(f"chunks.shard_factor must be >= 1 (1 = no sharding); got {config.chunks.shard_factor}.")
    return replace(config, io=io, grouping=grouping, concat=concat)


def apply_cli_overrides(
    config: AppConfig,
    *,
    overwrite: bool | None = None,
    consolidate_metadata: bool | None = None,
    layout: str | None = None,
    row_chunk: int | None = None,
    col_chunk: int | None = None,
    nnz_chunk: int | None = None,
    shard_factor: int | None = None,
    auto_shard: bool | None = None,
    cpus: int | None = None,
    lazy: bool | None = None,
    backend: str | None = None,
    sort_by: list[str] | None = None,
    obs_columns: list[str] | None = None,
) -> AppConfig:
    """Apply CLI flag overrides (``None`` = unset) onto a config."""
    io_cfg = config.io
    chunk_cfg = config.chunks
    grouping_cfg = config.grouping
    concat_cfg = config.concat

    if overwrite is not None:
        io_cfg = replace(io_cfg, overwrite=overwrite)
    if consolidate_metadata is not None:
        io_cfg = replace(io_cfg, consolidate_metadata=consolidate_metadata)
    if layout is not None:
        io_cfg = replace(io_cfg, layout=_normalize_layout(layout))
    # None means "don't touch it": --eager is the only lazy flag and defaults to None, so
    # io.lazy keeps its own default unless the flag is actually passed.
    if lazy is not None:
        io_cfg = replace(io_cfg, lazy=lazy)
    if backend is not None:
        io_cfg = replace(io_cfg, backend=_normalize_backend(backend))
    if row_chunk is not None:
        chunk_cfg = replace(chunk_cfg, row_chunk=row_chunk)
    if col_chunk is not None:
        chunk_cfg = replace(chunk_cfg, col_chunk=col_chunk)
    if nnz_chunk is not None:
        chunk_cfg = replace(chunk_cfg, nnz_chunk=nnz_chunk)
    if shard_factor is not None:
        chunk_cfg = replace(chunk_cfg, shard_factor=shard_factor)
    if auto_shard is not None:
        chunk_cfg = replace(chunk_cfg, auto_shard=auto_shard)
    if cpus is not None:
        chunk_cfg = replace(chunk_cfg, cpus=cpus)
    if sort_by is not None:
        grouping_cfg = replace(grouping_cfg, enabled=True, sort_by=tuple(sort_by))
    if obs_columns is not None:
        concat_cfg = replace(concat_cfg, obs_columns=tuple(obs_columns))

    return _validate_config(replace(config, io=io_cfg, chunks=chunk_cfg, grouping=grouping_cfg, concat=concat_cfg))


def resolve_backend_cfg(cfg: AppConfig) -> AppConfig:
    # lazy (io.lazy=True, the default) works into every backend: a lazy, not-thread-safe
    # reader feeding a non-local store (e.g. an Icechunk session) runs the read-ahead
    # pipeline instead of threads/processes (see writer_parallel_mode).
    if cfg.chunks.shard_factor > 1 and cfg.io.layout != "dense":
        logger.warning(
            f"shard_factor={cfg.chunks.shard_factor} only applies to dense X; "
            f"layout={cfg.io.layout!r} is sparse, so sharding is ignored."
        )
    return cfg
