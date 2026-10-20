from __future__ import annotations

import enum
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from annizarr._core._config import AppConfig
from annizarr._storage._backends import is_icechunk_repo
from annizarr._storage._uri import is_pathlike, is_remote, prepare_output_path, store_name, store_path
from annizarr._version import __version__
from annizarr.errors import RepoError, StorageError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import zarr

    from annizarr.ic import Repo
    from annizarr.typing import StoreLike

logger = logging.getLogger(__name__)


class _SwapState(enum.Enum):
    OPEN = "open"  # tmp store being written; target (if any) untouched
    SWAPPING = "swapping"  # mid-swap; target may momentarily be missing
    DONE = "done"  # swap complete; tmp no longer exists at its tmp path
    ABORTED = "aborted"  # writer gave up; tmp cleaned up (or never existed)


def op_metadata(op: str, **extra: Any) -> dict[str, Any]:
    return {"annizarr": __version__, "op": op, **{k: v for k, v in extra.items() if v is not None}}


def _supplied_repo(store: StoreLike) -> Repo | None:
    return None if is_pathlike(store) else store


def _as_repo(store: StoreLike, *, branch: str | None = None, icechunk: bool = False) -> Repo | None:
    # a path is checked first so plain zarr never imports annizarr.ic
    if is_pathlike(store):
        if not (icechunk or is_remote(store) or is_icechunk_repo(store)):
            return None
        from annizarr.ic import Repo

        return Repo(str(store), branch=branch)
    return store


def _checkout(repo: Repo, branch: str | None) -> None:
    if branch is not None and branch != repo.branch:
        repo.checkout(branch, create=True)


def check_output_target(output_path: StoreLike, cfg: AppConfig) -> None:
    # called before any expensive load; open_output_store repeats this right before writing.
    supplied = _supplied_repo(output_path)
    if supplied is not None or cfg.io.backend == "icechunk":
        from annizarr.ic import Repo

        if (supplied is not None or Repo.exists(str(output_path))) and not cfg.io.overwrite:
            raise StorageError(
                f"Icechunk repo already exists at '{store_path(output_path)}'. Pass --overwrite "
                "(or IOConfig(overwrite=True)) to replace its contents."
            )
        return
    path = store_path(output_path)
    if is_remote(path):
        return  # rejected later by open_output_store (remote requires the icechunk backend)
    target = Path(path)
    if target.exists() and not cfg.io.overwrite:
        raise StorageError(
            f"Output path already exists: {target}. Pass --overwrite (or IOConfig(overwrite=True)) to replace it."
        )


@dataclass
class OutputStore:
    root: zarr.Group
    finalize: Callable[[], str | None]
    abort: Callable[[], None]


def open_output_store(
    output_path: StoreLike,
    cfg: AppConfig,
    *,
    commit_message: str | None = None,
    commit_metadata: dict[str, Any] | None = None,
    branch: str | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> OutputStore:
    import zarr

    check_output_target(output_path, cfg)

    repo = _supplied_repo(output_path)
    if repo is not None or cfg.io.backend == "icechunk":
        from annizarr.ic import Repo

        if repo is None:
            path = str(output_path)
            exists = Repo.exists(path)
            if not exists and not is_remote(path):
                prepare_output_path(Path(path), cfg.io.overwrite)
            repo = Repo(path) if exists else Repo.create(path)
        elif repo.dirty:
            raise RepoError(f"Uncommitted changes on '{repo.branch}': commit() or discard() before writing onto it")
        _checkout(repo, branch)
        root: zarr.Group = repo.open_zarr("w", truncate=True)  # truncate: a re-run must not append to old data
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if committed is None:
                _verify_new_store(root, expected_shape=expected_shape)
                msg = commit_message or f"annizarr write → {store_name(output_path)}"
                committed = repo.commit(msg, metadata=commit_metadata)
                logger.info(f"icechunk commit {committed} on branch '{repo.branch}'")
            return committed

        def abort_icechunk() -> None:
            repo.discard()  # the session was clean when opened, so this drops only what the op staged

        return OutputStore(root, finalize_icechunk, abort_icechunk)

    output_path = str(output_path)
    if is_remote(output_path):
        raise StorageError(
            f"Remote output '{output_path}' requires the icechunk backend — pass --ic "
            "(or IOConfig(backend='icechunk'))."
        )

    # atomic swap: same-filesystem temp dir (plain rename); old store moved aside, not
    # deleted, until the new one is in place, so a crash mid-swap can always restore it.
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.tmp-{uuid4().hex[:8]}"
    root = zarr.open_group(str(tmp_path), mode="w")
    state = _SwapState.OPEN

    def finalize() -> str | None:
        nonlocal state
        if state is _SwapState.DONE:
            return None
        if state is not _SwapState.OPEN:
            return None  # a previous finalize() already failed and cleaned up; nothing to redo
        verify_root = zarr.open_group(str(tmp_path), mode="r")
        _verify_new_store(verify_root, expected_shape=expected_shape)
        if cfg.io.consolidate_metadata:
            zarr.consolidate_metadata(str(tmp_path))

        aside: Path | None = None
        if target.exists():
            if not cfg.io.overwrite:
                raise StorageError(f"Output path already exists: {target}.")
            aside = target.parent / f".{target.name}.old-{uuid4().hex[:8]}"

        state = _SwapState.SWAPPING
        try:
            if aside is not None:
                os.replace(target, aside)
            os.replace(tmp_path, target)
        except BaseException:
            # restore `aside` so the target is never left empty or gone.
            if aside is not None and not target.exists() and aside.exists():
                os.replace(aside, target)
            shutil.rmtree(tmp_path, ignore_errors=True)
            state = _SwapState.ABORTED
            raise

        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)
        state = _SwapState.DONE
        return None

    def abort() -> None:
        nonlocal state
        if state is _SwapState.DONE or state is _SwapState.ABORTED:
            return  # safe to call after finalize(), or after finalize() already cleaned up
        if state is _SwapState.OPEN:
            shutil.rmtree(tmp_path, ignore_errors=True)
        state = _SwapState.ABORTED

    return OutputStore(root, finalize, abort)


def _verify_new_store(root: zarr.Group, *, expected_shape: tuple[int, int] | None = None) -> None:
    # never ad.read_zarr here, which would materialise X
    import zarr

    from annizarr._core._zarr import shape_attr

    if root.attrs.get("encoding-type") != "anndata":
        raise StorageError("Refusing to finalize: missing anndata root encoding-type.")
    if "X" not in root:
        raise StorageError("Refusing to finalize: no X in the written store.")
    if expected_shape is None:
        return
    x = root["X"]
    actual = x.shape if isinstance(x, zarr.Array) else shape_attr(x)
    actual_shape = (int(actual[0]), int(actual[1]))
    if actual_shape != expected_shape:
        raise StorageError(f"X shape mismatch: expected {expected_shape}, got {actual_shape}.")


def verify_store_shape(
    root: zarr.Group, *, n_obs: int, n_vars: int, nnz: int | None = None, layers: Sequence[str] = ()
) -> None:
    # metadata plus one indptr read per CSR matrix; proves the store is shaped right, not which bytes it holds
    import numpy as np

    from annizarr._core._zarr import element_length, get_array, get_group, shape_attr, str_attr, str_list_attr

    indptr = np.asarray(get_array(get_group(root, "X"), "indptr")[:], dtype=np.int64)
    if len(indptr) != n_obs + 1 or (np.diff(indptr) < 0).any():
        raise StorageError(f"X/indptr is not {n_obs + 1} non-decreasing offsets.")
    if nnz is not None and int(indptr[-1]) != nnz:
        raise StorageError(f"X holds {int(indptr[-1])} nonzeros, expected {nnz}.")
    for path in ("X", *(f"layers/{k}" for k in layers)):
        matrix = get_group(root, path)
        if shape_attr(matrix) != (n_obs, n_vars):
            raise StorageError(f"{path} shape {shape_attr(matrix)} != {(n_obs, n_vars)}.")
        if path != "X" and not np.array_equal(get_array(matrix, "indptr")[:], indptr):
            raise StorageError(f"{path}/indptr differs from X's.")
        for name in ("data", "indices"):
            length = get_array(matrix, name).shape[0]
            if length != indptr[-1]:
                raise StorageError(f"{path}/{name} has {length} entries, indptr ends at {int(indptr[-1])}.")
    obs = get_group(root, "obs")
    for name in (str_attr(obs, "_index"), *str_list_attr(obs, "column-order")):
        length = element_length(obs[name])
        if length != n_obs:
            raise StorageError(f"obs/{name} has {length} rows, expected {n_obs}.")


def open_store_rw(
    store: StoreLike,
    cfg: AppConfig,
    *,
    commit_message: str | None = None,
    commit_metadata: dict[str, Any] | None = None,
    branch: str | None = None,
    commit: bool = True,
) -> tuple[zarr.Group, Callable[[], str | None]]:
    import zarr

    repo = _as_repo(store, icechunk=cfg.io.backend == "icechunk")
    if not commit:
        if repo is None:
            raise StorageError(f"commit=False only applies to Icechunk stores; '{store_path(store)}' is plain zarr.")
        if _supplied_repo(store) is None:
            raise StorageError(
                f"commit=False needs a Repo to hold the staged changes: pass az.Repo('{store_path(store)}') "
                "and call repo.commit() after the ops."
            )
    if repo is not None:
        _checkout(repo, branch)
        root: zarr.Group = repo.open_zarr("w")
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if commit and committed is None:
                msg = commit_message or f"annizarr update → {store_name(store)}"
                committed = repo.commit(msg, metadata=commit_metadata)
                logger.info(f"icechunk commit {committed} on branch '{repo.branch}'")
            return committed

        return root, finalize_icechunk

    path = Path(store_path(store))
    if not path.exists():
        raise StorageError(f"Store does not exist: {path}")
    # use_consolidated=False: anndata's write_elem refuses to edit a consolidated group.
    root = zarr.open_group(str(path), mode="r+", use_consolidated=False)

    meta_file = path / "zarr.json"
    had_consolidated = meta_file.exists() and json.loads(meta_file.read_text()).get("consolidated_metadata") is not None
    finalized = False

    def finalize() -> str | None:
        nonlocal finalized
        if not finalized:
            if had_consolidated or cfg.io.consolidate_metadata:
                zarr.consolidate_metadata(str(path))
            finalized = True
        return None

    return root, finalize


def open_input_group(store: StoreLike, *, branch: str | None = None, snapshot_id: str | None = None) -> zarr.Group:
    import zarr

    repo = _as_repo(store, branch=branch)
    if repo is None:
        return zarr.open_group(str(store), mode="r")
    if repo.dirty and snapshot_id is None:  # reads what the caller has staged; "w" on a clean repo would open a session
        return repo.open_zarr("w")
    return repo.open_zarr("r", snapshot_id=snapshot_id)
