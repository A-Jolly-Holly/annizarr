from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from annizarr.errors import StorageError
from annizarr.typing import PathLike, StoreLike

if TYPE_CHECKING:
    from typing_extensions import TypeIs  # typing.TypeIs is 3.13+, the floor is 3.12

_REMOTE_SCHEMES = frozenset({"s3", "gs", "gcs"})


def scheme(path: PathLike) -> str:
    return urlparse(str(path)).scheme


def is_remote(path: PathLike) -> bool:
    return scheme(path) in _REMOTE_SCHEMES


def bucket_prefix(path: PathLike) -> tuple[str, str | None]:
    # (bucket, prefix); prefix is None at the bucket root
    parsed = urlparse(str(path))
    if not parsed.netloc:
        raise StorageError(f"Missing bucket in '{path}'")
    return parsed.netloc, parsed.path.lstrip("/") or None


def canonical_location(path: PathLike) -> str:
    return str(path) if is_remote(path) else os.path.realpath(str(path))


def is_readonly_path(path: PathLike) -> bool:
    # a path that doesn't exist yet is judged by its nearest existing ancestor
    if is_remote(path):
        return False
    probe = os.path.abspath(str(path))
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return not os.access(probe, os.W_OK)


def is_pathlike(store: StoreLike) -> TypeIs[PathLike]:
    return isinstance(store, str | os.PathLike)


def store_path(store: StoreLike) -> str:
    # where a store lives: the path or URI as given, or a Repo's own path
    return str(store) if is_pathlike(store) else store.path


def store_name(store: StoreLike) -> str:
    # tail component of a local path or URI, for commit messages
    path = store_path(store)
    return path.rstrip("/").rsplit("/", 1)[-1] or path


def prepare_output_path(output_path: PathLike, overwrite: bool) -> None:
    # removes an existing LOCAL output path when overwrite is enabled, else raises
    path = Path(output_path)
    if path.exists():
        if not overwrite:
            raise StorageError(
                f"Output path already exists: {path}. Pass --overwrite (or IOConfig(overwrite=True)) to replace it."
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
