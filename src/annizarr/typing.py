from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from annizarr.ic import Repo

__all__ = ["Backend", "Layout", "PathLike", "StoreLike"]

type Layout = Literal["csr", "csc", "dense"]
type Backend = Literal["zarr", "icechunk"]
type PathLike = str | os.PathLike[str]
type StoreLike = PathLike | Repo  # PEP 695 aliases evaluate lazily, so annizarr.ic stays unimported
