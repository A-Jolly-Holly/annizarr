from __future__ import annotations

import os
from typing import Literal

__all__ = ["Backend", "Layout", "PathLike"]

type Layout = Literal["csr", "csc", "dense"]
type Backend = Literal["zarr", "icechunk"]
type PathLike = str | os.PathLike[str]
