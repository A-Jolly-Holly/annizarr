from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class Buckets:
    # per-band temp files read back with np.fromfile (memmap pages would count as resident, linear
    # in nonzeros). Workers write at precomputed offsets through their own descriptors, so no
    # process in the middle ever holds a payload. Plain state (paths, dtypes, counts) pickles.
    def __init__(self, root: Path, n_bands: int, fields: Mapping[str, np.dtype[Any]]) -> None:
        self.root = root
        self.fields = {name: np.dtype(dtype) for name, dtype in fields.items()}
        self.counts = [0] * n_bands

    @classmethod
    def create(
        cls,
        n_bands: int,
        fields: Mapping[str, np.dtype[Any]],
        *,
        dir: str | None,
        prefix: str,
        sizes: Sequence[int] | None = None,
    ) -> Buckets:
        buckets = cls(Path(tempfile.mkdtemp(prefix=prefix, dir=dir)), n_bands, fields)
        if sizes is not None:
            buckets.counts = [int(n) for n in sizes]
        return buckets

    @property
    def n_bands(self) -> int:
        return len(self.counts)

    def path(self, band: int, field: str) -> Path:
        return self.root / f"{field}{band}"

    def write_at(self, band: int, offset: int, **arrays: NDArray[Any]) -> None:
        # the band's size was fixed up front; pieces land at their offsets in any order
        assert set(arrays) == set(self.fields), "every field, all the same length"
        for field, arr in arrays.items():
            dtype = self.fields[field]
            fd = os.open(self.path(band, field), os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                os.pwrite(fd, memoryview(np.ascontiguousarray(arr, dtype=dtype)), offset * dtype.itemsize)
            finally:
                os.close(fd)

    def append(self, band: int, **arrays: NDArray[Any]) -> None:
        n = {len(a) for a in arrays.values()}
        assert len(n) == 1 and set(arrays) == set(self.fields), "every field, all the same length"
        for field, arr in arrays.items():
            # one raw write per piece: 1.4x the throughput of ndarray.tofile on a buffered file
            fd = os.open(self.path(band, field), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, memoryview(np.ascontiguousarray(arr, dtype=self.fields[field])))
            finally:
                os.close(fd)
        self.counts[band] += n.pop()

    def read(self, band: int, field: str) -> NDArray[Any]:
        dtype = self.fields[field]
        if self.counts[band] == 0:
            return np.empty(0, dtype=dtype)
        return np.fromfile(self.path(band, field), dtype=dtype, count=self.counts[band])

    def remove(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
