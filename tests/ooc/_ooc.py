"""Scaffolding shared by the large-store memory test and tests/ooc/bench.py:
a blockwise CSR h5ad builder that never holds the whole matrix, and a CLI runner that reports
wall time and peak RSS for the CLI process and for its whole process tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

CATEGORIES = ("A", "B", "C", "D")
MB = 1024**2


def build_h5ad(
    path: Path,
    *,
    n_obs: int,
    n_vars: int,
    density: float,
    seed: int,
    name_prefix: str = "cell",
    block_rows: int = 25_000,
) -> None:
    """Write a CSR h5ad with a four-level ``cell_type`` obs column, ``block_rows`` rows at a time.

    Entries are random (row, col) pairs at about ``density``, deduplicated; values are float32 in
    (0.1, 10.1). Deterministic for a given seed. Peak RAM is one block's entries, not the matrix.
    """
    from anndata.io import write_elem

    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = [n_obs, n_vars]
        data = x.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1 << 20,))
        indices = x.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,))
        indptr = np.zeros(n_obs + 1, dtype=np.int64)
        nnz = 0
        for r0 in range(0, n_obs, block_rows):
            r1 = min(r0 + block_rows, n_obs)
            rows, cols, vals = _block(rng, r1 - r0, n_vars, density)
            indptr[r0 + 1 : r1 + 1] = nnz + np.cumsum(np.bincount(rows, minlength=r1 - r0))
            m = len(vals)
            data.resize((nnz + m,))
            indices.resize((nnz + m,))
            data[nnz:] = vals
            indices[nnz:] = cols
            nnz += m
        x.create_dataset("indptr", data=indptr)
        obs = pd.DataFrame(
            {"cell_type": pd.Categorical(rng.choice(CATEGORIES, n_obs), categories=CATEGORIES)},
            index=[f"{name_prefix}_{i}" for i in range(n_obs)],
        )
        write_elem(f, "obs", obs)
        write_elem(f, "var", pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]))
        for key in ("uns", "obsm", "varm", "obsp", "varp", "layers"):
            write_elem(f, key, {})


def _block(
    rng: np.random.Generator, n_rows: int, n_vars: int, density: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # a Bernoulli mask per sub-block gives sorted, unique (row, col) pairs without a lexsort
    rows_parts, cols_parts = [], []
    step = max(1, (64 << 20) // (n_vars * 8))  # 64MB of random draws at a time
    for r0 in range(0, n_rows, step):
        r1 = min(r0 + step, n_rows)
        hit = np.nonzero(rng.random((r1 - r0, n_vars)) < density)
        rows_parts.append(hit[0] + r0)
        cols_parts.append(hit[1])
    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    vals = rng.random(len(rows)).astype(np.float32) * 10 + 0.1
    return rows, cols.astype(np.int32), vals


_RUNNER = """
import json, resource, sys, time
from annizarr._cli import main
argv = json.loads(sys.argv[1])
t0 = time.perf_counter()
rc = main(argv)
wall = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
self_peak = peak if sys.platform == "darwin" else peak * 1024
# Linux copies the pre-exec parent's high-water mark into ru_maxrss; VmHWM is this process only
try:
    with open("/proc/self/status") as f:
        self_peak = next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmHWM:"))
except (OSError, StopIteration):
    pass
print(json.dumps({"rc": rc, "wall_s": wall, "self_peak_bytes": self_peak}))
"""


@dataclass(frozen=True, slots=True)
class Measurement:
    rc: int
    wall_s: float
    self_peak_bytes: int  # the CLI process alone, exact (VmHWM on Linux, ru_maxrss elsewhere)
    tree_peak_bytes: int  # CLI process plus every descendant, sampled; never below self_peak_bytes
    tree_peak_procs: int  # processes alive at the peak sample
    stderr: str


def run_measured(argv: list[str], *, interval_s: float | None = None) -> Measurement:
    """Run ``annizarr argv`` in a fresh subprocess and measure it.

    Band workers are separate processes, so the CLI process's own peak misses them; the process
    tree is sampled every ``interval_s`` (20ms on Linux via /proc, 50ms elsewhere via ps) and the
    largest summed RSS is reported alongside the exact self peak.
    """
    if interval_s is None:
        interval_s = 0.02 if sys.platform.startswith("linux") else 0.05
    with tempfile.TemporaryFile(mode="w+") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, json.dumps(argv)], stdout=subprocess.PIPE, stderr=err, text=True
        )
        peak, peak_n = 0, 0
        while proc.poll() is None:
            total, n = _tree_rss(proc.pid)
            if total > peak:
                peak, peak_n = total, n
            time.sleep(interval_s)
        out = proc.stdout.read() if proc.stdout is not None else ""
        err.seek(0)
        stderr = err.read()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return Measurement(proc.returncode or 1, float("nan"), 0, peak, peak_n, stderr)
    r: dict[str, Any] = json.loads(lines[-1])
    self_peak = int(r["self_peak_bytes"])
    return Measurement(int(r["rc"]), float(r["wall_s"]), self_peak, max(peak, self_peak), max(peak_n, 1), stderr)


def _tree_rss(root: int) -> tuple[int, int]:
    # summed RSS (bytes) and process count of root plus all descendants, from one snapshot
    parent: dict[int, int] = {}
    rss: dict[int, int] = {}
    if sys.platform.startswith("linux"):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                with open(f"/proc/{pid}/status") as f:
                    ppid, kb = -1, 0
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                        elif line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            break
            except OSError:
                continue
            parent[pid], rss[pid] = ppid, kb * 1024
    else:
        ps = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, check=False)
        for line in ps.stdout.splitlines():
            parts = line.split()
            if len(parts) == 3:
                pid, ppid, kb = (int(p) for p in parts)
                parent[pid], rss[pid] = ppid, kb * 1024
    total, count = 0, 0
    frontier = [root]
    while frontier:
        pid = frontier.pop()
        if pid not in rss:
            continue
        total += rss[pid]
        count += 1
        frontier.extend(child for child, pp in parent.items() if pp == pid)
    return total, count


def format_table(rows: list[tuple[str, Measurement, int | None]]) -> str:
    """Rows of (label, measurement, ceiling bytes or None) as an aligned text table."""
    out = [f"{'step':<24} {'wall_s':>7} {'self_MB':>8} {'tree_MB':>8} {'procs':>5} {'ceiling_MB':>10}"]
    for label, m, ceiling in rows:
        c = f"{ceiling / MB:.0f}" if ceiling is not None else "-"
        out.append(
            f"{label:<24} {m.wall_s:>7.2f} {m.self_peak_bytes / MB:>8.0f} {m.tree_peak_bytes / MB:>8.0f} "
            f"{m.tree_peak_procs:>5} {c:>10}"
        )
    return "\n".join(out)


def worker_import_baseline(_: int) -> int:
    """Peak RSS in bytes of a fresh process after the imports a band worker makes.

    Lives here, not in the test module: a pool worker imports the function's module by name, and
    pytest's importlib mode registers test modules under names a worker cannot import.
    """
    import importlib
    import resource

    for module in ("numpy", "scipy.sparse", "pandas", "h5py", "zarr", "anndata"):
        importlib.import_module(module)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024
