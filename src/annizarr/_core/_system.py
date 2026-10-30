from __future__ import annotations

import math
import os
from pathlib import Path


def usable_cpus(root: str = "/") -> int:
    """Cores this process may run on: the affinity mask capped by a cgroup CPU quota.

    ``os.cpu_count()`` is the host's count; it ignores ``taskset``/cpusets and container CPU
    limits (``docker --cpus``, Kubernetes), and ``os.process_cpu_count()`` ignores the quota.
    A fractional quota rounds up, as dask and loky do. ``root`` points the cgroup reads at a
    fake tree in tests.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # 3.13+; honours PYTHON_CPU_COUNT
    if process_cpu_count is not None:
        count = process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0)) or 1
    else:
        count = os.cpu_count() or 1
    quota = _cgroup_cpu_quota(Path(root))
    return min(count, quota) if quota is not None else count


def _cgroup_cpu_quota(root: Path) -> int | None:
    # v2: the tightest cpu.max from this process's cgroup up to the mount root (a parent slice's limit
    # binds its children); v1: the cfs files. None = unlimited
    fs = root / "sys" / "fs" / "cgroup"
    limits = [q for path in _cgroup_dirs(root, fs) if (q := _read_v2(path / "cpu.max")) is not None]
    if limits:
        return min(limits)
    return _read_v1(fs / "cpu" / "cpu.cfs_quota_us", fs / "cpu" / "cpu.cfs_period_us")


def _cgroup_dirs(root: Path, fs: Path) -> list[Path]:
    # this process's v2 cgroup and each of its ancestors, down to the mount root
    try:
        lines = (root / "proc" / "self" / "cgroup").read_text().splitlines()
    except OSError:
        lines = []
    own = next((line[3:].strip("/") for line in lines if line.startswith("0::")), "")
    parts = [part for part in own.split("/") if part]
    return [fs.joinpath(*parts[:n]) for n in range(len(parts), -1, -1)]


def _read_v2(path: Path) -> int | None:
    try:
        quota, period = path.read_text().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return _ceil_ratio(quota, period)


def _read_v1(quota_path: Path, period_path: Path) -> int | None:
    try:
        quota, period = quota_path.read_text().strip(), period_path.read_text().strip()
    except OSError:
        return None
    return _ceil_ratio(quota, period)  # v1 writes -1 for no quota


def _ceil_ratio(quota: str, period: str) -> int | None:
    try:
        q, p = int(quota), int(period)
    except ValueError:
        return None
    return max(1, math.ceil(q / p)) if q > 0 and p > 0 else None
