from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Literal

import pytest

from annizarr._core._runtime import configure_runtime, map_parallel, pipeline_parallel, run_parallel
from annizarr._core._system import usable_cpus
from helpers import double, exit_abruptly, identity, raise_on, sleep_reverse, worker_zarr_threads


def test_configure_runtime_sizes_the_codec_pool_and_workers_cap_theirs() -> None:
    import zarr

    configure_runtime(2)
    configure_runtime(4)  # a second op must not keep the first call's sizing
    assert zarr.config.get("threading.max_workers") == max(4, usable_cpus())
    assert map_parallel(worker_zarr_threads, [(0,), (1,)], 2, mode="processes") == [2, 2]


def test_pipeline_parallel_consumes_every_job_once() -> None:
    results: list[int] = []
    lock = threading.Lock()

    def consume(payload: int) -> None:
        with lock:
            results.append(payload)

    pipeline_parallel(double, consume, [(i,) for i in range(20)], cpus=4)
    assert sorted(results) == [i * 2 for i in range(20)]


@pytest.mark.parametrize("mode", ["threads", "processes"])
def test_pipeline_parallel_ordered_consumes_in_job_order(mode: Literal["threads", "processes"]) -> None:
    seen: list[int] = []
    pipeline_parallel(sleep_reverse, seen.append, [(i,) for i in range(20)], cpus=4, mode=mode, ordered=True)
    assert seen == list(range(20))


@pytest.mark.parametrize("ordered", [False, True])
def test_pipeline_parallel_propagates_exceptions_from_either_stage(ordered: bool) -> None:
    def consume(payload: int) -> None:
        if payload == 3:
            raise RuntimeError("boom-consume")

    with pytest.raises(RuntimeError, match="boom-consume"):
        pipeline_parallel(identity, consume, [(i,) for i in range(10)], cpus=2, ordered=ordered)
    with pytest.raises(ValueError, match="boom-produce-5"):
        pipeline_parallel(raise_on, lambda _: None, [(5, i) for i in range(10)], cpus=2, ordered=ordered)


def test_run_parallel_fails_fast() -> None:
    def worker(i: int) -> None:
        time.sleep(0.05)
        if i == 3:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_parallel(worker, [(i,) for i in range(50)], cpus=4, mode="threads")


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"sys/fs/cgroup/cpu.max": "100000 100000"}, 1),
        ({"sys/fs/cgroup/cpu.max": "250000 100000"}, 3),  # a fractional quota rounds up
        ({"sys/fs/cgroup/cpu.max": "max 100000"}, None),
        ({"proc/self/cgroup": "0::/job", "sys/fs/cgroup/job/cpu.max": "200000 100000"}, 2),  # own cgroup first
        (  # a parent slice's limit binds the child below it
            {
                "proc/self/cgroup": "0::/slice/job",
                "sys/fs/cgroup/slice/cpu.max": "100000 100000",
                "sys/fs/cgroup/slice/job/cpu.max": "max 100000",
            },
            1,
        ),
        ({"sys/fs/cgroup/cpu/cpu.cfs_quota_us": "150000", "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}, 2),
        ({"sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1", "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}, None),
        ({}, None),
    ],
)
def test_usable_cpus_caps_the_affinity_count_by_a_cgroup_quota(
    tmp_path: Path, files: dict[str, str], expected: int | None
) -> None:
    from annizarr._core._system import usable_cpus

    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    unconstrained = usable_cpus(str(tmp_path / "nowhere"))
    assert usable_cpus(str(tmp_path)) == (unconstrained if expected is None else min(expected, unconstrained))


def test_a_killed_worker_names_cpus_as_the_fix() -> None:
    from annizarr.errors import ConversionError

    for run in (
        lambda: map_parallel(exit_abruptly, [(i,) for i in range(4)], 2, mode="processes"),
        lambda: run_parallel(exit_abruptly, [(i,) for i in range(4)], 2, mode="processes"),
        lambda: pipeline_parallel(exit_abruptly, lambda _: None, [(i,) for i in range(4)], 2, mode="processes"),
    ):
        with pytest.raises(ConversionError, match="lower --cpus"):
            run()
