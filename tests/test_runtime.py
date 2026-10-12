from __future__ import annotations

import logging
import os
import threading
import time
from typing import Literal

import pytest

from annizarr._core._runtime import configure_runtime, map_parallel, pipeline_parallel, run_parallel
from helpers import double, identity, raise_on, sleep_reverse, worker_zarr_threads


def test_configure_runtime_sizes_the_codec_pool_and_workers_cap_theirs() -> None:
    import zarr

    configure_runtime(2)
    configure_runtime(4)  # a second op must not keep the first call's sizing
    assert zarr.config.get("threading.max_workers") == max(4, os.cpu_count() or 1)
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


def test_check_worker_memory_warns_when_workers_outgrow_available_ram(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from annizarr._core import _layout, _runtime

    monkeypatch.setattr(_runtime, "available_memory", lambda: 4 * 1024**3)
    caplog.set_level(logging.WARNING, logger="annizarr")
    _runtime.check_worker_memory(2)  # 2 x (baseline + 512MiB) is well under 80% of 4GiB
    assert not caplog.records
    _runtime.check_worker_memory(16)
    assert any("pass a lower --cpus" in r.message for r in caplog.records)
    monkeypatch.setattr(_layout, "BATCH_BYTES", 16 * 1024**2)
    caplog.clear()
    _runtime.check_worker_memory(16)  # a smaller band budget shrinks the estimate
    assert not caplog.records


def test_available_memory_falls_back_to_three_quarters_of_ram_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    from annizarr._core import _runtime

    monkeypatch.setattr(_runtime.sys, "platform", "darwin")
    assert _runtime.available_memory() == os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 3 // 4
