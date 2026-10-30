from __future__ import annotations

import functools
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from annizarr._core._system import usable_cpus
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from annizarr._sources._readers import Reader

logger = logging.getLogger(__name__)

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_blas() -> None:
    # must run before numpy/scipy/blosc import: OpenBLAS reads these env vars once, at import.
    for var in _BLAS_VARS:
        os.environ.setdefault(var, "1")


def configure_runtime(cpus: int) -> None:
    import zarr

    pin_blas()
    # both zarr knobs: async.concurrency only dispatches fetches, max_workers runs decompression.
    workers = max(cpus, usable_cpus())
    zarr.config.set({"async.concurrency": 64, "threading.max_workers": workers})


def _init_worker() -> None:
    # per-process zarr pools multiply across the process pool, so keep each worker's small
    import zarr

    pin_blas()
    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 2})


def _executor(mode: Literal["threads", "processes"], cpus: int) -> Any:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    if mode == "processes":
        return ProcessPoolExecutor(max_workers=cpus, initializer=_init_worker)
    return ThreadPoolExecutor(max_workers=cpus)


@contextmanager
def _explain_killed_worker() -> Iterator[None]:
    # a pool worker that dies without raising (the OOM killer, most often) otherwise surfaces as a bare
    # "terminated abruptly"; one worker's peak is fixed by the band budget, so --cpus is the lever
    from concurrent.futures.process import BrokenProcessPool

    from annizarr._core import _layout

    try:
        yield
    except BrokenProcessPool as exc:
        worker_mb = (_layout.WORKER_BASELINE_BYTES + _layout.BATCH_BYTES) // 1024**2
        raise ConversionError(
            f"a worker process died abruptly ({exc}). The usual cause is the out-of-memory killer: each worker "
            f"holds up to about {worker_mb}MB, so try a lower --cpus."
        ) from exc


def run_parallel(
    worker: Callable[..., None],
    jobs: Sequence[tuple[object, ...]],
    cpus: int,
    *,
    mode: Literal["threads", "processes"] = "threads",
) -> None:
    # mode="processes" for h5py-backed readers (h5py is not thread-safe); "threads" otherwise.
    if cpus <= 1 or len(jobs) <= 1:
        for job in jobs:
            worker(*job)
        return
    with _explain_killed_worker(), _executor(mode, cpus) as ex:
        futures = [ex.submit(worker, *job) for job in jobs]
        try:
            for fut in futures:
                fut.result()
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise


def pipeline_parallel(
    produce: Callable[..., Any],
    consume: Callable[[Any], None],
    jobs: Sequence[tuple[object, ...]],
    cpus: int,
    *,
    mode: Literal["threads", "processes"] = "processes",
    ordered: bool = False,
) -> None:
    # workers read ahead, this process writes; the semaphore caps payloads read but not yet
    # written so memory stays bounded whichever stage is slower: 2*cpus, or fewer once that many
    # payloads would exceed the band budget (this process is then the bottleneck anyway).
    # ordered=True consumes in job order on one thread (the reorder buffer is bounded by the same
    # permits); processes for h5py readers, threads for thread-safe ones (no pickling, and
    # Icechunk sessions work)
    from annizarr._core import _layout

    if not jobs:
        return

    permits = max(2, min(2 * cpus, _layout.BATCH_BYTES // _layout.PIPELINE_BATCH_BYTES))
    inflight = threading.Semaphore(permits)
    lock = threading.Lock()
    ready_cv = threading.Condition(lock)
    ready: dict[int, Any] = {}
    errors: list[BaseException] = []

    def fail(exc: BaseException) -> None:
        with lock:
            errors.append(exc)
            ready_cv.notify_all()

    def release_ready() -> None:
        # with the lock held: give back the permits of payloads nobody will consume now
        for _ in ready:
            inflight.release()
        ready.clear()

    def do_write(payload: Any) -> None:
        try:
            consume(payload)
        except BaseException as exc:
            fail(exc)
        finally:
            inflight.release()

    def write_in_order() -> None:
        for idx in range(len(jobs)):
            with ready_cv:
                while idx not in ready and not errors:
                    ready_cv.wait()
                if errors:
                    release_ready()
                    return
                payload = ready.pop(idx)
            try:
                consume(payload)
            except BaseException as exc:
                with lock:
                    errors.append(exc)
                    release_ready()
                inflight.release()
                return
            inflight.release()

    def on_read(idx: int, fut: Any) -> None:
        try:
            payload = fut.result()
        except BaseException as exc:
            fail(exc)
            inflight.release()
            return
        if not ordered:
            writers.submit(do_write, payload)
            return
        with ready_cv:
            if errors:
                inflight.release()
                return
            ready[idx] = payload
            ready_cv.notify_all()

    with (
        _explain_killed_worker(),
        _executor(mode, cpus) as readers,
        _executor("threads", 1 if ordered else cpus) as writers,
    ):
        if ordered:
            writers.submit(write_in_order)
        for idx, job in enumerate(jobs):
            inflight.acquire()
            if errors:
                inflight.release()
                break
            readers.submit(produce, *job).add_done_callback(functools.partial(on_read, idx))

        for _ in range(permits):  # drain: block until every checked-out permit is back
            inflight.acquire()

    if errors:
        with _explain_killed_worker():  # a read failure is collected, so it is raised past the pool's block
            raise errors[0]


def map_parallel(
    worker: Callable[..., Any],
    jobs: Sequence[tuple[object, ...]],
    cpus: int,
    *,
    mode: Literal["threads", "processes"] = "threads",
) -> list[Any]:
    # like run_parallel, but returns one result per job, in job order (for a result-collecting
    # parallel step, e.g. a dense reader's row-banded nnz-counting pass).
    if cpus <= 1 or len(jobs) <= 1:
        return [worker(*job) for job in jobs]
    with _explain_killed_worker(), _executor(mode, cpus) as ex:
        futures = [ex.submit(worker, *job) for job in jobs]
        try:
            return [fut.result() for fut in futures]
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise


@dataclass(frozen=True, slots=True)
class ArrayTarget:
    # an open array (threads) or a LocalStore root + array path a worker process reopens
    array: Any | None
    root: str | None
    path: str | None

    @classmethod
    def attached(cls, array: Any) -> ArrayTarget:
        return cls(array=array, root=None, path=None)

    @classmethod
    def detached(cls, array: Any) -> ArrayTarget:
        from zarr.storage import LocalStore

        store = array.store_path.store
        assert isinstance(store, LocalStore), "process-mode writes need a local zarr store"
        return cls(array=None, root=str(store.root), path=array.store_path.path)

    @property
    def in_worker(self) -> bool:
        return self.array is None

    def resolve(self) -> Any:
        if self.array is not None:
            return self.array
        assert self.root is not None and self.path is not None, "a process-mode ArrayTarget needs root+path"
        import zarr
        from zarr.storage import LocalStore

        from annizarr._core._zarr import get_array

        # consolidated metadata, when present, predates arrays this op just created
        group = zarr.open_group(store=LocalStore(self.root), mode="r+", use_consolidated=False)
        return get_array(group, self.path)


def writer_parallel_mode(reader: Reader, store: Any) -> Literal["threads", "processes", "pipeline"]:
    # h5py readers need processes; a non-local store (an Icechunk session) is in-process only,
    # so there the processes only read and this process's threads write
    if reader.thread_safe:
        return "threads"
    from zarr.storage import LocalStore

    if isinstance(store, LocalStore):
        return "processes"
    logger.info(f"lazy input into a non-local store: {reader.cpus} reader processes feed the writer threads")
    return "pipeline"


def progress(total: int, label: str) -> Callable[[], None]:
    lock = threading.Lock()
    step = max(1, total // 10)
    count = 0
    last_time = time.perf_counter()
    last_count = 0

    def tick() -> None:
        nonlocal count, last_time, last_count
        with lock:
            count += 1
            now = time.perf_counter()
            done = count >= total
            if done or count - last_count >= step or now - last_time >= 5.0:
                logger.info(f"{label}: {count}/{total}" + (" done" if done else ""))
                last_time = now
                last_count = count

    return tick


@contextmanager
def stage(label: str) -> Iterator[None]:
    logger.info(f"{label} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"  done ({time.perf_counter() - t0:.1f}s)")
