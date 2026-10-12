"""Memory-bounded proof: every conversion/edit path streams band by band on a synthetic
300 000 x 30 000 (~1% density, ~90M nnz) CSR store, at 1 and at 4 workers.

Each step runs the real ``annizarr`` CLI in a fresh subprocess and is measured by
``_ooc.run_measured``: the CLI process's own exact peak and the sampled peak of its whole process
tree, which is what band workers in separate processes cost. The ceiling for each step comes from
the working-set model in ``_layout`` (bands from the planners, bytes per nonzero per pass, one band
per worker and nothing but small tables in the parent), with the worker import baseline and the
parent's own baseline measured on the running machine and a 15% margin.
All steps run, the table prints model and measurement side by side, and the ceilings are
asserted together at the end so one run shows the whole picture.

Marked ``slow`` and excluded from the default run by ``addopts = "-m 'not slow'"``; it is a memory
ceiling, not a regression test. Run it explicitly: ``pytest -m slow --no-cov -s``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from anndata.io import read_elem, sparse_dataset

from _ooc import MB, Measurement, build_h5ad, format_table, run_measured, worker_import_baseline
from annizarr._core import _layout
from annizarr._core._runtime import _init_worker
from annizarr.config import ChunkConfig

pytestmark = pytest.mark.slow

N_OBS, N_VARS = 300_000, 30_000
DENSITY = 0.01
N_APPEND = 2_000
CPUS = (1, 4)
H5_CHUNK = 1 << 20  # build_h5ad's data/indices chunk, the lazy reader's grain
MARGIN = 1.15


def _baselines(tiny_h5ad: Path, out: Path) -> tuple[int, int]:
    """(worker import baseline, parent baseline) in bytes, measured here rather than assumed."""
    with ProcessPoolExecutor(1, initializer=_init_worker) as ex:
        worker = int(ex.submit(worker_import_baseline, 0).result())
    parent = run_measured(["-q", "convert", str(tiny_h5ad), "-o", str(out), "--cpus", "1"]).self_peak_bytes
    return worker, parent


def _model(cpus: int, nnz: int, worker: int, parent: int) -> dict[str, int]:
    """Peak process-tree bytes per step from the working-set model; see the module docstring.

    Every pass is a parallel map over bands; this process holds only small tables, so a step is
    the parent baseline plus the busy workers' baselines and band working sets. At one worker the
    thread-mode passes run in this process, so the worker baseline is not paid twice.
    """
    budget = _layout.BATCH_BYTES
    unit = min(ChunkConfig().nnz_chunk, nnz)
    copy_bpn = _layout.copy_bytes_per_item(4, 4)
    copy_bands = _layout.plan_bands(nnz, unit=unit, bytes_per_item=copy_bpn, cpus=cpus, grain=H5_CHUNK)
    copy_band = max(s1 - s0 for s0, s1 in copy_bands)
    busy = min(cpus, len(copy_bands))

    def workers(n: int, band_bytes: int, *, in_process: bool) -> int:
        # n workers each holding one band; a thread shares this process's baseline
        base = 0 if in_process else worker
        return n * (base + band_bytes)

    scatter_nnz = budget // _layout.SCATTER_BYTES_PER_NNZ
    place_nnz = budget // _layout.PLACE_BYTES_PER_NNZ
    # a place worker reads its band back (12 B/nnz), sorts it, permutes it and writes it (zarr's copy)
    place_band = place_nnz * (_layout.PLACE_BYTES_PER_NNZ + int(8 * _layout.WRITE_COPY_FRACTION))
    scatter_band = scatter_nnz * _layout.SCATTER_BYTES_PER_NNZ
    lognorm_scatter_band = scatter_nnz * (_layout.SCATTER_BYTES_PER_NNZ + 8)
    bucket_nnz = budget // _layout.BUCKET_BYTES_PER_NNZ
    # append: the seam chunk is read-modify-written (about 4 x its raw bytes) while the obs index is scanned
    seam = 4 * unit * 4
    index_scan = (N_OBS + N_APPEND) * 100
    # a lazy h5 source reads in worker processes; a zarr-backed source (add-expr, sort) in this process's
    # threads. A process's RSS does not fall between passes, so once this process has run a pass in its
    # own threads it carries that peak through the worker-process passes that follow.
    h5_scatter = parent + workers(cpus, scatter_band, in_process=False)
    zarr_scatter = parent + workers(cpus, lognorm_scatter_band, in_process=True)
    place_workers = workers(cpus, place_band, in_process=False)
    sort_threads = max(
        parent + workers(cpus, bucket_nnz * _layout.BUCKET_BYTES_PER_NNZ, in_process=True),
        parent + workers(cpus, copy_band * copy_bpn, in_process=True),
        zarr_scatter,
    )
    return {
        "convert_csr": parent + workers(busy, copy_band * copy_bpn, in_process=cpus == 1),
        "convert_csc": max(h5_scatter, parent + place_workers),
        "add_expr_csc": zarr_scatter + place_workers,
        "sort": sort_threads + place_workers,
        "append": parent + seam + index_scan,
    }


@pytest.fixture(scope="session")
def large_h5ad(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("large_store_src") / "in.h5ad"
    build_h5ad(path, n_obs=N_OBS, n_vars=N_VARS, density=DENSITY, seed=0, name_prefix="cell")
    return path


@pytest.fixture(scope="session")
def extra_h5ad(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("large_store_extra") / "extra.h5ad"
    build_h5ad(path, n_obs=N_APPEND, n_vars=N_VARS, density=DENSITY, seed=1, name_prefix="extra")
    return path


@pytest.fixture(scope="session")
def baselines(tmp_path_factory: pytest.TempPathFactory) -> tuple[int, int]:
    d = tmp_path_factory.mktemp("large_store_baseline")
    build_h5ad(d / "tiny.h5ad", n_obs=200, n_vars=N_VARS, density=DENSITY, seed=2)
    return _baselines(d / "tiny.h5ad", d / "tiny.zarr")


def _sample_rows(n: int, k: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def _rows_from_h5ad(path: Path, rows: np.ndarray) -> np.ndarray:
    with h5py.File(path) as f:
        return np.asarray(sparse_dataset(f["X"])[rows].toarray())


@pytest.mark.parametrize("cpus", CPUS)
def test_large_store_pipeline_is_memory_bounded(
    tmp_path_factory: pytest.TempPathFactory, large_h5ad: Path, extra_h5ad: Path, baselines: tuple[int, int], cpus: int
) -> None:
    out_dir = tmp_path_factory.mktemp(f"large_store_out_c{cpus}")
    csr_store = out_dir / "csr.zarr"
    csc_store = out_dir / "csc.zarr"
    sorted_store = out_dir / "sorted.zarr"
    extra_store = out_dir / "extra.zarr"
    results: list[tuple[str, Measurement, int | None]] = []
    n = str(cpus)
    with h5py.File(large_h5ad) as f:
        nnz = int(f["X/indptr"][-1])
    worker, parent = baselines
    model = _model(cpus, nnz, worker, parent)
    ceilings = {k: int(v * MARGIN) for k, v in model.items()}

    def _step(name: str, argv: list[str]) -> Measurement:
        m = run_measured(argv)
        assert m.rc == 0, f"annizarr {argv} failed (rc={m.rc}); stderr:\n{m.stderr}"
        results.append((name, m, ceilings[name]))
        return m

    rows = _sample_rows(N_OBS, 5, seed=42)
    expected = _rows_from_h5ad(large_h5ad, rows)

    # 1. convert -> csr (lazy is the default, so no flag needed)
    _step("convert_csr", ["-q", "convert", str(large_h5ad), "-o", str(csr_store), "--cpus", n])
    got = sparse_dataset(zarr.open_group(str(csr_store), mode="r")["X"])[rows].toarray()
    np.testing.assert_array_equal(got, expected)

    # 2. convert --layout csc (lazy CSR -> csc through the streamed transpose)
    _step("convert_csc", ["-q", "convert", str(large_h5ad), "-o", str(csc_store), "--layout", "csc", "--cpus", n])
    got_csc = sparse_dataset(zarr.open_group(str(csc_store), mode="r")["X"])[rows].toarray()
    np.testing.assert_array_equal(got_csc, expected)

    # 3. add-expr --layout csc on the csr store (streamed lognorm CSR -> csc, same engine)
    _step("add_expr_csc", ["-q", "add-expr", str(csr_store), "--layout", "csc", "--cpus", n])
    root = zarr.open_group(str(csr_store), mode="r")
    gexp = sparse_dataset(root["layers/gexp"])[rows].toarray()
    x_rows = expected.astype(np.float64)
    sums = x_rows.sum(axis=1)
    factor = np.divide(1e4, sums, out=np.zeros_like(sums), where=sums > 0)
    np.testing.assert_allclose(gexp, np.log1p(x_rows * factor[:, None]).astype(np.float32), rtol=1e-4)

    # 4. sort --by cell_type (the csr store carries a lone gexp layer, which sort re-derives)
    _step("sort", ["-q", "sort", str(csr_store), "-o", str(sorted_store), "--by", "cell_type", "--cpus", n])
    sorted_root = zarr.open_group(str(sorted_store), mode="r")
    codes = read_elem(sorted_root["obs"])["cell_type"].cat.codes.to_numpy()
    assert np.all(np.diff(codes) >= 0), "sort did not produce contiguous cell_type blocks"
    assert "gexp" in list(sorted_root["layers"])

    # 5. append a 2 000-cell converted store onto the csr store
    assert run_measured(["-q", "convert", str(extra_h5ad), "-o", str(extra_store), "--cpus", n]).rc == 0
    _step("append", ["-q", "append", str(csr_store), str(extra_store), "--drop-derived", "--cpus", n])
    appended_root = zarr.open_group(str(csr_store), mode="r")
    assert appended_root["X"].attrs["shape"][0] == N_OBS + N_APPEND
    appended_tail = sparse_dataset(appended_root["X"])[np.arange(N_OBS, N_OBS + 5)].toarray()
    np.testing.assert_array_equal(appended_tail, _rows_from_h5ad(extra_h5ad, np.arange(5)))

    print(f"\ncpus={cpus}  worker baseline={worker / MB:.0f}MB  parent baseline={parent / MB:.0f}MB  nnz={nnz}")
    print(format_table(results))
    print("model (MB): " + "  ".join(f"{k}={v / MB:.0f}" for k, v in model.items()))
    over = [
        f"{name}: {m.tree_peak_bytes / MB:.0f}MB >= {c / MB:.0f}MB"
        for name, m, c in results
        if c and m.tree_peak_bytes >= c
    ]
    assert not over, f"cpus={cpus}: peak tree RSS over ceiling: " + "; ".join(over)
