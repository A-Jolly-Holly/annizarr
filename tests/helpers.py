"""Builders and assertions shared by the suite: small synthetic inputs, stores, and checks."""

from __future__ import annotations

import filecmp
from dataclasses import replace
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr

from annizarr.config import AppConfig, ChunkConfig, ConcatConfig, GroupingConfig, IOConfig
from annizarr.ops import convert_adata

# small chunks so every fixture spans several chunks; two workers so parallel paths run without
# spawning a pool per core
CHUNKS = ChunkConfig(row_chunk=16, col_chunk=4, nnz_chunk=64, cpus=2)


def chunks(**kw: Any) -> ChunkConfig:
    return replace(CHUNKS, **kw)


def cfg(
    *,
    chunks: ChunkConfig = CHUNKS,
    grouping: GroupingConfig | None = None,
    concat: ConcatConfig | None = None,
    **io: Any,
) -> AppConfig:
    return AppConfig(
        io=IOConfig(**io),
        chunks=chunks,
        grouping=grouping or GroupingConfig(),
        concat=concat or ConcatConfig(),
    )


def make_adata(n_obs: int = 100, n_vars: int = 50, *, seed: int = 0, density: float = 0.3) -> ad.AnnData:
    """A synthetic CSR AnnData with a categorical obs column and an obsm entry."""
    rng = np.random.default_rng(seed)
    x = sp.random(n_obs, n_vars, density=density, format="csr", dtype=np.float32, random_state=seed)
    x.data = np.abs(x.data) + 1.0
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["a", "b", "c"], n_obs), categories=["a", "b", "c"])},
        index=[f"cell{seed}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    return ad.AnnData(X=x, obs=obs, var=var, obsm={"X_pca": rng.random((n_obs, 2)).astype(np.float32)})


def make_h5ad(tmp_path: Path, name: str = "in.h5ad", *, adata: ad.AnnData | None = None, **kw: Any) -> Path:
    """Write a synthetic (or given) AnnData to ``tmp_path/name``; return the path."""
    path = tmp_path / name
    (adata if adata is not None else make_adata(**kw)).write_h5ad(path)
    return path


def make_store(
    tmp_path: Path,
    name: str = "store.zarr",
    *,
    adata: ad.AnnData | None = None,
    cfg: AppConfig | None = None,
    **kw: Any,
) -> Path:
    """Convert a synthetic (or given) AnnData to a zarr store; return the path."""
    out = tmp_path / name
    convert_adata(adata if adata is not None else make_adata(**kw), output=out, cfg=cfg or AppConfig(chunks=CHUNKS))
    return out


def make_labelled_h5ad(path: Path) -> None:
    """A 6-cell CSR h5ad whose X[:, 0] is a unique 1-based id, with two sort keys and an obsm."""
    cell_type = ["B", "A", "A", "B", "A", "B"]
    demographic = ["y", "x", "y", "x", "x", "y"]
    dense = np.zeros((6, 3), dtype=np.float32)
    dense[:, 0] = np.arange(1, 7)
    dense[:, 1] = np.arange(1, 7) * 10.0
    obs = pd.DataFrame({"cell_type": cell_type, "demographic": demographic}, index=[str(i) for i in range(6)])
    adata = ad.AnnData(X=sp.csr_matrix(dense), obs=obs)
    adata.obsm["coords"] = np.arange(12, dtype=np.float64).reshape(6, 2)
    adata.write_h5ad(path)


def make_10x_v3_h5(path: Path, *, include_non_gex: bool = False, int32_data: bool = False) -> Path:
    """A Cell Ranger v3 h5 (``matrix/``); dense X == [[1, 0, 3], [0, 2, 0]], plus an antibody feature if asked."""
    dtype = np.int32 if int32_data else np.float32
    ids = [b"ENSG001", b"ENSG002", b"ENSG003"]
    names = [b"GENEA", b"GENEB", b"GENEC"]
    ftypes = [b"Gene Expression"] * 3
    data, indices, indptr = [1, 3, 2], [0, 2, 1], [0, 2, 3]
    if include_non_gex:
        ids.append(b"AB001")
        names.append(b"ABC1")
        ftypes.append(b"Antibody Capture")
        data, indices, indptr = [1, 3, 9, 2, 5], [0, 2, 3, 1, 3], [0, 3, 5]
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
        m.create_dataset("data", data=np.array(data, dtype=dtype))
        m.create_dataset("indices", data=np.array(indices, dtype=np.int32))
        m.create_dataset("indptr", data=np.array(indptr, dtype=np.int32))
        m.create_dataset("shape", data=np.array([len(ids), 2], dtype=np.int32))
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array(ids))
        feat.create_dataset("name", data=np.array(names))
        feat.create_dataset("feature_type", data=np.array(ftypes))
        feat.create_dataset("genome", data=np.array([b"GRCh38"] * len(ids)))
    return path


def make_10x_v2_h5(path: Path, *, genomes: tuple[str, ...] = ("GRCh38",)) -> Path:
    """A Cell Ranger v2 h5: one ``<genome>`` group per entry; the same 2 x 3 matrix as v3."""
    with h5py.File(path, "w") as f:
        for genome in genomes:
            g = f.create_group(genome)
            g.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
            g.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
            g.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
            g.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
            g.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
            g.create_dataset("genes", data=np.array([b"ENSG001", b"ENSG002", b"ENSG003"]))
            g.create_dataset("gene_names", data=np.array([b"GENEA", b"GENEB", b"GENEC"]))
    return path


TENX_DENSE = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=np.float32)


def random_dense(n_obs: int, n_vars: int, *, seed: int, density: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n_obs, n_vars)) < density
    return (mask * rng.random((n_obs, n_vars))).astype(np.float32)


def read_x(store: Path | str) -> np.ndarray:
    x = ad.read_zarr(str(store)).X
    return np.asarray(x.todense() if sp.issparse(x) else x)


def tree_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}


def assert_byte_identical(a: Path, b: Path) -> None:
    fa, fb = tree_files(a), tree_files(b)
    assert set(fa) == set(fb), (
        f"file set differs: only in {a.name}: {set(fa) - set(fb)}, only in {b.name}: {set(fb) - set(fa)}"
    )
    for rel in sorted(fa):
        assert filecmp.cmp(fa[rel], fb[rel], shallow=False), f"differs: {rel}"


def assert_anndata_readable(store: Path | str | zarr.Group) -> None:
    """``ad.read_zarr`` opens it and every element carries encoding-type and encoding-version."""
    root = store if isinstance(store, zarr.Group) else zarr.open_group(str(store), mode="r")
    got = ad.read_zarr(root)
    assert got.n_obs >= 0 and got.n_vars >= 0
    assert root.attrs.get("encoding-type") == "anndata" and root.attrs.get("encoding-version")

    def walk(node: zarr.Array | zarr.Group) -> None:
        enc = node.attrs.get("encoding-type")
        assert enc, f"missing encoding-type on {node.path!r}"
        assert node.attrs.get("encoding-version"), f"missing encoding-version on {node.path!r}"
        if enc in ("csr_matrix", "csc_matrix"):
            return  # data/indices/indptr are structural children, not individually encoded
        if isinstance(node, zarr.Group):
            for child in node:
                walk(node[child])

    for top in root:
        walk(root[top])


def messages(caplog: Any) -> list[str]:
    return [r.message for r in caplog.records]


# workers for the process-pool tests: a spawned worker imports this module by name, which it could
# not do for a test module registered under importlib mode
def worker_zarr_threads(_: int) -> object:
    import zarr

    return zarr.config.get("threading.max_workers")


def double(x: int) -> int:
    return x * 2


def identity(x: int) -> int:
    return x


def raise_on(bad: int, x: int) -> int:
    if x == bad:
        raise ValueError(f"boom-produce-{x}")
    return x


def sleep_reverse(x: int) -> int:
    import time

    time.sleep(0.002 * (20 - x))  # later jobs finish first
    return x


def exit_abruptly(x: int) -> int:
    # stands in for a worker the kernel kills (the OOM killer): no exception, the process just ends
    import os

    if x == 1:
        os._exit(1)
    return x
