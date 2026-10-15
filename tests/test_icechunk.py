"""Everything the icechunk extra adds: the install guard, one commit per op, branches and
snapshots, lazy inputs streaming into a session, and repo copies through S3."""

from __future__ import annotations

import importlib
import sys
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import zarr
from anndata.io import read_elem, sparse_dataset

import annizarr as az
from annizarr._storage import open_input_group, open_output_store
from annizarr.config import AppConfig, ChunkConfig, GroupingConfig, IOConfig
from annizarr.errors import RepoError, StorageError
from annizarr.ops import add_expr, append, convert_h5ad, sort
from helpers import (
    assert_anndata_readable,
    cfg,
    expected_lognorm,
    make_adata,
    make_h5ad,
    make_labelled_h5ad,
    make_store,
)


def test_missing_icechunk_extra_raises_the_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import annizarr
    from annizarr._storage import require_icechunk
    from annizarr.ic._copy import check_copyable

    monkeypatch.setitem(sys.modules, "icechunk", None)  # any later `import icechunk` raises ImportError
    monkeypatch.delitem(sys.modules, "annizarr.ic", raising=False)
    assert hasattr(importlib.import_module("annizarr.ic"), "Repo")  # the facade imports without the extra
    with pytest.raises(StorageError, match=r"annizarr\[icechunk\]"):
        annizarr.Repo(str(tmp_path / "nonexistent"))
    with pytest.raises(StorageError, match=r"annizarr\[icechunk\]"):
        require_icechunk()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RepoError, match=r"annizarr\[icechunk\]"):
        check_copyable("some/local/repo", "s3://bucket/prefix")


def test_remote_output_without_the_backend_errors_before_importing_icechunk() -> None:
    saved = sys.modules.pop("icechunk", None)
    try:
        with pytest.raises(StorageError, match="--ic"):
            open_output_store("s3://bucket/store.zarr", AppConfig())
        assert "icechunk" not in sys.modules
    finally:
        if saved is not None:
            sys.modules["icechunk"] = saved


pytestmark_ic = pytest.mark.usefixtures("_icechunk")


@pytest.fixture
def _icechunk() -> None:
    pytest.importorskip("icechunk")


@pytestmark_ic
def test_lifecycle_is_one_commit_per_op(tmp_path: Path) -> None:
    """convert, add-expr, append and sort each commit exactly once; sort re-derives a lone gexp
    layer inside its own commit; a plain-zarr result carries no snapshot id."""
    from annizarr.ic import Repo

    a, b = make_adata(n_obs=30, seed=0), make_adata(n_obs=12, seed=1)
    repo_path = tmp_path / "repo.icechunk"
    r_convert = convert_h5ad(
        str(make_h5ad(tmp_path, "a.h5ad", adata=a)), output=str(repo_path), cfg=cfg(backend="icechunk")
    )
    r_expr = add_expr(str(repo_path), layout="csr", cfg=cfg(backend="icechunk"))
    cells = make_store(tmp_path, "b.zarr", adata=b)
    r_append = append(str(repo_path), cells=str(cells), drop_derived=True, cfg=cfg())  # icechunk auto-detected
    add_expr(str(repo_path), layout="csc", cfg=cfg())
    sorted_path = tmp_path / "sorted.icechunk"
    r_sort = sort(str(repo_path), output=str(sorted_path), by=("cell_type",), cfg=cfg(backend="icechunk"))
    assert all(isinstance(r.snapshot_id, str) and r.snapshot_id for r in (r_convert, r_expr, r_append, r_sort))
    plain = convert_h5ad(str(tmp_path / "a.h5ad"), output=str(tmp_path / "plain.zarr"), cfg=cfg())
    assert plain.snapshot_id is None  # a plain-zarr result carries no snapshot id

    log = Repo(str(repo_path)).log(branch="main")
    assert len(log) == 5 and log[-1].message == "Repository initialized"
    assert [s.id for s in log[1:4]] == [r_append.snapshot_id, r_expr.snapshot_id, r_convert.snapshot_id]
    assert (
        "annizarr append" in log[1].message
        and "annizarr add-expr csr" in log[2].message
        and "annizarr convert" in log[3].message
    )
    assert log[1].metadata == {"annizarr": az.__version__, "op": "append", "cells": "b.zarr"}
    assert log[3].metadata == {"annizarr": az.__version__, "op": "convert", "source": "a.h5ad"}
    sorted_log = Repo(str(sorted_path)).log(branch="main")
    assert (
        len(sorted_log) == 2 and sorted_log[0].id == r_sort.snapshot_id and "sort by cell_type" in sorted_log[0].message
    )

    root = open_input_group(str(sorted_path))
    assert_anndata_readable(root)
    obs = read_elem(root["obs"])
    assert len(obs) == 42 and (np.diff(obs["cell_type"].cat.codes.to_numpy()) >= 0).all()
    x = sparse_dataset(root["X"])[:]
    by_name = {n: r for src in (a, b) for n, r in zip(src.obs_names, src.X.toarray(), strict=True)}
    for i, n in enumerate(obs.index):
        np.testing.assert_allclose(x[i].toarray().ravel(), by_name[n])
    np.testing.assert_allclose(sparse_dataset(root["layers/gexp"])[:].toarray(), expected_lognorm(x), rtol=1e-5)


@pytestmark_ic
def test_branches_messages_and_snapshots(tmp_path: Path) -> None:
    from annizarr.ic import Repo

    repo_path = tmp_path / "repo.icechunk"
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=20, seed=0))
    r_convert = convert_h5ad(str(h5), output=str(repo_path), cfg=cfg(backend="icechunk"), message="a custom message")
    assert Repo(str(repo_path)).log(branch="main")[0].message == "a custom message"

    r_dev = add_expr(str(repo_path), layout="csr", cfg=cfg(), branch="dev")  # created off main's tip
    assert Repo(str(repo_path), branch="main").log(branch="main")[0].id == r_convert.snapshot_id
    dev_log = Repo(str(repo_path), branch="dev").log(branch="dev")
    assert dev_log[0].id == r_dev.snapshot_id and r_convert.snapshot_id in {s.id for s in dev_log}

    r_main = add_expr(str(repo_path), layout="csr", cfg=cfg())
    assert r_main.snapshot_id is not None
    old = open_input_group(str(repo_path), snapshot_id=r_convert.snapshot_id)
    assert "gexp" not in (list(old["layers"]) if "layers" in old else [])
    assert "gexp" in list(open_input_group(str(repo_path))["layers"])

    out = open_output_store(tmp_path / "idem.icechunk", AppConfig(io=IOConfig(backend="icechunk")))
    out.root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    out.root.require_array("X", shape=(1, 1), dtype="float32")
    first = out.finalize()
    assert first is not None and out.finalize() == first  # cached, not a second empty commit


@pytestmark_ic
@pytest.mark.parametrize("layout", ["csr", "dense", "csc"])
def test_lazy_input_streams_into_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A session cannot be written from a worker process: csr and dense run the read-ahead pipeline,
    csc transposes with workers reading and this process placing; all in one commit."""
    monkeypatch.setattr("annizarr._core._layout.BATCH_BYTES", 2000)
    adata = make_adata(n_obs=60, n_vars=9, seed=5)
    h5 = make_h5ad(tmp_path, adata=adata)
    out = tmp_path / f"{layout}.icechunk"
    config = AppConfig(
        io=IOConfig(backend="icechunk", layout=layout, lazy=True),
        chunks=ChunkConfig(row_chunk=8, col_chunk=4, nnz_chunk=50, cpus=2),
    )
    with caplog.at_level("INFO", logger="annizarr"):
        result = convert_h5ad(str(h5), output=str(out), cfg=config)
    assert result.snapshot_id is not None
    if layout != "csc":
        assert any("reader processes feed the writer threads" in r.message for r in caplog.records)
    root = open_input_group(str(out))
    got = root["X"][:] if layout == "dense" else sparse_dataset(root["X"])[:]
    if layout == "csc":
        assert root["X"].attrs["encoding-type"] == "csc_matrix" and got.has_sorted_indices
    np.testing.assert_array_equal(np.asarray(got if layout == "dense" else got.toarray()), adata.X.toarray())
    assert not list(tmp_path.glob("annizarr_transpose_*"))


@pytestmark_ic
def test_sorted_convert_through_icechunk_names_the_sort_keys(tmp_path: Path) -> None:
    from annizarr.ic import Repo

    make_labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "repo.icechunk"
    grouping = GroupingConfig(enabled=True, sort_by=("cell_type", "demographic"))
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg(backend="icechunk", lazy=True, grouping=grouping))
    assert (
        Repo(str(out)).log(branch="main")[0].message
        == "annizarr convert (sorted by cell_type,demographic) → repo.icechunk"
    )
    root = open_input_group(str(out), branch="main")
    assert list(np.asarray(sparse_dataset(root["X"])[:][:, 0].todense()).ravel().astype(int)) == [2, 5, 3, 4, 1, 6]


@pytestmark_ic
def test_failed_convert_leaves_no_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from annizarr.ic import Repo

    def boom(*_a: object, **_kw: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(importlib.import_module("annizarr.ops._convert"), "write_adata", boom)
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(tmp_path / "repo.icechunk"), cfg=cfg(backend="icechunk"))
    assert [s.message for s in Repo(str(tmp_path / "repo.icechunk")).log(branch="main")] == ["Repository initialized"]


@pytestmark_ic
def test_a_repo_is_a_store(tmp_path: Path) -> None:
    """A Repo is used as given: its branch receives the op, and a staged append is what a plan reads."""
    from annizarr.ic import Repo
    from annizarr.ops import plan_append

    repo_path = tmp_path / "repo.icechunk"
    convert_h5ad(
        str(make_h5ad(tmp_path, adata=make_adata(n_obs=20, seed=0))), output=str(repo_path), cfg=cfg(backend="icechunk")
    )
    main_tip = Repo(str(repo_path)).log()[0].id
    repo = Repo(str(repo_path))
    repo.checkout("dev", create=True)
    result = add_expr(repo, layout="csr", cfg=cfg())
    assert result.path == str(repo_path) and repo.branch == "dev" and not repo.dirty
    assert repo.log()[0].id == result.snapshot_id and Repo(str(repo_path)).log()[0].id == main_tip

    cells = make_store(tmp_path, "b.zarr", adata=make_adata(n_obs=12, seed=1))
    assert plan_append(repo, cells=str(cells)).extendable_layers == ("gexp",)
    del repo.open_zarr("w")["layers/gexp"]
    assert repo.dirty and plan_append(repo, cells=str(cells)).extendable_layers == ()


@pytestmark_ic
def test_deferred_ops_land_in_the_callers_commit(tmp_path: Path) -> None:
    from annizarr.ic import Repo

    repo_path = tmp_path / "repo.icechunk"
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=20, seed=0))
    convert_h5ad(str(h5), output=str(repo_path), cfg=cfg(backend="icechunk"))
    cells = make_store(tmp_path, "b.zarr", adata=make_adata(n_obs=12, seed=1))
    repo = Repo(str(repo_path))
    n_before = len(repo.log())
    r_append = append(repo, cells=str(cells), drop_derived=True, cfg=cfg(), commit=False)
    r_expr = add_expr(repo, layout="csr", cfg=cfg(), commit=False)
    assert r_append.snapshot_id is None and r_expr.snapshot_id is None and repo.dirty
    repo.commit("ingest", metadata={"run": "42"})
    log = repo.log()
    assert len(log) == n_before + 1 and log[0].metadata == {"run": "42"}
    root = open_input_group(str(repo_path))
    assert root["X"].attrs["shape"][0] == 32 and "gexp" in root["layers"]

    with pytest.raises(StorageError, match="needs a Repo"):
        add_expr(str(repo_path), cfg=cfg(), commit=False)
    plain = make_store(tmp_path, "plain.zarr", adata=make_adata(n_obs=5, seed=2))
    with pytest.raises(StorageError, match="only applies to Icechunk"):
        add_expr(str(plain), cfg=cfg(), commit=False)


@pytest.fixture
def moto_s3(monkeypatch: pytest.MonkeyPatch):
    """A live moto S3 server with two fresh buckets; icechunk's Rust core needs a real endpoint."""
    pytest.importorskip("icechunk")
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    for var, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_DEFAULT_REGION", "us-east-1"),
        ("AWS_ENDPOINT_URL", endpoint),
        ("AWS_ALLOW_HTTP", "true"),
    ):
        monkeypatch.setenv(var, value)
    client = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
    buckets = [f"bucket-{uuid.uuid4().hex[:8]}" for _ in range(2)]
    for b in buckets:
        client.create_bucket(Bucket=b)
    try:
        yield client, endpoint, *buckets
    finally:
        server.stop()


def _objects_local(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _objects_s3(client: object, bucket: str, prefix: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):  # type: ignore[attr-defined]
        for obj in page.get("Contents", []):
            out[obj["Key"][len(prefix) + 1 :]] = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()  # type: ignore[attr-defined]
    return out


def test_copy_repo_round_trips_through_s3(moto_s3, tmp_path: Path) -> None:
    import icechunk

    from annizarr.ic import Repo, copy_repo

    client, endpoint, bucket_a, bucket_b = moto_s3
    repo_path = tmp_path / "repo.icechunk"
    convert_h5ad(
        str(make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))), output=str(repo_path), cfg=cfg(backend="icechunk")
    )
    local = _objects_local(repo_path)

    def open_on_moto(bucket: str, prefix: str):
        storage = icechunk.s3_storage(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint,
            allow_http=True,
            region="us-east-1",
            access_key_id="testing",
            secret_access_key="testing",
            force_path_style=True,
        )
        return icechunk.Repository.open(storage)

    copy_repo(str(repo_path), f"s3://{bucket_a}/p1")
    assert _objects_s3(client, bucket_a, "p1") == local and open_on_moto(bucket_a, "p1").list_branches() == {"main"}
    copy_repo(f"s3://{bucket_a}/p1", f"s3://{bucket_b}/p2")
    assert _objects_s3(client, bucket_b, "p2") == local
    copy_repo(f"s3://{bucket_b}/p2", str(tmp_path / "back.icechunk"))
    assert _objects_local(tmp_path / "back.icechunk") == local and Repo(tmp_path / "back.icechunk").branches() == [
        "main"
    ]
    assert ad.read_zarr(zarr.open_group(open_input_group(str(tmp_path / "back.icechunk")).store, mode="r")).n_obs == 10
