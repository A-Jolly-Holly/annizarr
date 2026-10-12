"""Input detection, the source registry, and the loaders behind it."""

from __future__ import annotations

import filecmp
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

import annizarr
from annizarr._sources._tenx import load_10x_h5
from annizarr.config import AppConfig, IOConfig
from annizarr.errors import ConversionError
from annizarr.ops import convert
from annizarr.sources import Source, detect_format, open_source, register_source
from helpers import (
    TENX_DENSE,
    assert_anndata_readable,
    cfg,
    make_10x_v2_h5,
    make_10x_v3_h5,
    make_adata,
    make_h5ad,
    tree_files,
)


def test_detect_format_by_content_not_extension(tmp_path: Path) -> None:
    assert detect_format(make_h5ad(tmp_path, "in.h5ad", n_obs=4, n_vars=3)) == "h5ad"
    assert detect_format(make_h5ad(tmp_path, "in.h5", n_obs=4, n_vars=3)) == "h5ad"
    assert detect_format(make_10x_v3_h5(tmp_path / "matrix.h5")) == "10x"
    assert detect_format(make_10x_v2_h5(tmp_path / "raw_gene_bc_matrices.h5")) == "10x"
    zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    assert detect_format(tmp_path / "store.zarr") == "zarr"
    (tmp_path / "repo.icechunk" / "snapshots").mkdir(parents=True)
    (tmp_path / "repo.icechunk" / "repo").write_text("stub")
    assert detect_format(tmp_path / "repo.icechunk") == "icechunk"
    assert detect_format("s3://bucket/prefix") == "icechunk" and detect_format("gs://bucket/prefix") == "icechunk"


def test_detect_format_guards(tmp_path: Path) -> None:
    (tmp_path / "plain_dir").mkdir()
    with pytest.raises(ConversionError, match="--from"):
        detect_format(tmp_path / "plain_dir")
    with h5py.File(tmp_path / "mystery.h5", "w") as f:
        f.create_dataset("foo", data=np.arange(4))
    with pytest.raises(ConversionError, match="not recognised as h5ad or 10x"):
        detect_format(tmp_path / "mystery.h5")
    (tmp_path / "notreal.h5ad").write_bytes(b"not an hdf5 file at all")
    with pytest.raises(ConversionError, match="not a readable HDF5"):
        detect_format(tmp_path / "notreal.h5ad")
    with pytest.raises(ConversionError, match="does not exist"):
        detect_format(tmp_path / "nope.h5ad")


def test_open_source_dispatches_and_reports_lazy_state(tmp_path: Path) -> None:
    h5ad = make_h5ad(tmp_path, "in.h5ad", n_obs=4, n_vars=3)
    eager = open_source(str(h5ad), cfg(lazy=False))
    assert isinstance(eager, Source) and eager.kind == "h5ad" and eager.adata.n_obs == 4 and not eager.lazy
    eager.close()
    eager.close()  # safe twice
    lazy = open_source(str(h5ad), cfg(lazy=True))
    assert lazy.lazy and lazy.adata.isbacked
    lazy.close()

    tenx = open_source(str(make_10x_v3_h5(tmp_path / "matrix.h5")), cfg())
    assert tenx.kind == "10x" and tenx.lazy and tenx.x_reader is not None and tenx.adata.n_obs == 2
    tenx.close()
    tenx_eager = open_source(str(tmp_path / "matrix.h5"), cfg(lazy=False))
    assert not tenx_eager.lazy and sp.isspmatrix_csr(tenx_eager.adata.X)

    backed = ad.read_h5ad(h5ad, backed="r")
    assert open_source(backed, cfg()).lazy is True  # the AnnData's own state, not the cfg's
    backed.file.close()
    memory = ad.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.float32)))
    src = open_source(memory, cfg())
    assert src.kind == "anndata" and not src.lazy and src.adata is memory

    with pytest.raises(ConversionError, match="registered kinds"):
        open_source(str(h5ad), cfg(), fmt="not-a-kind")


def test_registered_source_round_trips_through_convert(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    def loader(path: object, config: AppConfig) -> Source:
        adata = ad.AnnData(X=sp.csr_matrix(np.eye(5, 3, dtype=np.float32)))
        return Source(adata=adata, kind="toy", lazy=False, warnings=("toy loaded",))

    register_source("toy", loader, sniffer=lambda p: "toy" if p.suffix == ".toy" else None)
    (tmp_path / "input.toy").write_text("toy marker")
    result = convert(str(tmp_path / "input.toy"), output=str(tmp_path / "out.zarr"), cfg=cfg())
    assert any("toy loaded" in r.message for r in caplog.records)
    assert (result.n_obs, result.n_vars) == (5, 3) and ad.read_zarr(str(tmp_path / "out.zarr")).n_obs == 5


def test_convert_dispatch_and_guards(tmp_path: Path) -> None:
    """One path, a one-element sequence, two paths (concat) and an AnnData all route correctly."""
    p = make_h5ad(tmp_path, "in.h5ad", n_obs=4, n_vars=3)
    convert([str(p)], output=str(tmp_path / "single.zarr"), cfg=cfg())
    assert ad.read_zarr(str(tmp_path / "single.zarr")).n_obs == 4
    b = make_h5ad(tmp_path, "b.h5ad", n_obs=3, n_vars=3, seed=1)
    convert([str(p), str(b)], output=str(tmp_path / "concat.zarr"), cfg=cfg())
    assert_anndata_readable(tmp_path / "concat.zarr")
    assert ad.read_zarr(str(tmp_path / "concat.zarr")).n_obs == 7
    result = convert(make_adata(n_obs=6, n_vars=4, seed=3), output=str(tmp_path / "adata.zarr"), cfg=cfg())
    assert (result.n_obs, result.n_vars, result.snapshot_id) == (6, 4, None)

    with pytest.raises(ConversionError, match="at least one input"):
        convert([], output="unused.zarr", cfg=cfg())
    with pytest.raises(ConversionError, match="already a store"):
        convert(str(tmp_path / "single.zarr"), output=str(tmp_path / "out2.zarr"), cfg=cfg())
    with pytest.raises(ConversionError, match="Failed to read 10x H5 file"):  # fmt overrides detection
        convert(str(p), output=str(tmp_path / "out3.zarr"), cfg=cfg(), fmt="10x")
    assert all(hasattr(annizarr.sources, n) for n in ("Source", "Loader", "Sniffer", "open_source", "register_source"))


def test_10x_eager_loader_filters_features_widens_counts_and_guards(tmp_path: Path) -> None:
    v3 = load_10x_h5(make_10x_v3_h5(tmp_path / "v3.h5", include_non_gex=True, int32_data=True))
    assert sp.isspmatrix_csr(v3.X) and v3.X.dtype == np.float32 and v3.shape == (2, 3)
    np.testing.assert_array_equal(v3.X.toarray(), TENX_DENSE)
    assert list(v3.obs_names) == ["CELL1-1", "CELL2-1"] and list(v3.var_names) == ["GENEA", "GENEB", "GENEC"]
    assert list(v3.var.columns) == ["gene_ids", "feature_types", "genome"] and "ABC1" not in v3.var_names
    assert all(pd.api.types.is_string_dtype(v3.var[c]) for c in v3.var.columns)
    v2 = load_10x_h5(make_10x_v2_h5(tmp_path / "v2.h5"))
    np.testing.assert_array_equal(v2.X.toarray(), TENX_DENSE)
    assert list(v2.var.columns) == ["gene_ids"]

    with pytest.raises(ValueError, match="more than one genome"):
        load_10x_h5(make_10x_v2_h5(tmp_path / "multi.h5", genomes=("GRCh38", "mm10")))
    with h5py.File(tmp_path / "empty.h5", "w"):
        pass
    with pytest.raises(ValueError, match="not a recognised 10x v2 layout"):
        load_10x_h5(tmp_path / "empty.h5")
    with pytest.raises(ConversionError, match="Failed to read 10x H5 file"):  # the registry wraps it
        open_source(str(tmp_path / "empty.h5"), cfg(), fmt="10x")


@pytest.mark.parametrize("non_gex", [False, True])
def test_lazy_10x_convert_is_byte_identical_to_eager(tmp_path: Path, non_gex: bool) -> None:
    p = make_10x_v3_h5(tmp_path / "v3.h5", include_non_gex=non_gex, int32_data=True)
    lazy_out, eager_out = tmp_path / "lazy.zarr", tmp_path / "eager.zarr"
    annizarr.convert(str(p), output=str(lazy_out), cfg=cfg())
    annizarr.convert(str(p), output=str(eager_out), cfg=AppConfig(io=IOConfig(lazy=False), chunks=cfg().chunks))
    lazy_files, eager_files = tree_files(lazy_out), tree_files(eager_out)
    assert set(lazy_files) == set(eager_files)
    assert all(filecmp.cmp(lazy_files[k], eager_files[k], shallow=False) for k in lazy_files)
    np.testing.assert_array_equal(ad.read_zarr(str(lazy_out)).X.toarray(), TENX_DENSE)
