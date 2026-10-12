"""The argparse layer: flag mapping, exit codes, consent prompts, help, and the generated reference."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest
import zarr

from annizarr._cli import main
from annizarr._core._config import ChunkConfig
from annizarr._version import __version__
from annizarr.ops import add_expr
from helpers import assert_anndata_readable, expected_lognorm, make_10x_v3_h5, make_adata, make_h5ad

ROOT = Path(__file__).resolve().parents[1]


def test_convert_flags_map_to_config(tmp_path: Path) -> None:
    assert (
        main(["convert", str(make_10x_v3_h5(tmp_path / "m.h5")), "-o", str(tmp_path / "from.zarr"), "--from", "10x"])
        == 0
    )
    assert ad.read_zarr(str(tmp_path / "from.zarr")).n_vars == 3
    h5 = make_h5ad(tmp_path, "in.h5ad")
    with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
        assert main(["convert", str(h5), "-o", str(tmp_path / "eager.zarr"), "--eager", "--cpus", "2"]) == 0
    assert "backed" not in mocked.call_args_list[0].kwargs
    big = make_h5ad(tmp_path, "big.h5ad", adata=make_adata(n_obs=30, n_vars=20))
    assert (
        main(
            ["convert", str(big), "-o", str(tmp_path / "shard.zarr"), "--layout", "csr", "--auto-shard", "--cpus", "2"]
        )
        == 0
    )
    assert_anndata_readable(tmp_path / "shard.zarr")
    assert zarr.open_group(str(tmp_path / "shard.zarr"), mode="r")["X/data"].shards is not None


def test_errors_exit_codes_and_quiet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad")
    with pytest.raises(SystemExit) as exc:  # a usage error is argparse's exit 2
        main(["convert", str(h5), "-o", str(tmp_path / "out.zarr"), "--obs-columns", "cell_type"])
    assert exc.value.code == 2
    assert main(["convert", str(tmp_path / "missing.h5ad"), "-o", str(tmp_path / "out.zarr")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["convert", str(h5), "-o", "s3://bucket/out.zarr"]) == 1
    assert "--ic" in capsys.readouterr().err

    out = tmp_path / "out.zarr"
    assert main(["convert", str(h5), "-o", str(out), "--cpus", "2"]) == 0
    assert "wrote" in capsys.readouterr().err
    assert main(["convert", str(h5), "-o", str(out), "--cpus", "2"]) == 1  # exists, no --overwrite
    assert "error:" in capsys.readouterr().err
    assert main(["-q", "convert", str(h5), "-o", str(out), "--overwrite", "--cpus", "2"]) == 0
    assert "wrote" not in capsys.readouterr().err
    assert main(["add-expr", str(tmp_path / "missing.zarr")]) == 1


def test_op_subcommands_dispatch(tmp_path: Path) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=8, n_vars=5, seed=2))
    store = tmp_path / "store.zarr"
    assert main(["convert", str(h5), "-o", str(store), "--cpus", "2"]) == 0
    assert main(["add-expr", str(store), "--col-chunk", "2", "--cpus", "2"]) == 0
    assert_anndata_readable(store)
    assert "gexp" in ad.read_zarr(str(store)).layers
    assert main(["rechunk", str(store), "-o", str(tmp_path / "rechunked.zarr"), "--row-chunk", "2", "--cpus", "2"]) == 0
    assert ad.read_zarr(str(tmp_path / "rechunked.zarr")).n_obs == 8
    assert main(["sort", str(store), "-o", str(tmp_path / "sorted.zarr"), "--by", "cell_type", "--cpus", "2"]) == 0
    assert (np.diff(ad.read_zarr(str(tmp_path / "sorted.zarr")).obs["cell_type"].cat.codes.to_numpy()) >= 0).all()


def test_append_consent_and_extend_layers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = make_adata(n_obs=4, seed=0), make_adata(n_obs=2, seed=1)
    sa, sb = tmp_path / "a.zarr", tmp_path / "b.zarr"
    assert main(["convert", str(make_h5ad(tmp_path, "a.h5ad", adata=a)), "-o", str(sa), "--cpus", "2"]) == 0
    assert main(["convert", str(make_h5ad(tmp_path, "b.h5ad", adata=b)), "-o", str(sb), "--cpus", "2"]) == 0
    add_expr(str(sa), layout="csr")
    assert main(["append", str(sa), str(sb)]) == 1  # obsm would drop and stdin is not a TTY: no prompt, exit 1
    err = capsys.readouterr().err
    assert "error:" in err and "--drop-derived" in err and ad.read_zarr(str(sa)).n_obs == 4
    assert main(["append", str(sa), str(sb), "--extend-layers", "--drop-derived", "--cpus", "2"]) == 0
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 6 and "X_pca" not in got.obsm
    np.testing.assert_allclose(
        got.layers["gexp"].toarray(), expected_lognorm(np.vstack([a.X.toarray(), b.X.toarray()])), rtol=1e-5
    )

    a.obsm.clear()  # nothing to drop, so no consent needed
    assert (
        main(["convert", str(make_h5ad(tmp_path, "c.h5ad", adata=a)), "-o", str(tmp_path / "c.zarr"), "--cpus", "2"])
        == 0
    )
    assert main(["append", str(tmp_path / "c.zarr"), str(sb)]) == 0
    assert ad.read_zarr(str(tmp_path / "c.zarr")).n_obs == 6


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["convert", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "default: 2048" in out and f"{ChunkConfig().nnz_chunk} nonzeros" in out  # config defaults reach --help
    with pytest.raises(SystemExit):
        main(["--help"])
    assert capsys.readouterr().out.startswith("usage: annizarr")
    assert main(["--version"]) == 0 and capsys.readouterr().out.strip() == f"annizarr {__version__}"
    for exe in ("annizarr", "anz"):
        path = shutil.which(exe)
        if path is not None:
            assert (
                subprocess.run([path, "--version"], capture_output=True, text=True).stdout.strip()
                == f"annizarr {__version__}"
            )


def test_cli_import_leaves_numpy_unimported() -> None:
    code = "import sys\nimport annizarr._cli\nassert 'numpy' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_cli_reference_is_current() -> None:
    spec = importlib.util.spec_from_file_location("gen_cli_reference", ROOT / "docs" / "gen_cli_reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / "docs" / "cli.md").read_text(encoding="utf-8") == module.render(), "run docs/gen_cli_reference.py"
