import os

import pytest

from annizarr._core._config import resolve_backend_cfg
from annizarr.config import AppConfig, ChunkConfig, IOConfig, apply_cli_overrides


def test_defaults() -> None:
    cfg = AppConfig()
    assert cfg.chunks.x_row_chunk == 2048
    assert cfg.io.x_storage == "csr"
    assert cfg.chunks.auto_shard is False
    assert cfg.io.lazy is True
    assert cfg.chunks.cpus == (os.cpu_count() or 1)


def test_resolve_backend_cfg_leaves_lazy_untouched_for_every_backend() -> None:
    # icechunk no longer forces eager or rejects lazy input: a lazy, not-thread-safe reader
    # into an icechunk session runs the read-ahead pipeline at write time instead (see
    # writer_parallel_mode / test_features.py's icechunk-lazy roundtrip).
    for backend in ("zarr", "icechunk"):
        for lazy in (True, False):
            resolved = resolve_backend_cfg(AppConfig(io=IOConfig(backend=backend, lazy=lazy)))
            assert resolved.io.lazy == lazy


def test_resolve_backend_cfg_warns_on_shard_factor_with_sparse_storage(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="annizarr")
    resolve_backend_cfg(AppConfig(chunks=ChunkConfig(x_shard_factor=2), io=IOConfig(x_storage="csr")))
    assert any("only applies to dense X" in r.message for r in caplog.records)


def test_cli_overrides() -> None:
    cfg = AppConfig()
    cfg2 = apply_cli_overrides(cfg, x_row_chunk=128, overwrite=True, x_storage="csr")
    assert cfg2.chunks.x_row_chunk == 128
    assert cfg2.io.overwrite is True
    assert cfg2.io.x_storage == "csr"

    assert cfg.chunks.auto_shard is False
    cfg3 = apply_cli_overrides(cfg, auto_shard=True)
    assert cfg3.chunks.auto_shard is True
    # None means "don't touch it" (a caller other than the CLI's own --auto-shard, which
    # always passes a bool)
    cfg4 = apply_cli_overrides(cfg3, auto_shard=None)
    assert cfg4.chunks.auto_shard is True


def test_cli_overrides_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="x_storage"):
        apply_cli_overrides(AppConfig(), x_storage="not-a-mode")
    with pytest.raises(ValueError, match="x_shard_factor"):
        apply_cli_overrides(AppConfig(), x_shard_factor=0)
