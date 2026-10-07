from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annizarr._core._runtime import stage
from annizarr._core._zarr import get_group
from annizarr._sources._readers import as_reader
from annizarr._writers._encoding import (
    autoshard_setting,
    prepare_frame,
    set_anndata_root_attrs,
    set_raw_group_attrs,
    write_elem,
)
from annizarr._writers._matrix import write_matrix
from annizarr._writers._sparse import _local_tmp_dir

if TYPE_CHECKING:
    import anndata as ad
    import zarr

    from annizarr._core._config import AppConfig


def named_layers(adata: ad.AnnData) -> dict[str, Any]:
    """The layers proper: anndata>=0.13 also lists X in ``.layers`` under the key ``None``."""
    return {name: data for name, data in adata.layers.items() if name is not None}


def write_adata(adata: ad.AnnData, store: zarr.Group, cfg: AppConfig) -> None:
    set_anndata_root_attrs(store)

    with autoshard_setting(cfg.chunks.auto_shard), stage("Writing metadata (obs/var/uns/obsm/varm/obsp/varp)"):
        write_elem(store, "obs", prepare_frame(adata.obs))
        write_elem(store, "var", prepare_frame(adata.var))
        write_elem(store, "uns", dict(adata.uns))
        write_elem(store, "obsm", dict(adata.obsm))
        write_elem(store, "varm", dict(adata.varm))
        write_elem(store, "obsp", dict(adata.obsp))
        write_elem(store, "varp", dict(adata.varp))

    tmp_dir = _local_tmp_dir(store)
    x_reader = as_reader(adata.X, cfg=cfg, tmp_dir=tmp_dir)
    try:
        with stage(f"Writing X (shape={x_reader.shape}, {cfg.io.layout})"):
            write_matrix(store, "X", x_reader, cfg)
    finally:
        x_reader.close()

    layers = named_layers(adata)
    if layers:
        with autoshard_setting(cfg.chunks.auto_shard):
            write_elem(store, "layers", {})
        layers_group = get_group(store, "layers")
        for name, data in layers.items():
            reader = as_reader(data, cfg=cfg, tmp_dir=tmp_dir)
            try:
                with stage(f"Writing layers/{name} (shape={reader.shape})"):
                    write_matrix(layers_group, name, reader, cfg)
            finally:
                reader.close()

    if adata.raw is not None:
        raw: Any = adata.raw  # anndata 0.13 types Raw.varm against AnnData rather than Raw
        raw_group = store.require_group("raw")
        set_raw_group_attrs(raw_group)
        with autoshard_setting(cfg.chunks.auto_shard):
            write_elem(raw_group, "var", prepare_frame(raw.var))
            write_elem(raw_group, "varm", dict(raw.varm))
        reader = as_reader(raw.X, cfg=cfg, tmp_dir=tmp_dir)
        try:
            with stage(f"Writing raw/X (shape={reader.shape})"):
                write_matrix(raw_group, "X", reader, cfg)
        finally:
            reader.close()
