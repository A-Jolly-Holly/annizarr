from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os

    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    from annizarr._sources._readers import Reader

_FEATURE_EXTRA_EXCLUDE = frozenset({"name", "feature_type", "id", "gene_id", "_all_tag_keys"})
_GEX = "Gene Expression"


@dataclass(frozen=True, slots=True)
class _TenxFrames:
    group: str  # the matrix group's path inside the file
    obs: pd.DataFrame
    var: pd.DataFrame
    keep: NDArray[np.bool_] | None  # features to keep (None: all), applied to var already
    n_obs: int
    n_features: int


def load_10x_h5(path: str | os.PathLike[str]) -> ad.AnnData:
    """Load a 10x Cell Ranger h5 whole, as an in-memory CSR AnnData of its Gene Expression features."""
    import anndata as ad
    import h5py
    from scipy.sparse import csr_matrix

    with h5py.File(str(path), "r") as f:
        frames = _frames(f)
        g = f[frames.group]
        data = _as_float32_if_int32(g["data"][()])
        indices = g["indices"][()]
        indptr = g["indptr"][()]
    matrix = csr_matrix((data, indices, indptr), shape=(frames.n_obs, frames.n_features))
    if frames.keep is not None:
        matrix = matrix[:, frames.keep]
    return ad.AnnData(matrix, obs=frames.obs, var=frames.var)


def load_10x_lazy(path: str | os.PathLike[str]) -> tuple[ad.AnnData, Reader]:
    """Open a 10x Cell Ranger h5 for streaming: obs and var in an X-less AnnData, X as a reader.

    ``matrix/`` is already cells-by-genes CSR on disk (``indptr`` over barcodes, ``indices`` over
    features), so the reader streams it directly; non-Gene-Expression features are masked out
    band by band.
    """
    import anndata as ad
    import h5py

    from annizarr._sources._readers import TenxCSRReader

    with h5py.File(str(path), "r") as f:
        frames = _frames(f)
    reader = TenxCSRReader(str(path), frames.group, n_obs=frames.n_obs, n_features=frames.n_features, keep=frames.keep)
    return ad.AnnData(obs=frames.obs, var=frames.var), reader


def _as_float32_if_int32(data: NDArray[Any]) -> NDArray[Any]:
    import numpy as np

    # some Cell Ranger versions tag float32 counts as int32; widen to match
    return data.astype(np.float32) if data.dtype == np.dtype("int32") else data


def _frames(f: h5py.File) -> _TenxFrames:
    if "matrix" in f:
        return _v3_frames(f)
    return _legacy_frames(f)


def _v3_frames(f: h5py.File) -> _TenxFrames:
    import h5py
    import numpy as np
    import pandas as pd

    m = f["matrix"]
    try:
        shape = m["shape"][()]  # on-disk shape is (n_features, n_barcodes)
        n_features, n_obs = int(shape[0]), int(shape[1])
        barcodes = m["barcodes"][()].astype(str)
        if "features" not in m:
            raise ValueError("10x v3 h5 file has no features group")
        feat = m["features"]
        names = feat["name"][()].astype(str)
        feature_types = feat["feature_type"][()].astype(str)
        if "gene_id" in feat:
            # probe-barcode matrix: "id" is the probe id, "gene_id" the underlying gene
            gene_ids = feat["gene_id"][()].astype(str)
            probe_ids: NDArray[Any] | None = feat["id"][()].astype(str)
        else:
            gene_ids = feat["id"][()].astype(str)
            probe_ids = None
    except KeyError as exc:
        raise ValueError(f"10x v3 h5 file is missing a required dataset: {exc}") from exc

    var_cols: dict[str, Any] = {"gene_ids": gene_ids}
    if probe_ids is not None:
        var_cols["probe_ids"] = probe_ids
    var_cols["feature_types"] = feature_types
    for key, dset in feat.items():
        if isinstance(dset, h5py.Dataset) and key not in _FEATURE_EXTRA_EXCLUDE:
            values = dset[()]
            var_cols[key] = values.astype(bool) if values.dtype.kind == "b" else values.astype(str)

    obs_cols: dict[str, Any] = {}
    if "filtered_barcodes" in m:
        obs_cols["filtered_barcodes"] = m["filtered_barcodes"][()].astype(bool)
    obs = pd.DataFrame(obs_cols, index=barcodes)

    keep = np.asarray(feature_types == _GEX)
    var = pd.DataFrame(var_cols, index=names)
    if keep.all():
        return _TenxFrames("matrix", obs, var, None, n_obs, n_features)
    return _TenxFrames("matrix", obs, var[keep], keep, n_obs, n_features)


def _legacy_frames(f: h5py.File) -> _TenxFrames:
    import pandas as pd

    children = list(f.keys())
    if not children:
        raise ValueError(f"{f.filename} has no top-level groups; not a recognised 10x v2 layout.")
    if len(children) > 1:
        raise ValueError(
            f"{f.filename} contains more than one genome. Legacy 10x h5 files with multiple "
            f"genomes are not supported here. Available genomes: {children}"
        )
    g = f[children[0]]
    try:
        shape = g["shape"][()]  # on-disk shape is (n_genes, n_barcodes)
        n_features, n_obs = int(shape[0]), int(shape[1])
        barcodes = g["barcodes"][()].astype(str)
        gene_names = g["gene_names"][()].astype(str)
        gene_ids = g["genes"][()].astype(str)
    except KeyError as exc:
        raise ValueError(f"10x v2 h5 file is missing a required dataset: {exc}") from exc
    var = pd.DataFrame({"gene_ids": gene_ids}, index=gene_names)
    return _TenxFrames(children[0], pd.DataFrame(index=barcodes), var, None, n_obs, n_features)
