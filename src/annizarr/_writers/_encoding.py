from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pandas as pd
from anndata._io.specs import write_elem  # private API; checked against anndata 0.12.7 through 0.13.4
from pandas.api.types import infer_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    import zarr

__all__ = [
    "mark_nullable_strings",
    "prepare_frame",
    "sparse_shards",
    "suppress_autoshard_warning",
    "write_elem",
    "write_frame",
    "write_settings",
]

ARRAY_ENCODING_TYPE = "array"
ARRAY_ENCODING_VERSION = "0.2.0"
ANNDATA_ENCODING_TYPE = "anndata"
ANNDATA_ENCODING_VERSION = "0.1.0"
RAW_ENCODING_TYPE = "raw"
RAW_ENCODING_VERSION = "0.1.0"
CSR_ENCODING_TYPE = "csr_matrix"
CSC_ENCODING_TYPE = "csc_matrix"
SPARSE_ENCODING_VERSION = "0.1.0"
STRING_DTYPE = pd.StringDtype(na_value=float("nan"))  # pandas>=3's default `str`, available back to pandas 2.3


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A shallow copy of ``df`` laid out the way anndata's own writers store a frame.

    String columns whose values repeat become categoricals, by the same rule
    ``AnnData.write_zarr``/``write_h5ad`` apply (``strings_to_categoricals``: a column stays a
    string array only when every value is unique, e.g. barcodes or ids). The index and every
    column that stays a string are coerced to pandas>=3's ``str`` dtype, so they are written as
    a ``nullable-string-array`` on every pandas version instead of following whichever dtype the
    installed pandas infers for text. That is byte for byte what anndata's own writers produce
    on pandas>=3; :func:`write_frame` keeps it so on every anndata version. The caller's frame is
    left untouched.
    """
    import anndata as ad

    out = df.copy(deep=False)
    if _is_string_like(out.index):
        out.index = out.index.astype(STRING_DTYPE)
    ad.AnnData().strings_to_categoricals(out)  # converts columns on the copy, in place
    for key in out.columns:
        if _is_string_like(out[key]):
            out[key] = out[key].astype(STRING_DTYPE)
    return out


def _is_string_like(values: pd.Series[Any] | pd.Index[Any]) -> bool:
    if isinstance(values.dtype, pd.StringDtype):
        return True
    return bool(values.dtype == object) and infer_dtype(values) == "string"


def write_frame(parent: zarr.Group, key: str, df: pd.DataFrame) -> None:
    """Write ``df`` under ``parent[key]`` the way anndata's own writers would on pandas>=3.

    ``prepare_frame`` lays the frame out, anndata's ``write_elem`` encodes it (inside
    :func:`write_settings`, which lets anndata<0.13 write nullable strings at all), and the
    ``na-value`` attribute anndata<0.13 leaves off every ``nullable-string-array`` is filled in, so
    the store is byte-identical across anndata versions and reads back as pandas>=3's ``str``.
    """
    write_elem(parent, key, prepare_frame(df))
    mark_nullable_strings(parent.require_group(key))


def mark_nullable_strings(group: zarr.Group) -> None:
    """Tag every ``nullable-string-array`` under ``group`` with ``na-value: NaN`` when it has none.

    anndata>=0.13 writes the attribute itself and uses it to pick pandas>=3's ``str`` dtype on
    read; anndata 0.12 writes the same values and mask without it. The missing-value sentinel is
    the only thing the attribute changes, so adding it never alters data.
    """
    import zarr

    for _, node in group.members(max_depth=None):
        if isinstance(node, zarr.Group) and node.attrs.get("encoding-type") == "nullable-string-array":
            if "na-value" not in node.attrs:
                node.attrs.put({"na-value": "NaN", **node.attrs})  # first, where anndata>=0.13 puts it: same bytes


def set_array_attrs(arr: zarr.Array[Any]) -> None:
    arr.attrs["encoding-type"] = ARRAY_ENCODING_TYPE
    arr.attrs["encoding-version"] = ARRAY_ENCODING_VERSION


def set_anndata_root_attrs(group: zarr.Group) -> None:
    group.attrs["encoding-type"] = ANNDATA_ENCODING_TYPE
    group.attrs["encoding-version"] = ANNDATA_ENCODING_VERSION


def set_raw_group_attrs(group: zarr.Group) -> None:
    group.attrs["encoding-type"] = RAW_ENCODING_TYPE
    group.attrs["encoding-version"] = RAW_ENCODING_VERSION


@contextmanager
def write_settings(auto_shard: bool) -> Iterator[None]:
    # anndata's write-side settings for the duration of our write_elem calls, restored on exit:
    # shards="auto" when asked, and nullable strings allowed (anndata<0.13 defaults that to False
    # and refuses pandas string arrays; >=0.13 defers to pandas' infer_string, False on pandas 2)
    import anndata as ad

    previous = ad.settings.auto_shard_zarr_v3, ad.settings.allow_write_nullable_strings
    with suppress_autoshard_warning(auto_shard):  # anndata<0.12.14 lets zarr's warning out of its own writes
        ad.settings.auto_shard_zarr_v3 = auto_shard
        ad.settings.allow_write_nullable_strings = True
        try:
            yield
        finally:
            ad.settings.auto_shard_zarr_v3, ad.settings.allow_write_nullable_strings = previous


@contextmanager
def suppress_autoshard_warning(enabled: bool) -> Iterator[None]:
    # mirrors anndata's own decorator, for raw require_array calls it never sees
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Automatic shard shape inference is experimental", UserWarning)
        yield


def sparse_shards(auto_shard: bool) -> Any:
    return "auto" if auto_shard else None


def make_sparse_group(parent: zarr.Group, key: str, *, csr: bool, shape: tuple[int, int]) -> zarr.Group:
    group = parent.require_group(key)
    group.attrs["encoding-type"] = CSR_ENCODING_TYPE if csr else CSC_ENCODING_TYPE
    group.attrs["encoding-version"] = SPARSE_ENCODING_VERSION
    group.attrs["shape"] = list(shape)
    return group
