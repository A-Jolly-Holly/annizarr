from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from anndata.io import read_elem

from annizarr._core import _layout
from annizarr._core._config import AppConfig
from annizarr._core._runtime import configure_runtime, run_parallel, stage
from annizarr._core._zarr import (
    as_array,
    as_group,
    get_array,
    get_group,
    has_element,
    matrix_shape,
    shape_attr,
    str_attr,
    str_list_attr,
)
from annizarr._storage import (
    op_metadata,
    open_input_group,
    open_store_rw,
    store_name,
    store_path,
    verify_store_shape,
)
from annizarr._writers._copy import copy_matrix
from annizarr._writers._encoding import nullable_series, write_frame_column, write_settings
from annizarr._writers._matrix import MatrixLayout, write_matrix
from annizarr._writers._sparse import _local_tmp_dir
from annizarr.errors import AnzError, ConversionError
from annizarr.ops._layer import SOURCE_ATTR
from annizarr.ops._result import AppendPlan, OpResult

if TYPE_CHECKING:
    import pandas as pd

    from annizarr._sources._readers import Reader
    from annizarr.typing import StoreLike

logger = logging.getLogger(__name__)

_INDEX_SCAN_ROWS = 1 << 20
_NULLABLE_ENCODINGS = ("nullable-integer", "nullable-boolean", "nullable-string-array")
# mutually appendable: the store column keeps its encoding and the incoming values are converted
_STRINGLIKE_ENCODINGS = ("categorical", "string-array", "nullable-string-array")
_CODE_WIDTHS = (np.int8, np.int16, np.int32, np.int64)
# a plain array of one of these kinds and its nullable encoding are one family: they append onto each other
_NULLABLE_FAMILY = {"i": "nullable-integer", "u": "nullable-integer", "b": "nullable-boolean"}


def plan_append(store: StoreLike, *, cells: StoreLike) -> AppendPlan:
    """Validate an append and describe what it would do, without mutating anything.

    Reads metadata only (attrs, obs schema, small index/indptr arrays, the cells' nullable masks) from both stores.

    Parameters
    ----------
    store
        Existing AnnData zarr store, or an Icechunk repo (path or :class:`~annizarr.ic.Repo`), the cells
        would be appended onto.
    cells
        Another AnnData zarr (or Icechunk) store whose cells would be appended.

    Returns
    -------
    AppendPlan
        What appending would do: new-cell count, derived elements that would be
        dropped, layers both stores carry (appended), and other notes.

    Raises
    ------
    ConversionError
        ``store``/``cells`` is not a CSR AnnData zarr store, ``var`` names/order
        mismatch, a shared obs column's encoding or dtype mismatch, a plain string obs
        column the cells lack, an X dtype mismatch, a shared layer whose cells dtype cannot be
        cast to the store's without loss, or every appended cell's obs name is already present
        in the store (already appended?).
    """
    root = open_input_group(store)
    src = open_input_group(cells)

    for g, label in ((root, "store"), (src, "cells")):
        if "X" not in g:
            raise ConversionError(f"no X in {label} store — not an AnnData zarr store?")
        x = get_group(g, "X")
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise ConversionError(f"append requires CSR X in {label}; got {x.attrs.get('encoding-type')!r}.")
    if has_element(root, "raw") and len(list(get_group(root, "raw"))) > 0:
        raise ConversionError("append does not extend raw (it is obs-aligned); drop raw first.")

    layer_keys = list(get_group(root, "layers")) if "layers" in root else []
    obsp_keys = list(get_group(root, "obsp")) if "obsp" in root else []
    obsm_keys = list(get_group(root, "obsm")) if "obsm" in root else []

    var_t: pd.DataFrame = read_elem(root["var"])
    var_s: pd.DataFrame = read_elem(src["var"])
    if len(var_t) != len(var_s) or not (var_t.index == var_s.index).all():
        raise ConversionError("var mismatch: names + order must be identical between stores.")

    obs_notes, obs_extra, obs_filled, obs_promoted = _check_obs_schema(get_group(root, "obs"), get_group(src, "obs"))

    x_t, x_s = get_group(root, "X"), get_group(src, "X")
    data_t, data_s = get_array(x_t, "data"), get_array(x_s, "data")
    if data_t.dtype != data_s.dtype:
        raise ConversionError(f"X dtype mismatch: {data_t.dtype} vs {data_s.dtype}.")

    n_t, _ = shape_attr(x_t)
    n_s, n_vars = shape_attr(x_s)

    cells_layer_keys = list(get_group(src, "layers")) if "layers" in src else []
    append_layers = [k for k in layer_keys if k in cells_layer_keys]
    drop_layers = [k for k in layer_keys if k not in cells_layer_keys]
    for k in append_layers:
        _check_layer(k, root[f"layers/{k}"], src[f"layers/{k}"], (n_s, n_vars))

    notes: list[str] = list(obs_notes)
    extras = []
    cells_only = [k for k in cells_layer_keys if k not in layer_keys]
    if cells_only:
        extras.append(f"layers {cells_only}")
    if has_element(src, "raw") and len(list(get_group(src, "raw"))) > 0:
        extras.append("raw")
    if "obsm" in src and list(get_group(src, "obsm")):
        extras.append(f"obsm {list(get_group(src, 'obsm'))}")
    if extras:
        notes.append("left behind (not carried from the cells store): " + ", ".join(extras))

    n_duplicate_names = _count_duplicate_names(get_group(root, "obs"), get_group(src, "obs"), n_t)
    if n_s > 0 and n_duplicate_names == n_s:
        raise ConversionError(
            f"append would add no new cells: all {n_s} appended cells are already present "
            "in the store (already appended?)."
        )

    return AppendPlan(
        n_new=n_s,
        drop_obsm=tuple(obsm_keys),
        drop_obsp=tuple(obsp_keys),
        drop_layers=tuple(drop_layers),
        append_layers=tuple(append_layers),
        n_duplicate_names=n_duplicate_names,
        notes=tuple(notes),
        obs_extra=obs_extra,
        obs_filled=obs_filled,
        obs_promoted=obs_promoted,
    )


def append(
    store: StoreLike,
    *,
    cells: StoreLike,
    drop_derived: bool = False,
    include_new_obs: bool = False,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
    commit: bool = True,
) -> OpResult:
    """Append the cells of another zarr store onto this one, in place.

    Extends X, obs, and every layer both stores carry. Obs columns are matched by name: one the
    cells lack gets missing values on the appended rows, one only the cells carry is left out
    unless ``include_new_obs``, and a plain int or bool store column is rewritten as nullable when
    it must hold a missing value. A shared layer is appended in the store layer's encoding, its
    cells values cast to the store's dtype when that loses nothing (CSR and dense layers grow in
    place; a CSC layer is rewritten, since every column grows in the middle). Derived obs-aligned
    elements the cells cannot extend (obsm embeddings, obsp graphs, layers only the store
    carries) are invalidated by new cells; :func:`plan_append` is called first, and if it would
    drop anything this raises unless ``drop_derived=True`` (re-create a dropped layer afterwards
    with :func:`~annizarr.ops._layer.add_layer`).

    Parameters
    ----------
    store
        Existing AnnData zarr store, or an Icechunk repo (path or :class:`~annizarr.ic.Repo`) to append onto, in place.
    cells
        Another AnnData zarr (or Icechunk) store whose cells are appended.
    drop_derived
        Consent to dropping derived obs-aligned elements the plan says would drop.
    include_new_obs
        Add obs columns only the cells carry, with missing values on the existing cells.
    cfg
        Resolved configuration; ``None`` uses the :class:`~annizarr.config.AppConfig` defaults.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet.
        Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the two stores involved.
    commit
        ``False`` stages the append without committing, so several ops land in one commit made
        with ``store.commit()``. Requires ``store`` to be a :class:`~annizarr.ic.Repo`.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        The plan (see :func:`plan_append`) would drop derived elements and
        ``drop_derived`` was not given, every appended cell is already present (see
        :func:`plan_append`), or the mutation fails partway through.
    """
    if cfg is None:
        cfg = AppConfig()

    plan = plan_append(store, cells=cells)
    drops = plan.drops()
    if drops and not drop_derived:
        raise ConversionError(
            "append will drop derived elements (invalidated by appended cells): "
            + ", ".join(drops)
            + ". Pass drop_derived=True to proceed; re-create a dropped layer with add-layer afterwards."
        )

    for note in plan.notes:
        logger.warning(note)

    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(cells)
    commit_message = message or f"annizarr append {store_name(cells)} → {store_name(store)}"
    root, finalize = open_store_rw(
        store,
        cfg,
        commit_message=commit_message,
        commit_metadata=op_metadata("append", cells=store_name(cells)),
        branch=branch,
        commit=commit,
    )

    x_t = get_group(root, "X")
    n_t, n_vars = shape_attr(x_t)
    n_s, _ = shape_attr(get_group(src, "X"))
    nnz = sum(int(np.asarray(get_array(x, "indptr")[-1:])[0]) for x in (x_t, get_group(src, "X")))

    try:
        hints = _drop_derived(root, plan)
        if hints:
            logger.warning("dropped " + ", ".join(hints) + " (invalidated by appended cells).")
        with stage(f"Appending X ({n_s} cells)"):
            _append_matrix(root, "X", src["X"], n_t, cfg)
        for k in plan.append_layers:
            with stage(f"Appending layers/{k} ({n_s} cells)"):
                _append_matrix(root, f"layers/{k}", src[f"layers/{k}"], n_t, cfg)
        with stage(f"Appending obs ({n_s} cells)"), write_settings(cfg.chunks.auto_shard):
            _append_obs(root["obs"], src["obs"], n_t, n_t + n_s, plan, include_new_obs)
        if plan.obs_extra:
            done = "added (missing on the existing cells)" if include_new_obs else "not appended"
            logger.warning(f"obs columns {list(plan.obs_extra)} {done}.")
    except AnzError:
        raise
    except Exception as e:
        raise ConversionError(
            f"append failed mid-mutation; {store_path(store)} may be inconsistent "
            f"(plain zarr cannot roll back — icechunk discards uncommitted changes): {e}"
        ) from e

    if plan.n_duplicate_names:
        logger.warning(
            f"appended cells introduce {plan.n_duplicate_names} duplicate obs name(s) "
            "(partial overlap with existing cells)."
        )
    logger.warning("appended cells break any sorted-store contiguity; re-run `annizarr sort` if the store was sorted.")
    verify_store_shape(root, n_obs=n_t + n_s, n_vars=n_vars, nnz=nnz, layers=plan.append_layers)
    snapshot_id = finalize()
    return OpResult(path=store_path(store), n_obs=n_t + n_s, n_vars=n_vars, snapshot_id=snapshot_id)


def _check_layer(key: str, node_t: Any, node_s: Any, cells_shape: tuple[int, int]) -> None:
    # a shared layer appends when the cells' values fit the store's dtype exactly; anything else refuses up front
    if matrix_shape(node_s) != cells_shape:
        raise ConversionError(f"layers/{key} in the cells is {matrix_shape(node_s)}, not cells x var {cells_shape}.")
    MatrixLayout.of(node_t)  # refuses a store element that is not a matrix
    dtype_t, dtype_s = _matrix_dtype(node_t), _matrix_dtype(node_s)
    if not _lossless(dtype_s, dtype_t):
        raise ConversionError(
            f"layers/{key}: the cells hold {dtype_s}, which does not fit the store's {dtype_t} without loss; "
            "cast one side before appending."
        )


def _lossless(src: np.dtype[Any], dst: np.dtype[Any]) -> bool:
    # numpy's "safe" casting allows int64 -> float64, which rounds above 2**53; every value must survive
    if not np.can_cast(src, dst, casting="safe"):
        return False
    if src.kind in "iu" and dst.kind == "f":
        return bool(np.iinfo(src).bits - (src.kind == "i") <= np.finfo(dst).nmant + 1)
    return True


def _matrix_dtype(node: Any) -> np.dtype[Any]:
    return np.dtype(node.dtype if isinstance(node, zarr.Array) else get_array(node, "data").dtype)


def _drop_derived(root: zarr.Group, plan: AppendPlan) -> list[str]:
    # deletes what the plan drops; returns one line per element group, with add-layer hints for copied layers
    lines: list[str] = []
    for group_key, keys in (("obsm", plan.drop_obsm), ("obsp", plan.drop_obsp), ("layers", plan.drop_layers)):
        if not keys:
            continue
        group = get_group(root, group_key)
        sources = {k: group[k].attrs.get(SOURCE_ATTR) for k in keys} if group_key == "layers" else {}
        for k in keys:
            del group[k]
        hints = [f"add-layer --layer {k} --matrix {m}" for k, m in sources.items() if isinstance(m, str)]
        lines.append(f"{group_key} {list(keys)}" + (f" (re-create with {'; '.join(hints)})" if hints else ""))
    return lines


def _append_matrix(root: zarr.Group, key: str, cells_node: Any, n_t: int, cfg: AppConfig) -> None:
    # the cells' element, read as rows in any encoding and cast to the store's dtype, appended in the store's encoding
    from annizarr._sources._readers import CastReader, as_reader

    node = root[key]
    tmp_dir = _local_tmp_dir(root)
    reader: Reader = as_reader(cells_node, cfg=cfg, tmp_dir=tmp_dir)
    try:
        dtype = _matrix_dtype(node)
        if reader.dtype != dtype:
            reader = CastReader(reader, dtype)
        layout = MatrixLayout.of(node).layout
        if layout == "csr":
            _append_csr(as_group(node), reader, n_t, cfg.chunks.cpus)
        elif layout == "dense":
            _append_dense(as_array(node), reader, n_t, cfg.chunks.cpus)
        else:
            _rewrite_appended(root, key, reader, tmp_dir, cfg)
    finally:
        reader.close()


def _append_csr(group: zarr.Group, reader: Reader, n_t: int, cpus: int) -> None:
    indptr_t = np.asarray(get_array(group, "indptr")[:], dtype=np.int64)
    nnz_t, nnz_s = int(indptr_t[-1]), reader.nnz
    _extend_flat(get_array(group, "data"), _FlatSource(reader.flat_data), nnz_t, nnz_s, cpus)
    _extend_flat(get_array(group, "indices"), _FlatSource(reader.flat_indices), nnz_t, nnz_s, cpus)
    _rewrite_indptr(group, indptr_t, reader.indptr, n_t + reader.shape[0])
    group.attrs["shape"] = [n_t + reader.shape[0], reader.shape[1]]


def _append_dense(arr: zarr.Array[Any], reader: Reader, n_t: int, cpus: int) -> None:
    n_s, n_vars = reader.shape
    arr.resize((n_t + n_s, n_vars))
    row_bytes = _layout.copy_bytes_per_item(arr.dtype.itemsize) * n_vars
    cuts = _append_cuts(n_t, n_s, unit=_layout.write_grid(arr)[0], bytes_per_item=row_bytes, cpus=cpus)
    run_parallel(_write_rows, [(reader, arr, r0, r1, n_t) for r0, r1 in cuts], cpus)


def _rewrite_appended(root: zarr.Group, key: str, cells: Reader, tmp_dir: str | None, cfg: AppConfig) -> None:
    # a CSC element cannot grow in place (every column gains entries mid-array): the store's rows and the
    # cells' rows are written whole to a temp store in the element's own layout, then copied into place
    import shutil
    import tempfile

    from annizarr._sources._readers import ConcatReader, as_reader

    node = root[key]
    layer_cfg = MatrixLayout.of(node).cfg(cfg)
    kept_attrs = {k: v for k, v in node.attrs.items() if k not in ("encoding-type", "encoding-version", "shape")}
    # the element keeps its own codecs; the temp store is only scaffolding
    codecs = {k: get_array(as_group(node), k).compressors for k in ("data", "indices")}
    logger.warning(f"{key} is {layer_cfg.io.layout}: rewriting it whole to append (a full pass over the element).")
    scratch = tempfile.mkdtemp(prefix="annizarr_rewrite_", dir=tmp_dir)
    try:
        tmp_root = zarr.open_group(scratch, mode="w")
        stored = as_reader(node, cfg=cfg, tmp_dir=scratch)
        try:
            write_matrix(tmp_root, "m", ConcatReader([stored, cells]), layer_cfg)
        finally:
            stored.close()
        parent_key, _, name = key.rpartition("/")
        parent = get_group(root, parent_key) if parent_key else root
        del parent[name]
        copy_matrix(tmp_root["m"], parent, name, cfg, rechunk=False, codecs=codecs)
        parent[name].attrs.update(kept_attrs)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _check_obs_schema(
    obs_t: zarr.Group, obs_s: zarr.Group
) -> tuple[list[str], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    # (notes, extra, filled, promoted): every obs decision append acts on is made here, once
    cols_t = str_list_attr(obs_t, "column-order")
    cols_s = str_list_attr(obs_s, "column-order")
    for g, label, cols in ((obs_t, "store", cols_t), (obs_s, "cells", cols_s)):
        stray = sorted(set(g) - set(cols) - {str_attr(g, "_index")})
        if stray:
            raise ConversionError(f"obs in {label} has elements outside column-order: {stray}.")

    extra = tuple(c for c in cols_s if c not in cols_t)
    filled = tuple(c for c in cols_t if c not in cols_s)
    promoted: list[str] = [c for c in filled if _fill_needs_promotion(obs_t[c], c)]
    notes: list[str] = []
    pairs = [(c, obs_t[c], obs_s[c]) for c in cols_t if c in cols_s]
    pairs.append(("<index>", obs_t[str_attr(obs_t, "_index")], obs_s[str_attr(obs_s, "_index")]))
    for name, t, s in pairs:
        enc, enc_s = t.attrs.get("encoding-type"), s.attrs.get("encoding-type")
        stringlike = enc in _STRINGLIKE_ENCODINGS and enc_s in _STRINGLIKE_ENCODINGS
        family = _family(t)
        if enc != enc_s and family is not None and family == _family(s):
            if _values(t).dtype != _values(s).dtype:
                raise ConversionError(f"obs column '{name}' dtype mismatch ({_values(t).dtype} vs {_values(s).dtype}).")
            if isinstance(t, zarr.Array) and bool(np.asarray(get_array(as_group(s), "mask")[:]).any()):
                promoted.append(name)
            continue
        if enc != enc_s and not stringlike:
            raise ConversionError(
                f"obs column '{name}' encoding mismatch ({enc!r} vs {enc_s!r}); reconcile before append."
            )
        if enc == "categorical":
            t_grp = as_group(t)
            cats_arr = get_array(t_grp, "categories")
            cat_t = np.asarray(cats_arr[:]).tolist()
            if bool(t_grp.attrs.get("ordered", False)):
                # ordered categories carry meaning in their order, so they must match exactly
                s_grp = as_group(s) if enc_s == "categorical" else None
                cat_s = np.asarray(get_array(s_grp, "categories")[:]).tolist() if s_grp is not None else None
                if s_grp is None or not bool(s_grp.attrs.get("ordered", False)) or cat_t != cat_s:
                    raise ConversionError(
                        f"obs column '{name}' categorical dtype mismatch "
                        "(an ordered categorical's categories, order, and ordered flag must be identical); "
                        "reconcile before append."
                    )
            else:
                new = _new_categories(cat_t, _stringlike_labels(s))
                if new:
                    codes_arr = get_array(t_grp, "codes")
                    want = _codes_dtype(cats_arr, codes_arr, len(cat_t) + len(new), new, name)
                    shown = ", ".join(map(str, new[:5])) + (", …" if len(new) > 5 else "")
                    plural = "y" if len(new) == 1 else "ies"
                    widen = f"; codes widen {codes_arr.dtype} → {want}" if want != codes_arr.dtype else ""
                    notes.append(f"obs column '{name}': {len(new)} new categor{plural} appended ({shown}){widen}.")
        elif enc == "string-array" and enc_s != "string-array":
            if any(label is None for label in _stringlike_labels(s)):
                raise ConversionError(
                    f"obs column '{name}': cells hold missing values; the store's plain string column cannot."
                )
        elif enc in _NULLABLE_ENCODINGS and enc_s == enc:
            t_grp, s_grp = as_group(t), as_group(s)
            t_values, s_values = get_array(t_grp, "values"), get_array(s_grp, "values")
            if t_values.dtype != s_values.dtype:
                raise ConversionError(f"obs column '{name}' dtype mismatch ({t_values.dtype} vs {s_values.dtype}).")
        elif enc == "array":
            t_arr, s_arr = as_array(t), as_array(s)
            if t_arr.dtype != s_arr.dtype:
                raise ConversionError(
                    f"obs column '{name}' dtype mismatch ({t_arr.dtype} vs {s_arr.dtype}); "
                    "obs columns extend in place, so dtypes must match exactly."
                )
        elif not stringlike:
            raise ConversionError(f"obs column '{name}': unsupported encoding {enc!r} for in-place append.")

    if extra:
        notes.append(f"obs columns in the cells, not in the store: {list(extra)}; include_new_obs adds them.")
    if filled:
        notes.append(f"obs columns in the store, not in the cells: {list(filled)}; appended cells get missing values.")
    if promoted:
        shown = ", ".join(f"'{c}' → {_family(obs_t[c])}" for c in promoted)
        notes.append(f"obs columns rewritten as nullable to hold missing values: {shown}.")
    return notes, extra, filled, tuple(promoted)


def _family(node: Any) -> str | None:
    # the nullable encoding a column shares a family with, None when it has none
    if isinstance(node, zarr.Array):
        return _NULLABLE_FAMILY.get(node.dtype.kind) if node.attrs.get("encoding-type") == "array" else None
    enc = node.attrs.get("encoding-type")
    return enc if enc in _NULLABLE_FAMILY.values() else None


def _values(node: Any) -> zarr.Array[Any]:
    return node if isinstance(node, zarr.Array) else get_array(as_group(node), "values")


def _fill_needs_promotion(node: Any, name: str) -> bool:
    # whether a store column the cells lack must be rewritten as nullable to hold their missing values
    enc = node.attrs.get("encoding-type")
    if enc == "string-array":
        raise ConversionError(
            f"obs column '{name}': the cells do not carry it, and a plain string-array holds no missing values. "
            "Add the column to the cells store. (A future release may instead rewrite the column as nullable.)"
        )
    if _family(node) is not None:
        return isinstance(node, zarr.Array)
    if enc in ("categorical", *_NULLABLE_ENCODINGS) or (enc == "array" and as_array(node).dtype.kind == "f"):
        return False
    raise ConversionError(f"obs column '{name}': the cells do not carry it, and {enc!r} holds no missing values.")


def _stringlike_labels(node: Any) -> np.ndarray[Any, Any]:
    # a string-like obs column as an object array of labels, None where the value is missing
    enc = node.attrs.get("encoding-type")
    if enc == "categorical":
        g = as_group(node)
        cats = np.array([*np.asarray(get_array(g, "categories")[:]).tolist(), None], dtype=object)
        return np.asarray(cats[np.asarray(get_array(g, "codes")[:])], dtype=object)  # -1 -> trailing None
    if enc == "nullable-string-array":
        g = as_group(node)
        values = np.asarray(np.asarray(get_array(g, "values")[:]).tolist(), dtype=object)
        values[np.asarray(get_array(g, "mask")[:], dtype=bool)] = None
        return values
    return np.asarray(np.asarray(as_array(node)[:]).tolist(), dtype=object)


def _new_categories(categories: list[Any], labels: np.ndarray[Any, Any]) -> list[Any]:
    import pandas as pd

    known = set(categories)
    return [label for label in pd.unique(labels) if label is not None and label not in known]


def _codes_dtype(cats_arr: Any, codes_arr: Any, n_categories: int, new: list[Any], name: str) -> np.dtype[Any]:
    # the stored code dtype, or the narrowest wider one that holds n_categories; -1 is missing, so signed
    if cats_arr.dtype.kind in "iufb":
        raise ConversionError(f"obs column '{name}': categories are {cats_arr.dtype}, cannot add {new[:5]}.")
    current: np.dtype[Any] = codes_arr.dtype
    if current.kind != "i":
        raise ConversionError(f"obs column '{name}': codes are {current}, expected signed integers.")
    if n_categories - 1 <= np.iinfo(current).max:
        return current
    return next(np.dtype(w) for w in _CODE_WIDTHS if np.iinfo(w).max >= n_categories - 1)


def _index_values(obs: zarr.Group) -> zarr.Array[Any]:
    # the 1-D array of obs names: the index element itself (string-array), or its values array
    # when anndata wrote a pandas>=3 string index as a nullable-string-array group
    node = obs[str_attr(obs, "_index")]
    return node if isinstance(node, zarr.Array) else get_array(as_group(node), "values")


def _count_duplicate_names(obs_t: zarr.Group, obs_s: zarr.Group, n_t: int) -> int:
    # counted once per row, not once per colliding pair
    idx_s = np.asarray(_index_values(obs_s)[:])

    order = np.argsort(idx_s, kind="stable")
    repeats_earlier = np.zeros(len(idx_s), dtype=bool)
    if len(idx_s) > 1:
        sorted_idx = idx_s[order]
        is_repeat_sorted = np.empty(len(idx_s), dtype=bool)
        is_repeat_sorted[0] = False
        is_repeat_sorted[1:] = sorted_idx[1:] == sorted_idx[:-1]
        repeats_earlier[order] = is_repeat_sorted

    import pandas as pd

    # pandas hashes the names; numpy's isin sorts object arrays and was 70x slower at 300k rows
    t_arr = _index_values(obs_t)
    chunk0 = t_arr.chunks[0]
    step = max(chunk0, (_INDEX_SCAN_ROWS // max(1, chunk0)) * chunk0)
    incoming = pd.Index(idx_s)
    in_target = np.zeros(len(idx_s), dtype=bool)
    for i0 in range(0, n_t, step):
        in_target |= incoming.isin(np.asarray(t_arr[i0 : min(i0 + step, n_t)]))
        if in_target.all():
            break
    return int((in_target | repeats_earlier).sum())


def _extend_flat(dst_a: Any, src_a: Any, off: int, n_src: int, cpus: int) -> None:
    dst_a.resize((off + n_src,))
    bytes_per_item = _layout.copy_bytes_per_item(dst_a.dtype.itemsize)
    cuts = _append_cuts(off, n_src, unit=_layout.write_grid(dst_a)[0], bytes_per_item=bytes_per_item, cpus=cpus)
    run_parallel(_copy_shifted, [(src_a, dst_a, s0, s1, off) for s0, s1 in cuts], cpus)


def _append_cuts(off: int, n_src: int, *, unit: int, bytes_per_item: int, cpus: int) -> list[tuple[int, int]]:
    # the part landing in the existing partial write unit first, then bands on the write grid (no read-modify-write)
    seam = min((-off) % unit, n_src)
    cuts: list[tuple[int, int]] = [(0, seam)] if seam else []
    cuts += [
        (seam + s0, seam + s1)
        for s0, s1 in _layout.plan_bands(n_src - seam, unit=unit, bytes_per_item=bytes_per_item, cpus=cpus)
    ]
    assert not n_src or (cuts[0][0] == 0 and cuts[-1][1] == n_src), "the cuts must cover the whole append"
    return cuts


def _rewrite_indptr(parent: Any, indptr_t: Any, indptr_s: Any, n_new: int) -> None:
    nnz_t = int(indptr_t[-1])
    nnz_new = nnz_t + int(indptr_s[-1])
    indptr_dtype = np.int64 if nnz_new > np.iinfo(np.int32).max else parent["indptr"].dtype
    del parent["indptr"]
    ip = parent.require_array("indptr", shape=(n_new + 1,), dtype=indptr_dtype, chunks=(n_new + 1,), overwrite=True)
    ip.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    ip[:] = np.concatenate([indptr_t, indptr_s[1:] + nnz_t]).astype(indptr_dtype)


class _FlatSource:
    # a reader's flat values or indices as a sliceable source for _extend_flat
    __slots__ = ("_read",)

    def __init__(self, read: Any) -> None:
        self._read = read

    def __getitem__(self, item: slice) -> Any:
        return self._read(item.start or 0, item.stop)


def _append_obs(obs_t: Any, obs_s: Any, n_t: int, n_new: int, plan: AppendPlan, include_new_obs: bool) -> None:
    for name in str_list_attr(obs_t, "column-order"):
        if name in plan.obs_promoted:
            incoming = None if name in plan.obs_filled else obs_s[name]
            write_frame_column(obs_t, name, _full_height(obs_t[name], incoming, n_t, n_new))
        elif name in plan.obs_filled:
            _fill_missing(obs_t[name], n_t, n_new)
        else:
            _extend_column(obs_t[name], obs_s[name], n_t, n_new, name)
    index_t = str_attr(obs_t, "_index")
    _extend_column(obs_t[index_t], obs_s[str_attr(obs_s, "_index")], n_t, n_new, index_t)
    if include_new_obs:
        for name in plan.obs_extra:
            write_frame_column(obs_t, name, _full_height(None, obs_s[name], n_t, n_new))


def _extend_column(t: Any, s: Any, n_t: int, n_new: int, name: str) -> None:
    enc, enc_s = t.attrs.get("encoding-type"), s.attrs.get("encoding-type")
    if enc == "categorical":
        _extend_categorical(t, _stringlike_labels(s), n_t, n_new, name)
    elif isinstance(t, zarr.Array):
        if isinstance(s, zarr.Array):
            _extend_1d(t, s[:], n_t, n_new)
        elif enc_s in _NULLABLE_FAMILY.values():  # no missing values: a column that has them was promoted
            _extend_1d(t, s["values"][:], n_t, n_new)
        else:  # string-array target fed from a categorical / nullable column (no missing values: checked in plan)
            _extend_1d(t, np.asarray(_stringlike_labels(s), dtype=str), n_t, n_new)
    elif enc == "nullable-string-array" and enc_s != enc:
        labels = _stringlike_labels(s)
        missing = np.fromiter((label is None for label in labels), dtype=bool, count=len(labels))
        _extend_1d(t["values"], np.asarray(np.where(missing, "", labels), dtype=str), n_t, n_new)
        _extend_1d(t["mask"], missing, n_t, n_new)
    elif isinstance(s, zarr.Array):  # nullable-integer/boolean store fed a plain column of its family
        _extend_1d(t["values"], s[:], n_t, n_new)
        _extend_1d(t["mask"], np.zeros(n_new - n_t, dtype=bool), n_t, n_new)
    else:  # nullable-*: values + mask
        _extend_1d(t["values"], s["values"][:], n_t, n_new)
        _extend_1d(t["mask"], s["mask"][:], n_t, n_new)


def _fill_missing(t: Any, n_t: int, n_new: int) -> None:
    # a store column the cells lack, in an encoding that holds missing values (checked in plan)
    n = n_new - n_t
    if t.attrs.get("encoding-type") == "categorical":
        _extend_1d(t["codes"], np.full(n, -1), n_t, n_new)
    elif isinstance(t, zarr.Array):
        _extend_1d(t, np.full(n, np.nan), n_t, n_new)
    else:
        values = t["values"]
        filler = np.zeros(n, dtype=values.dtype) if values.dtype.kind in "iufb" else np.full(n, "", dtype=str)
        _extend_1d(values, filler, n_t, n_new)
        _extend_1d(t["mask"], np.ones(n, dtype=bool), n_t, n_new)


def _full_height(t: Any, s: Any, n_t: int, n_new: int) -> pd.Series[Any]:
    # the store's values (if any) over the incoming ones (if any), missing wherever neither reaches
    import pandas as pd

    parts = [] if t is None else [nullable_series(read_elem(t))]
    if s is not None:
        parts.append(nullable_series(read_elem(s)).set_axis(range(n_t, n_new)))
    return pd.concat(parts).reindex(range(n_new))


def _extend_categorical(t: Any, labels: np.ndarray[Any, Any], n_t: int, n_new: int, name: str) -> None:
    # the store's categories stay as they are; unseen labels are appended as new categories and
    # the incoming values re-coded against the result, so codes already on disk never move
    import pandas as pd

    cats_arr, codes_arr = t["categories"], t["codes"]
    cats = np.asarray(cats_arr[:]).tolist()
    new = _new_categories(cats, labels)
    if new:
        want = _codes_dtype(cats_arr, codes_arr, len(cats) + len(new), new, name)
        cats_arr.resize((len(cats) + len(new),))
        cats_arr[len(cats) :] = np.asarray(new, dtype=str)
        if want != codes_arr.dtype:
            codes_arr = _widen_codes(t, codes_arr, want, n_t)
    codes = pd.Categorical(labels, categories=[*cats, *new]).codes  # -1 where the label is None
    _extend_1d(codes_arr, codes, n_t, n_new)


def _widen_codes(t: Any, codes_arr: Any, dtype: np.dtype[Any], n_t: int) -> Any:
    # zarr cannot change an array's dtype, so the codes are rewritten into a replacement at the same path
    # (the serializer is left automatic: a narrow array's carries no endianness, which a wide one needs)
    existing = codes_arr[:n_t]
    wide = t.create_array(
        "codes",
        shape=codes_arr.shape,
        dtype=dtype,
        chunks=codes_arr.chunks,
        shards=codes_arr.shards,
        fill_value=codes_arr.fill_value,
        compressors=codes_arr.compressors,
        attributes=dict(codes_arr.attrs),
        overwrite=True,
    )
    wide[:n_t] = existing.astype(dtype)
    return wide


def _extend_1d(dst: Any, vals: Any, n_t: int, n_new: int) -> None:
    vals = np.asarray(vals)
    if vals.dtype != dst.dtype and dst.dtype.kind in "iufb":  # e.g. categorical codes stored at different widths
        vals = vals.astype(dst.dtype)
    dst.resize((n_new,))
    dst[n_t:n_new] = vals


def _copy_shifted(src: Any, dst: Any, s0: int, s1: int, off: int) -> None:
    dst[off + s0 : off + s1] = src[s0:s1]


def _write_rows(reader: Reader, dst: Any, r0: int, r1: int, off: int) -> None:
    dst[off + r0 : off + r1, :] = reader.dense_block(r0, r1, 0, reader.shape[1])
