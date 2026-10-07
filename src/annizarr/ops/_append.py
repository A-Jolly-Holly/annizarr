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
    shape_attr,
    str_attr,
    str_list_attr,
)
from annizarr._storage import open_input_group, open_store_rw, store_name
from annizarr.errors import ConversionError
from annizarr.ops._expr import lognorm_band, target_sum_attr
from annizarr.ops._result import AppendPlan, OpResult

if TYPE_CHECKING:
    import pandas as pd

    from annizarr.typing import PathLike

logger = logging.getLogger(__name__)

_INDEX_SCAN_ROWS = 1 << 20
_NULLABLE_ENCODINGS = ("nullable-integer", "nullable-boolean", "nullable-string-array")
# mutually appendable: the store column keeps its encoding and the incoming values are converted
_STRINGLIKE_ENCODINGS = ("categorical", "string-array", "nullable-string-array")


def plan_append(store: PathLike, *, cells: PathLike) -> AppendPlan:
    """Validate an append and describe what it would do, without mutating anything.

    Reads metadata only (attrs, obs schema, small index/indptr arrays) from both stores.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store the cells would be appended onto.
    cells
        Another AnnData zarr store whose cells would be appended.

    Returns
    -------
    AppendPlan
        What appending would do: new-cell count, derived elements that would be
        dropped, layers eligible for in-place extension, and other notes.

    Raises
    ------
    ConversionError
        ``store``/``cells`` is not a CSR AnnData zarr store, ``var`` names/order
        mismatch, obs schema mismatch, an X dtype mismatch, or every appended cell's
        obs name is already present in the store (already appended?).
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

    obs_notes = _check_obs_schema(get_group(root, "obs"), get_group(src, "obs"))

    x_t, x_s = get_group(root, "X"), get_group(src, "X")
    data_t, data_s = get_array(x_t, "data"), get_array(x_s, "data")
    if data_t.dtype != data_s.dtype:
        raise ConversionError(f"X dtype mismatch: {data_t.dtype} vs {data_s.dtype}.")

    n_t, _ = shape_attr(x_t)
    n_s, _ = shape_attr(x_s)
    indptr_t = np.asarray(get_array(x_t, "indptr")[:], dtype=np.int64)

    ext_layers: list[str] = []
    bad_layers: list[str] = []
    if layer_keys:
        ext_layers, bad_layers = _extendable_layers(get_group(root, "layers"), layer_keys, indptr_t)
    drop_layers = [k for k in layer_keys if k not in ext_layers]

    notes: list[str] = list(obs_notes)
    if bad_layers:
        notes.append(f"layers {bad_layers}: sparsity differs from X, cannot extend.")
    extras = []
    if "layers" in src and list(get_group(src, "layers")):
        extras.append(f"layers {list(get_group(src, 'layers'))}")
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
        extendable_layers=tuple(ext_layers),
        n_duplicate_names=n_duplicate_names,
        notes=tuple(notes),
    )


def append(
    store: PathLike,
    *,
    cells: PathLike,
    drop_derived: bool = False,
    extend_layers: bool = False,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Append the cells of another zarr store onto this one, in place.

    Extends X and obs only. Derived obs-aligned elements on the store (obsm embeddings,
    obsp graphs, layers) are invalidated by new cells; :func:`plan_append` is called first,
    and if it would drop anything this raises unless ``drop_derived=True`` (re-derive
    layers afterwards with :func:`~annizarr.ops._expr.add_expr`). With ``extend_layers``,
    CSR layers created by add-expr (recorded target_sum, X's exact sparsity) are extended
    in place instead — the lognorm transform runs on the appended cells only.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store to append onto, in place.
    cells
        Another AnnData zarr store whose cells are appended.
    drop_derived
        Consent to dropping derived obs-aligned elements the plan says would drop.
    extend_layers
        Extend eligible add-expr CSR layers in place instead of dropping them.
    cfg
        Resolved configuration; ``None`` uses the :class:`~annizarr.config.AppConfig` defaults.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet.
        Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the two stores involved.

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
    drop_layers = list(plan.drop_layers)
    ext_layers = list(plan.extendable_layers) if extend_layers else []
    if not extend_layers:
        drop_layers += list(plan.extendable_layers)

    drops = plan.drops(extend_layers=extend_layers)
    if drops and not drop_derived:
        raise ConversionError(
            "append will drop derived elements (invalidated by appended cells): "
            + ", ".join(drops)
            + ". Pass drop_derived=True to proceed; re-derive layers with add-expr afterwards."
        )

    for note in plan.notes:
        logger.warning(note)

    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(cells)
    commit_message = message or f"annizarr append {store_name(cells)} → {store_name(store)}"
    root, finalize = open_store_rw(store, cfg, commit_message=commit_message, branch=branch)

    x_t, x_s = get_group(root, "X"), get_group(src, "X")
    n_t, n_vars = shape_attr(x_t)
    n_s, _ = shape_attr(x_s)
    indptr_t = np.asarray(get_array(x_t, "indptr")[:], dtype=np.int64)
    indptr_s = np.asarray(get_array(x_s, "indptr")[:], dtype=np.int64)

    obsm_keys, obsp_keys = list(plan.drop_obsm), list(plan.drop_obsp)
    try:
        for group_key, keys in (("obsm", obsm_keys), ("obsp", obsp_keys), ("layers", drop_layers)):
            for k in keys:
                del get_group(root, group_key)[k]
        drop_groups = (("obsm", obsm_keys), ("obsp", obsp_keys), ("layers", drop_layers))
        dropped = [f"{g} {list(k)}" for g, k in drop_groups if k]
        if dropped:
            hint = "; re-derive layers with add-expr" if drop_layers else ""
            logger.warning("dropped " + ", ".join(dropped) + f" (invalidated by appended cells{hint}).")
        _append_arrays(root, src, x_t, x_s, indptr_t, indptr_s, n_t, n_s, n_vars, cfg)
        if ext_layers:
            _extend_lognorm_layers(root, x_s, ext_layers, indptr_t, indptr_s, n_t + n_s, n_vars, cfg)
            logger.warning(
                f"extended layers {ext_layers} in place (lognorm applied to the appended "
                "cells at each layer's recorded target_sum)."
            )
    except ConversionError:
        raise
    except Exception as e:
        raise ConversionError(
            f"append failed mid-mutation; {store} may be inconsistent "
            f"(plain zarr cannot roll back — icechunk discards uncommitted changes): {e}"
        ) from e

    if plan.n_duplicate_names:
        logger.warning(
            f"appended cells introduce {plan.n_duplicate_names} duplicate obs name(s) "
            "(partial overlap with existing cells)."
        )
    logger.warning("appended cells break any sorted-store contiguity; re-run `annizarr sort` if the store was sorted.")
    snapshot_id = finalize()
    return OpResult(path=str(store), n_obs=n_t + n_s, n_vars=n_vars, snapshot_id=snapshot_id)


def _check_obs_schema(obs_t: zarr.Group, obs_s: zarr.Group) -> list[str]:
    cols_t = str_list_attr(obs_t, "column-order")
    cols_s = str_list_attr(obs_s, "column-order")
    if cols_t != cols_s:
        raise ConversionError(f"obs schema mismatch: store {cols_t} vs cells {cols_s}.")
    for g, label, cols in ((obs_t, "store", cols_t), (obs_s, "cells", cols_s)):
        stray = sorted(set(g) - set(cols) - {str_attr(g, "_index")})
        if stray:
            raise ConversionError(f"obs in {label} has elements outside column-order: {stray}.")

    notes: list[str] = []
    pairs = [(c, obs_t[c], obs_s[c]) for c in cols_t]
    pairs.append(("<index>", obs_t[str_attr(obs_t, "_index")], obs_s[str_attr(obs_s, "_index")]))
    for name, t, s in pairs:
        enc, enc_s = t.attrs.get("encoding-type"), s.attrs.get("encoding-type")
        stringlike = enc in _STRINGLIKE_ENCODINGS and enc_s in _STRINGLIKE_ENCODINGS
        if enc != enc_s and not stringlike:
            raise ConversionError(
                f"obs column '{name}' encoding mismatch ({enc!r} vs {enc_s!r}); reconcile before append."
            )
        if enc == "categorical":
            t_grp = as_group(t)
            cat_t = np.asarray(get_array(t_grp, "categories")[:]).tolist()
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
                    _check_category_room(t_grp, len(cat_t) + len(new), new, name)
                    shown = ", ".join(map(str, new[:5])) + (", …" if len(new) > 5 else "")
                    plural = "y" if len(new) == 1 else "ies"
                    notes.append(f"obs column '{name}': {len(new)} new categor{plural} appended ({shown}).")
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
    return notes


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


def _check_category_room(t_grp: zarr.Group, n_categories: int, new: list[Any], name: str) -> None:
    cats_arr, codes_arr = get_array(t_grp, "categories"), get_array(t_grp, "codes")
    if cats_arr.dtype.kind in "iufb":
        raise ConversionError(f"obs column '{name}': categories are {cats_arr.dtype}, cannot add {new[:5]}.")
    if n_categories - 1 > np.iinfo(codes_arr.dtype).max:
        raise ConversionError(
            f"obs column '{name}': {n_categories} categories exceed its {codes_arr.dtype} code width; "
            "rewrite the store with wider codes before appending."
        )


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

    t_arr = _index_values(obs_t)
    chunk0 = t_arr.chunks[0]
    step = max(chunk0, (_INDEX_SCAN_ROWS // max(1, chunk0)) * chunk0)
    in_target = np.zeros(len(idx_s), dtype=bool)
    for i0 in range(0, n_t, step):
        in_target |= np.isin(idx_s, np.asarray(t_arr[i0 : min(i0 + step, n_t)]))
    return int((in_target | repeats_earlier).sum())


def _append_arrays(
    root: Any,
    src: Any,
    x_t: Any,
    x_s: Any,
    indptr_t: Any,
    indptr_s: Any,
    n_t: int,
    n_s: int,
    n_vars: int,
    cfg: AppConfig,
) -> None:
    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    n_new = n_t + n_s

    with stage(f"Appending X ({n_s} cells, nnz={nnz_s})"):
        for name in ("data", "indices"):
            _extend_flat(x_t[name], x_s[name], nnz_t, nnz_s, cfg.chunks.cpus)
        _rewrite_indptr(x_t, indptr_t, indptr_s, n_new)
        x_t.attrs["shape"] = [n_new, n_vars]

    with stage(f"Appending obs ({n_s} cells)"):
        _append_obs(root["obs"], src["obs"], n_t, n_new)


def _extend_flat(dst_a: Any, src_a: Any, off: int, n_src: int, cpus: int) -> None:
    # after the seam, cuts land on dst's write-grid multiples (no read-modify-write).
    dst_a.resize((off + n_src,))
    grid0 = _layout.write_grid(dst_a)[0]
    step = max(grid0, (_layout.BATCH_BYTES // max(1, grid0 * dst_a.dtype.itemsize)) * grid0)
    cuts = [0]
    seam = (-off) % grid0
    if 0 < seam < n_src:
        cuts.append(seam)
    while cuts[-1] < n_src:
        cuts.append(min(n_src, cuts[-1] + step))
    jobs = [(src_a, dst_a, cuts[i], cuts[i + 1], off) for i in range(len(cuts) - 1)]
    run_parallel(_copy_shifted, jobs, cpus)


def _rewrite_indptr(parent: Any, indptr_t: Any, indptr_s: Any, n_new: int) -> None:
    nnz_t = int(indptr_t[-1])
    nnz_new = nnz_t + int(indptr_s[-1])
    indptr_dtype = np.int64 if nnz_new > np.iinfo(np.int32).max else parent["indptr"].dtype
    del parent["indptr"]
    ip = parent.require_array("indptr", shape=(n_new + 1,), dtype=indptr_dtype, chunks=(n_new + 1,), overwrite=True)
    ip.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    ip[:] = np.concatenate([indptr_t, indptr_s[1:] + nnz_t]).astype(indptr_dtype)


def _extendable_layers(layers: Any, keys: list[str], indptr_t: Any) -> tuple[list[str], list[str]]:
    # a sparsity mismatch goes to `bad` instead — extending it in place would corrupt it.
    ext, bad = [], []
    for k in keys:
        node = layers[k]
        if isinstance(node, zarr.Array) or node.attrs.get("encoding-type") != "csr_matrix":
            continue
        if target_sum_attr(node.attrs) is None:
            continue
        lp = np.asarray(node["indptr"][:], dtype=np.int64)
        if len(lp) != len(indptr_t) or not (lp == indptr_t).all():
            bad.append(k)
            continue
        ext.append(k)
    return ext, bad


def _extend_lognorm_layers(
    root: Any, x_s: Any, keys: list[str], indptr_t: Any, indptr_s: Any, n_new: int, n_vars: int, cfg: AppConfig
) -> None:
    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    row_nnz_s = np.diff(indptr_s)
    n_s = len(row_nnz_s)
    row_step = max(1_000, min(200_000, _layout.BATCH_BYTES // (max(1, nnz_s // max(1, n_s)) * 12)))
    for k in keys:
        g = root["layers"][k]
        target_sum = target_sum_attr(g.attrs)  # eligibility already checked this is present
        assert target_sum is not None
        with stage(f"Extending layers/{k} ({n_s} cells, nnz={nnz_s})"):
            _extend_flat(g["indices"], x_s["indices"], nnz_t, nnz_s, cfg.chunks.cpus)
            data = g["data"]
            data.resize((nnz_t + nnz_s,))
            for b0 in range(0, n_s, row_step):
                b1 = min(b0 + row_step, n_s)
                s0, s1, vals = lognorm_band(x_s["data"], indptr_s, row_nnz_s, target_sum, b0, b1)
                data[nnz_t + s0 : nnz_t + s1] = vals
            _rewrite_indptr(g, indptr_t, indptr_s, n_new)
            g.attrs["shape"] = [n_new, n_vars]


def _append_obs(obs_t: Any, obs_s: Any, n_t: int, n_new: int) -> None:
    pairs = [(c, c) for c in obs_t.attrs["column-order"]]
    pairs.append((obs_t.attrs["_index"], obs_s.attrs["_index"]))
    for name_t, name_s in pairs:
        t, s = obs_t[name_t], obs_s[name_s]
        enc, enc_s = t.attrs.get("encoding-type"), s.attrs.get("encoding-type")
        if enc == "categorical":
            _extend_categorical(t, _stringlike_labels(s), n_t, n_new, name_t)
        elif isinstance(t, zarr.Array):
            if isinstance(s, zarr.Array):
                _extend_1d(t, s[:], n_t, n_new)
            else:  # string-array target fed from a categorical / nullable column (no missing values: checked in plan)
                _extend_1d(t, np.asarray(_stringlike_labels(s), dtype=str), n_t, n_new)
        elif enc == "nullable-string-array" and enc_s != enc:
            labels = _stringlike_labels(s)
            missing = np.fromiter((label is None for label in labels), dtype=bool, count=len(labels))
            _extend_1d(t["values"], np.asarray(np.where(missing, "", labels), dtype=str), n_t, n_new)
            _extend_1d(t["mask"], missing, n_t, n_new)
        else:  # nullable-*: values + mask
            _extend_1d(t["values"], s["values"][:], n_t, n_new)
            _extend_1d(t["mask"], s["mask"][:], n_t, n_new)


def _extend_categorical(t: Any, labels: np.ndarray[Any, Any], n_t: int, n_new: int, name: str) -> None:
    # the store's categories stay as they are; unseen labels are appended as new categories and
    # the incoming values re-coded against the result, so codes already on disk never move
    import pandas as pd

    cats_arr, codes_arr = t["categories"], t["codes"]
    cats = np.asarray(cats_arr[:]).tolist()
    new = _new_categories(cats, labels)
    if new:
        _check_category_room(t, len(cats) + len(new), new, name)
        cats_arr.resize((len(cats) + len(new),))
        cats_arr[len(cats) :] = np.asarray(new, dtype=str)
    codes = pd.Categorical(labels, categories=[*cats, *new]).codes  # -1 where the label is None
    _extend_1d(codes_arr, codes, n_t, n_new)


def _extend_1d(dst: Any, vals: Any, n_t: int, n_new: int) -> None:
    vals = np.asarray(vals)
    if vals.dtype != dst.dtype and dst.dtype.kind in "iufb":  # e.g. categorical codes stored at different widths
        vals = vals.astype(dst.dtype)
    dst.resize((n_new,))
    dst[n_t:n_new] = vals


def _copy_shifted(src: Any, dst: Any, s0: int, s1: int, off: int) -> None:
    dst[off + s0 : off + s1] = src[s0:s1]
