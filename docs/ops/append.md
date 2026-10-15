# append

Append the cells of one converted store onto another, in place. Both are annizarr (or anndata) zarr
stores or Icechunk repositories: convert the new cells first, then append. X grows along the cell axis
on the store's existing chunk grid, obs grows with it, and nothing else is rewritten.

## Command line

```bash
annizarr convert batch2.h5ad -o batch2.zarr
annizarr append atlas.zarr batch2.zarr                    # prompts before dropping obsm/obsp/layers
annizarr append atlas.zarr batch2.zarr --drop-derived     # consent up front, for scripts
annizarr append atlas.zarr batch2.zarr --extend-layers    # extend add-expr CSR layers instead of dropping them
annizarr append atlas.zarr batch2.zarr --include-new-obs  # add obs columns only batch2 carries
annizarr append repo.icechunk batch2.zarr --branch dev -m "batch 2"
```

## Python

```python
import annizarr as az

plan = az.plan_append("atlas.zarr", cells="batch2.zarr")  # metadata only, nothing is written
plan.n_new, plan.drops(extend_layers=True), plan.notes
az.append("atlas.zarr", cells="batch2.zarr", drop_derived=True, extend_layers=True)
```

## Options

- `--drop-derived`: consent to dropping the obs-aligned elements the appended cells invalidate: obsm,
  obsp, and layers that cannot be extended. Without it `append` prompts on a terminal and errors in a
  script.
- `--extend-layers`: extend CSR layers written by `add-expr` in place, applying each layer's recorded
  normalization to the new cells, instead of dropping them.
- `--include-new-obs`: add obs columns that only the cells carry, with missing values on the existing
  cells, instead of leaving them out.
- `--cpus N`, `--branch`, `-m`: see the [CLI reference](../cli.md#append). There is no chunk option;
  the appended cells continue the store's chunk grid.

## Behaviour

- Both stores need CSR X of the same dtype and identical var (gene names and order). A non-empty raw
  in the target is refused; drop it first.
- Obs columns are matched by name, and the store's schema wins:

  | Column | What happens |
  |---|---|
  | in the store and the cells | appended (encodings below) |
  | in the store, not the cells | the appended rows are missing, with a warning |
  | in the cells, not the store | left out with a warning; `--include-new-obs` adds it, missing on the existing cells, at the end of `column-order` |

  A missing value is written in the store column's own encoding: code `-1` in a categorical, the
  mask in a nullable column, `NaN` in a float array. A plain int or bool array has no way to hold
  one, so it is rewritten as `nullable-integer` or `nullable-boolean` first, and reads back as
  `Int32`/`boolean` rather than `int32`/`bool`. A plain `string-array` is never filled: a column is
  stored that way only when every value is unique, so it is an identifier the cells must carry, and
  the plan refuses with an error.
- Shared obs columns keep the store's encoding. A categorical column gains new categories for values
  it has not seen and re-codes the incoming cells against them; an ordered categorical must match
  exactly. A plain string column accepts categorical cells as strings. A plain int or bool column and
  its nullable encoding append onto each other; when the incoming nullable values have a missing
  value, the store column is rewritten as nullable first. Dtypes must otherwise match exactly
  (`int32` onto `int64` errors), and other encodings across families are refused.
- Rewriting a column as nullable, or adding one with `--include-new-obs`, holds that one column in
  memory at full height (one value per cell) and rewrites it once.
- A categorical whose categories outgrow the code width it was written with (128 for `int8` codes,
  32768 for `int16`) has its codes array rewritten at the next signed width. `plan_append` reports
  the widening before anything is touched, and the rewrite is part of the same in-place edit.
- Duplicate cells: `append` errors when every incoming cell is already present and warns on a partial
  overlap, since barcodes legitimately collide across samples.
- A sorted store loses its contiguity. `append` warns about it, and `sort` restores it.
- Before committing, `append` checks the store it wrote: X's offsets, nonzero count and shape, the
  height of the index and every obs column, and any extended layer's offsets against X's. A mismatch
  raises `StorageError`, and on Icechunk nothing is committed. The check proves the store is shaped
  right, not that the values are the ones you appended.
- On plain zarr the edit is in place and cannot be rolled back if the run dies midway. On Icechunk it
  is one commit, and a failed run leaves the repository as it was. With a `Repo` and `commit=False`
  the append is staged for a commit you make yourself; see
  [batching ops into one commit](../icechunk.md#batching-ops-into-one-commit).
- `plan_append` returns an `AppendPlan(n_new, drop_obsm, drop_obsp, drop_layers, extendable_layers,
  n_duplicate_names, notes, obs_extra, obs_filled, obs_promoted)`; `plan.drops(extend_layers)` lists
  the element paths that would go. `obs_extra` are the cells-only columns, `obs_filled` the store
  columns the cells lack, and `obs_promoted` the store columns that would be rewritten as nullable.
