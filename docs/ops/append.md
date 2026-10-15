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
- `--cpus N`, `--branch`, `-m`: see the [CLI reference](../cli.md#append). There is no chunk option;
  the appended cells continue the store's chunk grid.

## Behaviour

- Both stores need CSR X of the same dtype, identical var (gene names and order) and the same obs
  columns. A non-empty raw in the target is refused; drop it first.
- Obs columns keep the store's encoding. A categorical column gains new categories for values it has
  not seen and re-codes the incoming cells against them; an ordered categorical must match exactly. A
  plain string column accepts categorical cells as strings.
- A categorical whose categories outgrow the code width it was written with (128 for `int8` codes,
  32768 for `int16`) has its codes array rewritten at the next signed width. `plan_append` reports
  the widening before anything is touched, and the rewrite is part of the same in-place edit.
- Duplicate cells: `append` errors when every incoming cell is already present and warns on a partial
  overlap, since barcodes legitimately collide across samples.
- A sorted store loses its contiguity. `append` warns about it, and `sort` restores it.
- On plain zarr the edit is in place and cannot be rolled back if the run dies midway. On Icechunk it
  is one commit, and a failed run leaves the repository as it was.
- `plan_append` returns an `AppendPlan(n_new, drop_obsm, drop_obsp, drop_layers, extendable_layers,
  n_duplicate_names, notes)`; `plan.drops(extend_layers)` lists the element paths that would go.
