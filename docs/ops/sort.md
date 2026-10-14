# sort

Physically order the rows of a store by one or more obs columns, primary key first, into a new store.
Cells that share a label become contiguous on disk, so reading one cell type or one sample touches
only its own chunks. `convert --sort-by` does the same while converting.

## Command line

```bash
annizarr sort sample.zarr -o sorted.zarr --by cell_type
annizarr sort sample.zarr -o sorted.zarr --by donor cell_type
annizarr convert sample.h5ad -o sorted.zarr --sort-by cell_type   # sort while converting
```

## Python

```python
import annizarr as az

az.sort("sample.zarr", output="sorted.zarr", by=["donor", "cell_type"])
```

## Options

- `--by COL [COL ...]`: obs columns to sort by, primary key first. Required.
- `--cpus N`, `--auto-shard`, `--overwrite`, `--consolidate-metadata`, `--ic`, `--branch`, `-m`: see
  the [CLI reference](../cli.md#sort).

## Behaviour

- Requires CSR X and writes CSR. obs and obsm are reordered with X; var, uns, varm and varp are copied
  as they are.
- A lone `layers/gexp` written by `add-expr` is re-derived on the sorted output with the layer's own
  layout, chunking and target sum. Any other layer, and a non-empty raw or obsp, is refused: drop it,
  or sort at convert time with `--eager`.
- Streams X through per-group temp files under the output, so memory stays bounded. The output has to
  be local for that reason.
- A new store, written the way `convert` writes: sibling temp directory, verify, rename. The source is
  never modified.
- `append` breaks the ordering; run `sort` again afterwards.
