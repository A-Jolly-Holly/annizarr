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
- `--clevel N`: compression level of every matrix the sort writes, 0 to 9. Default 3.
- `--cpus N`, `--auto-shard`, `--overwrite`, `--consolidate-metadata`, `--ic`, `--branch`, `-m`: see
  the [CLI reference](../cli.md#sort).

## Behaviour

- Requires CSR X and writes CSR. Everything aligned to the cells moves with the rows: obs, obsm,
  every layer, `raw/X`, and every obsp graph. var, uns, varm, varp and `raw/var` are copied as they
  are.
- Each layer, `raw/X` and obsp graph keeps its own encoding (CSR, CSC or dense) and chunking.
- An obsp graph is cells by cells, so both its axes are reordered: a neighbour graph still links the
  same cells afterwards, and `uns["neighbors"]` still names it.
- `convert --sort-by` carries layers, raw and obsp the same way.
- Streams each matrix through per-group temp files under the output, one matrix at a time, so memory
  stays bounded. A CSC layer passes through two on-disk transposes. The output has to be local for
  that reason.
- A new store, written the way `convert` writes: sibling temp directory, verify, rename. The source is
  never modified.
- `append` breaks the ordering; run `sort` again afterwards.
