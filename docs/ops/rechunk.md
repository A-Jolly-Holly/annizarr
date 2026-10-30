# rechunk

Rewrite one matrix of an existing store with new chunking and stream-copy everything else unchanged
into a new store. The layout stays what it is; only the chunk grid, and the shard grid if any,
changes. Use it when a store written for one access pattern needs larger or smaller chunks, or when a
remote store has too many small objects.

## Command line

```bash
annizarr rechunk sample.zarr -o rechunked.zarr --row-chunk 2000
annizarr rechunk sample.zarr -o rechunked.zarr --matrix layers/counts_csc --col-chunk 500
annizarr rechunk sample.zarr -o rechunked.zarr --auto-shard
```

## Python

```python
import annizarr as az
from annizarr.config import AppConfig, ChunkConfig

az.rechunk("sample.zarr", output="rechunked.zarr", cfg=AppConfig(chunks=ChunkConfig(row_chunk=2000)))
az.rechunk(
    "sample.zarr",
    output="rechunked.zarr",
    matrix="layers/counts_csc",
    cfg=AppConfig(chunks=ChunkConfig(col_chunk=500)),
)
```

## Options

- `--matrix X|layers/<name>|raw/X`: which matrix to rewrite. Default `X`. Everything else is copied
  as it is.
- `--row-chunk N`, `--col-chunk N`: the new chunk shape, with the same meaning as on `convert`: exact
  for dense, about N cells (CSR) or N genes (CSC) per chunk. The axis that does not apply to a sparse
  layout is ignored with a warning.
- `--clevel N`: re-encode the rewritten matrix at this compression level, 0 to 9. Default: keep the
  source's level. Copied matrices always keep theirs.
- `--auto-shard`: shard the 1-D sparse arrays and the anndata-written elements with zarr's automatic
  shard shape, which cuts the object count on remote stores.
- `--cpus N`, `--overwrite`, `--consolidate-metadata`, `--ic`, `--branch`, `-m`: see the
  [CLI reference](../cli.md#rechunk).

## Behaviour

- A new store, written the way `convert` writes: into a sibling temp directory, verified, then renamed
  onto the target. The source is never modified.
- The matrix is read band by band and written on the new grid in parallel, so memory is bounded by the
  band budget, not by the matrix.
- Layout and dtype are unchanged. To change layout, convert again from the original file with
  `--layout`.
