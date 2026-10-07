# What annizarr writes

## anndata-readable zarr v3

Every store is a zarr v3 group that `anndata.read_zarr` (anndata 0.13) opens directly. annizarr
writes the `encoding-type` / `encoding-version` attrs itself to match anndata's on-disk spec:

| Element | `encoding-type` | `encoding-version` |
|---|---|---|
| root group | `anndata` | 0.1.0 |
| dense X and layers | `array` | 0.2.0 |
| sparse X and layers | `csr_matrix` or `csc_matrix` | 0.1.0 |
| raw | `raw` | 0.1.0 |

obs, var, uns, obsm, varm, obsp and varp go through anndata's own `write_elem`, so anything anndata
can write round-trips, spatial coordinates and images included. Before writing, obs and var get the
same treatment anndata's own writers apply: string columns with repeated values become categoricals
(unique-per-row strings such as barcodes stay string arrays), and the index is always a plain
`string-array`, even on pandas 3 where anndata would otherwise emit a nullable-string group.

## Inputs

h5ad or 10x HDF5 (sniffed from the file contents; `--from` overrides), and for the edit commands an
existing annizarr/anndata zarr or Icechunk store. X and layers may be dense, CSR or CSC on input, and
a multi-input `convert` may mix file kinds and layouts. The only requirements are at least one cell,
at least one gene, and an X.

## Layout and chunking

- X defaults to CSR (`--layout csr`); layers follow X's layout on `convert`. `add-expr` layers
  default to CSC, for column reads.
- Dense: chunks of `row_chunk` by `col_chunk` (2048 by 2048 by default), optionally packed into
  shards of `shard_factor` chunks per axis (Python-only knob).
- Sparse: `data` and `indices` are 1-D arrays chunked by `nnz_chunk` nonzeros (9,000,000 by
  default). Setting the major-axis chunk (`--row-chunk` for CSR, `--col-chunk` for CSC) sizes them
  to about that many cells or genes per chunk instead, using the average nonzeros per row or
  column. The other axis's chunk setting is ignored for sparse output (a usage error on `convert`
  and `add-expr`, where the layout is known up front; a warning on `rechunk`).
- `--auto-shard` shards the 1-D sparse arrays and the anndata-written elements with zarr's
  automatic shard shape, which cuts object count on remote stores.
- Every matrix is compressed with zstd (level 5) plus byte shuffle.

## Streaming and parallelism

Inputs stream band by band by default (`--eager` loads the whole input first), so peak memory is
set by the band budget, not by store size. Band workers default to all cores (`--cpus`): threads for
in-memory or zarr-backed readers, processes for lazy h5py-backed ones, and a read-ahead pipeline
when a lazy reader feeds an Icechunk session.

## Atomic writes

`convert`, `rechunk` and `sort` write into a sibling `OUT.tmp-<id>` directory, verify the new store
opens with the expected shape, then rename it onto the target. A killed run never leaves a partial
store at the destination, and an existing target is only replaced with `--overwrite`.

## Icechunk

`--ic` (or `IOConfig(backend="icechunk")`) writes through an Icechunk repository. Every op is exactly
one commit on `--branch` (default `main`, created off main if missing) with `-m` as the message, or
an auto-generated one naming the op. `append` and `add-expr` work on an existing repo without `--ic`.
Remote targets (`s3://`, `gs://`) require Icechunk; credentials come from the environment.

## Re-running an op

- `add-expr` errors if the layer already exists unless `--overwrite`.
- `append` errors if every appended cell is already present ("already appended?") and only warns
  on a partial overlap, since barcodes legitimately collide across samples. It never drops
  obsm/obsp/layers without consent: it prompts on a terminal, or needs `--drop-derived` in a
  script. `--extend-layers` extends add-expr CSR layers in place instead of dropping them. Obs
  columns keep the store's encoding: a categorical column gains new categories for values it has
  not seen and re-codes incoming strings or differently coded categoricals against them (ordered
  categoricals must match exactly); a plain string column accepts categorical cells as strings.
- `sort` re-derives a lone `layers/gexp` on the sorted output with the layer's own layout and
  chunking.
