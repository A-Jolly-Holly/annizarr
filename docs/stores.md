# What annizarr writes

## anndata-readable zarr v3

Every store is a zarr v3 group that `anndata.read_zarr` (anndata 0.12.7 or newer) opens directly. annizarr
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
(unique-per-row strings such as barcodes stay string arrays), and the index and every column that
stays a string are coerced to pandas 3's `str` dtype, so they are written as `nullable-string-array`
whichever pandas or anndata is installed -- byte for byte what `AnnData.write_zarr` produces on pandas
3 (anndata 0.12 omits the `na-value` attribute that makes them read back as `str`; annizarr fills it
in, and `rechunk` adds it to stores that lack it). Stores written by earlier versions, whose index is a
plain `string-array`, are still read and appended to.

## Inputs

h5ad or 10x HDF5 (sniffed from the file contents; `--from` overrides), and for the edit commands an
existing annizarr/anndata zarr or Icechunk store. X and layers may be dense, CSR or CSC on input, and
a multi-input `convert` may mix file kinds and layouts. The only requirements are at least one cell,
at least one gene, and an X.

## Layout and chunking

- X defaults to CSR (`--layout csr`); layers follow X's layout on `convert`. `add-layer` copies
  default to CSC, for column reads.
- Dense: chunks of `row_chunk` by `col_chunk` (2048 by 2048 by default), optionally packed into
  shards of `shard_factor` chunks per axis (Python-only knob).
- Sparse: `data` and `indices` are 1-D arrays chunked by `nnz_chunk` nonzeros (9,000,000 by
  default). Setting the major-axis chunk (`--row-chunk` for CSR, `--col-chunk` for CSC) sizes them
  to about that many cells or genes per chunk instead, using the average nonzeros per row or
  column. The other axis's chunk setting is ignored for sparse output (a usage error on `convert`
  and `add-layer`, where the layout is known up front; a warning on `rechunk`).
- `--auto-shard` shards the 1-D sparse arrays and the anndata-written elements with zarr's
  automatic shard shape, which cuts object count on remote stores.
- Every matrix is compressed with Blosc zstd plus byte shuffle, at level 3 by default (`--clevel`,
  0 to 9, on `convert`, `sort` and `add-layer`). Level 3 writes 2 to 3.6 times faster than level 5
  for 1 to 7% more bytes on real counts. `rechunk` keeps the source's level unless `--clevel` is
  given, and `append` continues each array with the codec it already has.

## Streaming and parallelism

Inputs stream band by band by default (`--eager` loads the whole input first), so peak memory is
set by the band budget, not by store size. A band is a whole number of write units (the chunk, or
the shard when sharded) sized so one worker's working set stays within 512 MiB; there are as many
bands as that needs and never fewer than there are workers. Every command streams this way,
including CSC output and `add-layer` (a transpose through per-band temp files under the output),
`sort` (per-group temp files), and 10x input, whose matrix is already cells-by-genes CSR on disk.

Band workers default to every core the process may use (`--cpus`): its CPU affinity, capped by a
container's CPU limit (`docker --cpus`, Kubernetes, a SLURM cpuset), not the host's core count. They
are threads for in-memory or zarr-backed readers, processes for lazy h5py-backed ones, and a
read-ahead pipeline in which worker processes read and this process writes when a lazy reader feeds
an Icechunk session, a transpose, or a sort.

Memory scales with the worker count, not with the store: each worker holds up to about 630 MB (its
512 MiB band plus imports), so a run takes about `cpus x 630 MB` plus a small parent process. Ops on
a zarr store run their workers as threads, so that memory sits in the one process. annizarr does not
guess how much of the machine it may take; on a host with less memory per core, pass a lower
`--cpus`. A worker process killed for memory raises an error that says so. A warning is logged when
a single chunk is larger than the budget (no band can be smaller than one chunk).

## Atomic writes

`convert`, `rechunk` and `sort` write into a sibling `OUT.tmp-<id>` directory, verify the new store
opens with the expected shape, then rename it onto the target. A killed run never leaves a partial
store at the destination, and an existing target is only replaced with `--overwrite`.

## Icechunk

`--ic` (or `IOConfig(backend="icechunk")`) writes through an Icechunk repository. Every op is exactly
one commit on `--branch` (default `main`, created off main if missing) with `-m` as the message, or
an auto-generated one naming the op. `append` and `add-layer` work on an existing repo without `--ic`.
Remote targets (`s3://`, `gs://`) require Icechunk; credentials come from the environment.

## Re-running an op

- `add-layer` errors if the layer already exists unless `--overwrite`.
- `append` errors if every appended cell is already present ("already appended?") and only warns
  on a partial overlap, since barcodes legitimately collide across samples. It never drops
  obsm/obsp/layers without consent: it prompts on a terminal, or needs `--drop-derived` in a
  script; layers both stores carry are appended either way. Obs
  columns keep the store's encoding: a categorical column gains new categories for values it has
  not seen and re-codes incoming strings or differently coded categoricals against them (ordered
  categoricals must match exactly); a plain string column accepts categorical cells as strings. A
  categorical that outgrows the code width it was written with has its codes array rewritten at the
  next signed width. Obs columns are matched by name: one the cells lack gets missing values on the
  appended rows, one only the cells carry is left out unless `--include-new-obs`.
- `sort` writes a new store each time; every layer, `raw/X` and obsp graph keeps its own layout and
  chunking.
