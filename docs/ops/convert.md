# convert

Convert one or more h5ad or 10x HDF5 files into a single anndata-readable zarr v3 store, or into an
Icechunk repository. The input format is sniffed from the file contents; `--from` overrides it. Two or
more inputs are concatenated, in any mix of file kinds and matrix layouts. The input streams band by
band, so a file with millions of cells converts in bounded memory; `--eager` loads it whole first,
which is faster for small files.

## Command line

```bash
annizarr convert sample.h5ad -o sample.zarr
annizarr convert filtered_feature_bc_matrix.h5 -o sample.zarr --from 10x
annizarr convert a.h5ad b.h5ad c.h5ad -o merged.zarr --obs-columns sample cell_type
annizarr convert sample.h5ad -o dense.zarr --layout dense --row-chunk 4096 --col-chunk 1024
annizarr convert sample.h5ad -o sorted.zarr --sort-by cell_type
annizarr convert sample.h5ad -o repo.icechunk --ic -m "initial import"
```

## Python

```python
import annizarr as az
from annizarr.config import AppConfig, ChunkConfig, IOConfig

az.convert("sample.h5ad", output="sample.zarr")
az.convert(["a.h5ad", "b.h5ad"], output="merged.zarr")
az.convert(adata, output="from_memory.zarr")  # an in-memory AnnData works too

cfg = AppConfig(io=IOConfig(layout="dense", overwrite=True), chunks=ChunkConfig(row_chunk=4096, cpus=8))
result = az.convert("sample.h5ad", output="dense.zarr", cfg=cfg, fmt="h5ad")
result.n_obs, result.n_vars, result.snapshot_id  # snapshot_id is None for plain zarr
```

## Options

- `--layout csr|csc|dense`: on-disk layout of X; layers follow it. Default `csr`. CSR suits reading
  cells, CSC and dense suit reading genes across cells.
- `--row-chunk N`, `--col-chunk N`: chunk shape. Exact for dense (2048 by 2048 by default); for
  sparse, about N cells (CSR) or N genes (CSC) per chunk. The other axis is a usage error for sparse
  output.
- `--cpus N`: parallel band workers. Default all cores; lower it to leave headroom on a shared host.
- `--eager`: load the whole input first instead of streaming it. Faster for small files.
- `--sort-by COL [COL ...]`: physically order rows by obs columns while converting, primary key first.
  Streams for CSR output; dense or CSC output, and inputs with layers, raw or obsp, need `--eager`.
- `--obs-columns COL [COL ...]`: with two or more inputs, project obs to these columns. Without it the
  inputs need identical obs schemas.
- `--auto-shard`, `--consolidate-metadata`, `--overwrite`, `--ic`, `--branch`, `-m`: see the
  [CLI reference](../cli.md#convert) for every flag.

## Behaviour

- Writes into a sibling `OUT.tmp-<id>` directory, verifies the new store opens with the expected
  shape, then renames it onto the target. A killed run never leaves a partial store, and an existing
  target is only replaced with `--overwrite`.
- A single input is converted whole: X, layers, raw, obs, var, uns, obsm, varm, obsp and varp. The
  frames and the unstructured elements go through anndata's own writers, so anything anndata can
  write round-trips. See [What annizarr writes](../stores.md).
- Concatenation keeps X, obs and var. The inputs need identical var (gene names and order); obs
  columns must match or be projected with `--obs-columns`, and a column that is categorical in one
  input but not another, or with different categories, is refused rather than coerced. Layers, raw,
  uns, obsm, varm, obsp and varp are not carried over; the output has them empty.
- With `--ic` the whole conversion is one commit on `--branch`, with `-m` as the message.
