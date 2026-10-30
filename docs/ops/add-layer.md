# add-layer

Copy X, another layer, or `raw/X` into a new layer of an existing store, in place, in another layout.
The values are copied exactly; nothing is normalized or transformed. The usual use is a CSC copy of X,
so one gene's values across all cells read from a few chunks instead of every chunk of a CSR X.

## Command line

```bash
annizarr add-layer sample.zarr --layer counts_csc                       # CSC copy of X
annizarr add-layer sample.zarr --layer log1p_csc --matrix layers/log1p
annizarr add-layer sample.zarr --layer raw_csc --matrix raw/X           # raw/var must match var
annizarr add-layer sample.zarr --layer x_dense --layout dense --row-chunk 2048 --col-chunk 512
annizarr add-layer sample.zarr --layer counts_csc --overwrite           # replace an existing layer
annizarr add-layer repo.icechunk --layer counts_csc --branch dev -m "csc copy for gene reads"
```

## Python

```python
import annizarr as az

az.add_layer("sample.zarr", layer="counts_csc")
az.add_layer("sample.zarr", layer="log1p_csc", matrix="layers/log1p", layout="csc", overwrite=True)
```

## Options

- `--layer NAME`: name of the new layer. Required.
- `--matrix PATH`: what to copy: `X`, `layers/<name>`, or `raw/X`. Default `X`.
- `--layout csc|csr|dense`: layout of the new layer. Default `csc`.
- `--row-chunk N`, `--col-chunk N`: chunking of the layer, with the same meaning as on `convert`:
  about N cells per chunk for CSR, about N genes for CSC, exact `row_chunk` by `col_chunk` tiles for
  dense.
- `--clevel N`: compression level of the layer, 0 to 9. Default 3.
- `--cpus N`, `--auto-shard`, `--overwrite`, `--branch`, `-m`: see the
  [CLI reference](../cli.md#add-layer).

## Behaviour

- The source can be any encoding (CSR, CSC or dense) and must be cells by var; `raw/X` qualifies only
  when `raw/var` has the same names in the same order as `var`. Values and dtype are kept as they are.
- Normalization is analysis, not layout: run it in scanpy or rapids-singlecell, write the result as a
  layer, then copy that layer with `--matrix layers/<name>` if you want it in another layout.
- Errors when the layer already exists unless `--overwrite`, and refuses to copy a layer onto itself.
- Streams through the same writers as X, so memory is bounded at any store size. CSC output goes
  through an on-disk transpose in temp files under the store.
- Records the source on the layer (`annizarr-source`), so an `append` that has to drop the layer can
  say how to re-create it.
- On Icechunk the layer is one commit on `--branch`; an existing repository is detected without
  `--ic`. With a `Repo` and `commit=False` the layer is staged for a commit you make yourself; see
  [batching ops into one commit](../icechunk.md#batching-ops-into-one-commit).
