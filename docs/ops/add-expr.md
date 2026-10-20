# add-expr

Add a log-normalized expression layer derived from X to an existing store, in place. Each cell is
scaled to a common library size and log-transformed, `log1p(x * target_sum / row_sum)`, with empty
cells left at zero. The layer is `layers/gexp` by default, written as CSC so one gene's values across
all cells read from a few chunks; CSR and dense are available too.

## Command line

```bash
annizarr add-expr sample.zarr                                  # layers/gexp, CSC, target sum 10000
annizarr add-expr sample.zarr --layout csr
annizarr add-expr sample.zarr --layout dense --row-chunk 2048 --col-chunk 512
annizarr add-expr sample.zarr --layer lognorm_1e6 --target-sum 1e6
annizarr add-expr sample.zarr --overwrite                       # replace an existing layer
annizarr add-expr repo.icechunk --branch dev -m "add lognorm layer"
```

## Python

```python
import annizarr as az

az.add_expr("sample.zarr")
az.add_expr("sample.zarr", layout="csr", layer="lognorm", target_sum=1e4, overwrite=True)
```

## Options

- `--layout csr|csc|dense`: layout of the new layer. Default `csc`.
- `--layer NAME`: layer name. Default `gexp`.
- `--target-sum S`: library-size target. Default `10000`.
- `--row-chunk N`, `--col-chunk N`: chunking of the layer, with the same meaning as on `convert`:
  about N cells per chunk for CSR, about N genes for CSC, exact `row_chunk` by `col_chunk` tiles for
  dense.
- `--cpus N`, `--auto-shard`, `--overwrite`, `--branch`, `-m`: see the
  [CLI reference](../cli.md#add-expr).

## Behaviour

- Requires CSR X. The layer has the same nonzeros as X, stored as float32.
- Errors when the layer already exists unless `--overwrite`.
- Streams through the same writers as X: one pass over X computes the per-cell factors, a second
  writes the layer band by band. CSC output goes through an on-disk transpose in temp files under the
  store. Memory is bounded at any store size.
- Records the target sum on the layer. That is what lets `append --extend-layers` extend the layer and
  `sort` re-derive it on a sorted store.
- On Icechunk the layer is one commit on `--branch`; an existing repository is detected without
  `--ic`. With a `Repo` and `commit=False` the layer is staged for a commit you make yourself; see
  [batching ops into one commit](../icechunk.md#batching-ops-into-one-commit).
