# AnniZarr

Convert h5ad and 10x files into anndata-readable zarr v3 stores, then edit those stores: rechunk,
sort, append cells, copy a matrix into a layer in another layout. Every operation streams the matrix band by band and
uses every core, so memory stays bounded at any store size, and every operation can write through an
[Icechunk](icechunk.md) repository for versioned history.

## Install

```bash
pip install annizarr
pip install "annizarr[icechunk]"   # Icechunk repositories, and s3:// or gs:// locations
```

## Quickstart

```bash
annizarr convert sample.h5ad -o sample.zarr
annizarr convert a.h5ad b.h5ad -o merged.zarr                 # two or more inputs are concatenated
annizarr rechunk merged.zarr -o rechunked.zarr --row-chunk 2000
annizarr sort merged.zarr -o sorted.zarr --by cell_type
annizarr append sorted.zarr more_cells.zarr --drop-derived
annizarr add-layer merged.zarr --layer counts_csc             # a CSC copy of X, for gene reads
annizarr convert sample.h5ad -o repo.icechunk --ic -m "initial import"
```

`anz` is an alias for `annizarr`. The same operations exist in Python, with the flag names as keyword
arguments:

```python
import annizarr as az

az.convert("sample.h5ad", output="sample.zarr")
az.add_layer("sample.zarr", layer="counts_csc")
repo = az.Repo("repo.icechunk")
repo.log()
```

## Where to go next

- One page per operation: [convert](ops/convert.md), [rechunk](ops/rechunk.md), [sort](ops/sort.md),
  [append](ops/append.md), [add-layer](ops/add-layer.md).
- [Icechunk](icechunk.md): writing through a repository, branches, snapshots and the `Repo` API.
- Reference: the [Python API](python-api.md) (configuration, errors, extension points), the
  [CLI](cli.md) (every flag, generated from `--help`) and [What annizarr writes](stores.md) (on-disk
  encoding, chunking, streaming).
