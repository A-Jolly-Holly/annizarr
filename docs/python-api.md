# Python API

Everything below is importable from `annizarr` (`import annizarr as az`). The CLI is a thin layer
over these functions, so anything the CLI does the API does too, and the flag names match the
keyword arguments. `anz` is an alias for the `annizarr` command.

## Operations

One function per command, documented on its own page: [convert](ops/convert.md),
[rechunk](ops/rechunk.md), [sort](ops/sort.md), [append](ops/append.md), [add-expr](ops/add-expr.md).
Every op returns an `OpResult(path, n_obs, n_vars, snapshot_id)`. `snapshot_id` is `None` for plain
zarr and the committed Icechunk snapshot id otherwise. `cfg` is an [`AppConfig`](#configuration)
(defaults when omitted); `branch` and `message` only matter for Icechunk targets.

`store`, `cells` and `output` are `annizarr.typing.StoreLike`: a local path, an `s3://` or `gs://`
URI, or an [`az.Repo`](icechunk.md#a-repo-as-the-store), used as given with its branch and origin.

| Function | Page |
|---|---|
| `az.convert(inputs, *, output, cfg=None, fmt=None, branch=None, message=None)` | [convert](ops/convert.md) |
| `az.rechunk(store, *, output, matrix="X", cfg=None, branch=None, message=None)` | [rechunk](ops/rechunk.md) |
| `az.sort(store, *, output, by, cfg=None, branch=None, message=None)` | [sort](ops/sort.md) |
| `az.plan_append(store, *, cells)` | [append](ops/append.md) |
| `az.append(store, *, cells, drop_derived=False, extend_layers=False, include_new_obs=False, cfg=None, branch=None, message=None, commit=True)` | [append](ops/append.md) |
| `az.add_expr(store, *, layout="csc", layer="gexp", target_sum=1e4, overwrite=False, cfg=None, branch=None, message=None, commit=True)` | [add-expr](ops/add-expr.md) |

## Configuration

```python
from annizarr.config import AppConfig, ChunkConfig, IOConfig

cfg = AppConfig(
    io=IOConfig(layout="dense", overwrite=True),
    chunks=ChunkConfig(row_chunk=4096, col_chunk=1024, cpus=8),
)
az.convert("sample.h5ad", output="sample.zarr", cfg=cfg)
```

All config objects are frozen dataclasses: build them with keyword arguments or
`dataclasses.replace`. `annizarr.config.apply_cli_overrides(cfg, **flags)` overlays flag-style
keyword arguments (`None` means unset) onto a config; the CLI uses it.

| Field | Default | Meaning |
|---|---|---|
| `io.layout` | `"csr"` | On-disk layout of X and layers: `"csr"`, `"csc"`, or `"dense"`. |
| `io.lazy` | `True` | Stream h5ad and 10x input band by band; `False` loads it whole first (faster for small files). |
| `io.backend` | `"zarr"` | `"icechunk"` writes through a versioned repository, one commit per op. |
| `io.overwrite` | `False` | Replace an existing output store. |
| `io.consolidate_metadata` | `False` | Consolidate zarr metadata after writing (plain zarr only). |
| `chunks.row_chunk` | `None` | Rows per chunk. Exact for dense X (2048 when unset); for CSR, about this many cells' worth of nonzeros per chunk. Ignored for CSC. |
| `chunks.col_chunk` | `None` | Columns per chunk. Exact for dense X (2048 when unset); for CSC, about this many genes' worth of nonzeros per chunk. Ignored for CSR. |
| `chunks.nnz_chunk` | `9_000_000` | Flat `data`/`indices` chunk length for sparse output when the matching axis chunk is unset. |
| `chunks.cpus` | all cores | Parallel band workers for matrix writes. |
| `chunks.shard_factor` | `1` | Chunks per shard along each axis of dense X; `1` means no sharding. Python-only. |
| `chunks.auto_shard` | `False` | Shard the 1-D sparse arrays and anndata-written elements with zarr's `shards="auto"`. |
| `grouping.sort_by` | `()` | `convert` only: physically sort rows by these obs columns (set `enabled=True`, or pass `--sort-by`). |
| `concat.obs_columns` | `()` | Multi-input `convert`: project obs to these columns; empty requires identical obs schemas. |

## Icechunk

`az.Repo` and writing through a repository are on the [Icechunk](icechunk.md) page.

## Extension points

`annizarr.sources.register_source(kind, loader, sniffer=None)` adds an input kind to `convert`:
`loader(path, cfg)` returns a `Source`, and the optional `sniffer(path)` lets content detection
recognise it. `annizarr.sources.detect_format(path)` and `open_source(path, cfg)` are the detection
and loading entry points `convert` itself uses.

## Errors

All exceptions derive from `az.AnzError` (a `RuntimeError`): `ConversionError`, `StorageError`,
`ValidationError` (also a `ValueError`), and `RepoError`. The CLI prints them as `error: ...` and
exits 1.
