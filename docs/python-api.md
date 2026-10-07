# Python API

Everything below is importable from `annizarr` (`import annizarr as az`). The CLI is a thin layer
over these functions, so anything the CLI does the API does too, and the flag names match the
keyword arguments. `anz` is an alias for the `annizarr` command.

## Operations

Every op returns an `OpResult(path, n_obs, n_vars, snapshot_id)`. `snapshot_id` is `None` for plain
zarr and the committed Icechunk snapshot id otherwise. `cfg` is an [`AppConfig`](#configuration)
(defaults when omitted); `branch` and `message` only matter for Icechunk targets.

| Function | What it does |
|---|---|
| `az.convert(inputs, *, output, cfg=None, fmt=None, branch=None, message=None)` | One path, an in-memory `AnnData`, or a sequence of paths; two or more paths are concatenated. `fmt="h5ad"` or `"10x"` overrides content sniffing for a single path. |
| `az.rechunk(store, *, output, matrix="X", cfg=None, branch=None, message=None)` | Rewrite `matrix` (`"X"`, `"layers/<name>"`, or `"raw/X"`) with the chunking in `cfg.chunks`; stream-copy everything else unchanged. |
| `az.sort(store, *, output, by, cfg=None, branch=None, message=None)` | Physically order rows by the obs columns in `by` into a new store. A lone `layers/gexp` is re-derived on the sorted output. |
| `az.plan_append(store, *, cells)` | Metadata-only dry run of `append`: returns an `AppendPlan` saying what would be dropped or extended. |
| `az.append(store, *, cells, drop_derived=False, extend_layers=False, cfg=None, branch=None, message=None)` | Append the cells of another converted store in place. Raises if it would drop obsm/obsp/layers unless `drop_derived=True`; `extend_layers=True` extends add-expr CSR layers instead of dropping them. |
| `az.add_expr(store, *, layout="csc", layer="gexp", target_sum=1e4, overwrite=False, cfg=None, branch=None, message=None)` | Add a log-normalized layer derived from CSR X. Chunking comes from `cfg.chunks`. |

`AppendPlan(n_new, drop_obsm, drop_obsp, drop_layers, extendable_layers, n_duplicate_names, notes)`
is a frozen dataclass; `plan.drops(extend_layers=False)` lists the element paths that would go.

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
| `io.lazy` | `True` | Stream h5ad input band by band; `False` loads it whole first (faster for small files). |
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

## Icechunk: `az.Repo`

```python
repo = az.Repo("repo.icechunk")  # opens on main; Repo(path, branch="dev") pins a branch
repo.branches()
repo.log()
repo.tree()  # git-style reprs in a notebook
root = repo.open_zarr("w")  # writable root group; edits stage until commit
root.attrs["step"] = "lognorm"
repo.commit("normalize")
repo.checkout("experiment", create=True)  # switches this object only; nothing is persisted
old = repo.open_zarr("r", snapshot_id=repo.log()[1].id)  # time travel, read-only
az.Repo("gs://bucket/store", anonymous=True)  # public bucket, no credentials
repo.copy("/work/store")  # clone with every branch and snapshot id intact
```

| Member | What it does |
|---|---|
| `Repo(path, *, branch=None, origin=None, anonymous=False)` | Open an existing repo at a local directory, `s3://` or `gs://` URI, on `branch` (default `main`). `anonymous=True` reads a public bucket with unsigned requests. `origin` is a writable location for a read-only `path`. |
| `Repo.create(path)` | New, empty repo. |
| `Repo.init(zarr_path, out_path, message=None)` | New repo seeded from a plain zarr store as one commit. |
| `Repo.exists(path, anonymous=False)` | Whether a repo exists there; never creates anything. |
| `repo.branch`, `repo.branches()` | Current branch name; all branches. |
| `repo.checkout(branch, create=False)` | Switch this object to `branch`; `create=True` branches off the current tip. Returns the tip snapshot id. |
| `repo.open_zarr("r", snapshot_id=None)` | Read-only root group at the branch tip, or at a past snapshot. |
| `repo.open_zarr("w", truncate=False)` | Writable root group on a session at the branch tip; `truncate=True` starts from an empty root. |
| `repo.commit(message, metadata=None)` | Commit the open session and return the snapshot id. |
| `repo.discard()` | Drop uncommitted changes. |
| `repo.log(branch=None)`, `repo.tree()` | Snapshots newest first; every branch's history as a text graph. |
| `repo.cherrypick(snapshot_id)` | Point the current branch at `snapshot_id` (like `git reset --hard`). |
| `repo.copy(dest)` | Copy the repo, every branch and snapshot id intact, to a fresh local directory or `s3://` prefix. |

Credentials for `s3://` and `gs://` come from the environment (AWS variables, profile or instance
role; gcloud application-default credentials).

## Extension points

`annizarr.sources.register_source(kind, loader, sniffer=None)` adds an input kind to `convert`:
`loader(path, cfg)` returns a `Source`, and the optional `sniffer(path)` lets content detection
recognise it. `annizarr.sources.detect_format(path)` and `open_source(path, cfg)` are the detection
and loading entry points `convert` itself uses.

## Errors

All exceptions derive from `az.AnzError` (a `RuntimeError`): `ConversionError`, `StorageError`,
`ValidationError` (also a `ValueError`), and `RepoError`. The CLI prints them as `error: ...` and
exits 1.
