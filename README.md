# AnniZarr

[![CI](https://github.com/A-Jolly-Holly/annizarr/actions/workflows/ci.yml/badge.svg)](https://github.com/A-Jolly-Holly/annizarr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/annizarr)](https://pypi.org/project/annizarr/)

Convert, edit, and version AnnData Zarr stores — streaming, memory-bounded, and
Icechunk-versioned.

## Install

Not yet on PyPI — install from this directory. `annizarr` and the shorter `anz` are the same command.

```bash
pip install .
pip install ".[icechunk]"
```

Or use the pixi dev environment:

```bash
pixi install
```

## Quickstart usage

### Convert

The input format is sniffed from the HDF5 contents (h5ad vs 10x); `--from h5ad|10x` overrides it. Configuration can enable chunking, sharding, format ...
Two or more inputs are concatenated. `--ic` writes an Icechunk repository instead of plain zarr.

```bash
annizarr convert sample.h5ad -o sample.zarr

annizarr convert a.h5ad b.h5ad -o merged.zarr
annizarr convert sample.h5ad -o repo.icechunk --ic -m "initial import"
```

### Zarr operations and edits

`rechunk` rewrites one matrix with new chunking and streams everything else through unchanged.<br>
`sort` physically orders rows by obs column(s) into a new store.<br> `append` adds cells to an existing zarr/ic store; it prompts before dropping obsm/obsp/layers (`--drop-derived` consents up front)<br>
`add-expr` adds a log-normalized layer (`layers/gexp`) derived from X/. Csc by default for fast column reads

```bash
annizarr rechunk merged.zarr -o rechunked.zarr --x-row-chunk 2000

annizarr sort merged.zarr -o sorted.zarr --by cell_type
annizarr append sorted.zarr more_cells.zarr --drop-derived

annizarr add-expr merged.zarr --format csr
annizarr add-expr repo.icechunk --format csr --branch dev -m "add lognorm layer"
```

Icechunk history, branches, cherry-picks live in the Python API `annizarr.Repo` below

## Configuration and zarr stores

<b>Every flag is documented in `annizarr <command> --help`.</b><br> Inputs streams inputs lazily by default, so memory stays bounded at any store size; `--eager` can overwrite this for small files. Matrix writes use every core unless `--cpus` says otherwise.

| Flag | Default | Commands | Effect |
|---|---|---|---|
| `--cpus N` | all cores | all | Parallel band workers for matrix writes. |
| `--x-storage csr\|csc\|dense` | `csr` | convert | On-disk layout of X: CSR for row-wise access, CSC or dense for column queries. |
| `--x-row-chunk`, `--x-col-chunk` | 2048 | convert, rechunk | X chunk shape; the column chunk applies to dense X only. |
| `--sparse-flat-chunk` | 1,000,000 | convert, rechunk | Flat chunk size of sparse `data`/`indices`. |
| `--auto-shard` | off | all but append | Shard the 1-D sparse arrays and anndata-written elements with zarr's auto shard shape. |
| `--ic` | off | convert, rechunk, sort | Write through an Icechunk repository; `--branch B` and `-m MSG` pick the branch and commit message. `append`/`add-expr` detect an existing repo on their own. |

Each command has more flags than this; `--help` lists them.

**Zarr conventions:** Every store is anndata-readable zarr v3 with `encoding-type`/`encoding-version`
attrs matching anndata 0.12's on-disk spec. X and layers from the input file may be dense, CSR or CSC (h5ad, 10x h5, other filetypes) and are written to the format set in configuration above, defaults to CSR.<br> 
`convert`/`rechunk`/`sort` write to a sibling temp store, verify it opens, then rename it onto the
target, so a killed run never leaves a partial store. On Icechunk every op is exactly one commit,
and remote (`s3://`, `gs://`) outputs require it.

## Python API for Icechunk usage

```python
import annizarr as az

repo = az.Repo("repo.icechunk")

repo.branches()
repo.log()
repo.tree()

root = repo.open_zarr("w")
root.attrs["step"] = "lognorm"
repo.commit("normalize")

repo.checkout("experiment", create=True)  # switches this object only; nothing is persisted
old = repo.open_zarr("r", snapshot_id=repo.log()[1].id)

repo.copy("/work/store")  # clone with every branch and snapshot id intact, good for readonly cases
```

`OpResult(path, n_obs, n_vars, snapshot_id)` — `snapshot_id` is `None` for plain zarr,
the committed Icechunk snapshot id otherwise.

## Development

```bash
pixi install
pixi run -e default pytest              # excludes -m slow by default
pixi run -e default pytest -m slow      # large synthetic-store memory-ceiling test
pixi run -e default ruff check .
pixi run -e default mypy --strict src/annizarr
pre-commit run --all-files
```

Golden stores for regression tests live in `tests/golden/`.

## License

MIT — see [LICENSE](LICENSE).
