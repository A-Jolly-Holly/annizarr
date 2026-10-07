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
annizarr rechunk merged.zarr -o rechunked.zarr --row-chunk 2000

annizarr sort merged.zarr -o sorted.zarr --by cell_type
annizarr append sorted.zarr more_cells.zarr --drop-derived

annizarr add-expr merged.zarr --layout csr
annizarr add-expr repo.icechunk --layout csr --branch dev -m "add lognorm layer"
```

Icechunk history, branches, cherry-picks live in the Python API `annizarr.Repo` below

## Configuration and zarr stores

Every flag is documented in `annizarr <command> --help`; [docs/cli.md](docs/cli.md) has the same
reference in one place. Inputs stream band by band by default, so memory stays bounded at any store
size (`--eager` loads the whole input first, faster for small files), and matrix writes use every
core unless `--cpus` says otherwise.

Some of the main options; the docs have all of them:

- `--cpus N` — parallel band workers (default: all cores). Every command.
- `--layout csr|csc|dense` — on-disk layout of the matrix being written. Default `csr` for X on
  `convert`, `csc` for the `add-expr` layer. CSR suits row-wise access, CSC and dense column queries.
- `--row-chunk N`, `--col-chunk N` — chunk shape. Exact rows and columns for dense; for sparse, about
  N cells (csr) or N genes (csc) per chunk, sized from the average nonzeros. Defaults: 2048 for dense,
  about 9,000,000 nonzeros for sparse. `convert`, `rechunk`, `add-expr`.
- `--auto-shard` — shard the 1-D sparse arrays and anndata-written elements with zarr's automatic
  shard shape (default: off). Every command but `append`.
- `--ic` — write through an Icechunk repository; `--branch B` and `-m MSG` pick the branch and commit
  message. `append` and `add-expr` detect an existing repo on their own. `convert`, `rechunk`, `sort`.
- `--sort-by COL…` on `convert` / `--by COL…` on `sort` — physically order rows by obs columns,
  primary key first

**What gets written.** Every store is anndata-readable zarr v3 with `encoding-type`/`encoding-version`
attrs matching anndata 0.13's on-disk spec. X and layers may be dense, CSR or CSC on input (h5ad or
10x, mixed across inputs) and are written in whatever `--layout` asks for; X defaults to CSR.
`convert`/`rechunk`/`sort` write to a sibling temp store, verify it opens, then rename it onto the
target, so a killed run never leaves a partial store. On Icechunk every op is exactly one commit,
and remote (`s3://`, `gs://`) outputs require it. More in [docs/stores.md](docs/stores.md).

## Python API for Icechunk usage

The full Python API, including the config dataclasses and the ops functions, is in
[docs/python-api.md](docs/python-api.md).

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
