# Icechunk

[Icechunk](https://icechunk.io) is a transactional storage engine for zarr: every change is a commit,
history is kept as snapshots, and branches let you work on a copy without touching `main`. annizarr
writes any store through it, and `az.Repo` wraps the parts of its API an AnnData store needs.

Install the extra first: `pip install "annizarr[icechunk]"`. It also brings the clients for `s3://` and
`gs://` locations, which require Icechunk.

## Writing through Icechunk

Every operation takes `--ic` (or `IOConfig(backend="icechunk")`) and is exactly one commit on
`--branch` (default `main`, created off main if missing), with `-m` as the message or an auto-generated
one naming the op. `append` and `add-expr` detect an existing repository on their own.

```bash
annizarr convert sample.h5ad -o repo.icechunk --ic -m "initial import"
annizarr add-expr repo.icechunk --branch dev -m "add lognorm layer"
annizarr append repo.icechunk batch2.zarr -m "batch 2"
annizarr sort repo.icechunk -o sorted.icechunk --by cell_type --ic
```

Every op returns an `OpResult` whose `snapshot_id` is the committed snapshot. A failed op leaves the
repository at the previous snapshot, because Icechunk discards uncommitted changes.

Each commit carries metadata alongside its message: the annizarr version, the op, and the input it
read where there is one (`{"annizarr": "0.2.3", "op": "append", "cells": "batch2.zarr"}`). It is on
`repo.log()[0].metadata`.

## `az.Repo`

```python
import annizarr as az

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
| `repo.dirty` | Whether the open session holds uncommitted changes. |
| `repo.checkout(branch, create=False)` | Switch this object to `branch`; `create=True` branches off the current tip. Returns the tip snapshot id. |
| `repo.open_zarr("r", snapshot_id=None)` | Read-only root group at the branch tip, or at a past snapshot. |
| `repo.open_zarr("w", truncate=False)` | Writable root group on a session at the branch tip; `truncate=True` starts from an empty root. |
| `repo.commit(message, metadata=None, allow_empty=False)` | Commit the open session and return the snapshot id. |
| `repo.discard()` | Drop uncommitted changes. |
| `repo.log(branch=None)`, `repo.tree()` | Snapshots newest first; every branch's history as a text graph. |
| `repo.cherrypick(snapshot_id)` | Point the current branch at `snapshot_id` (like `git reset --hard`). |
| `repo.copy(dest)` | Copy the repo, every branch and snapshot id intact, to a fresh local directory or `s3://` prefix. |

## A `Repo` as the store

Every op accepts a `Repo` wherever it takes a path, for `store`, `cells` and `output` alike. The op
uses the object as given: its branch, its `origin` and its credentials stand, so a read-only mount
with a writable origin is reachable from any op. A `branch=` that differs from the repo's checks it
out (created off the tip if missing). `plan_append(repo, cells=...)` reads what the repo has staged,
so it sees an uncommitted append. Writing a new store onto a `Repo` (`convert`, `rechunk`, `sort`)
needs `overwrite`, like a path, and a repo with uncommitted changes is refused.

```python
repo = az.Repo("/mnt/atlas.icechunk", origin="s3://bucket/atlas.icechunk", branch="ingest")
az.append(repo, cells="batch2.zarr")  # reads the mount, commits to the origin
```

## Batching ops into one commit

`append` and `add_expr` take `commit=False`, which stages the edit on the `Repo` you pass and leaves
the commit to you, so several edits land as one snapshot with your own message and metadata. The
results carry `snapshot_id=None`, as a plain-zarr result does. It needs a `Repo` instance (a path
would open a repo the op then drops, losing the staged session) and an Icechunk store; anything
else raises `StorageError`. Deferring skips the commit, never the checks `append` runs before it.

```python
repo = az.Repo("s3://bucket/atlas.icechunk", branch="ingest-42")
az.append(repo, cells="batch42.zarr", drop_derived=True, commit=False)
az.add_expr(repo, layout="csr", commit=False)
repo.commit("ingest 42", metadata={"run": "42"})
```

There is no CLI flag: one op per process has nothing to batch.

Credentials for `s3://` and `gs://` come from the environment: AWS variables, profile or instance
role; gcloud application-default credentials.
