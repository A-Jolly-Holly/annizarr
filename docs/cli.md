# CLI reference

Generated from the argparse definitions by `python docs/gen_cli_reference.py`; `annizarr <command> --help`
prints the same text. `anz` is an alias for `annizarr`.

Global options go before the command: `-v`/`--verbose` (debug logging), `-q`/`--quiet` (warnings only),
`--version`.

## convert

Convert h5ad/10x input(s) into an AnnData zarr (or Icechunk) store

```text
usage: annizarr convert [-h] -o OUTPUT [--from {h5ad,10x}] [--eager] [--layout {csr,csc,dense}]
                        [--cpus CPUS] [--row-chunk N] [--col-chunk N] [--auto-shard]
                        [--sort-by COL [COL ...]] [--obs-columns COL [COL ...]]
                        [--consolidate-metadata] [--overwrite] [--ic] [--branch B] [-m MSG]
                        INPUT [INPUT ...]

positional arguments:
  INPUT                 input file(s); two or more inputs are concatenated (any mix of registered
                        kinds)

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   output store path or URI
  --from {h5ad,10x}     override content-based format detection
  --eager               load the whole input into memory first; faster for small files (default:
                        stream it band by band)
  --layout {csr,csc,dense}
                        output X layout (default: csr)
  --cpus CPUS           parallel band workers (default: all cores)
  --row-chunk N         rows per chunk: exact for dense output (default: 2048); for csr, about N
                        cells' worth of nonzeros per chunk (default: 1000000 nonzeros). Not for
                        csc.
  --col-chunk N         columns per chunk: exact for dense output (default: 2048); for csc, about
                        N genes' worth of nonzeros per chunk (default: 1000000 nonzeros). Not for
                        csr.
  --auto-shard          shard the anndata-written elements and the 1-D sparse arrays with zarr's
                        automatic shard shape (default: off)
  --sort-by COL [COL ...]
                        physically sort rows by these obs column(s), primary key first
  --obs-columns COL [COL ...]
                        project obs to this subset before concatenating (two or more inputs only)
  --consolidate-metadata
                        consolidate zarr metadata after writing
  --overwrite           replace an existing output
  --ic                  write through an Icechunk repository
  --branch B            Icechunk branch to read/write, created off main if missing (default:
                        main); ignored for plain zarr
  -m, --message MSG     Icechunk commit message (default: an auto-generated one naming the op);
                        ignored for plain zarr
```

## rechunk

Rewrite one matrix with new chunking; stream-copy the rest as-is

```text
usage: annizarr rechunk [-h] -o OUTPUT [--matrix MATRIX] [--row-chunk N] [--col-chunk N]
                        [--auto-shard] [--cpus CPUS] [--overwrite] [--consolidate-metadata] [--ic]
                        [--branch B] [-m MSG]
                        STORE

positional arguments:
  STORE

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   output store path or URI
  --matrix MATRIX       which matrix to rechunk: X, layers/<name>, or raw/X (default: X)
  --row-chunk N         rows per chunk: exact for dense output (default: 2048); for csr, about N
                        cells' worth of nonzeros per chunk (default: 1000000 nonzeros). Not for
                        csc.
  --col-chunk N         columns per chunk: exact for dense output (default: 2048); for csc, about
                        N genes' worth of nonzeros per chunk (default: 1000000 nonzeros). Not for
                        csr.
  --auto-shard          shard the anndata-written elements and the 1-D sparse arrays with zarr's
                        automatic shard shape (default: off)
  --cpus CPUS           parallel band workers (default: all cores)
  --overwrite           replace an existing output
  --consolidate-metadata
                        consolidate zarr metadata after writing
  --ic                  write through an Icechunk repository
  --branch B            Icechunk branch to read/write, created off main if missing (default:
                        main); ignored for plain zarr
  -m, --message MSG     Icechunk commit message (default: an auto-generated one naming the op);
                        ignored for plain zarr
```

## sort

Physically sort a store's rows by obs column(s) into a new store

```text
usage: annizarr sort [-h] -o OUTPUT --by COL [COL ...] [--cpus CPUS] [--auto-shard] [--overwrite]
                     [--consolidate-metadata] [--ic] [--branch B] [-m MSG]
                     STORE

positional arguments:
  STORE

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   output store path or URI
  --by COL [COL ...]    obs column(s) to sort by, primary key first
  --cpus CPUS           parallel band workers (default: all cores)
  --auto-shard          shard the anndata-written elements and the 1-D sparse arrays with zarr's
                        automatic shard shape (default: off)
  --overwrite           replace an existing output
  --consolidate-metadata
                        consolidate zarr metadata after writing
  --ic                  write through an Icechunk repository
  --branch B            Icechunk branch to read/write, created off main if missing (default:
                        main); ignored for plain zarr
  -m, --message MSG     Icechunk commit message (default: an auto-generated one naming the op);
                        ignored for plain zarr
```

## append

Append another store's cells onto this one, in place

```text
usage: annizarr append [-h] [--drop-derived] [--extend-layers] [--cpus CPUS] [--branch B] [-m MSG]
                       STORE CELLS

positional arguments:
  STORE
  CELLS

options:
  -h, --help         show this help message and exit
  --drop-derived     consent up front to dropping derived obs-aligned elements
                     (obsm/obsp/incompatible layers); without it you are prompted
  --extend-layers    extend eligible add-expr CSR layers in place instead of dropping them
  --cpus CPUS        parallel band workers (default: all cores)
  --branch B         Icechunk branch to read/write, created off main if missing (default: main);
                     ignored for plain zarr
  -m, --message MSG  Icechunk commit message (default: an auto-generated one naming the op);
                     ignored for plain zarr
```

## add-expr

Add a log-normalized expression layer derived from CSR X

```text
usage: annizarr add-expr [-h] [--layout {csr,csc,dense}] [--layer LAYER] [--target-sum TARGET_SUM]
                         [--row-chunk N] [--col-chunk N] [--auto-shard] [--cpus CPUS]
                         [--overwrite] [--branch B] [-m MSG]
                         STORE

positional arguments:
  STORE

options:
  -h, --help            show this help message and exit
  --layout {csr,csc,dense}
                        on-disk layout of the new layer (default: csc)
  --layer LAYER         layer name (default: gexp)
  --target-sum TARGET_SUM
                        library-size normalization target (default: 10000.0)
  --row-chunk N         rows per chunk: exact for dense output (default: 2048); for csr, about N
                        cells' worth of nonzeros per chunk (default: 1000000 nonzeros). Not for
                        csc.
  --col-chunk N         columns per chunk: exact for dense output (default: 2048); for csc, about
                        N genes' worth of nonzeros per chunk (default: 1000000 nonzeros). Not for
                        csr.
  --auto-shard          shard the anndata-written elements and the 1-D sparse arrays with zarr's
                        automatic shard shape (default: off)
  --cpus CPUS           parallel band workers (default: all cores)
  --overwrite           replace an existing output
  --branch B            Icechunk branch to read/write, created off main if missing (default:
                        main); ignored for plain zarr
  -m, --message MSG     Icechunk commit message (default: an auto-generated one naming the op);
                        ignored for plain zarr
```
