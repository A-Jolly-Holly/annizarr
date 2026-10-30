"""Out-of-core bench: every annizarr command in every layout, across store sizes and worker counts.

Reports wall time and peak RSS of the CLI process and of its whole process tree. Not run by CI or
any workflow. Results append to ``OUT_DIR/results.tsv`` as each command
finishes, so a partial run still has data, and a markdown table prints at the end.

    python tests/ooc/bench.py OUT_DIR
    python tests/ooc/bench.py OUT_DIR --sizes 150000,300000,600000 --cpus 1,4,12
    python tests/ooc/bench.py /tmp/smoke --sizes 3000 --cpus 1,2   # smoke run

Per size the h5ad inputs are built once (blockwise, bounded RAM). Per worker count the
commands run in dependency order on a fresh CSR store; outputs that nothing else needs are
deleted right away so peak disk stays near one input plus two stores.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ooc import MB, Measurement, build_h5ad, run_measured  # needs the path insert above

N_VARS = 30_000
NNZ_PER_CELL = 2_000  # about what 10x 3' data carries; the slow test runs sparser
N_APPEND = 2_000
COLUMNS = ("size", "cpus", "command", "layout", "rc", "wall_s", "self_MB", "tree_MB", "procs")


def _commands(d: Path, n: str) -> list[tuple[str, str, list[str], list[Path]]]:
    # (command, layout, argv, paths to delete afterwards); csr.zarr persists through the cpus pass
    h5, extra = d / "in.h5ad", d / "extra.h5ad"
    csr, csc, dense = d / "csr.zarr", d / "csc.zarr", d / "dense.zarr"
    concat, sort, rechunk, extra_store = d / "concat.zarr", d / "sorted.zarr", d / "rechunk.zarr", d / "extra.zarr"
    q = ["-q"]
    return [
        ("convert", "csr", [*q, "convert", str(h5), "-o", str(csr), "--cpus", n], []),
        ("convert", "csc", [*q, "convert", str(h5), "-o", str(csc), "--layout", "csc", "--cpus", n], [csc]),
        ("convert", "dense", [*q, "convert", str(h5), "-o", str(dense), "--layout", "dense", "--cpus", n], [dense]),
        ("concat", "csr", [*q, "convert", str(h5), str(extra), "-o", str(concat), "--cpus", n], [concat]),
        ("sort", "csr", [*q, "sort", str(csr), "-o", str(sort), "--by", "cell_type", "--cpus", n], [sort]),
        (
            "rechunk",
            "csr",
            [*q, "rechunk", str(csr), "-o", str(rechunk), "--row-chunk", "10000", "--cpus", n],
            [rechunk],
        ),
        ("add-layer", "csr", [*q, "add-layer", str(csr), "--layer", "l_csr", "--layout", "csr", "--cpus", n], []),
        ("add-layer", "csc", [*q, "add-layer", str(csr), "--layer", "l_csc", "--layout", "csc", "--cpus", n], []),
        ("add-layer", "dense", [*q, "add-layer", str(csr), "--layer", "l_dense", "--layout", "dense", "--cpus", n], []),
        ("convert", "extra", [*q, "convert", str(extra), "-o", str(extra_store), "--cpus", n], []),
        (
            "append",
            "csr",
            [*q, "append", str(csr), str(extra_store), "--drop-derived", "--cpus", n],
            [csr, extra_store],
        ),
    ]


def _row(size: int, cpus: int, command: str, layout: str, m: Measurement) -> tuple[str, ...]:
    return (
        str(size),
        str(cpus),
        command,
        layout,
        str(m.rc),
        f"{m.wall_s:.2f}",
        f"{m.self_peak_bytes / MB:.0f}",
        f"{m.tree_peak_bytes / MB:.0f}",
        str(m.tree_peak_procs),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the bench matrix given on the command line; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--sizes", default="150000,300000,600000", help="cells per store (default: 150000,300000,600000)")
    ap.add_argument("--cpus", default="1,4,12", help="worker counts (default: 1,4,12)")
    ap.add_argument("--n-vars", type=int, default=N_VARS, help=f"genes per store (default: {N_VARS})")
    ap.add_argument(
        "--nnz-per-cell", type=int, default=NNZ_PER_CELL, help=f"average nonzeros per cell (default: {NNZ_PER_CELL})"
    )
    ap.add_argument("--only", default="", help="comma-separated command names to run (default: all)")
    ap.add_argument("--keep", action="store_true", help="keep every output store (default: delete as you go)")
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",")]
    density = args.nnz_per_cell / args.n_vars
    cpus_list = [int(c) for c in args.cpus.split(",")]
    only = {c for c in args.only.split(",") if c}
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.tsv"
    if not results.exists():
        results.write_text("\t".join(COLUMNS) + "\n")
    rows: list[tuple[str, ...]] = []
    free = shutil.disk_usage(out_dir).free
    print(
        f"out_dir={out_dir}  free_disk={free / 1024**3:.1f}GB  sizes={sizes}  cpus={cpus_list}  "
        f"nnz_per_cell={args.nnz_per_cell}",
        flush=True,
    )

    for size in sizes:
        d = out_dir / f"n{size}"
        d.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        if not (d / "in.h5ad").exists():
            build_h5ad(d / "in.h5ad", n_obs=size, n_vars=args.n_vars, density=density, seed=0, name_prefix="cell")
        if not (d / "extra.h5ad").exists():
            build_h5ad(
                d / "extra.h5ad", n_obs=N_APPEND, n_vars=args.n_vars, density=density, seed=1, name_prefix="extra"
            )
        print(f"\n== size={size}: inputs ready in {time.perf_counter() - t0:.0f}s", flush=True)
        for cpus in cpus_list:
            for command, layout, argv_, cleanup in _commands(d, str(cpus)):
                if only and command not in only:
                    continue
                m = run_measured(argv_)
                row = _row(size, cpus, command, layout, m)
                rows.append(row)
                with results.open("a") as f:
                    f.write("\t".join(row) + "\n")
                status = (
                    ""
                    if m.rc == 0
                    else f"  FAILED rc={m.rc}: {m.stderr.strip().splitlines()[-1] if m.stderr.strip() else ''}"
                )
                print(
                    f"size={size:<7} cpus={cpus:<2} {command + ' ' + layout:<16} wall={m.wall_s:7.1f}s "
                    f"self={m.self_peak_bytes / MB:6.0f}MB tree={m.tree_peak_bytes / MB:6.0f}MB "
                    f"procs={m.tree_peak_procs}{status}",
                    flush=True,
                )
                if not args.keep:
                    for p in cleanup:
                        shutil.rmtree(p, ignore_errors=True)
            if not args.keep:  # the stores the next worker count rebuilds, even when --only skipped append
                for p in (d / "csr.zarr", d / "extra.zarr"):
                    shutil.rmtree(p, ignore_errors=True)
        if not args.keep:
            shutil.rmtree(d, ignore_errors=True)

    print("\n| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
