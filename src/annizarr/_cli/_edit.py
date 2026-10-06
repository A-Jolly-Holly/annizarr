from __future__ import annotations

import argparse
import logging
import sys

from annizarr._cli._args import (
    add_autoshard_arg,
    add_branch_arg,
    add_chunk_args,
    add_consolidate_arg,
    add_cpus_arg,
    add_ic_arg,
    add_message_arg,
    add_overwrite_arg,
    build_config,
    reject_off_axis_chunks,
)

_LOG = logging.getLogger(__name__)


def add_add_expr_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("add-expr", help="Add a log-normalized expression layer derived from CSR X")
    p.add_argument("store", metavar="STORE")
    # defaults below are literals, kept in sync by hand with add_expr's signature: importing
    # it here would pull in numpy/zarr on every CLI invocation, even --version.
    p.add_argument(
        "--layout",
        dest="layer_layout",  # not cfg.io.layout (X's layout): this is the new layer's
        choices=("csr", "csc", "dense"),
        default="csc",
        help="on-disk layout of the new layer (default: csc)",
    )
    p.add_argument("--layer", default="gexp", help="layer name (default: gexp)")
    p.add_argument("--target-sum", type=float, default=1e4, help="library-size normalization target (default: 10000.0)")
    add_chunk_args(p)
    add_autoshard_arg(p)
    add_cpus_arg(p)
    add_overwrite_arg(p)
    add_branch_arg(p)
    add_message_arg(p)
    p.set_defaults(func=_run_add_expr, _parser=p)


def _run_add_expr(args: argparse.Namespace) -> int:
    from annizarr.ops import add_expr

    cfg = build_config(args)
    reject_off_axis_chunks(args._parser, args, args.layer_layout)
    result = add_expr(
        args.store,
        layout=args.layer_layout,
        layer=args.layer,
        target_sum=args.target_sum,
        overwrite=bool(args.overwrite),
        cfg=cfg,
        branch=args.branch,
        message=args.message,
    )
    _LOG.info(f"updated {result.path} ({result.n_obs} x {result.n_vars})")
    if result.snapshot_id is not None:
        _LOG.info(f"icechunk snapshot {result.snapshot_id}")
    return 0


def add_rechunk_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("rechunk", help="Rewrite one matrix with new chunking; stream-copy the rest as-is")
    p.add_argument("store", metavar="STORE")
    p.add_argument("-o", "--output", required=True, help="output store path or URI")
    p.add_argument("--matrix", default="X", help="which matrix to rechunk: X, layers/<name>, or raw/X (default: X)")
    add_chunk_args(p)
    add_autoshard_arg(p)
    add_cpus_arg(p)
    add_overwrite_arg(p)
    add_consolidate_arg(p)
    add_ic_arg(p)
    add_branch_arg(p)
    add_message_arg(p)
    p.set_defaults(func=_run_rechunk)


def _run_rechunk(args: argparse.Namespace) -> int:
    from annizarr.ops import rechunk

    cfg = build_config(args)
    result = rechunk(
        args.store, output=args.output, matrix=args.matrix, cfg=cfg, branch=args.branch, message=args.message
    )
    _LOG.info(f"wrote {result.path} ({result.n_obs} x {result.n_vars})")
    if result.snapshot_id is not None:
        _LOG.info(f"icechunk snapshot {result.snapshot_id}")
    return 0


def add_sort_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("sort", help="Physically sort a store's rows by obs column(s) into a new store")
    p.add_argument("store", metavar="STORE")
    p.add_argument("-o", "--output", required=True, help="output store path or URI")
    p.add_argument(
        "--by",
        dest="sort_by",
        nargs="+",
        required=True,
        metavar="COL",
        help="obs column(s) to sort by, primary key first",
    )
    add_cpus_arg(p)
    add_autoshard_arg(p)
    add_overwrite_arg(p)
    add_consolidate_arg(p)
    add_ic_arg(p)
    add_branch_arg(p)
    add_message_arg(p)
    p.set_defaults(func=_run_sort)


def _run_sort(args: argparse.Namespace) -> int:
    from annizarr.ops import sort

    cfg = build_config(args)
    result = sort(args.store, output=args.output, by=args.sort_by, cfg=cfg, branch=args.branch, message=args.message)
    _LOG.info(f"wrote {result.path} ({result.n_obs} x {result.n_vars})")
    if result.snapshot_id is not None:
        _LOG.info(f"icechunk snapshot {result.snapshot_id}")
    return 0


def add_append_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("append", help="Append another store's cells onto this one, in place")
    p.add_argument("store", metavar="STORE")
    p.add_argument("cells", metavar="CELLS")
    p.add_argument(
        "--drop-derived",
        action="store_true",
        help="consent up front to dropping derived obs-aligned elements (obsm/obsp/incompatible layers); "
        "without it you are prompted",
    )
    p.add_argument(
        "--extend-layers",
        action="store_true",
        help="extend eligible add-expr CSR layers in place instead of dropping them",
    )
    add_cpus_arg(p)
    add_branch_arg(p)
    add_message_arg(p)
    p.set_defaults(func=_run_append)


def _run_append(args: argparse.Namespace) -> int:
    from annizarr.ops import append, plan_append

    cfg = build_config(args)
    plan = plan_append(args.store, cells=args.cells)
    _LOG.info(f"append plan: {plan.n_new} new cell(s)")
    if plan.drop_obsm:
        _LOG.warning(f"would drop obsm {list(plan.drop_obsm)} (invalidated by appended cells)")
    if plan.drop_obsp:
        _LOG.warning(f"would drop obsp {list(plan.drop_obsp)} (invalidated by appended cells)")
    if plan.drop_layers:
        _LOG.warning(f"would drop layers {list(plan.drop_layers)} (invalidated by appended cells)")
    if plan.extendable_layers:
        verb = "extend in place" if args.extend_layers else "drop unless --extend-layers"
        _LOG.warning(f"layers {list(plan.extendable_layers)} eligible to {verb}")
    if plan.n_duplicate_names:
        _LOG.warning(f"appending would introduce {plan.n_duplicate_names} duplicate obs name(s)")
    for note in plan.notes:
        _LOG.warning(note)

    # with --extend-layers those layers are spared, so only genuinely non-extendable
    # drops still require consent
    would_drop = bool(plan.drops(extend_layers=args.extend_layers))

    drop_derived = bool(args.drop_derived)
    if would_drop and not drop_derived:
        if sys.stdin.isatty():
            sys.stderr.write("Proceed? [y/N] ")
            sys.stderr.flush()
            answer = sys.stdin.readline().strip().lower()
            if answer not in ("y", "yes"):
                _LOG.info("aborted")
                return 1
            drop_derived = True
        else:
            _LOG.error("error: append would drop derived elements; pass --drop-derived to confirm")
            return 1

    result = append(
        args.store,
        cells=args.cells,
        drop_derived=drop_derived,
        extend_layers=args.extend_layers,
        cfg=cfg,
        branch=args.branch,
        message=args.message,
    )
    _LOG.info(f"updated {result.path} ({result.n_obs} x {result.n_vars})")
    if result.snapshot_id is not None:
        _LOG.info(f"icechunk snapshot {result.snapshot_id}")
    return 0
