from __future__ import annotations

import argparse

from annizarr._core._config import DENSE_CHUNK, AppConfig, ChunkConfig, apply_cli_overrides


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", default=None, help="replace an existing output")


def add_consolidate_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        default=None,
        help="consolidate zarr metadata after writing",
    )


def add_ic_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ic", action="store_true", default=None, help="write through an Icechunk repository")


def add_branch_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--branch",
        default=None,
        metavar="B",
        help="Icechunk branch to read/write, created off main if missing (default: main); ignored for plain zarr",
    )


def add_message_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--message",
        default=None,
        metavar="MSG",
        help="Icechunk commit message (default: an auto-generated one naming the op); ignored for plain zarr",
    )


def add_cpus_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpus", type=int, help="parallel band workers (default: all cores)")


def add_chunk_args(parser: argparse.ArgumentParser) -> None:
    nnz = ChunkConfig().nnz_chunk
    parser.add_argument(
        "--row-chunk",
        type=int,
        metavar="N",
        help=f"rows per chunk: exact for dense output (default: {DENSE_CHUNK}); for csr, about N cells' worth of "
        f"nonzeros per chunk (default: {nnz} nonzeros). Not for csc.",
    )
    parser.add_argument(
        "--col-chunk",
        type=int,
        metavar="N",
        help=f"columns per chunk: exact for dense output (default: {DENSE_CHUNK}); for csc, about N genes' worth of "
        f"nonzeros per chunk (default: {nnz} nonzeros). Not for csr.",
    )


def add_autoshard_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auto-shard",
        dest="auto_shard",
        action="store_true",
        default=None,
        help="shard the anndata-written elements and the 1-D sparse arrays with zarr's automatic shard shape "
        "(default: off)",
    )


def reject_off_axis_chunks(parser: argparse.ArgumentParser, args: argparse.Namespace, layout: str) -> None:
    # convert/add-expr know the output layout up front, so a chunk flag for the wrong sparse axis
    # is a usage error here; rechunk learns the layout from the store and only warns at write time
    if layout == "csr" and getattr(args, "col_chunk", None) is not None:
        parser.error("--col-chunk does not apply to csr output (chunks follow rows); use --row-chunk")
    if layout == "csc" and getattr(args, "row_chunk", None) is not None:
        parser.error("--row-chunk does not apply to csc output (chunks follow columns); use --col-chunk")


def build_config(args: argparse.Namespace) -> AppConfig:
    # overlays whichever override flags the calling subcommand's parser defined onto the
    # defaults; flags absent from args are simply skipped
    return apply_cli_overrides(
        AppConfig(),
        overwrite=getattr(args, "overwrite", None),
        consolidate_metadata=getattr(args, "consolidate_metadata", None),
        layout=getattr(args, "layout", None),
        row_chunk=getattr(args, "row_chunk", None),
        col_chunk=getattr(args, "col_chunk", None),
        auto_shard=getattr(args, "auto_shard", None),
        cpus=getattr(args, "cpus", None),
        lazy=getattr(args, "lazy", None),
        backend=("icechunk" if getattr(args, "ic", False) else None),
        sort_by=getattr(args, "sort_by", None),
        obs_columns=getattr(args, "obs_columns", None),
    )
