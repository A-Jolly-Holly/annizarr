"""Regenerate docs/cli.md from the CLI's argparse definitions.

    python docs/gen_cli_reference.py

tests/test_docs.py fails when the checked-in page is stale.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMANDS = ("convert", "rechunk", "sort", "append", "add-expr")
HEADER = """\
# CLI reference

Generated from the argparse definitions by `python docs/gen_cli_reference.py`; `annizarr <command> --help`
prints the same text. `anz` is an alias for `annizarr`.

Global options go before the command: `-v`/`--verbose` (debug logging), `-q`/`--quiet` (warnings only),
`--version`.
"""


def render() -> str:
    """Return the Markdown page: one section per subcommand with its --help text verbatim."""
    from annizarr._cli import _build_parser

    previous = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "100"  # argparse wraps to the terminal width; pin it so the page is stable
    try:
        parser = _build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        summaries = {a.dest: a.help or "" for a in sub._choices_actions}
        parts = [HEADER]
        for name in COMMANDS:
            parts.append(f"## {name}\n\n{summaries[name]}\n\n```text\n{sub.choices[name].format_help()}```\n")
        return "\n".join(parts)
    finally:
        if previous is None:
            del os.environ["COLUMNS"]
        else:
            os.environ["COLUMNS"] = previous


if __name__ == "__main__":
    (HERE / "cli.md").write_text(render(), encoding="utf-8")
    print(f"wrote {HERE / 'cli.md'}")
