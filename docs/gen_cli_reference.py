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


class _PinnedFormatter(argparse.HelpFormatter):
    """Python 3.13's option formatting ("-m, --message MSG") on every version.

    3.12 prints "-m MSG, --message MSG" and aligns columns differently, which would make the
    generated page depend on the interpreter that produced it.
    """

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        return ", ".join(action.option_strings) + " " + self._format_args(action, default)


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
            sub.choices[name].formatter_class = _PinnedFormatter
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
