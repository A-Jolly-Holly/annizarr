from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_reference_is_current() -> None:
    """docs/cli.md is generated from the argparse parsers; fail loudly when it drifts."""
    spec = importlib.util.spec_from_file_location("gen_cli_reference", ROOT / "docs" / "gen_cli_reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.render()
    actual = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    assert actual == expected, "docs/cli.md is stale: run `python docs/gen_cli_reference.py`"
