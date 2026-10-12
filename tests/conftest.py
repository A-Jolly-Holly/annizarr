from __future__ import annotations

import logging

import pytest
from anndata import AnnData

from helpers import make_adata


@pytest.fixture
def adata() -> AnnData:
    return make_adata()


@pytest.fixture(autouse=True)
def _annizarr_warning_capture(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="annizarr")
