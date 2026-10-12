from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from annizarr._core._config import AppConfig
from annizarr._sources._h5ad import close_lazy_if_needed
from annizarr.typing import PathLike

if TYPE_CHECKING:
    from anndata import AnnData

    from annizarr._sources._readers import Reader

__all__ = ["Loader", "Sniffer", "Source"]


@dataclass(frozen=True, slots=True)
class Source:
    """An opened conversion input, ready to write.

    ``x_reader`` is set when X streams from a reader of its own rather than ``adata.X``
    (a lazy 10x file); ``adata`` then carries obs and var only.
    """

    adata: AnnData
    kind: str
    lazy: bool
    warnings: tuple[str, ...]
    x_reader: Reader | None = None

    @property
    def x(self) -> Any:
        return self.adata.X if self.x_reader is None else self.x_reader

    def close(self) -> None:
        """Close a backed h5 handle or reader; a no-op for an eager or in-memory source."""
        close_lazy_if_needed(self.adata)
        if self.x_reader is not None:
            self.x_reader.close()


# Loader/Sniffer are plain Callable aliases; a custom kind is not restricted to the
# builtin kinds below, so the return type here is documentation, not an enforced bound.
type Loader = Callable[[PathLike, AppConfig], Source]
type Sniffer = Callable[[Path], str | None]
