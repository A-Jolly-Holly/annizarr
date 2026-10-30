from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AppendPlan", "OpResult"]


@dataclass(frozen=True, slots=True)
class OpResult:
    """Outcome of a completed write operation."""

    path: str
    n_obs: int
    n_vars: int
    snapshot_id: str | None


@dataclass(frozen=True, slots=True)
class AppendPlan:
    """Pure plan for :func:`~annizarr.ops._append.append`, computed from metadata only.

    ``obs_extra`` are obs columns in the cells but not the store (appended only with
    ``include_new_obs``), ``obs_filled`` columns in the store but not the cells (the appended rows
    get missing values), and ``obs_promoted`` store columns rewritten in their nullable encoding
    to hold a missing value. ``append_layers`` are the layers both stores carry, appended whatever
    ``drop_derived`` says; ``drop_layers`` those only the store carries.
    """

    n_new: int
    drop_obsm: tuple[str, ...]
    drop_obsp: tuple[str, ...]
    drop_layers: tuple[str, ...]
    append_layers: tuple[str, ...]
    n_duplicate_names: int
    notes: tuple[str, ...]
    obs_extra: tuple[str, ...] = ()
    obs_filled: tuple[str, ...] = ()
    obs_promoted: tuple[str, ...] = ()

    def drops(self) -> tuple[str, ...]:
        """Element paths (``obsm/<k>``, ``obsp/<k>``, ``layers/<k>``) the plan would drop."""
        return (
            *(f"obsm/{k}" for k in self.drop_obsm),
            *(f"obsp/{k}" for k in self.drop_obsp),
            *(f"layers/{k}" for k in self.drop_layers),
        )
