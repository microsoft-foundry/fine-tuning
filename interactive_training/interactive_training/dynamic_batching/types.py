"""Core value types for the dynamic-batching API.

``Allocation`` is the tri-state result of a strategy's roll-side
decision (A-axis); ``Selection`` is the per-group result of its
train-side decision (B-axis). Both are deliberately tiny, immutable-ish
data carriers: all behaviour lives in the strategy methods, all mutable
state lives in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True)
class Allocation:
    """Tri-state result of ``DynamicBatchStrategy.allocate``.

    Use the three canonical singletons, never guess ``kind``:

    * ``Allocation.MORE``   -- re-roll this instance (needs more signal).
    * ``Allocation.DROP``   -- discard this group (coarse group filter).
    * ``Allocation.ADMIT``  -- admit this group into the batch.

    The loop reads only ``is_more`` / ``is_drop`` (admit is the default
    fall-through). There is deliberately no per-group "credit" weight:
    the production loop gates a batch as full on group count, not on an
    accumulated allocation weight, so a credit number would be computed
    and never read. If a future scheduler needs a weight budget it can
    carry that itself without widening this type.
    """

    kind: str  # "more" | "drop" | "admit"

    MORE: ClassVar["Allocation"]
    DROP: ClassVar["Allocation"]
    ADMIT: ClassVar["Allocation"]

    @property
    def is_more(self) -> bool:
        return self.kind == "more"

    @property
    def is_drop(self) -> bool:
        return self.kind == "drop"

    @property
    def is_admit(self) -> bool:
        return self.kind == "admit"


# Singletons. Assigned after the class body because they are instances of
# the class itself. ``frozen=True`` only blocks *instance* mutation, so
# binding class attributes here is fine.
Allocation.MORE = Allocation(kind="more")
Allocation.DROP = Allocation(kind="drop")
Allocation.ADMIT = Allocation(kind="admit")


@dataclass
class Selection:
    """Result of ``DynamicBatchStrategy.select`` for a single group.

    ``kept_indices`` index into the input group's samples. ``weights[k]``
    is the per-survivor importance weight aligned with ``kept_indices``
    (``1.0`` when the subset is deterministic / no reweighting).
    ``advantages`` is the per-survivor advantage tensor, with baseline,
    importance weight, and any objective scale already folded in, ready
    to hand to ``assemble_training_data``.
    ``token_advantage_adjustments`` optionally supplies additive credit for
    each selected trajectory's action tokens, shaped as
    ``[survivor][transition][action_token]``. The training consumer computes
    ``final_advantage = trajectory_advantage + token_adjustment``. Strategies
    that use only trajectory-level credit leave it as ``None``.

    The surviving set may be SMALLER than the input group -- that is the
    B-axis shrink, and it is why ``select`` returns this struct rather
    than a bare ``list[float]``.
    """

    kept_indices: list[int]
    weights: list[float]
    advantages: torch.Tensor
    token_advantage_adjustments: list[list[list[float]]] | None = None
