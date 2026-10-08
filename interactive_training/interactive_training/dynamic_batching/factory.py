"""CLI-friendly factory for constructing dynamic-batching strategies.

Recipes expose strategy selection as flat string / scalar overrides (so the
choice can ride ``key=value`` CLI args and benchmark ``recipe_overrides``);
this module turns a strategy *name* plus scalar params into the matching
``DynamicBatchStrategy`` object. ``None`` (or ``"none"``) yields ``None`` --
the vanilla training path.
"""

from __future__ import annotations

from interactive_training.dynamic_batching.strategy import (
    DapoStrategy,
    DynamicBatchStrategy,
    FixedStrategy,
    PilotCommitStrategy,
    PodsStrategy,
)

# Public names accepted on the CLI (lower-cased, whitespace-stripped). Aliases
# map to a canonical builder below.
STRATEGY_NAMES = (
    "fixed",
    "dapo",
    "pods",
    "pilot_commit",
)


def build_strategy(
    name: str | None,
    *,
    group_size: int | None = None,
    pods_keep: int | None = None,
    pilot_p_lower: float = 0.125,
    pilot_p_upper: float = 0.75,
    pilot_rollouts: int | None = None,
    commit_rollouts: int = 0,
) -> DynamicBatchStrategy | None:
    """Build a strategy from a name + scalar params (or ``None`` for baseline).

    ``group_size`` is the sampling group size; it supplies sensible defaults
    for methods whose parameters are tied to it (PODS keeps half the group;
    Pilot-Commit's pilot wave must equal the group size).

    Raises ``ValueError`` on an unknown name so a benchmark typo fails fast
    instead of silently running the wrong method.
    """
    if name is None:
        return None
    key = name.strip().lower()
    if key in ("", "none", "baseline"):
        return None
    if key in ("fixed", "grpo"):
        return FixedStrategy()
    if key == "dapo":
        return DapoStrategy()
    if key == "pods":
        if pods_keep is not None:
            keep = int(pods_keep)
        elif group_size is not None:
            keep = max(1, int(group_size) // 2)
        else:
            raise ValueError("pods requires pods_keep or group_size to size the subset")
        return PodsStrategy(keep_per_group=keep)
    if key in ("pilot_commit", "pilotcommit", "pilot"):
        pilot = pilot_rollouts if pilot_rollouts is not None else group_size
        if pilot is None:
            raise ValueError(
                "pilot_commit requires pilot_rollouts or group_size "
                "(the pilot wave must equal the sampling group size)"
            )
        return PilotCommitStrategy(
            p_lower=float(pilot_p_lower),
            p_upper=float(pilot_p_upper),
            pilot_rollouts=int(pilot),
            commit_rollouts=int(commit_rollouts),
        )
    raise ValueError(
        f"unknown strategy {name!r}; valid names: {', '.join(STRATEGY_NAMES)} (or none)"
    )
