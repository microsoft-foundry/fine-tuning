"""Generic dynamic-batching API for Interactive Training RL training.

This package factors the batch-construction logic shared by PODS,
Pilot-Commit, and the GRPO/DAPO baselines behind two named primitives on
a single stateless ``DynamicBatchStrategy`` object:

* ``allocate(group)``  -- roll-side (A-axis): grow the rollout pool
  (``MORE`` / ``DROP`` / ``ADMIT``).
* ``select(batch)``    -- train-side (B-axis): shrink to the informative
  subset and assign advantages.

There is no batch-full gate on the strategy: the production loop decides
fullness on group count, so that concern stays in the loop.

The strategy is a pure function of rewards; all mutable bookkeeping
(in-flight pools, staleness, the dataset cursor) lives in the
scheduler/loop that drives it. See the "Custom batching strategies"
section of ``docs/training.md`` for a usage guide.
"""

from interactive_training.dynamic_batching.strategy import (
    DapoStrategy as DapoStrategy,
    DynamicBatchStrategy as DynamicBatchStrategy,
    FixedStrategy as FixedStrategy,
    PilotCommitStrategy as PilotCommitStrategy,
    PodsStrategy as PodsStrategy,
)
from interactive_training.dynamic_batching.factory import (
    STRATEGY_NAMES as STRATEGY_NAMES,
    build_strategy as build_strategy,
)
from interactive_training.dynamic_batching.types import (
    Allocation as Allocation,
    Selection as Selection,
)
