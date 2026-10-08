from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from interactive_training.dynamic_batching import build_strategy
from interactive_training.dynamic_batching.strategy import DynamicBatchStrategy, PodsStrategy


@dataclass(frozen=True)
class NerDynamicBatchingPolicy:
    """Default online signal-selection policy for graded exact-F1 rewards."""

    group_size: int = 8
    pods_keep: int = 4
    max_oversample_rounds: int = 3
    oversample_cushion: float = 1.2

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if not 2 <= self.pods_keep <= self.group_size:
            raise ValueError("pods_keep must be between 2 and group_size")
        if self.max_oversample_rounds < 1:
            raise ValueError("max_oversample_rounds must be positive")
        if self.oversample_cushion < 1.0:
            raise ValueError("oversample_cushion must be at least 1.0")

    def build_strategy(self) -> DynamicBatchStrategy:
        strategy = build_strategy(
            "pods",
            group_size=self.group_size,
            pods_keep=self.pods_keep,
        )
        assert isinstance(strategy, PodsStrategy)
        return strategy

    def training_overrides(self) -> dict[str, Any]:
        """Return the exact shared-trainer settings for this policy."""
        return {
            "strategy": self.build_strategy(),
            "dynamic_sampling": True,
            "remove_constant_reward_groups": True,
            "max_oversample_rounds": self.max_oversample_rounds,
            "oversample_cushion": self.oversample_cushion,
            "refill_on_drop": False,
            "max_rolls_per_group": None,
        }