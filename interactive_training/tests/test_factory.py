"""Tests for the CLI-friendly build_strategy factory."""

from __future__ import annotations

import pytest

from interactive_training.dynamic_batching import build_strategy
from interactive_training.dynamic_batching.strategy import (
    DapoStrategy,
    FixedStrategy,
    PilotCommitStrategy,
    PodsStrategy,
)


def test_none_and_baseline_yield_no_strategy():
    assert build_strategy(None) is None
    assert build_strategy("none") is None
    assert build_strategy("baseline") is None
    assert build_strategy("  None  ") is None


def test_fixed_and_dapo():
    assert isinstance(build_strategy("fixed"), FixedStrategy)
    assert isinstance(build_strategy("grpo"), FixedStrategy)
    assert isinstance(build_strategy("dapo"), DapoStrategy)


def test_pods_uses_explicit_keep_or_half_group():
    s = build_strategy("pods", pods_keep=3)
    assert isinstance(s, PodsStrategy) and s.keep == 3
    # Falls back to half the group size.
    s2 = build_strategy("pods", group_size=8)
    assert s2.keep == 4
    # No keep and no group_size -> fail fast.
    with pytest.raises(ValueError, match="pods_keep or group_size"):
        build_strategy("pods")


def test_pilot_commit_pilot_defaults_to_group_size():
    s = build_strategy("pilot_commit", group_size=8, commit_rollouts=8)
    assert isinstance(s, PilotCommitStrategy)
    assert s.pilot_rollouts == 8 and s.commit_rollouts == 8


def test_unknown_name_fails_fast_short():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("nope")


def test_pilot_commit_pilot_rollouts_and_fail_fast():
    # Explicit pilot_rollouts wins over group_size.
    s2 = build_strategy("pilot_commit", pilot_rollouts=4, group_size=8)
    assert s2.pilot_rollouts == 4
    # Neither -> fail fast.
    with pytest.raises(ValueError, match="pilot_rollouts or group_size"):
        build_strategy("pilot_commit")


def test_pilot_commit_band_params():
    s = build_strategy(
        "pilot_commit", group_size=8, pilot_p_lower=0.2, pilot_p_upper=0.6
    )
    assert s.p_lower == 0.2 and s.p_upper == 0.6


def test_unknown_name_fails_fast():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("supernova")  # not a public name
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("typo_pods")


def test_case_and_whitespace_insensitive():
    assert isinstance(build_strategy("  DAPO "), DapoStrategy)
    assert isinstance(build_strategy("Pods", group_size=4), PodsStrategy)
