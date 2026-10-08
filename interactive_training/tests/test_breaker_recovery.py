"""Integration test for the failure-rate breaker recovery loop in
``do_group_rollout_and_filter_constant_reward``.

When the circuit breaker trips, the wrapper should reset the tracker, back off,
and retry the group instead of aborting the run — bounded by
``rollout_failure_max_resets``.
"""

import asyncio
from types import SimpleNamespace

import pytest

import interactive_training.rl.train as train_mod
from interactive_training.rl.rollouts import (
    RolloutFailureRateExceeded,
    RolloutFailureTracker,
    RolloutRetryPolicy,
)
from interactive_training.rl.types import TrajectoryGroup


def _make_group() -> TrajectoryGroup:
    return TrajectoryGroup(trajectories_G=[], final_rewards_G=[], metrics_G=[{}])


def _install_fake_rollout(monkeypatch, trips_before_success: int):
    calls = {"n": 0}

    async def fake_do_group_rollout(
        env_group_builder, policy, retry_policy=None, failure_tracker=None
    ):
        calls["n"] += 1
        if calls["n"] <= trips_before_success:
            raise RolloutFailureRateExceeded("simulated breaker trip")
        return _make_group()

    monkeypatch.setattr(train_mod, "do_group_rollout", fake_do_group_rollout)
    return calls


async def _run_wrapper(tracker):
    return await train_mod.do_group_rollout_and_filter_constant_reward(
        sampling_client=SimpleNamespace(),
        env_group_builder=SimpleNamespace(),
        max_tokens=16,
        temperature=1.0,
        do_remove_constant_reward_groups=False,
        enable_logging=False,
        retry_policy=RolloutRetryPolicy(
            max_retries_per_trajectory=1, max_extra_attempts_per_group=1
        ),
        failure_tracker=tracker,
    )


def test_breaker_recovers_and_retries_within_budget(monkeypatch):
    calls = _install_fake_rollout(monkeypatch, trips_before_success=2)
    tracker = RolloutFailureTracker(
        window_size=10, min_attempts=10, abort_rate=0.20,
        max_resets=2, reset_backoff_sec=0.0,
    )

    group = asyncio.run(_run_wrapper(tracker))

    assert group is not None            # run continued
    assert calls["n"] == 3              # 2 trips + 1 success
    assert tracker.resets_used == 2


def test_breaker_aborts_when_budget_exhausted(monkeypatch):
    calls = _install_fake_rollout(monkeypatch, trips_before_success=5)
    tracker = RolloutFailureTracker(
        window_size=10, min_attempts=10, abort_rate=0.20,
        max_resets=1, reset_backoff_sec=0.0,
    )

    with pytest.raises(RolloutFailureRateExceeded):
        asyncio.run(_run_wrapper(tracker))

    # One recovery consumed, second trip exhausts the budget → abort.
    assert calls["n"] == 2
    assert tracker.resets_used == 1
