"""Phase-2 tests for the A-axis re-roll (Allocation.MORE) seam.

``do_group_rollout_and_filter_constant_reward`` grows a prompt's pool by
re-rolling whole sampling-group-sized waves while the strategy returns
``MORE`` and the pool stays under ``max_rolls_per_group``. These tests
drive that function with a mocked ``do_group_rollout`` so each wave's size
and reward/length signal is controllable, and count the rollout calls to
prove the re-roll actually fires (and stops).

Covered: Pilot-Commit commits to its target, MORE-without-a-cap collapses
to a single wave, and a settled DROP after allocation still filters the
group.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import interactive_training.rl.train as T
from interactive_training.dynamic_batching.strategy import (
    FixedStrategy,
    PilotCommitStrategy,
)
from interactive_training.rl.types import AutonomousRolloutGroupBuilder, EnvGroupBuilder

from tests.test_train_seams import _group


def _env_group_builder():
    # Python 3.11 protocol checks use hasattr: an unrestricted MagicMock
    # invents run_autonomous_rollout and takes the wrong dispatch branch.
    builder = MagicMock(spec_set=EnvGroupBuilder)
    assert not hasattr(builder, "run_autonomous_rollout")
    assert not isinstance(builder, AutonomousRolloutGroupBuilder)
    return builder


def _run_rollout(
    monkeypatch,
    *,
    waves,
    strategy,
    max_rolls_per_group,
    do_remove=False,
    seed=None,
    prefilter_rewards=None,
):
    """Drive the rollout function with a mocked wave generator.

    ``waves`` is a list of (rewards, lengths) tuples; each successive call to
    the mocked ``do_group_rollout`` returns the next wave as a group. If more
    waves are requested than provided, the last one repeats (so a strategy
    that keeps asking for MORE hits the cap deterministically).

    Returns (result_group_or_None, n_rollout_calls).
    """
    calls = {"n": 0}

    async def _fake_do_group_rollout(env_group_builder, policy):
        i = min(calls["n"], len(waves) - 1)
        rewards, lengths = waves[i]
        calls["n"] += 1
        return _group(rewards, lengths)

    monkeypatch.setattr(T, "do_group_rollout", _fake_do_group_rollout)
    monkeypatch.setattr(T, "SessionTokenCompleter", lambda *a, **k: MagicMock())

    result = asyncio.run(
        T.do_group_rollout_and_filter_constant_reward(
            sampling_client=MagicMock(),
            env_group_builder=_env_group_builder(),
            max_tokens=8,
            temperature=1.0,
            do_remove_constant_reward_groups=do_remove,
            enable_logging=False,
            seed=seed,
            strategy=strategy,
            max_rolls_per_group=max_rolls_per_group,
            on_prefilter_group=(
                None if prefilter_rewards is None else prefilter_rewards.add
            ),
        )
    )
    return result, calls["n"]


# ---------------------------------------------------------------------------
# Pilot-Commit: re-roll until the target pool size, then admit
# ---------------------------------------------------------------------------


def test_pilot_commit_commits_to_target(monkeypatch):
    # pilot=2 (wave size), commit=2 -> target pool 4. In-band p_hat=0.5.
    strategy = PilotCommitStrategy(
        p_lower=0.125, p_upper=0.75, pilot_rollouts=2, commit_rollouts=2
    )
    result, n_calls = _run_rollout(
        monkeypatch,
        waves=[([1.0, 0.0], [3, 3])],  # every wave: 2 samples, p_hat=0.5, in-band
        strategy=strategy,
        max_rolls_per_group=4,
    )
    assert result is not None
    # Wave 1 (n=2, MORE) + wave 2 (n=4, ADMIT) => 2 rollout calls, pool of 4.
    assert n_calls == 2
    assert len(result.trajectories_G) == 4


def test_reroll_assigns_stable_distinct_selection_seeds(monkeypatch):
    strategy = PilotCommitStrategy(
        p_lower=0.125, p_upper=0.75, pilot_rollouts=2, commit_rollouts=2
    )

    result, _ = _run_rollout(
        monkeypatch,
        waves=[([1.0, 0.0], [3, 3])],
        strategy=strategy,
        max_rolls_per_group=4,
        seed=500,
    )

    assert result is not None
    assert [trajectory.selection_seed for trajectory in result.trajectories_G] == [
        500,
        501,
        502,
        503,
    ]


def test_reroll_uses_latest_dynamic_batching_metadata(monkeypatch):
    calls = {"n": 0}

    async def _fake_do_group_rollout(env_group_builder, policy):
        calls["n"] += 1
        group = _group([1.0, 0.0], [3, 3])
        group.dynamic_batching_metadata = {
            "wave": calls["n"],
            "cumulative_count": 2 * calls["n"],
        }
        return group

    monkeypatch.setattr(T, "do_group_rollout", _fake_do_group_rollout)
    monkeypatch.setattr(T, "SessionTokenCompleter", lambda *a, **k: MagicMock())
    strategy = PilotCommitStrategy(
        p_lower=0.125, p_upper=0.75, pilot_rollouts=2, commit_rollouts=2
    )

    result = asyncio.run(
        T.do_group_rollout_and_filter_constant_reward(
            sampling_client=MagicMock(),
            env_group_builder=_env_group_builder(),
            max_tokens=8,
            temperature=1.0,
            do_remove_constant_reward_groups=False,
            enable_logging=False,
            strategy=strategy,
            max_rolls_per_group=4,
        )
    )

    assert result is not None
    assert result.dynamic_batching_metadata == {
        "wave": 2,
        "cumulative_count": 4,
    }


def test_pilot_commit_out_of_band_drops(monkeypatch):
    # p_hat=1.0 (all correct) is above p_upper -> DROP at the pilot wave.
    strategy = PilotCommitStrategy(
        p_lower=0.125, p_upper=0.75, pilot_rollouts=2, commit_rollouts=2
    )
    result, n_calls = _run_rollout(
        monkeypatch,
        waves=[([1.0, 1.0], [3, 3])],
        strategy=strategy,
        max_rolls_per_group=4,
    )
    assert result is None
    # No re-roll: pilot wave is out of band, so only one call.
    assert n_calls == 1


def test_prefilter_rewards_include_dropped_groups(monkeypatch):
    accumulator = T.PrefilterRewardAccumulator()
    strategy = PilotCommitStrategy(
        p_lower=0.125, p_upper=0.75, pilot_rollouts=2, commit_rollouts=2
    )

    result, _ = _run_rollout(
        monkeypatch,
        waves=[([1.0, 1.0], [3, 3])],
        strategy=strategy,
        max_rolls_per_group=4,
        prefilter_rewards=accumulator,
    )

    assert result is None
    assert accumulator.metrics() == {
        "sampling/pre_filter/reward/total": 1.0,
        "sampling/pre_filter/reward/group_mean": 1.0,
        "sampling/pre_filter/total_groups": 1,
        "sampling/pre_filter/total_episodes": 2,
        "sampling/pre_filter/by_group/frac_mixed": 0.0,
        "sampling/pre_filter/by_group/frac_all_good": 1.0,
        "sampling/pre_filter/by_group/frac_all_bad": 0.0,
    }


# ---------------------------------------------------------------------------
# MORE without a cap collapses to a single wave (backward compatible)
# ---------------------------------------------------------------------------


def test_more_without_cap_is_single_wave(monkeypatch):
    # Pilot-Commit would ask for MORE, but max_rolls_per_group=None disables
    # re-roll, so the pool stays at the single initial wave.
    strategy = PilotCommitStrategy(
        p_lower=0.125, p_upper=0.75, pilot_rollouts=2, commit_rollouts=2
    )
    result, n_calls = _run_rollout(
        monkeypatch,
        waves=[([1.0, 0.0], [3, 3])],  # in-band p_hat=0.5 -> MORE
        strategy=strategy,
        max_rolls_per_group=None,
    )
    assert result is not None
    assert n_calls == 1
    assert len(result.trajectories_G) == 2


def test_fixed_strategy_single_wave_admits(monkeypatch):
    result, n_calls = _run_rollout(
        monkeypatch,
        waves=[([0.9, 0.1], [3, 5])],
        strategy=FixedStrategy(),
        max_rolls_per_group=8,
    )
    assert result is not None
    assert n_calls == 1
    assert len(result.trajectories_G) == 2
