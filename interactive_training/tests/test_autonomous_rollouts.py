"""Focused tests for cookbook-managed and autonomous rollout dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import get_type_hints

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

import interactive_training.rl.train as train_module
from interactive_training.completers import TokensWithLogprobs
from interactive_training.rl.data_processing import trajectory_to_data
from interactive_training.rl.rollouts import RolloutFailureTracker, RolloutRetryPolicy
from interactive_training.rl.types import (
    AutonomousRolloutGroupBuilder,
    Env,
    EnvGroupBuilder,
    RolloutSamplingOptions,
    SamplingCheckpoint,
    StepResult,
    Trajectory,
    TrajectoryGroup,
    Transition,
)


def _ob(tokens: list[int]) -> ModelInput:
    return ModelInput(chunks=[ModelInputChunk(tokens=tokens)])


def _trajectory_group(rewards: list[float]) -> TrajectoryGroup:
    trajectory = Trajectory(
        transitions=[
            Transition(
                ob=_ob([1, 2]),
                ac=TokensWithLogprobs(tokens=[3], maybe_logprobs=[-0.3]),
                reward=0.0,
                episode_done=True,
            )
        ],
        final_ob=_ob([1, 2, 3]),
    )
    return TrajectoryGroup(
        trajectories_G=[trajectory for _ in rewards],
        final_rewards_G=rewards,
        metrics_G=[{} for _ in rewards],
    )


class _AutonomousBuilder:
    def __init__(self, group: TrajectoryGroup | None):
        self.group = group
        self.calls: list[tuple[SamplingCheckpoint, RolloutSamplingOptions]] = []
        self.make_envs_calls = 0

    async def run_autonomous_rollout(self, sampling_checkpoint, sampling_options):
        self.calls.append((sampling_checkpoint, sampling_options))
        return self.group

    async def make_envs(self):
        self.make_envs_calls += 1
        raise AssertionError("autonomous rollout must not create cookbook environments")

    def logging_tags(self) -> list[str]:
        return ["autonomous"]


class _AutonomousSamplingClient:
    def __init__(self):
        self.sampling_checkpoint = SamplingCheckpoint("session-1", "checkpoint-2")
        self.sample_calls = 0

    async def sample_async(self, *args, **kwargs):
        self.sample_calls += 1
        raise AssertionError("autonomous rollout must not sample through the cookbook policy")


@pytest.mark.parametrize("rewards", [[0.0, 1.0], [1.0, 1.0]])
def test_autonomous_prefilter_callback_runs_before_drop(rewards):
    group = _trajectory_group(rewards)
    observed = []
    result = asyncio.run(train_module.do_group_rollout_and_filter_constant_reward(
        _AutonomousSamplingClient(), _AutonomousBuilder(group),
        max_tokens=8, temperature=1.0, do_remove_constant_reward_groups=True,
        on_prefilter_group=observed.append,
    ))
    assert observed == [group]
    assert result is (group if len(set(rewards)) > 1 else None)


def test_autonomous_rollout_protocol_allows_dropped_groups():
    assert get_type_hints(
        AutonomousRolloutGroupBuilder.run_autonomous_rollout
    )["return"] == TrajectoryGroup | None

    observed = []
    result = asyncio.run(train_module.do_group_rollout_and_filter_constant_reward(
        _AutonomousSamplingClient(),
        _AutonomousBuilder(None),
        max_tokens=8,
        temperature=1.0,
        do_remove_constant_reward_groups=False,
        on_prefilter_group=observed.append,
    ))

    assert result is None
    assert observed == []


def test_autonomous_rollout_bypasses_cookbook_policy_and_preserves_group(monkeypatch):
    group = _trajectory_group([0.0, 1.0])
    builder = _AutonomousBuilder(group)
    client = _AutonomousSamplingClient()

    def unexpected_policy(*args, **kwargs):
        raise AssertionError("autonomous rollout must not construct SessionTokenCompleter")

    async def unexpected_group_rollout(*args, **kwargs):
        raise AssertionError("autonomous rollout must not call do_group_rollout")

    monkeypatch.setattr(train_module, "SessionTokenCompleter", unexpected_policy)
    monkeypatch.setattr(train_module, "do_group_rollout", unexpected_group_rollout)

    result = asyncio.run(
        train_module.do_group_rollout_and_filter_constant_reward(
            client,
            builder,
            max_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            seed=123,
            sample_timeout_sec=15.0,
            do_remove_constant_reward_groups=False,
        )
    )

    assert isinstance(builder, AutonomousRolloutGroupBuilder)
    assert result is group
    assert builder.calls == [
        (
            SamplingCheckpoint("session-1", "checkpoint-2"),
            RolloutSamplingOptions(
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                seed=123,
                sample_timeout_sec=15.0,
            ),
        )
    ]
    assert client.sample_calls == 0
    assert builder.make_envs_calls == 0


def test_autonomous_rollout_keeps_constant_reward_filter_and_rejects_reroll_settings():
    builder = _AutonomousBuilder(_trajectory_group([1.0, 1.0]))
    client = _AutonomousSamplingClient()

    assert (
        asyncio.run(
            train_module.do_group_rollout_and_filter_constant_reward(
                client,
                builder,
                max_tokens=8,
                temperature=1.0,
                do_remove_constant_reward_groups=True,
            )
        )
        is None
    )

    with pytest.raises(ValueError, match="does not support rollout strategy"):
        asyncio.run(
            train_module.do_group_rollout_and_filter_constant_reward(
                client,
                builder,
                max_tokens=8,
                temperature=1.0,
                do_remove_constant_reward_groups=False,
                strategy=object(),
            )
        )

    with pytest.raises(ValueError, match="retry_policy"):
        asyncio.run(
            train_module.do_group_rollout_and_filter_constant_reward(
                client,
                builder,
                max_tokens=8,
                temperature=1.0,
                do_remove_constant_reward_groups=False,
                retry_policy=RolloutRetryPolicy(max_retries_per_trajectory=1),
            )
        )

    retry_compatible_result = asyncio.run(
        train_module.do_group_rollout_and_filter_constant_reward(
            client,
            builder,
            max_tokens=8,
            temperature=1.0,
            do_remove_constant_reward_groups=False,
            sample_timeout_sec=10.0,
            retry_policy=RolloutRetryPolicy(),
            failure_tracker=RolloutFailureTracker(),
        )
    )
    assert retry_compatible_result is builder.group


class _OrdinaryEnv(Env):
    def __init__(self):
        self.observation = _ob([7])

    async def initial_observation(self):
        return self.observation, []

    async def step(self, action):
        return StepResult(
            reward=0.25,
            episode_done=True,
            next_observation=_ob([7, *action]),
            next_stop_condition=[],
        )


class _OrdinaryBuilder(EnvGroupBuilder):
    def __init__(self):
        self.make_envs_calls = 0
        self.compute_group_rewards_calls = 0

    async def make_envs(self):
        self.make_envs_calls += 1
        return [_OrdinaryEnv()]

    async def compute_group_rewards(self, trajectory_group, env_group):
        self.compute_group_rewards_calls += 1
        return [(0.75, {})]


class _OrdinarySamplingClient:
    def __init__(self):
        self.sample_calls = 0

    async def sample_async(self, **kwargs):
        self.sample_calls += 1
        return SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[8], logprobs=[-0.8])]
        )


def test_env_group_builder_uses_existing_cookbook_rollout_path():
    builder = _OrdinaryBuilder()
    client = _OrdinarySamplingClient()

    result = asyncio.run(
        train_module.do_group_rollout_and_filter_constant_reward(
            client,
            builder,
            max_tokens=8,
            temperature=1.0,
            do_remove_constant_reward_groups=False,
        )
    )

    assert result is not None
    assert builder.make_envs_calls == 1
    assert builder.compute_group_rewards_calls == 1
    assert client.sample_calls == 1
    assert result.final_rewards_G == [0.75]
    assert result.trajectories_G[0].transitions[0].ac.tokens == [8]


def test_force_new_training_sequence_splits_an_otherwise_mergeable_transition():
    first_ob = _ob([1, 2])
    second_ob = _ob([1, 2, 3, 4, 5])
    trajectory = Trajectory(
        transitions=[
            Transition(
                ob=first_ob,
                ac=TokensWithLogprobs(tokens=[3, 4], maybe_logprobs=[-0.3, -0.4]),
                reward=0.0,
                episode_done=False,
            ),
            Transition(
                ob=second_ob,
                ac=TokensWithLogprobs(tokens=[6], maybe_logprobs=[-0.6]),
                reward=0.0,
                episode_done=True,
                force_new_training_sequence=True,
            ),
        ],
        final_ob=_ob([1, 2, 3, 4, 5, 6]),
    )

    data = trajectory_to_data(trajectory, traj_advantage=0.5)

    assert len(data) == 2
    assert data[0].model_input.chunks[0].tokens == [1, 2, 3]
    assert data[0].loss_fn_inputs["target_tokens"].data == [2, 3, 4]
    assert data[0].loss_fn_inputs["logprobs"].data == pytest.approx([0.0, -0.3, -0.4])
    assert data[0].loss_fn_inputs["mask"].data == [0.0, 1.0, 1.0]
    assert data[1].model_input.chunks[0].tokens == [1, 2, 3, 4, 5]
    assert data[1].loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5, 6]
    assert data[1].loss_fn_inputs["logprobs"].data == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, -0.6]
    )
    assert data[1].loss_fn_inputs["mask"].data == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_unflagged_mergeable_transitions_remain_one_training_sequence():
    trajectory = Trajectory(
        transitions=[
            Transition(
                ob=_ob([1, 2]),
                ac=TokensWithLogprobs(tokens=[3, 4], maybe_logprobs=[-0.3, -0.4]),
                reward=0.0,
                episode_done=False,
            ),
            Transition(
                ob=_ob([1, 2, 3, 4, 5]),
                ac=TokensWithLogprobs(tokens=[6], maybe_logprobs=[-0.6]),
                reward=0.0,
                episode_done=True,
            ),
        ],
        final_ob=_ob([1, 2, 3, 4, 5, 6]),
    )

    data = trajectory_to_data(trajectory, traj_advantage=0.5)

    assert len(data) == 1
    assert data[0].model_input.chunks[0].tokens == [1, 2, 3, 4, 5]
    assert data[0].loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5, 6]
    assert data[0].loss_fn_inputs["logprobs"].data == pytest.approx(
        [0.0, -0.3, -0.4, 0.0, -0.6]
    )
    assert data[0].loss_fn_inputs["mask"].data == [0.0, 1.0, 1.0, 0.0, 1.0]
