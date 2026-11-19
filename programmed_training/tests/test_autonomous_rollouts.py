"""Focused tests for cookbook-managed and autonomous rollout dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

import programmed_training.rl.train as train_module
from programmed_training.completers import TokensWithLogprobs
from programmed_training.rl.data_processing import trajectory_to_data
from programmed_training.rl.rollouts import RolloutFailureTracker, RolloutRetryPolicy
from programmed_training.rl.types import (
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


@pytest.mark.parametrize("skip_after_preparation", [True, False])
@pytest.mark.parametrize("resample", [True, False])
def test_sync_rollouts_and_evaluations_keep_the_sampled_policy_step(
    tmp_path, monkeypatch, skip_after_preparation, resample,
):
    class SamplingClient:
        def __init__(self, checkpoint_id):
            self.sampling_checkpoint = SamplingCheckpoint("session", checkpoint_id)

        def set_sampling_checkpoint_step(self, step):
            self.sampling_checkpoint = replace(self.sampling_checkpoint, policy_step=step)

    builders = [_AutonomousBuilder(_trajectory_group([0.0, 1.0])) for _ in range(4)]
    if not skip_after_preparation:
        builders[1].group = _trajectory_group([1.0, 1.0])
    dataset = SimpleNamespace(get_batch=lambda index: [builders[index]])

    async def recover(sample, batch_builders):
        assert len(batch_builders) == 1
        await sample()
        return await sample()

    config = train_module.Config(
        learning_rate=1e-4,
        dataset_builder=lambda: None,
        model_name="model",
        max_tokens=8,
        log_path=str(tmp_path),
        eval_every=1,
        save_every=0,
        num_groups_to_log=0,
        remove_constant_reward_groups=True,
        rollout_batch_recovery=recover if resample else None,
    )
    initial = SamplingClient("step0")
    saved_initial = initial.sampling_checkpoint
    evaluations = []
    saved_policy_steps = []
    final_loop_state = {}
    ml_logger = MagicMock()

    async def evaluate(_evaluators, sampling_client, _cfg, step):
        evaluations.append((step, sampling_client.sampling_checkpoint))
        return {}

    async def train_step(_cfg, step, *_args, current_sampling_client, loop_state_extra, **_kwargs):
        if step == 1:
            return current_sampling_client, {"train/skipped_no_valid_groups": 1}
        saved_policy_steps.append(loop_state_extra["policy_step"])
        return SamplingClient(f"step{step + 1}"), {}

    monkeypatch.setattr(
        train_module, "save_checkpoint_and_get_sampling_client",
        AsyncMock(return_value=(initial, {})),
    )
    monkeypatch.setattr(train_module, "run_evaluations_parallel", evaluate)
    monkeypatch.setattr(train_module, "do_train_step_and_get_sampling_client", train_step)

    progress = asyncio.run(train_module.do_sync_training(
        start_batch=0, end_batch=4, num_batches=4, cfg=config,
        training_client=MagicMock(), kl_reference_client=None,
        evaluators=[MagicMock()], dataset=dataset, ml_logger=ml_logger,
        tokenizer=MagicMock(), final_loop_state=final_loop_state,
    ))

    checkpoints = [builder.calls[0][0] for builder in builders]
    assert [checkpoint.policy_step for checkpoint in checkpoints] == [0, 1, 1, 2]
    assert [checkpoint.checkpoint_id for checkpoint in checkpoints] == [
        "step0", "step1", "step1", "step3",
    ]
    assert evaluations == [(step, checkpoint) for step, checkpoint in enumerate(checkpoints)]
    assert saved_initial.policy_step is None
    assert saved_policy_steps == [1, 2, 3]
    assert final_loop_state == {"policy_step": 3}
    assert progress == (4, 3)
    assert all(len(builder.calls) == (2 if resample else 1) for builder in builders)
    for call in ml_logger.log_metrics.call_args_list:
        metrics = call.args[0]
        if "sampling/pre_filter/total_groups" in metrics:
            assert metrics["sampling/pre_filter/total_groups"] == 1
            assert metrics["sampling/pre_filter/total_episodes"] == 2


@pytest.mark.parametrize("unsupported", [
    {"dynamic_sampling": True}, {"async_config": MagicMock()},
    {"stream_minibatch_config": MagicMock()}, {"strategy": MagicMock()},
    {"strategy": MagicMock(), "refill_on_drop": True},
])
def test_batch_recovery_rejects_unsupported_sampling_modes(tmp_path, unsupported):
    with pytest.raises(ValueError, match="rollout_batch_recovery requires"):
        train_module.Config(
            learning_rate=1e-4, dataset_builder=lambda: None, model_name="model",
            max_tokens=8, log_path=str(tmp_path), rollout_batch_recovery=AsyncMock(),
            **unsupported,
        )


@pytest.mark.parametrize(
    "start_batch,resume_step,expected,skipped",
    [(5, 2, 2, True), (5, None, None, True), (0, None, None, True),
     (5, 2, 2, False), (5, None, None, False), (0, 0, 0, False)],
)
def test_sync_resume_preserves_known_or_unknown_policy_step(
    tmp_path, monkeypatch, start_batch, resume_step, expected, skipped,
):
    class SamplingClient:
        def __init__(self):
            self.sampling_checkpoint = SamplingCheckpoint("session", "step0")

        def set_sampling_checkpoint_step(self, step):
            self.sampling_checkpoint = replace(self.sampling_checkpoint, policy_step=step)

    builder = _AutonomousBuilder(_trajectory_group([0.0, 1.0]))
    dataset = SimpleNamespace(get_batch=lambda _index: [builder])
    config = train_module.Config(
        learning_rate=1e-4, dataset_builder=lambda: None, model_name="model",
        max_tokens=8, log_path=str(tmp_path), eval_every=0, save_every=0,
        num_groups_to_log=0,
    )
    sampling_client = SamplingClient()
    monkeypatch.setattr(
        train_module, "save_checkpoint_and_get_sampling_client",
        AsyncMock(return_value=(sampling_client, {})),
    )
    monkeypatch.setattr(
        train_module, "do_train_step_and_get_sampling_client",
        AsyncMock(return_value=(sampling_client, {"train/skipped_no_valid_groups": 1} if skipped else {})),
    )
    final_loop_state = {}

    asyncio.run(train_module.do_sync_training(
        start_batch=start_batch, end_batch=start_batch + 1, num_batches=start_batch + 1,
        cfg=config, training_client=MagicMock(), kl_reference_client=None,
        evaluators=[], dataset=dataset, ml_logger=MagicMock(), tokenizer=MagicMock(),
        resume_sampling_policy_step=resume_step, has_resume_checkpoint=True,
        final_loop_state=final_loop_state,
    ))

    assert builder.calls[0][0].policy_step == expected
    next_step = expected if skipped or expected is None else expected + 1
    assert sampling_client.sampling_checkpoint.policy_step == next_step
    assert final_loop_state == {"policy_step": next_step}


def test_empty_minibatch_keeps_current_policy_without_checkpointing(monkeypatch, tmp_path):
    config = train_module.Config(
        learning_rate=1e-4, dataset_builder=lambda: None, model_name="model",
        max_tokens=8, log_path=str(tmp_path), num_groups_to_log=0,
    )
    sampler = object()
    training_client = MagicMock()
    monkeypatch.setattr(
        train_module, "prepare_minibatch", AsyncMock(return_value=([], {})),
    )
    train_step = AsyncMock()
    save_metrics = AsyncMock()
    monkeypatch.setattr(train_module, "train_step", train_step)
    monkeypatch.setattr(
        train_module, "compute_full_batch_metrics_and_get_sampling_client", save_metrics,
    )

    result, metrics = asyncio.run(train_module.do_train_step_and_get_sampling_client(
        config, 2, training_client, None, MagicMock(), [], [],
        current_sampling_client=sampler,
    ))

    assert result is sampler
    assert metrics["train/skipped_no_valid_groups"] == 1
    train_step.assert_not_awaited()
    save_metrics.assert_not_awaited()


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
