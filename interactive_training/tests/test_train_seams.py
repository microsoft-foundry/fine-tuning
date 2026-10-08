"""Phase-1 integration tests for the dynamic-batching seams in ``rl/train.py``.

These pin the two seams wired into the production training loop:

* **B-axis** -- ``prepare_minibatch`` routes advantage/subset selection
  through ``strategy.select`` (via ``_apply_selections`` +
  ``_materialize_subgroup``) when a strategy is set, and is byte-identical
  to the vanilla ``compute_advantages`` path when it is ``None`` or
  ``FixedStrategy``.
* **A-axis DROP** -- ``do_group_rollout_and_filter_constant_reward``
  honours ``strategy.allocate(...).is_drop`` (``DapoStrategy`` reproduces
  constant-reward dropping).

They exercise the seam helpers directly, so no training client or tokenizer is
required; async strategy hooks are driven with a local event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest
import torch

from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.completers import TokensWithLogprobs
from interactive_training.dynamic_batching.strategy import (
    DapoStrategy,
    DynamicBatchStrategy,
    FixedStrategy,
    PilotCommitStrategy,
    PodsStrategy,
)
from interactive_training.dynamic_batching.types import Allocation, Selection
from interactive_training.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
    trajectory_to_data,
)
from interactive_training.rl import train as train_module
from interactive_training.rl.rollouts import do_group_rollout
from interactive_training.rl.train import (
    _apply_selections,
    _assemble_training_data_async,
    _call_sampling_evaluator,
    _materialize_subgroup,
)
from interactive_training.rl.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    TrajectoryGroup,
    Trajectory,
    Transition,
)
from interactive_training.utils.misc_utils import all_same


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


async def test_sampling_evaluator_preserves_legacy_call_signature():
    sampling_client = object()

    class LegacyEvaluator:
        async def __call__(self, client):
            assert client is sampling_client
            return {"legacy": 1.0}

    result = await _call_sampling_evaluator(LegacyEvaluator(), sampling_client, 7)

    assert result == {"legacy": 1.0}


async def test_sampling_evaluator_passes_step_when_supported():
    sampling_client = object()

    class StepAwareEvaluator:
        async def __call__(self, client, *, step=None):
            assert client is sampling_client
            return {"step": step}

    result = await _call_sampling_evaluator(StepAwareEvaluator(), sampling_client, 7)

    assert result == {"step": 7}


def _traj(n_tokens: int) -> Trajectory:
    """Real single-transition trajectory with ``n_tokens`` action tokens."""
    ob = ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])])
    ac = TokensWithLogprobs(
        tokens=list(range(10, 10 + n_tokens)),
        maybe_logprobs=[-0.1] * n_tokens,
    )
    return Trajectory(
        transitions=[Transition(ob=ob, ac=ac, reward=0.0, episode_done=True)],
        final_ob=ob,
    )


def _group(rewards: list[float], lengths: list[int] | None = None) -> TrajectoryGroup:
    """Build a group whose per-trajectory reward and response length are set.

    The reward rides ``final_rewards_G`` (per-transition reward is 0), so
    ``get_total_rewards()`` returns exactly ``rewards``. ``lengths`` (default
    2 each) controls action-token counts so length-based strategies have a
    signal independent of reward.
    """
    if lengths is None:
        lengths = [2] * len(rewards)
    return TrajectoryGroup(
        trajectories_G=[_traj(n) for n in lengths],
        final_rewards_G=list(rewards),
        metrics_G=[{"i": i} for i in range(len(rewards))],
    )


def test_materialize_subgroup_preserves_selection_seeds():
    group = _group([0.1, 0.5, 0.9])
    for trajectory, seed in zip(
        group.trajectories_G, [101, 202, 303], strict=True
    ):
        object.__setattr__(trajectory, "selection_seed", seed)

    subgroup = _materialize_subgroup(group, [2, 0])

    assert [trajectory.selection_seed for trajectory in subgroup.trajectories_G] == [
        303,
        101,
    ]


def test_rollout_carries_upstream_instance_budget_metadata():
    ob = ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])])

    class TestEnv(Env):
        async def initial_observation(self):
            return ob, [0]

        async def step(self, action):
            return StepResult(
                reward=0.5,
                episode_done=True,
                next_observation=ob,
                next_stop_condition=[0],
            )

    class TestBuilder(EnvGroupBuilder):
        async def make_envs(self):
            return [TestEnv()]

        def dynamic_batching_metadata(self) -> dict[str, object]:
            return {"custom_budget": 0.25, "custom_target": 12.0}

    class TestPolicy:
        async def __call__(self, model_input, stop, max_tokens=None):
            return TokensWithLogprobs(tokens=[4], maybe_logprobs=[-0.1])

    group = asyncio.run(do_group_rollout(TestBuilder(), TestPolicy()))

    assert group.dynamic_batching_metadata == {
        "custom_budget": 0.25,
        "custom_target": 12.0,
    }


# ---------------------------------------------------------------------------
# B-axis: FixedStrategy parity with the vanilla compute_advantages path
# ---------------------------------------------------------------------------


def test_builtin_strategies_remain_synchronous() -> None:
    group = _group([0.0, 1.0, 0.0, 1.0])
    strategies = [
        FixedStrategy(),
        DapoStrategy(),
        PodsStrategy(keep_per_group=2),
        PilotCommitStrategy(),
    ]

    for strategy in strategies:
        assert isinstance(strategy.allocate(group), Allocation)
        assert isinstance(strategy.select([group]), list)


def test_fixed_strategy_advantages_match_compute_advantages():
    groups = [_group([0.9, 0.1, 0.5]), _group([1.0, 0.0]), _group([0.2, 0.2, 0.8, 0.4])]
    fixed = FixedStrategy()

    selections = fixed.select(groups)
    baseline = compute_advantages(groups)

    assert len(selections) == len(baseline)
    for sel, adv, group in zip(selections, baseline, groups):
        # Keeps every sample, in order.
        assert sel.kept_indices == list(range(len(group.trajectories_G)))
        assert sel.weights == [1.0] * len(group.trajectories_G)
        assert torch.allclose(sel.advantages, adv)


def test_apply_selections_fixed_is_identity_for_assemble():
    """FixedStrategy through the seam yields the same training data as vanilla."""
    groups = [_group([0.9, 0.1, 0.5]), _group([1.0, 0.0])]

    # Vanilla path
    vanilla_adv = compute_advantages(groups)
    vanilla_data, vanilla_meta = assemble_training_data(groups, vanilla_adv)

    # Seam path
    metrics: dict = {}
    kept_groups, sel_adv, token_adjustments = asyncio.run(
        _apply_selections(groups, FixedStrategy(), metrics)
    )
    seam_data, seam_meta = assemble_training_data(kept_groups, sel_adv)

    assert len(seam_data) == len(vanilla_data)
    assert seam_meta == vanilla_meta
    assert token_adjustments == [None, None]
    for a, b in zip(sel_adv, vanilla_adv):
        assert torch.allclose(a, b)
    # Metrics report a full keep.
    assert metrics["strategy/n_groups_in"] == 2
    assert metrics["strategy/n_groups_kept"] == 2
    assert metrics["strategy/select_keep_rate"] == 1.0


@pytest.mark.parametrize("strategy", [None, FixedStrategy(), PodsStrategy(keep_per_group=2)])
async def test_async_assembly_matches_synchronous_for_strategy_paths(strategy):
    groups = [_group([0.9, 0.1, 0.5]), _group([1.0, 0.0])]
    if strategy is None:
        assembly_groups = groups
        advantages = compute_advantages(groups)
        token_adjustments = None
    else:
        assembly_groups, advantages, token_adjustments = await _apply_selections(
            groups, strategy, {}
        )

    expected = assemble_training_data(
        assembly_groups, advantages, token_adjustments
    )
    actual = await _assemble_training_data_async(
        assembly_groups, advantages, token_adjustments
    )

    assert actual == expected


async def test_async_assembly_keeps_loop_drift_under_one_second(monkeypatch):
    groups = [_group([float(i % 3) for i in range(96)], lengths=[128] * 96)]
    advantages = compute_advantages(groups)
    original_assemble = train_module.assemble_training_data

    def slow_assemble(trajectory_groups, trajectory_advantages):
        time.sleep(1.05)
        return original_assemble(trajectory_groups, trajectory_advantages)

    async def measure(run_assembly):
        interval = 0.01
        ticker_started = asyncio.Event()
        stop_ticker = asyncio.Event()
        loop_drifts = []

        async def ticker():
            previous_tick = asyncio.get_running_loop().time()
            ticker_started.set()
            while not stop_ticker.is_set():
                await asyncio.sleep(interval)
                current_tick = asyncio.get_running_loop().time()
                loop_drifts.append(max(0.0, current_tick - previous_tick - interval))
                previous_tick = current_tick

        ticker_task = asyncio.create_task(ticker())
        await ticker_started.wait()
        started_at = time.monotonic()
        await run_assembly()
        wall_time = time.monotonic() - started_at
        stop_ticker.set()
        await ticker_task
        return wall_time, max(loop_drifts, default=0.0)

    async def run_synchronously():
        train_module.assemble_training_data(groups, advantages)

    async def run_off_loop():
        await _assemble_training_data_async(groups, advantages)

    monkeypatch.setattr(train_module, "assemble_training_data", slow_assemble)
    sync_wall_time, sync_max_drift = await measure(run_synchronously)
    async_wall_time, async_max_drift = await measure(run_off_loop)

    assert sync_wall_time >= 1.0
    assert async_wall_time >= 1.0
    assert sync_max_drift >= 1.0
    assert async_max_drift < 1.0


async def test_async_assembly_repeated_cancellation_drains_worker(monkeypatch):
    assembly_started = threading.Event()
    release_assembly = threading.Event()
    assembly_finished = threading.Event()

    def slow_assemble(trajectory_groups, advantages):
        del trajectory_groups, advantages
        assembly_started.set()
        assert release_assembly.wait(timeout=2)
        assembly_finished.set()
        return [], []

    monkeypatch.setattr(train_module, "assemble_training_data", slow_assemble)
    assembly_task = asyncio.create_task(_assemble_training_data_async([], []))
    while not assembly_started.is_set():
        await asyncio.sleep(0)

    assembly_task.cancel()
    await asyncio.sleep(0)
    assembly_task.cancel()
    await asyncio.sleep(0)
    assert not assembly_task.done()

    release_assembly.set()
    with pytest.raises(asyncio.CancelledError):
        await assembly_task
    assert assembly_finished.is_set()


async def test_async_assembly_observes_worker_failure_after_cancellation(
    monkeypatch, caplog
):
    assembly_started = threading.Event()
    release_assembly = threading.Event()

    def failing_assemble(trajectory_groups, advantages):
        del trajectory_groups, advantages
        assembly_started.set()
        assert release_assembly.wait(timeout=2)
        raise RuntimeError("assembly failed")

    monkeypatch.setattr(train_module, "assemble_training_data", failing_assemble)
    caplog.set_level(logging.ERROR, logger=train_module.__name__)
    assembly_task = asyncio.create_task(_assemble_training_data_async([], []))
    while not assembly_started.is_set():
        await asyncio.sleep(0)

    assembly_task.cancel()
    await asyncio.sleep(0)
    release_assembly.set()

    with pytest.raises(asyncio.CancelledError):
        await assembly_task
    assert "assemble_training_data failed after cancellation" in caplog.text


def test_trajectory_to_data_adds_nested_token_advantage_adjustments() -> None:
    trajectory = _traj(3)

    (datum,) = trajectory_to_data(
        trajectory,
        0.5,
        token_advantage_adjustments=[[0.1, -0.2, 0.3]],
    )

    action_advantages = [
        advantage
        for advantage, mask in zip(
            datum.loss_fn_inputs["advantages"].data,
            datum.loss_fn_inputs["mask"].data,
            strict=True,
        )
        if mask == 1.0
    ]
    assert action_advantages == pytest.approx([0.6, 0.3, 0.8])


def test_trajectory_to_data_rejects_misaligned_token_advantage_adjustments() -> None:
    trajectory = _traj(2)

    with pytest.raises(ValueError, match="token advantage adjustments"):
        trajectory_to_data(
            trajectory,
            0.5,
            token_advantage_adjustments=[[0.1]],
        )


def test_custom_strategy_can_add_per_token_credit() -> None:
    class DenseCreditStrategy(DynamicBatchStrategy):
        async def allocate(self, group, *, history=None):
            return Allocation.ADMIT

        async def select(self, batch):
            return [
                Selection(
                    kept_indices=list(range(len(group.trajectories_G))),
                    weights=[1.0] * len(group.trajectories_G),
                    advantages=torch.zeros(len(group.trajectories_G)),
                    token_advantage_adjustments=[
                        [[0.1] * len(transition.ac.tokens) for transition in trajectory.transitions]
                        for trajectory in group.trajectories_G
                    ],
                )
                for group in batch
            ]

    group = _group([0.2, 0.8], lengths=[2, 3])
    kept_groups, advantages, token_adjustments = asyncio.run(
        _apply_selections([group], DenseCreditStrategy(), {})
    )
    data, _ = assemble_training_data(
        kept_groups, advantages, token_adjustments
    )

    assert len(data) == 2
    for datum in data:
        action_advantages = [
            advantage
            for advantage, mask in zip(
                datum.loss_fn_inputs["advantages"].data,
                datum.loss_fn_inputs["mask"].data,
                strict=True,
            )
            if mask == 1.0
        ]
        assert action_advantages == pytest.approx([0.1] * len(action_advantages))


def test_materialize_subgroup_slices_all_fields():
    group = _group([0.1, 0.2, 0.3, 0.4], lengths=[2, 5, 9, 3])
    sub = _materialize_subgroup(group, [0, 2])

    assert sub.get_total_rewards() == [0.1, 0.3]
    assert sub.metrics_G == [{"i": 0}, {"i": 2}]
    assert sub.trajectories_G[0] is group.trajectories_G[0]
    assert sub.trajectories_G[1] is group.trajectories_G[2]


# ---------------------------------------------------------------------------
# B-axis: PODS shrinks the group
# ---------------------------------------------------------------------------


def test_pods_select_shrinks_to_keep_and_centers_on_subset():
    group = _group([0.0, 0.1, 0.5, 0.9, 1.0])
    kept_groups, adv, token_adjustments = asyncio.run(
        _apply_selections([group], PodsStrategy(keep_per_group=2), {})
    )

    assert len(kept_groups) == 1
    assert len(kept_groups[0].trajectories_G) == 2
    # Max-variance size-2 subset of these rewards is the extremes {0.0, 1.0}.
    assert sorted(kept_groups[0].get_total_rewards()) == [0.0, 1.0]
    # Advantages centered over the kept subset -> +-0.5.
    assert torch.allclose(adv[0].abs(), torch.tensor([0.5, 0.5]))
    assert token_adjustments == [None]


# ---------------------------------------------------------------------------
# A-axis DROP: DapoStrategy reproduces constant-reward dropping
# ---------------------------------------------------------------------------


def test_dapo_allocate_drops_only_constant_reward_groups():
    dapo = DapoStrategy()
    varied = _group([0.9, 0.1, 0.5])
    constant = _group([0.5, 0.5, 0.5])
    singleton = _group([0.5])

    assert dapo.allocate(varied).is_admit
    assert dapo.allocate(constant).is_drop
    # Degenerate single-sample group is admitted (no variance to judge).
    assert dapo.allocate(singleton).is_admit


def test_dapo_drop_matches_remove_constant_reward_groups():
    """The A-axis DROP predicate matches today's batch-level filter."""
    dapo = DapoStrategy()
    groups = [_group([0.9, 0.1]), _group([0.5, 0.5]), _group([0.2, 0.8, 0.4])]

    # Per-group strategy DROP.
    kept_by_strategy = [
        g for g in groups if not dapo.allocate(g).is_drop
    ]
    # Batch-level filter.
    kept_by_filter = remove_constant_reward_groups(groups)

    assert [g.get_total_rewards() for g in kept_by_strategy] == [
        g.get_total_rewards() for g in kept_by_filter
    ]
    # Sanity: the constant group is the one dropped.
    assert all(not all_same(g.get_total_rewards()) for g in kept_by_strategy)


def test_fixed_strategy_never_drops():
    fixed = FixedStrategy()
    for rewards in ([0.5, 0.5, 0.5], [0.9, 0.1], [0.3]):
        assert fixed.allocate(_group(rewards)).is_admit
