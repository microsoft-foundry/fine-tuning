"""End-to-end: shipped strategies through the real training loops.

The seam unit tests (``tests/test_train_seams.py`` + ``test_train_reroll.py``) validate the mechanism.
These drive the REAL loops with the actual shipped strategies to catch
integration risks the seam tests can't.

Sync-loop DROP / B-axis select / keep-all parity is covered against a vanilla
baseline in ``test_strategy_apples_to_apples.py``; to avoid duplicating that,
this file focuses on the paths that file does not exercise: the async loop and
the streaming-minibatch loop.

Reuses the mock-session harness and the fingerprint helpers.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from interactive_training.dynamic_batching.strategy import (
    DapoStrategy,
    DynamicBatchStrategy,
    FixedStrategy,
    PilotCommitStrategy,
    PodsStrategy,
)
from interactive_training.dynamic_batching.types import Allocation, Selection
from interactive_training.rl.metrics import incorporate_kl_penalty
from interactive_training.rl.train import (
    PrefilterRewardAccumulator,
    StreamMinibatchConfig,
    WrappedTrajectoryGroup,
    _trainable_wrapped_groups,
    do_train_step_streaming_and_get_sampling_client,
    do_sync_training_with_stream_minibatch,
)

from tests.test_dynamic_sampling import (
    AsyncConfig,
    ControlledDataset,
    FixedRewardGroupBuilder,
    MixedRewardGroupBuilder,
    _make_config,
    _make_logger,
    _make_mock_session,
    _read_metric_steps,
)
from tests.test_train_loop_characterization import _fingerprint, _run_async, _run_sync
from tests.test_train_seams import _group


def _run_stream(cfg, dataset, session, tmp_path, start=0, end=1, num_batches=1):
    import asyncio

    from tests.test_dynamic_sampling import _make_logger

    logger = _make_logger(tmp_path)
    asyncio.run(
        do_sync_training_with_stream_minibatch(
            start_batch=start,
            end_batch=end,
            num_batches=num_batches,
            cfg=cfg,
            training_client=session,
            kl_reference_client=None,
            evaluators=[],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=session.get_tokenizer(),
        )
    )
    return logger


# ---------------------------------------------------------------------------
# A-axis DROP through the real async loop (builder/group/key alignment)
# ---------------------------------------------------------------------------
#
# Sync-loop DROP parity vs a baseline is covered in
# test_strategy_apples_to_apples.py (test_dapo_drops_constant_groups_vs_baseline).


def test_dapo_strategy_drops_through_async_loop(tmp_path):
    # Same DROP semantics on the async path (cyclic dataloader backfills).
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            FixedRewardGroupBuilder(0.0, group_size=4),
        ],
        [
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
            FixedRewardGroupBuilder(1.0, group_size=4),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, strategy=DapoStrategy(),
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    _run_async(cfg, dataset, session, tmp_path)

    assert _read_metric_steps(tmp_path) == [0, 1]
    import json

    with open(tmp_path / "metrics.jsonl") as f:
        training_rows = [
            row
            for row in (json.loads(line) for line in f)
            if "training_client/step" in row
        ]
    assert all(row["sampling/pre_filter/total_groups"] > 2 for row in training_rows)
    assert all(
        row["sampling/pre_filter/by_group/frac_all_good"]
        + row["sampling/pre_filter/by_group/frac_all_bad"]
        > 0
        for row in training_rows
    )


def test_pods_strategy_shrinks_batch_through_async_loop(tmp_path):
    # ASYNC B-axis select: PodsStrategy(keep=2) on 4-sample mixed groups trains
    # on the down-sampled subset. Each of the 2 steps assembles groups_per_batch
    # groups; Pods never drops (always ADMIT), so both steps train on exactly 2
    # groups * 2 kept = 4 datums -> n_datum reflects keep=2 per surviving group,
    # not the full 4. n_datum/n_fwd_bwd are tied to training steps and so are
    # deterministic even though async in-flight rollout counts are not.
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.2, 0.5, 0.9]),
            MixedRewardGroupBuilder([0.9, 0.5, 0.2, 1.0]),
        ],
        [
            MixedRewardGroupBuilder([0.2, 1.0, 0.9, 0.5]),
            MixedRewardGroupBuilder([0.5, 0.9, 1.0, 0.2]),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        strategy=PodsStrategy(keep_per_group=2),
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    _run_async(cfg, dataset, session, tmp_path)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 2          # two training steps
    assert fp["n_datum"] == 8            # 2 steps * 2 groups * keep 2 (not 16)


def test_pilot_commit_strategy_rerolls_through_async_loop(tmp_path):
    # ASYNC A-axis MORE (re-roll): PilotCommitStrategy is in-band (p_hat=0.5 in
    # [0.125, 0.75]) and below the pilot+commit target, so it asks MORE. With
    # pilot_rollouts=4 (== the sampling group size) and commit_rollouts=4, the
    # group grows one wave (4) -> 8 == target, then ADMITs. max_rolls_per_group=8
    # caps the re-roll. A single wave would train on 4 datums; MORE grows the
    # trained pool to 8. Single group / single step keeps it deterministic.
    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])]])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        strategy=PilotCommitStrategy(
            p_lower=0.125, p_upper=0.75, pilot_rollouts=4, commit_rollouts=4
        ),
        max_rolls_per_group=8,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=1),
    )
    _run_async(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 1          # one training step
    assert fp["n_datum"] == 8            # re-rolled pool (2 waves), not 4


def test_fixed_strategy_async_matches_default_fingerprint(tmp_path):
    # ASYNC keep-all parity: FixedStrategy is a no-op seam on the async path --
    # it produces the SAME training-tied fingerprint as the default no-strategy
    # async run (mirrors the sync apples-to-apples parity anchor). n_sample is
    # excluded because async in-flight rollout counts are not deterministic; the
    # training-tied counts (n_fwd_bwd/n_optim/n_datum) are.
    def _make_dataset():
        return ControlledDataset([
            [
                MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
                MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
            ],
            [
                MixedRewardGroupBuilder([1.0, 1.0, 0.0, 0.0]),
                MixedRewardGroupBuilder([0.0, 0.0, 1.0, 1.0]),
            ],
        ])

    def _run(strategy):
        dataset = _make_dataset()
        session = _make_mock_session()
        cfg = _make_config(
            dataset, tmp_path / (strategy and "fixed" or "default"),
            sampling_seed=123, strategy=strategy,
            async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
        )
        _run_async(cfg, dataset, session, tmp_path / (strategy and "fixed" or "default"))
        fp = _fingerprint(session)
        return {k: fp[k] for k in ("n_fwd_bwd", "n_optim", "n_datum")}

    default_fp = _run(None)
    fixed_fp = _run(FixedStrategy())

    assert fixed_fp == default_fp
    assert default_fp == {"n_fwd_bwd": 2, "n_optim": 2, "n_datum": 16}


def test_legacy_synchronous_strategy_runs_through_sync_loop(tmp_path):
    class LegacySyncStrategy(DynamicBatchStrategy):
        def allocate(self, group, *, history=None):
            return Allocation.ADMIT

        def select(self, batch):
            return [
                Selection(
                    kept_indices=list(range(len(group.trajectories_G))),
                    weights=[1.0] * len(group.trajectories_G),
                    advantages=torch.zeros(len(group.trajectories_G)),
                )
                for group in batch
            ]

    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, strategy=LegacySyncStrategy()
    )

    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    assert _fingerprint(session)["n_datum"] == 4


def test_streaming_filter_drops_wrappers_without_trajectory_groups() -> None:
    builder = MixedRewardGroupBuilder([1.0, 0.0])
    valid_group = object()
    dropped = WrappedTrajectoryGroup(
        trajectory_group=None,
        env_group_builder=builder,
        sampling_client_step=0,
    )
    valid = WrappedTrajectoryGroup(
        trajectory_group=valid_group,
        env_group_builder=builder,
        sampling_client_step=0,
    )

    assert _trainable_wrapped_groups([None, dropped, valid]) == [valid]


def test_streaming_minibatch_backfills_dropped_wrapper(tmp_path, monkeypatch) -> None:
    import interactive_training.rl.train as train_module

    async def skip_post_train_metrics(training_client, *_args, **_kwargs):
        return training_client.create_sampling_client(), {}

    monkeypatch.setattr(
        train_module,
        "compute_full_batch_metrics_and_get_sampling_client",
        skip_post_train_metrics,
    )

    async def run() -> None:
        builder = MixedRewardGroupBuilder([1.0, 0.0])
        dropped_prefilter = PrefilterRewardAccumulator()
        dropped_prefilter.add(_group([0.0, 0.0]))
        valid_prefilter = PrefilterRewardAccumulator()
        valid_prefilter.add(_group([1.0, 0.0]))
        aggregate_prefilter = PrefilterRewardAccumulator()
        queue = asyncio.Queue()
        await queue.put(
            WrappedTrajectoryGroup(
                trajectory_group=None,
                env_group_builder=builder,
                sampling_client_step=0,
                prefilter_rewards=dropped_prefilter,
            )
        )
        await queue.put(
            WrappedTrajectoryGroup(
                trajectory_group=_group([1.0, 0.0]),
                env_group_builder=builder,
                sampling_client_step=0,
                metrics={"time/trajectory_group_worker_loop/total": 0.01},
                prefilter_rewards=valid_prefilter,
            )
        )
        dataset = ControlledDataset([[builder]])
        session = _make_mock_session()
        cfg = _make_config(
            dataset,
            tmp_path,
            stream_minibatch_config=StreamMinibatchConfig(
                groups_per_batch=1, num_minibatches=1
            ),
        )

        _, metrics = await do_train_step_streaming_and_get_sampling_client(
            cfg,
            0,
            queue,
            session,
            None,
            session.get_tokenizer(),
            prefilter_rewards=aggregate_prefilter,
        )

        assert _fingerprint(session)["n_datum"] == 2
        assert metrics["sampling/pre_filter/total_groups"] == 2
        assert metrics["sampling/pre_filter/total_episodes"] == 4

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Streaming-minibatch path (stream_minibatch_config) + stateless strategy
# ---------------------------------------------------------------------------


def test_dapo_strategy_drops_constant_group_through_streaming_loop(tmp_path):
    # The streaming worker threads the strategy too: DapoStrategy DROPs the
    # constant group (worker enqueues None), the streaming consumer strips it,
    # and only the mixed group trains.
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            FixedRewardGroupBuilder(0.0, group_size=4),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, strategy=DapoStrategy(),
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=2, num_minibatches=1
        ),
    )
    _run_stream(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 8   # both groups rolled
    assert fp["n_datum"] == 4    # constant dropped by allocate


def test_dapo_all_groups_dropped_through_streaming_loop_does_not_crash(tmp_path):
    # Regression: when a strategy DROPs *every* group in a streaming minibatch,
    # the batch must be a no-op instead of crashing or advancing the optimizer.
    dataset = ControlledDataset([
        [
            FixedRewardGroupBuilder(0.0, group_size=4),
            FixedRewardGroupBuilder(1.0, group_size=4),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, strategy=DapoStrategy(),
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=2, num_minibatches=1
        ),
    )
    _run_stream(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 8   # both groups rolled
    assert fp["n_datum"] == 0    # both constant groups dropped -> empty batch
    assert fp["n_fwd_bwd"] == 0
    assert fp["n_optim"] == 0


def test_selection_drops_all_groups_through_streaming_loop_is_noop(tmp_path):
    class DropAllAtSelectionStrategy(DynamicBatchStrategy):
        async def allocate(self, group, *, history=None):
            return Allocation.ADMIT

        async def select(self, batch):
            return [
                Selection(
                    kept_indices=[],
                    weights=[],
                    advantages=torch.empty(0),
                )
                for _ in batch
            ]

    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        sampling_seed=123,
        strategy=DropAllAtSelectionStrategy(),
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=1, num_minibatches=1
        ),
    )
    _run_stream(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 4
    assert fp["n_datum"] == 0
    assert fp["n_fwd_bwd"] == 0
    assert fp["n_optim"] == 0


def test_selection_drops_all_groups_through_sync_loop_is_noop(tmp_path):
    class DropAllAtSelectionStrategy(DynamicBatchStrategy):
        async def allocate(self, group, *, history=None):
            return Allocation.ADMIT

        async def select(self, batch):
            return [
                Selection(
                    kept_indices=[],
                    weights=[],
                    advantages=torch.empty(0),
                )
                for _ in batch
            ]

    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        sampling_seed=123,
        strategy=DropAllAtSelectionStrategy(),
    )
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 4
    assert fp["n_datum"] == 0
    assert fp["n_fwd_bwd"] == 0
    assert fp["n_optim"] == 0


def test_incorporate_kl_penalty_empty_batch_is_noop():
    # Regression: prepare_minibatch calls incorporate_kl_penalty(data_D, ...)
    # when kl_penalty_coef > 0, BEFORE the full-batch metric guards. A train-side
    # strategy selection that keeps no groups makes data_D == [], and the old
    # code computed sum([]) / sum([]) -> ZeroDivisionError. The empty batch must
    # be a no-op (no client calls, no crash, no metrics).
    import asyncio

    metrics = asyncio.run(
        incorporate_kl_penalty([], base_sampling_client=None, kl_penalty_coef=0.1, kl_discount_factor=1.0)
    )
    assert metrics == {}


def test_pods_strategy_shrinks_batch_through_streaming_loop(tmp_path):
    # B-axis select runs in the streaming prepare_minibatch: Pods keeps 2 of 4.
    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.2, 0.5, 0.9])]])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, strategy=PodsStrategy(keep_per_group=2),
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=1, num_minibatches=1
        ),
    )
    _run_stream(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 4   # rolled the full group
    assert fp["n_datum"] == 2    # B-axis kept only 2


def test_stateless_strategy_streaming_matches_default_datum_count(tmp_path):
    # A keep-all strategy through streaming trains on every rolled trajectory,
    # same as the default streaming path would.
    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])]])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        strategy=PodsStrategy(keep_per_group=4),
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=1, num_minibatches=1
        ),
    )
    _run_stream(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_datum"] == 4


def test_pilot_commit_strategy_rerolls_through_streaming_loop(tmp_path):
    # STREAMING A-axis MORE (re-roll): the streaming worker threads the strategy
    # + max_rolls_per_group too. PilotCommit is in-band (p_hat=0.5) and below the
    # pilot+commit target, so it asks MORE; with pilot_rollouts=4 (== group size)
    # and commit_rollouts=4 the group grows one wave (4) -> 8 == target, then
    # ADMITs (cap max_rolls_per_group=8). A single wave would roll+train on 4;
    # MORE grows both the rolled pool and the trained datums to 8.
    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])]])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        strategy=PilotCommitStrategy(
            p_lower=0.125, p_upper=0.75, pilot_rollouts=4, commit_rollouts=4
        ),
        max_rolls_per_group=8,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=1, num_minibatches=1
        ),
    )
    _run_stream(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 8   # re-rolled a second wave (2 * 4)
    assert fp["n_datum"] == 8    # trained on the grown pool, not 4
