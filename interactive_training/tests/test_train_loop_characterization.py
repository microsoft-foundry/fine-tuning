"""Golden-master characterization of the train.py loop's DEFAULT path.

These tests pin the observable behavior of ``do_sync_training`` /
``do_async_training`` with **no strategy** (the vanilla path) via the mock
FineTuningSession: how many rollouts happen, how many train steps run, how
many Datum reach the trainer, how constant-reward dropping shrinks the batch,
and that runs are deterministic.

They deliberately avoid every symbol added by the dynamic-batching work
(``strategy``, ``max_rolls_per_group``, ...), so the SAME file imports and runs
against the pre-change ``train.py`` on master. The intended use is differential:

    1. run against master's train.py  -> establishes the golden behavior
    2. run against the modified train.py with no strategy set -> must match

If both passes are green with identical assertions, the additive strategy
changes did not perturb the default loop.
"""

from __future__ import annotations

import asyncio

from interactive_training.rl.train import do_async_training, do_sync_training

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


def _fingerprint(session) -> dict:
    """Observable counts the loop produced against the mock session."""
    fwd_bwd_calls = session.forward_backward_async.call_args_list
    n_datum = sum(len(c.args[0]) for c in fwd_bwd_calls)
    return {
        "n_sample": session.sample_async.call_count,
        "n_fwd_bwd": session.forward_backward_async.call_count,
        "n_optim": session.optim_step_async.call_count,
        "n_datum": n_datum,
    }


def _run_sync(cfg, dataset, session, tmp_path, start=0, end=2, num_batches=2, **kw):
    logger = _make_logger(tmp_path)
    asyncio.run(
        do_sync_training(
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
            **kw,
        )
    )
    return logger


def _run_async(cfg, dataset, session, tmp_path, start=0, end=2, num_batches=2, timeout=30.0):
    logger = _make_logger(tmp_path)

    async def _driver():
        await asyncio.wait_for(
            do_async_training(
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
            ),
            timeout=timeout,
        )

    asyncio.run(_driver())
    logger.close()
    return logger


# ---------------------------------------------------------------------------
# Sync path
# ---------------------------------------------------------------------------


def test_sync_two_steps_fingerprint(tmp_path):
    # 2 batches, each one mixed group of 4 envs -> 2 steps, 4 rollouts/step.
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, sampling_seed=123)
    _run_sync(cfg, dataset, session, tmp_path)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 2
    assert fp["n_optim"] == 2
    assert fp["n_sample"] == 8   # 2 steps * 1 group * 4 envs
    assert fp["n_datum"] == 8    # keep-all: one Datum per trajectory


def test_sync_multi_group_batch_fingerprint(tmp_path):
    # One batch with two mixed groups, run one step.
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, sampling_seed=123)
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 1
    assert fp["n_sample"] == 8   # 2 groups * 4 envs
    assert fp["n_datum"] == 8


def test_sync_constant_reward_dropping_shrinks_batch(tmp_path):
    # One batch: one mixed group (kept) + one constant group (dropped).
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            FixedRewardGroupBuilder(0.0, group_size=4),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, remove_constant_reward_groups=True
    )
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    # Both groups are rolled (8 samples), but the constant one is dropped, so
    # only the 4 mixed trajectories reach the trainer.
    assert fp["n_sample"] == 8
    assert fp["n_datum"] == 4
    assert fp["n_fwd_bwd"] == 1


def test_sync_deadline_breaks_before_any_training(tmp_path):
    import time as _time

    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])]])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path)
    _run_sync(
        cfg, dataset, session, tmp_path, end=5, num_batches=1,
        deadline=_time.time() - 1.0,
    )

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 0
    assert fp["n_sample"] == 0


def test_sync_run_is_deterministic(tmp_path):
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    fps = []
    for i in range(2):
        session = _make_mock_session()
        cfg = _make_config(dataset, tmp_path / f"run{i}", sampling_seed=777)
        _run_sync(cfg, dataset, session, tmp_path / f"run{i}")
        fps.append(_fingerprint(session))
    assert fps[0] == fps[1]


# ---------------------------------------------------------------------------
# Async path
# ---------------------------------------------------------------------------


def test_async_two_steps_all_mixed(tmp_path):
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])] * 2,
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])] * 2,
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        remove_constant_reward_groups=True,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    _run_async(cfg, dataset, session, tmp_path)

    assert _read_metric_steps(tmp_path) == [0, 1]
    assert session.forward_backward_async.call_count >= 2


def test_async_completes_despite_constant_reward_drops(tmp_path):
    # Cyclic dataloader must wrap to backfill dropped constant-reward groups.
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
        dataset, tmp_path, sampling_seed=123,
        remove_constant_reward_groups=True,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    _run_async(cfg, dataset, session, tmp_path)

    assert _read_metric_steps(tmp_path) == [0, 1]


# ---------------------------------------------------------------------------
# Edge cases: filtering eliminates all / most prompts in a batch
# ---------------------------------------------------------------------------


def test_sync_all_constant_batch_trains_on_fallback_singleton(tmp_path):
    # Every group is constant-reward and remove_constant_reward_groups=True.
    # remove_constant_reward_groups() returns a 1-group fallback ([0:1]) rather
    # than an empty batch (logs "All rewards are uniform"), so the step still
    # trains -- on exactly that one group. Pins that master behavior.
    dataset = ControlledDataset([
        [
            FixedRewardGroupBuilder(0.0, group_size=4),
            FixedRewardGroupBuilder(1.0, group_size=4),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, remove_constant_reward_groups=True
    )
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 8   # both groups rolled
    assert fp["n_fwd_bwd"] == 1  # trained on the fallback singleton
    assert fp["n_datum"] == 4    # exactly one group's trajectories


def test_sync_all_constant_batch_no_filter_trains_on_all(tmp_path):
    # No filtering: every constant group is kept and trained on (zero-gradient
    # but the loop still runs). Pins the unfiltered all-constant behavior.
    dataset = ControlledDataset([
        [
            FixedRewardGroupBuilder(0.0, group_size=4),
            FixedRewardGroupBuilder(1.0, group_size=4),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123, remove_constant_reward_groups=False
    )
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_sample"] == 8
    assert fp["n_fwd_bwd"] == 1
    assert fp["n_datum"] == 8   # both groups kept


def test_sync_dynamic_sampling_all_constant_skips_step(tmp_path):
    # DAPO dynamic sampling: an all-constant batch never yields a valid group,
    # so after max_oversample_rounds the step is skipped with no training. Pins
    # that the rounds are bounded and no forward/backward happens.
    dataset = ControlledDataset([[FixedRewardGroupBuilder(0.0, group_size=4)]])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        dynamic_sampling=True, max_oversample_rounds=2,
    )
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 0   # step skipped: no valid groups
    assert fp["n_optim"] == 0
    assert fp["n_sample"] == 8    # 2 oversample rounds * 4 envs


def test_sync_partial_filter_keeps_only_survivors(tmp_path):
    # Two batches; each has one mixed (kept) + one constant (dropped) group.
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
        dataset, tmp_path, sampling_seed=123, remove_constant_reward_groups=True
    )
    _run_sync(cfg, dataset, session, tmp_path, end=2, num_batches=2)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 2   # both steps train
    assert fp["n_sample"] == 16   # 2 steps * 2 groups * 4 envs (all rolled)
    assert fp["n_datum"] == 8     # only the mixed group per step survives


def test_async_all_constant_stops_without_training(tmp_path):
    # Every group is constant-reward: no trainable batch can ever be assembled,
    # so the watchdog stops the run after enough consecutive drops -- with zero
    # training steps recorded.
    dataset = ControlledDataset([[FixedRewardGroupBuilder(0.0, group_size=4)]])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        remove_constant_reward_groups=True, max_oversample_rounds=2,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    _run_async(cfg, dataset, session, tmp_path, end=3, num_batches=1)

    assert _read_metric_steps(tmp_path) == []
    assert session.forward_backward_async.call_count == 0


# ---------------------------------------------------------------------------
# Resume: DAPO dynamic-sampling prompt cursor (master-supported behavior)
# ---------------------------------------------------------------------------


class _RecordingDataset(ControlledDataset):
    """ControlledDataset that records which (wrapped) batch indices were drawn."""

    def __init__(self, batch_builders):
        super().__init__(batch_builders)
        self.requested: list[int] = []

    def get_batch(self, index):
        self.requested.append(index % len(self._batch_builders))
        return super().get_batch(index)


def test_sync_dynamic_sampling_resumes_from_prompt_cursor(tmp_path):
    # The DAPO gotcha: dynamic_sampling decouples the training step from the
    # dataset cursor, so on resume the loop must start drawing from the saved
    # prompt_cursor, not from batch 0. Pin that resume_prompt_cursor is honored.
    dataset = _RecordingDataset([
        [MixedRewardGroupBuilder([1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        dynamic_sampling=True, max_oversample_rounds=2,
    )
    _run_sync(
        cfg, dataset, session, tmp_path, end=1, num_batches=2,
        resume_prompt_cursor=1,
    )

    # The very first dataset batch touched reflects the resumed cursor (1),
    # not a fresh start at 0.
    assert dataset.requested[0] == 1


def test_sync_dynamic_sampling_fresh_start_draws_from_zero(tmp_path):
    # Contrast: with no resume cursor the first batch drawn is 0.
    dataset = _RecordingDataset([
        [MixedRewardGroupBuilder([1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, sampling_seed=123,
        dynamic_sampling=True, max_oversample_rounds=2,
    )
    _run_sync(cfg, dataset, session, tmp_path, end=1, num_batches=2)

    assert dataset.requested[0] == 0
