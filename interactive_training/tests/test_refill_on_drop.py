"""End-to-end: refill-on-drop keeps the trained batch at target size.

Option B. When a strategy's A-axis ``allocate`` DROPs a group, ``refill_on_drop``
makes the sync loop pull fresh prompts until ``groups_per_batch`` groups survive,
instead of letting the batch shrink for that step. This makes drop-based
strategies (DAPO, Pilot-Commit, ...) faithful to their papers, which hold the
trained batch size constant. It reuses the ``dynamic_sampling`` refill machinery
but the *survivor* criterion is the strategy's drop, not native constant-reward
filtering.

Reuses the mock-session harness from ``test_dynamic_sampling`` /
``test_train_loop_characterization``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from interactive_training.dynamic_batching.strategy import DapoStrategy
from interactive_training.rl.train import AsyncConfig, Config

from tests.test_dynamic_sampling import (
    ControlledDataset,
    FixedRewardGroupBuilder,
    MixedRewardGroupBuilder,
    _make_config,
    _make_mock_session,
)
from tests.test_train_loop_characterization import (
    _RecordingDataset,
    _fingerprint,
    _run_sync,
)


# ---------------------------------------------------------------------------
# Behavior: refill vs shrink on a strategy DROP
# ---------------------------------------------------------------------------


def test_refill_on_drop_fills_batch_after_strategy_drop(tmp_path):
    """Dropped groups are replaced with fresh prompts up to the target size.

    Batch 0's two groups are constant-reward, so DapoStrategy DROPs both; the
    loop pulls batch 1 (two mixed groups) and trains on those 2 -> the batch is
    refilled to the target (2 groups x 4 = 8 datums), not shrunk to 0.
    """
    dataset = ControlledDataset([
        [
            FixedRewardGroupBuilder(1.0, group_size=4),  # constant -> DROP
            FixedRewardGroupBuilder(0.0, group_size=4),  # constant -> DROP
        ],
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),  # kept
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),  # kept
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, strategy=DapoStrategy(), refill_on_drop=True)
    _run_sync(cfg, dataset, session, tmp_path, start=0, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 1
    assert fp["n_datum"] == 8  # refilled to 2 groups, not shrunk


def test_no_refill_shrinks_batch_on_strategy_drop(tmp_path):
    """Without refill_on_drop (default), a DROP shrinks the batch for the step."""
    dataset = ControlledDataset([
        [
            FixedRewardGroupBuilder(1.0, group_size=4),  # constant -> DROP
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),  # kept
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, strategy=DapoStrategy())  # refill_on_drop=False
    _run_sync(cfg, dataset, session, tmp_path, start=0, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 1
    assert fp["n_datum"] == 4  # batch shrank 2 -> 1


def test_refill_on_drop_single_round_when_strategy_never_drops(tmp_path):
    """A never-dropping strategy refills in one round -> batch == target."""
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),  # never constant
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, strategy=DapoStrategy(), refill_on_drop=True)
    _run_sync(cfg, dataset, session, tmp_path, start=0, end=1, num_batches=1)

    fp = _fingerprint(session)
    assert fp["n_fwd_bwd"] == 1
    assert fp["n_datum"] == 8  # both groups train, no extra rounds


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_refill_on_drop_requires_strategy(tmp_path):
    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0])]])
    with pytest.raises(ValueError, match="refill_on_drop requires a `strategy`"):
        _make_config(dataset, tmp_path, refill_on_drop=True)  # strategy=None


def test_refill_on_drop_rejects_async():
    with pytest.raises(ValueError, match="synchronous"):
        Config(
            learning_rate=1e-5,
            dataset_builder=MagicMock(),
            model_name="test",
            max_tokens=10,
            log_path="/tmp/test",
            strategy=DapoStrategy(),
            refill_on_drop=True,
            async_config=AsyncConfig(max_steps_off_policy=2, groups_per_batch=4),
        )


def test_refill_on_drop_stateless_sync_accepted(tmp_path):
    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0])]])
    # Should not raise.
    _make_config(dataset, tmp_path, strategy=DapoStrategy(), refill_on_drop=True)


# ---------------------------------------------------------------------------
# Resume: refill_on_drop advances prompt_cursor, so it must be checkpointed and
# honored on resume (the same gotcha dynamic_sampling handles).
# ---------------------------------------------------------------------------


def test_refill_on_drop_checkpoints_prompt_cursor(tmp_path, monkeypatch):
    """refill_on_drop must persist prompt_cursor via loop_state_extra.

    Without this, a resumed run resets the cursor to start_batch and re-reads
    already-consumed prompts (dataset drift).
    """
    import interactive_training.rl.train as T

    captured: dict = {}
    orig = T.do_train_step_and_get_sampling_client

    async def _spy(*args, **kw):
        captured["loop_state_extra"] = kw.get("loop_state_extra")
        return await orig(*args, **kw)

    monkeypatch.setattr(T, "do_train_step_and_get_sampling_client", _spy)

    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, strategy=DapoStrategy(), refill_on_drop=True)
    _run_sync(cfg, dataset, session, tmp_path, start=0, end=1, num_batches=2)

    assert captured["loop_state_extra"] is not None
    assert "prompt_cursor" in captured["loop_state_extra"]


def test_no_refill_does_not_checkpoint_prompt_cursor(tmp_path, monkeypatch):
    """Default (no refill): prompt_cursor is not decoupled, so loop_state_extra
    stays None (byte-identical to the pre-refill path)."""
    import interactive_training.rl.train as T

    captured: dict = {}
    orig = T.do_train_step_and_get_sampling_client

    async def _spy(*args, **kw):
        captured["loop_state_extra"] = kw.get("loop_state_extra")
        return await orig(*args, **kw)

    monkeypatch.setattr(T, "do_train_step_and_get_sampling_client", _spy)

    dataset = ControlledDataset([[MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])]])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, strategy=DapoStrategy())  # refill_on_drop=False
    _run_sync(cfg, dataset, session, tmp_path, start=0, end=1, num_batches=1)

    assert captured["loop_state_extra"] is None


def test_refill_on_drop_resumes_from_prompt_cursor(tmp_path):
    """resume_prompt_cursor is honored: the first dataset batch drawn reflects
    the resumed cursor, not a fresh start at 0."""
    dataset = _RecordingDataset([
        [MixedRewardGroupBuilder([1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, strategy=DapoStrategy(), refill_on_drop=True)
    _run_sync(
        cfg, dataset, session, tmp_path, start=0, end=1, num_batches=2,
        resume_prompt_cursor=1,
    )

    assert dataset.requested[0] == 1


def test_refill_on_drop_returns_final_prompt_cursor(tmp_path):
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset, tmp_path, strategy=DapoStrategy(), refill_on_drop=True
    )
    final_loop_state: dict = {}

    _run_sync(
        cfg,
        dataset,
        session,
        tmp_path,
        start=0,
        end=1,
        num_batches=2,
        final_loop_state=final_loop_state,
    )

    assert final_loop_state == {"prompt_cursor": 1}
