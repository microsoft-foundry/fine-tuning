"""Unit tests for crash-recovery / auto-resume of the math_rl Azure SDK recipe.

These cover the local ``checkpoints.jsonl`` ledger that drives auto-resume
(see docs/training.md#resuming-after-an-interruption):

  * ``checkpoint_utils.get_last_checkpoint`` — picks the LAST fully-resumable
    row (one that has ``state_path``), skipping sampler-only rows.
  * ``checkpoint_utils.save_checkpoint_async`` — writes a well-formed row whose
    fields are exactly what resume reads back (``name``, ``batch``,
    ``state_path``, ``sampler_path``).
  * The resume data contract that ``rl/train_azure.main`` and
    ``recipes/math_rl/train_azure.cli_main`` rely on: the resumable row carries
    both the server checkpoint pointer (``state_path``) and the loop position
    (``batch``).

Offline only — no backend, no GPU.

Run with:
    cd interactive_training && uv run pytest tests/test_resume.py -v
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from interactive_training import checkpoint_utils
from interactive_training.recipes.math_rl.train_azure import _parse_checkpoint_path
from interactive_training.rl.train import save_checkpoint_and_get_sampling_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_checkpoints(log_dir: str, rows: list[dict]) -> str:
    """Write rows to ``<log_dir>/checkpoints.jsonl`` and return the path."""
    path = os.path.join(log_dir, checkpoint_utils.CHECKPOINTS_BASE_NAME)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _make_training_client(state_path: str = "model_abc/4",
                          sampler_path: str = "4") -> MagicMock:
    """Mock AzureSDKTrainingClient whose save_* calls resolve to given paths.

    Mirrors the real contract: ``save_state_async`` / ``save_weights_for_sampler_async``
    return an awaitable future-like object exposing ``result_async()`` whose
    result has a ``.path`` attribute.
    """
    def _future_with_path(path: str) -> MagicMock:
        result = MagicMock()
        result.path = path
        fut = MagicMock()
        fut.result_async = AsyncMock(return_value=result)
        return fut

    client = MagicMock()
    client.save_state_async = AsyncMock(return_value=_future_with_path(state_path))
    client.save_weights_for_sampler_async = AsyncMock(
        return_value=_future_with_path(sampler_path)
    )
    # Sampling-client surface used by save_checkpoint_and_get_sampling_client:
    client.create_sampling_client = MagicMock(return_value="sampling-client")
    client.save_weights_and_get_sampling_client_async = AsyncMock(
        return_value="sampling-client-no-save"
    )
    return client


# ---------------------------------------------------------------------------
# get_last_checkpoint
# ---------------------------------------------------------------------------


class TestGetLastCheckpoint:
    """Auto-resume reads the LAST resumable row from checkpoints.jsonl."""

    def test_returns_none_when_file_missing(self, tmp_path):
        """Fresh run (no checkpoints.jsonl) → None, start fresh."""
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None

    def test_returns_none_when_empty_file(self, tmp_path):
        """Empty checkpoints.jsonl → None."""
        _write_checkpoints(str(tmp_path), [])
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None

    def test_returns_last_resumable_row(self, tmp_path):
        """With several rows, the LAST one with state_path wins."""
        _write_checkpoints(str(tmp_path), [
            {"name": "2", "batch": 2, "state_path": "model_abc/2", "sampler_path": "2"},
            {"name": "4", "batch": 4, "state_path": "model_abc/4", "sampler_path": "4"},
        ])
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row is not None
        assert row["batch"] == 4
        assert row["state_path"] == "model_abc/4"

    def test_skips_trailing_sampler_only_row(self, tmp_path):
        """A sampler-only row (no state_path) is NOT resumable; fall back to the
        last row that does have state_path."""
        _write_checkpoints(str(tmp_path), [
            {"name": "2", "batch": 2, "state_path": "model_abc/2", "sampler_path": "2"},
            # ephemeral sampler sync between steps — no state_path
            {"name": "step3", "sampler_path": "step3"},
        ])
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row is not None
        assert row["batch"] == 2
        assert row["state_path"] == "model_abc/2"

    def test_none_when_only_sampler_rows(self, tmp_path):
        """If NO row has state_path, there is nothing resumable → None."""
        _write_checkpoints(str(tmp_path), [
            {"name": "step1", "sampler_path": "step1"},
            {"name": "step2", "sampler_path": "step2"},
        ])
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None

    def test_custom_required_key(self, tmp_path):
        """The required_key filter is honoured (defaults to state_path)."""
        _write_checkpoints(str(tmp_path), [
            {"name": "2", "batch": 2, "sampler_path": "2"},
        ])
        # Default key (state_path) → not resumable
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None
        # Explicit sampler_path key → row is found
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path), required_key="sampler_path")
        assert row is not None and row["name"] == "2"


# ---------------------------------------------------------------------------
# save_checkpoint_async — row schema
# ---------------------------------------------------------------------------


class TestSaveCheckpointRow:
    """The row written by save_checkpoint_async is exactly what resume reads."""

    @pytest.mark.asyncio
    async def test_both_writes_full_row(self, tmp_path):
        """kind='both' writes name + loop_state + state_path + sampler_path."""
        client = _make_training_client(
            state_path="model_abc/4", sampler_path="4"
        )
        paths = await checkpoint_utils.save_checkpoint_async(
            training_client=client,
            name="4",
            log_path=str(tmp_path),
            loop_state={"batch": 4, "prompt_cursor": 9},
            kind="both",
        )
        assert paths == {
            "state_path": "model_abc/4",
            "sampler_path": "4",
        }
        rows = _read_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0] == {
            "name": "4",
            "batch": 4,
            "prompt_cursor": 9,
            "state_path": "model_abc/4",
            "sampler_path": "4",
        }

    @pytest.mark.asyncio
    async def test_state_only_omits_sampler(self, tmp_path):
        """kind='state' writes a resumable row with state_path but no sampler_path."""
        client = _make_training_client(state_path="model_abc/7")
        await checkpoint_utils.save_checkpoint_async(
            training_client=client,
            name="7",
            log_path=str(tmp_path),
            loop_state={"batch": 7},
            kind="state",
        )
        client.save_weights_for_sampler_async.assert_not_called()
        row = _read_rows(tmp_path)[0]
        assert row["state_path"] == "model_abc/7"
        assert "sampler_path" not in row
        assert row["batch"] == 7

    @pytest.mark.asyncio
    async def test_sampler_only_row_is_not_resumable(self, tmp_path):
        """kind='sampler' writes a row WITHOUT state_path → get_last_checkpoint skips it."""
        client = _make_training_client(sampler_path="step3")
        await checkpoint_utils.save_checkpoint_async(
            training_client=client,
            name="step3",
            log_path=str(tmp_path),
            loop_state={"batch": 3},
            kind="sampler",
        )
        client.save_state_async.assert_not_called()
        row = _read_rows(tmp_path)[0]
        assert "state_path" not in row
        # End-to-end: this row must NOT be picked up as resumable.
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None

    @pytest.mark.asyncio
    async def test_rows_append(self, tmp_path):
        """Successive saves append; get_last_checkpoint returns the newest."""
        client = _make_training_client(state_path="model_abc/1", sampler_path="1")
        await checkpoint_utils.save_checkpoint_async(
            training_client=client, name="1", log_path=str(tmp_path),
            loop_state={"batch": 1}, kind="both",
        )
        client2 = _make_training_client(state_path="model_abc/2", sampler_path="2")
        await checkpoint_utils.save_checkpoint_async(
            training_client=client2, name="2", log_path=str(tmp_path),
            loop_state={"batch": 2}, kind="both",
        )
        assert len(_read_rows(tmp_path)) == 2
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row["batch"] == 2


# ---------------------------------------------------------------------------
# Resume data contract — round-trip save → get_last_checkpoint → parse
# ---------------------------------------------------------------------------


class TestResumeContract:
    """The saved row carries both the server pointer and the loop position,
    and parses back into (session_id, checkpoint_name) for from_checkpoint."""

    @pytest.mark.asyncio
    async def test_round_trip_state_path_and_batch(self, tmp_path):
        client = _make_training_client(
            state_path="model_6ff3628a/4", sampler_path="4"
        )
        await checkpoint_utils.save_checkpoint_async(
            training_client=client, name="4", log_path=str(tmp_path),
            loop_state={"batch": 4}, kind="both",
        )

        # Resume side: read newest resumable row.
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row is not None

        # start_batch recovery (as in rl/train_azure.main).
        start_batch = row["batch"]
        assert start_batch == 4

        # from_checkpoint construction (as in recipes/math_rl/train_azure.cli_main).
        session_id, checkpoint_id = _parse_checkpoint_path(row["state_path"])
        assert session_id == "session_6ff3628a"
        assert checkpoint_id == "4"

    def test_start_batch_defaults_zero_without_checkpoint(self, tmp_path):
        """No resumable row → start_batch falls back to 0 (clean start)."""
        resume_info = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        start_batch = resume_info["batch"] if resume_info else 0
        assert start_batch == 0


# ---------------------------------------------------------------------------
# Checkpoint precedence — auto-resume wins over an explicit load_checkpoint_path
# (mirrors recipes/*/train_azure.cli_main; see docs/continual-fine-tuning.md)
# ---------------------------------------------------------------------------


class TestSelectResumeCheckpoint:
    """``select_resume_checkpoint`` decides which checkpoint bootstraps a session.

    Resume wins when a resumable ledger row is present so the restored weights
    stay consistent with the dataset cursor; an explicit ``load_checkpoint_path``
    only applies to a fresh ``log_path``.
    """

    def test_clean_start_when_nothing_provided(self):
        sel = checkpoint_utils.select_resume_checkpoint(None, None)
        assert sel.checkpoint_path is None
        assert sel.is_resume is False
        assert sel.ignored_load_checkpoint_path is None

    def test_explicit_path_used_on_fresh_log_path(self):
        sel = checkpoint_utils.select_resume_checkpoint(None, "model_x/final")
        assert sel.checkpoint_path == "model_x/final"
        assert sel.is_resume is False
        assert sel.ignored_load_checkpoint_path is None

    def test_resume_used_when_ledger_present(self):
        resume_info = {"state_path": "model_abc/4", "batch": 4}
        sel = checkpoint_utils.select_resume_checkpoint(resume_info, None)
        assert sel.checkpoint_path == "model_abc/4"
        assert sel.is_resume is True
        assert sel.ignored_load_checkpoint_path is None

    def test_resume_preempts_explicit_path(self):
        """The key precedence rule: a present ledger wins, and the ignored
        explicit path is surfaced so the recipe can warn."""
        resume_info = {"state_path": "model_abc/4", "batch": 4}
        sel = checkpoint_utils.select_resume_checkpoint(resume_info, "model_x/final")
        assert sel.checkpoint_path == "model_abc/4"
        assert sel.is_resume is True
        assert sel.ignored_load_checkpoint_path == "model_x/final"

    def test_sampler_only_row_is_not_a_resume(self):
        """A row without ``state_path`` isn't resumable; explicit path applies."""
        # get_last_checkpoint already filters these out, so resume_info is None.
        sel = checkpoint_utils.select_resume_checkpoint(None, "model_x/final")
        assert sel.checkpoint_path == "model_x/final"
        assert sel.is_resume is False


# ---------------------------------------------------------------------------
# Weights-and-cursor consistency — the cross-file regression guard
#
# The original bug was split across two files: the recipe chose *weights* with
# load_checkpoint_path winning, while the loop always read the *cursor* from the
# ledger — so the two could disagree (weights from one run, batch from another).
# These tests wire together the REAL building blocks from both seams against a
# shared ledger fixture and assert they reference the same checkpoint:
#   * recipe weight-selection: select_resume_checkpoint + _parse_checkpoint_path
#     (recipes/*/train_azure.cli_main)
#   * loop cursor-read: get_last_checkpoint(...)["batch"]
#     (rl/train_azure.main)
# ---------------------------------------------------------------------------


class TestResumeConsistency:
    """Weights chosen by the recipe and the cursor read by the loop must agree."""

    def _select_and_cursor(self, log_dir: str, load_checkpoint_path: str | None):
        """Run the two real seams against the same ledger; return what each picks."""
        resume_info = checkpoint_utils.get_last_checkpoint(log_dir)
        selection = checkpoint_utils.select_resume_checkpoint(
            resume_info, load_checkpoint_path
        )
        start_batch = resume_info["batch"] if resume_info else 0
        return selection, start_batch

    def test_resume_weights_and_cursor_share_the_ledger_checkpoint(self, tmp_path):
        """Ledger present + explicit load_checkpoint_path: weights come from the
        ledger's state_path (not the explicit path) AND the cursor comes from the
        same ledger row — so they can't drift apart."""
        state_path = "model_abc/7"
        _write_checkpoints(
            str(tmp_path),
            [{"name": "step-7", "batch": 7, "state_path": state_path, "sampler_path": "7"}],
        )

        selection, start_batch = self._select_and_cursor(
            str(tmp_path), load_checkpoint_path="model_other/final"
        )

        # Auto-resume wins; the explicit path is ignored (and surfaced for the warning).
        assert selection.is_resume is True
        assert selection.ignored_load_checkpoint_path == "model_other/final"
        # Weights and cursor reference the SAME prior run.
        assert _parse_checkpoint_path(selection.checkpoint_path) == _parse_checkpoint_path(
            state_path
        )
        assert start_batch == 7

    def test_continual_ft_uses_explicit_weights_and_starts_at_batch_zero(self, tmp_path):
        """Fresh log_path (no ledger): weights come from the explicit path and the
        cursor starts at 0 — a clean continual-FT pass, consistently."""
        selection, start_batch = self._select_and_cursor(
            str(tmp_path), load_checkpoint_path="model_x/final"
        )

        assert selection.is_resume is False
        assert selection.ignored_load_checkpoint_path is None
        assert _parse_checkpoint_path(selection.checkpoint_path) == _parse_checkpoint_path(
            "model_x/final"
        )
        assert start_batch == 0

    def test_clean_start_has_no_checkpoint_and_batch_zero(self, tmp_path):
        """No ledger and no explicit path: no weights restored, cursor at 0."""
        selection, start_batch = self._select_and_cursor(
            str(tmp_path), load_checkpoint_path=None
        )

        assert selection.checkpoint_path is None
        assert selection.is_resume is False
        assert start_batch == 0


# ---------------------------------------------------------------------------
# Producer side — the real training-loop save path, with only the backend mocked
# ---------------------------------------------------------------------------


class TestPeriodicSaveProducer:
    """Drives the real ``save_checkpoint_and_get_sampling_client`` (the per-batch
    save path in ``rl/train.py``) so the producer's loop_state key (``batch``) is
    pinned to what the resume side reads back. This is the link the schema-only
    tests can't catch: if the loop stopped emitting ``batch`` (e.g. renamed to
    ``step``), resume would break in production but the pure-ledger tests would
    still pass.
    """

    @pytest.mark.asyncio
    async def test_save_boundary_writes_resumable_batch_row(self, tmp_path):
        """On a save_every boundary the loop writes a row whose ``batch`` is the
        current iteration and round-trips into the right ``start_batch``."""
        client = _make_training_client(
            state_path="model_abc/4", sampler_path="4"
        )
        _, _metrics = await save_checkpoint_and_get_sampling_client(
            training_client=client,
            i_batch=4,
            log_path=str(tmp_path),
            save_every=2,
            start_batch=0,
        )
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row is not None
        assert row["name"] == "4"
        assert row["batch"] == 4  # producer must emit the key resume reads
        assert row["state_path"] == "model_abc/4"
        # Resume side recovers the exact loop position.
        assert (row["batch"] if row else 0) == 4

    @pytest.mark.asyncio
    async def test_off_boundary_writes_no_row(self, tmp_path):
        """Off both save_every and eval_every boundaries: no checkpoint row, and a
        fresh sampling client comes back without persisting."""
        client = _make_training_client()
        await save_checkpoint_and_get_sampling_client(
            training_client=client,
            i_batch=3,
            log_path=str(tmp_path),
            save_every=2,
            eval_every=0,
            start_batch=0,
        )
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None
        client.save_state_async.assert_not_called()
        client.save_weights_and_get_sampling_client_async.assert_awaited()

    @pytest.mark.asyncio
    async def test_start_batch_guard_skips_resumed_batch(self, tmp_path):
        """The ``i_batch > start_batch`` guard means the just-resumed batch is not
        re-saved (avoids rewriting the row we resumed from)."""
        client = _make_training_client()
        await save_checkpoint_and_get_sampling_client(
            training_client=client,
            i_batch=4,
            log_path=str(tmp_path),
            save_every=2,
            start_batch=4,  # resumed exactly here
        )
        assert checkpoint_utils.get_last_checkpoint(str(tmp_path)) is None

    @pytest.mark.asyncio
    async def test_eval_boundary_also_saves(self, tmp_path):
        """An eval_every boundary saves a resumable row even when not on a
        save_every boundary (so the just-evaluated iteration is recoverable)."""
        client = _make_training_client(
            state_path="model_abc/3", sampler_path="3"
        )
        await save_checkpoint_and_get_sampling_client(
            training_client=client,
            i_batch=3,
            log_path=str(tmp_path),
            save_every=1000,  # not on a save boundary
            eval_every=3,     # but on an eval boundary
            start_batch=0,
        )
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row is not None and row["batch"] == 3

    @pytest.mark.asyncio
    async def test_loop_state_extra_is_persisted(self, tmp_path):
        """Dynamic sampling threads extra loop state (prompt_cursor) through to the
        row, alongside ``batch``."""
        client = _make_training_client(
            state_path="model_abc/4", sampler_path="4"
        )
        await save_checkpoint_and_get_sampling_client(
            training_client=client,
            i_batch=4,
            log_path=str(tmp_path),
            save_every=2,
            start_batch=0,
            loop_state_extra={"prompt_cursor": 9},
        )
        row = checkpoint_utils.get_last_checkpoint(str(tmp_path))
        assert row is not None
        assert row["batch"] == 4
        assert row["prompt_cursor"] == 9


def _read_rows(tmp_path) -> list[dict]:
    path = os.path.join(str(tmp_path), checkpoint_utils.CHECKPOINTS_BASE_NAME)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
