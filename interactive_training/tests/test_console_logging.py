"""Regression tests for the console-logging safeguards added to bound
``print_group`` output (the change that prevents a single long agentic rollout
from flooding the console and starving the async SDK heartbeat).

Covers the three behaviors surfaced in review:

1. ``_truncate_text_lines`` never returns more than ``max_lines`` and inserts an
   elision marker (off-by-one fix).
2. ``Config`` rejects negative logging knobs at the config boundary rather than
   silently treating a typo'd negative as "unlimited".
3. In the streaming-minibatch path the per-datum log cap is applied *per training
   step*, not per minibatch: the total groups handed to ``print_group`` across all
   minibatches/substeps of one step never exceeds ``num_groups_to_log``.
"""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import chz
import pytest

import interactive_training.rl.train as train
from interactive_training.rl.train import (
    Config,
    StreamMinibatchConfig,
    _config_for_logging,
    _effective_num_groups_to_log,
    _truncate_text_lines,
    do_train_step_streaming_and_get_sampling_client,
)
from interactive_training.rl.types import RLDatasetBuilder


# ---------------------------------------------------------------------------
# 1. _truncate_text_lines
# ---------------------------------------------------------------------------


class TestTruncateTextLines:
    @pytest.mark.parametrize("max_lines", [1, 2, 3, 7, 200])
    def test_truncated_output_never_exceeds_cap(self, max_lines):
        """A long input is capped to at most ``max_lines`` lines (off-by-one fix).

        Before the fix the head + marker + tail composition returned
        ``max_lines + 1`` lines (e.g. 201 at a cap of 200).
        """
        text = "\n".join(f"line {i}" for i in range(1000))
        out = _truncate_text_lines(text, max_lines)
        assert len(out.splitlines()) <= max_lines

    def test_truncation_inserts_elision_marker(self):
        text = "\n".join(f"line {i}" for i in range(1000))
        out = _truncate_text_lines(text, 10)
        assert "lines omitted" in out
        # Head and tail content is preserved around the marker.
        assert out.splitlines()[0] == "line 0"
        assert out.splitlines()[-1] == "line 999"

    def test_cap_of_one_emits_only_marker(self):
        """At a cap of 1 there is no room for head/tail, only the marker — and it
        must not fall back to dumping every line (``lines[-0:]`` guard)."""
        text = "\n".join(f"line {i}" for i in range(1000))
        out = _truncate_text_lines(text, 1)
        assert len(out.splitlines()) == 1
        assert "lines omitted" in out

    def test_short_text_passes_through_unchanged(self):
        text = "a\nb\nc"
        assert _truncate_text_lines(text, 200) == text
        # Exactly at the cap is still a passthrough (no marker).
        assert _truncate_text_lines(text, 3) == text

    @pytest.mark.parametrize("max_lines", [0, -1, -100])
    def test_non_positive_cap_is_unlimited(self, max_lines):
        """0 disables truncation; negatives are treated as no-cap (never raises) so
        the diagnostics helper can never abort a training step."""
        text = "\n".join(f"line {i}" for i in range(1000))
        assert _truncate_text_lines(text, max_lines) == text


# ---------------------------------------------------------------------------
# 2. Config validation of logging knobs
# ---------------------------------------------------------------------------


@chz.chz
class _NoopDatasetBuilder(RLDatasetBuilder):
    async def __call__(self):
        return None, None


def _make_config(tmp_path, **overrides) -> Config:
    defaults = dict(
        learning_rate=1e-5,
        dataset_builder=_NoopDatasetBuilder(),
        model_name="test-model",
        max_tokens=10,
        log_path=str(tmp_path),
        eval_every=0,
        save_every=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestConsoleLoggingValidation:
    def test_negative_num_groups_to_log_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="num_groups_to_log must be >= 0"):
            _make_config(tmp_path, num_groups_to_log=-1)

    def test_negative_max_console_lines_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_console_lines_per_datum must be >= 0"):
            _make_config(tmp_path, max_console_lines_per_datum=-5)

    def test_zero_values_accepted(self, tmp_path):
        cfg = _make_config(
            tmp_path, num_groups_to_log=0, max_console_lines_per_datum=0
        )
        assert cfg.num_groups_to_log == 0
        assert cfg.max_console_lines_per_datum == 0

    def test_sanitize_logs_disables_trajectory_logging(self, tmp_path):
        cfg = _make_config(tmp_path, num_groups_to_log=4, sanitize_logs=True)

        assert _effective_num_groups_to_log(cfg) == 0
        assert cfg.num_groups_to_log == 4

    def test_unsanitized_logs_preserve_configured_group_count(self, tmp_path):
        cfg = _make_config(tmp_path, num_groups_to_log=3, sanitize_logs=False)

        assert _effective_num_groups_to_log(cfg) == 3

    def test_sanitize_logs_redacts_dataset_builder_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, sanitize_logs=True)

        logged_config = _config_for_logging(cfg)

        assert isinstance(logged_config, dict)
        assert logged_config["dataset_builder"] == "<redacted>"
        assert "test-model" in str(logged_config)
        assert "_NoopDatasetBuilder" not in str(logged_config)


# ---------------------------------------------------------------------------
# 3. Per-step console-log budget in the streaming-minibatch path
# ---------------------------------------------------------------------------


class TestStreamingPerStepLogBudget:
    def test_log_budget_is_per_step_not_per_minibatch(self, tmp_path, monkeypatch):
        """``prepare_minibatch`` is invoked once per minibatch, so without a shared
        budget the cap would apply per minibatch and print up to
        ``num_groups_to_log * num_substeps * num_minibatches`` groups per step.

        Drive one streaming step with 4 minibatches of 2 groups each and a budget
        of 3, and assert the total number of groups actually offered to
        ``print_group`` across the step is exactly the budget (and never more).
        """
        num_groups_to_log = 3
        # groups_per_batch / num_substeps / num_minibatches => groups_per_minibatch
        # 8 / 1 / 4 = 2 groups per minibatch.
        cfg = _make_config(
            tmp_path,
            num_substeps=1,
            num_groups_to_log=num_groups_to_log,
            stream_minibatch_config=StreamMinibatchConfig(
                groups_per_batch=8, num_minibatches=4
            ),
        )

        # Capture the (budget, num_groups_in_minibatch) handed to each minibatch.
        calls: list[tuple[int, int]] = []

        async def fake_prepare_minibatch(env_builders, traj_groups, tokenizer, kl_ref, **kwargs):
            calls.append((kwargs["num_groups_to_log"], len(traj_groups)))
            return [], {}

        monkeypatch.setattr(train, "prepare_minibatch", fake_prepare_minibatch)
        monkeypatch.setattr(
            train, "compute_sampling_client_metrics", MagicMock(return_value={})
        )
        monkeypatch.setattr(
            train, "compute_trajectory_metrics", MagicMock(return_value={})
        )
        monkeypatch.setattr(
            train, "_training_logprobs_from_fwd_bwd", MagicMock(return_value=[])
        )
        monkeypatch.setattr(
            train,
            "compute_full_batch_metrics_and_get_sampling_client",
            AsyncMock(return_value=(MagicMock(), {})),
        )

        # training_client: forward_backward + optim_step return awaitable futures.
        def _make_fwd_bwd_future(*args, **kwargs):
            fut = MagicMock()
            res = MagicMock()
            res.metrics = None
            fut.result_async = AsyncMock(return_value=res)
            return fut

        training_client = MagicMock()
        training_client.forward_backward_async = AsyncMock(side_effect=_make_fwd_bwd_future)
        optim_future = MagicMock()
        optim_res = MagicMock()
        optim_res.metrics = None
        optim_future.result_async = AsyncMock(return_value=optim_res)
        training_client.optim_step_async = AsyncMock(return_value=optim_future)

        # Queue holds one full step worth of groups: 8 lightweight stand-ins.
        queue: asyncio.Queue = asyncio.Queue()

        async def _driver():
            for _ in range(8):
                queue.put_nowait(
                    types.SimpleNamespace(
                        env_group_builder=MagicMock(),
                        trajectory_group=MagicMock(),
                    )
                )
            return await do_train_step_streaming_and_get_sampling_client(
                cfg=cfg,
                i_batch=0,
                trajectory_groups_queue=queue,
                training_client=training_client,
                kl_reference_client=None,
                tokenizer=MagicMock(),
            )

        asyncio.run(_driver())

        # print_group prints min(budget, num_groups_in_minibatch) groups per call.
        total_printed = sum(min(budget, n) for budget, n in calls)
        assert total_printed <= num_groups_to_log
        assert total_printed == num_groups_to_log
        # Budget is monotonically debited across minibatches and hits 0.
        budgets = [budget for budget, _ in calls]
        assert budgets == sorted(budgets, reverse=True)
        assert budgets[-1] == 0
