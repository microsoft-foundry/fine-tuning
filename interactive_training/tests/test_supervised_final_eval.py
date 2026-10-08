"""Regression tests for SFT final-evaluation step co-location.

Training uses zero-based step indices. To keep ``train_loss`` and ``eval_loss``
on the same final step, the SFT loop must log the final evaluation alongside the
last training step, not at ``last_step + 1``. A one-based display then places
both on step ``N`` rather than adding a phantom ``N + 1`` row.

These tests pin that producer-side contract (``final_eval_step``) against the
same loop arithmetic the training loop uses, and prove it matches the RL loop's
convention (final eval at ``num_batches - 1``).
"""

from __future__ import annotations

import pytest

from interactive_training.supervised.train import (
    final_eval_step,
    should_evaluate_after_epoch,
    should_evaluate_before_step,
    should_evaluate_final_model,
)


def _simulate_last_step(
    *, start_epoch: int, start_batch: int, step_cap: int | None,
    n_batches: int, num_epochs: int,
) -> int | None:
    """Last optimizer step index executed, mirroring the SFT training loop."""
    last_step: int | None = None
    stop = False
    for epoch_idx in range(start_epoch, num_epochs):
        if stop:
            break
        sb = start_batch if epoch_idx == start_epoch else 0
        for batch_idx in range(sb, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if step_cap is not None and step >= step_cap:
                stop = True
                break
            last_step = step
    return last_step


@pytest.mark.parametrize("last_step", [0, 1, 2, 7, 100])
def test_final_eval_co_locates_with_last_step(last_step):
    # Co-located with the last training step, NOT last_step + 1 (the old phantom).
    assert final_eval_step(last_step) == last_step
    assert final_eval_step(last_step) != last_step + 1


@pytest.mark.parametrize(
    "n_batches,num_epochs,step_cap",
    [
        (1, 1, None),   # single-step run
        (9, 1, None),   # single epoch
        (9, 1, 1),      # max_steps cap
        (9, 2, None),   # multi-epoch
        (4, 3, 5),      # multi-epoch capped mid-run
    ],
)
def test_final_eval_step_matches_loop_last_step(n_batches, num_epochs, step_cap):
    last_step = _simulate_last_step(
        start_epoch=0, start_batch=0, step_cap=step_cap,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    assert last_step is not None
    # The final eval is emitted on the same step as the last training row, so
    # results.csv/W&B show a single contiguous step axis with no trailing phantom.
    assert final_eval_step(last_step) == last_step


def test_matches_rl_final_eval_convention_for_full_run():
    # RL logs its final eval at num_batches - 1 (rl/train.py). A full SFT run's
    # last_step is also num_batches - 1, so both backends co-locate identically.
    n_batches, num_epochs = 9, 1
    last_step = _simulate_last_step(
        start_epoch=0, start_batch=0, step_cap=None,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    assert final_eval_step(last_step) == n_batches - 1


@pytest.mark.parametrize("step,expected", [(0, True), (1, False), (10, True), (11, False)])
def test_step_strategy_preserves_existing_periodic_cadence(step, expected):
    assert should_evaluate_before_step(
        eval_strategy="steps",
        eval_every=10,
        step=step,
        is_fresh_run=True,
    ) is expected


def test_step_strategy_eval_every_zero_remains_disabled():
    assert not should_evaluate_before_step(
        eval_strategy="steps",
        eval_every=0,
        step=0,
        is_fresh_run=True,
    )


def test_epoch_strategy_evaluates_initial_weights_only_on_fresh_run():
    assert should_evaluate_before_step(
        eval_strategy="epoch",
        eval_every=1,
        step=0,
        is_fresh_run=True,
    )
    assert not should_evaluate_before_step(
        eval_strategy="epoch",
        eval_every=1,
        step=0,
        is_fresh_run=False,
    )
    assert not should_evaluate_before_step(
        eval_strategy="epoch",
        eval_every=1,
        step=1,
        is_fresh_run=True,
    )


def test_epoch_strategy_evaluates_only_completed_epochs():
    assert should_evaluate_after_epoch(eval_strategy="epoch", epoch_completed=True)
    assert not should_evaluate_after_epoch(eval_strategy="epoch", epoch_completed=False)
    assert not should_evaluate_after_epoch(eval_strategy="steps", epoch_completed=True)


@pytest.mark.parametrize("helper", [should_evaluate_before_step, should_evaluate_after_epoch])
def test_evaluation_helpers_reject_unknown_strategy(helper):
    kwargs = (
        {"eval_every": 10, "step": 0, "is_fresh_run": True}
        if helper is should_evaluate_before_step
        else {"epoch_completed": True}
    )
    with pytest.raises(ValueError, match="Unsupported eval_strategy"):
        helper(eval_strategy="unknown", **kwargs)


@pytest.mark.parametrize("stopped_early", [True, False])
@pytest.mark.parametrize("last_evaluated_step", [None, 0, 5])
def test_final_model_steps_strategy_tracks_eval_every(stopped_early, last_evaluated_step):
    # In steps mode the final eval mirrors legacy behavior: emitted whenever
    # periodic evaluation is enabled (eval_every > 0), regardless of early-stop
    # or which step was last evaluated.
    assert should_evaluate_final_model(
        eval_strategy="steps",
        eval_every=10,
        stopped_early=stopped_early,
        last_step=5,
        last_evaluated_step=last_evaluated_step,
    )
    assert not should_evaluate_final_model(
        eval_strategy="steps",
        eval_every=0,
        stopped_early=stopped_early,
        last_step=5,
        last_evaluated_step=last_evaluated_step,
    )


def test_final_model_never_evaluates_when_nothing_trained():
    # last_step is None only when no optimizer step ran this call (already at the
    # max_steps cap on resume); there is nothing new to evaluate.
    for strategy, eval_every in (("steps", 10), ("epoch", 1)):
        assert not should_evaluate_final_model(
            eval_strategy=strategy,
            eval_every=eval_every,
            stopped_early=True,
            last_step=None,
            last_evaluated_step=None,
        )


def test_final_model_epoch_strategy_only_on_unevaluated_early_stop():
    # A completed epoch is already evaluated at its boundary -> no final eval.
    assert not should_evaluate_final_model(
        eval_strategy="epoch",
        eval_every=1,
        stopped_early=False,
        last_step=5,
        last_evaluated_step=None,
    )
    # Mid-epoch early stop at a step not yet evaluated -> evaluate the final model.
    assert should_evaluate_final_model(
        eval_strategy="epoch",
        eval_every=1,
        stopped_early=True,
        last_step=5,
        last_evaluated_step=3,
    )
    # Early stop that lands exactly on an already-evaluated epoch boundary must
    # NOT double-evaluate the same step.
    assert not should_evaluate_final_model(
        eval_strategy="epoch",
        eval_every=1,
        stopped_early=True,
        last_step=5,
        last_evaluated_step=5,
    )


def test_final_model_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported eval_strategy"):
        should_evaluate_final_model(
            eval_strategy="unknown",
            eval_every=1,
            stopped_early=True,
            last_step=5,
            last_evaluated_step=None,
        )
