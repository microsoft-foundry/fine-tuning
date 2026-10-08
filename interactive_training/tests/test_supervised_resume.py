"""Regression tests for SFT resume after a ``max_steps``-capped run.

Bug (surfaced by the interactive-post-training-bench e2e ``sft_resume`` stage): the supervised final
checkpoint hardcoded ``loop_state={"epoch": num_epochs, "batch": n_batches}``, so
a run stopped early by ``max_steps`` stamped itself *fully complete*. A later
auto-resume then read ``start_epoch == num_epochs``, the training loop
``range(start_epoch, num_epochs)`` was empty, and the run no-op'd
("Training was already complete; nothing to do") without writing a new
checkpoint.

``compute_resume_loop_state`` fixes this by recording the *actual* stopped
position. These tests pin that contract and, crucially, replay the exact
continual→resume chain the e2e exercises to prove a capped run can be resumed
and advanced.
"""

from __future__ import annotations

import pytest

from interactive_training.supervised.train import compute_resume_loop_state


# --------------------------------------------------------------------------- #
# Faithful mirror of the supervised training loop's stop condition, so the test
# exercises the same (epoch, batch, step_cap) arithmetic as train.main().
# --------------------------------------------------------------------------- #
def _simulate_run(
    *, start_epoch: int, start_batch: int, step_cap: int | None,
    n_batches: int, num_epochs: int,
) -> tuple[int | None, bool]:
    """Return ``(last_step, entered_save_branch)`` for one train() call.

    Mirrors ``for epoch_idx in range(start_epoch, num_epochs): for batch_idx ...``
    with the ``step >= step_cap`` early stop. ``entered_save_branch`` reflects the
    ``if start_epoch < num_epochs`` guard that gates the final checkpoint (and
    otherwise logs "already complete; nothing to do").
    """
    entered_save = start_epoch < num_epochs
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
            last_step = step  # this batch trains
    return last_step, entered_save


# --------------------------------------------------------------------------- #
# 1. Unit contract for compute_resume_loop_state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "last_step,start_epoch,start_batch,n_batches,num_epochs,expected",
    [
        # max_steps=1 cap on a 9-batch/1-epoch dataset: trained step 0, resume at
        # batch 1 — NOT the old {epoch:1, batch:9} "complete" stamp.
        (0, 0, 0, 9, 1, {"epoch": 0, "batch": 1}),
        # resume-to-extend: started at batch 1, trained step 1, resume at batch 2.
        (1, 0, 1, 9, 1, {"epoch": 0, "batch": 2}),
        # genuinely complete single-epoch run: epoch advances to num_epochs.
        (8, 0, 0, 9, 1, {"epoch": 1, "batch": 0}),
        # multi-epoch: finished epoch 0, resume at epoch 1 batch 0 (still < 2).
        (8, 0, 0, 9, 2, {"epoch": 1, "batch": 0}),
        # multi-epoch complete: epoch advances to num_epochs=2.
        (17, 0, 0, 9, 2, {"epoch": 2, "batch": 0}),
        # nothing trained this call (already at cap): keep the start position.
        (None, 0, 3, 9, 1, {"epoch": 0, "batch": 3}),
        # degenerate empty batch set: preserve historical complete marker, no /0.
        (None, 0, 0, 0, 1, {"epoch": 1, "batch": 0}),
    ],
)
def test_compute_resume_loop_state(
    last_step, start_epoch, start_batch, n_batches, num_epochs, expected
):
    assert compute_resume_loop_state(
        last_step=last_step,
        start_epoch=start_epoch,
        start_batch=start_batch,
        n_batches=n_batches,
        num_epochs=num_epochs,
    ) == expected


def test_capped_run_stays_resumable_not_complete():
    """The core regression: a max_steps-capped run must NOT stamp itself complete."""
    n_batches, num_epochs = 9, 1
    ls = compute_resume_loop_state(
        last_step=0, start_epoch=0, start_batch=0,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    # Old (buggy) value was {"epoch": num_epochs, "batch": n_batches}, which makes
    # range(start_epoch, num_epochs) empty on resume -> no-op.
    assert ls != {"epoch": num_epochs, "batch": n_batches}
    assert ls["epoch"] < num_epochs, "capped run must remain resumable"


# --------------------------------------------------------------------------- #
# 2. End-to-end chain replay: e2e sft_continual (max_steps=1) -> sft_resume
#    (max_steps=2). Proves the second run actually trains and the checkpoint
#    position advances — the exact assertion the e2e require_advance check makes.
# --------------------------------------------------------------------------- #
def test_continual_then_resume_advances():
    n_batches, num_epochs = 9, 1

    # Stage 1 (behavior=delete, max_steps=1): fresh run, trains step 0.
    last1, saved1 = _simulate_run(
        start_epoch=0, start_batch=0, step_cap=1,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    assert saved1 and last1 == 0
    ls1 = compute_resume_loop_state(
        last_step=last1, start_epoch=0, start_batch=0,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    assert ls1 == {"epoch": 0, "batch": 1}

    # Stage 2 (behavior=resume, max_steps=2): reads ls1 as its start position.
    se, sb = ls1["epoch"], ls1["batch"]
    # With the OLD hardcoded {num_epochs, n_batches}, se would equal num_epochs and
    # this stage would log "already complete; nothing to do".
    assert se < num_epochs, "resume must enter the training loop, not no-op"

    last2, saved2 = _simulate_run(
        start_epoch=se, start_batch=sb, step_cap=2,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    assert saved2, "resume stage must enter the save branch"
    assert last2 == 1, "resume must train the 2nd optimizer step (step index 1)"

    ls2 = compute_resume_loop_state(
        last_step=last2, start_epoch=se, start_batch=sb,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    # Checkpoint position advanced past stage 1 (batch 1 -> batch 2): the
    # e2e's require_advance / "checkpoint count grows" bar is satisfied.
    assert ls2 == {"epoch": 0, "batch": 2}
    assert (ls2["epoch"], ls2["batch"]) > (ls1["epoch"], ls1["batch"])


def test_old_hardcoded_state_would_have_no_op_d():
    """Documents why the old behavior failed: resume reads 'complete' and stops."""
    n_batches, num_epochs = 9, 1
    old_ls = {"epoch": num_epochs, "batch": n_batches}  # pre-fix hardcode
    # A resume reading this starts at epoch == num_epochs -> empty loop -> no-op.
    start_epoch = old_ls["epoch"]
    assert not (start_epoch < num_epochs)
    last, entered_save = _simulate_run(
        start_epoch=start_epoch, start_batch=old_ls["batch"], step_cap=2,
        n_batches=n_batches, num_epochs=num_epochs,
    )
    assert last is None and not entered_save, "old state => 'already complete', no training"
