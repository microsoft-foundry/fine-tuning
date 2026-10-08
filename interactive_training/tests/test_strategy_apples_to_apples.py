"""Apples-to-apples sanity checks for strategy comparison runs.

The benchmark compares dynamic-batching strategies against a vanilla baseline
with an identical config + seed. These tests pin what parity we DO get and
make the necessary divergences explicit:

* **Prompt draw is identical.** Every strategy (and the baseline) draws the
  same dataset batches in the same order per step -- the strategy never
  perturbs which prompts enter a step. This is the strongest apples-to-apples
  guarantee and is fully controllable.
* **Datum count diverges by design.** Strategies that DROP (dapo) or SELECT a
  subset (pods) deliberately train on fewer datums than the baseline; a
  keep-all strategy (fixed) matches the baseline exactly. That divergence is
  the effect being measured, not a confound -- we assert its shape so it can't
  regress silently.
"""

from __future__ import annotations

from interactive_training.dynamic_batching import build_strategy

from tests.test_dynamic_sampling import (
    FixedRewardGroupBuilder,
    MixedRewardGroupBuilder,
    _make_config,
    _make_mock_session,
)
from tests.test_train_loop_characterization import _RecordingDataset, _fingerprint, _run_sync

# 8-wide mixed group (p_hat = 0.5, so nothing is constant-reward).
_MIXED8 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def _run_and_record(strategy_name, tmp_path, batch_builders, *, end, num_batches):
    """Run do_sync_training with a given strategy; return (requested, fingerprint)."""
    dataset = _RecordingDataset(batch_builders)
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path / (strategy_name or "baseline"),
        sampling_seed=2,
        strategy=build_strategy(strategy_name, group_size=8, pods_keep=4, commit_rollouts=8),
        max_rolls_per_group=16,
    )
    _run_sync(cfg, dataset, session, tmp_path / (strategy_name or "baseline"),
              end=end, num_batches=num_batches)
    return dataset.requested, _fingerprint(session)


def test_all_strategies_draw_identical_prompt_sequence(tmp_path):
    # Two batches of mixed groups; run 3 steps (wraps: 0,1,0). The batch-index
    # sequence must be identical for the baseline and every strategy -- the
    # strategy decides what to do with rolled groups, never which prompts to
    # draw.
    batch_builders = [
        [MixedRewardGroupBuilder(list(_MIXED8)), MixedRewardGroupBuilder(list(_MIXED8))],
        [MixedRewardGroupBuilder(list(_MIXED8)), MixedRewardGroupBuilder(list(_MIXED8))],
    ]
    baseline_req, _ = _run_and_record(None, tmp_path, batch_builders, end=3, num_batches=2)
    assert baseline_req == [0, 1, 0]

    for name in ("fixed", "dapo", "pods", "pilot_commit"):
        req, _ = _run_and_record(name, tmp_path, batch_builders, end=3, num_batches=2)
        assert req == baseline_req, f"{name} perturbed the prompt sequence: {req}"


def test_fixed_strategy_matches_baseline_datum_count(tmp_path):
    # FixedStrategy is keep-all: identical training datums to the no-strategy
    # baseline. The parity anchor.
    batch_builders = [[MixedRewardGroupBuilder(list(_MIXED8))]]
    _, base_fp = _run_and_record(None, tmp_path, batch_builders, end=1, num_batches=1)
    _, fixed_fp = _run_and_record("fixed", tmp_path, batch_builders, end=1, num_batches=1)
    assert base_fp["n_datum"] == fixed_fp["n_datum"] == 8
    assert base_fp["n_fwd_bwd"] == fixed_fp["n_fwd_bwd"] == 1


def test_pods_trains_on_fewer_datums_than_baseline(tmp_path):
    # PODS keeps 4 of 8 per group -> half the baseline's datums, SAME prompts.
    batch_builders = [[MixedRewardGroupBuilder(list(_MIXED8))]]
    _, base_fp = _run_and_record(None, tmp_path, batch_builders, end=1, num_batches=1)
    _, pods_fp = _run_and_record("pods", tmp_path, batch_builders, end=1, num_batches=1)
    assert base_fp["n_datum"] == 8
    assert pods_fp["n_datum"] == 4         # B-axis shrink
    assert pods_fp["n_sample"] == base_fp["n_sample"]  # same rollouts drawn


def test_dapo_drops_constant_groups_vs_baseline(tmp_path):
    # Batch = one mixed (kept) + one constant (dropped by dapo). Baseline keeps
    # both; dapo trains on the mixed group only. Prompts drawn are identical.
    batch_builders = [[
        MixedRewardGroupBuilder(list(_MIXED8)),
        FixedRewardGroupBuilder(0.0, group_size=8),
    ]]
    base_req, base_fp = _run_and_record(None, tmp_path, batch_builders, end=1, num_batches=1)
    dapo_req, dapo_fp = _run_and_record("dapo", tmp_path, batch_builders, end=1, num_batches=1)
    assert base_req == dapo_req              # identical prompt draw
    assert base_fp["n_datum"] == 16          # both groups trained
    assert dapo_fp["n_datum"] == 8           # constant group dropped
    assert base_fp["n_sample"] == dapo_fp["n_sample"] == 16  # both rolled
