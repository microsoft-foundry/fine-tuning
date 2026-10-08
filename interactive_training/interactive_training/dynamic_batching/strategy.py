"""``DynamicBatchStrategy`` ABC and the open-algorithm baselines.

The ABC defines the two-primitive contract (``allocate``/``select``);
``FixedStrategy`` (plain GRPO) and ``DapoStrategy`` (constant-reward
dropping) are the two
baselines whose behaviour needs only plain mean-baseline math. They
double as parity anchors: ``FixedStrategy`` must reproduce the vanilla
``compute_advantages`` path, and ``DapoStrategy`` must reproduce
DAPO-style dynamic sampling.

References (each strategy re-implements the cited method's decision rule
against this ABC; see the per-class docstrings for the exact mapping):

* DAPO -- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning
  System at Scale", https://arxiv.org/abs/2503.14476
* PODS -- Xu et al., "Not All Rollouts are Useful: Down-Sampling Rollouts
  in LLM Reinforcement Learning" (TMLR 2026),
  https://arxiv.org/abs/2504.13818
* Pilot-Commit -- Kim et al., "Spend Your Rollouts Where It Counts:
  Rollout Allocation for Group-Based RL Post-Training",
  https://arxiv.org/abs/2605.26606

Each reports its gain as an efficiency win at *fixed accuracy* or *fixed
budget* (cumulative rollouts, training compute, or wall-clock), NOT an
equal-step accuracy comparison -- see
docs/experiment-write-ups/dynamic-batching-strategies.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

import torch

from interactive_training.dynamic_batching.types import Allocation, Selection

if TYPE_CHECKING:
    from interactive_training.rl.types import TrajectoryGroup


class DynamicBatchStrategy(ABC):
    """Pluggable batch-construction strategy.

    Decides how to *allocate* rollouts (A-axis) and how to *select*
    which rolled samples train (B-axis). Implementations are stateless:
    every method is a pure function of its inputs, and all mutable
    bookkeeping (in-flight pools, accumulated weight, staleness, and the
    dataset cursor) lives in the loop/scheduler that drives the strategy.
    The same strategy object
    therefore runs unchanged under a simple synchronous test scheduler or
    the production async one.
    """

    # ---- A-axis: roll-side allocation --------------------------
    @abstractmethod
    def allocate(
        self, group: "TrajectoryGroup", *, history: Any = None
    ) -> Allocation | Awaitable[Allocation]:
        """Decide this group's fate after a roll wave.

        ``history`` is reserved for a future stateful extension. Current loops
        always pass ``None`` and current stateless strategies ignore it.
        Returns one of:

        * ``Allocation.MORE``   -- re-roll this instance (needs more signal).
        * ``Allocation.DROP``   -- discard (coarse group-level filter).
        * ``Allocation.ADMIT``  -- admit this group into the batch.
        """
        ...

    # ---- B-axis: train-side selection --------------------------
    @abstractmethod
    def select(
        self, batch: "list[TrajectoryGroup]"
    ) -> list[Selection] | Awaitable[list[Selection]]:
        """Per group, choose surviving samples and assign advantages.

        Returns one :class:`Selection` per input group (aligned by
        position). The surviving set may be smaller than the input.
        """
        ...


def _mean_baseline_selection(group: "TrajectoryGroup") -> Selection:
    """Identity keep + GRPO mean-centered advantages for one group.

    Mirrors :func:`interactive_training.rl.data_processing.compute_advantages`:
    every sample survives with unit weight and advantage ``r - mean(r)``.
    """
    rewards = group.get_total_rewards()
    n = len(rewards)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = rewards_t - rewards_t.mean() if n > 0 else rewards_t
    return Selection(
        kept_indices=list(range(n)),
        weights=[1.0] * n,
        advantages=advantages,
    )


def _subset_mean_baseline_selection(
    group: "TrajectoryGroup", kept_indices: list[int]
) -> Selection:
    """Deterministic subset + mean-centered advantages over the survivors.

    Shared by the deterministic-subset strategies (PODS): keep an
    explicit index set with unit importance weights (the subset is biased
    by construction, which these methods accept) and center advantages
    within the kept rewards. Indices are emitted in ascending order so
    they line up with the loop's ``_materialize_subgroup`` field-slicing.
    """
    kept = sorted(set(kept_indices))
    rewards = group.get_total_rewards()
    kept_rewards = torch.tensor([rewards[i] for i in kept], dtype=torch.float32)
    advantages = (
        kept_rewards - kept_rewards.mean() if len(kept) > 0 else kept_rewards
    )
    return Selection(
        kept_indices=kept,
        weights=[1.0] * len(kept),
        advantages=advantages,
    )


def _max_variance_subset(rewards: list[float], m: int) -> list[int]:
    """Indices of the size-``m`` subset with maximal reward variance.

    Implements PODS max-variance down-sampling (Xu et al., 2504.13818,
    Lemma 3.1 / Algorithm 2). By Lemma 3.1 the variance-maximizing subset
    is always ``k`` highest-reward rollouts plus ``(m - k)`` lowest for
    some ``k in {0..m}``, so we sort once and scan ``k`` with prefix sums
    in ``O(n log n)``. For binary rewards this reduces to ``m/2`` highest
    + ``m/2`` lowest (their Theorem 2), but for the general (non-binary)
    reward case the optimal split is not symmetric -- which is why we scan
    rather than hard-code halves.

    Returns original indices (unsorted); the caller sorts.
    """
    n = len(rewards)
    if m >= n:
        return list(range(n))
    order = sorted(range(n), key=lambda i: rewards[i])  # ascending reward
    sorted_r = [rewards[i] for i in order]
    pre = [0.0] * (n + 1)
    pre2 = [0.0] * (n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] + sorted_r[i]
        pre2[i + 1] = pre2[i] + sorted_r[i] * sorted_r[i]
    best_k = 0
    best_var = -1.0
    for k in range(0, m + 1):
        low = m - k  # take `low` lowest (front) + `k` highest (back)
        s = pre[low] + (pre[n] - pre[n - k])
        s2 = pre2[low] + (pre2[n] - pre2[n - k])
        mean = s / m
        var = s2 / m - mean * mean
        if var > best_var:
            best_var = var
            best_k = k
    low = m - best_k
    positions = list(range(0, low)) + list(range(n - best_k, n))
    return [order[p] for p in positions]


class FixedStrategy(DynamicBatchStrategy):
    """Plain GRPO: fixed rollout budget, keep everything, mean baseline.

    A-axis is the identity (always ``admit``, never ``MORE``/``DROP``);
    B-axis keeps all samples with GRPO mean-centered advantages. This is
    the parity anchor for the vanilla ``compute_advantages`` path.
    """

    def allocate(self, group: "TrajectoryGroup", *, history: Any = None) -> Allocation:
        return Allocation.ADMIT

    def select(self, batch: "list[TrajectoryGroup]") -> list[Selection]:
        return [_mean_baseline_selection(g) for g in batch]


class DapoStrategy(DynamicBatchStrategy):
    """DAPO-style dynamic sampling: drop constant-reward groups.

    A-axis drops any group whose rewards are all equal (zero gradient),
    otherwise admits with no over-sampling; B-axis keeps everything with
    GRPO mean-centered advantages. Parity anchor for today's
    ``dynamic_sampling`` / ``remove_constant_reward_groups`` behaviour.
    """

    def allocate(self, group: "TrajectoryGroup", *, history: Any = None) -> Allocation:
        rewards = group.get_total_rewards()
        if len(rewards) > 1 and min(rewards) != max(rewards):
            return Allocation.ADMIT
        if len(rewards) <= 1:
            # Degenerate single-sample group: admit (no variance to judge).
            return Allocation.ADMIT
        return Allocation.DROP

    def select(self, batch: "list[TrajectoryGroup]") -> list[Selection]:
        return [_mean_baseline_selection(g) for g in batch]


class PodsStrategy(DynamicBatchStrategy):
    """PODS: no over-sampling, keep a max-variance reward subset.

    A-axis is the identity (``admit`` everything, never ``MORE``); the
    whole method is the B-axis. From the rolled rewards it keeps the
    size-``keep`` subset of maximal reward variance (Xu et al.,
    2504.13818, Lemma 3.1): the ``k`` highest- and ``keep - k`` lowest-
    reward responses for the ``k`` that maximizes variance, found by an
    ``O(n log n)`` scan. For binary rewards this collapses to ``keep/2``
    highest + ``keep/2`` lowest (their Theorem 2); for graded rewards the
    optimal split is asymmetric. Advantages are mean-centered over the
    *selected* subset, matching the paper's Appendix A.3 finding that
    normalizing over the down-sampled batch beats normalizing over the
    full pool. The kept subset is deterministic, so importance weights are
    unit -- PODS accepts the selection bias by construction.

    This exercises the contract's claim that ``select`` may return a
    SMALLER group than ``allocate`` admitted, with no re-roll involved.
    """

    def __init__(self, keep_per_group: int) -> None:
        self.keep = int(keep_per_group)

    def allocate(self, group: "TrajectoryGroup", *, history: Any = None) -> Allocation:
        return Allocation.ADMIT

    def select(self, batch: "list[TrajectoryGroup]") -> list[Selection]:
        out: list[Selection] = []
        for group in batch:
            rewards = group.get_total_rewards()
            n = len(rewards)
            if n <= self.keep or self.keep <= 0:
                # Nothing to prune: keep everything (mean baseline).
                out.append(_mean_baseline_selection(group))
                continue
            kept = _max_variance_subset(rewards, self.keep)
            out.append(_subset_mean_baseline_selection(group, kept))
        return out


class PilotCommitStrategy(DynamicBatchStrategy):
    """Pilot-Commit: spend rollouts on high reward-variance prompts.

    Kim et al. (2605.26606) split the per-step rollout budget into two
    stages. A cheap **pilot** wave estimates each prompt's success rate
    ``p_hat`` (the mean binary reward); for binary rewards the reward
    variance is ``p_hat * (1 - p_hat)``, so ``p_hat`` is a direct proxy
    for how informative the prompt's gradient will be. Prompts whose
    ``p_hat`` lands in a high-variance band ``[p_lower, p_upper]`` get the
    remaining **commit** rollouts; prompts outside the band are dropped
    (too easy -> evict, too hard -> defer). Training then uses the union
    of pilot and commit rollouts with plain GRPO advantages.

    On this contract the whole method is the **A-axis**: it is
    ``allocate`` returning ``DROP`` (out of band), ``MORE`` (in band, keep
    committing) or ``ADMIT`` (committed to target). The band filter is a
    *continuous generalization of DAPO's constant-reward DROP*: with
    ``p_lower -> 0+`` and ``p_upper -> 1-`` it recovers "drop only the
    all-same groups." The B-axis is untouched -- ``select`` is identical
    to :class:`FixedStrategy` (keep everything, GRPO mean baseline),
    because Pilot-Commit's entire contribution is *where* rollouts are
    spent, not how survivors are scored.

    Two concerns stay in the loop by the contract's own boundaries and are
    NOT the strategy's job:

    * the batch-full cap (retain at most ``b_t`` committed prompts) -- the
      loop's batch-fullness gate;
    * the evict-vs-defer dataset fate of a dropped prompt, plus the replay
      buffer and pilot/commit binding -- dataset-lifecycle + systems
      concerns the loop owns. The strategy only says ``DROP``; the loop
      reads the same ``p_hat`` if it wants to choose evict vs defer.

    ``pilot_rollouts`` must equal the sampling group size (the initial
    roll wave); commit rollouts are added by the loop in wave-sized
    increments up to ``pilot_rollouts + commit_rollouts``.
    """

    def __init__(
        self,
        *,
        p_lower: float = 0.125,
        p_upper: float = 0.75,
        pilot_rollouts: int = 1,
        commit_rollouts: int = 0,
    ) -> None:
        self.p_lower = float(p_lower)
        self.p_upper = float(p_upper)
        self.pilot_rollouts = int(pilot_rollouts)
        self.commit_rollouts = int(commit_rollouts)

    def allocate(self, group: "TrajectoryGroup", *, history: Any = None) -> Allocation:
        rewards = group.get_total_rewards()
        n = len(rewards)
        if n == 0:
            return Allocation.ADMIT
        target = self.pilot_rollouts + self.commit_rollouts
        if n <= self.pilot_rollouts:
            # Pilot decision point: gate on the success-rate band ONCE,
            # while the pool is still at pilot size. p_hat = mean binary
            # reward proxies reward variance p_hat * (1 - p_hat).
            p_hat = sum(rewards) / n
            if p_hat < self.p_lower or p_hat > self.p_upper:
                # Out of the high-variance band: too hard (defer) or too
                # easy (evict). The strategy only drops; the loop owns the
                # evict-vs-defer dataset fate.
                return Allocation.DROP
            # In band: begin committing if there is commit budget left.
            return Allocation.MORE if target > n else Allocation.ADMIT
        # Committed: draw commit rollouts until the target, then admit. A
        # committed prompt is never dropped -- commit is unconditional, so
        # the band is not re-checked as the pool grows.
        return Allocation.MORE if n < target else Allocation.ADMIT

    def select(self, batch: "list[TrajectoryGroup]") -> list[Selection]:
        # Train on the union of pilot + commit rollouts with plain GRPO
        # advantages: identical to FixedStrategy, since Pilot-Commit's
        # contribution is entirely the A-axis allocation.
        return [_mean_baseline_selection(g) for g in batch]
