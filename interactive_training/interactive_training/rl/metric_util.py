import asyncio
import inspect
import itertools
import logging
import math
from collections import defaultdict
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import numpy as np
from azure.ai.finetuningsessions import FineTuningSession
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk
from interactive_training.completers import SessionTokenCompleter, TokenCompleter
from interactive_training.eval.evaluators import SamplingClientEvaluator
from interactive_training.rl.rollouts import (
    AuthoritativeGroupGradingError,
    RolloutGroupExhausted,
    RolloutRetryPolicy,
    do_group_rollout,
)
from interactive_training.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    Trajectory,
    TrajectoryGroup,
)
from interactive_training.tokenizer_utils import Tokenizer
from interactive_training.utils.misc_utils import all_same, dict_mean
from interactive_training.utils import logtree

logger = logging.getLogger(__name__)


def _model_input_length(model_input: ModelInput) -> int:
    return sum(
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in model_input.chunks
    )


def _compute_by_group_metrics(trajectory_groups_P: List[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def compute_trajectory_metrics(
    trajectory_groups_P: List[TrajectoryGroup],
    taglist_P: List[list[str]],
    max_tokens: int | None = None,
) -> Dict[str, float]:
    if not trajectory_groups_P:
        # A fully-dropped batch has no trajectories to summarize.
        return {}
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)
    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P) for trajgroups in tag2trajgroups.values()
    )  # check if any tag gives us a strict subset of the full trajectory groups
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups, max_tokens).items()
            }
            out.update(prefixed_metrics)
    out.update(
        {
            f"env/all/{k}": v
            for k, v in _compute_trajectory_metrics(trajectory_groups_P, max_tokens).items()
        }
    )
    return out


def _compute_trajectory_metrics(
    trajectory_groups_P: List[TrajectoryGroup], max_tokens: int | None = None
) -> Dict[str, float]:
    """Compute metrics for the trajectory groups."""
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs_PG for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        _model_input_length(transition.ob)
        for traj in flat_trajs_PG
        for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    # Compute metrics
    metrics = {
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs_PG),
        "total_episodes": len(flat_trajs_PG),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }
    # Truncation / completion-length tail metrics. The mean ac_tokens_per_turn
    # above hides heavy-tail truncation, so we always emit the tail (max/p95/p99)
    # and, when the sampling cap (max_tokens) is known, the fraction of turns
    # that hit (or nearly hit) that cap -- i.e. completions the sampler very
    # likely truncated.
    if ac_tokens_by_turn:
        ac_arr = np.array(ac_tokens_by_turn)
        metrics["ac_tokens_max"] = float(ac_arr.max())
        metrics["ac_tokens_p95"] = float(np.percentile(ac_arr, 95))
        metrics["ac_tokens_p99"] = float(np.percentile(ac_arr, 99))
        if max_tokens is not None and max_tokens > 0:
            metrics["ac_tokens_max_configured"] = float(max_tokens)
            # "at max": within 4 tokens of the cap (effectively truncated).
            # Clamp the threshold to >= 1 so absurdly small caps (max_tokens < 5)
            # don't collapse the threshold to <= 0 and mark every turn as
            # truncated.
            at_max_threshold = max(1, max_tokens - 4)
            metrics["ac_tokens_frac_at_max"] = float(
                (ac_arr >= at_max_threshold).mean()
            )
            # "near max": used >= 95% of the cap.
            metrics["ac_tokens_frac_near_max"] = float(
                (ac_arr >= 0.95 * max_tokens).mean()
            )
    metrics["reward/total"] = np.mean(
        [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    ).item()
    # Per-transition metrics
    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    # combine traj_metrics and transition_metrics in case there's some key
    # (like format error) that appears in the per-step metrics for some envs
    # but the compute_group_rewards metric for other envs.
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


@dataclass(frozen=True)
class ValidationRecord:
    step: int
    input: str
    response: str
    ground_truth: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ValidationSkipped:
    """An application explicitly omitted this pass; do not publish reward metrics."""

    reason: str


class ValidationObserver(Protocol):
    """Observe a pass, supply rewards, or return ValidationSkipped to omit its metrics."""

    name: str
    max_tokens: int | None

    def create_record(
        self,
        step: int,
        trajectory: Trajectory,
        env: Env,
        tags: tuple[str, ...],
    ) -> ValidationRecord: ...

    def on_validation_complete(
        self,
        records: list[ValidationRecord],
    ) -> (
        Awaitable[Sequence[tuple[float, Metrics]] | ValidationSkipped | None]
        | Sequence[tuple[float, Metrics]]
        | ValidationSkipped
        | None
    ): ...


class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: RLDataset,
        max_tokens: int,
        name: str = "test",
        num_groups_to_log: int = 4,
        sample_timeout_sec: float | None = None,
        max_retries_per_trajectory: int = 0,
        max_extra_trajectory_attempts_per_group: int = 0,
        observer: ValidationObserver | None = None,
        response_format: dict[str, Any] | None = None,
        require_full_validation: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
    ):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.require_full_validation = require_full_validation
        if self.require_full_validation and not self.env_group_builders_P:
            raise ValueError(
                "Full validation requires a non-empty validation dataset"
            )
        self.max_tokens = (
            observer.max_tokens
            if observer is not None and observer.max_tokens is not None
            else max_tokens
        )
        self.name = observer.name if observer is not None else name
        self.num_groups_to_log = num_groups_to_log
        # Client-side failure tolerance for eval sampling. Mirrors the training
        # rollout path (timeout + per-slot retry) so a single wedged sample
        # (e.g. sustained HTTP 500 on one request) cannot ride the SDK error
        # budget to exhaustion and crash the whole run. By default a group
        # that still fails after retries is dropped and training continues.
        # Full validation instead rejects an incomplete pass before publication.
        self.sample_timeout_sec = sample_timeout_sec
        self.max_retries_per_trajectory = max_retries_per_trajectory
        self.max_extra_trajectory_attempts_per_group = max_extra_trajectory_attempts_per_group
        self.observer = observer
        self.response_format = response_format
        self.temperature = temperature
        self.seed = seed

    async def eval_token_completer(
        self,
        policy: TokenCompleter,
        *,
        step: int | None = None,
    ) -> dict[str, float]:
        if self.observer is not None and step is None:
            raise ValueError(
                "RLTestSetEvaluator validation publication requires a training step"
            )
        retry_policy = (
            RolloutRetryPolicy(
                max_retries_per_trajectory=self.max_retries_per_trajectory,
                max_extra_attempts_per_group=self.max_extra_trajectory_attempts_per_group,
            )
            if (
                self.max_retries_per_trajectory > 0
                or self.max_extra_trajectory_attempts_per_group > 0
            )
            else None
        )

        async def run_group_rollout(builder, i):
            enable_logging = i < self.num_groups_to_log
            with logtree.optional_enable_logging(enable=enable_logging):
                try:
                    if self.observer is None:
                        group = await do_group_rollout(
                            builder,
                            policy,
                            retry_policy=retry_policy,
                        )
                        return group, []

                    assert step is not None
                    tags = tuple(builder.logging_tags())

                    def validation_records(trajectory_group, env_group):
                        return [
                            self.observer.create_record(step, trajectory, env, tags)
                            for trajectory, env in zip(
                                trajectory_group.trajectories_G,
                                env_group,
                                strict=True,
                            )
                        ]

                    return await do_group_rollout(
                        builder,
                        policy,
                        retry_policy=retry_policy,
                        result_mapper=validation_records,
                    )
                except AuthoritativeGroupGradingError:
                    # Validation rewards are part of the training contract. Continuing
                    # without them would publish incomplete or misleading metrics.
                    raise
                except RolloutGroupExhausted:
                    logger.warning(
                        "Eval group %d exhausted its retry budget; dropping it from eval.",
                        i,
                    )
                    return None
                except Exception:
                    # Collect failed groups before enforcing full validation, so
                    # all concurrent rollouts finish their environment cleanup.
                    logger.exception(
                        "Eval group %d failed; dropping it from eval.",
                        i,
                    )
                    return None

        results_P = await asyncio.gather(
            *[run_group_rollout(builder, i) for i, builder in enumerate(self.env_group_builders_P)]
        )
        # Keep only successful groups, with their tags aligned.
        kept = [
            (result[0], builder.logging_tags(), result[1])
            for result, builder in zip(results_P, self.env_group_builders_P)
            if result is not None
        ]
        dropped = len(results_P) - len(kept)
        if self.require_full_validation and dropped:
            raise ValueError(
                "Full validation failed: "
                f"{dropped} of {len(results_P)} validation groups were dropped; "
                "refusing to publish incomplete validation metrics"
            )
        if not kept:
            logger.warning(
                "All %d eval groups failed; returning empty eval metrics (run continues).",
                len(results_P),
            )
            if self.observer is not None:
                return {
                    f"{self.name}/rows_emitted": 0.0,
                    f"{self.name}/eval_groups_dropped": float(dropped),
                    f"{self.name}/empty_boundary": 1.0,
                }
            return {}
        trajectory_groups_P = [group for group, _, _ in kept]
        taglist_P = [tags for _, tags, _ in kept]
        callback_failed = False
        records: list[ValidationRecord] = []
        validation_rewards: Sequence[tuple[float, Metrics]] | ValidationSkipped | None = None
        if self.observer is not None:
            records = list(
                itertools.chain.from_iterable(group_records for _, _, group_records in kept)
            )
            if records:
                try:
                    callback_result = self.observer.on_validation_complete(records)
                    validation_rewards = (
                        await callback_result
                        if inspect.isawaitable(callback_result)
                        else callback_result
                    )
                except AuthoritativeGroupGradingError:
                    raise
                except Exception:
                    if self.require_full_validation:
                        raise
                    logger.exception(
                        "Validation boundary callback failed; training will continue."
                    )
                    callback_failed = True

        if isinstance(validation_rewards, ValidationSkipped):
            logger.warning(
                "Skipping validation metrics at step=%s: %s", step, validation_rewards.reason,
            )
            return {
                f"{self.name}/validation_skipped": 1.0,
                f"{self.name}/rows_emitted": float(len(records)),
            }

        if validation_rewards is not None:
            validation_rewards = list(validation_rewards)
            if len(validation_rewards) != len(records):
                raise ValueError(
                    "Validation observer returned "
                    f"{len(validation_rewards)} rewards for {len(records)} records"
                )
            reward_index = 0
            for trajectory_group in trajectory_groups_P:
                group_size = len(trajectory_group.trajectories_G)
                group_rewards = validation_rewards[
                    reward_index : reward_index + group_size
                ]
                for reward, _ in group_rewards:
                    if not math.isfinite(reward):
                        raise ValueError(
                            f"Validation observer returned non-finite reward {reward}"
                        )
                trajectory_group.final_rewards_G = [
                    reward for reward, _ in group_rewards
                ]
                trajectory_group.metrics_G = [
                    dict(reward_metrics) for _, reward_metrics in group_rewards
                ]
                reward_index += group_size

        metrics = compute_trajectory_metrics(
            trajectory_groups_P, taglist_P, max_tokens=self.max_tokens
        )
        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        if dropped:
            metrics[f"{self.name}/eval_groups_dropped"] = float(dropped)
        if self.observer is not None:
            metrics[f"{self.name}/rows_emitted"] = float(len(records))
            if not records:
                metrics[f"{self.name}/empty_boundary"] = 1.0
            if callback_failed:
                metrics[f"{self.name}/callback_failed"] = 1.0
        return metrics

    async def __call__(
        self,
        sampling_client: FineTuningSession,
        *,
        step: int | None = None,
    ) -> dict[str, float]:
        policy = SessionTokenCompleter(
            sampling_client,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            response_format=self.response_format,
            sample_timeout_sec=self.sample_timeout_sec,
        )
        return await self.eval_token_completer(policy, step=step)
