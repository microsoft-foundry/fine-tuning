"""
Implements RL on general MDPs
"""

import asyncio
import inspect
import io
import json
import logging
import os
import secrets
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Coroutine, Iterable, Iterator, List, Literal, Sequence, TypeVar

import chz
import numpy as np
import torch
from azure.ai.finetuningsessions import FineTuningSession, FineTuningSessionClient
from azure.ai.finetuningsessions.models import (
    AdamParams,
    Datum,
    ForwardBackwardOperationResult,
    LossFn as LossFnType,
    OptimStepOperationResult,
)
from tqdm import tqdm

from interactive_training import checkpoint_utils
from interactive_training.completers import SessionTokenCompleter
from interactive_training.display import colorize_example
from interactive_training.dynamic_batching.strategy import DynamicBatchStrategy
from interactive_training.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from interactive_training.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from interactive_training.rl.metric_util import (
    RLTestSetEvaluator,
    ValidationObserver,
    compute_trajectory_metrics,
)
from interactive_training.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
from interactive_training.rl.rollouts import (
    GroupDropCallback,
    GroupDropEvent,
    IncompleteGroupGradingError,
    RolloutFailureRateExceeded,
    RolloutFailureTracker,
    RolloutGroupExhausted,
    RolloutRetryPolicy,
    do_group_rollout,
)
from interactive_training.rl.types import (
    AutonomousRolloutGroupBuilder,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    RolloutSamplingOptions,
    SamplingCheckpoint,
    TrainingRolloutContext,
    TrajectoryGroup,
)
from interactive_training.tokenizer_utils import Tokenizer
from interactive_training.utils import logtree, ml_log
from interactive_training.utils.misc_utils import all_same, safezip, split_list, timed
from interactive_training.utils.trace import scope, trace_init, update_scope_context

logger = logging.getLogger(__name__)

EvaluationStrategy = Literal["steps", "epoch"]

T = TypeVar("T")


def compute_effective_end(
    *,
    start_batch: int,
    num_batches: int,
    max_steps: int | None,
) -> int:
    """Return the effective end batch index given an optional ``max_steps`` cap.

    Shared by the direct (``rl/train.py``) and Azure (``rl/train_azure.py``)
    entrypoints so both honor the ``max_steps`` knob on ``Config`` (and the
    recipe CLIs) identically.

    - ``max_steps=None`` → train through the whole dataset.
    - ``max_steps`` set  → cap at ``start_batch + max_steps`` (never beyond
      ``num_batches``; never below ``start_batch``).
    """
    if max_steps is None:
        return num_batches
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    return min(num_batches, start_batch + max_steps)


def _derive_group_seed(base_seed: int | None, batch_idx: int, group_idx: int) -> int | None:
    """Derive a deterministic per-group seed from the base config seed.

    Returns None when base_seed is None (non-reproducible mode -- server
    picks a random seed per call).  When set, produces a unique seed per
    (batch, group) combination so that:
    - Different groups within a batch get different seeds (diversity).
    - The same (batch, group) pair always gets the same seed (reproducibility).
    """
    if base_seed is None:
        return None
    # Use large primes to spread seeds and avoid collisions across batches/groups.
    return (base_seed + batch_idx * 100003 + group_idx * 1009) % (2**31)


def _derive_async_group_seed(
    base_seed: int | None,
    group_ordinal: int,
    groups_per_batch: int,
) -> int | None:
    """Derive an async seed from deterministic enqueue order only."""
    return _derive_group_seed(
        base_seed,
        group_ordinal // groups_per_batch,
        group_ordinal % groups_per_batch,
    )


async def _resolve_strategy_result(result: Any) -> Any:
    """Accept both legacy direct strategy results and async implementations."""
    return await result if inspect.isawaitable(result) else result


@chz.chz
class KLReferenceConfig:
    """Configuration for the KL penalty reference model.

    If not specified in Config, the training model's base model is used.
    """

    base_model: str
    load_checkpoint_path: str | None = None


async def gather_with_progress(
    coroutines: Iterable[Coroutine[Any, Any, T]],
    desc: str,
    max_concurrency: int | None = None,
) -> list[T]:
    """
    Run coroutines concurrently with a progress bar that updates as each completes.

    This preserves the order of results (like asyncio.gather) while providing
    real-time progress feedback as individual coroutines complete.

    Args:
        max_concurrency: If set, limits concurrent coroutines via a semaphore.
            Useful for RL sampling where each group spawns many HTTP requests
            and unbounded concurrency overwhelms the request store.
    """
    coroutine_list = list(coroutines)
    pbar = tqdm(total=len(coroutine_list), desc=desc)
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def track(coro: Coroutine[Any, Any, T]) -> T:
        if sem is not None:
            async with sem:
                result = await coro
        else:
            result = await coro
        pbar.update(1)
        return result

    try:
        results = await asyncio.gather(*[track(coro) for coro in coroutine_list])
    finally:
        pbar.close()

    return results


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    return (
        evaluator.name
        if isinstance(evaluator, RLTestSetEvaluator) and evaluator.name is not None
        else ""
    )


_LOGTREE_EXPLANATION = (
    "This HTML log was generated by logtree during RL training. "
    "It shows rollouts and rewards for a subset of trajectory groups in this iteration. "
    "To customize what gets logged, modify the logtree calls in your Env implementation "
    "(see examples in interactive_training/recipes/)."
)


@contextmanager
def _get_logtree_scope(
    log_path: str | None, num_groups_to_log: int, f_name: str, scope_name: str
) -> Iterator[None]:
    """
    Creates a context manager; all log inside this context will be logged under the section `scope_name`.
    It will create a file with the path of log_path/f_name.html
    If num_groups_to_log is 0, it will disable logging (but note that this function does not actually implement the logic for logging itself!)
    """
    if log_path is not None and num_groups_to_log > 0:
        logtree_path = os.path.join(log_path, f"{f_name}.html")
        with logtree.init_trace(scope_name, path=logtree_path):
            logtree.log_text(_LOGTREE_EXPLANATION)
            yield
    else:
        yield


@scope
def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


def _truncate_text_lines(text: str, max_lines: int) -> str:
    """Cap a multi-line string to ``max_lines`` by keeping a head and tail with an
    elision marker. Prevents a single long (agentic/tool-calling) trajectory from
    flooding the console with hundreds of thousands of decoded-token lines.

    ``max_lines == 0`` disables truncation (unlimited). Callers must pass a
    non-negative value; ``max_console_lines_per_datum`` is validated to be >= 0 on
    the Config, so negatives never reach here. A non-positive value is treated as
    "no cap" rather than raised, to keep this diagnostics helper from ever aborting
    a training step."""
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    # Reserve one line for the elision marker so the result stays within max_lines.
    keep = max_lines - 1
    head = keep // 2
    tail = keep - head
    omitted = len(lines) - (head + tail)
    marker = f"... [{omitted} lines omitted] ..."
    return "\n".join(lines[:head] + [marker] + (lines[-tail:] if tail > 0 else []))


@scope
def print_group(
    traj_group: TrajectoryGroup, tokenizer: Tokenizer, max_lines_per_datum: int = 200
) -> None:
    """
    Print a subset of the trajectory group to the console.
    """
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    @scope
    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(_truncate_text_lines(colorize_example(datum, tokenizer, key="advantages"), max_lines_per_datum))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


def _training_logprobs_from_fwd_bwd(
    fwd_bwd_result: ForwardBackwardOperationResult,
) -> list[torch.Tensor]:
    return [torch.tensor(output["logprobs"].data) for output in fwd_bwd_result.loss_fn_outputs]


@scope
async def train_step(
    data_D: List[Datum],
    training_client: FineTuningSession,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    loss_fn_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories.

    Pipelines forward_backward and optim_step so they land on the same clock cycle.
    """
    if not data_D:
        return []

    batches = split_list(data_D, min(num_substeps, len(data_D)))

    adam_params = AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs_D: list[torch.Tensor] = []
    optim_result: OptimStepOperationResult | None = None

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(
        batches[0], loss_fn=loss_fn, loss_fn_config=loss_fn_config
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue next batch before consuming current results (to stay on same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                batches[i + 1],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None
        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
        if metrics is not None and getattr(fwd_bwd_result, "metrics", None):
            metrics.update(fwd_bwd_result.metrics)
        optim_result = await optim_future.result_async()
        # Move to next iteration
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future

    if metrics is not None and optim_result is not None and optim_result.metrics:
        metrics.update(optim_result.metrics)

    return training_logprobs_D


@chz.chz
class StreamMinibatchConfig:
    """
    Configuration for training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready.
    """

    # Total number of trajectory groups across all minibatches and substeps
    groups_per_batch: int
    # For each substep, we will divide up the number of trajectory groups
    # into this many minibatches.
    # We will do num_minibatches forward_backward() passes and one optim_step()
    # per substep.
    num_minibatches: int


@chz.chz
class AsyncConfig:
    """Configuration for async RL training"""

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int
    # Advisory watchdog: if no training step completes for this many seconds,
    # log an actionable "no progress" warning (does NOT kill the run). Helps an
    # operator distinguish a healthy-but-slow run (e.g. cold inference fleet)
    # from a wedged one. Set <= 0 to disable. Default 30 min is generous enough
    # to avoid false alarms on a cold large-model first step.
    #
    # This is deliberately ~3x the SDK's per-operation poll-warn trigger
    # (_POLL_WARN_SEC, default 600s in _patch.py). The two watch different
    # scopes: the SDK warning fires on a single stuck backend op (one sampling
    # submit, one checkpoint upload, etc.), whereas this heartbeat watches a
    # whole training step end-to-end (rollouts + forward/backward + optim + KL
    # ref + possible checkpoint upload), which legitimately takes much longer.
    # Keeping it above the SDK trigger means the fine-grained per-op warning
    # surfaces first and usually already explains the stall, while this coarse
    # last-resort line only speaks up when those didn't -- avoiding duplicate
    # noise for the same underlying stall. Do not "fix" the 1800-vs-600 gap;
    # it is intentional.
    no_progress_warn_sec: float = 1800.0


@chz.chz
class Config:
    """Configuration for RL training."""

    # -------------------------------------------------------------------------
    # Core parameters (recommended to set for nearly all runs)
    # -------------------------------------------------------------------------
    # Base learning rate used by Adam.
    learning_rate: float
    # Builds the RL dataset; also determines number of groups per batch.
    dataset_builder: RLDatasetBuilder
    # Model name (base weights) to train.
    model_name: str
    # Tokenizer name (defaults to model_name if not specified).
    tokenizer_name: str | None = None
    # Maximum number of generated tokens per rollout trajectory.
    max_tokens: int
    # Directory for checkpoints, logs, and traces.
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    # Evaluation scheduling. RL performs one dataset traversal, so "epoch"
    # means initial weights plus the model after that traversal completes.
    eval_strategy: EvaluationStrategy = "steps"
    # Evaluation cadence: steps use training iterations (0 = disabled), while
    # epoch strategy requires 1.
    eval_every: int = 20
    # Checkpoint cadence in training iterations (0 = disabled).
    save_every: int = 20
    # Optional evaluators run during training.
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    # Optional observer for the built-in validation evaluator.
    validation_observer: ValidationObserver | None = None
    # Reject empty/incomplete built-in validation passes and observer failures.
    require_full_validation: bool = False
    # Start training from weights at this checkpoint (fresh optimizer state).
    load_checkpoint_path: str | None = None
    # Optional W&B project and run name.
    wandb_project: str | None = None
    wandb_name: str | None = None

    # -------------------------------------------------------------------------
    # KL penalty configuration (advanced)
    # -------------------------------------------------------------------------
    # KL penalty coefficient against reference policy (0 = disabled).
    kl_penalty_coef: float = 0.0
    # Optional position discount for KL penalty terms.
    kl_discount_factor: float = 0.0
    # Required when kl_penalty_coef > 0.
    kl_reference_config: KLReferenceConfig | None = None

    # -------------------------------------------------------------------------
    # Loss and optimizer behavior (advanced)
    # -------------------------------------------------------------------------
    # Loss function.  Either an SDK built-in (see azure-ai-finetuningsessions
    # docs) or the sentinel "custom" indicating a Python-side loss closure is
    # in use (see ``forward_backward_custom_async``).  When set to "custom",
    # ``custom_loss_name`` must be the registry key naming the closure, and
    # the training client must intercept ``forward_backward_async`` to route
    # to the surrogate-gradient path -- the "custom" sentinel never reaches
    # the SDK wire.
    loss_fn: LossFnType | Literal["custom"] = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    # Required iff ``loss_fn == "custom"``: the user-defined name of the
    # Python-side loss closure (e.g. "dapo").  Surfaced in run_meta and on
    # the dashboard so the effective objective is identifiable.
    custom_loss_name: str | None = None

    @chz.validate
    def _check_custom_loss_consistency(self) -> None:
        is_custom = self.loss_fn == "custom"
        has_name = self.custom_loss_name is not None
        if is_custom and not has_name:
            raise ValueError(
                'Config.loss_fn == "custom" requires custom_loss_name to be set.'
            )
        if has_name and not is_custom:
            raise ValueError(
                'Config.custom_loss_name is set but loss_fn != "custom" '
                f"(got {self.loss_fn!r}); these must be in sync."
            )

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1
    # LoRA rank for the training adapter.
    lora_rank: int = 16
    # LoRA scaling factor (effective scale = alpha / rank).
    lora_alpha: int = 32

    # -------------------------------------------------------------------------
    # Sampling and diagnostics (advanced)
    # -------------------------------------------------------------------------
    # Changing sampling temperature is not generally recommended; T=1 is near-optimal
    # for most post-trained models, and non-1 temperatures currently do not play
    # well with KL penalty.
    temperature: float = 1.0
    # Nucleus sampling threshold (1.0 = disabled, i.e. consider all tokens).
    top_p: float = 1.0
    # Top-k filtering (-1 = disabled, i.e. no filtering).
    top_k: int = -1
    # Optional OpenAI-compatible response format for constrained generation.
    response_format: dict[str, Any] | None = None
    # RNG seed for reproducible sampling. When set, each /sample call receives
    # a deterministic seed derived from this value, making rollout trajectories
    # reproducible across runs with the same configuration. None = server
    # picks a random seed per call (non-reproducible).
    sampling_seed: int | None = None
    # Compute extra post-update KL metrics (adds overhead).
    compute_post_kl: bool = False
    # Remove groups where all trajectories have identical reward.
    remove_constant_reward_groups: bool = False
    # Opt in only for typed, exhausted incomplete grading; validation is unchanged.
    allow_incomplete_grading_group_drop: bool = False
    # Application-owned diagnostics/policy. Callback exceptions abort training.
    on_group_drop: GroupDropCallback | None = None
    # (experimental) Pluggable dynamic-batching strategy (two-axis API in
    # interactive_training.dynamic_batching). None = today's behavior exactly
    # (compute_advantages + optional constant-reward dropping). When set,
    # the strategy's `select` chooses which rolled samples train and assigns
    # advantages (B-axis, applied in prepare_minibatch), and its `allocate`
    # may DROP a rolled group or ask for MORE rolls (A-axis, applied at
    # rollout time). FixedStrategy reproduces the vanilla path; DapoStrategy
    # reproduces constant-reward dropping.
    strategy: DynamicBatchStrategy | None = None
    # (experimental) Cap on the total number of rolled trajectories per group
    # when a strategy requests re-rolls (Allocation.MORE, the A-axis "grow this
    # prompt's pool" signal). None disables the re-roll seam entirely -- MORE is
    # then treated as admit and every prompt is rolled exactly once. Only
    # meaningful together with `strategy`; set it at or above the strategy's
    # target pool size (e.g. Pilot-Commit's pilot + commit rollouts). Each
    # re-roll adds one sampling-group-sized wave.
    max_rolls_per_group: int | None = None

    # --- Rollout resilience (advanced; opt-in) ---
    # Per-sample-call wall-clock timeout. When set, a sample that exceeds this
    # many seconds is treated as a (retryable) rollout failure.
    sample_timeout_sec: float | None = None
    # Per-trajectory retry budget (0 disables retries).
    max_retries_per_trajectory: int = 0
    # Extra whole-group attempts allowed to backfill failed trajectory slots.
    max_extra_trajectory_attempts_per_group: int = 0
    # Rolling window (in trajectory attempts) for the failure-rate circuit breaker.
    rollout_failure_window_size: int = 100
    # Warn when the windowed failure rate reaches this fraction.
    rollout_failure_rate_warn: float = 0.10
    # Abort the run when the windowed failure rate reaches this fraction.
    rollout_failure_rate_abort: float = 0.20
    # Minimum attempts observed before the circuit breaker can fire.
    rollout_failure_min_attempts: int = 25
    # When the breaker trips, recover this many times (reset the window and back
    # off) before aborting the run.
    rollout_failure_max_resets: int = 0
    # Seconds to pause after a breaker trip before retrying, giving the sampling
    # backend time to recover.
    rollout_failure_reset_backoff_sec: float = 60.0

    # (experimental) Dynamic sampling: keep sampling new prompt groups until
    # groups_per_batch post-filter groups are collected (DAPO-style). When
    # enabled, constant-reward filtering is always applied (the value of
    # remove_constant_reward_groups is ignored in this path).
    dynamic_sampling: bool = False
    # (experimental) Maximum number of sampling rounds (including the initial
    # round) before giving up when dynamic_sampling is enabled. With a ~50%
    # filter rate the first round typically yields half the target, so 3
    # rounds is usually sufficient. Prevents infinite loops on datasets that
    # are too easy or too hard.
    max_oversample_rounds: int = 3
    # (experimental) Safety cushion applied when right-sizing the round-2+
    # top-up sample using the observed filter rate. Higher values (e.g. 1.5)
    # reduce the chance of needing a third round at the cost of slightly more
    # rollout compute. 1.0 = no cushion.
    oversample_cushion: float = 1.2
    # (experimental) Refill-on-drop: when a `strategy`'s A-axis `allocate`
    # DROPs a group, keep sampling fresh prompts until `groups_per_batch`
    # groups survive (instead of letting the batch shrink for that step).
    # This makes drop-based strategies (DAPO, Pilot-Commit, ...) faithful to
    # their papers, which hold the trained batch size constant. Reuses the
    # dynamic_sampling refill machinery (max_oversample_rounds /
    # oversample_cushion / prompt carryover), but the *survivor* criterion is
    # the strategy's drop decision, not native constant-reward filtering. No
    # effect unless `strategy` is set; default False keeps the batch-shrink
    # behavior (and the default path byte-identical).
    refill_on_drop: bool = False
    # Emit async trace events for debugging/profiling.
    enable_trace: bool = False

    # -------------------------------------------------------------------------
    # Execution mode knobs (advanced)
    # -------------------------------------------------------------------------
    # Enable async/off-policy training mode when set.
    async_config: AsyncConfig | None = None
    # Enable sync training with streaming minibatches when set.
    stream_minibatch_config: StreamMinibatchConfig | None = None
    # Optional service base URL override (primarily internal/dev use).
    base_url: str | None = None

    # -------------------------------------------------------------------------
    # Training budget (advanced)
    # -------------------------------------------------------------------------
    # Cap on total number of training iterations (None = train through the
    # whole dataset). Without this, large RL datasets run until exhausted
    # or the run crashes -- there is no graceful early stop.
    max_steps: int | None = None
    # Wall-clock budget in seconds (None = no limit). When set, the training
    # loop breaks at the next iteration boundary once elapsed time since
    # main() started exceeds this value. Useful for time-boxed experiments,
    # smoke tests, and CI runs.
    max_wall_clock_seconds: float | None = None

    # -------------------------------------------------------------------------
    # Checkpoint retention and logging detail (advanced)
    # -------------------------------------------------------------------------
    # TTL for checkpoints in seconds, i.e. how long checkpoints should live
    # before expiry (None = no expiry).
    ttl_seconds: int | None = 604800  # 7 days
    num_groups_to_log: int = 4  # Number of groups to log per iteration (0 = disable logging)
    # Suppress trajectory content in console and logtree output while preserving
    # scalar metrics and the trajectories used for training and evaluation.
    sanitize_logs: bool = False
    # Max lines of colorized per-trajectory output written to the console for each
    # logged group. Long agentic/tool-calling rollouts can decode into hundreds of
    # thousands of lines; this caps the console echo (head + tail with an elision
    # marker). Set to 0 for unlimited / full transcripts. Note: this only bounds the
    # console log, not training data or logtree HTML logging.
    max_console_lines_per_datum: int = 200

    @chz.validate
    def _validate_dynamic_sampling(self):
        if self.dynamic_sampling and self.async_config is not None:
            raise ValueError(
                "dynamic_sampling is only supported with sync training. "
                "It cannot be combined with async_config."
            )
        if self.dynamic_sampling and self.stream_minibatch_config is not None:
            raise ValueError(
                "dynamic_sampling is only supported with sync training. "
                "It cannot be combined with stream_minibatch_config."
            )

    @chz.validate
    def _validate_refill_on_drop(self):
        # refill_on_drop reuses the sync-only refill loop and refills based on
        # the strategy's drop decision, so it requires a strategy and the
        # standard synchronous path.
        if not self.refill_on_drop:
            return
        if self.strategy is None:
            raise ValueError(
                "refill_on_drop requires a `strategy`; it refills the batch "
                "when the strategy's allocate() DROPs a group. Set `strategy` "
                "or leave refill_on_drop=False."
            )
        if self.async_config is not None or self.stream_minibatch_config is not None:
            raise ValueError(
                "refill_on_drop is only supported with standard synchronous "
                "training (not async_config or stream_minibatch_config)."
            )

    @chz.validate
    def _validate_console_logging(self):
        if self.num_groups_to_log < 0:
            raise ValueError(
                f"num_groups_to_log must be >= 0 (0 disables logging), got {self.num_groups_to_log}"
            )
        if self.max_console_lines_per_datum < 0:
            raise ValueError(
                "max_console_lines_per_datum must be >= 0 (0 = unlimited), "
                f"got {self.max_console_lines_per_datum}"
            )

    @chz.validate
    def _validate_eval_cadence(self):
        # Under the epoch strategy, eval_every is expressed in epochs rather
        # than training iterations. Only 1 is currently supported: evaluate
        # the initial model and after the single dataset traversal.
        if self.eval_strategy == "epoch" and self.eval_every != 1:
            raise ValueError(
                "eval_strategy='epoch' requires eval_every=1; other epoch "
                "cadences are not supported. This evaluates the initial model "
                "and the model after the dataset traversal. To disable "
                "evaluation entirely, use eval_strategy='steps' with "
                "eval_every=0."
            )

    # Maximum concurrent group rollouts per training step.
    # Each group spawns group_size HTTP requests; unbounded concurrency
    # (groups_per_batch × group_size) can overwhelm the request store.
    # None = unlimited (all groups in parallel, correct for production).
    # For local dev with the Cosmos emulator, set to ~32.
    max_concurrent_groups: int | None = None


def _effective_num_groups_to_log(cfg: Config) -> int:
    """Return the content-log budget after applying privacy sanitization."""
    return 0 if getattr(cfg, "sanitize_logs", False) is True else cfg.num_groups_to_log


def _config_for_logging(cfg: Config) -> Config | dict[str, Any]:
    """Redact the dataset builder because prepared builders can contain raw rows."""
    if getattr(cfg, "sanitize_logs", False) is not True:
        return cfg
    logged_config = chz.asdict(cfg, shallow=True, exclude={"dataset_builder"})
    logged_config["dataset_builder"] = "<redacted>"
    return logged_config


def _sampling_evaluator_accepts_step(evaluator) -> bool:
    try:
        parameters = inspect.signature(evaluator).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "step"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
        for parameter in parameters
    )


async def _call_sampling_evaluator(evaluator, sampling_client, step):
    if _sampling_evaluator_accepts_step(evaluator):
        return await evaluator(sampling_client, step=step)
    return await evaluator(sampling_client)


@scope
async def run_single_evaluation(evaluator, cfg, i_batch, sampling_client):
    ev_name = _get_evaluator_name(evaluator)
    with _get_logtree_scope(
        log_path=cfg.log_path,
        num_groups_to_log=_effective_num_groups_to_log(cfg),
        f_name=f"eval_{ev_name}_iteration_{i_batch:06d}",
        scope_name=f"Running evaluation {ev_name} {i_batch}",
    ):
        eval_metrics = await _call_sampling_evaluator(
            evaluator, sampling_client, i_batch
        )
        return eval_metrics


@scope
async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: FineTuningSession,
    cfg: Config,
    i_batch: int,
) -> dict[str, Any]:
    """Run all evaluators in parallel and return aggregated metrics."""

    # Create tasks for all evaluators with names for better traceability
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        task = asyncio.create_task(
            run_single_evaluation(evaluator, cfg, i_batch, sampling_client),
            name=f"eval_{ev_name or i}_iteration_{i_batch:06d}",
        )
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Merge all metrics
    metrics = {}
    for result in results:
        metrics.update(result)

    return metrics


def should_evaluate_rl_iteration(
    *,
    eval_strategy: EvaluationStrategy,
    eval_every: int,
    step: int,
    is_fresh_run: bool,
) -> bool:
    """Return whether RL should evaluate before this training iteration."""
    if eval_strategy == "steps":
        return eval_every > 0 and step % eval_every == 0
    if eval_strategy == "epoch":
        return is_fresh_run and step == 0
    raise ValueError(f"Unsupported eval_strategy: {eval_strategy!r}")


def should_evaluate_rl_final(
    *,
    eval_strategy: EvaluationStrategy,
    eval_every: int,
    actual_next_batch: int,
    num_batches: int,
) -> bool:
    """Return whether RL should evaluate the final saved model."""
    if eval_strategy == "steps":
        return eval_every > 0
    if eval_strategy == "epoch":
        return actual_next_batch > 0
    raise ValueError(f"Unsupported eval_strategy: {eval_strategy!r}")


def checkpoint_eval_every(cfg: Config) -> int:
    """Return the step cadence that should force periodic checkpoints."""
    return cfg.eval_every if cfg.eval_strategy == "steps" else 0


@scope
async def do_sync_training_with_stream_minibatch(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: FineTuningSession,
    kl_reference_client: FineTuningSession | None,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    deadline: float | None = None,
    is_fresh_run: bool | None = None,
):
    """
    Implements fully synchronous on-policy training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready. This allows us to overlap sampling and training.
    """
    fresh_run = start_batch == 0 if is_fresh_run is None else is_fresh_run

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch,
        cfg.ttl_seconds,
        eval_every=checkpoint_eval_every(cfg),
    )
    retry_policy, failure_tracker = _make_rollout_failure_policy(cfg)

    actual_next_batch = start_batch
    last_completed_step: int | None = None
    for i_batch in range(start_batch, end_batch):
        if deadline is not None and time.time() >= deadline:
            logger.info(
                "[stream_minibatch] wall-clock budget reached at step %d/%d; stopping",
                i_batch, end_batch,
            )
            break
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if should_evaluate_rl_iteration(
            eval_strategy=cfg.eval_strategy,
            eval_every=cfg.eval_every,
            step=i_batch,
            is_fresh_run=fresh_run,
        ):
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        with _get_logtree_scope(
            cfg.log_path,
            _effective_num_groups_to_log(cfg),
            f"train_iteration_{i_batch:06d}",
            f"RL Iteration {i_batch}",
        ):
            # Samplers will produce trajectory groups asynchronously,
            # and the trainer will consume them as soon as they are ready
            trajectory_groups_queue = asyncio.Queue[_TrajectoryGroupQueueItem]()
            env_group_builders_P = dataset.get_batch(i_batch)

            @scope
            async def trajectory_group_worker_task(
                builder: EnvGroupBuilder, group_index: int,
                enable_logging: bool, group_seed: int | None = None,
            ) -> None:
                try:
                    metrics = {}
                    t_start = time.time()
                    trajectory_group = await do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                        enable_logging=enable_logging,
                        top_p=cfg.top_p,
                        top_k=cfg.top_k,
                        response_format=cfg.response_format,
                        seed=group_seed,
                        strategy=cfg.strategy,
                        max_rolls_per_group=cfg.max_rolls_per_group,
                        sample_timeout_sec=cfg.sample_timeout_sec,
                        retry_policy=retry_policy,
                        failure_tracker=failure_tracker,
                        allow_incomplete_grading_group_drop=cfg.allow_incomplete_grading_group_drop,
                        on_group_drop=cfg.on_group_drop,
                        training_step=i_batch,
                        training_group_index=group_index,
                        replenish_exhausted_group=not isinstance(
                            builder, AutonomousRolloutGroupBuilder
                        ),
                    )
                    metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                    if trajectory_group is not None:
                        trajectory_groups_queue.put_nowait(
                            WrappedTrajectoryGroup(
                                trajectory_group=trajectory_group,
                                env_group_builder=builder,
                                sampling_client_step=i_batch,
                                metrics=metrics,
                            )
                        )
                    else:
                        trajectory_groups_queue.put_nowait(None)
                except Exception as error:
                    trajectory_groups_queue.put_nowait(error)

            # Sample all trajectories asynchronously. If we have multiple minibatches,
            # then sampling can overlap with training.
            worker_tasks = []
            for i, builder in enumerate(env_group_builders_P):
                group_seed = _derive_group_seed(cfg.sampling_seed, i_batch, i)
                worker_tasks.append(
                    asyncio.create_task(
                        trajectory_group_worker_task(
                            builder,
                            group_index=i,
                            enable_logging=i < _effective_num_groups_to_log(cfg),
                            group_seed=group_seed,
                        ),
                        name=f"trajectory_group_worker_task_{i}",
                    )
                )

            # Run multiple optimizer substeps per training iteration
            try:
                (
                    next_sampling_client,
                    full_batch_metrics,
                ) = await do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    training_client,
                    kl_reference_client,
                    tokenizer,
                )
            finally:
                for worker_task in worker_tasks:
                    if not worker_task.done():
                        worker_task.cancel()
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            if next_sampling_client is not None:
                sampling_client = next_sampling_client

        # Log metrics
        metrics.update(full_batch_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)
        actual_next_batch = i_batch + 1
        if full_batch_metrics.get("train/skipped_no_valid_groups") != 1:
            last_completed_step = i_batch

    return actual_next_batch, last_completed_step


@dataclass
class PrefilterRewardAccumulator:
    total_groups: int = 0
    total_episodes: int = 0
    reward_sum: float = 0.0
    group_reward_mean_sum: float = 0.0
    mixed_groups: int = 0
    all_good_groups: int = 0
    all_bad_groups: int = 0

    def add(self, group: TrajectoryGroup) -> None:
        rewards = group.get_total_rewards()
        if not rewards:
            return
        self.total_groups += 1
        self.total_episodes += len(rewards)
        self.reward_sum += sum(rewards)
        self.group_reward_mean_sum += sum(rewards) / len(rewards)
        if all_same(rewards):
            if rewards[0] >= 0.5:
                self.all_good_groups += 1
            else:
                self.all_bad_groups += 1
        else:
            self.mixed_groups += 1

    def merge(self, other: "PrefilterRewardAccumulator") -> None:
        self.total_groups += other.total_groups
        self.total_episodes += other.total_episodes
        self.reward_sum += other.reward_sum
        self.group_reward_mean_sum += other.group_reward_mean_sum
        self.mixed_groups += other.mixed_groups
        self.all_good_groups += other.all_good_groups
        self.all_bad_groups += other.all_bad_groups

    def metrics(self) -> dict[str, float | int]:
        if self.total_groups == 0 or self.total_episodes == 0:
            return {}
        return {
            "sampling/pre_filter/reward/total": self.reward_sum / self.total_episodes,
            "sampling/pre_filter/reward/group_mean": (
                self.group_reward_mean_sum / self.total_groups
            ),
            "sampling/pre_filter/total_groups": self.total_groups,
            "sampling/pre_filter/total_episodes": self.total_episodes,
            "sampling/pre_filter/by_group/frac_mixed": (
                self.mixed_groups / self.total_groups
            ),
            "sampling/pre_filter/by_group/frac_all_good": (
                self.all_good_groups / self.total_groups
            ),
            "sampling/pre_filter/by_group/frac_all_bad": (
                self.all_bad_groups / self.total_groups
            ),
        }


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup | None
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)
    prefilter_rewards: PrefilterRewardAccumulator = chz.field(
        default_factory=PrefilterRewardAccumulator
    )
    group_ordinal: int | None = None


@dataclass(frozen=True)
class _QueuedEnvGroupBuilder:
    builder: EnvGroupBuilder
    ordinal: int


def _trainable_wrapped_groups(
    groups: Sequence[WrappedTrajectoryGroup | None],
) -> list[WrappedTrajectoryGroup]:
    return [
        group
        for group in groups
        if group is not None and group.trajectory_group is not None
    ]


def _is_stale_trajectory_group(
    *,
    training_step: int,
    sampling_client_step: int,
    max_steps_off_policy: int,
) -> bool:
    return training_step - sampling_client_step > max_steps_off_policy


def _async_rollout_worker_count(
    groups_per_batch: int, max_concurrent_groups: int | None
) -> int:
    if max_concurrent_groups is None:
        return groups_per_batch
    if max_concurrent_groups < 1:
        raise ValueError("max_concurrent_groups must be at least 1")
    return min(groups_per_batch, max_concurrent_groups)


_TrajectoryGroupQueueItem = WrappedTrajectoryGroup | GroupDropEvent | Exception | None


@scope
async def do_async_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: FineTuningSession,
    kl_reference_client: FineTuningSession | None,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    deadline: float | None = None,
    is_fresh_run: bool | None = None,
    resume_group_ordinal: int | None = None,
    final_loop_state: dict[str, Any] | None = None,
):
    """Implements async off-policy training, capped at K steps off policy."""
    assert cfg.async_config is not None
    fresh_run = start_batch == 0 if is_fresh_run is None else is_fresh_run
    seed_state = final_loop_state if final_loop_state is not None else {}
    next_group_ordinal = (
        start_batch * cfg.async_config.groups_per_batch
        if resume_group_ordinal is None else resume_group_ordinal
    )
    if not isinstance(next_group_ordinal, int) or next_group_ordinal < 0:
        raise ValueError("resume_group_ordinal must be a non-negative integer")
    seed_state["async_group_ordinal"] = next_group_ordinal

    shutdown_event = asyncio.Event()
    # Tracks the last time training made forward progress (a step completed, or
    # the run just started). Read by the heartbeat watchdog to detect stalls.
    last_progress_at = time.time()
    last_progress_step = start_batch
    # We will have groups_per_batch worker generating rollouts, so cap the
    # queue size to be groups_per_batch.
    env_group_builders_queue = asyncio.Queue[_QueuedEnvGroupBuilder | None](
        maxsize=cfg.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[_TrajectoryGroupQueueItem]()
    retry_policy, failure_tracker = _make_rollout_failure_policy(cfg)

    # In epoch mode, evaluate before writing the resumable batch-0 checkpoint.
    # An interruption during evaluation will therefore retry the baseline
    # instead of resuming past an evaluation that never completed.
    if cfg.eval_strategy == "epoch" and fresh_run and evaluators:
        initial_sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )
        metrics = await run_evaluations_parallel(
            evaluators, initial_sampling_client, cfg, start_batch
        )
        ml_logger.log_metrics(metrics, step=start_batch)

    # RL needs both the trainer state (for resume) and a sampler snapshot (to
    # start generating rollouts), so save both before launching worker loops.
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=str(start_batch),
        log_path=cfg.log_path,
        loop_state={"batch": start_batch, **seed_state},
        kind="both",
        ttl_seconds=cfg.ttl_seconds,
    )

    # This will be updated by the training loop
    sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
    sampling_client_step = start_batch
    sampling_group_step = start_batch
    next_sampling_group_index = 0
    actual_next_batch = start_batch
    last_completed_step: int | None = None
    sampling_client_updated_event = asyncio.Event()
    if (
        cfg.eval_strategy == "steps"
        and cfg.eval_every > 0
        and not fresh_run
        and evaluators
    ):
        metrics = {}
        with timed("run_evals", metrics):
            eval_metrics = await run_evaluations_parallel(
                evaluators, sampling_client, cfg, start_batch
            )
            metrics.update(eval_metrics)
        ml_logger.log_metrics(metrics, step=start_batch)
    if cfg.eval_strategy == "steps" and fresh_run:
        sampling_client_updated_event.set()

    @scope
    def shutdown_loops():
        """Trigger all loops to shutdown"""
        shutdown_event.set()
        assert cfg.async_config is not None
        for _ in range(cfg.async_config.groups_per_batch):
            try:
                env_group_builders_queue.put_nowait(None)
            except asyncio.QueueFull:
                break
        sampling_client_updated_event.set()

    async def _put_builder_or_shutdown(builder: _QueuedEnvGroupBuilder) -> bool:
        """Put a builder on the queue; return False if shutdown wins first.

        Lets the (bounded, back-pressured) dataloader exit promptly on shutdown
        instead of blocking forever on a full queue once the consumers stop.
        """
        put_task = asyncio.ensure_future(env_group_builders_queue.put(builder))
        shutdown_task = asyncio.ensure_future(shutdown_event.wait())
        try:
            done, _ = await asyncio.wait(
                {put_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
            )
            return put_task in done
        finally:
            shutdown_task.cancel()
            if not put_task.done():
                put_task.cancel()

    async def _get_builder_or_shutdown() -> _QueuedEnvGroupBuilder | None:
        """Get the next builder, or None if shutdown is signalled first.

        Treated identically to the None sentinel by the worker, so workers exit
        on shutdown even when no sentinel reaches them (e.g. the queue is full
        of builders the cyclic dataloader produced when training finished).
        """
        get_task = asyncio.ensure_future(env_group_builders_queue.get())
        shutdown_task = asyncio.ensure_future(shutdown_event.wait())
        try:
            done, _ = await asyncio.wait(
                {get_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                return get_task.result()
            return None
        finally:
            shutdown_task.cancel()
            if not get_task.done():
                get_task.cancel()

    @scope
    async def dataloader_loop():
        """Continuously supplies env builders, cycling through the dataset.

        A single pass emits exactly num_batches * batch_size builders. When the
        constant-reward filter drops a group (see trajectory_group_worker_loop)
        it consumes a builder without producing a trainable group, so a
        single-pass loader can leave the training loop permanently short of the
        groups_per_batch valid groups it needs to assemble a step -- a tail-batch
        starvation deadlock. Instead, keep supplying builders (wrapping around
        the dataset, like sync dynamic-sampling's `% len(dataset)`) until the
        training loop signals completion. The bounded env_group_builders_queue
        provides backpressure, so we only ever read a few prompts ahead of
        demand.
        """
        i_batch = start_batch
        group_ordinal = next_group_ordinal
        while not shutdown_event.is_set():
            env_group_builders_P = dataset.get_batch(i_batch % num_batches)
            for env_group_builder in env_group_builders_P:
                queued_builder = _QueuedEnvGroupBuilder(
                    builder=env_group_builder,
                    ordinal=group_ordinal,
                )
                group_ordinal += 1
                seed_state["async_group_ordinal"] = group_ordinal
                if not await _put_builder_or_shutdown(queued_builder):
                    return
            i_batch += 1
            # Wrapping the dataset means drops have forced us to reuse prompts to
            # keep the training loop fed. Log once per full pass (cheap, naturally
            # throttled) so high drop pressure is visible without per-rollout spam.
            if (i_batch - start_batch) % num_batches == 0:
                logger.info(
                    "[dataloader_loop] cycled through dataset (%d batches supplied, "
                    "%d full passes); replenishing dropped/stale groups",
                    i_batch - start_batch,
                    (i_batch - start_batch) // num_batches,
                )

    @scope
    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        nonlocal sampling_group_step, next_sampling_group_index
        grading_drop: GroupDropEvent | None = None

        async def record_grading_drop(event: GroupDropEvent) -> None:
            nonlocal grading_drop
            await _handle_incomplete_group_drop(
                event, allowed=True, callback=cfg.on_group_drop,
            )
            grading_drop = event

        while not shutdown_event.is_set():
            queued_builder = await _get_builder_or_shutdown()
            if queued_builder is None:
                break
            env_group_builder = queued_builder.builder

            metrics = {}
            prefilter_rewards = PrefilterRewardAccumulator()
            t_start = time.time()
            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step
            # Allocate across workers before awaiting; completion order is irrelevant.
            if sampling_group_step != sampling_client_step_copy:
                sampling_group_step = sampling_client_step_copy
                next_sampling_group_index = 0
            training_group_index = next_sampling_group_index
            next_sampling_group_index += 1
            group_seed = _derive_async_group_seed(
                cfg.sampling_seed,
                queued_builder.ordinal,
                cfg.async_config.groups_per_batch,
            )
            grading_drop = None
            trajectory_group = await do_group_rollout_and_filter_constant_reward(
                sampling_client,
                env_group_builder,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                response_format=cfg.response_format,
                seed=group_seed,
                strategy=cfg.strategy,
                max_rolls_per_group=cfg.max_rolls_per_group,
                sample_timeout_sec=cfg.sample_timeout_sec,
                retry_policy=retry_policy,
                failure_tracker=failure_tracker,
                allow_incomplete_grading_group_drop=cfg.allow_incomplete_grading_group_drop,
                on_group_drop=record_grading_drop,
                training_step=sampling_client_step_copy,
                training_group_index=training_group_index,
                replenish_exhausted_group=not isinstance(
                    env_group_builder, AutonomousRolloutGroupBuilder
                ),
                on_prefilter_group=prefilter_rewards.add,
            )
            if trajectory_group is None and grading_drop is not None:
                trajectory_groups_queue.put_nowait(grading_drop)
                continue
            if trajectory_group is not None:
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
            trajectory_groups_queue.put_nowait(
                WrappedTrajectoryGroup(
                    trajectory_group=trajectory_group,
                    env_group_builder=env_group_builder,
                    sampling_client_step=sampling_client_step_copy,
                    metrics=metrics,
                    prefilter_rewards=prefilter_rewards,
                    group_ordinal=queued_builder.ordinal,
                )
            )

    @scope
    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert cfg.async_config is not None

        i_batch = start_batch
        wrapped_trajectory_groups = []
        # Guard against a degenerate batch where every prompt keeps yielding a
        # constant-reward (zero-advantage) group: the cyclic dataloader would
        # otherwise replenish supply forever without ever assembling a step.
        # If we drop this many groups in a row with no trainable group in
        # between, there is no learnable signal -- stop cleanly.
        consecutive_dropped_groups = 0
        # The drop counter increments per *prompt group*, not per dataset batch.
        # Base the threshold on the number of valid groups we need per training
        # step, times the allowed number of "oversample rounds" worth of retries.
        max_consecutive_dropped_groups = max(cfg.async_config.groups_per_batch, 1) * max(
            cfg.max_oversample_rounds, 1
        )
        # Total groups dropped since the last successful train
        # step, surfaced as a per-step metric (drops are otherwise invisible).
        dropped_groups_since_train = 0
        prefilter_rewards_since_train = PrefilterRewardAccumulator()
        while i_batch < end_batch:
            if deadline is not None and time.time() >= deadline:
                logger.info(
                    "[async] wall-clock budget reached at step %d/%d; stopping",
                    i_batch, end_batch,
                )
                shutdown_event.set()
                break
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if isinstance(wrapped_trajectory_group, Exception):
                raise wrapped_trajectory_group
            if isinstance(wrapped_trajectory_group, GroupDropEvent):
                # Unavailable rewards are not evidence of no learnable signal.
                # The application's callback owns the grading-failure budget.
                consecutive_dropped_groups = 0
                dropped_groups_since_train += 1
                continue

            @scope
            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Requeue on a separate coroutine to avoid blocking the training loop
                assert cfg.async_config is not None
                if _is_stale_trajectory_group(
                    training_step=i_batch,
                    sampling_client_step=wrapped_trajectory_group.sampling_client_step,
                    max_steps_off_policy=cfg.async_config.max_steps_off_policy,
                ):
                    logger.info(f"[training_loop] Step {i_batch}: Samples are too stale, skipping")
                    asyncio.create_task(
                        env_group_builders_queue.put(
                            _QueuedEnvGroupBuilder(
                                builder=wrapped_trajectory_group.env_group_builder,
                                ordinal=wrapped_trajectory_group.group_ordinal,
                            )
                        ),
                        name="requeue_stale_sample_task",
                    )
                    return False
                return True

            if wrapped_trajectory_group is not None:
                if not filter_stale_trajectory_group(wrapped_trajectory_group):
                    continue
                if cfg.stream_minibatch_config is None:
                    prefilter_rewards_since_train.merge(
                        wrapped_trajectory_group.prefilter_rewards
                    )
            if wrapped_trajectory_group is None or wrapped_trajectory_group.trajectory_group is None:
                # A worker dropped a group; the cyclic dataloader will replenish
                # supply. Only bail if we never recover.
                consecutive_dropped_groups += 1
                dropped_groups_since_train += 1
                if consecutive_dropped_groups >= max_consecutive_dropped_groups:
                    logger.warning(
                        "[training_loop] Step %d: %d consecutive groups dropped "
                        "(threshold %d); no trainable signal -- stopping.",
                        i_batch,
                        consecutive_dropped_groups,
                        max_consecutive_dropped_groups,
                    )
                    shutdown_loops()
                    break
                continue
            consecutive_dropped_groups = 0

            metrics = {
                "training_client/step": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
                "async/dropped_groups": dropped_groups_since_train,
            }
            metrics.update(prefilter_rewards_since_train.metrics())
            if not cfg.allow_incomplete_grading_group_drop:
                metrics["async/dropped_constant_reward_groups"] = dropped_groups_since_train
            t_start = time.time()

            nonlocal sampling_client
            nonlocal sampling_client_step
            nonlocal last_progress_at
            nonlocal last_progress_step
            nonlocal actual_next_batch
            nonlocal last_completed_step
            if cfg.stream_minibatch_config is not None:
                await trajectory_groups_queue.put(wrapped_trajectory_group)
                (
                    next_sampling_client,
                    train_step_metrics,
                ) = await do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    training_client,
                    kl_reference_client,
                    tokenizer,
                    filter_stale_trajectory_group,
                    prefilter_rewards_since_train,
                    loop_state_extra=seed_state,
                )
                if next_sampling_client is not None:
                    sampling_client = next_sampling_client
                train_step_metrics["async/dropped_groups"] = (
                    dropped_groups_since_train
                    + train_step_metrics.get("async/dropped_groups", 0)
                )
            else:
                # Dynamic sampling: Wait for enough trajectories to accumulate to
                # ensure all batch sizes are the same size. This avoids needing to adjust
                # the learning rate for different batch sizes.
                wrapped_trajectory_groups.append(wrapped_trajectory_group)
                if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                    continue
                logger.info(
                    f"[training_loop] Step {i_batch}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
                )

                # Compute sampling client metrics, as samples may have been generated with
                # different sampler versions
                metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

                # TODO: For proper checkpointing, we also need to save dataloader state and
                # all queued trajectory groups that haven't been trained on yet
                sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                    cfg,
                    i_batch,
                    training_client,
                    kl_reference_client,
                    tokenizer,
                    [g.env_group_builder for g in wrapped_trajectory_groups],
                    [
                        g.trajectory_group
                        for g in wrapped_trajectory_groups
                        if g.trajectory_group is not None
                    ],
                    loop_state_extra=seed_state,
                )
            if train_step_metrics.get("train/skipped_no_valid_groups") != 1:
                sampling_client_step = i_batch + 1
                sampling_client_updated_event.set()

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/training_loop/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)
            i_batch += 1
            wrapped_trajectory_groups = []
            dropped_groups_since_train = 0
            prefilter_rewards_since_train = PrefilterRewardAccumulator()
            last_progress_at = time.time()
            last_progress_step = i_batch
            actual_next_batch = i_batch
            if train_step_metrics.get("train/skipped_no_valid_groups") != 1:
                last_completed_step = i_batch - 1

        shutdown_loops()

    @scope
    async def evaluation_loop():
        """Runs evals periodically"""
        if len(evaluators) == 0 or (
            cfg.eval_strategy == "steps" and cfg.eval_every == 0
        ):
            return

        while not shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()
            # shutdown_loops() sets this event to release us; re-check so we
            # don't run a spurious final evaluation (in epoch mode this would
            # duplicate the step-0 baseline when the run dies before batch 0
            # completes).
            if shutdown_event.is_set():
                break

            metrics = {}
            t_start = time.time()
            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if should_evaluate_rl_iteration(
                eval_strategy=cfg.eval_strategy,
                eval_every=cfg.eval_every,
                step=sampling_client_eval_step,
                is_fresh_run=fresh_run,
            ):
                with timed("run_evals", metrics):
                    for evaluator in evaluators:
                        eval_metrics = await _call_sampling_evaluator(
                            evaluator,
                            sampling_client_eval,
                            sampling_client_eval_step,
                        )
                        # Evaluator already returns keys prefixed with "test/" —
                        # don't double-prefix them
                        metrics.update(eval_metrics)
                metrics["time/evaluation_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)

    @scope
    async def heartbeat_loop():
        """Advisory watchdog: warn (don't kill) when training stops progressing.

        Defense-in-depth breadcrumb. The cyclic dataloader removes the known
        constant-reward starvation deadlock, but other stalls are still possible
        anywhere the training loop awaits between steps -- rollouts not arriving
        (slow/cold inference fleet), the train backend wedged in
        forward_backward/optim_step, or a checkpoint upload hanging. Rather than
        hang silently for hours, periodically emit a log line so an operator can
        tell a healthy-but-slow run from a wedged one and knows to go look at the
        per-stage logs to localize the stall.
        """
        assert cfg.async_config is not None
        warn_sec = cfg.async_config.no_progress_warn_sec
        if warn_sec <= 0:
            return
        poll_sec = max(min(warn_sec / 4.0, 60.0), 5.0)
        last_warn_at = 0.0
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=poll_sec)
                return  # shutdown signalled while waiting
            except asyncio.TimeoutError:
                pass
            now = time.time()
            idle = now - last_progress_at
            # Re-warn at most once per warn_sec window so logs stay readable.
            if idle >= warn_sec and (now - last_warn_at) >= warn_sec:
                last_warn_at = now
                logger.warning(
                    "[heartbeat] Advisory only (not an error): no training step "
                    "has completed for %.1f min (last completed step %d/%d). The "
                    "run is still alive -- heavy jobs can legitimately take this "
                    "long per step, so this may be normal. Only investigate if "
                    "this far exceeds your expected step time; if so, check the "
                    "run logs to see which stage is slow -- sampling, the train "
                    "backend, or checkpoint upload. To silence expected long "
                    "steps, raise async_config.no_progress_warn_sec.",
                    idle / 60.0,
                    last_progress_step,
                    end_batch,
                )

    num_rollout_workers = cfg.async_config.groups_per_batch
    if cfg.max_concurrent_groups is not None:
        if cfg.max_concurrent_groups <= 0:
            raise ValueError("max_concurrent_groups must be positive when set")
        num_rollout_workers = min(num_rollout_workers, cfg.max_concurrent_groups)
    logger.info(
        "[async_training] Starting %d rollout workers "
        "(groups_per_batch=%d, max_concurrent_groups=%s)",
        num_rollout_workers,
        cfg.async_config.groups_per_batch,
        cfg.max_concurrent_groups,
    )

    await asyncio.gather(
        asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
        *[
            asyncio.create_task(
                trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
            )
            for i in range(num_rollout_workers)
        ],
        asyncio.create_task(training_loop(), name="training_loop"),
        asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
        asyncio.create_task(heartbeat_loop(), name="heartbeat_loop"),
    )

    return actual_next_batch, last_completed_step


def _merge_trajectory_groups(
    a: TrajectoryGroup, b: TrajectoryGroup
) -> TrajectoryGroup:
    """Concatenate two roll waves of the same prompt into one pool.

    Used by the A-axis re-roll seam: each ``Allocation.MORE`` adds another
    sampling-group-sized wave, and the strategy re-judges the grown pool.
    """
    return TrajectoryGroup(
        trajectories_G=a.trajectories_G + b.trajectories_G,
        final_rewards_G=a.final_rewards_G + b.final_rewards_G,
        metrics_G=a.metrics_G + b.metrics_G,
        dynamic_batching_metadata={
            **a.dynamic_batching_metadata,
            **b.dynamic_batching_metadata,
        },
    )


def _assign_selection_seeds(
    group: TrajectoryGroup, base_seed: int | None
) -> TrajectoryGroup:
    """Attach stable opaque selector identities to one rollout wave."""

    trajectories = [
        replace(
            trajectory,
            selection_seed=(
                secrets.randbits(32)
                if base_seed is None
                else (base_seed + index) % (2**32)
            ),
        )
        for index, trajectory in enumerate(group.trajectories_G)
    ]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=group.final_rewards_G,
        metrics_G=group.metrics_G,
        dynamic_batching_metadata=group.dynamic_batching_metadata,
    )


@scope
def _make_rollout_failure_policy(
    cfg: "Config",
) -> tuple[RolloutRetryPolicy | None, RolloutFailureTracker | None]:
    """Build the rollout retry policy + failure tracker, or (None, None) when disabled.

    Resilience is opt-in: it activates only when a timeout or a retry budget is
    configured, so default runs keep the previous single-attempt behavior.
    """
    enabled = (
        cfg.sample_timeout_sec is not None
        or cfg.max_retries_per_trajectory > 0
        or cfg.max_extra_trajectory_attempts_per_group > 0
    )
    if not enabled:
        return None, None

    return (
        RolloutRetryPolicy(
            max_retries_per_trajectory=cfg.max_retries_per_trajectory,
            max_extra_attempts_per_group=cfg.max_extra_trajectory_attempts_per_group,
        ),
        RolloutFailureTracker(
            window_size=cfg.rollout_failure_window_size,
            warn_rate=cfg.rollout_failure_rate_warn,
            abort_rate=cfg.rollout_failure_rate_abort,
            min_attempts=cfg.rollout_failure_min_attempts,
            max_resets=cfg.rollout_failure_max_resets,
            reset_backoff_sec=cfg.rollout_failure_reset_backoff_sec,
        ),
    )


async def _handle_incomplete_group_drop(
    event: GroupDropEvent,
    *,
    allowed: bool,
    callback: GroupDropCallback | None,
) -> None:
    if not allowed:
        raise event.error
    if callback is not None:
        result = callback(event)
        if inspect.isawaitable(result):
            await result
    else:
        logger.warning(
            "Dropping entire training group before advantage computation; "
            "training_step=%s trajectories_dropped=%d: %s",
            event.training_step, event.trajectory_count, event.error,
        )


async def do_group_rollout_and_filter_constant_reward(
    sampling_client: FineTuningSession,
    env_group_builder: EnvGroupBuilder | AutonomousRolloutGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
    top_p: float = 1.0,
    top_k: int = -1,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    strategy: DynamicBatchStrategy | None = None,
    max_rolls_per_group: int | None = None,
    sample_timeout_sec: float | None = None,
    retry_policy: RolloutRetryPolicy | None = None,
    failure_tracker: RolloutFailureTracker | None = None,
    replenish_exhausted_group: bool = False,
    on_prefilter_group: Callable[[TrajectoryGroup], None] | None = None,
    allow_incomplete_grading_group_drop: bool = False,
    on_group_drop: GroupDropCallback | None = None,
    training_step: int | None = None,
    training_group_index: int | None = None,
) -> TrajectoryGroup | None:
    if isinstance(env_group_builder, AutonomousRolloutGroupBuilder):
        incompatible_settings: list[str] = []
        if strategy is not None:
            incompatible_settings.append("strategy")
        if max_rolls_per_group is not None:
            incompatible_settings.append("max_rolls_per_group")
        if retry_policy is not None and (
            retry_policy.max_retries_per_trajectory > 0
            or retry_policy.max_extra_attempts_per_group > 0
        ):
            incompatible_settings.append("retry_policy")
        if failure_tracker is not None and (
            failure_tracker.window_size != 100
            or failure_tracker.warn_rate != 0.10
            or failure_tracker.abort_rate != 0.20
            or failure_tracker.min_attempts != 25
            or failure_tracker.max_resets != 0
            or failure_tracker.reset_backoff_sec != 60.0
        ):
            incompatible_settings.append("failure_tracker")
        if replenish_exhausted_group:
            incompatible_settings.append("replenish_exhausted_group")
        if incompatible_settings:
            raise ValueError(
                "AutonomousRolloutGroupBuilder does not support rollout "
                "strategy, retry, or refill settings: "
                + ", ".join(incompatible_settings)
            )

        sampling_checkpoint = getattr(sampling_client, "sampling_checkpoint", None)
        if not isinstance(sampling_checkpoint, SamplingCheckpoint):
            raise TypeError(
                "AutonomousRolloutGroupBuilder requires a sampling client with "
                "a SamplingCheckpoint-valued sampling_checkpoint property"
            )
        sampling_options = RolloutSamplingOptions(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            sample_timeout_sec=sample_timeout_sec,
        )
        with logtree.optional_enable_logging(enable_logging):
            trajectory_group = await env_group_builder.run_autonomous_rollout(
                sampling_checkpoint,
                sampling_options,
            )
        if trajectory_group is None:
            # Autonomous builders can reject a group after exhausting a
            # recipe-owned recovery policy; the normal rollout filter replenishes it.
            return None
        if on_prefilter_group is not None:
            on_prefilter_group(trajectory_group)
        if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
            return None
        return trajectory_group

    if training_step is not None and training_group_index is not None:
        env_group_builder = env_group_builder.with_training_context(
            TrainingRolloutContext(training_step, training_group_index)
        )

    policy = SessionTokenCompleter(
        sampling_client, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k, seed=seed, response_format=response_format,
        sample_timeout_sec=sample_timeout_sec,
    )

    with logtree.optional_enable_logging(enable_logging):
        # Primary roll with optional resilience (retry budget + failure-rate
        # circuit breaker). Falls back to a single attempt when disabled.
        while True:
            try:
                if retry_policy is None and failure_tracker is None:
                    trajectory_group = await do_group_rollout(
                        env_group_builder, policy
                    )
                else:
                    trajectory_group = await do_group_rollout(
                        env_group_builder,
                        policy,
                        retry_policy=retry_policy,
                        failure_tracker=failure_tracker,
                    )
            except IncompleteGroupGradingError as error:
                await _handle_incomplete_group_drop(
                    GroupDropEvent(error, training_step),
                    allowed=allow_incomplete_grading_group_drop,
                    callback=on_group_drop,
                )
                return None
            except RolloutGroupExhausted:
                if not replenish_exhausted_group:
                    raise
                logger.exception(
                    "Rollout group exhausted its retry budget; dropping the group "
                    "(it will be replenished)"
                )
                return None
            except RolloutFailureRateExceeded as exc:
                # Instead of aborting the run, optionally reset the breaker, back
                # off to let the backend recover, and retry the group (bounded by
                # rollout_failure_max_resets).
                backoff = (
                    failure_tracker.note_breaker_tripped()
                    if failure_tracker is not None
                    else None
                )
                if backoff is None:
                    raise
                logger.error(
                    "Rollout failure-rate breaker tripped (%s); recovery %d/%d — "
                    "pausing %.0fs then retrying the group",
                    exc,
                    failure_tracker.resets_used,
                    failure_tracker.max_resets,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            break
        trajectory_group = _assign_selection_seeds(
            trajectory_group, seed
        )

        # A-axis: ask the strategy what to do with this rolled group.
        allocation = None
        if strategy is not None:
            allocation = await _resolve_strategy_result(
                strategy.allocate(trajectory_group)
            )
            # MORE: re-roll the SAME prompt to grow its pool while the strategy
            # keeps asking for more signal and we stay under the roll cap. The
            # seam is opt-in via max_rolls_per_group; without a cap, MORE is
            # treated as admit (single-wave behavior).
            if max_rolls_per_group is not None:
                reroll_wave = 1
                while (
                    allocation.is_more
                    and len(trajectory_group.trajectories_G) < max_rolls_per_group
                ):
                    # Derive a distinct deterministic seed per re-roll wave so a
                    # seeded run doesn't replay the same rollout on every MORE
                    # wave. (The backend does not currently honor the sample seed
                    # at generation time, so this is defensive — it keeps MORE
                    # meaningful if that ever changes.)
                    wave_seed = None if seed is None else (seed + reroll_wave * 1009) % (2**31)
                    reroll_policy = SessionTokenCompleter(
                        sampling_client, max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p, top_k=top_k, seed=wave_seed,
                        response_format=response_format,
                        sample_timeout_sec=sample_timeout_sec,
                    )
                    try:
                        extra_group = await do_group_rollout(
                            env_group_builder, reroll_policy
                        )
                    except IncompleteGroupGradingError as error:
                        await _handle_incomplete_group_drop(
                            GroupDropEvent(
                                error, training_step, len(trajectory_group.trajectories_G),
                            ),
                            allowed=allow_incomplete_grading_group_drop,
                            callback=on_group_drop,
                        )
                        return None
                    extra_group = _assign_selection_seeds(
                        extra_group,
                        (
                            None
                            if seed is None
                            else (seed + len(trajectory_group.trajectories_G))
                            % (2**32)
                        ),
                    )
                    trajectory_group = _merge_trajectory_groups(trajectory_group, extra_group)
                    allocation = await _resolve_strategy_result(
                        strategy.allocate(trajectory_group)
                    )
                    reroll_wave += 1

    # Surface how often the generation budget had to be clamped to fit the model
    # context window, so it shows up on the dashboard.
    if policy.context_clamp_count and trajectory_group.metrics_G:
        trajectory_group.metrics_G[0]["rollout/context_clamps"] = float(
            policy.context_clamp_count
        )

    if on_prefilter_group is not None:
        on_prefilter_group(trajectory_group)

    # Remove if all trajectories have the same reward
    if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
        return None
    # A-axis DROP: honor an explicit DROP from the settled allocation. MORE
    # (cap hit) and ADMIT both keep the group.
    if allocation is not None and allocation.is_drop:
        return None
    return trajectory_group


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: FineTuningSession,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
    ttl_seconds: int | None = None,
    eval_every: int = 0,
    loop_state_extra: dict[str, Any] | None = None,
    checkpoint_metrics: dict[str, Any] | None = None,
) -> tuple[FineTuningSession | None, dict[str, Any]]:
    """
    Save a checkpoint on save_every boundaries (full retention via ttl_seconds)
    OR on eval_every boundaries (so the iteration whose test-set metric we just
    logged is recoverable). Otherwise just return a fresh sampling client without
    persisting a checkpoint row to checkpoints.jsonl.
    """
    metrics = {}
    with timed("save_checkpoint", metrics):
        on_save_boundary = save_every > 0 and i_batch % save_every == 0
        on_eval_boundary = eval_every > 0 and i_batch % eval_every == 0
        if i_batch > start_batch and (on_save_boundary or on_eval_boundary):
            loop_state = {"batch": i_batch}
            if loop_state_extra:
                loop_state.update(loop_state_extra)
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=str(i_batch),
                log_path=log_path,
                loop_state=loop_state,
                kind="both",
                ttl_seconds=ttl_seconds,
                step_number=i_batch - 1,
                metrics=checkpoint_metrics,
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics


def _materialize_subgroup(
    group: TrajectoryGroup, kept_indices: list[int]
) -> TrajectoryGroup:
    """Build the surviving sub-group from a strategy ``Selection``.

    Slices the same fields ``assemble_training_data`` reads, so the
    all-kept case (FixedStrategy) reproduces the original group's training
    data exactly.
    """
    return TrajectoryGroup(
        trajectories_G=[group.trajectories_G[i] for i in kept_indices],
        final_rewards_G=[group.final_rewards_G[i] for i in kept_indices],
        metrics_G=[group.metrics_G[i] for i in kept_indices],
        dynamic_batching_metadata=group.dynamic_batching_metadata,
    )


async def _apply_selections(
    trajectory_groups_P: list[TrajectoryGroup],
    strategy: DynamicBatchStrategy,
    metrics: dict[str, Any],
) -> tuple[
    list[TrajectoryGroup],
    list[torch.Tensor],
    list[list[list[list[float]]] | None],
]:
    """B-axis seam: let the strategy choose survivors + advantages.

    Returns the kept sub-groups and their aligned advantage tensors, ready
    for ``assemble_training_data``. Groups whose selection keeps nothing
    are dropped. Records selection metrics into ``metrics``.
    """
    selections = await _resolve_strategy_result(
        strategy.select(trajectory_groups_P)
    )
    n_traj_before = sum(len(g.trajectories_G) for g in trajectory_groups_P)
    kept_groups: list[TrajectoryGroup] = []
    advantages_P: list[torch.Tensor] = []
    token_advantage_adjustments_P: list[
        list[list[list[float]]] | None
    ] = []
    for group, selection in safezip(trajectory_groups_P, selections):
        if not selection.kept_indices:
            continue
        kept_groups.append(_materialize_subgroup(group, selection.kept_indices))
        advantages_P.append(selection.advantages)
        token_advantage_adjustments_P.append(
            selection.token_advantage_adjustments
        )
    n_traj_after = sum(len(g.trajectories_G) for g in kept_groups)
    metrics["strategy/n_groups_in"] = len(trajectory_groups_P)
    metrics["strategy/n_groups_kept"] = len(kept_groups)
    metrics["strategy/n_traj_kept"] = n_traj_after
    metrics["strategy/select_keep_rate"] = (
        n_traj_after / n_traj_before if n_traj_before > 0 else 0.0
    )
    return kept_groups, advantages_P, token_advantage_adjustments_P


async def _assemble_training_data_async(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
    token_advantage_adjustments_P: list[
        list[list[list[float]]] | None
    ]
    | None = None,
) -> tuple[list[Datum], list[dict[str, int]]]:
    """Assemble training data off-loop without abandoning work on cancellation."""
    assembly_args = (
        (trajectory_groups_P, advantages_P)
        if token_advantage_adjustments_P is None
        else (
            trajectory_groups_P,
            advantages_P,
            token_advantage_adjustments_P,
        )
    )
    worker = asyncio.create_task(
        asyncio.to_thread(assemble_training_data, *assembly_args),
        name="assemble_training_data",
    )
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            if worker.cancelled():
                raise
            cancelled = exc
        except Exception:
            if cancelled is None:
                raise
            logger.exception("assemble_training_data failed after cancellation")
            raise cancelled

    if cancelled is not None:
        raise cancelled
    return result


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    kl_reference_client: FineTuningSession | None,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    max_tokens: int | None = None,
    num_groups_to_log: int = 2,
    max_console_lines_per_datum: int = 200,
    strategy: DynamicBatchStrategy | None = None,
) -> tuple[list[Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P, max_tokens=max_tokens))

    # Print a bounded number of trajectory groups. Offloaded to a worker thread so
    # the heavy tokenizer decoding/colorizing cannot block the event loop and starve
    # the SDK heartbeat task (which would let the server expire/unload the session).
    for traj_group in trajectory_groups_P[:num_groups_to_log]:
        await asyncio.to_thread(
            print_group, traj_group, tokenizer, max_console_lines_per_datum
        )

    # Assemble training data
    with timed("assemble_training_data", metrics):
        if strategy is None:
            # Default path: unchanged vanilla GRPO advantages over all samples.
            advantages_P = compute_advantages(trajectory_groups_P)
            data_D, _metadata_D = await _assemble_training_data_async(
                trajectory_groups_P, advantages_P
            )
        else:
            # B-axis: the strategy chooses survivors + advantages.
            (
                kept_groups,
                advantages_P,
                token_advantage_adjustments_P,
            ) = await _apply_selections(trajectory_groups_P, strategy, metrics)
            if not kept_groups:
                logger.warning(
                    "Strategy select kept no groups this minibatch; batch will be empty"
                )
            data_D, _metadata_D = await _assemble_training_data_async(
                kept_groups,
                advantages_P,
                token_advantage_adjustments_P,
            )

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0 and kl_reference_client is not None:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                kl_reference_client,
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: FineTuningSession,
    i_batch: int,
    data_D: list[Datum],
    training_logprobs_D: list[torch.Tensor],
    log_path: str,
    save_every: int,
    do_compute_post_kl: bool,
    ttl_seconds: int | None = None,
    eval_every: int = 0,
    loop_state_extra: dict[str, Any] | None = None,
) -> tuple[FineTuningSession, dict[str, Any]]:
    """
    At the end of the iteration, this will compute metrics for the full batch
    and return the latest sampling client.

    The reason we return a sampling client is that if do_compute_post_kl is True,
    we need to create a sampling client from the post-update policy.
    """
    metrics = {}

    # Compute KL metrics
    with timed("compute_kl_sample_train", metrics):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client,
        # Save/resume batch is one past the internal step. The checkpoint row's
        # step_number remains the internal 0-based step for metric display.
        i_batch + 1,
        log_path,
        save_every,
        ttl_seconds=ttl_seconds,
        eval_every=eval_every,
        loop_state_extra=loop_state_extra,
        checkpoint_metrics=kl_sample_train_metrics,
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if do_compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


@scope
async def do_train_step_streaming_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[_TrajectoryGroupQueueItem],
    training_client: FineTuningSession,
    kl_reference_client: FineTuningSession | None,
    tokenizer: Tokenizer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool] = lambda _: True,
    prefilter_rewards: PrefilterRewardAccumulator | None = None,
    loop_state_extra: dict[str, Any] | None = None,
) -> tuple[FineTuningSession | None, dict[str, Any]]:
    """
    As soon as we have enough trajectories for a minibatch, we will train on them.
    This allows us to overlap sampling and training.
    """
    assert cfg.stream_minibatch_config is not None
    assert cfg.stream_minibatch_config.groups_per_batch % cfg.num_substeps == 0, (
        f"{cfg.stream_minibatch_config.groups_per_batch=} must be divisible by {cfg.num_substeps=}"
    )
    # Number of groups across all minibatches in each optimizer substep
    groups_per_substep = cfg.stream_minibatch_config.groups_per_batch // cfg.num_substeps
    assert groups_per_substep % cfg.stream_minibatch_config.num_minibatches == 0, (
        f"{groups_per_substep} must be divisible by {cfg.stream_minibatch_config.num_minibatches=}"
    )
    # Number of groups per minibatch in each optimizer substep
    groups_per_minibatch = groups_per_substep // cfg.stream_minibatch_config.num_minibatches

    update_scope_context({"step": i_batch})

    metrics = {}

    # Console-logging budget for the whole training step. prepare_minibatch is
    # called once per minibatch/substep, so without a shared budget the per-datum
    # cap would apply per minibatch and print up to
    # num_groups_to_log * num_substeps * num_minibatches groups per step. Track a
    # remaining budget here and hand each minibatch only what's left so the
    # documented "per training step" cap holds.
    remaining_groups_to_log = _effective_num_groups_to_log(cfg)

    # Run multiple optimizer substeps per training iteration
    all_data_D = []
    all_training_logprobs_D = []
    all_wrapped_trajectory_groups = []
    for i_substep in range(cfg.num_substeps):
        # Run multiple minibatches per substep
        # Once we have enough trajectories for a minibatch, train on them
        wrapped_trajectory_groups = []
        forward_backward_futures: list[Any] = []  # APIFuture removed
        i_minibatch = 0
        while i_minibatch < cfg.stream_minibatch_config.num_minibatches:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if isinstance(wrapped_trajectory_group, Exception):
                raise wrapped_trajectory_group
            if isinstance(wrapped_trajectory_group, GroupDropEvent):
                # Async workers preserve grading drops while refilling a minibatch.
                metrics["async/dropped_groups"] = metrics.get("async/dropped_groups", 0) + 1
                continue
            if (
                wrapped_trajectory_group is not None
                and wrapped_trajectory_group.trajectory_group is None
            ):
                if prefilter_rewards is not None:
                    prefilter_rewards.merge(wrapped_trajectory_group.prefilter_rewards)
                continue
            if not trajectory_group_filter(wrapped_trajectory_group):
                continue
            if wrapped_trajectory_group is not None and prefilter_rewards is not None:
                prefilter_rewards.merge(wrapped_trajectory_group.prefilter_rewards)
            wrapped_trajectory_groups.append(wrapped_trajectory_group)

            if len(wrapped_trajectory_groups) < groups_per_minibatch:
                continue
            logger.info(
                f"[stream_minibatch] Step {i_batch}, Substep {i_substep}/{cfg.num_substeps}, Minibatch {i_minibatch}/{cfg.stream_minibatch_config.num_minibatches}: Will train on minibatch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Note: we may have removed trajectory groups that have the same reward.
            # To have the same results as the sync implementation, we will
            # remove these and train on a smaller batch.
            wrapped_trajectory_groups = _trainable_wrapped_groups(wrapped_trajectory_groups)
            if len(wrapped_trajectory_groups) == 0:
                i_minibatch += 1
                continue

            data_D, prepare_minibatch_metrics = await prepare_minibatch(
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                tokenizer,
                kl_reference_client,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
                max_tokens=cfg.max_tokens,
                num_groups_to_log=remaining_groups_to_log,
                max_console_lines_per_datum=cfg.max_console_lines_per_datum,
                strategy=cfg.strategy,
            )
            # prepare_minibatch prints min(num_groups_to_log, len(groups)) groups;
            # debit the per-step budget by however many it actually consumed.
            remaining_groups_to_log = max(
                0, remaining_groups_to_log - len(wrapped_trajectory_groups)
            )
            metrics.update(prepare_minibatch_metrics)

            if not data_D:
                i_minibatch += 1
                wrapped_trajectory_groups = []
                continue

            # Enqueue forward-backward (we'll await results after all minibatches are enqueued)
            with timed(f"train/fwd_bwd_substep_{i_substep}_mb_{i_minibatch}_enqueue", metrics):
                forward_backward_futures.append(
                    await training_client.forward_backward_async(
                        data_D,
                        loss_fn=cfg.loss_fn,
                        loss_fn_config=cfg.loss_fn_config,
                    )
                )
            all_data_D.extend(data_D)
            all_wrapped_trajectory_groups.extend(wrapped_trajectory_groups)
            i_minibatch += 1
            wrapped_trajectory_groups = []

        if not forward_backward_futures:
            continue

        # Enqueue optim_step before awaiting results (so they land on same clock cycle)
        adam_params = AdamParams(
            learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        with timed(f"train/optim_substep_{i_substep}_enqueue", metrics):
            optim_future = await training_client.optim_step_async(adam_params)

        # Now consume all forward-backward results
        for i_mb, fwd_bwd_future in enumerate(forward_backward_futures):
            with timed(f"train/fwd_bwd_substep_{i_substep}_mb_{i_mb}_consume", metrics):
                fwd_bwd_result = await fwd_bwd_future.result_async()
                all_training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
                if getattr(fwd_bwd_result, "metrics", None):
                    metrics.update(fwd_bwd_result.metrics)

        with timed(f"train/optim_substep_{i_substep}_consume", metrics):
            optim_result = await optim_future.result_async()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

    if not all_wrapped_trajectory_groups:
        if prefilter_rewards is not None:
            metrics.update(prefilter_rewards.metrics())
        metrics["train/skipped_no_valid_groups"] = 1
        logger.warning(
            "Step %d: no valid trajectory groups after streaming/filtering, skipping training step",
            i_batch,
        )
        return None, metrics

    # Aggregate metrics across the entire batch
    metrics.update(compute_sampling_client_metrics(all_wrapped_trajectory_groups))
    if prefilter_rewards is not None:
        metrics.update(prefilter_rewards.metrics())
    metrics.update(
        compute_trajectory_metrics(
            [g.trajectory_group for g in all_wrapped_trajectory_groups],
            [g.env_group_builder.logging_tags() for g in all_wrapped_trajectory_groups],
            max_tokens=cfg.max_tokens,
        )
    )
    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch,
        all_data_D,
        all_training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
        cfg.ttl_seconds,
        eval_every=checkpoint_eval_every(cfg),
        loop_state_extra=loop_state_extra,
    )
    metrics.update(full_batch_metrics)
    return sampling_client, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: FineTuningSession,
    kl_reference_client: FineTuningSession | None,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    loop_state_extra: dict[str, Any] | None = None,
) -> tuple[FineTuningSession, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        kl_reference_client,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        max_tokens=cfg.max_tokens,
        num_groups_to_log=_effective_num_groups_to_log(cfg),
        max_console_lines_per_datum=cfg.max_console_lines_per_datum,
        strategy=cfg.strategy,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=metrics,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
        cfg.ttl_seconds,
        eval_every=checkpoint_eval_every(cfg),
        loop_state_extra=loop_state_extra,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: FineTuningSession,
    kl_reference_client: FineTuningSession | None,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    resume_prompt_cursor: int | None = None,
    final_loop_state: dict[str, Any] | None = None,
    deadline: float | None = None,
    is_fresh_run: bool | None = None,
):
    """Implements fully synchronous on-policy training"""
    fresh_run = start_batch == 0 if is_fresh_run is None else is_fresh_run

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch,
        cfg.ttl_seconds,
        eval_every=checkpoint_eval_every(cfg),
    )
    retry_policy, failure_tracker = _make_rollout_failure_policy(cfg)

    # Refill-on-drop: run the batch-refill loop when either dynamic_sampling is
    # on (survivor = non-constant-reward) OR a strategy asked for refill_on_drop
    # (survivor = not dropped by the strategy's allocate). Both keep the trained
    # batch at groups_per_batch instead of letting drops shrink it. Config
    # validation guarantees refill_on_drop implies a (stateless) strategy on the
    # sync path.
    do_refill = cfg.dynamic_sampling or (
        cfg.strategy is not None and cfg.refill_on_drop
    )

    # When dynamic_sampling is enabled, we decouple the training step counter
    # from the dataset batch cursor. The training step stays the same, but we
    # may consume more dataset batches to fill the target group count.
    prompt_cursor = resume_prompt_cursor if resume_prompt_cursor is not None else start_batch
    if cfg.dynamic_sampling:
        warnings.warn(
            "dynamic_sampling is experimental and may change or be removed in future releases.",
            FutureWarning,
            stacklevel=2,
        )

    # Compute the target batch size once from the first dataset batch we will
    # actually draw from, rather than re-querying every step (and hard-coding
    # batch 0 as the source of truth for sizing).
    target_groups: int | None = (
        len(dataset.get_batch(prompt_cursor % len(dataset))) if do_refill else None
    )

    # Carry over excess prompts from one step to the next so we don't
    # waste prompts that were already drawn from the dataset.
    carryover_builders: list[EnvGroupBuilder] = []

    actual_next_batch = start_batch
    last_completed_step: int | None = None
    for i_batch in range(start_batch, end_batch):
        if deadline is not None and time.time() >= deadline:
            logger.info(
                "[sync] wall-clock budget reached at step %d/%d; stopping",
                i_batch, end_batch,
            )
            break
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()
        prefilter_rewards = PrefilterRewardAccumulator()

        # Run evaluations
        if should_evaluate_rl_iteration(
            eval_strategy=cfg.eval_strategy,
            eval_every=cfg.eval_every,
            step=i_batch,
            is_fresh_run=fresh_run,
        ):
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        sampling_started_at = time.time()
        if do_refill:
            # Dynamic sampling / refill-on-drop: keep pulling prompt batches and
            # sampling until we have enough surviving groups to fill the batch.
            # Survivor = passes the active filter(s): native constant-reward
            # drop (dynamic_sampling) and/or the strategy's allocate() DROP.
            assert target_groups is not None
            carryover_in = len(carryover_builders)
            # Prepend carried-over prompts so they get fresh rollouts first
            pending_builders: list[EnvGroupBuilder] = list(carryover_builders)
            carryover_builders.clear()
            all_trajectory_groups: list[TrajectoryGroup] = []
            all_env_group_builders: list[EnvGroupBuilder] = []
            rounds = 0
            total_groups_sampled = 0

            while len(all_trajectory_groups) < target_groups:
                rounds += 1

                if pending_builders:
                    # Use carried-over prompts first (fresh rollouts)
                    env_group_builders_P = pending_builders
                    pending_builders = []
                else:
                    batch_idx = prompt_cursor % len(dataset)
                    prompt_cursor += 1
                    env_group_builders_P = dataset.get_batch(batch_idx)

                if any(
                    isinstance(builder, AutonomousRolloutGroupBuilder)
                    for builder in env_group_builders_P
                ):
                    raise ValueError(
                        "AutonomousRolloutGroupBuilder cannot be used with "
                        "dynamic_sampling or refill_on_drop."
                    )

                # After round 1, right-size the top-up using the observed
                # filter rate so we sample just enough to fill the deficit.
                if rounds > 1 and total_groups_sampled > 0:
                    deficit = target_groups - len(all_trajectory_groups)
                    pass_rate = len(all_trajectory_groups) / total_groups_sampled
                    needed = int(np.ceil(deficit / max(pass_rate, 0.1) * cfg.oversample_cushion))
                    env_group_builders_P = env_group_builders_P[:needed]

                with _get_logtree_scope(
                    log_path=cfg.log_path,
                    num_groups_to_log=(
                        _effective_num_groups_to_log(cfg) if rounds == 1 else 0
                    ),
                    f_name=f"train_iteration_{i_batch:06d}",
                    scope_name=f"RL Iteration {i_batch}",
                ):
                    trajectory_groups_P = await gather_with_progress(
                        (
                            do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                max_tokens=cfg.max_tokens,
                                temperature=cfg.temperature,
                                # Native constant-reward drop applies when
                                # dynamic_sampling forces it, or when the user
                                # explicitly asks for it (parity with the
                                # standard branch's post-filter). For a pure
                                # refill_on_drop run the strategy owns drops, so
                                # this stays False unless requested.
                                do_remove_constant_reward_groups=(
                                    cfg.dynamic_sampling or cfg.remove_constant_reward_groups
                                ),
                                enable_logging=(
                                    rounds == 1
                                    and i < _effective_num_groups_to_log(cfg)
                                ),
                                top_p=cfg.top_p,
                                top_k=cfg.top_k,
                                response_format=cfg.response_format,
                                seed=_derive_group_seed(
                                    cfg.sampling_seed,
                                    i_batch,
                                    total_groups_sampled + i,
                                ),
                                strategy=cfg.strategy,
                                max_rolls_per_group=cfg.max_rolls_per_group,
                                sample_timeout_sec=cfg.sample_timeout_sec,
                                retry_policy=retry_policy,
                                failure_tracker=failure_tracker,
                                allow_incomplete_grading_group_drop=cfg.allow_incomplete_grading_group_drop,
                                on_group_drop=cfg.on_group_drop,
                                training_step=i_batch,
                                training_group_index=total_groups_sampled + i,
                                replenish_exhausted_group=not isinstance(
                                    builder, AutonomousRolloutGroupBuilder
                                ),
                                on_prefilter_group=prefilter_rewards.add,
                            )
                            for i, builder in enumerate(env_group_builders_P)
                        ),
                        desc=f"Sampling batch {i_batch} (round {rounds})",
                        max_concurrency=cfg.max_concurrent_groups,
                    )

                # Collect non-None groups (constant-reward groups were filtered)
                total_groups_sampled += len(trajectory_groups_P)
                for builder, group in zip(env_group_builders_P, trajectory_groups_P):
                    if group is not None:
                        all_trajectory_groups.append(group)
                        all_env_group_builders.append(builder)

                if rounds >= cfg.max_oversample_rounds:
                    logger.warning(
                        f"Dynamic sampling hit max_oversample_rounds={cfg.max_oversample_rounds} "
                        f"with {len(all_trajectory_groups)}/{target_groups} valid groups"
                    )
                    break

            # Trim to target size; save excess prompts for next step
            trajectory_groups_P = all_trajectory_groups[:target_groups]
            env_group_builders_P = all_env_group_builders[:target_groups]
            carryover_builders = all_env_group_builders[target_groups:]
            metrics["dynamic_sampling/rounds"] = rounds
            metrics["dynamic_sampling/valid_groups"] = len(trajectory_groups_P)
            metrics["dynamic_sampling/carryover_in"] = carryover_in
            metrics["dynamic_sampling/carryover_out"] = len(carryover_builders)
            metrics["dynamic_sampling/total_groups_sampled"] = total_groups_sampled
            metrics["dynamic_sampling/filtered_groups"] = total_groups_sampled - len(all_trajectory_groups)
            metrics["dynamic_sampling/filter_rate"] = (
                (total_groups_sampled - len(all_trajectory_groups)) / total_groups_sampled
                if total_groups_sampled > 0
                else 0.0
            )
            metrics["dynamic_sampling/prompt_cursor"] = prompt_cursor
            # Mirror the strategy-drop count under the same key the shrink path
            # uses, so apples-to-apples analysis reads one metric regardless of
            # whether the batch was refilled. Here it is the number of groups
            # dropped (then replaced) rather than the net batch shrink.
            if cfg.strategy is not None:
                key = (
                    "sampling/n_dropped_groups" if cfg.allow_incomplete_grading_group_drop
                    else "sampling/n_dropped_by_strategy"
                )
                metrics[key] = total_groups_sampled - len(all_trajectory_groups)
            # TODO: Add dedicated dashboard panels for dynamic sampling metrics.
            # Key signals: filter_rate trending over time (model mastery),
            # prompt_cursor vs training step (dataset consumption rate),
            # and pre-filter reward distribution (unbiased accuracy estimate).

        else:
            # Standard fixed-batch sampling
            env_group_builders_P = dataset.get_batch(i_batch)

            # Initialize logtree trace for this iteration if logging is enabled
            with _get_logtree_scope(
                log_path=cfg.log_path,
                num_groups_to_log=_effective_num_groups_to_log(cfg),
                f_name=f"train_iteration_{i_batch:06d}",
                scope_name=f"RL Iteration {i_batch}",
            ):
                # Note: do_remove_constant_reward_groups=False here because we remove
                # constant reward groups after all rollouts are collected (below)
                trajectory_groups_P = await gather_with_progress(
                    (
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            do_remove_constant_reward_groups=False,
                            enable_logging=i < _effective_num_groups_to_log(cfg),
                            top_p=cfg.top_p,
                            top_k=cfg.top_k,
                            response_format=cfg.response_format,
                            seed=_derive_group_seed(cfg.sampling_seed, i_batch, i),
                            strategy=cfg.strategy,
                            max_rolls_per_group=cfg.max_rolls_per_group,
                            sample_timeout_sec=cfg.sample_timeout_sec,
                            retry_policy=retry_policy,
                            failure_tracker=failure_tracker,
                            allow_incomplete_grading_group_drop=cfg.allow_incomplete_grading_group_drop,
                            on_group_drop=cfg.on_group_drop,
                            training_step=i_batch,
                            training_group_index=i,
                            replenish_exhausted_group=not isinstance(
                                builder, AutonomousRolloutGroupBuilder
                            ),
                            on_prefilter_group=prefilter_rewards.add,
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ),
                    desc=f"Sampling batch {i_batch}",
                    max_concurrency=cfg.max_concurrent_groups,
                )

            # A-axis DROP (strategy) and/or an exhausted retry budget may have
            # returned None for some groups; drop those (and their
            # builders/keys, keeping the lists aligned) before the rest of the
            # step consumes them. This filter must run unconditionally --
            # rollout groups can be exhausted regardless of whether a
            # strategy is configured, and leaving a None in trajectory_groups_P
            # crashes remove_constant_reward_groups (or any other downstream
            # consumer) with an AttributeError.
            n_before_drop = len(trajectory_groups_P)
            kept_pairs = [
                (builder, group)
                for builder, group in zip(env_group_builders_P, trajectory_groups_P)
                if group is not None
            ]
            env_group_builders_P = [b for b, _ in kept_pairs]
            trajectory_groups_P = [g for _, g in kept_pairs]
            n_dropped = n_before_drop - len(trajectory_groups_P)
            if cfg.allow_incomplete_grading_group_drop:
                metrics["sampling/n_dropped_groups"] = n_dropped
            elif cfg.strategy is not None:
                # Surface how many groups the strategy's A-axis dropped, so an
                # apples-to-apples comparison can see per-step batch shrink.
                metrics["sampling/n_dropped_by_strategy"] = n_dropped
            elif n_dropped:
                metrics["sampling/n_dropped_exhausted"] = n_dropped

            if cfg.remove_constant_reward_groups:
                trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)

        metrics["time/sampling"] = time.time() - sampling_started_at
        metrics.update(prefilter_rewards.metrics())

        # Skip training if no valid groups remain (e.g. all were constant-reward)
        if len(trajectory_groups_P) == 0:
            logger.warning(
                f"Step {i_batch}: no valid trajectory groups after filtering, skipping training step"
            )
            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)
            actual_next_batch = i_batch + 1
            continue

        # Train step
        # Rewind prompt_cursor by 1 when we have leftover carryover so that
        # on resume (where in-memory carryover_builders is lost) the most
        # recently consumed batch is re-drawn and those prompts are not
        # silently dropped.
        #
        # Subtle invariant (intentionally NOT guarded): if a step were filled
        # entirely from prior-step carryover (no new batch consumed) AND still
        # produced leftover carryover, this would rewind below the true cursor.
        # That state is unreachable with the default oversample_cushion=1.2:
        # max single-step overshoot is ~`cushion * deficit < target`, so prior
        # carryover can never reach `target_groups`, meaning every step always
        # consumes at least one new batch. Worst-case impact if ever triggered
        # (e.g. with cushion >= 2.0): one batch of prompts re-read on resume.
        #
        # Gated on `do_refill` (not just dynamic_sampling): refill_on_drop
        # advances prompt_cursor the same way, so its cursor must be
        # checkpointed too or a resumed run re-reads already-consumed prompts.
        loop_state_extra = (
            {"prompt_cursor": prompt_cursor - (1 if carryover_builders else 0)}
            if do_refill
            else None
        )
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            kl_reference_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            loop_state_extra=loop_state_extra,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)
        actual_next_batch = i_batch + 1
        if train_step_metrics.get("train/skipped_no_valid_groups") != 1:
            last_completed_step = i_batch

    if final_loop_state is not None and do_refill:
        final_loop_state["prompt_cursor"] = prompt_cursor - (
            1 if carryover_builders else 0
        )
    return actual_next_batch, last_completed_step


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    e2e_start_time = time.time()

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=_config_for_logging(cfg),
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python interactive_training/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = FineTuningSessionClient(base_url=cfg.base_url)
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(
            RLTestSetEvaluator(
                maybe_test_dataset,
                max_tokens=cfg.max_tokens,
                num_groups_to_log=_effective_num_groups_to_log(cfg),
                sample_timeout_sec=cfg.sample_timeout_sec,
                max_retries_per_trajectory=cfg.max_retries_per_trajectory,
                max_extra_trajectory_attempts_per_group=(
                    cfg.max_extra_trajectory_attempts_per_group
                ),
                observer=cfg.validation_observer,
                response_format=cfg.response_format,
                require_full_validation=cfg.require_full_validation,
            )
        )

    num_batches = len(dataset)
    end_batch = compute_effective_end(
        start_batch=start_batch, num_batches=num_batches, max_steps=cfg.max_steps
    )
    if cfg.max_steps is not None:
        logger.info(
            f"Will train on {end_batch - start_batch} batches "
            f"(max_steps={cfg.max_steps}, dataset has {num_batches})"
        )
    else:
        logger.info(f"Will train on {num_batches} batches")

    # Create KL reference client once if KL penalty is enabled
    if cfg.kl_penalty_coef > 0:
        if cfg.kl_reference_config is None:
            raise ValueError("kl_reference_config must be specified when kl_penalty_coef > 0")
        kl_reference_client = service_client.create_sampling_client(
            base_model=cfg.kl_reference_config.base_model,
            model_path=cfg.kl_reference_config.load_checkpoint_path,
        )
    else:
        kl_reference_client = None

    # Training loop
    if cfg.async_config is not None:
        training_func = do_async_training
    elif cfg.stream_minibatch_config is not None:
        training_func = do_sync_training_with_stream_minibatch
    else:
        training_func = do_sync_training

    # Build kwargs common to all training functions
    training_kwargs: dict[str, Any] = dict(
        start_batch=start_batch,
        end_batch=end_batch,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        kl_reference_client=kl_reference_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        is_fresh_run=resume_info is None,
    )
    # Pass prompt_cursor resume state only for sync training
    if training_func is do_sync_training and resume_info:
        training_kwargs["resume_prompt_cursor"] = resume_info.get("prompt_cursor")
    if training_func is do_async_training and resume_info:
        training_kwargs["resume_group_ordinal"] = resume_info.get("async_group_ordinal")
    final_loop_state: dict[str, Any] = {}
    if training_func in (do_sync_training, do_async_training):
        training_kwargs["final_loop_state"] = final_loop_state
    # Wall-clock budget: convert the configured duration into an absolute
    # deadline so each training function breaks at its next iteration boundary
    # once the budget elapses. None/0 => no limit.
    if cfg.max_wall_clock_seconds is not None and cfg.max_wall_clock_seconds > 0:
        training_kwargs["deadline"] = e2e_start_time + cfg.max_wall_clock_seconds
        logger.info(
            f"max_wall_clock_seconds={cfg.max_wall_clock_seconds} — will stop at the "
            f"next iteration boundary after the budget elapses"
        )
    actual_next_batch, last_completed_step = await training_func(**training_kwargs)

    # Save final checkpoint
    if actual_next_batch > start_batch:
        final_checkpoint_kwargs: dict[str, Any] = {}
        if last_completed_step is not None:
            final_checkpoint_kwargs["step_number"] = last_completed_step
        final_paths = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": actual_next_batch, **final_loop_state},
            ttl_seconds=cfg.ttl_seconds,
            **final_checkpoint_kwargs,
        )
    else:
        logger.info("Training was already complete; nothing to do")
        final_paths = None

    # Final evaluation on the truly final model (after all training steps).
    if (
        final_paths is not None
        and last_completed_step is not None
        and len(evaluators) > 0
        and should_evaluate_rl_final(
            eval_strategy=cfg.eval_strategy,
            eval_every=cfg.eval_every,
            actual_next_batch=actual_next_batch,
            num_batches=num_batches,
        )
    ):
        sampling_client = training_client.create_sampling_client(final_paths["sampler_path"])
        metrics = {}
        with timed("run_evals", metrics):
            eval_metrics = await run_evaluations_parallel(
                evaluators, sampling_client, cfg, actual_next_batch
            )
            metrics.update(eval_metrics)
        ml_logger.log_metrics(metrics, step=actual_next_batch)

    # Record e2e wall-clock time
    e2e_seconds = time.time() - e2e_start_time
    h, rem = divmod(int(e2e_seconds), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Total wall-clock time: {h}h {m}m {s}s ({e2e_seconds/60:.1f} min)")
    try:
        timing_path = os.path.join(cfg.log_path, "timing.json")
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump({
                "e2e_wall_clock_seconds": round(e2e_seconds, 1),
                "e2e_wall_clock_minutes": round(e2e_seconds / 60, 1),
                "e2e_wall_clock_formatted": f"{h}h {m}m {s}s",
            }, f, indent=4)
    except Exception:
        pass

    # Cleanup
    service_client.close()
    ml_logger.close()
    logger.info("Training completed successfully")
