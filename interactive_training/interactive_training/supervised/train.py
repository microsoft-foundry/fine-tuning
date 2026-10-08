"""
Supervised fine-tuning (SFT)

This module implements a pipelined supervised learning training loop.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import chz
from azure.ai.finetuningsessions.models import AdamParams, TensorData

from interactive_training import checkpoint_utils
from interactive_training.display import colorize_example
from interactive_training.eval.evaluators import (
    Evaluator,
    EvaluatorBuilder,
    SamplingClientEvaluator,
    TrainingClientEvaluator,
)
from interactive_training.supervised.common import _chunk_len, compute_mean_nll
from interactive_training.supervised.nll_evaluator import NLLEvaluator
from interactive_training.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.utils import ml_log
from interactive_training.utils.lr_scheduling import compute_schedule_lr_multiplier, LRSchedule
from interactive_training.utils.misc_utils import timed
from interactive_training.utils.trace import scope, update_scope_context, trace_init

logger = logging.getLogger(__name__)

EvaluationStrategy = Literal["steps", "epoch"]


@chz.chz
class Config:
    """Configuration for supervised fine-tuning."""

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str
    load_checkpoint_path: str | None = None
    dataset_builder: SupervisedDatasetBuilder

    # Training parameters
    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    # Optional hard cap on training steps. When set (> 0), training stops after
    # this many optimizer steps even if `num_epochs` * len(dataset) is larger.
    # The LR schedule still uses `total_steps = n_batches * num_epochs` as its
    # denominator so caps act like early stopping (LR will not have decayed to
    # zero). Use 0/None to disable.
    max_steps: int | None = None

    # Wall-clock budget in seconds (None/0 = no limit). The training loop breaks
    # at the next batch boundary once elapsed time since the loop started exceeds
    # this. Mirrors rl.train.Config.max_wall_clock_seconds.
    max_wall_clock_seconds: float | None = None

    # Run seed. Threaded into the per-epoch dataset shuffle as
    # ``epoch_idx + seed`` for reproducibility; seed=0 reproduces the historical
    # epoch-index-only shuffle (back-compat).
    seed: int = 0

    # Model parameters
    lora_rank: int = 16

    # Checkpointing and evaluation. For steps, eval_every=0 disables evaluation;
    # epoch strategy requires eval_every=1.
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 20
    eval_every: int = 10
    infrequent_eval_every: int = 100
    ttl_seconds: int | None = 604800  # 7 days

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    enable_trace: bool = False

    # When True, skip logging of training example content (e.g. chat messages)
    sanitize_logs: bool = False

    # Number of training steps kept in flight (submitted but not yet awaited) so
    # the engine stays fed while the client polls results off the critical path.
    # 1 = classic one-step-ahead (submit step N, await N-1). Higher (e.g. 3)
    # pre-queues more fwd/optim pairs so the engine never idles waiting for the
    # client to discover a result and submit the next step -- the dominant
    # per-step coordination cost for small-batch SFT. The server runs the queued
    # ops in order, so gradient semantics are identical at any depth.
    #
    # Periodic eval and checkpointing ARE supported: the loop drains the in-flight
    # window at each eval/save boundary so they operate on SETTLED weights (a
    # checkpoint must capture its labeled step or resume double-applies; eval
    # snapshots the pre-step model). Between boundaries the window stays full, so
    # this costs only an occasional flush, not steady-state throughput.
    #
    # SFT-only by design: on-policy RL (``rl/train``) cannot use this -- each
    # step's rollouts are sampled from the PREVIOUS step's synced weights (a
    # per-step sampler barrier), so future fwd/optim can't be pre-queued. Do not
    # port this to ``rl/train``.
    pipeline_depth: int = 1

    # Render every dataset row before submitting any remote training operation.
    # Launchers can call prepare_datasets before creating a remote session to
    # move this validation ahead of session allocation as well.
    preflight_dataset: bool = False

    @chz.validate
    def _validate_pipeline_depth(self):
        if self.pipeline_depth < 1:
            raise ValueError(f"pipeline_depth must be >= 1, got {self.pipeline_depth}")

    @chz.validate
    def _validate_eval_cadence(self):
        # Under the epoch strategy, eval_every is expressed in epochs rather
        # than optimizer steps. Only 1 is currently supported: evaluate the
        # initial model and after every completed epoch.
        if self.eval_strategy == "epoch" and self.eval_every != 1:
            raise ValueError(
                "eval_strategy='epoch' requires eval_every=1; other epoch "
                "cadences are not supported. This evaluates before training "
                "and after every completed epoch. To disable evaluation "
                "entirely, use eval_strategy='steps' with eval_every=0."
            )


@dataclass
class SubmittedBatch:
    fwd_bwd_future: object  # _SDKFuture with .result_async()
    optim_step_future: object  # _SDKFuture with .result_async()
    metrics: dict[str, int | float | str]
    data: list
    step: int
    epoch_idx: int
    batch_idx: int
    batch_start_time: float
    eval_metrics: dict[str, float] | None = None
    infrequent_eval_metrics: dict[str, float] | None = None


PreparedDatasets = tuple[SupervisedDataset, SupervisedDataset | None]


def prepare_datasets(config: Config) -> PreparedDatasets:
    """Build and fully render datasets so invalid examples fail before a session starts."""
    dataset, maybe_test_dataset = config.dataset_builder.build()
    for split_name, split in (
        ("training", dataset),
        ("evaluation", maybe_test_dataset),
    ):
        if split is None:
            continue
        for batch_index in range(len(split)):
            try:
                split.get_batch(batch_index)
            except ValueError as exc:
                raise ValueError(
                    f"{split_name.capitalize()} dataset preflight failed: {exc}"
                ) from exc
    return dataset, maybe_test_dataset


@scope
async def run_evals(
    evaluators: list[Evaluator],
    training_client,
    step: int,
) -> dict[str, float]:
    """Evaluate the current model weights and prefix results with ``test/``.

    The helper is called immediately before optimizer step `step` is submitted, so it
    measures the weights produced after step `step-1` (or the initial weights for step 0).
    Training-client evaluators run against the mutable training client, while sampling
    evaluators request a fresh `SamplingClient` snapshot via
    `save_weights_and_get_sampling_client_async` to ensure their work uses a fixed
    checkpoint. Returned metrics are prefixed with ``test/`` so they can be logged next
    to the same-step training metrics.
    """
    update_scope_context({"step": step})

    metrics = {}
    sampling_client = None

    @scope
    async def run_evaluator(evaluator: Evaluator) -> dict[str, float]:
        update_scope_context(
            {
                "step": step,
                "evaluator_name": type(evaluator).__name__,
            }
        )
        if isinstance(evaluator, TrainingClientEvaluator):
            update_scope_context({"evaluator_type": "TrainingClientEvaluator"})
            return await evaluator(training_client)
        elif isinstance(evaluator, SamplingClientEvaluator):
            update_scope_context({"evaluator_type": "SamplingClientEvaluator"})
            # Create sampling client lazily, only when needed
            nonlocal sampling_client
            if sampling_client is None:
                # Snapshot the current pre-step weights and create a new sampling client.
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                    f"evals_step_{step}"
                )
            return await evaluator(sampling_client)
        else:
            raise ValueError(f"Unknown evaluator type: {type(evaluator)}")

    for evaluator in evaluators:
        eval_metrics = await run_evaluator(evaluator)
        # Add test/ prefix to all metrics
        metrics.update(eval_metrics)

    return metrics


def compute_resume_loop_state(
    *,
    last_step: int | None,
    start_epoch: int,
    start_batch: int,
    n_batches: int,
    num_epochs: int,
) -> dict[str, int]:
    """The ``(epoch, batch)`` a resume should continue from after this run.

    Written into the ``final`` checkpoint's ``loop_state`` so auto-resume picks
    up where training actually stopped — not a hardcoded "fully complete" state.

    ``last_step`` is the last optimizer step index executed this call
    (``epoch_idx * n_batches + batch_idx``), or ``None`` if nothing trained. The
    next step to run is ``last_step + 1``; if nothing trained (already at the
    ``max_steps`` cap) we keep the position we started at so the checkpoint does
    not move. A genuinely complete run yields ``epoch == num_epochs`` (so
    ``range(num_epochs, num_epochs)`` is empty), which is how the training loop
    detects "already complete" on the next resume.

    Hardcoding ``{num_epochs, n_batches}`` (the old behavior) made a
    ``max_steps``-capped run stamp itself complete, so a later resume that raised
    ``max_steps`` would no-op instead of continuing.
    """
    if n_batches <= 0:
        # Degenerate (dataset smaller than a batch): nothing trains; preserve the
        # historical "complete" marker rather than dividing by zero.
        return {"epoch": num_epochs, "batch": n_batches}
    if last_step is not None:
        next_step = last_step + 1
    else:
        next_step = start_epoch * n_batches + start_batch
    epoch, batch = divmod(next_step, n_batches)
    return {"epoch": epoch, "batch": batch}


def final_eval_step(last_step: int) -> int:
    """The step index the post-training final evaluation is logged at.

    Co-located with the last completed training step (``last_step``), NOT
    ``last_step + 1``. The final eval measures the model *after* the last
    optimizer step. Reporting it on the last training step matches the shared
    RL loop's ``num_batches - 1`` convention and keeps ``train_loss`` and
    ``eval_loss`` on the same final step. A one-based display places both on
    step ``N``, avoiding a phantom ``N + 1`` row in ``results.csv``.

    This is a logging-only step selection; it feeds no control-flow index
    (LR schedule, eval/save cadence, checkpoint names, or resume loop state).
    """
    return last_step


def should_evaluate_before_step(
    *,
    eval_strategy: EvaluationStrategy,
    eval_every: int,
    step: int,
    is_fresh_run: bool,
) -> bool:
    """Return whether evaluation should snapshot the model before this step."""
    if eval_strategy == "steps":
        return eval_every > 0 and step % eval_every == 0
    if eval_strategy == "epoch":
        return is_fresh_run and step == 0
    raise ValueError(f"Unsupported eval_strategy: {eval_strategy!r}")


def should_evaluate_after_epoch(
    *,
    eval_strategy: EvaluationStrategy,
    epoch_completed: bool,
) -> bool:
    """Return whether evaluation should run after a fully completed epoch."""
    if eval_strategy == "steps":
        return False
    if eval_strategy == "epoch":
        return epoch_completed
    raise ValueError(f"Unsupported eval_strategy: {eval_strategy!r}")


def should_evaluate_final_model(
    *,
    eval_strategy: EvaluationStrategy,
    eval_every: int,
    stopped_early: bool,
    last_step: int | None,
    last_evaluated_step: int | None,
) -> bool:
    """Return whether to evaluate the trained model once after the loop ends.

    - ``steps``: preserves the legacy behavior of co-locating a final evaluation
      with the last optimizer step whenever periodic evaluation is enabled.
    - ``epoch``: completed epochs are already evaluated at their boundary, so a
      final evaluation is only needed when the run stopped mid-epoch
      (``max_steps`` / wall-clock) and the last trained step has not already
      been evaluated at an epoch boundary. This mirrors the RL loop, which
      evaluates the final model after a partial dataset traversal.
    """
    if last_step is None:
        return False
    if eval_strategy == "steps":
        return eval_every > 0
    if eval_strategy == "epoch":
        return stopped_early and last_step != last_evaluated_step
    raise ValueError(f"Unsupported eval_strategy: {eval_strategy!r}")


@scope
async def main(
    config: Config,
    training_client,
    prepared_datasets: PreparedDatasets | None = None,
):
    """Run the standard supervised learning loop.

    Responsibilities:
    1. Initialize logging, build the dataset/evaluator objects, and determine the
       ``epoch``/``batch`` indices to start from.
    2. Iterate over batches: fetch data, optionally run evaluations before submitting the
       optimizer step (so they observe pre-step weights), issue `forward_backward` and
       `optim_step` requests, and log metrics once the futures resolve.
    3. Save checkpoints at the configured cadence so runs can resume or export weights,
       then emit a final checkpoint when training completes.

    Training and evaluation metrics share the same ``step`` index to keep dashboards easy
    to read.
    """
    e2e_start_time = time.time()
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    is_fresh_run = resume_info is None
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0
    # (start_epoch, start_batch) now represent the next batch to execute if resuming.

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
    )
    if config.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(config.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        trace_init(output_file=os.path.join(config.log_path, "trace_events.jsonl"))

    if prepared_datasets is not None:
        dataset, maybe_test_dataset = prepared_datasets
    elif config.preflight_dataset:
        dataset, maybe_test_dataset = prepare_datasets(config)
    else:
        dataset, maybe_test_dataset = config.dataset_builder.build()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    # The LR schedule still uses `total_steps` as its denominator (see
    # Config.max_steps docstring), but for user-facing progress we report
    # against the actually planned number of steps so the dashboard reads true.
    # `max_train_examples` is already reflected here because the dataset
    # builder truncates before `len(dataset)`.
    if config.max_steps and config.max_steps > 0:
        planned_steps = min(total_steps, config.max_steps)
    else:
        planned_steps = total_steps
    progress_denominator = planned_steps if planned_steps > 0 else 1
    tokenizer = get_tokenizer(config.model_name)

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(NLLEvaluator.from_dataset(maybe_test_dataset))

    infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
    if planned_steps != total_steps:
        logger.info(
            f"Training for {planned_steps} steps "
            f"(capped by max_steps={config.max_steps}; "
            f"would otherwise be {n_batches} batches x {config.num_epochs} epochs = {total_steps} steps)"
        )
    else:
        logger.info(
            f"Training for {n_batches} batches x {config.num_epochs} epochs = {total_steps} steps"
        )

    @scope
    async def submit_batch(epoch_idx: int, batch_idx: int) -> SubmittedBatch:
        step = epoch_idx * n_batches + batch_idx
        update_scope_context({"step": step})

        set_ctx = getattr(training_client, "set_operation_context", None)
        reset_ctx = getattr(training_client, "reset_operation_context", None)
        ctx_token = set_ctx(f"step={step}") if set_ctx is not None else None

        batch_start_time = time.time()
        logger.info(
            f"[step_timeline] step={step} submit_start "
            f"epoch={epoch_idx} batch={batch_idx}"
        )
        metrics: dict[str, int | float | str] = {"epoch": epoch_idx}
        metrics["progress"] = step / progress_denominator

        learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=config.lr_schedule,
            step=step,
            total_steps=total_steps,
        )
        metrics["learning_rate"] = learning_rate

        adam_params = AdamParams(
            learning_rate=learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )

        try:
            with timed("get_batch", metrics):
                data = dataset.get_batch(batch_idx)
            if data and not config.sanitize_logs:
                logger.info(colorize_example(data[0], tokenizer))

            checkpoint_path = None
            if config.save_every > 0 and step % config.save_every == 0 and step > 0:
                with timed("save_checkpoint", metrics):
                    checkpoint_paths = await checkpoint_utils.save_checkpoint_async(
                        training_client=training_client,
                        name=str(step),
                        log_path=config.log_path,
                        loop_state={"epoch": epoch_idx, "batch": batch_idx},
                        kind="both",
                        ttl_seconds=config.ttl_seconds,
                        step_number=step,
                    )
                    checkpoint_path = checkpoint_paths.get("state_path")

            # Trigger evaluations BEFORE submitting training operations so they snapshot pre-step weights
            eval_metrics = None
            if evaluators and should_evaluate_before_step(
                eval_strategy=config.eval_strategy,
                eval_every=config.eval_every,
                step=step,
                is_fresh_run=is_fresh_run,
            ):
                with timed("evals", metrics):
                    eval_metrics = await run_evals(evaluators, training_client, step)
                if checkpoint_path:
                    checkpoint_utils.record_checkpoint_evaluation(
                        log_path=config.log_path,
                        step=step,
                        checkpoint_path=checkpoint_path,
                        metrics=eval_metrics,
                    )

            infrequent_eval_metrics = None
            if (
                infrequent_evaluators
                and config.infrequent_eval_every > 0
                and step % config.infrequent_eval_every == 0
            ):
                with timed("infrequent_evals", metrics):
                    infrequent_eval_metrics = await run_evals(
                        infrequent_evaluators, training_client, step
                    )

            fwd_bwd_future = await training_client.forward_backward_async(
                data, loss_fn="cross_entropy"
            )
            optim_step_future = await training_client.optim_step_async(adam_params)
            logger.info(
                f"[step_timeline] step={step} submit_done "
                "(fwd+optim enqueued)"
            )
        finally:
            if reset_ctx is not None and ctx_token is not None:
                reset_ctx(ctx_token)

        return SubmittedBatch(
            fwd_bwd_future=fwd_bwd_future,
            optim_step_future=optim_step_future,
            metrics=metrics,
            data=data,
            step=step,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batch_start_time=batch_start_time,
            eval_metrics=eval_metrics,
            infrequent_eval_metrics=infrequent_eval_metrics,
        )

    @scope
    async def finish_batch(submitted: SubmittedBatch):
        update_scope_context({"step": submitted.step})

        metrics = submitted.metrics
        metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)

        set_ctx = getattr(training_client, "set_operation_context", None)
        reset_ctx = getattr(training_client, "reset_operation_context", None)
        ctx_token = (
            set_ctx(f"step={submitted.step}") if set_ctx is not None else None
        )
        try:
            with timed("step", metrics):
                logger.info(
                    f"[step_timeline] step={submitted.step} retire_start "
                    "(awaiting fwd+optim results)"
                )
                fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
                optim_step_result = await submitted.optim_step_future.result_async()
                logger.info(
                    f"[step_timeline] step={submitted.step} retire_done "
                    "(results consumed)"
                )
        finally:
            if reset_ctx is not None and ctx_token is not None:
                reset_ctx(ctx_token)

        if optim_step_result.metrics:
            metrics.update(optim_step_result.metrics)

        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in submitted.data]
        train_nll = compute_mean_nll(logprobs, weights)

        metrics.update(
            num_sequences=len(submitted.data),
            num_tokens=sum(_datum_token_count(datum) for datum in submitted.data),
            num_loss_tokens=sum(
                sum(datum.loss_fn_inputs["weights"].data) for datum in submitted.data
            ),
            train_mean_nll=train_nll,
        )
        metrics["time/total"] = time.time() - submitted.batch_start_time

        # Merge evaluation metrics gathered before the training step was submitted
        if submitted.eval_metrics is not None:
            metrics.update(submitted.eval_metrics)

        if submitted.infrequent_eval_metrics is not None:
            metrics.update(submitted.infrequent_eval_metrics)

        # Emit all metrics for this step (train and eval) on the `submitted.step` row.
        ml_logger.log_metrics(metrics=metrics, step=submitted.step)

    # Warm up the inference engine before training starts. The first sampler-touching
    # call on a fresh session triggers inference-engine initialization on the service
    # side; doing it now, while trainer GPU memory is at its minimum (model weights
    # only, no optimizer state or activations yet), avoids contending with peak
    # training memory at the first periodic save. RL recipes get this priming for
    # free via the initial sampling client they build before the training loop, and
    # this call deliberately mirrors that pattern (unconditional, before the loop).
    #
    # The call is ephemeral: no checkpoint row is written and nothing is persisted.
    #
    # TODO(deploy-checkpoint-serialization): remove this warm-up once persisted
    # sampler/deploy checkpoints no longer require a live inference engine on the
    # service side. Producing a deploy artifact (especially for LoRA adapters) is a
    # pure serialization step and shouldn't depend on the inference runtime.
    logger.info("Warming up inference engine before training loop")
    await training_client.save_weights_and_get_sampling_client_async()

    inflight: deque[SubmittedBatch] = deque()
    # Hard step cap (e.g. to match a published recipe length without rebuilding
    # the dataset). 0/None means "run all epochs to completion".
    step_cap = config.max_steps if config.max_steps and config.max_steps > 0 else None
    if step_cap is not None:
        logger.info(f"max_steps={step_cap} — will stop early after {step_cap} optimizer steps")

    stop_early = False
    last_step: int | None = None
    # Tracks the step at which the trained model was last evaluated (epoch
    # boundary evals). Used to avoid double-evaluating the final model when an
    # early stop lands exactly on an already-evaluated epoch boundary.
    last_evaluated_step: int | None = None
    # Wall-clock budget (mirror rl.train): break at the next batch boundary once
    # elapsed time since the loop started exceeds the configured budget.
    wall_clock_deadline: float | None = None
    if config.max_wall_clock_seconds is not None and config.max_wall_clock_seconds > 0:
        wall_clock_deadline = time.time() + config.max_wall_clock_seconds
        logger.info(
            f"max_wall_clock_seconds={config.max_wall_clock_seconds} — will stop at "
            f"the next batch boundary after the budget elapses"
        )
    for epoch_idx in range(start_epoch, config.num_epochs):
        if stop_early:
            break
        logger.info(f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx + config.seed)

        epoch_completed = True
        start_batch_idx = start_batch if epoch_idx == start_epoch else 0
        for batch_idx in range(start_batch_idx, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if step_cap is not None and step >= step_cap:
                stop_early = True
                epoch_completed = False
                break
            if wall_clock_deadline is not None and time.time() >= wall_clock_deadline:
                logger.info(f"Wall-clock budget reached at step {step}; stopping.")
                stop_early = True
                epoch_completed = False
                break
            # Eval and periodic checkpointing both snapshot the same pre-step
            # weights inside submit_batch, so flush the in-flight window at either
            # boundary. Between boundaries the window stays full.
            will_eval = (
                bool(evaluators)
                and should_evaluate_before_step(
                    eval_strategy=config.eval_strategy,
                    eval_every=config.eval_every,
                    step=step,
                    is_fresh_run=is_fresh_run,
                )
            ) or (
                bool(infrequent_evaluators)
                and config.infrequent_eval_every > 0
                and step % config.infrequent_eval_every == 0
            )
            will_save = config.save_every > 0 and step % config.save_every == 0 and step > 0
            if will_eval or will_save:
                # Settle weights at step-1 before a pre-step snapshot runs.
                while inflight:
                    await finish_batch(inflight.popleft())

            submitted_batch = await submit_batch(epoch_idx, batch_idx)
            inflight.append(submitted_batch)
            last_step = step

            if will_save:
                # Keep later steps from being submitted until this boundary step
                # completes, preserving settled weights at the next boundary.
                while inflight:
                    await finish_batch(inflight.popleft())
            elif len(inflight) > config.pipeline_depth:
                # Keep `pipeline_depth` steps in flight; retire the oldest (which
                # is ~pipeline_depth steps behind the engine, so already complete,
                # so the await rarely blocks). This keeps the next fwd request
                # pre-queued so the engine never idles waiting for the client.
                await finish_batch(inflight.popleft())

        if evaluators and should_evaluate_after_epoch(
            eval_strategy=config.eval_strategy,
            epoch_completed=epoch_completed,
        ):
            while inflight:
                await finish_batch(inflight.popleft())
            if last_step is not None:
                logger.info(f"Running evaluation after epoch {epoch_idx} at step {last_step}")
                metrics: dict[str, int | float | str] = {}
                with timed("evals", metrics):
                    metrics.update(await run_evals(evaluators, training_client, last_step))
                ml_logger.log_metrics(metrics=metrics, step=last_step)
                last_evaluated_step = last_step

    while inflight:
        await finish_batch(inflight.popleft())

    if start_epoch < config.num_epochs:
        # Final checkpoint: state + sampler so the run produces a deployable artifact.
        # The training pipeline has drained by this point (last `finish_batch` has
        # already awaited fwd_bwd + optim_step), so this is the safest moment to take
        # a sampler snapshot even if the periodic save above turns out to be risky.
        #
        # Record the *actual* resume position, not a hardcoded "fully complete"
        # state. Mirrors rl/train_azure.py, which saves the effective end batch.
        # See compute_resume_loop_state for the semantics and the bug it fixes.
        loop_state = compute_resume_loop_state(
            last_step=last_step,
            start_epoch=start_epoch,
            start_batch=start_batch,
            n_batches=n_batches,
            num_epochs=config.num_epochs,
        )
        final_checkpoint_paths = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state=loop_state,
            ttl_seconds=config.ttl_seconds,
            step_number=last_step,
        )
        final_checkpoint_path = final_checkpoint_paths.get("state_path")

        # Final evaluation on the truly final model (after the last optimizer step).
        # Mirrors the pattern in rl/train.py: periodic evals run on `step %
        # eval_every == 0`, which won't necessarily land on the last step
        # (especially with `max_steps`). Without this, the last logged eval can
        # be hundreds of steps stale. SFT evaluators take a `training_client`
        # (not a sampling client), so this is just a forward pass — no vLLM
        # launch, no GPU OOM risk.
        run_final_eval = evaluators and should_evaluate_final_model(
            eval_strategy=config.eval_strategy,
            eval_every=config.eval_every,
            stopped_early=stop_early,
            last_step=last_step,
            last_evaluated_step=last_evaluated_step,
        )
        if run_final_eval and config.eval_strategy == "steps":
            # Co-locate the final evaluation with the last training step (see
            # final_eval_step), matching the RL loop. A one-based display then
            # puts it on the same step as the final training row. Logging-only;
            # no control-flow index changes. The later, post-training evaluation
            # value is logged at this step.
            eval_step = final_eval_step(last_step)
            logger.info(f"Running final evaluation at step {eval_step}")
            metrics: dict[str, int | float | str] = {}
            with timed("evals", metrics):
                for ev in evaluators:
                    metrics.update(await ev(training_client))
            if final_checkpoint_path:
                checkpoint_utils.record_checkpoint_evaluation(
                    log_path=config.log_path,
                    step=eval_step,
                    checkpoint_path=final_checkpoint_path,
                    metrics=metrics,
                )
            ml_logger.log_metrics(metrics=metrics, step=eval_step)
        elif run_final_eval:
            # epoch mode, early-stopped mid-epoch: the partial final epoch never
            # hit its boundary eval, so evaluate the trained model once here so
            # the run always reports a post-training metric (parity with RL).
            logger.info(
                f"Running final evaluation on early-stopped model at step {last_step}"
            )
            metrics: dict[str, int | float | str] = {}
            with timed("evals", metrics):
                metrics.update(await run_evals(evaluators, training_client, last_step))
            if final_checkpoint_path:
                checkpoint_utils.record_checkpoint_evaluation(
                    log_path=config.log_path,
                    step=last_step,
                    checkpoint_path=final_checkpoint_path,
                    metrics=metrics,
                )
            ml_logger.log_metrics(metrics=metrics, step=last_step)
    else:
        logger.info("Training was already complete; nothing to do")

    e2e_seconds = time.time() - e2e_start_time
    h, rem = divmod(int(e2e_seconds), 3600)
    m, s = divmod(rem, 60)
    logger.info(
        f"Total wall-clock time: {h}h {m}m {s}s ({e2e_seconds / 60:.1f} min)"
    )
    try:
        timing_path = os.path.join(config.log_path, "timing.json")
        with open(timing_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "e2e_wall_clock_seconds": round(e2e_seconds, 1),
                    "e2e_wall_clock_minutes": round(e2e_seconds / 60, 1),
                    "e2e_wall_clock_formatted": f"{h}h {m}m {s}s",
                },
                stream,
                indent=4,
            )
    except OSError as exc:
        logger.warning("Failed to write SFT timing artifact %s: %s", timing_path, exc)

    ml_logger.close()
    logger.info("Training completed successfully")


def _datum_token_count(datum) -> int:
    """Get total token count from a Datum's model_input."""
    return sum(_chunk_len(chunk) for chunk in datum.model_input.chunks)
