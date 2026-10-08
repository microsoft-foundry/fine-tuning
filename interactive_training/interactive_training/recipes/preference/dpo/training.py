"""Direct Preference Optimization training on Interactive Training."""

import asyncio
import logging
import os
import time
from pathlib import Path

import chz
import torch
import torch.nn.functional as F
from azure.ai.finetuningsessions.models import (
    AdamParams,
    Datum,
    ModelInput,
    ModelInputChunk,
)

from interactive_training import checkpoint_utils
from interactive_training.supervised.nll_evaluator import NLLEvaluator
from interactive_training.supervised.types import ChatDatasetBuilder
from interactive_training.utils import ml_log
from interactive_training.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    log_path: str = chz.field(munger=lambda _, value: str(Path(value).expanduser()))
    model_name: str
    dataset_builder: ChatDatasetBuilder
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    max_steps: int | None = None
    dpo_beta: float = 0.1
    save_every: int = 20
    eval_every: int = 10
    reference_concurrency: int = 32
    ttl_seconds: int | None = 604800
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    wandb_project: str | None = None
    wandb_name: str | None = None

    @chz.validate
    def _validate(self):
        if self.dpo_beta <= 0 or self.learning_rate <= 0 or self.num_epochs <= 0:
            raise ValueError("dpo_beta, learning_rate, and num_epochs must be positive")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be non-negative or None")
        if self.eval_every < 0 or self.save_every < 0:
            raise ValueError("eval_every and save_every must be non-negative")
        if self.reference_concurrency <= 0:
            raise ValueError("reference_concurrency must be positive")


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_reference_logprobs: list[torch.Tensor],
    rejected_reference_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_ratio = torch.stack(
        [
            policy - reference
            for policy, reference in zip(
                chosen_logprobs, chosen_reference_logprobs, strict=True
            )
        ]
    )
    rejected_ratio = torch.stack(
        [
            policy - reference
            for policy, reference in zip(
                rejected_logprobs, rejected_reference_logprobs, strict=True
            )
        ]
    )
    losses = -F.logsigmoid(dpo_beta * (chosen_ratio - rejected_ratio))
    loss = losses.mean()
    chosen_rewards = dpo_beta * chosen_ratio
    rejected_rewards = dpo_beta * rejected_ratio
    return loss, {
        "dpo_loss": loss.item(),
        "accuracy": (chosen_ratio > rejected_ratio).float().mean().item(),
        "margin": (chosen_rewards - rejected_rewards).mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


def _full_sequence(datum: Datum) -> ModelInput:
    tokens: list[int] = []
    for chunk in datum.model_input.chunks:
        if not isinstance(chunk, ModelInputChunk):
            raise TypeError("DPO currently supports text-only examples")
        tokens.extend(chunk.tokens)
    targets = datum.loss_fn_inputs["target_tokens"].data
    if targets:
        tokens.append(int(targets[-1]))
    return ModelInput(chunks=[ModelInputChunk(tokens=tokens)])


def build_dpo_loss_fn(
    data: list[Datum],
    reference_logprobs: list[torch.Tensor],
    dpo_beta: float,
):
    if not data or len(data) % 2:
        raise ValueError("DPO batches must contain chosen/rejected pairs")
    if len(reference_logprobs) != len(data):
        raise ValueError("Reference logprobs must align one-to-one with the DPO batch")

    def dpo_loss_fn(
        loss_data: list[Datum], policy_logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if loss_data is not data:
            raise ValueError(
                "DPO loss received a different batch than it was built for"
            )
        weighted_policy: list[torch.Tensor] = []
        weighted_reference: list[torch.Tensor] = []
        for datum, policy, reference in zip(
            data, policy_logprobs, reference_logprobs, strict=True
        ):
            weights = torch.tensor(
                datum.loss_fn_inputs["weights"].data,
                dtype=policy.dtype,
                device=policy.device,
            )
            reference = reference.to(dtype=policy.dtype, device=policy.device)
            if (
                policy.numel() != weights.numel()
                or reference.numel() != weights.numel()
            ):
                raise ValueError("Policy, reference, and weight lengths must match")
            weighted_policy.append(torch.dot(policy, weights))
            weighted_reference.append(torch.dot(reference, weights))
        return compute_dpo_loss(
            weighted_policy[0::2],
            weighted_policy[1::2],
            weighted_reference[0::2],
            weighted_reference[1::2],
            dpo_beta,
        )

    return dpo_loss_fn


async def _reference_logprobs(
    reference_client, data: list[Datum], concurrency: int = 32
) -> list[torch.Tensor]:
    if concurrency <= 0:
        raise ValueError("reference_concurrency must be positive")
    results = []
    for offset in range(0, len(data), concurrency):
        results.extend(
            await asyncio.gather(
                *(
                    reference_client.compute_logprobs_async(_full_sequence(datum))
                    for datum in data[offset : offset + concurrency]
                )
            )
        )
    aligned: list[torch.Tensor] = []
    for datum, result in zip(data, results, strict=True):
        values = result[1:]
        expected = len(datum.loss_fn_inputs["target_tokens"].data)
        if len(values) != expected or any(value is None for value in values):
            raise ValueError(
                "Reference prompt logprobs did not align with the datum targets"
            )
        aligned.append(torch.tensor(values, dtype=torch.float32))
    return aligned


async def evaluate_preferences(
    training_client, data, reference_logprobs, dpo_beta, prefix
):
    future = await training_client.forward_async(data, loss_fn="cross_entropy")
    result = await future.result_async()
    policy_logprobs = [
        torch.tensor(entry["logprobs"].data, dtype=torch.float32)
        for entry in result.loss_fn_outputs
    ]
    with torch.no_grad():
        _, metrics = build_dpo_loss_fn(data, reference_logprobs, dpo_beta)(
            data, policy_logprobs
        )
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


async def main(
    config: Config,
    training_client,
    reference_client,
    reference_state_path: str,
    prepared_datasets=None,
) -> None:
    dataset, test_dataset = prepared_datasets or config.dataset_builder.build()
    n_batches = len(dataset)
    full_steps = n_batches * config.num_epochs
    total_steps = min(full_steps, config.max_steps) if config.max_steps else full_steps
    if total_steps <= 0:
        raise ValueError("DPO dataset produced no full batches")
    evaluator = None
    if config.eval_every > 0 and test_dataset is not None:
        evaluator = NLLEvaluator.from_dataset(test_dataset)
        if not evaluator.data:
            evaluator = None
    os.makedirs(config.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
    )
    resume = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_epoch = int(resume.get("epoch", 0)) if resume else 0
    start_batch = int(resume.get("batch", 0)) if resume else 0
    completed_updates = (
        int(resume.get("completed_updates", start_epoch * n_batches + start_batch))
        if resume
        else 0
    )
    if resume and "completed_updates" not in resume:
        logger.warning(
            "Legacy checkpoint has no optimizer-update count; assuming all prior "
            "batches were trainable. The count may be too high if batches were skipped."
        )
    start_updates = completed_updates
    next_epoch, next_batch = start_epoch, start_batch

    try:
        probes = []
        if config.eval_every > 0:
            probe_data = dataset.get_batch(0)
            if probe_data:
                probes.append(("train_probe", probe_data))
            if evaluator is not None:
                probes.append(("test", evaluator.data))
        scored_probes = [
            (
                prefix,
                data,
                await _reference_logprobs(
                    reference_client, data, config.reference_concurrency
                ),
            )
            for prefix, data in probes
        ]

        async def evaluate():
            metrics = await evaluator(training_client) if evaluator is not None else {}
            for prefix, data, reference in scored_probes:
                metrics.update(
                    await evaluate_preferences(
                        training_client, data, reference, config.dpo_beta, prefix
                    )
                )
            return metrics

        if scored_probes:
            ml_logger.log_metrics(
                {"evaluation_phase": "initial", **await evaluate()},
                step=completed_updates,
            )
        stop = False
        for epoch in range(start_epoch, config.num_epochs):
            dataset.set_epoch(epoch)
            first_batch = start_batch if epoch == start_epoch else 0
            for batch_index in range(first_batch, n_batches):
                step = completed_updates
                if step >= total_steps:
                    stop = True
                    break
                batch_start = time.monotonic()
                data = dataset.get_batch(batch_index)
                if len(data) % 2:
                    raise ValueError("DPO batches must contain chosen/rejected pairs")
                if not data:
                    logger.warning(
                        "Skipping empty DPO batch at epoch %d, batch %d",
                        epoch,
                        batch_index,
                    )
                    next_epoch, next_batch = divmod(
                        epoch * n_batches + batch_index + 1, n_batches
                    )
                    continue

                if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                    await checkpoint_utils.save_checkpoint_async(
                        training_client,
                        name=f"step_{step:06d}",
                        log_path=config.log_path,
                        loop_state={
                            "epoch": epoch,
                            "batch": batch_index,
                            "completed_updates": completed_updates,
                            "reference_state_path": reference_state_path,
                        },
                        ttl_seconds=config.ttl_seconds,
                    )

                eval_metrics = {}
                if (
                    scored_probes
                    and step % config.eval_every == 0
                    and step != start_updates
                ):
                    eval_metrics = await evaluate()
                reference_logprobs = await _reference_logprobs(
                    reference_client, data, config.reference_concurrency
                )
                loss_fn = build_dpo_loss_fn(data, reference_logprobs, config.dpo_beta)
                learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
                    config.lr_schedule, step, total_steps
                )
                adam = AdamParams(
                    learning_rate=learning_rate,
                    beta1=config.adam_beta1,
                    beta2=config.adam_beta2,
                    eps=config.adam_eps,
                )
                forward = await training_client.forward_backward_custom_async(
                    data, loss_fn
                )
                optim = await training_client.optim_step_async(adam)
                forward_result, optim_result = await asyncio.gather(
                    forward.result_async(), optim.result_async()
                )
                completed_updates += 1
                next_epoch, next_batch = divmod(
                    epoch * n_batches + batch_index + 1, n_batches
                )
                metrics = {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "num_pairs": len(data) // 2,
                    "num_tokens": sum(
                        len(datum.loss_fn_inputs["target_tokens"].data)
                        for datum in data
                    ),
                    "progress": (step + 1) / total_steps,
                    "batch_time": time.monotonic() - batch_start,
                    **eval_metrics,
                    **forward_result.metrics,
                    **optim_result.metrics,
                }
                ml_logger.log_metrics(metrics, step=step)
            if stop:
                break

        if completed_updates > 0 and (next_epoch, next_batch) != (
            start_epoch,
            start_batch,
        ):
            await checkpoint_utils.save_checkpoint_async(
                training_client,
                name="final",
                log_path=config.log_path,
                loop_state={
                    "epoch": next_epoch,
                    "batch": next_batch,
                    "completed_updates": completed_updates,
                    "reference_state_path": reference_state_path,
                },
                kind="both",
                ttl_seconds=config.ttl_seconds,
                step_number=completed_updates - 1,
            )
            if scored_probes:
                ml_logger.log_metrics(
                    {"evaluation_phase": "final", **await evaluate()},
                    step=completed_updates - 1,
                )
        elif completed_updates == 0:
            raise ValueError("No valid preference pairs were available for training")
    finally:
        ml_logger.close()
