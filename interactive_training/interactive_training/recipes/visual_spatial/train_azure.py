"""Run visual-spatial SFT warmup followed by RFT with one command."""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chz

from interactive_training import cli_utils
from interactive_training.recipes.visual_spatial import train_rft_azure, train_sft_azure
from interactive_training.recipes.visual_spatial.training import TaskVariant
from interactive_training.renderers import TrainOnWhat
from interactive_training.training_types import TrainingType, training_type_field
from interactive_training.utils import file_utils
from interactive_training.utils.lr_scheduling import LRSchedule


@dataclass(frozen=True)
class CheckpointEvaluation:
    checkpoint_path: str
    step: int
    accuracy: float
    correct: int | None = None
    examples: int | None = None


@chz.chz
class CLIConfig:
    """Configuration for the staged visual-spatial SFT-to-RFT workflow."""

    model_name: str = "meta-models/Muse-Glimmer-30B"
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32
    freeze_vision_tower: bool = True
    freeze_multi_modal_projector: bool = True
    load_checkpoint_path: str | None = None

    seed: int = 0
    data_path: str | None = None
    images_per_example: int = 1
    task_variant: TaskVariant = "count_category"
    model_context_length: int | None = None
    max_train_examples: int | None = 320
    max_test_examples: int | None = None
    preflight_only: bool = False
    prompt_cache_max_mb: int = 128
    log_examples: bool = False

    sft_learning_rate: float = 3e-5
    sft_lr_schedule: LRSchedule = "linear"
    sft_batch_size: int = 8
    sft_num_epochs: int = 2
    sft_max_steps: int | None = 60
    sft_max_wall_clock_seconds: float | None = 1200
    sft_max_length: int | None = None
    train_on_what: TrainOnWhat | None = None
    sft_eval_every: int = 20
    sft_save_every: int = 20
    sft_pipeline_depth: int = 1

    rft_learning_rate: float = 3e-6
    rft_group_size: int = 16
    rft_groups_per_batch: int = 7
    rft_max_concurrent_groups: int | None = 7
    rft_num_epochs: int = 1
    rft_max_steps: int | None = 10
    rft_max_wall_clock_seconds: float | None = 1500
    rft_temperature: float = 1.0
    rft_max_tokens: int = 16
    rft_eval_max_tokens: int = 16
    rft_eval_concurrency: int = 16
    rft_eval_every: int = 5
    rft_save_every: int = 5
    rft_remove_constant_reward_groups: bool = True

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    project_endpoint: str
    verbose_http: bool = False
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _optional_int(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return int(value)


def _select_best_sft_checkpoint(log_path: str) -> CheckpointEvaluation:
    """Select the highest-accuracy SFT checkpoint with a matching evaluation."""
    root = Path(log_path)
    checkpoints_by_step: dict[int, str] = {}
    checkpoint_steps_by_path: dict[str, int] = {}
    for record in _read_jsonl(root / "checkpoints.jsonl"):
        step = _optional_int(record, "step")
        state_path = record.get("state_path")
        if step is not None and isinstance(state_path, str) and state_path:
            checkpoints_by_step[step] = state_path
            checkpoint_steps_by_path[state_path] = step

    evaluations_by_checkpoint: dict[str, CheckpointEvaluation] = {}
    checkpoint_evaluations_path = root / "checkpoint_evaluations.jsonl"
    checkpoint_evaluation_records = (
        _read_jsonl(checkpoint_evaluations_path)
        if checkpoint_evaluations_path.exists()
        else []
    )
    for record in checkpoint_evaluation_records:
        step = _optional_int(record, "step")
        checkpoint_path = record.get("checkpoint_path")
        accuracy = record.get("test/visual_spatial_accuracy")
        if (
            step is None
            or not isinstance(checkpoint_path, str)
            or not checkpoint_path
            or checkpoint_steps_by_path.get(checkpoint_path) != step
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy)
        ):
            continue
        evaluations_by_checkpoint[checkpoint_path] = CheckpointEvaluation(
            checkpoint_path=checkpoint_path,
            step=step,
            accuracy=float(accuracy),
            correct=_optional_int(record, "test/visual_spatial_correct"),
            examples=_optional_int(record, "test/visual_spatial_examples"),
        )

    legacy_evaluations_by_step: dict[int, CheckpointEvaluation] = {}
    for record in _read_jsonl(root / "metrics.jsonl"):
        step = _optional_int(record, "step")
        accuracy = record.get("test/visual_spatial_accuracy")
        if (
            step is None
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy)
        ):
            continue

        if step not in checkpoints_by_step:
            continue
        legacy_evaluations_by_step[step] = CheckpointEvaluation(
            checkpoint_path=checkpoints_by_step[step],
            step=step,
            accuracy=float(accuracy),
            correct=_optional_int(record, "test/visual_spatial_correct"),
            examples=_optional_int(record, "test/visual_spatial_examples"),
        )

    correlated_steps = {
        evaluation.step for evaluation in evaluations_by_checkpoint.values()
    }
    candidates = list(evaluations_by_checkpoint.values())
    candidates.extend(
        evaluation
        for step, evaluation in legacy_evaluations_by_step.items()
        if step not in correlated_steps
    )
    if not candidates:
        raise RuntimeError(
            "SFT completed without a held-out evaluation that matches a durable "
            f"checkpoint in {log_path}. Keep sft_eval_every and sft_save_every "
            "aligned and greater than zero."
        )
    return max(candidates, key=lambda candidate: (candidate.accuracy, -candidate.step))


def _write_staged_summary(
    log_path: str,
    selected_sft_checkpoint: CheckpointEvaluation,
    rft_accuracy_summary: dict[str, Any] | None = None,
) -> None:
    summary = {
        "schema_version": 1,
        "selected_sft_checkpoint": asdict(selected_sft_checkpoint),
        "sft_log_path": os.path.join(log_path, "sft"),
        "rft_log_path": os.path.join(log_path, "rft"),
        "rft_accuracy_summary": rft_accuracy_summary,
    }
    destination = Path(log_path) / "staged_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    os.replace(temporary, destination)


async def cli_main(cli_config: CLIConfig) -> None:
    run_name = (
        f"visual-spatial-staged-{cli_config.model_name.replace('/', '-')}"
        f"-{cli_config.task_variant}-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = cli_config.log_path or str(
        file_utils.default_logs_root() / "visual_spatial_staged" / run_name
    )
    cli_utils.check_log_dir(
        log_path,
        behavior_if_exists=cli_config.behavior_if_log_dir_exists,
    )
    os.makedirs(log_path, exist_ok=True)
    sft_log_path = os.path.join(log_path, "sft")
    rft_log_path = os.path.join(log_path, "rft")
    run_meta = {"azure": {"session_ids": {"sft": None, "rft": None}}}
    metadata_path = os.path.join(log_path, "run_meta.json")

    def write_run_meta() -> None:
        descriptor = os.open(
            metadata_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            file_utils.set_private_file_permissions(descriptor, metadata_path)
            json.dump(run_meta, output, indent=2)

    async def run_stage(stage: str, launch: Any, config: Any) -> None:
        try:
            await launch(config)
        finally:
            stage_metadata = Path(log_path) / stage / "run_meta.json"
            if stage_metadata.exists():
                data = json.loads(stage_metadata.read_text(encoding="utf-8"))
                run_meta["azure"]["session_ids"][stage] = data["azure"]["session_id"]
                write_run_meta()

    write_run_meta()

    shared = {
        "project_endpoint": cli_config.project_endpoint,
        "model_name": cli_config.model_name,
        "tokenizer_name": cli_config.tokenizer_name,
        "renderer_name": cli_config.renderer_name,
        "lora_rank": cli_config.lora_rank,
        "freeze_vision_tower": cli_config.freeze_vision_tower,
        "freeze_multi_modal_projector": cli_config.freeze_multi_modal_projector,
        "seed": cli_config.seed,
        "data_path": cli_config.data_path,
        "images_per_example": cli_config.images_per_example,
        "task_variant": cli_config.task_variant,
        "model_context_length": cli_config.model_context_length,
        "max_train_examples": cli_config.max_train_examples,
        "max_test_examples": cli_config.max_test_examples,
        "prompt_cache_max_mb": cli_config.prompt_cache_max_mb,
        "log_examples": cli_config.log_examples,
        "verbose_http": cli_config.verbose_http,
        "training_type": cli_config.training_type,
        "user_metadata": cli_config.user_metadata,
    }

    print("Stage 1/2: supervised warmup")
    await run_stage("sft", train_sft_azure.cli_main,
        train_sft_azure.CLIConfig(
            **shared,
            preflight_only=cli_config.preflight_only,
            load_checkpoint_path=cli_config.load_checkpoint_path,
            learning_rate=cli_config.sft_learning_rate,
            lr_schedule=cli_config.sft_lr_schedule,
            batch_size=cli_config.sft_batch_size,
            num_epochs=cli_config.sft_num_epochs,
            max_steps=cli_config.sft_max_steps,
            max_wall_clock_seconds=cli_config.sft_max_wall_clock_seconds,
            max_length=cli_config.sft_max_length,
            train_on_what=cli_config.train_on_what,
            eval_max_tokens=cli_config.rft_eval_max_tokens,
            eval_concurrency=cli_config.rft_eval_concurrency,
            eval_every=cli_config.sft_eval_every,
            save_every=cli_config.sft_save_every,
            pipeline_depth=cli_config.sft_pipeline_depth,
            log_path=sft_log_path,
            wandb_project=cli_config.wandb_project,
            wandb_name=(
                f"{cli_config.wandb_name}-sft" if cli_config.wandb_name else None
            ),
            behavior_if_log_dir_exists=(
                "resume" if os.path.exists(sft_log_path) else "raise"
            ),
        ),
    )

    if cli_config.preflight_only:
        print("Stage 2/2: reinforcement fine-tuning preflight")
        await run_stage("rft", train_rft_azure.cli_main,
            train_rft_azure.CLIConfig(
                **shared,
                preflight_only=True,
                load_checkpoint_path=None,
                learning_rate=cli_config.rft_learning_rate,
                group_size=cli_config.rft_group_size,
                groups_per_batch=cli_config.rft_groups_per_batch,
                max_concurrent_groups=cli_config.rft_max_concurrent_groups,
                num_epochs=cli_config.rft_num_epochs,
                max_steps=cli_config.rft_max_steps,
                max_wall_clock_seconds=cli_config.rft_max_wall_clock_seconds,
                temperature=cli_config.rft_temperature,
                max_tokens=cli_config.rft_max_tokens,
                eval_max_tokens=cli_config.rft_eval_max_tokens,
                eval_concurrency=cli_config.rft_eval_concurrency,
                eval_every=cli_config.rft_eval_every,
                save_every=cli_config.rft_save_every,
                remove_constant_reward_groups=(
                    cli_config.rft_remove_constant_reward_groups
                ),
                log_path=rft_log_path,
                wandb_project=cli_config.wandb_project,
                wandb_name=(
                    f"{cli_config.wandb_name}-rft" if cli_config.wandb_name else None
                ),
                behavior_if_log_dir_exists=(
                    "resume" if os.path.exists(rft_log_path) else "raise"
                ),
            ),
        )
        print("Staged preflight completed; no training session was created.")
        return

    selected_checkpoint = _select_best_sft_checkpoint(sft_log_path)
    print(
        "Selected SFT checkpoint: "
        f"step={selected_checkpoint.step} "
        f"accuracy={selected_checkpoint.accuracy:.1%} "
        f"path={selected_checkpoint.checkpoint_path}"
    )
    _write_staged_summary(log_path, selected_checkpoint)

    print("Stage 2/2: reinforcement fine-tuning")
    await run_stage("rft", train_rft_azure.cli_main,
        train_rft_azure.CLIConfig(
            **shared,
            preflight_only=False,
            load_checkpoint_path=selected_checkpoint.checkpoint_path,
            learning_rate=cli_config.rft_learning_rate,
            group_size=cli_config.rft_group_size,
            groups_per_batch=cli_config.rft_groups_per_batch,
            max_concurrent_groups=cli_config.rft_max_concurrent_groups,
            num_epochs=cli_config.rft_num_epochs,
            max_steps=cli_config.rft_max_steps,
            max_wall_clock_seconds=cli_config.rft_max_wall_clock_seconds,
            temperature=cli_config.rft_temperature,
            max_tokens=cli_config.rft_max_tokens,
            eval_max_tokens=cli_config.rft_eval_max_tokens,
            eval_concurrency=cli_config.rft_eval_concurrency,
            eval_every=cli_config.rft_eval_every,
            save_every=cli_config.rft_save_every,
            remove_constant_reward_groups=cli_config.rft_remove_constant_reward_groups,
            log_path=rft_log_path,
            wandb_project=cli_config.wandb_project,
            wandb_name=(
                f"{cli_config.wandb_name}-rft" if cli_config.wandb_name else None
            ),
            behavior_if_log_dir_exists=(
                "resume" if os.path.exists(rft_log_path) else "raise"
            ),
        ),
    )
    rft_summary_path = Path(rft_log_path) / "visual_spatial_accuracy_summary.json"
    rft_accuracy_summary = (
        json.loads(rft_summary_path.read_text(encoding="utf-8"))
        if rft_summary_path.exists()
        else None
    )
    _write_staged_summary(log_path, selected_checkpoint, rft_accuracy_summary)
    print(f"Staged summary: {Path(log_path) / 'staged_summary.json'}")


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(config)))