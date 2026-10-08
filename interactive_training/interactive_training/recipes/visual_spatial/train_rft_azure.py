"""Visual-spatial RL training with Azure AI Fine-Tuning Sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import chz
from azure.ai.finetuningsessions.models import LossFn as LossFnType

from interactive_training import checkpoint_utils, cli_utils, model_info
from interactive_training.recipes.visual_spatial.training import (
    TaskVariant,
    VisualSpatialAccuracyEvaluator,
    VisualSpatialRLDatasetBuilder,
)
from interactive_training.rl import train_azure as rl_azure
from interactive_training.rl.train import AsyncConfig, Config, EvaluationStrategy
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils

logger = logging.getLogger(__name__)

_ACCURACY_METRIC = "test/visual_spatial_accuracy"
_ACCURACY_SUMMARY_FILE = "visual_spatial_accuracy_summary.json"


def _accuracy_observation(record: Any) -> dict[str, int | float] | None:
    if not isinstance(record, dict):
        return None

    accuracy = record.get(_ACCURACY_METRIC)
    step = record.get("step")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
        or isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(step)
    ):
        return None

    observation: dict[str, int | float] = {
        "step": int(step),
        "accuracy": float(accuracy),
    }
    for source, destination in (
        ("test/visual_spatial_correct", "correct"),
        ("test/visual_spatial_examples", "examples"),
    ):
        value = record.get(source)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            observation[destination] = int(value)
    return observation


def _write_accuracy_summary(log_path: str) -> dict[str, Any] | None:
    """Persist and print the first-to-final visual-spatial accuracy change."""
    metrics_path = os.path.join(log_path, "metrics.jsonl")
    observations: list[dict[str, int | float]] = []
    try:
        with open(metrics_path, encoding="utf-8") as metrics_file:
            for line_number, line in enumerate(metrics_file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Ignoring malformed metrics line %d in %s",
                        line_number,
                        metrics_path,
                    )
                    continue
                observation = _accuracy_observation(record)
                if observation is not None:
                    observations.append(observation)
    except FileNotFoundError:
        observations = []

    if len(observations) < 2:
        print(
            "Visual-spatial accuracy comparison unavailable: baseline and "
            "post-training evaluations did not both complete. Ensure eval_every > 0."
        )
        return None

    before_training = observations[0]
    after_training = observations[-1]
    improvement = after_training["accuracy"] - before_training["accuracy"]
    baseline_accuracy = before_training["accuracy"]
    summary = {
        "schema_version": 1,
        "metric": _ACCURACY_METRIC,
        "before_training": before_training,
        "after_training": after_training,
        "absolute_improvement": improvement,
        "percentage_point_improvement": improvement * 100,
        "relative_improvement": (
            improvement / baseline_accuracy if baseline_accuracy != 0 else None
        ),
        "evaluations_recorded": len(observations),
    }

    os.makedirs(log_path, exist_ok=True)
    summary_path = os.path.join(log_path, _ACCURACY_SUMMARY_FILE)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{_ACCURACY_SUMMARY_FILE}.",
        suffix=".tmp",
        dir=log_path,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as summary_file:
            file_utils.set_private_file_permissions(descriptor, temporary_path)
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")
        os.replace(temporary_path, summary_path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    def format_result(observation: dict[str, int | float]) -> str:
        result = f"{observation['accuracy']:.1%}"
        if "correct" in observation and "examples" in observation:
            result += f" ({observation['correct']}/{observation['examples']})"
        return result

    print("Visual-spatial accuracy comparison:")
    print(f"  Before training: {format_result(before_training)}")
    print(f"  After training:  {format_result(after_training)}")
    print(f"  Improvement:     {improvement * 100:+.1f} percentage points")
    print(f"  Summary:         {summary_path}")
    return summary


@chz.chz
class CLIConfig:
    """Configuration for visual-spatial reinforcement learning."""

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
    task_variant: TaskVariant = "single_image"
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    num_epochs: int = 1
    model_context_length: int | None = None
    preflight_only: bool = False
    prompt_cache_max_mb: int = 128
    log_examples: bool = False

    group_size: int = 5
    groups_per_batch: int = 3
    max_concurrent_groups: int | None = None
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    compute_post_kl: bool = False
    remove_constant_reward_groups: bool = False
    max_oversample_rounds: int = 10

    max_steps_off_policy: int | None = None
    max_steps: int | None = None
    max_wall_clock_seconds: float | None = None

    eval_every: int = 20
    eval_strategy: EvaluationStrategy = "steps"
    eval_max_tokens: int = 512
    eval_concurrency: int = 6
    save_every: int = 20

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    project_endpoint: str
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0
    initial_idle_seconds: float = 0.0
    existing_session_id: str | None = None
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    if "://" in checkpoint_path:
        stripped = checkpoint_path.split("://", 1)[1]
        parts = stripped.split("/")
        if len(parts) >= 2:
            session_id = parts[0]
            if not session_id.startswith("session_"):
                session_id = f"session_{session_id.removeprefix('model_')}"
            return session_id, parts[-1]

    parts = checkpoint_path.split("/", 1)
    if len(parts) == 2 and all(parts):
        session_id = parts[0]
        if not session_id.startswith("session_"):
            session_id = f"session_{session_id.removeprefix('model_')}"
        return session_id, parts[1]

    raise ValueError(
        f"Invalid checkpoint_path={checkpoint_path!r}. Expected "
        "'<session_id>/<checkpoint_name>' or an SDK checkpoint URI."
    )


def _validate_cli_config(cli_config: CLIConfig) -> None:
    supported_task_variants = {
        "single_image",
        "absolute_difference",
        "count_category",
    }
    if cli_config.task_variant not in supported_task_variants:
        raise ValueError(
            f"task_variant must be one of {sorted(supported_task_variants)}, "
            f"got {cli_config.task_variant!r}"
        )
    positive_values = {
        "lora_rank": cli_config.lora_rank,
        "images_per_example": cli_config.images_per_example,
        "group_size": cli_config.group_size,
        "groups_per_batch": cli_config.groups_per_batch,
        "num_epochs": cli_config.num_epochs,
        "max_tokens": cli_config.max_tokens,
        "eval_max_tokens": cli_config.eval_max_tokens,
        "eval_concurrency": cli_config.eval_concurrency,
    }
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if cli_config.images_per_example > 64:
        raise ValueError("images_per_example must be at most 64")
    if cli_config.task_variant == "absolute_difference":
        if cli_config.images_per_example != 2:
            raise ValueError("absolute_difference requires images_per_example=2")
        if cli_config.data_path is not None:
            raise ValueError("absolute_difference does not support data_path")
    if cli_config.task_variant == "count_category":
        if cli_config.images_per_example != 1:
            raise ValueError("count_category requires images_per_example=1")
        if cli_config.data_path is not None:
            raise ValueError("count_category does not support data_path")
    for name, value in {
        "max_train_examples": cli_config.max_train_examples,
        "max_test_examples": cli_config.max_test_examples,
        "max_concurrent_groups": cli_config.max_concurrent_groups,
    }.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1 when set, got {value}")
    if cli_config.prompt_cache_max_mb < 0:
        raise ValueError(
            "prompt_cache_max_mb must not be negative, got "
            f"{cli_config.prompt_cache_max_mb}"
        )


async def cli_main(cli_config: CLIConfig) -> None:
    _validate_cli_config(cli_config)
    model_info.require_vision_language_model(
        cli_config.model_name,
        workload="visual-spatial reinforcement learning",
    )

    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )
    model_context_length = (
        cli_config.model_context_length
        or model_info.get_model_context_length(cli_config.model_name)
    )
    if model_context_length is None:
        raise ValueError(
            "Visual-spatial preflight requires model_context_length. Set it to "
            "the context length supported by the selected model."
        )

    run_name = (
        f"visual-spatial-{cli_config.model_name.replace('/', '-')}"
        f"-{cli_config.task_variant}"
        f"-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group"
        f"-{cli_config.groups_per_batch}batch"
        f"-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = cli_config.log_path or str(
        file_utils.default_logs_root() / "visual_spatial" / run_name
    )
    wandb_name = cli_config.wandb_name or run_name
    image_cache_dir = (
        os.path.join(log_path, "image_cache")
        if cli_config.data_path is not None
        else None
    )
    log_dir_prepared = False
    if image_cache_dir is not None:
        cli_utils.check_log_dir(
            log_path,
            behavior_if_exists=cli_config.behavior_if_log_dir_exists,
        )
        os.makedirs(log_path, exist_ok=True)
        log_dir_prepared = True

    dataset_builder = VisualSpatialRLDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name=cli_config.model_name,
        tokenizer_name=tokenizer_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
        max_train_examples=cli_config.max_train_examples,
        max_test_examples=cli_config.max_test_examples,
        images_per_example=cli_config.images_per_example,
        num_epochs=cli_config.num_epochs,
        data_path=cli_config.data_path,
        prompt_cache_max_mb=cli_config.prompt_cache_max_mb,
        image_cache_dir=image_cache_dir,
        log_examples=cli_config.log_examples,
        task_variant=cli_config.task_variant,
        max_tokens=cli_config.max_tokens,
    )
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=tokenizer_name,
        tokenizer_name=tokenizer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_strategy=cli_config.eval_strategy,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        evaluator_builders=[
            lambda: VisualSpatialAccuracyEvaluator(
                model_name=cli_config.model_name,
                renderer_name=renderer_name,
                seed=cli_config.seed,
                output_dir=log_path,
                max_examples=cli_config.max_test_examples,
                images_per_example=cli_config.images_per_example,
                max_tokens=cli_config.eval_max_tokens,
                concurrency=cli_config.eval_concurrency,
                data_path=cli_config.data_path,
                prompt_cache_max_mb=cli_config.prompt_cache_max_mb,
                image_cache_dir=image_cache_dir,
                log_examples=cli_config.log_examples,
                task_variant=cli_config.task_variant,
            )
        ],
        compute_post_kl=cli_config.compute_post_kl,
        async_config=(
            AsyncConfig(
                max_steps_off_policy=cli_config.max_steps_off_policy,
                groups_per_batch=cli_config.groups_per_batch,
            )
            if cli_config.max_steps_off_policy is not None
            else None
        ),
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        max_oversample_rounds=cli_config.max_oversample_rounds,
        max_concurrent_groups=cli_config.max_concurrent_groups,
        max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        sampling_seed=cli_config.seed,
    )

    print(
        "Validating visual-spatial prompts before creating a training session "
        f"(context length: {model_context_length} tokens)..."
    )
    prepared_datasets = await dataset_builder()
    reserved_generation_tokens = max(cli_config.max_tokens, cli_config.eval_max_tokens)
    for split in prepared_datasets:
        summary = await split.preflight(
            max_tokens=reserved_generation_tokens,
            model_context_length=model_context_length,
        )
        print(
            f"Preflight passed: split={summary.split} examples={summary.examples} "
            f"max_prompt_tokens={summary.max_prompt_tokens} "
            f"max_total_tokens={summary.max_total_tokens}/{model_context_length}"
        )
    if cli_config.preflight_only:
        print("Preflight completed; no training session was created.")
        return

    if not log_dir_prepared:
        cli_utils.check_log_dir(
            log_path,
            behavior_if_exists=cli_config.behavior_if_log_dir_exists,
        )
        os.makedirs(log_path, exist_ok=True)

    parsed_endpoint = urlparse(cli_config.project_endpoint or "")
    endpoint_host = parsed_endpoint.hostname or ""
    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "visual_spatial_rl_azure_sdk",
        "azure": {
            "endpoint_host": endpoint_host,
            "auth": "AzureKeyCredential" if os.environ.get("AZURE_AI_API_KEY") else "DefaultAzureCredential",
            "session_id": None,
            "from_checkpoint": None,
        },
        "model": {
            "model_name": cli_config.model_name,
            "tokenizer_name": tokenizer_name,
            "renderer_name": renderer_name,
            "lora_rank": cli_config.lora_rank,
            "freeze_vision_tower": cli_config.freeze_vision_tower,
            "freeze_multi_modal_projector": cli_config.freeze_multi_modal_projector,
        },
        "dataset": {
            "name": (
                "azure_openai_conversations_jsonl"
                if cli_config.data_path is not None
                else "microsoft/Do-You-See-Me:visual_spatial"
            ),
            "images_per_example": cli_config.images_per_example,
            "task_variant": cli_config.task_variant,
            "max_train_examples": cli_config.max_train_examples,
            "max_test_examples": cli_config.max_test_examples,
            "group_size": cli_config.group_size,
            "groups_per_batch": cli_config.groups_per_batch,
            "num_epochs": cli_config.num_epochs,
            "prompt_cache_max_mb": cli_config.prompt_cache_max_mb,
            "log_examples": cli_config.log_examples,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "temperature": cli_config.temperature,
            "max_tokens": cli_config.max_tokens,
            "loss_fn": str(cli_config.loss_fn),
            "seed": cli_config.seed,
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
            "max_steps": cli_config.max_steps,
            "max_wall_clock_seconds": cli_config.max_wall_clock_seconds,
            "max_steps_off_policy": cli_config.max_steps_off_policy,
        },
    }

    def write_run_meta() -> None:
        meta_path = os.path.join(log_path, "run_meta.json")
        meta_descriptor = os.open(
            meta_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(meta_descriptor, "w", encoding="utf-8") as file:
            file_utils.set_private_file_permissions(meta_descriptor, meta_path)
            json.dump(run_meta, file, indent=2, default=str)

    write_run_meta()

    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key)
    else:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()

    if cli_config.verbose_http:
        import azure.ai.finetuningsessions._patch as patch_module

        patch_module.VERBOSE_HTTP = True
        logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(logging.INFO)

    client_kwargs: dict[str, Any] = {
        "endpoint": cli_config.project_endpoint,
        "credential": credential,
    }
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    client = FineTuningSessionClient(**client_kwargs)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    selection = checkpoint_utils.select_resume_checkpoint(
        resume_info,
        cli_config.load_checkpoint_path,
    )
    checkpoint_path = selection.checkpoint_path
    if selection.ignored_load_checkpoint_path:
        print(
            f"Ignoring load_checkpoint_path={selection.ignored_load_checkpoint_path}: "
            f"resuming from the checkpoint ledger in {log_path}. Use a fresh log_path "
            "to start from an explicit checkpoint."
        )
    if selection.is_resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    elif checkpoint_path is not None:
        print(f"Starting from checkpoint: {checkpoint_path}")

    from_checkpoint: FromCheckpoint | None = None
    if checkpoint_path is not None:
        source_session_id, checkpoint_id = _parse_checkpoint_path(checkpoint_path)
        from_checkpoint = FromCheckpoint(
            source_session_id=source_session_id,
            checkpoint_id=checkpoint_id,
        )
        run_meta["azure"]["from_checkpoint"] = {
            "source_session_id": source_session_id,
            "checkpoint_id": checkpoint_id,
        }
        write_run_meta()

    try:
        if cli_config.existing_session_id:
            session_id = cli_config.existing_session_id
            print(f"Attaching to existing session: {session_id}")
        else:
            print(
                f"Creating session: base_model={cli_config.model_name} "
                f"lora_rank={cli_config.lora_rank} "
                f"endpoint={cli_config.project_endpoint}"
            )
            session_id = await client.create_session(
                base_model=cli_config.model_name,
                lora_config=LoRAConfig(
                    rank=cli_config.lora_rank,
                    freeze_vision_tower=cli_config.freeze_vision_tower,
                    freeze_multi_modal_projector=(
                        cli_config.freeze_multi_modal_projector
                    ),
                ),
                type="training",
                from_checkpoint=from_checkpoint,
                timeout_sec=cli_config.create_session_timeout_sec,
                user_metadata=cli_config.user_metadata,
                training_type=normalize_training_type(cli_config.training_type),
            )
        print(f"Session ready: session_id={session_id}")
        run_meta["azure"]["session_id"] = session_id
        write_run_meta()

        if cli_config.initial_idle_seconds > 0:
            await asyncio.sleep(cli_config.initial_idle_seconds)

        await rl_azure.main(
            config,
            client,
            session_id,
            prepared_datasets=prepared_datasets,
            training_type=cli_config.training_type,
        )
        try:
            _write_accuracy_summary(log_path)
        except Exception:
            logger.exception(
                "Training completed, but the visual-spatial accuracy summary "
                "could not be generated"
            )
            print(
                "Training completed, but the visual-spatial accuracy comparison "
                "could not be generated. See the logs for details."
            )
    finally:
        await client.close()
        if hasattr(credential, "close"):
            await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))
