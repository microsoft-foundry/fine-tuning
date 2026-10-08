"""Visual-spatial supervised fine-tuning with Azure Fine-Tuning Sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import chz

from interactive_training import checkpoint_utils, cli_utils, model_info
from interactive_training.recipes.visual_spatial.train_rft_azure import (
    _parse_checkpoint_path,
    _write_accuracy_summary,
)
from interactive_training.recipes.visual_spatial.training import (
    TaskVariant,
    VisualSpatialAccuracyEvaluator,
    VisualSpatialSftBuilder,
)
from interactive_training.renderers import TrainOnWhat
from interactive_training.rl.train_azure import AzureSDKTrainingClient
from interactive_training.supervised import train
from interactive_training.supervised.types import ChatDatasetBuilderCommonConfig
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils
from interactive_training.utils.lr_scheduling import LRSchedule

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Configuration for visual-spatial supervised fine-tuning."""

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
    batch_size: int = 8
    num_epochs: int = 1
    max_steps: int | None = None
    max_wall_clock_seconds: float | None = None
    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "constant"

    model_context_length: int | None = None
    max_length: int | None = None
    train_on_what: TrainOnWhat | None = None
    eval_max_tokens: int = 512
    eval_concurrency: int = 6
    prompt_cache_max_mb: int = 128
    log_examples: bool = False
    preflight_only: bool = False

    eval_every: int = 20
    save_every: int = 20
    pipeline_depth: int = 1

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
    for name, value in {
        "lora_rank": cli_config.lora_rank,
        "images_per_example": cli_config.images_per_example,
        "batch_size": cli_config.batch_size,
        "num_epochs": cli_config.num_epochs,
        "eval_max_tokens": cli_config.eval_max_tokens,
        "eval_concurrency": cli_config.eval_concurrency,
        "eval_every": cli_config.eval_every,
        "save_every": cli_config.save_every,
        "pipeline_depth": cli_config.pipeline_depth,
    }.items():
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
        "max_steps": cli_config.max_steps,
    }.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1 when set, got {value}")
    if cli_config.prompt_cache_max_mb < 0:
        raise ValueError(
            "prompt_cache_max_mb must not be negative, got "
            f"{cli_config.prompt_cache_max_mb}"
        )


def _write_run_meta(path: str, metadata: dict[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as metadata_file:
        file_utils.set_private_file_permissions(descriptor, path)
        json.dump(metadata, metadata_file, indent=2, default=str)
        metadata_file.write("\n")


async def cli_main(cli_config: CLIConfig) -> None:
    _validate_cli_config(cli_config)
    model_info.require_vision_language_model(
        cli_config.model_name,
        workload="visual-spatial supervised fine-tuning",
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
    max_length = cli_config.max_length or model_context_length
    if max_length > model_context_length:
        raise ValueError(
            f"max_length={max_length} exceeds model_context_length="
            f"{model_context_length}"
        )

    run_name = (
        f"visual-spatial-sft-{cli_config.model_name.replace('/', '-')}"
        f"-{cli_config.task_variant}"
        f"-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.batch_size}batch"
        f"-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = cli_config.log_path or str(
        file_utils.default_logs_root() / "visual_spatial_sft" / run_name
    )
    cli_utils.check_log_dir(
        log_path,
        behavior_if_exists=cli_config.behavior_if_log_dir_exists,
    )
    os.makedirs(log_path, exist_ok=True)
    image_cache_dir = (
        os.path.join(log_path, "image_cache")
        if cli_config.data_path is not None
        else None
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=cli_config.batch_size,
        train_on_what=cli_config.train_on_what,
        model_context_length=model_context_length,
        fail_on_truncation=True,
    )
    dataset_builder = VisualSpatialSftBuilder(
        common_config=common_config,
        max_train_examples=cli_config.max_train_examples,
        seed=cli_config.seed,
        images_per_example=cli_config.images_per_example,
        data_path=cli_config.data_path,
        image_cache_dir=image_cache_dir,
        task_variant=cli_config.task_variant,
    )
    evaluator = VisualSpatialAccuracyEvaluator(
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
    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=[lambda: evaluator],
        infrequent_evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        seed=cli_config.seed,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        pipeline_depth=cli_config.pipeline_depth,
        preflight_dataset=True,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
    )

    print("Validating every training example before creating a session...")
    prepared_datasets = train.prepare_datasets(config)
    evaluation_summary = evaluator.preflight(
        model_context_length=model_context_length
    )
    print(
        "Preflight passed: "
        f"train_batches={len(prepared_datasets[0])} "
        f"evaluation_examples={evaluation_summary.examples} "
        f"evaluation_max_total_tokens={evaluation_summary.max_total_tokens}/"
        f"{model_context_length}"
    )
    if cli_config.preflight_only:
        print("Preflight completed; no training session was created.")
        return

    parsed_endpoint = urlparse(cli_config.project_endpoint or "")
    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "visual_spatial_sft_azure_sdk",
        "azure": {
            "endpoint_host": parsed_endpoint.hostname or "",
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
            "batch_size": cli_config.batch_size,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "lr_schedule": cli_config.lr_schedule,
            "num_epochs": cli_config.num_epochs,
            "max_steps": cli_config.max_steps,
            "seed": cli_config.seed,
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
        },
    }
    metadata_path = os.path.join(log_path, "run_meta.json")
    _write_run_meta(metadata_path, run_meta)

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
        logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(
            logging.INFO
        )

    client_kwargs: dict[str, Any] = {
        "endpoint": cli_config.project_endpoint,
        "credential": credential,
    }
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    if cli_config.project_endpoint.startswith("http://"):
        client_kwargs["allow_insecure_http"] = True
    client = FineTuningSessionClient(**client_kwargs)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    selection = checkpoint_utils.select_resume_checkpoint(
        resume_info,
        cli_config.load_checkpoint_path,
    )
    checkpoint_path = selection.checkpoint_path
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
        _write_run_meta(metadata_path, run_meta)

    session_id: str | None = None
    training_client: AzureSDKTrainingClient | None = None
    try:
        if cli_config.existing_session_id:
            session_id = cli_config.existing_session_id
            print(f"Attaching to existing session: {session_id}")
        else:
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
        _write_run_meta(metadata_path, run_meta)

        if cli_config.initial_idle_seconds > 0:
            await asyncio.sleep(cli_config.initial_idle_seconds)

        training_client = AzureSDKTrainingClient(
            client,
            session_id,
            get_tokenizer(tokenizer_name),
        )
        await train.main(
            config,
            training_client,
            prepared_datasets=prepared_datasets,
        )
        try:
            _write_accuracy_summary(log_path)
        except Exception:
            logger.exception(
                "Training completed, but the visual-spatial accuracy summary "
                "could not be generated"
            )
    finally:
        if training_client is not None:
            await training_client.close_poller()
        if session_id is not None:
            await client.close_session(session_id)
        await client.close()
        if hasattr(credential, "close"):
            await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))