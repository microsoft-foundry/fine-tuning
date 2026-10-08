"""CLI for Direct Preference Optimization using Azure Fine-Tuning Sessions."""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Literal

import chz
from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

from interactive_training import checkpoint_utils, cli_utils, model_info
from interactive_training.recipes.preference.dpo.datasets import (
    DPODatasetBuilder,
    HelpSteer3ComparisonBuilder,
    HHHComparisonBuilder,
    UltraFeedbackComparisonBuilder,
)
from interactive_training.recipes.preference.dpo.training import Config, main
from interactive_training.rl.train_azure import (
    AzureSDKTrainingClient,
    _parse_reference_checkpoint_path,
)
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
DatasetName = Literal["hhh", "helpsteer3", "ultrafeedback"]


@chz.chz
class CLIConfig:
    project_endpoint: str
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    dataset: DatasetName = "hhh"
    load_checkpoint_path: str | None = None
    lora_rank: int = 16
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    dpo_beta: float = 0.1
    max_length: int | None = 8192
    batch_size: int = 32
    num_epochs: int = 1
    max_steps: int | None = None
    max_train_examples: int | None = None
    max_test_examples: int = 1024
    save_every: int = 20
    eval_every: int = 10
    reference_concurrency: int = 32
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None
    verbose_http: bool = False


def write_run_metadata(log_path: str, metadata: dict[str, Any]) -> None:
    with open(os.path.join(log_path, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)


def validate_resume_config(
    log_path: str, cli_config: CLIConfig, tokenizer_name: str, renderer_name: str
) -> None:
    metadata_path = os.path.join(log_path, "run_meta.json")
    if not os.path.exists(metadata_path):
        raise ValueError("Cannot validate DPO resume without run_meta.json")
    with open(metadata_path, encoding="utf-8") as handle:
        previous = json.load(handle)
    settings = chz.asdict(cli_config)
    previous_settings = previous.get("cli_config", {})
    fields = (
        "model_name",
        "lora_rank",
        "dataset",
        "batch_size",
        "max_length",
        "max_train_examples",
        "max_test_examples",
        "dpo_beta",
    )
    changed = [
        field
        for field in fields
        if field not in previous_settings or previous_settings[field] != settings[field]
    ]
    for field, resolved in (
        ("tokenizer_name", tokenizer_name),
        ("renderer_name", renderer_name),
    ):
        if previous.get(field) != resolved:
            changed.append(field)
    if changed:
        raise ValueError(
            "Incompatible DPO resume settings: "
            + ", ".join(changed)
            + ". Use the original settings or a new log_path."
        )


def get_dataset_builder(config: CLIConfig, tokenizer_name: str, renderer_name: str):
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )
    builders = {
        "hhh": HHHComparisonBuilder,
        "helpsteer3": HelpSteer3ComparisonBuilder,
        "ultrafeedback": UltraFeedbackComparisonBuilder,
    }
    comparison_builder = builders[config.dataset](
        test_size=config.max_test_examples,
        max_train_examples=config.max_train_examples,
    )
    return DPODatasetBuilder(
        common_config=common, comparison_builder=comparison_builder
    )


async def cli_main(cli_config: CLIConfig) -> None:
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )
    run_name = (
        f"dpo-{cli_config.dataset}-{cli_config.model_name.replace('/', '-')}"
        f"-{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = os.path.expanduser(
        cli_config.log_path
        or str(file_utils.default_logs_root() / "preference" / "dpo" / run_name)
    )
    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )

    config = Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        dataset_builder=get_dataset_builder(cli_config, tokenizer_name, renderer_name),
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        max_steps=cli_config.max_steps,
        dpo_beta=cli_config.dpo_beta,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        reference_concurrency=cli_config.reference_concurrency,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
    )
    resume = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume and not resume.get("reference_state_path"):
        raise ValueError("Cannot resume DPO without the original reference_state_path")
    if resume:
        validate_resume_config(log_path, cli_config, tokenizer_name, renderer_name)
    prepared_datasets = config.dataset_builder.build()
    if len(prepared_datasets[0]) == 0:
        raise ValueError("DPO dataset produced no full batches; lower batch_size")
    tokenizer = get_tokenizer(tokenizer_name)
    os.makedirs(log_path, exist_ok=True)
    run_meta = {
        "recipe": "preference/dpo",
        "run_name": run_name,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": cli_config.model_name,
        "tokenizer_name": tokenizer_name,
        "renderer_name": renderer_name,
        "dataset": cli_config.dataset,
        "cli_config": chz.asdict(cli_config),
    }
    await asyncio.to_thread(write_run_metadata, log_path, run_meta)

    checkpoint_path = (
        resume.get("state_path") if resume else cli_config.load_checkpoint_path
    )
    from_checkpoint = None
    if checkpoint_path:
        source_session, checkpoint_id = _parse_reference_checkpoint_path(
            checkpoint_path
        )
        from_checkpoint = FromCheckpoint(
            source_session_id=source_session, checkpoint_id=checkpoint_id
        )

    from azure.ai.finetuningsessions.aio import FineTuningSessionClient

    api_key = os.environ.get("AZURE_AI_API_KEY")
    if api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(api_key)
    else:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()
    client_kwargs: dict[str, Any] = {
        "endpoint": cli_config.project_endpoint,
        "credential": credential,
    }
    if not api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    client = FineTuningSessionClient(**client_kwargs)

    training_type = normalize_training_type(cli_config.training_type)
    session_ids: list[str] = []
    try:
        session_id = await client.create_session(
            base_model=cli_config.model_name,
            lora_config=LoRAConfig(rank=cli_config.lora_rank),
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=600.0,
            user_metadata=cli_config.user_metadata,
            training_type=training_type,
        )
        session_ids.append(session_id)
        training_client = AzureSDKTrainingClient(client, session_id, tokenizer)
        logger.info("DPO training session: %s", session_id)
        run_meta["azure"] = {"session_id": session_id}
        await asyncio.to_thread(write_run_metadata, log_path, run_meta)
        if resume:
            reference_state_path = resume["reference_state_path"]
            source_session, checkpoint_id = _parse_reference_checkpoint_path(
                reference_state_path
            )
            reference_session_id = await client.create_session(
                base_model=cli_config.model_name,
                lora_config=LoRAConfig(rank=cli_config.lora_rank),
                type="training",
                from_checkpoint=FromCheckpoint(
                    source_session_id=source_session, checkpoint_id=checkpoint_id
                ),
                timeout_sec=600.0,
                training_type=training_type,
            )
            session_ids.append(reference_session_id)
            reference_training_client = AzureSDKTrainingClient(
                client, reference_session_id, tokenizer
            )
        else:
            reference_future = await training_client.save_state_async("dpo_reference")
            reference_state_path = (await reference_future.result_async()).path
            reference_training_client = training_client
        reference_client = (
            await reference_training_client.save_weights_and_get_sampling_client_async(
                "dpo_reference"
            )
        )
        await main(
            config,
            training_client,
            reference_client,
            reference_state_path,
            prepared_datasets=prepared_datasets,
        )
    finally:
        for session_id in reversed(session_ids):
            try:
                await client.close_session(session_id)
            except Exception:
                logger.exception("Failed to close DPO session %s", session_id)
        try:
            await client.close()
        finally:
            if hasattr(credential, "close"):
                await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))
