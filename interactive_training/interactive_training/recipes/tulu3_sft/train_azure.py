"""Tulu3 SFT training using Azure AI Fine-Tuning Sessions SDK.

Supervised fine-tuning on the allenai/tulu-3-sft-mixture dataset using
cross-entropy loss through the Interactive Training platform.  Follows the same architecture
as ``math_rl/train_azure.py``:

    1. ``CLIConfig`` configures data, training, and the ``project_endpoint``.
    2. Authentication and session creation use the Azure SDK directly.
  3. The training loop is the shared ``supervised/train.py::main()``,
     driven by ``AzureSDKTrainingClient`` from ``rl/train_azure.py``.

Usage (from the cookbook root)::

    python -m interactive_training.recipes.tulu3_sft.train_azure \\
        project_endpoint="<your-project-endpoint>" \\
        model_name="Qwen/Qwen3.8-27B" \\
        lora_rank=16 \\
        learning_rate=5e-4 \\
        batch_size=128 \\
        max_length=16384 \\
        num_epochs=1 \\
        behavior_if_log_dir_exists=raise
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from importlib import import_module
from typing import Any
from urllib.parse import urlparse

import chz

from interactive_training import checkpoint_utils, cli_utils, model_info
from interactive_training.recipes.tulu3_sft.chat_datasets import Tulu3Builder
from interactive_training.renderers import TrainOnWhat
from interactive_training.supervised import train
from interactive_training.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils
from interactive_training.utils.lr_scheduling import LRSchedule

logger = logging.getLogger(__name__)


def _resolve_session_route(
    project_endpoint: str,
) -> tuple[str, bool, str, bool]:
    """Resolve endpoint, auth scope, and HTTP policy for the session client."""
    if not os.environ.get("INTERACTIVE_POST_TRAINING_API_ENDPOINT", "").strip():
        return (
            project_endpoint,
            False,
            "https://ai.azure.com/.default",
            project_endpoint.startswith("http://"),
        )

    try:
        direct_api = import_module("utils.direct_api")
    except ModuleNotFoundError as error:
        if error.name not in {"utils", "utils.direct_api"}:
            raise
        raise RuntimeError(
            "INTERACTIVE_POST_TRAINING_API_ENDPOINT requires the interactive-post-training-command-job runtime utilities."
        ) from error

    endpoint = direct_api.resolve_endpoint(project_endpoint)
    direct_api.preflight(endpoint)
    return (
        endpoint,
        True,
        direct_api.DIRECT_TOKEN_SCOPE,
        direct_api.allow_insecure_http(endpoint),
    )


def _build_session_credentials(
    *,
    direct_path: bool,
    azure_api_key: str | None,
) -> tuple[Any, Any, bool]:
    if not direct_path and azure_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key)
        return credential, credential, False

    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential

    return DefaultAzureCredential(), AsyncDefaultAzureCredential(), True


# =============================================================================
# CLI configuration for the shared SFT loop, with project_endpoint
# =============================================================================


@chz.chz
class CLIConfig:
    """Command-line config for Tulu3 SFT training via Azure SDK."""

    # Model
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    lora_rank: int = 16
    load_checkpoint_path: str | None = None

    # Training parameters
    learning_rate: float = 5e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    # Optional hard cap on optimizer steps. The historical Tulu3 benchmark
    # stops at 1740 steps (test/nll plateau). Set to 0/None to run all
    # `num_epochs` to completion.
    max_steps: int | None = None
    # Wall-clock budget in seconds (None = no limit). The training loop breaks
    # at the next batch boundary once elapsed time exceeds this.
    max_wall_clock_seconds: float | None = None
    # Run seed. Threaded into the dataset shuffle/split and per-epoch reshuffle;
    # seed=0 reproduces the historical split (back-compat).
    seed: int = 0

    # Dataset-specific parameters
    renderer_name: str | None = None
    train_on_what: TrainOnWhat | None = None
    max_length: int | None = 16384
    batch_size: int = 128
    model_context_length: int | None = None
    fail_on_truncation: bool = False
    preflight_dataset: bool = False
    preflight_only: bool = False

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing and evaluation
    save_every: int = 500
    eval_every: int = 500

    # Pipeline depth: steps kept in flight to keep the engine fed (1 = classic
    # one-step-ahead). >1 pre-queues fwd/optim so the engine never idles on the
    # client; eval and checkpointing still work (the loop drains at those
    # boundaries). See interactive_training.supervised.train.Config.pipeline_depth.
    pipeline_depth: int = 1

    # Dataset limits
    max_train_examples: int | None = None
    max_test_examples: int | None = None

    # Azure SDK
    project_endpoint: str
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0

    training_type: TrainingType | None = training_type_field()

    user_metadata: dict[str, Any] | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Parse a checkpoint path into ``(session_id, checkpoint_name)``.

    Accepts ``<session_id>/<name>`` URIs (as written to
    checkpoints.jsonl) or plain ``<session_id>/<checkpoint_name>`` paths.

    Raises ``ValueError`` on any other shape.
    """
    # Accept SDK-returned checkpoint URIs as well as session/checkpoint paths.
    if "://" in checkpoint_path:
        stripped = checkpoint_path.split("://", 1)[1]
        parts = stripped.split("/")
        if len(parts) >= 2:
            session = parts[0]
            name = parts[-1]
            if not session.startswith("session_"):
                session = f"session_{session.removeprefix('model_')}"
            return session, name

    parts = checkpoint_path.split("/", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        session = parts[0]
        if not session.startswith("session_"):
            session = f"session_{session.removeprefix('model_')}"
        return session, parts[1]

    raise ValueError(
        f"Invalid checkpoint_path={checkpoint_path!r}. "
        f"Expected format '<session_id>/<checkpoint_name>' (e.g. 'session_abc12345/final')."
    )


# =============================================================================
# cli_main
# =============================================================================


async def cli_main(cli_config: CLIConfig):
    """Build the shared SFT config, create an Azure session, and run training."""

    endpoint, direct_path, token_scope, allow_insecure_http = _resolve_session_route(
        cli_config.project_endpoint
    )
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name_safe = cli_config.model_name.replace("/", "-")
    run_name = (
        f"tulu3-{model_name_safe}"
        f"-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.batch_size}batch"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    logs_root = file_utils.default_logs_root()
    log_path = (
        cli_config.log_path
        if cli_config.log_path is not None
        else str(logs_root / "tulu3_sft" / run_name)
    )

    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    # ── Write run_meta.json: surfaces context the dashboard can show ─────────
    parsed_endpoint = urlparse(endpoint)
    endpoint_host = parsed_endpoint.netloc or endpoint
    host_parts = endpoint_host.split(".")
    endpoint_project: str | None = None
    pm = re.search(r"/projects/([^/?#]+)", parsed_endpoint.path or "")
    if pm:
        endpoint_project = pm.group(1)
    elif host_parts and host_parts[0] and host_parts[0].lower() != "www":
        endpoint_project = host_parts[0]

    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "tulu3_sft_azure_sdk",
        "azure": {
            "project_endpoint": cli_config.project_endpoint,
            "session_endpoint": endpoint,
            "project_name": endpoint_project,
            "endpoint_host": endpoint_host,
            "auth": (
                "DefaultAzureCredential"
                if direct_path or not os.environ.get("AZURE_AI_API_KEY")
                else "AzureKeyCredential"
            ),
            "session_id": None,  # filled after session creation
        },
        "model": {
            "model_name": cli_config.model_name,
            "tokenizer_name": tokenizer_name,
            "renderer_name": renderer_name,
            "lora_rank": cli_config.lora_rank,
        },
        "dataset": {
            "name": "allenai/tulu-3-sft-mixture",
            "max_train_examples": cli_config.max_train_examples,
            "max_test_examples": cli_config.max_test_examples,
            "batch_size": cli_config.batch_size,
            "max_length": cli_config.max_length,
            "model_context_length": cli_config.model_context_length,
            "fail_on_truncation": cli_config.fail_on_truncation,
            "preflight_dataset": cli_config.preflight_dataset or cli_config.preflight_only,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "lr_schedule": cli_config.lr_schedule,
            "num_epochs": cli_config.num_epochs,
            "max_steps": cli_config.max_steps,
            "max_wall_clock_seconds": cli_config.max_wall_clock_seconds,
            "seed": cli_config.seed,
            "loss_fn": "cross_entropy",
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
        },
        "command": " ".join(sys.argv),
    }

    def _write_meta() -> None:
        with open(os.path.join(log_path, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, default=str)

    _write_meta()

    common_dataset_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=cli_config.train_on_what,
        model_context_length=cli_config.model_context_length,
        fail_on_truncation=cli_config.fail_on_truncation,
    )
    dataset_builder: ChatDatasetBuilder = Tulu3Builder(
        common_config=common_dataset_config,
        max_train_examples=cli_config.max_train_examples,
        max_test_examples=cli_config.max_test_examples,
        seed=cli_config.seed,
    )

    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
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
        preflight_dataset=cli_config.preflight_dataset or cli_config.preflight_only,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )
    prepared_datasets = None
    if config.preflight_dataset:
        prepared_datasets = train.prepare_datasets(config)
        if cli_config.preflight_only:
            print("Preflight completed; no training session was created.")
            return

    # ── Azure SDK: auth + session creation ───────────────────────────────────
    from azure.ai.finetuningsessions import FineTuningSession, FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    configured_azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    azure_api_key = None if direct_path else configured_azure_api_key
    credential, async_credential, uses_token_credential = _build_session_credentials(
        direct_path=direct_path,
        azure_api_key=azure_api_key,
    )
    if not uses_token_credential:
        logger.info("Using AzureKeyCredential (AZURE_AI_API_KEY).")
    else:
        if direct_path and configured_azure_api_key:
            logger.info(
                "Ignoring AZURE_AI_API_KEY for the direct Interactive Training path; using the "
                "command-job managed identity."
            )
        logger.info(
            "Using DefaultAzureCredential (MI → CLI fallback; pin AZURE_CLIENT_ID for a specific UAMI)."
        )

    if cli_config.verbose_http:
        import azure.ai.finetuningsessions._patch as _patch_module

        _patch_module.VERBOSE_HTTP = True
        logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(logging.INFO)

    client_kwargs: dict[str, Any] = dict(
        endpoint=endpoint,
        credential=credential,
    )
    if uses_token_credential:
        client_kwargs["credential_scopes"] = [token_scope]

    if allow_insecure_http:
        client_kwargs["allow_insecure_http"] = True

    client = FineTuningSessionClient(**client_kwargs)

    # ── Resolve the checkpoint to start from ─────────────────────────────────
    # An explicit ``load_checkpoint_path`` (continual-FT, e.g. RL→SFT) wins; else
    # auto-resume from the last state saved in this log dir. Either way the new
    # session must be *created* ``from_checkpoint`` so its LoRA weights/optimizer
    # are bootstrapped from that checkpoint — the supervised loop only advances
    # the epoch/batch counters and never reloads weights itself, so omitting this
    # silently cold-starts from the base model.
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    from_checkpoint: FromCheckpoint | None = None
    if cli_config.load_checkpoint_path:
        checkpoint_path = cli_config.load_checkpoint_path
        print(f"Continual fine-tuning from checkpoint: {checkpoint_path}")
    elif resume_info and "state_path" in resume_info:
        checkpoint_path = resume_info["state_path"]
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

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
        _write_meta()

    print(
        f"Creating session: base_model={cli_config.model_name}  "
        f"lora_rank={cli_config.lora_rank}  endpoint={endpoint}"
        + (f"  from_checkpoint={from_checkpoint}" if from_checkpoint else "")
    )

    # FineTuningSession.create() is synchronous and blocks during model load
    # (up to timeout_sec).  Run it off the event loop thread.
    loop = asyncio.get_running_loop()
    session: FineTuningSession = await loop.run_in_executor(
        None,
        lambda: FineTuningSession.create(
            client,
            base_model=cli_config.model_name,
            lora_config=LoRAConfig(rank=cli_config.lora_rank),
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=cli_config.create_session_timeout_sec,
            user_metadata=cli_config.user_metadata,
            training_type=normalize_training_type(cli_config.training_type),
        ),
    )
    print(f"Session ready: session_id={session.session_id}")
    run_meta["azure"]["session_id"] = session.session_id
    _write_meta()

    # ── Wrap session in the async training client ─────────────────────────────
    from azure.ai.finetuningsessions.aio import FineTuningSessionClient as AsyncFineTuningSessionClient
    from interactive_training.rl.train_azure import AzureSDKTrainingClient
    from interactive_training.tokenizer_utils import get_tokenizer

    # AzureSDKTrainingClient needs the async SDK client for forward/forward_backward
    async_client_kwargs = dict(client_kwargs)
    async_client_kwargs["credential"] = async_credential
    async_client = AsyncFineTuningSessionClient(**async_client_kwargs)
    tokenizer = get_tokenizer(tokenizer_name)
    training_client = AzureSDKTrainingClient(async_client, session.session_id, tokenizer)

    # ── Run SFT training ──────────────────────────────────────────────────────
    workload_error: BaseException | None = None
    try:
        if prepared_datasets is None:
            await train.main(config, training_client)
        else:
            await train.main(config, training_client, prepared_datasets=prepared_datasets)
    except BaseException as exc:
        workload_error = exc
        raise
    finally:
        try:
            await training_client.close_poller()
            await async_client.close_session(session.session_id)
            await async_client.close()
            if uses_token_credential:
                await async_credential.close()
        except BaseException:
            if workload_error is None:
                raise
            logger.exception(
                "Session cleanup failed while preserving the workload failure"
            )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_main(cli_config))
