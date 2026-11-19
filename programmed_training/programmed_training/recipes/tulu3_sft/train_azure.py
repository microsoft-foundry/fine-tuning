"""Tulu3 SFT training using Azure AI Fine-Tuning Sessions SDK.

Supervised fine-tuning on the allenai/tulu-3-sft-mixture dataset using
cross-entropy loss through the Programmed Training platform.  Follows the same architecture
as ``math_rl/train_azure.py``:

    1. ``CLIConfig`` configures data, training, and the ``project_endpoint``.
    2. Authentication and session creation use the Azure SDK directly.
  3. The training loop is the shared ``supervised/train.py::main()``,
     driven by ``AzureSDKTrainingClient`` from ``rl/train_azure.py``.

Usage (from the cookbook root)::

    python -m programmed_training.recipes.tulu3_sft.train_azure \\
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
from collections.abc import Awaitable
from datetime import datetime
from importlib import import_module
from typing import Any
from urllib.parse import urlparse

import chz

from programmed_training import auto_resume, checkpoint_utils, cli_utils, model_info
from programmed_training.recipes.tulu3_sft.chat_datasets import Tulu3Builder
from programmed_training.renderers import TrainOnWhat
from programmed_training.supervised import train
from programmed_training.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from programmed_training.optimizers import AdamConfig
from programmed_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from programmed_training.utils import file_utils
from programmed_training.utils.lr_scheduling import LRSchedule

logger = logging.getLogger(__name__)


def _resolve_session_route(
    project_endpoint: str,
) -> tuple[str, bool, str, bool]:
    """Resolve endpoint, auth scope, and HTTP policy for the session client."""
    if not os.environ.get("LOOM_API_ENDPOINT", "").strip():
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
            "LOOM_API_ENDPOINT requires the loom-command-job runtime utilities."
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

    credential = DefaultAzureCredential()
    try:
        async_credential = AsyncDefaultAzureCredential()
    except BaseException:
        try:
            credential.close()
        except BaseException:
            logger.exception("Credential cleanup failed during initialization")
        raise
    return credential, async_credential, True


async def _await_owned_task(task: asyncio.Future[Any]) -> Any:
    """Finish acquisition or cleanup before propagating caller cancellation.

    Cancelling an executor future does not stop its worker. Shield it so a
    session returned after cancellation is still available to the cleanup scope.
    The same rule prevents a second cancellation from abandoning cleanup.
    """
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.done():
                # Executor futures can finish *with* CancelledError without
                # reporting cancelled(). Consume any completed failure once.
                try:
                    task.result()
                except asyncio.CancelledError as operation_error:
                    raise cancellation or operation_error
                except BaseException:
                    logger.exception("Owned lifecycle operation failed during cancellation")
                raise cancellation or exc
            if cancellation is None:
                cancellation = exc
        except BaseException:
            if cancellation is not None:
                logger.exception("Owned lifecycle operation failed during cancellation")
                raise cancellation
            raise
        else:
            if cancellation is not None:
                raise cancellation
            return result


_SESSION_CLOSE_HEARTBEAT_ATTEMPTS = 12
_HEARTBEAT_STILL_RUNNING = (
    "Heartbeat is still running; session lifecycle request was not sent"
)


def _close_owned_session(session: Any) -> None:
    """Drain the owning SDK heartbeat before releasing capacity and clients.

    The sync SDK joins for five seconds, shorter than its heartbeat's combined
    connect/read timeouts. Only this explicit *pre-request* failure is safe to
    retry: no completion POST has been sent. Repeated bounded joins allow up to
    a minute for in-flight I/O/retries to drain. Never replay a completion after
    an HTTP/transport failure with an ambiguous outcome, or hide exhaustion.
    """
    for attempt in range(_SESSION_CLOSE_HEARTBEAT_ATTEMPTS):
        try:
            session.close()
            return
        except RuntimeError as exc:
            if str(exc) != _HEARTBEAT_STILL_RUNNING or attempt + 1 == _SESSION_CLOSE_HEARTBEAT_ATTEMPTS:
                raise
            logger.warning(
                "Waiting for the owned heartbeat before retrying unsent session completion (%s/%s)",
                attempt + 1, _SESSION_CLOSE_HEARTBEAT_ATTEMPTS,
            )


# =============================================================================
# CLI configuration for the shared SFT loop, with project_endpoint
# =============================================================================


@chz.chz
class CLIConfig(auto_resume.AutoResumeConfig, AdamConfig):
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
    # boundaries). See programmed_training.supervised.train.Config.pipeline_depth.
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
        **cli_config.adam_kwargs(),
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
    client = None
    async_client = None
    session: FineTuningSession | None = None
    training_client = None
    workload_error: BaseException | None = None
    session_close_task: asyncio.Task[None] | None = None
    supervisor_cleanup_started = False

    async def close_initial_session() -> None:
        nonlocal session_close_task
        if session_close_task is None:
            session_close_task = asyncio.create_task(asyncio.to_thread(_close_owned_session, session))
        await _await_owned_task(session_close_task)

    async def close_supervised_session() -> None:
        nonlocal supervisor_cleanup_started
        supervisor_cleanup_started = True
        await close_initial_session()

    try:
        if not uses_token_credential:
            logger.info("Using AzureKeyCredential (AZURE_AI_API_KEY).")
        else:
            if direct_path and configured_azure_api_key:
                logger.info(
                    "Ignoring AZURE_AI_API_KEY for the direct Programmed Training path; using the "
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

        # Automatic recovery pairs ledger weights and progress. Opt-out keeps
        # the launcher's legacy explicit-checkpoint precedence.
        resume_info = checkpoint_utils.get_last_checkpoint(log_path)
        from_checkpoint: FromCheckpoint | None = None
        if cli_config.auto_resume and resume_info and "state_path" in resume_info:
            checkpoint_path = resume_info["state_path"]
            print(f"Resuming from checkpoint: {checkpoint_path}")
        elif cli_config.load_checkpoint_path:
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

        session_kwargs = dict(
            base_model=cli_config.model_name,
            lora_config=LoRAConfig(rank=cli_config.lora_rank),
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=cli_config.create_session_timeout_sec,
            user_metadata=cli_config.user_metadata,
            training_type=normalize_training_type(cli_config.training_type),
        )

        def create_owned_session() -> FineTuningSession:
            nonlocal session
            session = FineTuningSession.create(client, **session_kwargs)
            return session

        # Creation blocks during model load. Retain ownership if the caller is
        # cancelled while the worker is still acquiring the hosted session.
        loop = asyncio.get_running_loop()
        session = await _await_owned_task(
            loop.run_in_executor(None, create_owned_session)
        )
        print(f"Session ready: session_id={session.session_id}")
        run_meta["azure"]["session_id"] = session.session_id
        _write_meta()

        # Keep SDK/tokenizer imports lazy, but inside the ownership scope: any
        # import, constructor, or metadata failure must still release capacity.
        from azure.ai.finetuningsessions.aio import FineTuningSessionClient as AsyncFineTuningSessionClient
        from programmed_training.rl.train_azure import AzureSDKTrainingClient
        from programmed_training.tokenizer_utils import get_tokenizer

        async_client_kwargs = dict(client_kwargs)
        async_client_kwargs["credential"] = async_credential
        async_client = AsyncFineTuningSessionClient(**async_client_kwargs)
        tokenizer = get_tokenizer(tokenizer_name)
        training_client = AzureSDKTrainingClient(async_client, session.session_id, tokenizer)

        await auto_resume.run_supervised(
            config, training_client, recovery_config=cli_config,
            client=async_client, session_id=session.session_id,
            client_kwargs=async_client_kwargs, session_kwargs=session_kwargs,
            prepared_datasets=prepared_datasets,
            initial_session_closer=close_supervised_session,
        )
    except BaseException as exc:
        workload_error = exc
        raise
    finally:
        async def close_owned_resources() -> BaseException | None:
            cleanup_error: BaseException | None = None

            async def close_resource(name: str, operation: Awaitable[Any]) -> None:
                nonlocal cleanup_error
                try:
                    await operation
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    logger.exception("Failed to close owned %s", name)

            if training_client is not None and not supervisor_cleanup_started:
                await close_resource("training poller", training_client.close_poller())
            if session is not None:
                # The sync wrapper owns its heartbeat. Closing through the
                # async client alone would leave that heartbeat thread running.
                await close_resource("session", close_initial_session())
            if async_client is not None and not supervisor_cleanup_started:
                await close_resource("async SDK client", async_client.close())
            if client is not None:
                await close_resource("sync SDK client", asyncio.to_thread(client.close))
            if uses_token_credential:
                await close_resource("async credential", async_credential.close())
                await close_resource("sync credential", asyncio.to_thread(credential.close))
            return cleanup_error

        try:
            cleanup_error = await _await_owned_task(asyncio.create_task(close_owned_resources()))
            if cleanup_error is not None:
                raise cleanup_error
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
