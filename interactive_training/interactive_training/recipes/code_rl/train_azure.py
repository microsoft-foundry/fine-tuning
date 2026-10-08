"""DeepCoder RL training using Azure AI Fine-Tuning Sessions SDK.

Builds the DeepCoder dataset and drives the shared Azure RL loop. Follow the
recipe README for sandbox setup before starting a run.

Usage::

    python -m interactive_training.recipes.code_rl.train_azure \\
        project_endpoint="<your-project-endpoint>" \\
        model_name="Qwen/Qwen3.8-27B" \\
        learning_rate=1e-5 temperature=1.0 max_tokens=8192 lora_rank=32 \\
        group_size=8 groups_per_batch=64 loss_fn=importance_sampling seed=42 \\
        eval_every=20 save_every=20 behavior_if_log_dir_exists=raise
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import chz
from azure.ai.finetuningsessions.models import LossFn as LossFnType

from interactive_training import cli_utils, model_info
from interactive_training.recipes.code_rl.code_env import DeepcoderDatasetBuilder
from interactive_training.recipes.code_rl.code_grading import assert_sandbox_reachable
from interactive_training.rl import train_azure as rl_azure
from interactive_training.rl.train import AsyncConfig, Config, EvaluationStrategy
from interactive_training.sandbox import SandboxBackend
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils

logger = logging.getLogger(__name__)


# =============================================================================
# CLI configuration — identical to train.py except base_url → project_endpoint
# =============================================================================


@chz.chz
class CLIConfig:
    """Command-line config for Azure SDK-based DeepCoder RL training."""

    # Model
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Data
    seed: int = 0

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 64
    # Max assistant turns per episode (multi-turn tool-use). Threaded into
    # DeepcoderDatasetBuilder; default 2 matches the builder default.
    max_turns: int = 2
    # Preserve historical shaping defaults while allowing recipes to configure
    # malformed-output rewards explicitly.
    format_coef: float = 0.1
    failed_parse_reward: float = -0.1
    max_concurrent_groups: int | None = None
    learning_rate: float = 1e-5
    max_tokens: int = 8192
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    remove_constant_reward_groups: bool = False

    # Loss
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 20
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 20

    # Azure SDK endpoint (replaces base_url from train.py)
    project_endpoint: str

    # Auth / SDK options
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0

    training_type: TrainingType | None = training_type_field()

    user_metadata: dict[str, Any] | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None

    # Cap on total training iterations (None = use full dataset).
    # A safety net against runaway runs that burn GPU-minutes.
    max_steps: int | None = None
    # Wall-clock budget in seconds (None = no limit). The training loop
    # breaks at the next iteration boundary once elapsed time exceeds this.
    max_wall_clock_seconds: float | None = None

    # Rollout resilience (advanced; opt-in).
    max_retries_per_trajectory: int = 0
    max_extra_trajectory_attempts_per_group: int = 0

    # Dataset truncation — set to None for a full run.
    # Set small values (e.g. 1000 / 200) for a fast smoke run.
    max_train_examples: int | None = None
    max_test_examples: int | None = None

    # Code execution sandbox configuration
    sandbox_backend: SandboxBackend = SandboxBackend.SANDBOXFUSION



# =============================================================================
# Checkpoint parsing — identical to math_rl/train_azure.py
# =============================================================================


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
# cli_main — mirrors train.py cli_main() then creates session + calls rl_azure
# =============================================================================


async def cli_main(cli_config: CLIConfig):
    """Build Config (same as train.py), create Azure session, run training."""

    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_tag = cli_config.model_name.replace("/", "-")
    loss_fn_str = getattr(cli_config.loss_fn, "value", cli_config.loss_fn)
    run_name = (
        f"code_rl-azure-sdk-{model_tag}"
        f"-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group"
        f"-{cli_config.groups_per_batch}batch"
        f"-{loss_fn_str}"
        f"-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    log_path = (
        cli_config.log_path
        if cli_config.log_path is not None
        else str(file_utils.default_logs_root() / "code_rl" / run_name)
    )
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = DeepcoderDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
        sandbox_backend=cli_config.sandbox_backend,
        max_turns=cli_config.max_turns,
        format_coef=cli_config.format_coef,
        failed_parse_reward=cli_config.failed_parse_reward,
        max_train_examples=cli_config.max_train_examples,
        max_test_examples=cli_config.max_test_examples,
    )

    # Fail fast if the grading sandbox is unreachable: otherwise every rollout
    # silently grades as correct=0 and the entire run produces no learning
    # signal (see code_grading.assert_sandbox_reachable).
    await assert_sandbox_reachable(cli_config.sandbox_backend)

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
        compute_post_kl=cli_config.compute_post_kl,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        sampling_seed=cli_config.seed,
        max_concurrent_groups=cli_config.max_concurrent_groups,
        max_retries_per_trajectory=cli_config.max_retries_per_trajectory,
        max_extra_trajectory_attempts_per_group=cli_config.max_extra_trajectory_attempts_per_group,
    )

    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    os.makedirs(log_path, exist_ok=True)

    # ── Write run_meta.json ──────────────────────────────────────────────────
    import re as _re

    parsed_endpoint = urlparse(cli_config.project_endpoint or "")
    endpoint_host = parsed_endpoint.netloc or cli_config.project_endpoint or ""
    host_parts = endpoint_host.split(".")

    endpoint_project: str | None = None
    pm = _re.search(r"/projects/([^/?#]+)", parsed_endpoint.path or "")
    if pm:
        endpoint_project = pm.group(1)
    elif host_parts and host_parts[0] and host_parts[0].lower() != "www":
        endpoint_project = host_parts[0]

    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "code_rl_azure_sdk",
        "azure": {
            "project_endpoint": cli_config.project_endpoint,
            "project_name": endpoint_project,
            "endpoint_host": endpoint_host,
            "auth": "AzureKeyCredential" if os.environ.get("AZURE_AI_API_KEY") else "DefaultAzureCredential",
            "session_id": None,
            "reference_session_id": None,
            "from_checkpoint": None,
        },
        "model": {
            "model_name": cli_config.model_name,
            "tokenizer_name": tokenizer_name,
            "renderer_name": renderer_name,
            "lora_rank": cli_config.lora_rank,
        },
        "dataset": {
            "name": "agentica-org/DeepCoder-Preview-Dataset",
            "group_size": cli_config.group_size,
            "groups_per_batch": cli_config.groups_per_batch,
            "max_turns": cli_config.max_turns,
            "format_coef": cli_config.format_coef,
            "failed_parse_reward": cli_config.failed_parse_reward,
            "max_train_examples": cli_config.max_train_examples,
            "max_test_examples": cli_config.max_test_examples,
            "max_concurrent_groups": cli_config.max_concurrent_groups,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "temperature": cli_config.temperature,
            "max_tokens": cli_config.max_tokens,
            "loss_fn": loss_fn_str,
            "seed": cli_config.seed,
            "eval_strategy": cli_config.eval_strategy,
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
            "max_steps": cli_config.max_steps,
            "max_wall_clock_seconds": cli_config.max_wall_clock_seconds,
        },
        "sandbox": {
            "backend": str(cli_config.sandbox_backend),
        },
        "command": " ".join(sys.argv),
    }

    def _write_meta() -> None:
        with open(os.path.join(log_path, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, default=str)

    _write_meta()

    def record_reference_session(session_id: str) -> None:
        run_meta["azure"]["reference_session_id"] = session_id
        _write_meta()

    # ── Azure SDK: auth + session creation ───────────────────────────────────
    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key)
        logger.info("Using AzureKeyCredential (AZURE_AI_API_KEY).")
    else:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
        logger.info(
            "Using DefaultAzureCredential (managed identity excluded → AzureCliCredential)."
        )

    if cli_config.verbose_http:
        import azure.ai.finetuningsessions._patch as _patch_module

        _patch_module.VERBOSE_HTTP = True
        logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(logging.INFO)

    client_kwargs: dict[str, Any] = dict(
        endpoint=cli_config.project_endpoint,
        credential=credential,
    )
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]

    client = FineTuningSessionClient(**client_kwargs)

    # ── Determine if we're resuming from a checkpoint ────────────────────────
    from interactive_training import checkpoint_utils

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    selection = checkpoint_utils.select_resume_checkpoint(
        resume_info, cli_config.load_checkpoint_path
    )
    checkpoint_path = selection.checkpoint_path
    if selection.ignored_load_checkpoint_path:
        # Auto-resume preempts an explicit load_checkpoint_path so the restored
        # weights stay consistent with the dataset cursor read from this ledger.
        print(
            f"Ignoring load_checkpoint_path={selection.ignored_load_checkpoint_path}: "
            f"resuming from the existing ledger in {log_path} instead. Use a fresh "
            "log_path to continual-fine-tune from an explicit checkpoint."
        )
    if selection.is_resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    elif checkpoint_path is not None:
        print(f"Continual fine-tuning from checkpoint: {checkpoint_path}")

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
        _write_meta()

    print(
        f"Creating session: base_model={cli_config.model_name}  "
        f"lora_rank={cli_config.lora_rank}  endpoint={cli_config.project_endpoint}"
        + (f"  from_checkpoint={from_checkpoint}" if from_checkpoint else "")
    )

    try:
        session_id = await client.create_session(
            base_model=cli_config.model_name,
            lora_config=LoRAConfig(rank=cli_config.lora_rank),
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=cli_config.create_session_timeout_sec,
            user_metadata=cli_config.user_metadata,
            training_type=normalize_training_type(cli_config.training_type),
        )
        print(f"Session ready: session_id={session_id}")

        run_meta["azure"]["session_id"] = session_id
        _write_meta()

        await rl_azure.main(
            config, client, session_id, training_type=cli_config.training_type,
            on_reference_session_created=record_reference_session,
        )
    finally:
        await client.close()
        if hasattr(credential, "close"):
            await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))
