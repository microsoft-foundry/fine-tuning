"""Search-R1-style tool-use RL training with Azure AI Fine-Tuning Sessions SDK."""

from __future__ import annotations

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
from interactive_training.dynamic_batching import build_strategy
from interactive_training.recipes.search_tool.search_env import SearchR1DatasetBuilder
from interactive_training.recipes.search_tool.tools import RetrievalConfig
from interactive_training.rl import train_azure as rl_azure
from interactive_training.rl.train import AsyncConfig, Config, EvaluationStrategy
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line config for Azure SDK-based Search-R1 tool-use RL."""

    # Model
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Data / reproducibility
    seed: int = 2

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 128
    max_concurrent_groups: int | None = None
    learning_rate: float = 4e-5
    max_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    remove_constant_reward_groups: bool = False

    # Dynamic-batching strategy (experimental). None = vanilla GRPO (baseline).
    # Name is resolved to a strategy object by
    # interactive_training.dynamic_batching.build_strategy; the *_ params below tune
    # the individual methods. max_rolls_per_group caps A-axis re-rolls (MORE);
    # it must be set for oversampling methods (pilot_commit) to grow a
    # prompt's pool beyond the initial wave.
    strategy: str | None = None
    max_rolls_per_group: int | None = None
    # Refill the batch to groups_per_batch when the strategy DROPs a group
    # (instead of letting the batch shrink). Makes drop-based methods (dapo,
    # pilot_commit) faithful to their papers. Only meaningful with `strategy`.
    refill_on_drop: bool = False
    pods_keep: int | None = None
    pilot_p_lower: float = 0.125
    pilot_p_upper: float = 0.75
    pilot_rollouts: int | None = None
    commit_rollouts: int = 0

    # Loss
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Reward / env
    max_turns: int = 5
    format_coef: float = 0.1
    max_trajectory_tokens: int = 32 * 1024

    # Retrieval stack (FAISS + E5 retrieval server; see retrieval_server.py)
    retrieval_host: str = "localhost"
    retrieval_port: int = 8000
    n_results: int = 3

    # Dataset limits
    max_train_examples: int | None = None
    max_test_examples: int | None = 1000
    dataset_cache_dir: str | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 20
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 20

    # Optional run cap
    max_steps: int | None = None
    # Wall-clock budget in seconds (None = no limit). The training loop
    # breaks at the next iteration boundary once elapsed time exceeds this.
    max_wall_clock_seconds: float | None = None

    # Azure endpoint / auth
    project_endpoint: str
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0

    training_type: TrainingType | None = training_type_field()

    user_metadata: dict[str, Any] | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps_off_policy: int | None = None


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Parse checkpoint path into (session_id, checkpoint_name)."""
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
        "Expected format '<session_id>/<checkpoint_name>' (e.g. 'session_abc12345/final')."
    )


async def cli_main(cli_config: CLIConfig):
    """Build RL config, create Azure session, and launch Search-R1 training."""
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    retrieval_config = RetrievalConfig(
        n_results=cli_config.n_results,
    )

    model_tag = cli_config.model_name.replace("/", "-")
    loss_fn_str = getattr(cli_config.loss_fn, "value", cli_config.loss_fn)
    strategy_tag = cli_config.strategy or "baseline"
    run_name = (
        f"search_tool-azure-sdk-{model_tag}"
        f"-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group"
        f"-{cli_config.groups_per_batch}batch"
        f"-{loss_fn_str}"
        f"-{strategy_tag}"
        f"-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    log_path = (
        cli_config.log_path
        if cli_config.log_path is not None
        else str(file_utils.default_logs_root() / "search_tool" / run_name)
    )
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = SearchR1DatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        retrieval_host=cli_config.retrieval_host,
        retrieval_port=cli_config.retrieval_port,
        retrieval_config=retrieval_config,
        seed=cli_config.seed,
        max_turns=cli_config.max_turns,
        format_coef=cli_config.format_coef,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_train_examples=cli_config.max_train_examples,
        max_test_examples=cli_config.max_test_examples,
        cache_dir=cli_config.dataset_cache_dir,
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
        sampling_seed=cli_config.seed,
        max_steps=cli_config.max_steps,
        strategy=build_strategy(
            cli_config.strategy,
            group_size=cli_config.group_size,
            pods_keep=cli_config.pods_keep,
            pilot_p_lower=cli_config.pilot_p_lower,
            pilot_p_upper=cli_config.pilot_p_upper,
            pilot_rollouts=cli_config.pilot_rollouts,
            commit_rollouts=cli_config.commit_rollouts,
        ),
        max_rolls_per_group=cli_config.max_rolls_per_group,
        refill_on_drop=cli_config.refill_on_drop,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        max_concurrent_groups=cli_config.max_concurrent_groups,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    # Write run metadata for dashboard / reproducibility.
    import re as _re

    parsed_endpoint = urlparse(cli_config.project_endpoint or "")
    endpoint_host = parsed_endpoint.netloc or cli_config.project_endpoint or ""
    host_parts = endpoint_host.split(".")

    endpoint_project: str | None = None
    project_match = _re.search(r"/projects/([^/?#]+)", parsed_endpoint.path or "")
    if project_match:
        endpoint_project = project_match.group(1)
    elif host_parts and host_parts[0] and host_parts[0].lower() != "www":
        endpoint_project = host_parts[0]

    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "search_tool_azure_sdk",
        "azure": {
            "project_endpoint": cli_config.project_endpoint,
            "project_name": endpoint_project,
            "endpoint_host": endpoint_host,
            "auth": (
                "AzureKeyCredential"
                if os.environ.get("AZURE_AI_API_KEY")
                else "DefaultAzureCredential"
            ),
            "session_id": None,
            "from_checkpoint": None,
        },
        "model": {
            "model_name": cli_config.model_name,
            "tokenizer_name": tokenizer_name,
            "renderer_name": renderer_name,
            "lora_rank": cli_config.lora_rank,
        },
        "dataset": {
            "name": "PeterJinGo/nq_hotpotqa_train",
            "group_size": cli_config.group_size,
            "groups_per_batch": cli_config.groups_per_batch,
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
            "strategy": cli_config.strategy or "baseline",
            "max_rolls_per_group": cli_config.max_rolls_per_group,
            "refill_on_drop": cli_config.refill_on_drop,
            "max_wall_clock_seconds": cli_config.max_wall_clock_seconds,
        },
        "search": {
            "max_turns": cli_config.max_turns,
            "format_coef": cli_config.format_coef,
            "max_trajectory_tokens": cli_config.max_trajectory_tokens,
            "retrieval": {
                "host": cli_config.retrieval_host,
                "port": cli_config.retrieval_port,
                "n_results": cli_config.n_results,
                "encoder": "intfloat/e5-base-v2",
            },
        },
        "command": " ".join(sys.argv),
    }

    def _write_meta() -> None:
        with open(os.path.join(log_path, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, default=str)

    _write_meta()

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
            "Using DefaultAzureCredential (managed identity excluded -> AzureCliCredential)."
        )

    if cli_config.verbose_http:
        import azure.ai.finetuningsessions._patch as _patch_module

        _patch_module.VERBOSE_HTTP = True
        logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(logging.INFO)

    client_kwargs: dict[str, Any] = {
        "endpoint": cli_config.project_endpoint,
        "credential": credential,
    }
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]

    client = FineTuningSessionClient(**client_kwargs)

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
            config, client, session_id, training_type=cli_config.training_type
        )
    finally:
        await client.close()
        if hasattr(credential, "close"):
            await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))
