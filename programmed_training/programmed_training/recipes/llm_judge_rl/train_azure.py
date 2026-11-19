"""BillSum RL with a separately authenticated Azure OpenAI judge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chz
from azure.ai.finetuningsessions import RequestRetryableError
from azure.ai.finetuningsessions.models import LossFn as LossFnType
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from programmed_training import auto_resume, cli_utils, model_info
from programmed_training.recipes.llm_judge_rl.data import (
    BILLSUM_DATASET_NAME,
    BILLSUM_REVISION,
)
from programmed_training.recipes.llm_judge_rl.env import BillSumDatasetBuilder
from programmed_training.recipes.llm_judge_rl.grader import (
    RUBRIC_VERSION,
    AzureOpenAIJudge,
    JudgeGradingError,
)
from programmed_training.rl import train_azure as rl_azure
from programmed_training.rl.rollouts import GroupDropEvent
from programmed_training.rl.train import AsyncConfig, Config, EvaluationStrategy
from programmed_training.optimizers import AdamConfig
from programmed_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from programmed_training.utils import file_utils

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"
DEFAULT_RENDERER_NAME = "qwen3_disable_thinking"


@chz.chz
class CLIConfig(auto_resume.AutoResumeConfig, AdamConfig):
    """Customer-facing configuration for BillSum LLM-as-judge RL."""

    # Model
    model_name: str = DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    lora_rank: int = 16
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset and task
    dataset_name: str = BILLSUM_DATASET_NAME
    dataset_revision: str = BILLSUM_REVISION
    dataset_path: str | None = None
    validation_size: int = 128
    seed: int = 20261014
    max_train_examples: int | None = 800
    max_test_examples: int | None = 128
    word_limit: int = 175
    policy_prompt_token_budget: int = 12000
    judge_prompt_token_budget: int = 16000
    judge_candidate_token_reserve: int = 512
    judge_tokenizer_encoding: str = "o200k_base"

    # Azure OpenAI judge. Credentials are never CLI fields.
    judge_endpoint: str | None = None
    judge_deployment: str | None = None
    judge_api_version: str = "2025-04-01-preview"
    judge_api_key_env_var: str = "AZURE_OPENAI_JUDGE_API_KEY"
    judge_timeout_seconds: float = 120.0
    judge_max_transport_retries: int = 2
    judge_max_structured_retries: int = 1
    judge_max_concurrency: int = 8
    judge_max_response_tokens: int = 2400
    max_judge_failed_groups: int = 10
    rubric_version: str = RUBRIC_VERSION

    # Training
    group_size: int = 8
    groups_per_batch: int = 8
    max_concurrent_groups: int | None = 4
    learning_rate: float = 5e-6
    max_tokens: int = 320
    temperature: float = 0.7
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    max_steps_off_policy: int | None = None
    max_steps: int | None = 100
    max_wall_clock_seconds: float | None = None
    sample_timeout_sec: float | None = None
    max_retries_per_trajectory: int = 4
    max_extra_trajectory_attempts_per_group: int = 4
    rollout_failure_max_resets: int = 20
    rollout_failure_reset_backoff_sec: float = 120.0

    # Evaluation and logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 10
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 4
    require_full_validation: bool = True
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Azure Fine-Tuning Sessions
    project_endpoint: str
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
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
        f"Invalid checkpoint_path={checkpoint_path!r}; expected "
        "'<session_id>/<checkpoint_name>'"
    )


def _resolve_judge_setting(
    explicit: str | None, environment_name: str
) -> str:
    value = explicit or os.environ.get(environment_name)
    if not value:
        raise ValueError(
            f"Set {environment_name} or pass its non-secret CLI equivalent"
        )
    return value


def _resolve_renderer_name(cli_config: CLIConfig) -> str:
    if cli_config.renderer_name:
        return cli_config.renderer_name
    if cli_config.model_name == DEFAULT_MODEL_NAME:
        return DEFAULT_RENDERER_NAME
    return model_info.get_recommended_renderer_name(cli_config.model_name)


def _session_creation_retry_delay(
    exc: HttpResponseError | ServiceRequestError,
    attempt: int,
) -> float | None:
    if isinstance(exc, RequestRetryableError):
        retry_after = exc.retry_after_sec
        return retry_after if retry_after is not None and retry_after > 0 else 2**attempt
    if isinstance(exc, ServiceRequestError):
        return 2**attempt
    status = getattr(exc, "status_code", None)
    code = getattr(getattr(exc, "error", None), "code", None)
    if status == 408 or code == "Timeout":
        return 2**attempt
    return None


def _make_judge(cli_config: CLIConfig) -> AzureOpenAIJudge:
    return AzureOpenAIJudge(
        endpoint=_resolve_judge_setting(cli_config.judge_endpoint, "AZURE_OPENAI_JUDGE_ENDPOINT"),
        deployment=_resolve_judge_setting(cli_config.judge_deployment, "AZURE_OPENAI_JUDGE_DEPLOYMENT"),
        api_version=cli_config.judge_api_version,
        api_key_env_var=cli_config.judge_api_key_env_var,
        timeout_seconds=cli_config.judge_timeout_seconds,
        max_transport_retries=cli_config.judge_max_transport_retries,
        max_structured_retries=cli_config.judge_max_structured_retries,
        max_concurrency=cli_config.judge_max_concurrency,
        max_response_tokens=cli_config.judge_max_response_tokens,
        prompt_token_budget=cli_config.judge_prompt_token_budget,
        tokenizer_encoding=cli_config.judge_tokenizer_encoding,
        rubric_version=cli_config.rubric_version,
    )


def _make_dataset_builder(cli_config: CLIConfig, judge: AzureOpenAIJudge) -> BillSumDatasetBuilder:
    return BillSumDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.tokenizer_name or cli_config.model_name,
        renderer_name=_resolve_renderer_name(cli_config),
        judge=judge,
        dataset_name=cli_config.dataset_name,
        dataset_revision=cli_config.dataset_revision,
        dataset_path=cli_config.dataset_path,
        validation_size=cli_config.validation_size,
        seed=cli_config.seed,
        max_train_examples=cli_config.max_train_examples,
        max_test_examples=cli_config.max_test_examples,
        policy_prompt_token_budget=cli_config.policy_prompt_token_budget,
        judge_prompt_token_budget=cli_config.judge_prompt_token_budget,
        judge_candidate_token_reserve=cli_config.judge_candidate_token_reserve,
        word_limit=cli_config.word_limit,
        policy_max_tokens=cli_config.max_tokens,
    )


class JudgeDropRecorder:
    """Persist customer-safe grader failure counts without prompts or secrets."""

    def __init__(self, path: Path, *, max_failed_groups: int) -> None:
        if max_failed_groups < 0:
            raise ValueError("max_failed_groups must be non-negative")
        self.path = path
        self.max_failed_groups = max_failed_groups
        self._lock = asyncio.Lock()
        self.total_dropped_groups = 0

    async def __call__(self, event: GroupDropEvent) -> None:
        cause = event.error.__cause__
        category = (
            cause.category
            if isinstance(cause, JudgeGradingError)
            else type(cause).__name__ if cause is not None else "unknown"
        )
        async with self._lock:
            self.total_dropped_groups += 1
            total_dropped_groups = self.total_dropped_groups
            record = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "training_step": event.training_step,
                "trajectory_count": event.trajectory_count,
                "category": category,
                "total_dropped_groups": total_dropped_groups,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        logger.warning(
            "Dropped incomplete judge group: step=%s trajectories=%d category=%s "
            "total_dropped_groups=%d",
            event.training_step,
            event.trajectory_count,
            category,
            total_dropped_groups,
        )
        if total_dropped_groups > self.max_failed_groups:
            raise RuntimeError(
                "Azure OpenAI judge failure budget exceeded: "
                f"{total_dropped_groups}>{self.max_failed_groups} dropped groups"
            )


def build_run_metadata(
    *,
    cli_config: CLIConfig,
    run_name: str,
    log_path: str,
    tokenizer_name: str,
    renderer_name: str,
    judge: AzureOpenAIJudge,
) -> dict[str, Any]:
    """Build durable metadata containing identifiers but never credentials."""

    parsed_endpoint = urlparse(cli_config.project_endpoint)
    endpoint_host = parsed_endpoint.netloc or cli_config.project_endpoint
    return {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "llm_judge_rl",
        "azure": {
            "project_endpoint": cli_config.project_endpoint,
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
            "name": cli_config.dataset_name,
            "revision": cli_config.dataset_revision,
            "local_fixture": bool(cli_config.dataset_path),
            "validation_source": "deterministic subset of official train",
            "official_test_used_for_training": False,
            "official_ca_test_used_for_training": False,
            "validation_size": cli_config.validation_size,
            "max_train_examples": cli_config.max_train_examples,
            "max_test_examples": cli_config.max_test_examples,
            "policy_prompt_token_budget": cli_config.policy_prompt_token_budget,
            "judge_prompt_token_budget": cli_config.judge_prompt_token_budget,
            "judge_candidate_token_reserve": cli_config.judge_candidate_token_reserve,
        },
        "judge": judge.durable_config(),
        "task": {
            "word_limit": cli_config.word_limit,
            "failure_log": "judge_failures.jsonl",
            "max_judge_failed_groups": cli_config.max_judge_failed_groups,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "temperature": cli_config.temperature,
            "max_tokens": cli_config.max_tokens,
            "loss_fn": getattr(cli_config.loss_fn, "value", cli_config.loss_fn),
            "seed": cli_config.seed,
            "eval_strategy": cli_config.eval_strategy,
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
            "max_steps": cli_config.max_steps,
            "max_wall_clock_seconds": cli_config.max_wall_clock_seconds,
            "rollout_failure_max_resets": cli_config.rollout_failure_max_resets,
            "rollout_failure_reset_backoff_sec": (
                cli_config.rollout_failure_reset_backoff_sec
            ),
        },
    }


async def cli_main(cli_config: CLIConfig) -> None:
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = _resolve_renderer_name(cli_config)
    judge = _make_judge(cli_config)
    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"llm-judge-rl-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = cli_config.log_path or str(
        file_utils.default_logs_root() / "llm_judge_rl" / run_name
    )
    wandb_name = cli_config.wandb_name or run_name
    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    os.makedirs(log_path, exist_ok=True)
    drop_recorder = JudgeDropRecorder(
        Path(log_path) / "judge_failures.jsonl",
        max_failed_groups=cli_config.max_judge_failed_groups,
    )

    dataset_builder = _make_dataset_builder(cli_config, judge)
    config = Config(
        learning_rate=cli_config.learning_rate,
        **cli_config.adam_kwargs(),
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
        require_full_validation=cli_config.require_full_validation,
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
        max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        sample_timeout_sec=cli_config.sample_timeout_sec,
        max_retries_per_trajectory=cli_config.max_retries_per_trajectory,
        max_extra_trajectory_attempts_per_group=(
            cli_config.max_extra_trajectory_attempts_per_group
        ),
        rollout_failure_max_resets=cli_config.rollout_failure_max_resets,
        rollout_failure_reset_backoff_sec=(
            cli_config.rollout_failure_reset_backoff_sec
        ),
        sampling_seed=cli_config.seed,
        max_concurrent_groups=cli_config.max_concurrent_groups,
        allow_incomplete_grading_group_drop=True,
        on_group_drop=drop_recorder,
    )

    from programmed_training import checkpoint_utils

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    selection = checkpoint_utils.select_resume_checkpoint(
        resume_info, cli_config.load_checkpoint_path
    )
    checkpoint_path = selection.checkpoint_path
    checkpoint_identity = None
    if checkpoint_path is not None:
        source_session_id, checkpoint_id = _parse_checkpoint_path(checkpoint_path)
        checkpoint_identity = {
            "source_session_id": source_session_id,
            "checkpoint_id": checkpoint_id,
        }

    run_meta = build_run_metadata(
        cli_config=cli_config,
        run_name=run_name,
        log_path=log_path,
        tokenizer_name=tokenizer_name,
        renderer_name=renderer_name,
        judge=judge,
    )
    run_meta["azure"]["from_checkpoint"] = checkpoint_identity

    def write_meta() -> None:
        with (Path(log_path) / "run_meta.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(run_meta, handle, indent=2, default=str)

    write_meta()

    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    training_api_key = os.environ.get("AZURE_AI_API_KEY")
    if training_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(training_api_key)
        logger.info("Using AzureKeyCredential for Loom training.")
    else:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()
        logger.info("Using DefaultAzureCredential for Loom training.")

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
    if not training_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    client = FineTuningSessionClient(**client_kwargs)

    from_checkpoint: FromCheckpoint | None = None
    if checkpoint_identity is not None:
        from_checkpoint = FromCheckpoint(
            source_session_id=checkpoint_identity["source_session_id"],
            checkpoint_id=checkpoint_identity["checkpoint_id"],
        )

    session_kwargs = {
        "base_model": cli_config.model_name,
        "lora_config": LoRAConfig(rank=cli_config.lora_rank),
        "type": "training",
        "from_checkpoint": from_checkpoint,
        "timeout_sec": cli_config.create_session_timeout_sec,
        "user_metadata": cli_config.user_metadata,
        "training_type": normalize_training_type(cli_config.training_type),
    }

    try:
        for attempt in range(3):
            try:
                session_id = await client.create_session(**session_kwargs)
                break
            except (HttpResponseError, ServiceRequestError) as exc:
                delay = _session_creation_retry_delay(exc, attempt)
                if attempt == 2 or delay is None:
                    raise
                await asyncio.sleep(delay)
        run_meta["azure"]["session_id"] = session_id
        write_meta()
        await auto_resume.run_rl(
            config,
            client,
            session_id,
            recovery_config=cli_config,
            client_kwargs=client_kwargs,
            session_kwargs=session_kwargs,
            training_type=cli_config.training_type,
        )
    finally:
        await judge.close()
        await client.close()
        if hasattr(credential, "close"):
            await credential.close()
if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(config)))
