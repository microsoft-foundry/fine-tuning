"""Run long-context counting RL through Azure Fine-Tuning Sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chz
from azure.ai.finetuningsessions import TrainingEngineError
from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig, LossFn
from azure.core.rest import AsyncHttpResponse, HttpRequest

from programmed_training import auto_resume, checkpoint_utils, cli_utils, model_info
from programmed_training.optimizers import AdamConfig
from programmed_training.recipes.rl_numerics_check.env import CountingDatasetBuilder, CountingTaskConfig
from programmed_training.rl import train_azure as rl_azure
from programmed_training.rl.train import AsyncConfig, Config, EvaluationStrategy
from programmed_training.rl.train_azure import _parse_reference_checkpoint_path
from programmed_training.training_types import TrainingType, normalize_training_type, training_type_field
from programmed_training.utils.file_utils import default_logs_root

logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "Qwen/Qwen3.8-27B-256K"


class _RecipeClient(FineTuningSessionClient):
    """Share cleanup results between the launcher, shared loop and recovery."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allocated_session_id: str | None = None
        self._session_closures: dict[str, asyncio.Task[None]] = {}
        self._client_closure: asyncio.Task[None] | None = None

    async def send_request(
        self, request: HttpRequest, *, stream: bool = False, **kwargs: Any,
    ) -> AsyncHttpResponse:
        response = await super().send_request(request, stream=stream, **kwargs)
        if (
            request.method == "POST"
            and urlparse(request.url).path.rstrip("/").endswith("/fine_tuning/sessions")
            and response.status_code < 400
        ):
            payload = response.json()
            session_id = payload.get("session_id") if isinstance(payload, dict) else None
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("Session allocation response did not contain a session ID")
            self.allocated_session_id = session_id
            logger.info("Training session allocated: %s", session_id)
        return response

    async def close_session(self, session_id: str) -> None:
        if session_id not in self._session_closures:
            self._session_closures[session_id] = asyncio.create_task(
                super().close_session(session_id)
            )
        await self._session_closures[session_id]

    async def close(self) -> None:
        if self._client_closure is None:
            self._client_closure = asyncio.create_task(super().close())
        await self._client_closure

    @property
    def client_closed(self) -> bool:
        return (
            self._client_closure is not None and self._client_closure.done()
            and not self._client_closure.cancelled() and self._client_closure.exception() is None
        )


@chz.chz
class CLIConfig(auto_resume.AutoResumeConfig, AdamConfig):
    project_endpoint: str
    model_name: str = DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32
    load_checkpoint_path: str | None = None
    learning_rate: float = 4e-5
    loss_fn: LossFn = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    kl_penalty_coef: float = 0.0
    group_size: int = 4
    groups_per_batch: int = 1
    max_concurrent_groups: int | None = None
    max_tokens: int = 8192
    max_trajectory_tokens: int = 262144
    temperature: float = 1.0
    num_substeps: int = 1
    task: CountingTaskConfig = chz.field(default_factory=CountingTaskConfig)
    corpus_path: str | None = None
    seed: int = 0
    max_train_examples: int = 10
    max_test_examples: int = 0
    max_steps: int | None = 10
    max_wall_clock_seconds: float | None = 259200
    max_steps_off_policy: int | None = None
    eval_every: int = 0
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 1
    compute_post_kl: bool = False
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "raise"
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None

    @chz.validate
    def _validate(self) -> None:
        for name in (
            "group_size", "groups_per_batch", "max_tokens", "max_trajectory_tokens",
            "max_train_examples", "num_substeps", "lora_rank",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO advantages")
        if self.max_tokens >= self.max_trajectory_tokens:
            raise ValueError("max_tokens must be smaller than max_trajectory_tokens")
        for name in ("max_test_examples", "eval_every", "save_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("max_steps", "max_concurrent_groups", "max_wall_clock_seconds"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when set")
        if self.max_steps_off_policy is not None and self.max_steps_off_policy < 0:
            raise ValueError("max_steps_off_policy must be nonnegative when set")


def build_config(cli_config: CLIConfig) -> Config:
    tokenizer = cli_config.tokenizer_name or (
        DEFAULT_MODEL_NAME.removesuffix("-256K")
        if cli_config.model_name == DEFAULT_MODEL_NAME else cli_config.model_name
    )
    renderer = cli_config.renderer_name or (
        "qwen3_8" if cli_config.model_name == DEFAULT_MODEL_NAME
        else model_info.get_recommended_renderer_name(
            cli_config.model_name, prefer_non_thinking=True
        )
    )
    run_name = (
        f"rl-numerics-{cli_config.model_name.replace('/', '-')}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    return Config(
        model_name=cli_config.model_name, tokenizer_name=tokenizer,
        dataset_builder=CountingDatasetBuilder(
            model_name_for_tokenizer=tokenizer, renderer_name=renderer,
            batch_size=cli_config.groups_per_batch, group_size=cli_config.group_size,
            task=cli_config.task, corpus_path=cli_config.corpus_path, seed=cli_config.seed,
            max_train_examples=cli_config.max_train_examples,
            max_test_examples=cli_config.max_test_examples,
            max_trajectory_tokens=cli_config.max_trajectory_tokens,
            max_generation_tokens=cli_config.max_tokens,
        ),
        log_path=os.path.expanduser(
            cli_config.log_path or str(default_logs_root() / "rl_numerics_check" / run_name)
        ),
        learning_rate=cli_config.learning_rate, lora_rank=cli_config.lora_rank,
        **cli_config.adam_kwargs(),
        max_tokens=cli_config.max_tokens, temperature=cli_config.temperature,
        loss_fn=cli_config.loss_fn, loss_fn_config=cli_config.loss_fn_config,
        kl_penalty_coef=cli_config.kl_penalty_coef, num_substeps=cli_config.num_substeps,
        compute_post_kl=cli_config.compute_post_kl, sampling_seed=cli_config.seed,
        eval_every=cli_config.eval_every, eval_strategy=cli_config.eval_strategy,
        save_every=cli_config.save_every, max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        max_concurrent_groups=cli_config.max_concurrent_groups,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        async_config=(
            AsyncConfig(
                max_steps_off_policy=cli_config.max_steps_off_policy,
                groups_per_batch=cli_config.groups_per_batch,
            ) if cli_config.max_steps_off_policy is not None else None
        ),
    )


async def cli_main(cli_config: CLIConfig) -> None:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    from azure.core.credentials import AzureKeyCredential
    from azure.identity.aio import DefaultAzureCredential

    config = build_config(cli_config)
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    resume = checkpoint_utils.get_last_checkpoint(config.log_path)
    selection = checkpoint_utils.select_resume_checkpoint(resume, cli_config.load_checkpoint_path)
    if selection.ignored_load_checkpoint_path:
        logger.warning(
            "Using the existing checkpoint ledger instead of load_checkpoint_path=%s",
            selection.ignored_load_checkpoint_path,
        )
    from_checkpoint = None
    if selection.checkpoint_path is not None:
        source, checkpoint = _parse_reference_checkpoint_path(selection.checkpoint_path)
        from_checkpoint = FromCheckpoint(source_session_id=source, checkpoint_id=checkpoint)

    preparation_start = time.perf_counter()
    prepared = await config.dataset_builder()
    preparation_seconds = time.perf_counter() - preparation_start
    Path(config.log_path).mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("AZURE_AI_API_KEY")
    credential = (
        AzureKeyCredential(api_key) if api_key
        else DefaultAzureCredential(exclude_managed_identity_credential=True)
    )
    logger.info("Using %s", type(credential).__name__)
    kwargs: dict[str, Any] = {"endpoint": cli_config.project_endpoint, "credential": credential}
    if not api_key:
        kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    metadata = {
        "schema_version": 1,
        "recipe": "rl_numerics_check",
        "run_name": config.wandb_name,
        "started_at": started_at,
        "dataset_preparation_seconds": preparation_seconds,
        "model": {
            "model_name": cli_config.model_name,
            "tokenizer_name": config.tokenizer_name,
            "renderer_name": config.dataset_builder.renderer_name,
            "lora_rank": cli_config.lora_rank,
        },
        "azure": {
            "project_endpoint": cli_config.project_endpoint,
            "endpoint_host": urlparse(cli_config.project_endpoint).netloc,
            "session_id": None,
        },
        "cli_config": chz.asdict(cli_config),
    }

    def write_metadata() -> None:
        (Path(config.log_path) / "run_meta.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    write_metadata()
    client = _RecipeClient(**kwargs)
    session_id: str | None = None
    failed = False
    session_kwargs = dict(
        base_model=cli_config.model_name, lora_config=LoRAConfig(rank=cli_config.lora_rank),
        type="training", from_checkpoint=from_checkpoint,
        timeout_sec=cli_config.create_session_timeout_sec,
        user_metadata=cli_config.user_metadata,
        training_type=normalize_training_type(cli_config.training_type),
    )
    try:
        try:
            session_id = await client.create_session(**session_kwargs)
        except TrainingEngineError as exc:
            session_id = exc.session_id
            raise
        logger.info("Training session ready: %s", session_id)
        metadata["azure"]["session_id"] = session_id
        write_metadata()
        await auto_resume.run_rl(
            config, client, session_id,
            training_type=cli_config.training_type, prepared_datasets=prepared,
            recovery_config=cli_config, client_kwargs=kwargs,
            session_kwargs=session_kwargs,
        )
    except BaseException:
        failed = True
        raise
    finally:
        session_id = session_id or client.allocated_session_id
        cleanup_end = time.monotonic() + auto_resume.CLEANUP_TIMEOUT_SECONDS
        session_cleanup_end = cleanup_end - min(1.0, auto_resume.CLEANUP_TIMEOUT_SECONDS / 10)
        try:
            try:
                try:
                    # A successful supervisor has already handled the original session's close result.
                    if session_id is not None and not (
                        not failed and cli_config.auto_resume and client.client_closed
                    ):
                        await auto_resume._bounded(
                            client.close_session(session_id),
                            session_cleanup_end - time.monotonic(),
                        )
                finally:
                    await auto_resume._bounded(
                        client.close(), cleanup_end - time.monotonic(),
                    )
            except BaseException:
                logger.exception("Recipe session/client cleanup failed")
                if not failed:
                    raise
        finally:
            if isinstance(credential, DefaultAzureCredential):
                await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))
