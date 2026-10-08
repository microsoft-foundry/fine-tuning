"""Train a vision model to use an image-returning MCP tool with Interactive Training."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import chz
import httpx
from azure.ai.finetuningsessions.models import LossFn as LossFnType
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from interactive_training import checkpoint_utils, cli_utils, model_info
from interactive_training.recipes.mcp_image_tool.mcp_env import MCPImageToolDatasetBuilder
from interactive_training.recipes.mcp_image_tool.learning_report import build_learning_report
from interactive_training.recipes.mcp_image_tool.mixed_rl import evaluate_repeated, train_mixed
from interactive_training.recipes.mcp_image_tool.warmup import train_warmup
from interactive_training.recipes.mcp_image_tool.training_client import MCPImageTrainingClient
from interactive_training.rl.train import Config
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.tool_use import ToolInput, tools_from_mcp_session
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils

logger = logging.getLogger(__name__)
REQUIRED_TOOL = "reveal_alien_color"
MCPTransport = Literal["stdio", "streamable_http"]


@chz.chz
class CLIConfig:
    """Supervised warm-up followed by bounded mixed-task RL."""

    project_endpoint: str | None = None
    model_name: str = "meta-models/Muse-Glimmer-30B"
    tokenizer_name: str | None = None
    renderer_name: str | None = "muse_glimmer_low_reasoning"
    lora_rank: int = 32
    lora_seed: int = 552161550
    freeze_vision_tower: bool = True
    freeze_multi_modal_projector: bool = True
    load_checkpoint_path: str | None = None

    mcp_transport: MCPTransport = "stdio"
    mcp_server_command: str | None = None
    mcp_server_args: str = ""
    mcp_server_url: str | None = None
    mcp_bearer_token_env: str | None = None
    mcp_http_timeout_seconds: float = 30.0
    mcp_smoke_only: bool = False

    seed: int = 0
    max_train_examples: int = 40
    max_test_examples: int = 30
    group_size: int = 8
    groups_per_batch: int = 10
    max_concurrent_groups: int | None = 4
    max_tool_calls: int = 20
    max_trajectory_tokens: int = 8 * 1024

    mixed_train_examples: int = 60
    evaluation_repetitions: int = 1
    warmup_epochs: int = 3
    warmup_learning_rate: float = 2e-4
    learning_rate: float = 5e-6
    max_tokens: int = 2048
    temperature: float = 1.0
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    num_substeps: int = 1
    compute_post_kl: bool = False
    max_steps: int | None = 6
    max_wall_clock_seconds: float | None = None

    log_path: str | None = None
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    stripped = checkpoint_path.split("://", 1)[-1]
    parts = stripped.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "load_checkpoint_path must be '<session_id>/<checkpoint_name>' or an SDK checkpoint URI"
        )
    session_id = parts[0]
    if not session_id.startswith("session_"):
        session_id = f"session_{session_id.removeprefix('model_')}"
    return session_id, parts[1].rsplit("/", 1)[-1]


def _mcp_server_parameters(cli_config: CLIConfig) -> StdioServerParameters:
    if cli_config.mcp_server_command is None:
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "interactive_training.recipes.mcp_image_tool.server", "--server"],
        )
    return StdioServerParameters(
        command=cli_config.mcp_server_command,
        args=shlex.split(cli_config.mcp_server_args),
    )


@asynccontextmanager
async def _mcp_session(cli_config: CLIConfig) -> AsyncIterator[ClientSession]:
    if cli_config.mcp_transport == "stdio":
        async with (
            stdio_client(_mcp_server_parameters(cli_config)) as streams,
            ClientSession(*streams) as session,
        ):
            await session.initialize()
            yield session
        return

    assert cli_config.mcp_server_url is not None
    headers: dict[str, str] = {}
    if cli_config.mcp_bearer_token_env is not None:
        token = os.environ.get(cli_config.mcp_bearer_token_env)
        if not token:
            raise ValueError(
                f"MCP bearer token environment variable "
                f"{cli_config.mcp_bearer_token_env!r} is not set"
            )
        headers["Authorization"] = f"Bearer {token}"
    async with (
        httpx.AsyncClient(
            headers=headers,
            timeout=cli_config.mcp_http_timeout_seconds,
        ) as http_client,
        streamable_http_client(
            cli_config.mcp_server_url,
            http_client=http_client,
        ) as (read_stream, write_stream, _),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


def _validate_config(cli_config: CLIConfig) -> None:
    for name, value in {
        "lora_rank": cli_config.lora_rank,
        "max_train_examples": cli_config.max_train_examples,
        "max_test_examples": cli_config.max_test_examples,
        "group_size": cli_config.group_size,
        "groups_per_batch": cli_config.groups_per_batch,
        "max_tokens": cli_config.max_tokens,
        "max_tool_calls": cli_config.max_tool_calls,
        "num_substeps": cli_config.num_substeps,
        "warmup_epochs": cli_config.warmup_epochs,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if cli_config.mcp_http_timeout_seconds <= 0:
        raise ValueError("mcp_http_timeout_seconds must be greater than zero")
    if cli_config.warmup_learning_rate <= 0:
        raise ValueError("warmup_learning_rate must be positive")
    if cli_config.evaluation_repetitions < 1 or not 1 <= cli_config.mixed_train_examples <= 180:
        raise ValueError("evaluation_repetitions must be positive and mixed_train_examples in [1, 180]")
    if cli_config.max_steps is not None and cli_config.max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if not cli_config.mcp_smoke_only and not cli_config.project_endpoint:
        raise ValueError("project_endpoint is required for training")
    if cli_config.mcp_transport == "stdio":
        if cli_config.mcp_server_url is not None:
            raise ValueError("mcp_server_url requires mcp_transport=streamable_http")
        if cli_config.mcp_bearer_token_env is not None:
            raise ValueError(
                "mcp_bearer_token_env requires mcp_transport=streamable_http"
            )
    elif cli_config.mcp_transport == "streamable_http":
        if cli_config.mcp_server_url is None:
            raise ValueError("mcp_server_url is required for streamable_http")
        server_url = httpx.URL(cli_config.mcp_server_url)
        is_loopback_http = server_url.scheme == "http" and server_url.host in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
        if server_url.scheme != "https" and not is_loopback_http:
            raise ValueError(
                "mcp_server_url must use HTTPS, except for loopback HTTP servers"
            )
        if cli_config.mcp_server_command is not None or cli_config.mcp_server_args:
            raise ValueError(
                "mcp_server_command and mcp_server_args are only valid for stdio"
            )
    else:
        raise ValueError(f"unsupported mcp_transport: {cli_config.mcp_transport!r}")


def _session_client_kwargs(
    project_endpoint: str,
    azure_api_key: str | None,
) -> tuple[dict[str, Any], Any]:
    parsed_endpoint = urlparse(project_endpoint)
    is_loopback_http = (
        parsed_endpoint.scheme == "http"
        and parsed_endpoint.hostname
        in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
    )
    if azure_api_key or is_loopback_http:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key or "local-test-key")
        client_kwargs: dict[str, Any] = {
            "endpoint": project_endpoint,
            "credential": credential,
        }
        if is_loopback_http:
            client_kwargs["allow_insecure_http"] = True
        return client_kwargs, credential

    from azure.identity.aio import DefaultAzureCredential

    credential = DefaultAzureCredential()
    return {
        "endpoint": project_endpoint,
        "credential": credential,
        "credential_scopes": ["https://ai.azure.com/.default"],
    }, credential


async def _train_with_mcp(cli_config: CLIConfig) -> None:
    model_info.require_vision_language_model(
        cli_config.model_name,
        workload="MCP image tool reinforcement learning",
    )
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )
    run_name = (
        f"mcp-image-tool-{cli_config.model_name.replace('/', '-')}"
        f"-seed{cli_config.seed}-{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = cli_config.log_path or str(
        file_utils.default_logs_root() / "mcp_image_tool" / run_name
    )
    async with _mcp_session(cli_config) as mcp_session:
        tools = await tools_from_mcp_session(mcp_session)
        discovered_names = {tool.name for tool in tools}
        if REQUIRED_TOOL not in discovered_names:
            raise ValueError(
                f"MCP server must expose {REQUIRED_TOOL!r}; discovered "
                f"{sorted(discovered_names)}"
            )
        if cli_config.mcp_smoke_only:
            image_tool = next(tool for tool in tools if tool.name == REQUIRED_TOOL)
            result = await image_tool.run(
                ToolInput(arguments={"alien_word": "varkesh"}, call_id="smoke")
            )
            content = result.messages[0]["content"]
            if not isinstance(content, list) or not any(
                part["type"] in {"image", "image_url"} for part in content
            ):
                raise ValueError(f"MCP tool {REQUIRED_TOOL!r} did not return an image")
            print(
                f"MCP smoke passed: transport={cli_config.mcp_transport} "
                f"tools={sorted(discovered_names)}"
            )
            return

        cli_utils.check_log_dir(
            log_path,
            behavior_if_exists=cli_config.behavior_if_log_dir_exists,
        )
        os.makedirs(log_path, exist_ok=True)

        dataset_builder = MCPImageToolDatasetBuilder(
            model_name=cli_config.model_name,
            tokenizer_name=tokenizer_name,
            tools=tools,
            batch_size=cli_config.groups_per_batch,
            group_size=cli_config.group_size,
            renderer_name=renderer_name,
            max_tool_calls=cli_config.max_tool_calls,
            format_coef=0.0,
            max_trajectory_tokens=cli_config.max_trajectory_tokens,
            seed=cli_config.seed,
            train_examples=cli_config.max_train_examples,
            test_examples=cli_config.max_test_examples,
        )
        prepared_datasets = await dataset_builder()
        config = Config(
            learning_rate=cli_config.learning_rate,
            dataset_builder=dataset_builder,
            model_name=cli_config.model_name,
            tokenizer_name=tokenizer_name,
            lora_rank=cli_config.lora_rank,
            max_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
            kl_penalty_coef=0.0,
            num_substeps=cli_config.num_substeps,
            compute_post_kl=cli_config.compute_post_kl,
            async_config=None,
            log_path=log_path,
            load_checkpoint_path=cli_config.load_checkpoint_path,
            loss_fn=cli_config.loss_fn,
            loss_fn_config=cli_config.loss_fn_config,
            remove_constant_reward_groups=False,
            max_concurrent_groups=cli_config.max_concurrent_groups,
            max_steps=cli_config.max_steps,
            max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
            sampling_seed=cli_config.seed,
            num_groups_to_log=0,
        )

        from azure.ai.finetuningsessions.aio import FineTuningSessionClient
        from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

        if cli_config.verbose_http:
            import azure.ai.finetuningsessions._patch as patch_module

            patch_module.VERBOSE_HTTP = True
            logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(
                logging.INFO
            )

        assert cli_config.project_endpoint is not None
        client_kwargs, credential = _session_client_kwargs(
            cli_config.project_endpoint,
            os.environ.get("AZURE_AI_API_KEY"),
        )
        resume_info = checkpoint_utils.get_last_checkpoint(log_path)
        selection = checkpoint_utils.select_resume_checkpoint(
            resume_info,
            cli_config.load_checkpoint_path,
        )
        from_checkpoint = None
        if selection.checkpoint_path is not None:
            source_session_id, checkpoint_id = _parse_checkpoint_path(
                selection.checkpoint_path
            )
            from_checkpoint = FromCheckpoint(
                source_session_id=source_session_id,
                checkpoint_id=checkpoint_id,
            )

        client = FineTuningSessionClient(**client_kwargs)
        session_id = None
        try:
            session_id = await client.create_session(
                base_model=cli_config.model_name,
                lora_config=LoRAConfig(
                    rank=cli_config.lora_rank,
                    seed=cli_config.lora_seed,
                    freeze_vision_tower=cli_config.freeze_vision_tower,
                    freeze_multi_modal_projector=cli_config.freeze_multi_modal_projector,
                ),
                type="training",
                from_checkpoint=from_checkpoint,
                timeout_sec=cli_config.create_session_timeout_sec,
                user_metadata=cli_config.user_metadata,
                training_type=normalize_training_type(cli_config.training_type),
            )
            print(f"Session ready: session_id={session_id}")
            baseline = None
            tokenizer = get_tokenizer(tokenizer_name)
            training_client = MCPImageTrainingClient(client, session_id, tokenizer)
            if from_checkpoint is None:
                train_dataset, test_dataset = prepared_datasets
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                    "before_warmup"
                )
                baseline = await evaluate_repeated(
                    sampling_client, test_dataset, max_tokens=cli_config.max_tokens,
                    repetitions=cli_config.evaluation_repetitions,
                    expected_episodes=cli_config.max_test_examples, log_path=log_path,
                    stage="before_sft", step=-1, seed=cli_config.seed,
                )
                with open(os.path.join(log_path, "baseline_metrics.json"), "w") as output:
                    json.dump(baseline, output, indent=2)
                logger.info("Fresh-model baseline: %s", json.dumps(baseline))
                if baseline.get("test/env/all/total_episodes") != cli_config.max_test_examples:
                    raise RuntimeError("Incomplete baseline evaluation; refusing to start warm-up")
                await train_warmup(
                    training_client, train_dataset, epochs=cli_config.warmup_epochs,
                    learning_rate=cli_config.warmup_learning_rate,
                    seed=cli_config.seed, log_path=log_path,
                )
            rl_report = await train_mixed(
                config, training_client, tokenizer, *prepared_datasets,
                task_count=cli_config.mixed_train_examples,
                repetitions=cli_config.evaluation_repetitions,
                expected_episodes=cli_config.max_test_examples,
            )
            with open(os.path.join(log_path, "evaluations.jsonl")) as metrics_file:
                metrics = [json.loads(line) for line in metrics_file if line.strip()]
            post_sft = next((row for row in metrics if row.get("stage") != "before_sft"
                             and "test/env/all/correct" in row), None)
            report = build_learning_report(
                metrics, expected_episodes=cli_config.max_test_examples
            )
            report["warmup_epochs"] = (
                cli_config.warmup_epochs if from_checkpoint is None else 0
            )
            report["lora_seed"] = cli_config.lora_seed
            report["fresh_model_baseline"] = selection.checkpoint_path is None
            if selection.checkpoint_path is not None:
                report["accepted"] = False
            if baseline is not None and post_sft is not None:
                report["post_sft_accuracy"] = post_sft["test/env/all/correct"]
                report["rl_accuracy_gain"] = report.get("final_accuracy", 0.0) - report["post_sft_accuracy"]
            report.update(rl_report)
            report["final_checkpoint"] = f"{session_id}/final"
            with open(os.path.join(log_path, "learning_report.json"), "w") as report_file:
                json.dump(report, report_file, indent=2)
            logger.info("Learning acceptance: %s", json.dumps(report))
        finally:
            try:
                if session_id is not None:
                    await client.close_session(session_id)
            finally:
                await client.close()
                if hasattr(credential, "close"):
                    await credential.close()


async def cli_main(cli_config: CLIConfig) -> None:
    _validate_config(cli_config)
    await _train_with_mcp(cli_config)


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(config)))
