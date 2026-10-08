"""Bounded NER RL training with offline filtering and PODS dynamic batching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import chz
from azure.ai.finetuningsessions.models import LossFn as LossFnType

from interactive_training import cli_utils, model_info
from interactive_training.recipes.tool_ner_rl.dynamic_policy import NerDynamicBatchingPolicy
from interactive_training.recipes.tool_ner_rl.ner_env import (
    DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    NerDatasetBuilder,
    PromptVersion,
)
from interactive_training.recipes.tool_ner_rl.ner_task import ToolVariant
from interactive_training.rl.metric_util import RLTestSetEvaluator
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
    project_endpoint: str = os.environ.get("PROJECT_ENDPOINT", "")
    train_filter_manifest_path: str = str(Path(__file__).with_name("train_selection.json"))
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None

    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_seed: int = 42
    load_checkpoint_path: str | None = None

    dataset_revision: str = DEFAULT_DATASET_REVISION
    language: str = "en"
    seed: int = 20260828
    expected_filter_threshold: float = 0.96
    train_source_pool_size: int = 10_000
    validation_pool_size: int = 2_000
    max_test_examples: int = 200
    max_train_examples: int | None = None
    tab_fraction: float = 0.25
    tab_eval_documents: int = 8

    group_size: int = 8
    groups_per_batch: int = 16
    pods_keep: int = 4
    max_oversample_rounds: int = 4
    oversample_cushion: float = 1.2
    max_concurrent_groups: int | None = 4

    prompt_version: PromptVersion = "fewshot_fast"
    tool_variant: ToolVariant = "fast"
    max_turns: int = 2
    max_tokens: int = 768
    second_turn_max_tokens: int = 768
    max_trajectory_tokens: int = 16 * 1024

    learning_rate: float = 1e-5
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    compute_post_kl: bool = False
    max_steps_off_policy: int | None = None
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    max_steps: int = 20
    max_wall_clock_seconds: float | None = 14_400.0
    eval_every: int = 20
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 10
    sample_timeout_sec: float = 300.0
    max_retries_per_trajectory: int = 1
    max_extra_trajectory_attempts_per_group: int = 1
    ttl_seconds: int | None = 604_800

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0

    @chz.validate
    def _validate(self) -> None:
        if not self.project_endpoint:
            raise ValueError("project_endpoint is required; set PROJECT_ENDPOINT")
        if self.behavior_if_log_dir_exists == "resume":
            raise ValueError("In-place resume is unsupported; use load_checkpoint_path with a fresh log_path")
        if self.max_steps_off_policy is not None and self.max_steps_off_policy < 1:
            raise ValueError("max_steps_off_policy must be positive for async training")
        if self.tool_variant != "fast" or self.prompt_version != "fewshot_fast" or self.max_turns != 2:
            raise ValueError("This recipe supports only unfocused fewshot_fast/fast two-turn training")
        if self.tab_fraction != 0.25:
            raise ValueError("tab_fraction must be 0.25 for the mixed recipe")
        if not 1 <= self.tab_eval_documents <= 127:
            raise ValueError("tab_eval_documents must be between 1 and 127")
        if not 2 <= self.group_size <= 16:
            raise ValueError("group_size must be between 2 and 16")
        if not 2 <= self.pods_keep <= self.group_size:
            raise ValueError("pods_keep must be between 2 and group_size")
        if not 1 <= self.groups_per_batch <= 64:
            raise ValueError("groups_per_batch must be between 1 and 64")
        if not 1 <= self.max_tokens <= 768:
            raise ValueError("max_tokens must be between 1 and 768")
        if not 1 <= self.second_turn_max_tokens <= 768:
            raise ValueError("second_turn_max_tokens must be between 1 and 768")
        if not 1 <= self.max_test_examples <= 500:
            raise ValueError("max_test_examples must be between 1 and 500")
        if self.max_train_examples is not None and self.max_train_examples < 1:
            raise ValueError("max_train_examples must be positive")
        if not 1 <= self.max_steps <= 300:
            raise ValueError("max_steps must be between 1 and 300")
        if (
            self.max_wall_clock_seconds is not None
            and not 60.0 <= self.max_wall_clock_seconds <= 72_000.0
        ):
            raise ValueError("max_wall_clock_seconds must be between 60 and 72000")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest_summary(
    path: Path,
    expected_filter_threshold: float,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = manifest.get("summary")
    if not isinstance(summary, dict) or not summary.get("eligible_documents"):
        raise ValueError("Training filter manifest has no eligible document summary")
    threshold = manifest.get("config", {}).get("f1_threshold")
    if threshold != expected_filter_threshold:
        raise ValueError(
            f"Expected offline f1_threshold={expected_filter_threshold}, got {threshold!r}"
        )
    if len(manifest.get("sources", [])) < 2:
        raise ValueError("Training filter must be derived from at least two base replicates")
    return summary


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Parse a Interactive Training checkpoint URI or a ``session/checkpoint`` path."""
    if "://" in checkpoint_path:
        parts = checkpoint_path.split("://", 1)[1].split("/")
        if len(parts) >= 2 and parts[0] and parts[-1]:
            session_id = parts[0]
            if not session_id.startswith(("model_", "session_")):
                session_id = f"model_{session_id}"
            return session_id, parts[-1]
    parts = checkpoint_path.split("/", 1)
    if len(parts) == 2 and all(parts):
        session_id = parts[0]
        if not session_id.startswith(("model_", "session_")):
            session_id = f"model_{session_id}"
        return session_id, parts[1]
    raise ValueError(
        "load_checkpoint_path must be <session>/<name> or "
        "<session>/<name>"
    )


class TabTrainingEvaluator(RLTestSetEvaluator):
    def __init__(self, cli_config: CLIConfig, log_path: Path) -> None:
        from interactive_training.recipes.tool_ner_rl.eval_azure import CLIConfig as EvalConfig
        from interactive_training.recipes.tool_ner_rl.eval_azure import _tab_manifest
        from interactive_training.recipes.tool_ner_rl.ner_env import build_eval_dataset
        from interactive_training.recipes.tool_ner_rl.tab_data import load_tab_dev_tasks

        self.eval_config = EvalConfig(
            project_endpoint=cli_config.project_endpoint,
            model_name=cli_config.model_name,
            tokenizer_name=cli_config.tokenizer_name,
            renderer_name=cli_config.renderer_name,
            dataset_name="tab", dataset_split="dev",
            tab_max_documents=cli_config.tab_eval_documents,
            max_tokens=cli_config.max_tokens,
            second_turn_max_tokens=cli_config.second_turn_max_tokens,
        )
        collection = load_tab_dev_tasks(max_documents=cli_config.tab_eval_documents, seed=42)
        self.reference_tasks = collection.tasks
        self.output_root = log_path / "tab_dev"
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "manifest.json").write_text(
            json.dumps(_tab_manifest(collection, self.eval_config), indent=2), encoding="utf-8",
        )
        unique_tasks = {}
        for task in collection.tasks:
            unique_tasks.setdefault((task.document_uid, task.source_start), task)
        dataset = build_eval_dataset(
            list(unique_tasks.values()), model_name=cli_config.tokenizer_name or cli_config.model_name,
            renderer_name=cli_config.renderer_name, batch_size=8, samples_per_task=1,
            max_turns=2, prompt_version="fewshot_fast", tool_variant="fast",
            second_turn_max_tokens=cli_config.second_turn_max_tokens,
        )
        super().__init__(dataset, max_tokens=cli_config.max_tokens, name="tab", num_groups_to_log=0,
                         sample_timeout_sec=cli_config.sample_timeout_sec,
                         max_retries_per_trajectory=cli_config.max_retries_per_trajectory)

    async def __call__(self, sampling_client: Any, *, step: int | None = None) -> dict[str, float]:
        from interactive_training.recipes.tool_ner_rl.eval_azure import PromptSeededTokenCompleter
        from interactive_training.recipes.tool_ner_rl.eval_artifacts import NerEvalArtifactObserver

        output = self.output_root / f"step-{step:06d}"
        if output.exists():
            output = self.output_root / f"step-{step:06d}-repeat-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        output.mkdir(parents=True, exist_ok=False)
        self.observer = NerEvalArtifactObserver(
            output, seed=42, name="tab", tab_reference_tasks=self.reference_tasks,
            turn_token_limits=(self.eval_config.max_tokens, self.eval_config.second_turn_max_tokens),
        )
        metrics = await self.eval_token_completer(
            PromptSeededTokenCompleter(sampling_client, self.eval_config), step=step,
        )
        metrics.update({key.replace("base/", "tab/", 1): value
                        for key, value in self.observer.summary_metrics.items()})
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


async def cli_main(cli_config: CLIConfig) -> None:
    manifest_path = Path(cli_config.train_filter_manifest_path).expanduser().resolve()
    manifest_summary = _read_manifest_summary(
        manifest_path,
        cli_config.expected_filter_threshold,
    )
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name, prefer_non_thinking=True
    )
    async_training = cli_config.max_steps_off_policy is not None
    run_name = (
        f"ner-rl-{'async-pods' if async_training else 'pods'}-{cli_config.model_name.replace('/', '-')}-g{cli_config.group_size}"
        f"-b{cli_config.groups_per_batch}-k{cli_config.pods_keep}-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = (
        Path(cli_config.log_path).expanduser()
        if cli_config.log_path
        else file_utils.default_logs_root() / "tool_ner_rl" / run_name
    )
    cli_utils.check_log_dir(
        str(log_path),
        behavior_if_exists=cli_config.behavior_if_log_dir_exists,
    )
    log_path.mkdir(parents=True, exist_ok=True)
    if any(log_path.iterdir()):
        raise ValueError("In-place resume is unsupported; use load_checkpoint_path with a fresh log_path")
    copied_manifest = log_path / "train_filter_manifest.json"
    shutil.copy2(manifest_path, copied_manifest)

    policy = NerDynamicBatchingPolicy(
        group_size=cli_config.group_size,
        pods_keep=cli_config.pods_keep,
        max_oversample_rounds=cli_config.max_oversample_rounds,
        oversample_cushion=cli_config.oversample_cushion,
    )
    policy_overrides = policy.training_overrides()
    if async_training:
        policy_overrides.update(dynamic_sampling=False)
    dataset_builder = NerDatasetBuilder(
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        train_filter_manifest_path=str(copied_manifest),
        dataset_revision=cli_config.dataset_revision,
        language=cli_config.language,
        seed=cli_config.seed,
        train_source_pool_size=cli_config.train_source_pool_size,
        validation_pool_size=cli_config.validation_pool_size,
        max_test_examples=cli_config.max_test_examples,
        max_train_examples=cli_config.max_train_examples,
        max_turns=cli_config.max_turns,
        prompt_version=cli_config.prompt_version,
        tool_variant=cli_config.tool_variant,
        second_turn_max_tokens=cli_config.second_turn_max_tokens,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        tab_fraction=cli_config.tab_fraction,
        mixture_manifest_path=str(log_path / "mixture_manifest.json"),
    )
    training_config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=tokenizer_name,
        tokenizer_name=tokenizer_name,
        lora_rank=cli_config.lora_rank,
        lora_alpha=cli_config.lora_alpha,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        top_p=cli_config.top_p,
        top_k=cli_config.top_k,
        log_path=str(log_path),
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        compute_post_kl=cli_config.compute_post_kl,
        eval_strategy=cli_config.eval_strategy,
        async_config=(
            AsyncConfig(
                max_steps_off_policy=cli_config.max_steps_off_policy,
                groups_per_batch=cli_config.groups_per_batch,
            )
            if cli_config.max_steps_off_policy is not None else None
        ),
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        sampling_seed=cli_config.seed,
        max_concurrent_groups=None if async_training else cli_config.max_concurrent_groups,
        sample_timeout_sec=cli_config.sample_timeout_sec,
        max_retries_per_trajectory=cli_config.max_retries_per_trajectory,
        max_extra_trajectory_attempts_per_group=(
            cli_config.max_extra_trajectory_attempts_per_group
        ),
        ttl_seconds=cli_config.ttl_seconds,
        num_groups_to_log=0,
        evaluator_builders=[partial(TabTrainingEvaluator, cli_config, log_path)] if cli_config.tab_fraction else [],
        **policy_overrides,
    )

    metadata = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "bounded_ner_rl_training",
        "model": {
            "name": cli_config.model_name,
            "renderer": renderer_name,
            "lora_rank": cli_config.lora_rank,
            "lora_alpha": cli_config.lora_alpha,
            "lora_seed": cli_config.lora_seed,
        },
        "dataset": {
            "name": DATASET_NAME,
            "revision": cli_config.dataset_revision,
            "train_source_pool_size": cli_config.train_source_pool_size,
            "eligible_documents": manifest_summary["eligible_documents"],
            "eligible_gold_entities": manifest_summary["eligible_gold_entities"],
            "filter_threshold": cli_config.expected_filter_threshold,
            "manifest_path": str(copied_manifest),
            "manifest_sha256": _sha256(copied_manifest),
            "validation_selection": "random_unfiltered",
            "validation_pool_size": cli_config.validation_pool_size,
            "validation_examples": cli_config.max_test_examples,
            "full_document_reward": "exact_set_f1",
            "tab_fraction_before_pods": cli_config.tab_fraction,
            "tab_eval_documents": cli_config.tab_eval_documents if cli_config.tab_fraction else 0,
            "tab_setup": "v4_target_context_native_schemas_train_example_shared_dev_predictions",
        },
        "batching": {
            "strategy": "pods",
            "group_size": cli_config.group_size,
            "groups_per_batch": cli_config.groups_per_batch,
            "pods_keep": cli_config.pods_keep,
            "dynamic_sampling": not async_training,
            "max_steps_off_policy": cli_config.max_steps_off_policy,
            "constant_reward_filter": True,
            "max_oversample_rounds": cli_config.max_oversample_rounds,
            "oversample_cushion": cli_config.oversample_cushion,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "loss_fn": str(cli_config.loss_fn),
            "max_steps": cli_config.max_steps,
            "max_wall_clock_seconds": cli_config.max_wall_clock_seconds,
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
            "seed": cli_config.seed,
        },
        "latency_contract": {
            "max_turns": cli_config.max_turns,
            "turn_1_max_tokens": cli_config.max_tokens,
            "later_turn_max_tokens": cli_config.second_turn_max_tokens,
        },
        "azure": {
            "session_id": None,
            "reference_session_id": None,
            "project_endpoint": cli_config.project_endpoint,
            "from_checkpoint": None,
            "training_type": cli_config.training_type,
        },
        "command": " ".join(sys.argv),
    }

    def write_metadata() -> None:
        (log_path / "run_meta.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

    write_metadata()

    def record_reference_session(reference_session_id: str) -> None:
        metadata["azure"]["reference_session_id"] = reference_session_id
        write_metadata()

    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    credential: Any
    if azure_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key)
    else:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential(exclude_managed_identity_credential=True)

    client_kwargs: dict[str, Any] = {
        "endpoint": cli_config.project_endpoint,
        "credential": credential,
    }
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    client = FineTuningSessionClient(**client_kwargs)

    from_checkpoint: FromCheckpoint | None = None
    if cli_config.load_checkpoint_path is not None:
        source_session_id, checkpoint_id = _parse_checkpoint_path(
            cli_config.load_checkpoint_path
        )
        from_checkpoint = FromCheckpoint(
            source_session_id=source_session_id,
            checkpoint_id=checkpoint_id,
        )
        metadata["azure"]["from_checkpoint"] = {
            "source_session_id": source_session_id,
            "checkpoint_id": checkpoint_id,
        }
        write_metadata()

    session_id: str | None = None
    try:
        session_id = await client.create_session(
            base_model=cli_config.model_name,
            lora_config=LoRAConfig(
                rank=cli_config.lora_rank,
                alpha=cli_config.lora_alpha,
                seed=cli_config.lora_seed,
            ),
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=cli_config.create_session_timeout_sec,
            user_metadata=cli_config.user_metadata,
            training_type=normalize_training_type(cli_config.training_type),
        )
        metadata["azure"]["session_id"] = session_id
        write_metadata()
        await rl_azure.main(
            training_config, client, session_id, training_type=cli_config.training_type,
            on_reference_session_created=record_reference_session,
        )
    except BaseException:
        if session_id is not None:
            try:
                async with FineTuningSessionClient(**client_kwargs) as cleanup_client:
                    await cleanup_client.close_session(session_id)
            except Exception:
                logger.exception("Failed to close training session %s after failure", session_id)
        raise
    finally:
        await client.close()
        close = getattr(credential, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))