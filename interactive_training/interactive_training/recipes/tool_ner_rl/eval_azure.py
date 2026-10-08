"""Evaluate a frozen base model on tool-grounded NER without training."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import chz

from interactive_training import cli_utils, model_info, tokenizer_utils
from interactive_training.completers import SessionTokenCompleter, TokenCompleter
from interactive_training.recipes.tool_ner_rl.ner_env import (
    DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    FEWSHOT_TRAIN_UIDS,
    MAX_SAFE_EVAL_EXAMPLES,
    MAX_SAFE_TRAIN_SCORING_EXAMPLES,
    PromptVersion,
    build_eval_dataset,
    load_openpii_tasks,
    stratified_sample_tasks,
)
from interactive_training.recipes.tool_ner_rl.tab_data import (
    TAB_DATASET_NAME,
    TAB_DEV_SHA256,
    TAB_REVISION,
    TabTaskCollection,
    load_tab_dev_tasks,
)
from interactive_training.recipes.tool_ner_rl.eval_artifacts import NerEvalArtifactObserver
from interactive_training.recipes.tool_ner_rl.ner_task import ToolVariant
from interactive_training.recipes.tool_ner_rl.offline_filter import eligible_uids
from interactive_training.rl.metric_util import RLTestSetEvaluator
from interactive_training.rl.train_azure import AzureSDKTrainingClient
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import logtree

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Small, evaluation-only configuration with hard corpus limits."""

    project_endpoint: str = os.environ.get("PROJECT_ENDPOINT", "")
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 16
    lora_seed: int = 42
    load_checkpoint_path: str | None = None
    training_type: TrainingType | None = training_type_field()

    dataset_name: Literal["openpii", "tab"] = "openpii"
    dataset_revision: str = DEFAULT_DATASET_REVISION
    dataset_split: Literal["train", "validation", "dev"] = "validation"
    language: str = "en"
    max_examples: int = 8
    sample_pool_size: int = 200
    sample_seed: int = 42
    selection_method: Literal["stratified", "random"] = "stratified"
    uid_filter_manifest: str | None = None
    max_documents_scanned: int = 800
    tab_source_path: str | None = None
    tab_max_documents: int = 8
    tab_max_words: int = 128
    tab_overlap_words: int = 16
    samples_per_task: int = 1
    batch_size: int = 8

    prompt_version: PromptVersion = "fewshot_fast"
    tool_variant: ToolVariant = "fast"
    max_turns: int = 2
    max_tokens: int = 768
    second_turn_max_tokens: int = 768
    num_groups_to_log: int = 0
    error_sample_size: int = 12
    max_concurrent_samples: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    sample_timeout_sec: float = 300.0
    max_retries_per_trajectory: int = 1

    log_path: str | None = None
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0

    @chz.validate
    def _validate_bounds(self) -> None:
        if not self.project_endpoint:
            raise ValueError("project_endpoint is required; set PROJECT_ENDPOINT or pass it explicitly")
        if self.tool_variant != "fast" or self.prompt_version != "fewshot_fast" or self.max_turns != 2:
            raise ValueError("prompt_version/tool_variant/max_turns must select fewshot_fast/fast two-turn evaluation")
        if self.dataset_name == "tab":
            if self.dataset_split != "dev":
                raise ValueError("TAB evaluation requires dataset_split=dev")
            if not 1 <= self.tab_max_documents <= 127:
                raise ValueError("tab_max_documents must be between 1 and 127")
            if self.tab_max_words <= 0:
                raise ValueError("tab_max_words must be positive")
            if not 0 <= self.tab_overlap_words < self.tab_max_words:
                raise ValueError(
                    "tab_overlap_words must be between 0 and tab_max_words - 1"
                )
            if self.samples_per_task != 1:
                raise ValueError(
                    "TAB document aggregation currently requires samples_per_task=1"
                )
        else:
            if self.dataset_split == "dev":
                raise ValueError("OpenPII does not have a dev split")
            max_eval_examples = (
                MAX_SAFE_TRAIN_SCORING_EXAMPLES
                if self.dataset_split == "train"
                else MAX_SAFE_EVAL_EXAMPLES
            )
            if not 1 <= self.max_examples <= max_eval_examples:
                raise ValueError(
                    f"max_examples must be between 1 and {max_eval_examples} for "
                    f"dataset_split={self.dataset_split}"
                )
        if not 1 <= self.samples_per_task <= 8:
            raise ValueError("samples_per_task must be between 1 and 8")
        if self.dataset_name == "openpii" and not (
            self.max_examples <= self.sample_pool_size <= 12_000
        ):
            raise ValueError("sample_pool_size must be between max_examples and 12000")
        if not 1 <= self.max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024 during the constrained phase")
        if not 1 <= self.second_turn_max_tokens <= self.max_tokens:
            raise ValueError("second_turn_max_tokens must be between 1 and max_tokens")
        if not 1 <= self.max_concurrent_samples <= 64:
            raise ValueError("max_concurrent_samples must be between 1 and 64")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if (
            self.dataset_name == "openpii"
            and self.max_documents_scanned < self.sample_pool_size
        ):
            raise ValueError("max_documents_scanned must be at least sample_pool_size")


def _default_log_path(config: CLIConfig) -> Path:
    from interactive_training.utils.file_utils import default_logs_root

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    variant = f"{config.prompt_version}-{config.tool_variant}"
    return default_logs_root() / "tool_ner_rl" / f"eval-{variant}-{timestamp}"


def _run_metadata(config: CLIConfig, log_path: Path) -> dict[str, Any]:
    tokenizer_name = config.tokenizer_name or config.model_name
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name, prefer_non_thinking=True
    )
    return {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "base_model_evaluation_only",
        "training_steps": 0,
        "model": {
            "model_name": config.model_name,
            "tokenizer_name": tokenizer_name,
            "renderer_name": renderer_name,
            "lora_rank_for_session": config.lora_rank,
            "lora_seed": config.lora_seed,
            "load_checkpoint_path": config.load_checkpoint_path,
        },
        "dataset": {
            "name": (
                TAB_DATASET_NAME if config.dataset_name == "tab" else DATASET_NAME
            ),
            "revision": (
                TAB_REVISION if config.dataset_name == "tab" else config.dataset_revision
            ),
            "split": config.dataset_split,
            "language": config.language,
            "max_examples": config.max_examples,
            "sample_pool_size": config.sample_pool_size,
            "sample_seed": config.sample_seed,
            "max_documents_scanned": config.max_documents_scanned,
            "uid_filter_manifest": config.uid_filter_manifest,
            "tab_max_documents": config.tab_max_documents,
            "tab_max_words": config.tab_max_words,
            "tab_overlap_words": config.tab_overlap_words,
            "tab_dev_sha256": TAB_DEV_SHA256 if config.dataset_name == "tab" else None,
        },
        "evaluation": {
            "prompt_version": config.prompt_version,
            "tool_variant": config.tool_variant,
            "samples_per_task": config.samples_per_task,
            "max_turns": config.max_turns,
            "max_tokens": config.max_tokens,
            "second_turn_max_tokens": config.second_turn_max_tokens,
            "max_total_generated_tokens": config.max_tokens
            + (config.max_turns - 1) * config.second_turn_max_tokens,
            "max_concurrent_samples": config.max_concurrent_samples,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "sample_timeout_sec": config.sample_timeout_sec,
            "max_retries_per_trajectory": config.max_retries_per_trajectory,
        },
        "azure": {
            "project_endpoint": config.project_endpoint,
            "auth": "AzureKeyCredential"
            if os.environ.get("AZURE_AI_API_KEY")
            else "DefaultAzureCredential",
            "session_id": None,
        },
        "log_path": str(log_path),
        "command": " ".join(sys.argv),
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, default=str)


def add_micro_metrics(metrics: dict[str, float], prefix: str = "base/env/all") -> None:
    true_positives = metrics.get(f"{prefix}/exact_true_positives")
    predicted = metrics.get(f"{prefix}/predicted_entities")
    gold = metrics.get(f"{prefix}/gold_entities")
    if true_positives is None or predicted is None or gold is None:
        return
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / gold if gold else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics[f"{prefix}/micro_precision"] = precision
    metrics[f"{prefix}/micro_recall"] = recall
    metrics[f"{prefix}/micro_f1"] = f1


class PromptSeededTokenCompleter(TokenCompleter):
    """Apply a stable seed derived from each rendered prompt, independent of concurrency."""

    def __init__(self, sampling_client: Any, config: CLIConfig) -> None:
        self.sampling_client = sampling_client
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_samples)
        self.prompt_counts: dict[bytes, int] = {}

    async def __call__(self, model_input, stop, max_tokens=None):
        prompt_tokens = [token for chunk in model_input.chunks for token in chunk.tokens]
        prompt_key = hashlib.sha256(
            ",".join(str(token) for token in prompt_tokens).encode()
        ).digest()
        occurrence = self.prompt_counts.get(prompt_key, 0)
        self.prompt_counts[prompt_key] = occurrence + 1
        call_seed = prompt_seed(prompt_tokens, self.config.sample_seed, occurrence)
        delegate = SessionTokenCompleter(
            self.sampling_client,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            seed=call_seed,
            sample_timeout_sec=self.config.sample_timeout_sec,
        )
        async with self.semaphore:
            return await delegate(model_input, stop, max_tokens)


def prompt_seed(prompt_tokens: list[int], base_seed: int, occurrence: int = 0) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{occurrence}:".encode()
        + ",".join(str(token) for token in prompt_tokens).encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    if "://" in checkpoint_path:
        stripped = checkpoint_path.split("://", 1)[1]
        parts = [part for part in stripped.split("/") if part]
        if len(parts) >= 2:
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
        "<session>/<checkpoint>"
    )


def _manifest(
    tasks,
    pool_tasks,
    config: CLIConfig,
    uid_filter_info: dict[str, Any] | None,
) -> dict[str, Any]:
    def summarize(items) -> dict[str, Any]:
        label_counts: dict[str, int] = {}
        entity_counts = []
        for task in items:
            entity_counts.append(len(task.gold_entities))
            for entity in task.gold_entities:
                label_counts[entity.label] = label_counts.get(entity.label, 0) + 1
        return {
            "documents": len(items),
            "entities": sum(entity_counts),
            "entity_count_min": min(entity_counts),
            "entity_count_max": max(entity_counts),
            "entity_count_mean": sum(entity_counts) / len(entity_counts),
            "label_counts": dict(sorted(label_counts.items())),
        }

    return {
        "dataset": DATASET_NAME,
        "revision": config.dataset_revision,
        "split": config.dataset_split,
        "language": config.language,
        "excluded_prompt_example_uids": (
            sorted(FEWSHOT_TRAIN_UIDS) if config.dataset_split == "train" else []
        ),
        "uid_filter": uid_filter_info,
        "selection": config.selection_method,
        "seed": config.sample_seed,
        "pool": summarize(pool_tasks),
        "sample": summarize(tasks),
        "documents": [
            {"uid": task.uid, "gold_entities": len(task.gold_entities)} for task in tasks
        ],
    }


def _tab_manifest(collection: TabTaskCollection, config: CLIConfig) -> dict[str, Any]:
    return {
        "dataset": TAB_DATASET_NAME,
        "revision": TAB_REVISION,
        "split": "dev",
        "sha256": TAB_DEV_SHA256,
        "seed": config.sample_seed,
        "selection_unit": "document",
        "total_documents": collection.total_documents,
        "selected_documents": len(collection.selected_document_uids),
        "selected_document_uids": list(collection.selected_document_uids),
        "chunks": collection.chunk_count,
        "annotation_layers": collection.annotation_layer_count,
        "environment_groups": collection.chunk_count,
        "reference_tasks": len(collection.tasks),
        "reference_policy": "one_prediction_per_window_scored_against_every_annotator",
        "setup_version": "tab-v4-target-context-native-labels-train-example",
        "max_gold_entities_per_group": collection.max_gold_entities,
        "chunking": {
            "unit": "whitespace_token",
            "max_words": config.tab_max_words,
            "overlap_words": config.tab_overlap_words,
            "mention_owner": "every_fully_containing_chunk; deduplicate_document_coordinates",
        },
        "tasks": [
            {
                "uid": task.uid,
                "document_uid": task.document_uid,
                "annotation_layer": task.annotation_layer,
                "source_start": task.source_start,
                "source_characters": len(task.source_text),
                "gold_entities": len(task.gold_entities),
            }
            for task in collection.tasks
        ],
    }


async def cli_main(config: CLIConfig) -> dict[str, float]:
    log_path = Path(config.log_path).expanduser() if config.log_path else _default_log_path(config)
    if log_path.exists() and any(log_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty log directory: {log_path}")
    log_path.mkdir(parents=True, exist_ok=True)

    metadata = _run_metadata(config, log_path)

    tab_reference_tasks = ()
    if config.dataset_name == "tab":
        collection = load_tab_dev_tasks(
            source_path=config.tab_source_path,
            max_documents=config.tab_max_documents,
            seed=config.sample_seed,
            max_words=config.tab_max_words,
            overlap_words=config.tab_overlap_words,
        )
        tab_reference_tasks = collection.tasks
        unique_tasks = {}
        for task in collection.tasks:
            unique_tasks.setdefault((task.document_uid, task.source_start), task)
        tasks = list(unique_tasks.values())
        _write_json(log_path / "manifest.json", _tab_manifest(collection, config))
    else:
        excluded_uids = FEWSHOT_TRAIN_UIDS if config.dataset_split == "train" else frozenset()
        load_count = config.sample_pool_size + len(excluded_uids)
        pool_tasks = load_openpii_tasks(
            cast(Literal["train", "validation"], config.dataset_split),
            max_examples=load_count,
            language=config.language,
            revision=config.dataset_revision,
            max_documents_scanned=config.max_documents_scanned,
        )
        pool_tasks = [task for task in pool_tasks if task.uid not in excluded_uids][
            : config.sample_pool_size
        ]
        if len(pool_tasks) != config.sample_pool_size:
            raise RuntimeError(
                f"Only {len(pool_tasks)} tasks remain after prompt-example exclusion; "
                f"expected {config.sample_pool_size}"
            )
        uid_filter_info = None
        if config.uid_filter_manifest is not None:
            filter_path = Path(config.uid_filter_manifest).expanduser()
            filter_manifest = json.loads(filter_path.read_text(encoding="utf-8"))
            allowed_uids = eligible_uids(filter_manifest)
            pool_tasks = [task for task in pool_tasks if task.uid in allowed_uids]
            uid_filter_info = {
                "path": str(filter_path),
                "sha256": hashlib.sha256(filter_path.read_bytes()).hexdigest(),
                "eligible_uids": len(allowed_uids),
                "matched_pool_uids": len(pool_tasks),
            }
            if len(pool_tasks) < config.max_examples:
                raise ValueError(
                    f"UID filter matched {len(pool_tasks)} pool tasks, fewer than "
                    f"max_examples={config.max_examples}"
                )
        if config.selection_method == "stratified":
            tasks = stratified_sample_tasks(
                pool_tasks,
                sample_size=config.max_examples,
                seed=config.sample_seed,
            )
        else:
            tasks = random.Random(config.sample_seed).sample(
                pool_tasks,
                config.max_examples,
            )
        _write_json(
            log_path / "manifest.json",
            _manifest(tasks, pool_tasks, config, uid_filter_info),
        )
    _write_json(log_path / "run_meta.json", metadata)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name, prefer_non_thinking=True
    )
    dataset = build_eval_dataset(
        tasks,
        model_name=config.tokenizer_name or config.model_name,
        renderer_name=renderer_name,
        batch_size=config.batch_size,
        samples_per_task=config.samples_per_task,
        max_turns=config.max_turns,
        prompt_version=config.prompt_version,
        tool_variant=config.tool_variant,
        second_turn_max_tokens=config.second_turn_max_tokens,
    )
    observer = NerEvalArtifactObserver(
        log_path=log_path,
        seed=config.sample_seed,
        error_sample_size=config.error_sample_size,
        turn_token_limits=(config.max_tokens, config.second_turn_max_tokens),
        tab_reference_tasks=tab_reference_tasks,
    )
    evaluator = RLTestSetEvaluator(
        dataset,
        max_tokens=config.max_tokens,
        name="base",
        num_groups_to_log=min(config.num_groups_to_log, len(tasks)),
        sample_timeout_sec=config.sample_timeout_sec,
        max_retries_per_trajectory=config.max_retries_per_trajectory,
        observer=observer,
    )

    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    credential: Any
    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key)
    else:
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential(exclude_managed_identity_credential=True)

    if config.verbose_http:
        import azure.ai.finetuningsessions._patch as patch_module

        patch_module.VERBOSE_HTTP = True

    client_kwargs: dict[str, Any] = {
        "endpoint": config.project_endpoint,
        "credential": credential,
    }
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]
    client = FineTuningSessionClient(**client_kwargs)
    session_id: str | None = None

    try:
        logger.info(
            "Creating evaluation-only session: model=%s examples=%d samples_per_task=%d",
            config.model_name,
            len(tasks),
            config.samples_per_task,
        )
        from_checkpoint = None
        if config.load_checkpoint_path is not None:
            source_session_id, checkpoint_id = _parse_checkpoint_path(
                config.load_checkpoint_path
            )
            from_checkpoint = FromCheckpoint(
                source_session_id=source_session_id,
                checkpoint_id=checkpoint_id,
            )
        session_id = await client.create_session(
            base_model=config.model_name,
            lora_config=LoRAConfig(rank=config.lora_rank, seed=config.lora_seed),
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=config.create_session_timeout_sec,
            training_type=normalize_training_type(config.training_type),
        )
        metadata["azure"]["session_id"] = session_id
        _write_json(log_path / "run_meta.json", metadata)

        tokenizer = tokenizer_utils.get_tokenizer(config.tokenizer_name or config.model_name)
        training_client = AzureSDKTrainingClient(client, session_id, tokenizer)
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            "base-eval"
        )

        html_path = log_path / "eval_base.html"
        policy = PromptSeededTokenCompleter(sampling_client, config)
        with logtree.init_trace("NER base-model evaluation", path=html_path):
            metrics = await evaluator.eval_token_completer(policy, step=0)
        if not observer.summary_metrics:
            raise RuntimeError("NER evaluation artifacts were not produced")
        metrics.update(observer.summary_metrics)
        add_micro_metrics(metrics)
        _write_json(log_path / "metrics.json", metrics)
        logger.info("Base-model evaluation complete: %s", json.dumps(metrics, sort_keys=True))
        return metrics
    finally:
        if session_id is not None:
            try:
                await client.close_session(session_id)
            except Exception:
                logger.exception("Failed to close evaluation session %s", session_id)
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