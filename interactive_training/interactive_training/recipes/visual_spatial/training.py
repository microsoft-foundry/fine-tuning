from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence

import chz
import datasets
from azure.ai.finetuningsessions.models import (
    Datum,
    ImageChunk,
    ModelInput,
    ModelInputChunk,
    SamplingParams,
)

from interactive_training.eval.evaluators import SamplingClientEvaluator
from interactive_training.image_processing_utils import get_image_processor
from interactive_training.recipes.visual_spatial import azure_openai_jsonl
from interactive_training.recipes.visual_spatial.data import (
    COUNT_CATEGORY_CODES,
    absolute_difference_answer,
    absolute_difference_sft_messages,
    absolute_difference_user_message,
    answers_match,
    count_category_answer,
    count_category_sft_messages,
    count_category_user_message,
    grouped_sft_messages,
    grouped_user_message,
    load_dataset,
    sft_messages,
    user_message,
)
from interactive_training.renderers import Renderer, TrainOnWhat, get_renderer, get_text_content
from interactive_training.rl.problem_env import ProblemGroupBuilder
from interactive_training.rl.types import Action, Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from interactive_training.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from interactive_training.supervised.types import ChatDatasetBuilder, SupervisedDataset
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.utils import file_utils

TaskVariant = Literal["single_image", "absolute_difference", "count_category"]


def _load_dataset(
    seed: int,
    data_path: str | None,
    image_cache_dir: str | None = None,
):
    if data_path is not None:
        return azure_openai_jsonl.load_dataset(
            data_path,
            seed=seed,
            image_cache_dir=image_cache_dir,
        )
    return load_dataset(seed=seed)


def _prompt_messages(row: dict) -> list:
    if "prompt_messages" in row:
        return azure_openai_jsonl.prompt_messages(row)
    return [user_message(row)]


def _sft_messages(row: dict) -> list:
    if "prompt_messages" in row:
        return azure_openai_jsonl.sft_messages(row)
    return sft_messages(row)


def _answers_match(row: dict, candidate: str, expected: str) -> bool:
    if "prompt_messages" in row:
        return azure_openai_jsonl.answers_match(candidate, expected)
    return answers_match(candidate, expected)


def _is_single_integer_answer(candidate: str) -> bool:
    stripped = candidate.strip()
    return len(stripped) == 1 and stripped.isascii() and stripped.isdigit()


def _smallest_image_indices(dataset, count: int) -> list[int]:
    return sorted(
        range(len(dataset)),
        key=lambda index: (
            dataset[index]["image"].size[0] * dataset[index]["image"].size[1],
            index,
        ),
    )[:count]


def _group_rows(
    dataset,
    target_index: int,
    image_count: int,
    context_indices: list[int] | None = None,
) -> list[dict]:
    if image_count < 1:
        raise ValueError(f"image_count must be at least 1, got {image_count}")
    if len(dataset) == 0:
        raise ValueError("cannot group images from an empty dataset")
    if image_count == 1:
        return [dataset[target_index]]
    candidates = context_indices or list(range(len(dataset)))
    context = [index for index in candidates if index != target_index]
    if not context:
        context = [target_index]
    context = [context[offset % len(context)] for offset in range(image_count - 1)]
    return [dataset[target_index], *[dataset[index] for index in context]]


def _group_target_with_context(
    target_row: dict,
    context_dataset,
    image_count: int,
    context_indices: list[int] | None = None,
) -> list[dict]:
    if image_count < 1:
        raise ValueError(f"image_count must be at least 1, got {image_count}")
    if image_count == 1:
        return [target_row]
    if len(context_dataset) == 0:
        raise ValueError("cannot group images from an empty context dataset")
    candidates = context_indices or list(range(len(context_dataset)))
    context = [
        context_dataset[candidates[offset % len(candidates)]]
        for offset in range(image_count - 1)
    ]
    return [target_row, *context]


def _absolute_difference_rows(dataset, target_index: int) -> list[dict]:
    if len(dataset) < 2:
        raise ValueError("absolute_difference requires at least two examples per split")
    partner_index = (target_index + len(dataset) // 2) % len(dataset)
    if partner_index == target_index:
        partner_index = (target_index + 1) % len(dataset)
    return [dataset[target_index], dataset[partner_index]]


def _balanced_count_category_indices(
    dataset,
    target_count: int,
    seed: int,
) -> list[int]:
    category_indices = {category: [] for category in COUNT_CATEGORY_CODES}
    for index, answer in enumerate(dataset["answer"]):
        category = str(answer).strip()
        if category in category_indices:
            category_indices[category].append(index)

    missing_categories = [
        category for category, indices in category_indices.items() if not indices
    ]
    if missing_categories:
        raise ValueError(
            "count_category training split is missing categories: "
            f"{missing_categories}"
        )

    rng = random.Random(seed)
    for indices in category_indices.values():
        rng.shuffle(indices)

    categories = sorted(category_indices)
    offsets = {category: 0 for category in categories}
    selected: list[int] = []
    while len(selected) < target_count:
        rng.shuffle(categories)
        for category in categories:
            indices = category_indices[category]
            selected.append(indices[offsets[category] % len(indices)])
            offsets[category] += 1
            if len(selected) == target_count:
                break
    return selected


@dataclass(frozen=True)
class DatasetPreflightSummary:
    split: str
    examples: int
    max_prompt_tokens: int
    max_total_tokens: int


def _model_input_payload_bytes(prompt: ModelInput) -> int:
    return max(
        sum(
            len(chunk.data)
            if isinstance(chunk, ImageChunk)
            else len(chunk.tokens) * 8
            for chunk in prompt.chunks
        ),
        1,
    )


class _PreparedPromptCache:
    def __init__(self, max_bytes: int):
        if max_bytes < 0:
            raise ValueError("prompt cache size must not be negative")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._entries: OrderedDict[int, tuple[ModelInput, int]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_create(
        self,
        key: int,
        factory: Callable[[], ModelInput],
    ) -> ModelInput:
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                self.hits += 1
                return cached[0]
            self.misses += 1

        prompt = factory()
        prompt_bytes = _model_input_payload_bytes(prompt)
        if self.max_bytes == 0 or prompt_bytes > self.max_bytes:
            return prompt

        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                return cached[0]
            while self._entries and self.current_bytes + prompt_bytes > self.max_bytes:
                _, (_, evicted_bytes) = self._entries.popitem(last=False)
                self.current_bytes -= evicted_bytes
                self.evictions += 1
            self._entries[key] = (prompt, prompt_bytes)
            self.current_bytes += prompt_bytes
        return prompt


def _example_id(row: dict) -> str:
    identity = f"{row.get('question', '')}\0{row.get('answer', '')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _validate_prompt_budget(
    prompt: ModelInput,
    *,
    max_tokens: int,
    model_context_length: int,
    split: str,
    example_index: int,
) -> tuple[int, int]:
    image_chunks = [
        chunk for chunk in prompt.chunks if isinstance(chunk, ImageChunk)
    ]
    if len(image_chunks) > azure_openai_jsonl.MAX_IMAGES_PER_EXAMPLE:
        raise ValueError(
            f"{split} example {example_index} contains {len(image_chunks)} images; "
            f"at most {azure_openai_jsonl.MAX_IMAGES_PER_EXAMPLE} are allowed"
        )
    image_payload_bytes = sum(len(chunk.data) for chunk in image_chunks)
    if image_payload_bytes > azure_openai_jsonl.MAX_TOTAL_IMAGE_BYTES_PER_EXAMPLE:
        raise ValueError(
            f"{split} example {example_index} contains {image_payload_bytes} bytes "
            "of normalized image data; at most "
            f"{azure_openai_jsonl.MAX_TOTAL_IMAGE_BYTES_PER_EXAMPLE} are allowed"
        )
    chunk_lengths = [
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in prompt.chunks
    ]
    prompt_tokens = sum(chunk_lengths)
    total_tokens = prompt_tokens + max_tokens
    if total_tokens > model_context_length:
        excess_tokens = total_tokens - model_context_length
        excess_unit = "token" if excess_tokens == 1 else "tokens"
        max_supported_prompt_tokens = max(model_context_length - max_tokens, 0)
        image_token_lengths = [
            chunk.length for chunk in prompt.chunks if isinstance(chunk, ImageChunk)
        ]
        suggestions = [
            f"remove at least {excess_tokens} {excess_unit} from this example",
            "shorten its text or split it into multiple examples",
        ]
        image_details = ""
        if image_token_lengths:
            image_tokens = sum(image_token_lengths)
            context_image_lengths = sorted(image_token_lengths[1:], reverse=True)
            removed_tokens = 0
            images_to_remove = 0
            for image_tokens_to_remove in context_image_lengths:
                removed_tokens += image_tokens_to_remove
                images_to_remove += 1
                if removed_tokens >= excess_tokens:
                    break
            image_details = (
                f" The prompt contains {len(image_token_lengths)} images consuming "
                f"{image_tokens} tokens."
            )
            if removed_tokens >= excess_tokens:
                suggestions.append(
                    f"remove approximately {images_to_remove} of the largest context "
                    "images at their current processed sizes"
                )
            suggestions.append("reduce image resolution")
        available_generation_tokens = model_context_length - prompt_tokens
        if available_generation_tokens >= 0:
            suggestions.append(
                f"reduce max_tokens from {max_tokens} to at most "
                f"{available_generation_tokens}"
            )
        else:
            suggestions.append(
                "reducing max_tokens alone is insufficient because the prompt itself "
                "exceeds the context window"
            )
        raise ValueError(
            f"{split} example {example_index} requires {prompt_tokens} prompt tokens "
            f"+ {max_tokens} generation tokens = {total_tokens}, exceeding the model "
            f"context length of {model_context_length} tokens by {excess_tokens}. "
            f"With max_tokens={max_tokens}, the prompt may contain at most "
            f"{max_supported_prompt_tokens} tokens.{image_details} Suggested changes: "
            f"{'; '.join(suggestions)}."
        )
    return prompt_tokens, total_tokens


@chz.chz
class VisualSpatialSftBuilder(ChatDatasetBuilder):
    max_train_examples: int | None = None
    seed: int = 0
    images_per_example: int = 1
    data_path: str | None = None
    image_cache_dir: str | None = None
    task_variant: TaskVariant = "single_image"

    @property
    def renderer(self) -> Renderer:
        renderer = get_renderer(
            self.common_config.renderer_name,
            self.tokenizer,
            image_processor=get_image_processor(
                self.common_config.model_name_for_tokenizer
            ),
        )
        renderer.image_cache_dir = self.image_cache_dir
        return renderer

    def build(self) -> tuple[SupervisedDataset, None]:
        if self.task_variant == "absolute_difference" and self.images_per_example != 2:
            raise ValueError("absolute_difference requires images_per_example=2")
        if self.task_variant == "count_category" and self.images_per_example != 1:
            raise ValueError("count_category requires images_per_example=1")
        if self.data_path is not None and self.task_variant != "single_image":
            raise ValueError("data_path supports only task_variant=single_image")
        if self.data_path is not None and self.images_per_example != 1:
            raise ValueError("data_path requires images_per_example=1")
        train_dataset, _ = _load_dataset(
            self.seed,
            self.data_path,
            self.image_cache_dir,
        )
        target_count = (
            len(train_dataset)
            if self.max_train_examples is None
            else min(self.max_train_examples, len(train_dataset))
        )
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        source_dataset = train_dataset
        context_indices = (
            _smallest_image_indices(source_dataset, self.images_per_example)
            if self.task_variant == "single_image" and self.images_per_example > 1
            else None
        )
        target_indices = (
            _balanced_count_category_indices(source_dataset, target_count, self.seed)
            if self.task_variant == "count_category"
            else list(range(target_count))
        )
        indexed_dataset = datasets.Dataset.from_dict(
            {"target_index": target_indices}
        )

        def map_fn(row: dict) -> Datum:
            target_index = int(row["target_index"])
            if self.task_variant == "absolute_difference":
                messages = absolute_difference_sft_messages(
                    _absolute_difference_rows(source_dataset, target_index)
                )
            elif self.task_variant == "count_category":
                messages = count_category_sft_messages(source_dataset[target_index])
            else:
                grouped_rows = _group_rows(
                    source_dataset,
                    target_index,
                    self.images_per_example,
                    context_indices,
                )
                messages = (
                    _sft_messages(grouped_rows[0])
                    if self.images_per_example == 1
                    else grouped_sft_messages(grouped_rows)
                )
            return conversation_to_datum(
                messages,
                self.renderer,
                self.common_config.max_length,
                train_on_what,
                self.common_config.model_context_length,
                self.common_config.fail_on_truncation,
            )

        return (
            SupervisedDatasetFromHFDataset(
                indexed_dataset,
                batch_size=self.common_config.batch_size,
                map_fn=map_fn,
            ),
            None,
        )


class VisualSpatialEnv(Env):
    def __init__(
        self,
        row: dict,
        renderer: Renderer,
        image_rows: list[dict] | None = None,
        prepared_prompt: ModelInput | None = None,
        log_examples: bool = False,
        expected_answer: str | None = None,
        max_tokens: int = 512,
    ):
        self.row = row
        self.renderer = renderer
        self.image_rows = image_rows
        self.prepared_prompt = prepared_prompt
        self.log_examples = log_examples
        self.expected_answer = expected_answer
        self.max_tokens = max_tokens

    async def initial_observation(self):
        return (
            (
                self.prepared_prompt
                if self.prepared_prompt is not None
                else self.renderer.build_generation_prompt(
                    _prompt_messages(self.row)
                    if self.image_rows is None
                    else [grouped_user_message(self.image_rows)]
                )
            ),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: Action) -> StepResult:
        message, parsed = self.renderer.parse_response(action)
        prediction = get_text_content(message)
        expected = self.expected_answer or str(self.row["answer"])
        correct = float(
            parsed and _answers_match(self.row, prediction, expected)
        )
        logs = {"example_id": _example_id(self.row)}
        if self.log_examples:
            logs.update(
                question=str(self.row["question"]),
                expected=expected,
                prediction=prediction,
            )
        return StepResult(
            reward=correct,
            episode_done=True,
            next_observation=ModelInput(chunks=[]),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "correct": correct,
                "parsed": float(parsed),
                "single_integer_answer": float(
                    parsed and _is_single_integer_answer(prediction)
                ),
                "truncated": float(
                    len(action) >= max(1, self.max_tokens - 4)
                ),
            },
            logs=logs,
        )


class VisualSpatialRLDataset(RLDataset):
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        dataset_name: str,
        images_per_example: int = 1,
        max_examples: int | None = None,
        context_dataset=None,
        num_epochs: int = 1,
        prompt_cache_max_mb: int = 128,
        log_examples: bool = False,
        task_variant: TaskVariant = "single_image",
        max_tokens: int = 512,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if group_size < 1:
            raise ValueError(f"group_size must be at least 1, got {group_size}")
        if not 1 <= images_per_example <= azure_openai_jsonl.MAX_IMAGES_PER_EXAMPLE:
            raise ValueError(
                "images_per_example must be between 1 and "
                f"{azure_openai_jsonl.MAX_IMAGES_PER_EXAMPLE}, got {images_per_example}"
            )
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be at least 1, got {num_epochs}")
        if task_variant == "absolute_difference" and images_per_example != 2:
            raise ValueError("absolute_difference requires images_per_example=2")
        if task_variant == "count_category" and images_per_example != 1:
            raise ValueError("count_category requires images_per_example=1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.dataset_name = dataset_name
        self.images_per_example = images_per_example
        self.context_dataset = context_dataset or dataset
        self.num_epochs = num_epochs
        self.prompt_cache = _PreparedPromptCache(prompt_cache_max_mb * 1024 * 1024)
        self.log_examples = log_examples
        self.task_variant = task_variant
        self.max_tokens = max_tokens
        self.num_examples = (
            len(dataset) if max_examples is None else min(max_examples, len(dataset))
        )
        if self.num_examples < 1:
            raise ValueError(f"{dataset_name} must contain at least one example")
        self.context_indices = (
            _smallest_image_indices(self.context_dataset, images_per_example)
            if task_variant == "single_image" and images_per_example > 1
            else None
        )

    def _rows_and_expected(self, target_index: int) -> tuple[list[dict], str]:
        if self.task_variant == "absolute_difference":
            rows = _absolute_difference_rows(self.dataset, target_index)
            return rows, absolute_difference_answer(rows)
        if self.task_variant == "count_category":
            row = self.dataset[target_index]
            return [row], count_category_answer(row)
        rows = (
            [self.dataset[target_index]]
            if self.images_per_example == 1
            else _group_target_with_context(
                self.dataset[target_index],
                self.context_dataset,
                self.images_per_example,
                self.context_indices,
            )
        )
        return rows, str(self.dataset[target_index]["answer"]).strip()

    def _prepare_prompt(self, target_index: int) -> ModelInput:
        def build() -> ModelInput:
            image_rows, _ = self._rows_and_expected(target_index)
            return self.renderer.build_generation_prompt(
                [absolute_difference_user_message(image_rows)]
                if self.task_variant == "absolute_difference"
                else [count_category_user_message(image_rows[0])]
                if self.task_variant == "count_category"
                else (
                    _prompt_messages(self.dataset[target_index])
                    if self.images_per_example == 1
                    else [grouped_user_message(image_rows)]
                )
            )

        return self.prompt_cache.get_or_create(target_index, build)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batches_per_epoch = math.ceil(self.num_examples / self.batch_size)
        epoch_batch_index = index % batches_per_epoch
        start = epoch_batch_index * self.batch_size
        end = min(start + self.batch_size, self.num_examples)
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    VisualSpatialEnv,
                    self.dataset[target_index],
                    self.renderer,
                    prepared_prompt=self._prepare_prompt(target_index),
                    log_examples=self.log_examples,
                    expected_answer=self._rows_and_expected(target_index)[1],
                    max_tokens=self.max_tokens,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for target_index in range(start, end)
        ]

    def __len__(self) -> int:
        return math.ceil(self.num_examples / self.batch_size) * self.num_epochs

    async def preflight(
        self,
        *,
        max_tokens: int,
        model_context_length: int,
    ) -> DatasetPreflightSummary:
        max_prompt_tokens = 0
        max_total_tokens = 0
        for example_index in range(self.num_examples):
            prompt = self._prepare_prompt(example_index)
            prompt_tokens, total_tokens = _validate_prompt_budget(
                prompt,
                max_tokens=max_tokens,
                model_context_length=model_context_length,
                split=self.dataset_name,
                example_index=example_index,
            )
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            max_total_tokens = max(max_total_tokens, total_tokens)
        return DatasetPreflightSummary(
            split=self.dataset_name,
            examples=self.num_examples,
            max_prompt_tokens=max_prompt_tokens,
            max_total_tokens=max_total_tokens,
        )


@chz.chz
class VisualSpatialRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name: str
    tokenizer_name: str
    renderer_name: str
    group_size: int
    seed: int = 0
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    images_per_example: int = 1
    num_epochs: int = 1
    data_path: str | None = None
    prompt_cache_max_mb: int = 128
    image_cache_dir: str | None = None
    log_examples: bool = False
    task_variant: TaskVariant = "single_image"
    max_tokens: int = 512

    async def __call__(self) -> tuple[VisualSpatialRLDataset, VisualSpatialRLDataset]:
        if self.task_variant == "absolute_difference" and self.images_per_example != 2:
            raise ValueError("absolute_difference requires images_per_example=2")
        if self.task_variant == "count_category" and self.images_per_example != 1:
            raise ValueError("count_category requires images_per_example=1")
        if self.data_path is not None and self.task_variant != "single_image":
            raise ValueError("data_path supports only task_variant=single_image")
        if self.data_path is not None and self.images_per_example != 1:
            raise ValueError("data_path requires images_per_example=1")
        train_dataset, test_dataset = _load_dataset(
            self.seed,
            self.data_path,
            self.image_cache_dir,
        )
        train_max_examples = self.max_train_examples
        if self.task_variant == "count_category":
            target_count = (
                len(train_dataset)
                if self.max_train_examples is None
                else min(self.max_train_examples, len(train_dataset))
            )
            train_dataset = train_dataset.select(
                _balanced_count_category_indices(
                    train_dataset,
                    target_count,
                    self.seed,
                )
            )
            train_max_examples = None
        renderer = get_renderer(
            self.renderer_name,
            get_tokenizer(self.tokenizer_name),
            image_processor=get_image_processor(self.model_name),
        )
        renderer.image_cache_dir = self.image_cache_dir
        return (
            VisualSpatialRLDataset(
                train_dataset,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                dataset_name="visual_spatial/train",
                images_per_example=self.images_per_example,
                max_examples=train_max_examples,
                num_epochs=self.num_epochs,
                prompt_cache_max_mb=self.prompt_cache_max_mb,
                log_examples=self.log_examples,
                task_variant=self.task_variant,
                max_tokens=self.max_tokens,
            ),
            VisualSpatialRLDataset(
                test_dataset,
                batch_size=self.batch_size,
                group_size=1,
                renderer=renderer,
                dataset_name="visual_spatial/test",
                images_per_example=self.images_per_example,
                max_examples=self.max_test_examples,
                context_dataset=train_dataset,
                prompt_cache_max_mb=0,
                log_examples=self.log_examples,
                task_variant=self.task_variant,
                max_tokens=self.max_tokens,
            ),
        )


@dataclass
class VisualSpatialAccuracyEvaluator(SamplingClientEvaluator):
    model_name: str
    renderer_name: str
    seed: int
    output_dir: str
    max_examples: int | None = None
    images_per_example: int = 1
    max_tokens: int = 512
    concurrency: int = 16
    data_path: str | None = None
    prompt_cache_max_mb: int = 128
    image_cache_dir: str | None = None
    log_examples: bool = False
    task_variant: TaskVariant = "single_image"

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError(
                f"max_examples must be at least 1, got {self.max_examples}"
            )
        if self.data_path is not None and self.images_per_example != 1:
            raise ValueError("data_path requires images_per_example=1")
        if self.task_variant == "absolute_difference" and self.images_per_example != 2:
            raise ValueError("absolute_difference requires images_per_example=2")
        if self.task_variant == "count_category" and self.images_per_example != 1:
            raise ValueError("count_category requires images_per_example=1")
        if self.data_path is not None and self.task_variant != "single_image":
            raise ValueError("data_path supports only task_variant=single_image")
        train_dataset, test_dataset = _load_dataset(
            self.seed,
            self.data_path,
            self.image_cache_dir,
        )
        self.context_dataset = train_dataset
        self.pairing_dataset = test_dataset
        self.dataset = test_dataset
        if self.max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(self.max_examples, len(self.dataset)))
            )
        if len(self.dataset) < 1:
            raise ValueError(
                "visual_spatial/evaluation must contain at least one example"
            )
        self.context_indices = (
            _smallest_image_indices(self.context_dataset, self.images_per_example)
            if self.task_variant == "single_image" and self.images_per_example > 1
            else None
        )
        self.renderer = get_renderer(
            self.renderer_name,
            get_tokenizer(self.model_name),
            image_processor=get_image_processor(self.model_name),
        )
        self.renderer.image_cache_dir = self.image_cache_dir
        self.prompt_cache = _PreparedPromptCache(
            self.prompt_cache_max_mb * 1024 * 1024
        )
        self.call_index = 0

    def _build_prompt(self, index: int, row: dict) -> ModelInput:
        def build() -> ModelInput:
            if self.task_variant == "absolute_difference":
                messages = [
                    absolute_difference_user_message(
                        _absolute_difference_rows(self.pairing_dataset, index)
                    )
                ]
            elif self.task_variant == "count_category":
                messages = [count_category_user_message(row)]
            else:
                messages = (
                    _prompt_messages(row)
                    if self.images_per_example == 1
                    else [
                        grouped_user_message(
                            _group_target_with_context(
                                row,
                                self.context_dataset,
                                self.images_per_example,
                                self.context_indices,
                            )
                        )
                    ]
                )
            return self.renderer.build_generation_prompt(messages)

        return self.prompt_cache.get_or_create(index, build)

    def preflight(self, *, model_context_length: int) -> DatasetPreflightSummary:
        max_prompt_tokens = 0
        max_total_tokens = 0
        for example_index, row in enumerate(self.dataset):
            prompt_tokens, total_tokens = _validate_prompt_budget(
                self._build_prompt(example_index, row),
                max_tokens=self.max_tokens,
                model_context_length=model_context_length,
                split="visual_spatial/evaluation",
                example_index=example_index,
            )
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            max_total_tokens = max(max_total_tokens, total_tokens)
        return DatasetPreflightSummary(
            split="visual_spatial/evaluation",
            examples=len(self.dataset),
            max_prompt_tokens=max_prompt_tokens,
            max_total_tokens=max_total_tokens,
        )

    async def __call__(self, sampling_client) -> dict[str, float]:
        if not hasattr(sampling_client, "sample_async"):
            sampling_client = await sampling_client.save_weights_and_get_sampling_client_async(
                f"visual_spatial_eval_{self.call_index}"
            )
        predictions: list[dict | None] = [None] * len(self.dataset)
        work_queue: asyncio.Queue[int | None] = asyncio.Queue(
            maxsize=self.concurrency * 2
        )

        async def evaluate(index: int) -> dict:
            row, prompt = await asyncio.to_thread(
                lambda: (
                    self.dataset[index],
                    self._build_prompt(index, self.dataset[index]),
                )
            )
            result = await sampling_client.sample_async(
                prompt,
                sampling_params=SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    seed=self.seed + index,
                    stop_criteria=self.renderer.get_stop_sequences(),
                ),
            )
            sequence = result.sequences[0]
            message, parsed = self.renderer.parse_response(sequence.tokens)
            prediction = get_text_content(message)
            expected = (
                absolute_difference_answer(
                    _absolute_difference_rows(self.pairing_dataset, index)
                )
                if self.task_variant == "absolute_difference"
                else count_category_answer(row)
                if self.task_variant == "count_category"
                else str(row["answer"]).strip()
            )
            completion_tokens = len(sequence.tokens)
            prediction_record = {
                "index": index,
                "example_id": _example_id(row),
                "parsed": parsed,
                "correct": bool(
                    parsed and _answers_match(row, prediction, expected)
                ),
                "completion_tokens": completion_tokens,
                "hit_max_tokens": completion_tokens >= self.max_tokens,
                "single_integer_answer": bool(
                    parsed and _is_single_integer_answer(prediction)
                ),
            }
            if self.log_examples:
                prediction_record.update(
                    question=str(row["question"]).strip(),
                    expected=expected,
                    prediction=prediction,
                )
            return prediction_record

        async def produce() -> None:
            for index in range(len(self.dataset)):
                await work_queue.put(index)
            for _ in range(min(self.concurrency, len(self.dataset))):
                await work_queue.put(None)

        async def worker() -> None:
            while True:
                index = await work_queue.get()
                if index is None:
                    return
                predictions[index] = await evaluate(index)

        worker_count = min(self.concurrency, len(self.dataset))
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(produce())
            for _ in range(worker_count):
                task_group.create_task(worker())

        completed_predictions = [
            prediction for prediction in predictions if prediction is not None
        ]
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, f"visual_spatial_eval_{self.call_index}.jsonl"
        )
        output_descriptor = os.open(
            output_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(output_descriptor, "w", encoding="utf-8") as output_file:
            file_utils.set_private_file_permissions(output_descriptor, output_path)
            for prediction in completed_predictions:
                output_file.write(json.dumps(prediction) + "\n")
        self.call_index += 1
        correct = sum(prediction["correct"] for prediction in completed_predictions)
        parsed = sum(prediction["parsed"] for prediction in completed_predictions)
        hit_max_tokens = sum(
            prediction["hit_max_tokens"] for prediction in completed_predictions
        )
        single_integer_answers = sum(
            prediction["single_integer_answer"]
            for prediction in completed_predictions
        )
        example_count = len(completed_predictions)
        return {
            "test/visual_spatial_accuracy": correct / example_count,
            "test/visual_spatial_correct": float(correct),
            "test/visual_spatial_examples": float(example_count),
            "test/visual_spatial_parse_rate": parsed / example_count,
            "test/visual_spatial_frac_at_max_tokens": (
                hit_max_tokens / example_count
            ),
            "test/visual_spatial_single_integer_answer_rate": (
                single_integer_answers / example_count
            ),
            "test/visual_spatial_prompt_cache_hits": float(self.prompt_cache.hits),
            "test/visual_spatial_prompt_cache_misses": float(self.prompt_cache.misses),
            "test/visual_spatial_prompt_cache_evictions": float(
                self.prompt_cache.evictions
            ),
            "test/visual_spatial_prompt_cache_bytes": float(
                self.prompt_cache.current_bytes
            ),
        }