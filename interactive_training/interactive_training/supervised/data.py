"""
Supervised learning dataset implementations from HuggingFace datasets.
"""

import json
from typing import Any, Callable

import chz
import datasets
import fsspec
from azure.ai.finetuningsessions.models import Datum, ImageChunk, ModelInput, ModelInputChunk

from interactive_training.renderers import Message, Renderer, TrainOnWhat
from interactive_training.supervised.common import datum_from_model_input_weights
from interactive_training.supervised.types import ChatDatasetBuilder, SupervisedDataset


def _format_example_overflow(
    model_input: ModelInput,
    *,
    required_tokens: int,
    token_limit: int,
    limit_name: str,
) -> str:
    excess_tokens = required_tokens - token_limit
    excess_unit = "token" if excess_tokens == 1 else "tokens"
    image_token_lengths = [
        chunk.length for chunk in model_input.chunks if isinstance(chunk, ImageChunk)
    ]
    image_details = ""
    suggestions = [
        f"remove at least {excess_tokens} {excess_unit} from this example",
        "shorten its text or split it into multiple examples",
    ]
    if image_token_lengths:
        image_unit = "image" if len(image_token_lengths) == 1 else "images"
        image_details = (
            f" It contains {len(image_token_lengths)} {image_unit} consuming "
            f"{sum(image_token_lengths)} tokens."
        )
        context_image_lengths = sorted(image_token_lengths[1:], reverse=True)
        removed_tokens = 0
        images_to_remove = 0
        for image_tokens_to_remove in context_image_lengths:
            removed_tokens += image_tokens_to_remove
            images_to_remove += 1
            if removed_tokens >= excess_tokens:
                break
        if removed_tokens >= excess_tokens:
            suggestions.append(
                f"remove approximately {images_to_remove} of the largest context "
                "images at their current processed sizes"
            )
        suggestions.append("reduce image resolution")
    return (
        f"example requires {required_tokens} tokens, exceeding {limit_name} of "
        f"{token_limit} tokens by {excess_tokens}.{image_details} Suggested changes: "
        f"{'; '.join(suggestions)}."
    )


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    model_context_length: int | None = None,
    fail_on_truncation: bool = False,
) -> Datum:
    """Common function to process a list of messages into a Datum."""
    model_input, weights = renderer.build_supervised_example(
        conversation, train_on_what=train_on_what
    )
    rendered_length = sum(
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in model_input.chunks
    )
    if (
        fail_on_truncation
        and max_length is not None
        and rendered_length > max_length
    ):
        raise ValueError(
            _format_example_overflow(
                model_input,
                required_tokens=rendered_length,
                token_limit=max_length,
                limit_name="the configured training max_length",
            )
        )
    datum = datum_from_model_input_weights(model_input, weights, max_length)
    training_length = sum(
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in datum.model_input.chunks
    )
    if model_context_length is not None and training_length > model_context_length:
        raise ValueError(
            _format_example_overflow(
                datum.model_input,
                required_tokens=training_length,
                token_limit=model_context_length,
                limit_name="the model context length",
            )
        )
    return datum


def _one_of(a: Any, b: Any) -> bool:
    return (a is not None and b is None) or (a is None and b is not None)


class SupervisedDatasetFromHFDataset(SupervisedDataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        batch_size: int,
        map_fn: Callable[[dict], Datum] | None = None,
        flatmap_fn: Callable[[dict], list[Datum]] | None = None,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset
        self.shuffle_dataset = (
            hf_dataset  # Keep a reference to the original dataset to avoid statefulness
        )
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn

    def get_batch(self, index: int) -> list[Datum]:
        first_row_index = index * self.batch_size
        rows = self.shuffle_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        if self.map_fn is not None:
            mapped = []
            for offset, row in enumerate(rows.to_list()):
                try:
                    mapped.append(self.map_fn(row))
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset example {first_row_index + offset} is invalid: {exc}"
                    ) from exc
            return mapped
        else:
            assert self.flatmap_fn is not None
            mapped = []
            for offset, row in enumerate(rows.to_list()):
                try:
                    mapped.extend(self.flatmap_fn(row))
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset example {first_row_index + offset} is invalid: {exc}"
                    ) from exc
            return mapped

    def set_epoch(self, seed: int = 0):
        self.shuffle_dataset = self.hf_dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.hf_dataset) // self.batch_size


@chz.chz
class FromConversationFileBuilder(ChatDatasetBuilder):
    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    def build(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL file
        conversations = []
        with fsspec.open(self.file_path, mode="rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise ValueError(
                        f"Each line in the JSONL file must contain a 'messages' field. Got: {data.keys()}"
                    )
                conversations.append(data)

        # Create HuggingFace dataset from the loaded data
        dataset = datasets.Dataset.from_list(conversations)

        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split into train and test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            # If test_size is 0 or dataset is too small, use all data for training
            train_ds = dataset
            test_ds = None

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Define mapping function
        def map_fn(row: dict) -> Datum:
            return conversation_to_datum(
                row["messages"],
                self.renderer,
                self.common_config.max_length,
                train_on_what,
                self.common_config.model_context_length,
                self.common_config.fail_on_truncation,
            )

        # Create supervised dataset
        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        # Create evaluator if we have test data
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        else:
            test_dataset = None

        return supervised_dataset, test_dataset
