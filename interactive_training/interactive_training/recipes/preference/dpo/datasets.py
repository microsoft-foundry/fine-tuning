import logging
import re
from dataclasses import dataclass
from typing import Literal, cast

import chz
import datasets
from azure.ai.finetuningsessions.models import Datum

from interactive_training.renderers import Message
from interactive_training.supervised.common import datum_from_model_input_weights
from interactive_training.supervised.data import SupervisedDatasetFromHFDataset
from interactive_training.supervised.types import (
    ChatDatasetBuilder,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Comparison:
    prompt_conversation: list[Message]
    completion_a: list[Message]
    completion_b: list[Message]


@dataclass(frozen=True)
class LabeledComparison:
    comparison: Comparison
    label: Literal["A", "B"]


@chz.chz
class ComparisonDatasetBuilder:
    def get_train_and_test_datasets(
        self,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        raise NotImplementedError

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        raise NotImplementedError


def _parse_hhh_conversation(text: str) -> list[Message]:
    messages: list[Message] = []
    parts = re.split(r"(Human:|Assistant:)", text)
    if not parts[0].strip():
        parts = parts[1:]
    for index in range(0, len(parts), 2):
        if index + 1 >= len(parts):
            continue
        delimiter = parts[index].strip()
        if delimiter not in {"Human:", "Assistant:"}:
            continue
        role = "user" if delimiter == "Human:" else "assistant"
        messages.append({"role": role, "content": parts[index + 1].strip()})
    return messages


@chz.chz
class HHHComparisonBuilder(ComparisonDatasetBuilder):
    test_size: int = 1024
    max_train_examples: int | None = None

    def get_train_and_test_datasets(
        self,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = cast(
            datasets.DatasetDict,
            datasets.load_dataset("Anthropic/hh-rlhf"),
        )
        train = dataset["train"].shuffle(seed=0)
        if self.max_train_examples is not None:
            train = train.take(self.max_train_examples)
        test = dataset["test"].shuffle(seed=0).take(self.test_size)
        return train, test

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        chosen = _parse_hhh_conversation(example["chosen"])
        rejected = _parse_hhh_conversation(example["rejected"])
        if not chosen or len(chosen) != len(rejected):
            return None
        if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
            return None
        if chosen[:-1] != rejected[:-1] or chosen[-1] == rejected[-1]:
            return None
        return LabeledComparison(
            comparison=Comparison(chosen[:-1], [chosen[-1]], [rejected[-1]]),
            label="A",
        )


@chz.chz
class HelpSteer3ComparisonBuilder(ComparisonDatasetBuilder):
    test_size: int = 1024
    max_train_examples: int | None = None

    def get_train_and_test_datasets(
        self,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = cast(
            datasets.DatasetDict,
            datasets.load_dataset("nvidia/HelpSteer3", "preference"),
        )
        train = dataset["train"].shuffle(seed=0)
        if self.max_train_examples is not None:
            train = train.take(self.max_train_examples)
        test = dataset["validation"].shuffle(seed=0).take(self.test_size)
        return train, test

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        preference = example["overall_preference"]
        if preference == 0:
            return None
        return LabeledComparison(
            comparison=Comparison(
                prompt_conversation=example["context"],
                completion_a=[{"role": "assistant", "content": example["response1"]}],
                completion_b=[{"role": "assistant", "content": example["response2"]}],
            ),
            label="A" if preference < 0 else "B",
        )


@chz.chz
class UltraFeedbackComparisonBuilder(ComparisonDatasetBuilder):
    test_size: int = 1024
    max_train_examples: int | None = None

    def get_train_and_test_datasets(
        self,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = cast(
            datasets.Dataset,
            datasets.load_dataset(
                "argilla/ultrafeedback-binarized-preferences", split="train"
            ),
        ).shuffle(seed=0)
        test = dataset.take(self.test_size)
        train = dataset.skip(self.test_size)
        if self.max_train_examples is not None:
            train = train.take(self.max_train_examples)
        return train, test

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison:
        return LabeledComparison(
            comparison=Comparison(
                prompt_conversation=[
                    {"role": "user", "content": example["instruction"]}
                ],
                completion_a=[
                    {"role": "assistant", "content": example["chosen_response"]}
                ],
                completion_b=[
                    {"role": "assistant", "content": example["rejected_response"]}
                ],
            ),
            label="A",
        )


@chz.chz
class DPODatasetBuilder(ChatDatasetBuilder):
    comparison_builder: ComparisonDatasetBuilder

    def build(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        if self.common_config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        train, test = self.comparison_builder.get_train_and_test_datasets()
        renderer = self.renderer

        def example_to_pair(example: dict) -> list[Datum]:
            labeled = self.comparison_builder.example_to_labeled_comparison(example)
            if labeled is None:
                logger.warning("Skipping malformed preference example")
                return []
            comparison = labeled.comparison
            chosen = (
                comparison.completion_a
                if labeled.label == "A"
                else comparison.completion_b
            )
            rejected = (
                comparison.completion_b
                if labeled.label == "A"
                else comparison.completion_a
            )

            def render(completion: list[Message]) -> Datum:
                model_input, weights = renderer.build_supervised_example(
                    [*comparison.prompt_conversation, *completion]
                )
                return datum_from_model_input_weights(
                    model_input, weights, self.common_config.max_length
                )

            pair = [render(chosen), render(rejected)]
            if any(not any(datum.loss_fn_inputs["weights"].data) for datum in pair):
                logger.warning(
                    "Skipping preference pair with no response tokens after truncation"
                )
                return []
            return pair

        test_dataset = (
            SupervisedDatasetFromHFDataset(
                test, batch_size=len(test), flatmap_fn=example_to_pair
            )
            if test is not None and len(test) > 0
            else None
        )
        return (
            SupervisedDatasetFromHFDataset(
                train,
                batch_size=self.common_config.batch_size,
                flatmap_fn=example_to_pair,
            ),
            test_dataset,
        )
