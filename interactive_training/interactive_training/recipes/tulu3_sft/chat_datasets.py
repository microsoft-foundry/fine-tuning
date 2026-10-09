"""
Datasets for supervised learning (SFT) that use chat-formatted data, which we
convert to tokens using a Renderer.
"""

import logging
from typing import cast

import chz
import datasets
from azure.ai.finetuningsessions.models import Datum

from interactive_training.renderers import TrainOnWhat
from interactive_training.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from interactive_training.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


@chz.chz
class Tulu3Builder(ChatDatasetBuilder):
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    seed: int = 0

    def build(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=self.seed)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        if self.max_test_examples is not None:
            test_ds = test_ds.take(self.max_test_examples)
        if self.max_train_examples is not None:
            train_ds = train_ds.take(self.max_train_examples)

        # Use train_on_what from common_config if provided, otherwise default to LAST_ASSISTANT_MESSAGE
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        def map_fn(row: dict) -> Datum:
            return conversation_to_datum(
                row["messages"],
                self.renderer,
                self.common_config.max_length,
                train_on_what,
                model_context_length=self.common_config.model_context_length,
                fail_on_truncation=self.common_config.fail_on_truncation,
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
