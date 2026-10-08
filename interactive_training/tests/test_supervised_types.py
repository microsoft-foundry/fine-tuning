"""Minimal tests for the supervised dataset builder type hierarchy."""

import chz
import pytest
from azure.ai.finetuningsessions.models import Datum

from interactive_training.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)


# --- Concrete stubs for testing the ABC contracts ---


class StubDataset(SupervisedDataset):
    def __init__(self, size: int = 10):
        self._size = size

    def get_batch(self, index: int) -> list[Datum]:
        return []

    def __len__(self) -> int:
        return self._size


@chz.chz
class StubBuilder(SupervisedDatasetBuilder):
    size: int = 5

    def build(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        return StubDataset(self.size), None


@chz.chz
class StubChatBuilder(ChatDatasetBuilder):
    def build(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        return StubDataset(42), StubDataset(8)


# --- Tests ---


def test_cannot_instantiate_abstract_builder():
    with pytest.raises(TypeError):
        SupervisedDatasetBuilder()  # type: ignore[abstract]


def test_cannot_instantiate_abstract_chat_builder():
    with pytest.raises(TypeError):
        ChatDatasetBuilder(  # type: ignore[abstract]
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="x",
                renderer_name="x",
                max_length=None,
                batch_size=1,
            )
        )


def test_stub_builder_build():
    builder = StubBuilder(size=3)
    train_ds, eval_ds = builder.build()
    assert isinstance(train_ds, SupervisedDataset)
    assert eval_ds is None
    assert len(train_ds) == 3


def test_stub_chat_builder_build():
    builder = StubChatBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="x",
            renderer_name="x",
            max_length=None,
            batch_size=1,
        )
    )
    train_ds, eval_ds = builder.build()
    assert isinstance(train_ds, SupervisedDataset)
    assert eval_ds is not None
    assert len(train_ds) == 42
    assert len(eval_ds) == 8


def test_dataset_get_batch():
    ds = StubDataset(10)
    batch = ds.get_batch(0)
    assert isinstance(batch, list)
