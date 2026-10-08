from types import SimpleNamespace

import datasets
import pytest
import torch
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.image_types import ImageChunk
from interactive_training.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from interactive_training.supervised.train import prepare_datasets


def _model_input_length(model_input):
    return sum(
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in model_input.chunks
    )


class _Renderer:
    def __init__(self):
        self.model_input = ModelInput(
            chunks=[
                ModelInputChunk(tokens=[1, 2]),
                ImageChunk(
                    data=b"\xff\xd8\xffjpeg",
                    format="jpeg",
                    expected_tokens=6,
                ),
                ModelInputChunk(tokens=[3, 4]),
            ]
        )

    def build_supervised_example(self, conversation, *, train_on_what):
        del conversation, train_on_what
        return self.model_input, torch.ones(10)


def test_exact_multimodal_context_boundary_passes():
    datum = conversation_to_datum(
        [], _Renderer(), max_length=None, model_context_length=9
    )

    assert _model_input_length(datum.model_input) == 9


def test_multimodal_context_one_token_over_fails():
    with pytest.raises(
        ValueError,
        match="requires 9 tokens.*context length of 8 tokens by 1",
    ) as exc_info:
        conversation_to_datum(
            [], _Renderer(), max_length=None, model_context_length=8
        )

    message = str(exc_info.value)
    assert "1 image consuming 6 tokens" in message
    assert "remove at least 1 token" in message
    assert "reduce image resolution" in message


def test_strict_mode_rejects_would_be_truncation():
    with pytest.raises(
        ValueError,
        match="requires 10 tokens.*training max_length of 9",
    ) as exc_info:
        conversation_to_datum(
            [],
            _Renderer(),
            max_length=9,
            model_context_length=100,
            fail_on_truncation=True,
        )

    assert "training max_length of 9 tokens by 1" in str(exc_info.value)


def test_legacy_mode_keeps_truncating():
    datum = conversation_to_datum(
        [],
        _Renderer(),
        max_length=9,
        model_context_length=100,
        fail_on_truncation=False,
    )

    assert _model_input_length(datum.model_input) == 8


def test_preflight_reports_split_row_required_and_allowed_tokens():
    renderer = _Renderer()
    dataset = SupervisedDatasetFromHFDataset(
        datasets.Dataset.from_list([{"context_length": 9}, {"context_length": 8}]),
        batch_size=1,
        map_fn=lambda row: conversation_to_datum(
            [],
            renderer,
            max_length=None,
            model_context_length=row["context_length"],
        ),
    )
    config = SimpleNamespace(
        dataset_builder=SimpleNamespace(build=lambda: (dataset, None))
    )

    with pytest.raises(
        ValueError,
        match=(
            "Training dataset preflight failed: Dataset example 1 is invalid: "
            ".*requires 9 tokens.*context length of 8"
        ),
    ):
        prepare_datasets(config)