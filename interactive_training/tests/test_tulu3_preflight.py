"""The documented SFT preflight rejects bad data before allocating a session."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import datasets
import pytest
import torch
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.recipes.tulu3_sft import train_azure as recipe
from interactive_training.supervised.types import ChatDatasetBuilderCommonConfig


@pytest.fixture
def synthetic_tulu3(monkeypatch):
    """Exercise the real builder without dataset or tokenizer downloads."""
    source = datasets.Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": "example"}]}] * 1025
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        Mock(return_value=datasets.DatasetDict({"train": source})),
    )
    renderer = SimpleNamespace(
        build_supervised_example=Mock(
            return_value=(
                ModelInput(chunks=[ModelInputChunk(tokens=list(range(10)))]),
                torch.ones(10),
            )
        )
    )
    monkeypatch.setattr(recipe.Tulu3Builder, "renderer", property(lambda _: renderer))


@pytest.mark.parametrize("split_index", [0, 1], ids=["training", "evaluation"])
@pytest.mark.parametrize(
    ("max_length", "model_context_length", "fail_on_truncation", "message"),
    [
        (8, None, True, "training max_length of 8"),
        (None, 8, False, "model context length of 8"),
    ],
)
def test_tulu3_builder_enforces_configured_limits(
    synthetic_tulu3, split_index, max_length, model_context_length,
    fail_on_truncation, message,
):
    builder = recipe.Tulu3Builder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="unused",
            renderer_name="unused",
            batch_size=1,
            max_length=max_length,
            model_context_length=model_context_length,
            fail_on_truncation=fail_on_truncation,
        ),
        max_train_examples=1,
        max_test_examples=1,
    )
    split = builder.build()[split_index]
    with pytest.raises(ValueError, match=message):
        split.get_batch(0)


@pytest.mark.parametrize("split_index", [0, 1], ids=["training", "evaluation"])
@pytest.mark.parametrize(
    ("max_length", "fail_on_truncation", "expected_tokens"),
    [(10, True, 9), (8, False, 7)],
    ids=["exact-boundary", "legacy-truncation"],
)
def test_tulu3_builder_preserves_valid_and_default_behavior(
    synthetic_tulu3, split_index, max_length, fail_on_truncation, expected_tokens,
):
    builder = recipe.Tulu3Builder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="unused",
            renderer_name="unused",
            batch_size=1,
            max_length=max_length,
            model_context_length=9,
            fail_on_truncation=fail_on_truncation,
        ),
        max_train_examples=1,
        max_test_examples=1,
    )
    datum = builder.build()[split_index].get_batch(0)[0]
    assert sum(len(chunk.tokens) for chunk in datum.model_input.chunks) == expected_tokens


@pytest.mark.asyncio
async def test_real_tulu3_preflight_rejects_truncation_before_session(
    synthetic_tulu3, monkeypatch, tmp_path,
):
    monkeypatch.delenv("INTERACTIVE_POST_TRAINING_API_ENDPOINT", raising=False)
    create = Mock(side_effect=AssertionError("must not allocate a session"))
    monkeypatch.setattr("azure.ai.finetuningsessions.FineTuningSession.create", create)
    with pytest.raises(ValueError, match="Training dataset preflight failed"):
        await recipe.cli_main(
            recipe.CLIConfig(
                project_endpoint="https://example.invalid/api/projects/test",
                log_path=str(tmp_path / "run"),
                preflight_only=True,
                fail_on_truncation=True,
                max_length=8,
                batch_size=1,
                max_train_examples=1,
                max_test_examples=1,
            )
        )
    create.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_dataset_fails_before_sdk_session(monkeypatch, tmp_path):
    monkeypatch.delenv("INTERACTIVE_POST_TRAINING_API_ENDPOINT", raising=False)
    create = Mock(side_effect=AssertionError("must not allocate a session"))
    monkeypatch.setattr("azure.ai.finetuningsessions.FineTuningSession.create", create)
    preflight = Mock(side_effect=ValueError("Training dataset preflight failed"))
    monkeypatch.setattr(recipe.train, "prepare_datasets", preflight)
    config = recipe.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"),
        preflight_dataset=True,
        fail_on_truncation=True,
        model_context_length=4096,
        max_length=2048,
    )
    with pytest.raises(ValueError, match="preflight failed"):
        await recipe.cli_main(config)
    common = preflight.call_args.args[0].dataset_builder.common_config
    assert common.fail_on_truncation is True
    assert common.model_context_length == 4096
    create.assert_not_called()


@pytest.mark.asyncio
async def test_preflight_only_never_contacts_service(monkeypatch, tmp_path):
    monkeypatch.delenv("INTERACTIVE_POST_TRAINING_API_ENDPOINT", raising=False)
    create = Mock(side_effect=AssertionError("must not allocate a session"))
    monkeypatch.setattr("azure.ai.finetuningsessions.FineTuningSession.create", create)
    preflight = Mock(return_value=([], None))
    monkeypatch.setattr(recipe.train, "prepare_datasets", preflight)
    train = AsyncMock()
    monkeypatch.setattr(recipe.train, "main", train)
    await recipe.cli_main(
        recipe.CLIConfig(
            project_endpoint="https://example.invalid/api/projects/test",
            log_path=str(tmp_path / "run"),
            preflight_only=True,
        )
    )
    preflight.assert_called_once()
    create.assert_not_called()
    train.assert_not_called()