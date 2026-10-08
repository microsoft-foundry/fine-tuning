"""The documented SFT preflight rejects bad data before allocating a session."""

from unittest.mock import AsyncMock, Mock

import pytest

from interactive_training.recipes.tulu3_sft import train_azure as recipe


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