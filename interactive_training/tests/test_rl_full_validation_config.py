from unittest.mock import AsyncMock, MagicMock

import chz
import pytest

from interactive_training.rl import train, train_azure
from interactive_training.rl.metric_util import RLTestSetEvaluator
from interactive_training.rl.types import RLDataset, RLDatasetBuilder


class _Dataset(RLDataset):
    def __init__(self):
        self.builder = MagicMock()

    def __len__(self):
        return 1

    def get_batch(self, index):
        return [self.builder]


@chz.chz
class _DatasetBuilder(RLDatasetBuilder):
    train: RLDataset = chz.field(default_factory=_Dataset)
    validation: RLDataset = chz.field(default_factory=_Dataset)

    async def __call__(self):
        return self.train, self.validation


def _config(**overrides):
    return train.Config(
        learning_rate=1e-5,
        dataset_builder=_DatasetBuilder(),
        model_name="test-model",
        max_tokens=10,
        log_path="unused-validation-test-logs",
        enable_trace=False,
        **overrides,
    )


def test_shared_config_defaults_to_tolerant_validation():
    cfg = _config()

    assert train_azure.Config is train.Config
    assert cfg.require_full_validation is False
    assert cfg.validation_observer is None


@pytest.mark.parametrize("entrypoint", [train, train_azure], ids=["train", "train_azure"])
@pytest.mark.parametrize("strict", [None, False, True], ids=["default", "tolerant", "strict"])
@pytest.mark.parametrize("with_observer", [False, True])
async def test_entrypoints_forward_validation_requirement(
    monkeypatch, entrypoint, strict, with_observer
):
    observer = MagicMock(name="observer") if with_observer else None
    if observer is not None:
        observer.name = "validation"
        observer.max_tokens = None
    overrides = {"validation_observer": observer}
    if strict is not None:
        overrides["require_full_validation"] = strict
    cfg = _config(**overrides)
    training_client = MagicMock()
    client = MagicMock()
    client.create_lora_training_client_async = AsyncMock(return_value=training_client)
    client.close_session = AsyncMock()
    client.close = AsyncMock()

    class _ReachedTraining(Exception):
        pass

    training = AsyncMock(side_effect=_ReachedTraining)
    monkeypatch.setattr(entrypoint.ml_log, "setup_logging", MagicMock())
    monkeypatch.setattr(entrypoint.checkpoint_utils, "get_last_checkpoint", lambda path: None)
    monkeypatch.setattr(entrypoint, "do_sync_training", training)
    if entrypoint is train:
        monkeypatch.setattr(train, "FineTuningSessionClient", lambda **kwargs: client)
    else:
        monkeypatch.setattr(train_azure, "get_tokenizer", MagicMock())
        monkeypatch.setattr(
            train_azure, "AzureSDKTrainingClient", MagicMock(return_value=training_client)
        )

    # Stop after the real evaluator is built, before any training or log-file writes.
    with pytest.raises(_ReachedTraining):
        if entrypoint is train:
            await train.main(cfg)
        else:
            await train_azure.main(cfg, client, "test-session")

    training.assert_awaited_once()
    evaluators = training.await_args.kwargs["evaluators"]
    assert len(evaluators) == 1
    evaluator = evaluators[0]
    assert isinstance(evaluator, RLTestSetEvaluator)
    assert evaluator.env_group_builders_P == [cfg.dataset_builder.validation.builder]
    assert evaluator.require_full_validation is (strict is True)
    assert evaluator.observer is observer
