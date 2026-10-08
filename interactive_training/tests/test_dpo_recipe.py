import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import chz
import datasets
import pytest
import torch
from azure.ai.finetuningsessions.models import (
    Datum,
    ModelInput,
    ModelInputChunk,
    TensorData,
)

from interactive_training.recipes.preference.dpo.datasets import (
    DPODatasetBuilder,
    HelpSteer3ComparisonBuilder,
    HHHComparisonBuilder,
    UltraFeedbackComparisonBuilder,
    _parse_hhh_conversation,
)
from interactive_training.recipes.preference.dpo.train_azure import (
    CLIConfig,
    validate_resume_config,
)
from interactive_training.recipes.preference.dpo.training import (
    Config,
    _full_sequence,
    _reference_logprobs,
    build_dpo_loss_fn,
    compute_dpo_loss,
    evaluate_preferences,
    main,
)
from interactive_training.supervised.types import ChatDatasetBuilderCommonConfig


def _datum(weights: list[float]) -> Datum:
    return Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=[10, 20])]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[20, 30]),
            "weights": TensorData(data=weights),
        },
    )


def test_compute_dpo_loss_prefers_larger_chosen_ratio():
    loss, metrics = compute_dpo_loss(
        [torch.tensor(3.0)],
        [torch.tensor(1.0)],
        [torch.tensor(1.0)],
        [torch.tensor(1.0)],
        dpo_beta=1.0,
    )

    assert loss.item() == pytest.approx(
        -math.log(torch.sigmoid(torch.tensor(2.0)).item())
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["margin"] == 2.0


def test_dpo_loss_fn_pairs_datums_and_backpropagates():
    data = [_datum([0.0, 1.0]), _datum([0.0, 1.0])]
    reference = [torch.tensor([-2.0, -1.0]), torch.tensor([-2.0, -1.0])]
    policy = [
        torch.tensor([-2.0, -0.5], requires_grad=True),
        torch.tensor([-2.0, -1.5], requires_grad=True),
    ]

    loss, metrics = build_dpo_loss_fn(data, reference, dpo_beta=1.0)(data, policy)
    loss.backward()

    assert metrics["accuracy"] == 1.0
    assert policy[0].grad is not None
    assert policy[1].grad is not None
    assert policy[0].grad[0] == 0
    assert policy[1].grad[0] == 0
    assert policy[0].grad[1] < 0
    assert policy[1].grad[1] > 0


def test_full_sequence_restores_last_target_token():
    sequence = _full_sequence(_datum([0.0, 1.0]))
    assert sequence.chunks[0].tokens == [10, 20, 30]


def test_hhh_parser_preserves_turns():
    assert _parse_hhh_conversation("\n\nHuman: Hi\n\nAssistant: Hello") == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


@pytest.mark.asyncio
async def test_reference_logprobs_shift_and_validate():
    reference = SimpleNamespace(
        compute_logprobs_async=AsyncMock(return_value=[None, -1.0, -2.0])
    )
    data = [_datum([0.0, 1.0])]
    result = await _reference_logprobs(reference, data)
    assert result[0].tolist() == [-1.0, -2.0]
    reference.compute_logprobs_async.return_value = [None, -1.0]
    with pytest.raises(ValueError, match="did not align"):
        await _reference_logprobs(reference, data)


@pytest.mark.asyncio
async def test_training_loop_evaluates_and_keeps_reference_on_resume(
    tmp_path, monkeypatch
):
    from interactive_training.recipes.preference.dpo import training

    data = [_datum([0.0, 1.0]), _datum([0.0, 1.0])]
    dataset = MagicMock()
    dataset.__len__.return_value = 3
    dataset.get_batch.return_value = data
    builder = MagicMock()
    builder.build.return_value = (dataset, dataset)
    reference = SimpleNamespace(
        compute_logprobs_async=AsyncMock(return_value=[None, -1.0, -1.0])
    )
    logger = MagicMock()
    monkeypatch.setattr(training.ml_log, "setup_logging", lambda **kwargs: logger)
    save = AsyncMock()
    monkeypatch.setattr(training.checkpoint_utils, "save_checkpoint_async", save)
    monkeypatch.setattr(
        training.checkpoint_utils,
        "get_last_checkpoint",
        lambda _: {
            "epoch": 0,
            "batch": 1,
            "reference_state_path": "reference-checkpoint",
        },
    )
    future = SimpleNamespace(
        result_async=AsyncMock(return_value=SimpleNamespace(metrics={}))
    )

    async def custom_forward(batch, loss_fn):
        policy = [torch.tensor([-1.0, -1.0], requires_grad=True) for _ in batch]
        loss, metrics = loss_fn(batch, policy)
        loss.backward()
        return SimpleNamespace(
            result_async=AsyncMock(return_value=SimpleNamespace(metrics=metrics))
        )

    async def forward(batch, **kwargs):
        return SimpleNamespace(
            result_async=AsyncMock(
                return_value=SimpleNamespace(
                    loss_fn_outputs=[{"logprobs": TensorData(data=[-1.0, -1.0])}]
                    * len(batch)
                )
            )
        )

    client = SimpleNamespace(
        forward_backward_custom_async=AsyncMock(side_effect=custom_forward),
        optim_step_async=AsyncMock(return_value=future),
        forward_async=AsyncMock(side_effect=forward),
    )
    await main(
        Config(
            log_path=str(tmp_path),
            model_name="test",
            dataset_builder=builder,
            max_steps=2,
            eval_every=1,
            save_every=1,
        ),
        client,
        reference,
        "reference-checkpoint",
    )
    assert client.optim_step_async.await_count == 1
    assert client.forward_async.await_count == 6
    assert save.await_count == 2
    final = save.call_args.kwargs
    assert final["loop_state"] == {
        "epoch": 0,
        "batch": 2,
        "completed_updates": 2,
        "reference_state_path": "reference-checkpoint",
    }
    assert final["step_number"] == 1
    assert any("dpo_loss" in call.args[0] for call in logger.log_metrics.call_args_list)
    logger.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "valid_batches,max_steps,num_epochs,resume,updates,cursor",
    [
        ([False, True], 1, 1, None, 1, (1, 0)),
        ([False, False], 1, 1, None, 0, (1, 0)),
        ([True, False, True, True], 2, 1, None, 2, (0, 3)),
        ([True, False, True, False], 4, 1, None, 2, (1, 0)),
        ([False, True, False], 3, 2, None, 2, (2, 0)),
        ([False, True, False], 0, 2, None, 2, (2, 0)),
        (
            [False, True, False, True, False],
            2,
            1,
            {"epoch": 0, "batch": 3, "completed_updates": 1},
            1,
            (0, 4),
        ),
        (
            [True, False, False],
            3,
            1,
            {"epoch": 0, "batch": 1, "completed_updates": 1},
            0,
            (1, 0),
        ),
        (
            [False, True, False, True],
            1,
            1,
            {"epoch": 0, "batch": 2, "completed_updates": 1},
            0,
            (0, 2),
        ),
    ],
)
async def test_training_counts_updates_separately_from_dataset_cursor(
    tmp_path, monkeypatch, valid_batches, max_steps, num_epochs, resume, updates, cursor
):
    from interactive_training.recipes.preference.dpo import training

    data = [_datum([0.0, 1.0]), _datum([0.0, 1.0])]
    dataset = MagicMock()
    dataset.__len__.return_value = len(valid_batches)
    dataset.get_batch.side_effect = lambda index: data if valid_batches[index] else []
    builder = MagicMock()
    builder.build.return_value = (dataset, None)
    logger = MagicMock()
    monkeypatch.setattr(training.ml_log, "setup_logging", lambda **kwargs: logger)
    monkeypatch.setattr(
        training.checkpoint_utils, "get_last_checkpoint", lambda _: resume
    )
    save = AsyncMock()
    monkeypatch.setattr(training.checkpoint_utils, "save_checkpoint_async", save)
    future = SimpleNamespace(
        result_async=AsyncMock(return_value=SimpleNamespace(metrics={}))
    )
    client = SimpleNamespace(
        forward_backward_custom_async=AsyncMock(return_value=future),
        optim_step_async=AsyncMock(return_value=future),
    )
    reference = SimpleNamespace(
        compute_logprobs_async=AsyncMock(return_value=[None, -1.0, -1.0])
    )
    run = main(
        Config(
            log_path=str(tmp_path),
            model_name="test",
            dataset_builder=builder,
            max_steps=max_steps,
            num_epochs=num_epochs,
            eval_every=0,
            save_every=1,
        ),
        client,
        reference,
        "reference-checkpoint",
    )
    if not any(valid_batches):
        with pytest.raises(ValueError, match="No valid preference pairs"):
            await run
        client.optim_step_async.assert_not_awaited()
        save.assert_not_awaited()
        logger.close.assert_called_once()
        return
    await run
    initial_updates = resume["completed_updates"] if resume else 0
    assert client.optim_step_async.await_count == updates
    if resume and cursor == (resume["epoch"], resume["batch"]):
        save.assert_not_awaited()
        dataset.get_batch.assert_not_called()
        logger.close.assert_called_once()
        return
    final = save.call_args.kwargs
    assert final["name"] == "final"
    assert final["loop_state"] == {
        "epoch": cursor[0],
        "batch": cursor[1],
        "completed_updates": initial_updates + updates,
        "reference_state_path": "reference-checkpoint",
    }
    assert final["step_number"] == initial_updates + updates - 1
    assert [call.kwargs["step"] for call in logger.log_metrics.call_args_list] == list(
        range(initial_updates, initial_updates + updates)
    )
    total_steps = len(valid_batches) * num_epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    assert [
        call.args[0].learning_rate for call in client.optim_step_async.call_args_list
    ] == pytest.approx(
        [
            1e-5 * (1 - step / total_steps)
            for step in range(initial_updates, initial_updates + updates)
        ]
    )
    for call in save.call_args_list[:-1]:
        state = call.kwargs["loop_state"]
        assert call.kwargs["name"] == f"step_{state['completed_updates']:06d}"
        assert valid_batches[state["batch"]]
    logger.close.assert_called_once()


@pytest.mark.asyncio
async def test_fixed_preference_eval_detects_learning():
    data = [_datum([0.0, 1.0]), _datum([0.0, 1.0])]
    reference = [torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0])]

    def result(chosen, rejected):
        return SimpleNamespace(
            result_async=AsyncMock(
                return_value=SimpleNamespace(
                    loss_fn_outputs=[
                        {"logprobs": TensorData(data=[-1.0, chosen])},
                        {"logprobs": TensorData(data=[-1.0, rejected])},
                    ]
                )
            )
        )

    client = SimpleNamespace(
        forward_async=AsyncMock(side_effect=[result(-1.0, -1.0), result(-0.5, -1.5)])
    )
    before = await evaluate_preferences(client, data, reference, 0.1, "probe")
    after = await evaluate_preferences(client, data, reference, 0.1, "probe")
    assert after["probe/accuracy"] > before["probe/accuracy"]
    assert after["probe/margin"] > before["probe/margin"]
    assert after["probe/dpo_loss"] < before["probe/dpo_loss"]


@pytest.mark.parametrize("preference,label", [(-1, "A"), (1, "B"), (0, None)])
def test_helpsteer_preference_direction_and_ties(preference, label):
    comparison = HelpSteer3ComparisonBuilder().example_to_labeled_comparison(
        {
            "context": [{"role": "user", "content": "question"}],
            "response1": "first",
            "response2": "second",
            "overall_preference": preference,
        }
    )
    assert (comparison.label if comparison else None) == label


def test_hhh_rejects_mismatched_prompts_and_identical_responses():
    builder = HHHComparisonBuilder()
    chosen = "\n\nHuman: question\n\nAssistant: good"
    rejected = "\n\nHuman: question\n\nAssistant: bad"
    assert (
        builder.example_to_labeled_comparison(
            {"chosen": chosen, "rejected": rejected}
        ).label
        == "A"
    )
    assert (
        builder.example_to_labeled_comparison({"chosen": chosen, "rejected": chosen})
        is None
    )
    assert (
        builder.example_to_labeled_comparison(
            {"chosen": chosen, "rejected": rejected.replace("question", "different")}
        )
        is None
    )


def test_ultrafeedback_chosen_is_preferred():
    labeled = UltraFeedbackComparisonBuilder().example_to_labeled_comparison(
        {
            "instruction": "question",
            "chosen_response": "good",
            "rejected_response": "bad",
        }
    )
    assert labeled.label == "A"
    assert labeled.comparison.completion_a[0]["content"] == "good"


def test_rendered_pairs_keep_order_masks_and_skip_ties(monkeypatch):
    raw = datasets.Dataset.from_list(
        [
            {
                "context": [{"role": "user", "content": "question"}],
                "response1": "first",
                "response2": "second",
                "overall_preference": preference,
            }
            for preference in (-1, 0, 1)
        ]
    )
    builder = MagicMock()
    builder.get_train_and_test_datasets.return_value = (raw, raw.select([]))
    builder.example_to_labeled_comparison.side_effect = (
        HelpSteer3ComparisonBuilder().example_to_labeled_comparison
    )

    def render(conversation):
        answer = 30 if conversation[-1]["content"] == "first" else 40
        return (
            ModelInput(chunks=[ModelInputChunk(tokens=[10, 20, answer, 50])]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )

    renderer = SimpleNamespace(build_supervised_example=render)
    monkeypatch.setattr(DPODatasetBuilder, "renderer", property(lambda _: renderer))
    dataset, test = DPODatasetBuilder(
        comparison_builder=builder,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="test",
            renderer_name="test",
            max_length=32,
            batch_size=3,
        ),
    ).build()
    batch = dataset.get_batch(0)
    assert test is None
    assert [datum.loss_fn_inputs["target_tokens"].data[1] for datum in batch] == [
        30,
        40,
        40,
        30,
    ]
    assert all(
        datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0] for datum in batch
    )


@pytest.mark.asyncio
async def test_reference_concurrency_is_bounded_and_preserves_order():
    pending = 0
    peak = 0

    async def score(sequence):
        nonlocal pending, peak
        pending += 1
        peak = max(peak, pending)
        await asyncio.sleep(0)
        pending -= 1
        return [None, -1.0, -float(sequence.chunks[0].tokens[-1])]

    data = [_datum([0.0, 1.0]) for _ in range(7)]
    for index, datum in enumerate(data):
        datum.loss_fn_inputs["target_tokens"] = TensorData(data=[20, 30 + index])
    result = await _reference_logprobs(
        SimpleNamespace(compute_logprobs_async=score), data, concurrency=2
    )
    assert peak == 2
    assert [values[-1].item() for values in result] == [
        -float(30 + index) for index in range(7)
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 16),
        ("dataset", "helpsteer3"),
        ("lora_rank", 32),
        ("max_length", 512),
        ("max_train_examples", 8),
        ("max_test_examples", 32),
        ("dpo_beta", 0.2),
        ("model_name", "different"),
    ],
)
def test_resume_rejects_changed_training_contract(tmp_path, field, value):
    config = CLIConfig(project_endpoint="https://example.test")
    metadata = {
        "cli_config": chz.asdict(config),
        "tokenizer_name": "tokenizer",
        "renderer_name": "renderer",
    }
    (tmp_path / "run_meta.json").write_text(json.dumps(metadata))
    changed = chz.replace(config, **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_resume_config(str(tmp_path), changed, "tokenizer", "renderer")


def test_resume_allows_more_steps_but_checks_resolved_renderer(tmp_path):
    config = CLIConfig(project_endpoint="https://example.test")
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "cli_config": chz.asdict(config),
                "tokenizer_name": "tokenizer",
                "renderer_name": "renderer",
            }
        )
    )
    extended = chz.replace(config, max_steps=100, num_epochs=2)
    validate_resume_config(str(tmp_path), extended, "tokenizer", "renderer")
    with pytest.raises(ValueError, match="renderer_name"):
        validate_resume_config(str(tmp_path), extended, "tokenizer", "changed")


def test_resume_requires_metadata(tmp_path):
    with pytest.raises(ValueError, match="run_meta.json"):
        validate_resume_config(
            str(tmp_path),
            CLIConfig(project_endpoint="https://example.test"),
            "tokenizer",
            "renderer",
        )
