import asyncio
import io
import json
import os
import stat
import threading
from types import SimpleNamespace

import datasets
import pytest
from PIL import Image

from interactive_training.recipes.visual_spatial import data as visual_spatial
from interactive_training.recipes.visual_spatial import train_rft_azure, training


def _dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "question": [f" question {index} " for index in range(12)],
            "answer": [f" Answer {index} " for index in range(12)],
            "dataset_id": ["visual_spatial"] * 10 + ["other"] * 2,
            "sweep": ["easy"] * 12,
            "image": [f"image-{index}" for index in range(12)],
        }
    )


def test_split_filters_dataset_id_and_is_deterministic() -> None:
    train_a, test_a = visual_spatial.split_dataset(_dataset(), seed=7, test_size=2)
    train_b, test_b = visual_spatial.split_dataset(_dataset(), seed=7, test_size=2)

    assert len(train_a) == 8
    assert len(test_a) == 2
    assert set(train_a["dataset_id"]) == {"visual_spatial"}
    assert train_a["question"] == train_b["question"]
    assert test_a["question"] == test_b["question"]


def test_messages_use_one_image_first_and_share_the_answer_contract() -> None:
    row = _dataset()[0]

    user = visual_spatial.user_message(row)
    messages = visual_spatial.sft_messages(row)

    assert messages[0] == user
    assert user["content"] == [
        {"type": "image", "image": "image-0"},
        {
            "type": "text",
            "text": "Reply with only the exact short answer to this question: question 0",
        },
    ]
    assert messages[1] == {"role": "assistant", "content": "Answer 0"}


def test_grouped_messages_use_64_ordered_images_and_first_answer() -> None:
    rows = [_dataset()[index % 10] for index in range(64)]

    user = visual_spatial.grouped_user_message(rows)
    messages = visual_spatial.grouped_sft_messages(rows)

    assert messages[0] == user
    assert len(user["content"]) == 65
    assert [part["image"] for part in user["content"][:-1]] == [
        f"image-{index % 10}" for index in range(64)
    ]
    assert user["content"][-1] == {
        "type": "text",
        "text": (
            "Inspect all 64 images and reply with only the exact short answer "
            "to this question about the first image: question 0"
        ),
    }
    assert messages[1] == {"role": "assistant", "content": "Answer 0"}


def test_absolute_difference_messages_compare_two_ordered_images() -> None:
    rows = [
        {"image": "first", "question": "How many circles?", "answer": "5"},
        {"image": "second", "question": "How many squares?", "answer": "2"},
    ]

    user = visual_spatial.absolute_difference_user_message(rows)
    messages = visual_spatial.absolute_difference_sft_messages(rows)

    assert user["content"][:2] == [
        {"type": "image", "image": "first"},
        {"type": "image", "image": "second"},
    ]
    assert user["content"][2] == {
        "type": "text",
        "text": (
            "Answer each visual counting question, then reply with only the "
            "absolute difference between the two counts as one integer.\n"
            "First image: How many circles?\n"
            "Second image: How many squares?"
        ),
    }
    assert messages == [user, {"role": "assistant", "content": "3"}]


def test_count_category_messages_use_hidden_deranged_codebook() -> None:
    assert set(visual_spatial.COUNT_CATEGORY_CODES) == set("0123456")
    assert set(visual_spatial.COUNT_CATEGORY_CODES.values()) == set("0123456")
    assert all(
        count != code
        for count, code in visual_spatial.COUNT_CATEGORY_CODES.items()
    )
    row = {"image": "image", "question": "How many circles?", "answer": "2"}

    user = visual_spatial.count_category_user_message(row)
    messages = visual_spatial.count_category_sft_messages(row)

    assert user["content"] == [
        {"type": "image", "image": "image"},
        {
            "type": "text",
            "text": (
                "How many circles?\n"
                "Reply with only the learned category code as one integer."
            ),
        },
    ]
    assert messages[-1] == {"role": "assistant", "content": "0"}


def test_count_category_training_indices_are_balanced_and_deterministic() -> None:
    dataset = datasets.Dataset.from_dict(
        {
            "answer": ["0"] * 20
            + ["1"] * 12
            + ["2"] * 8
            + ["3"] * 5
            + ["4"] * 3
            + ["5"]
            + ["6"],
        }
    )

    first = training._balanced_count_category_indices(dataset, 35, seed=7)
    second = training._balanced_count_category_indices(dataset, 35, seed=7)
    selected_counts = [str(dataset[index]["answer"]) for index in first]

    assert first == second
    assert len(first) == 35
    assert {count: selected_counts.count(count) for count in "0123456"} == {
        count: 5 for count in "0123456"
    }
    assert len({index for index in first if dataset[index]["answer"] == "6"}) == 1


@pytest.mark.asyncio
async def test_count_category_rl_builder_balances_only_training_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dataset = datasets.Dataset.from_dict(
        {
            "answer": ["0"] * 20
            + ["1"] * 12
            + ["2"] * 8
            + ["3"] * 5
            + ["4"] * 3
            + ["5"]
            + ["6"],
        }
    )
    test_dataset = datasets.Dataset.from_dict({"answer": ["6", "0", "6"]})
    renderer = SimpleNamespace(image_cache_dir=None)
    monkeypatch.setattr(
        training,
        "_load_dataset",
        lambda *_args, **_kwargs: (train_dataset, test_dataset),
    )
    monkeypatch.setattr(training, "get_tokenizer", lambda _name: object())
    monkeypatch.setattr(training, "get_image_processor", lambda _name: object())
    monkeypatch.setattr(training, "get_renderer", lambda *_args, **_kwargs: renderer)

    train, test = await training.VisualSpatialRLDatasetBuilder(
        batch_size=5,
        model_name="example/model",
        tokenizer_name="example/model",
        renderer_name="example_renderer",
        group_size=2,
        max_train_examples=35,
        task_variant="count_category",
    )()

    assert train.num_examples == 35
    assert {
        category: train.dataset["answer"].count(category)
        for category in "0123456"
    } == {category: 5 for category in "0123456"}
    assert test.dataset["answer"] == ["6", "0", "6"]


def test_user_message_decodes_huggingface_image_dict() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color="red").save(buffer, format="PNG")
    row = {"image": {"bytes": buffer.getvalue(), "path": None}, "question": " locate "}

    image = visual_spatial.user_message(row)["content"][0]["image"]

    assert isinstance(image, Image.Image)
    assert image.size == (3, 2)


def test_exact_match_only_normalizes_case_and_whitespace() -> None:
    assert visual_spatial.answers_match("  ANSWER\n0 ", "Answer 0")
    assert not visual_spatial.answers_match("Answer: 0", "Answer 0")


@pytest.mark.parametrize("without_fchmod", [False, True])
def test_accuracy_summary_reports_before_and_after_training(
    tmp_path, capsys, monkeypatch, without_fchmod
) -> None:
    if without_fchmod:
        monkeypatch.delattr(os, "fchmod", raising=False)
    records = [
        {
            "step": 0,
            "test/visual_spatial_accuracy": 0.2,
            "test/visual_spatial_correct": 4.0,
            "test/visual_spatial_examples": 20.0,
        },
        {"step": 5, "optim/loss": 0.5},
        {
            "step": 10,
            "test/visual_spatial_accuracy": 0.65,
            "test/visual_spatial_correct": 13.0,
            "test/visual_spatial_examples": 20.0,
        },
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = train_rft_azure._write_accuracy_summary(str(tmp_path))

    assert summary == {
        "schema_version": 1,
        "metric": "test/visual_spatial_accuracy",
        "before_training": {"step": 0, "accuracy": 0.2, "correct": 4, "examples": 20},
        "after_training": {"step": 10, "accuracy": 0.65, "correct": 13, "examples": 20},
        "absolute_improvement": pytest.approx(0.45),
        "percentage_point_improvement": pytest.approx(45.0),
        "relative_improvement": pytest.approx(2.25),
        "evaluations_recorded": 2,
    }
    persisted = json.loads(
        (tmp_path / "visual_spatial_accuracy_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["before_training"]["step"] == 0
    assert persisted["after_training"]["step"] == 10
    # POSIX mode bits do not represent Windows ACL permissions.
    if os.name == "posix":
        assert stat.S_IMODE(
            (tmp_path / "visual_spatial_accuracy_summary.json").stat().st_mode
        ) == 0o600
    output = capsys.readouterr().out
    assert "Before training: 20.0% (4/20)" in output
    assert "After training:  65.0% (13/20)" in output
    assert "Improvement:     +45.0 percentage points" in output


def test_accuracy_summary_preserves_zero_baseline_across_resume(tmp_path) -> None:
    records = [
        {"step": 0, "test/visual_spatial_accuracy": 0.0},
        {"step": 20, "test/visual_spatial_accuracy": 0.25},
        {"step": 40, "test/visual_spatial_accuracy": 0.5},
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = train_rft_azure._write_accuracy_summary(str(tmp_path))

    assert summary is not None
    assert summary["before_training"] == {"step": 0, "accuracy": 0.0}
    assert summary["after_training"] == {"step": 40, "accuracy": 0.5}
    assert summary["percentage_point_improvement"] == 50.0
    assert summary["relative_improvement"] is None


def test_accuracy_summary_explains_missing_evaluation(tmp_path, capsys) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"step": 0, "test/visual_spatial_accuracy": 0.2}) + "\n",
        encoding="utf-8",
    )

    assert train_rft_azure._write_accuracy_summary(str(tmp_path)) is None
    assert not (tmp_path / "visual_spatial_accuracy_summary.json").exists()
    assert "comparison unavailable" in capsys.readouterr().out


def test_accuracy_summary_ignores_malformed_records(tmp_path) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        "not-json\n"
        + json.dumps(["not", "an", "object"])
        + "\n"
        + json.dumps({"step": None, "test/visual_spatial_accuracy": 0.1})
        + "\n"
        + json.dumps({"step": 0, "test/visual_spatial_accuracy": 0.2})
        + "\n"
        + json.dumps({"step": 10, "test/visual_spatial_accuracy": 0.6})
        + "\n",
        encoding="utf-8",
    )

    summary = train_rft_azure._write_accuracy_summary(str(tmp_path))

    assert summary is not None
    assert summary["before_training"] == {"step": 0, "accuracy": 0.2}
    assert summary["after_training"] == {"step": 10, "accuracy": 0.6}


@pytest.mark.asyncio
@pytest.mark.parametrize("without_fchmod", [False, True])
async def test_grouped_evaluator_sends_64_images(monkeypatch, tmp_path, without_fchmod) -> None:
    if without_fchmod:
        monkeypatch.delattr(os, "fchmod", raising=False)
    test_dataset = datasets.Dataset.from_dict(
        {
            "question": [f"question {index}" for index in range(70)],
            "answer": [f"answer {index}" for index in range(70)],
            "image": [Image.new("RGB", (index + 1, 1)) for index in range(70)],
        }
    )
    captured_messages = []

    class FakeRenderer:
        def build_generation_prompt(self, messages):
            from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

            captured_messages.extend(messages)
            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return {"role": "assistant", "content": "answer 0"}, True

    class FakeSamplingClient:
        async def sample_async(self, _prompt, *, sampling_params):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])])

    monkeypatch.setattr(
        training,
        "load_dataset",
        lambda seed: (test_dataset.select(range(64)), test_dataset),
    )
    monkeypatch.setattr(training, "get_tokenizer", lambda model_name: object())
    monkeypatch.setattr(training, "get_image_processor", lambda model_name: object())
    monkeypatch.setattr(
        training,
        "get_renderer",
        lambda renderer_name, tokenizer, image_processor: FakeRenderer(),
    )

    evaluator = training.VisualSpatialAccuracyEvaluator(
        model_name="model",
        renderer_name="renderer",
        seed=0,
        output_dir=str(tmp_path),
        max_examples=1,
        images_per_example=64,
    )
    output_path = tmp_path / "visual_spatial_eval_0.jsonl"
    output_path.write_text("stale", encoding="utf-8")
    os.chmod(output_path, 0o644)
    metrics = await evaluator(FakeSamplingClient())
    await evaluator(FakeSamplingClient())

    image_parts = [
        part
        for part in captured_messages[0]["content"]
        if part["type"] == "image"
    ]
    assert len(image_parts) == 64
    assert len(captured_messages) == 1
    assert metrics["test/visual_spatial_accuracy"] == 1.0
    assert metrics["test/visual_spatial_parse_rate"] == 1.0
    assert metrics["test/visual_spatial_frac_at_max_tokens"] == 0.0

    prediction = json.loads(output_path.read_text())
    if os.name == "posix":
        assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert prediction["completion_tokens"] == 1
    assert prediction["hit_max_tokens"] is False
    assert prediction["example_id"]
    assert "question" not in prediction
    assert "expected" not in prediction
    assert "prediction" not in prediction


@pytest.mark.asyncio
async def test_absolute_difference_evaluator_uses_held_out_pair(
    monkeypatch, tmp_path
) -> None:
    train_dataset = datasets.Dataset.from_dict(
        {
            "question": ["Training question one", "Training question two"],
            "answer": ["0", "0"],
            "image": ["training-image-one", "training-image-two"],
        }
    )
    test_dataset = datasets.Dataset.from_dict(
        {
            "question": ["How many circles?", "How many squares?"],
            "answer": ["5", "2"],
            "image": ["first-image", "second-image"],
        }
    )
    captured_messages = []

    class FakeRenderer:
        def build_generation_prompt(self, messages):
            captured_messages.extend(messages)
            from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return {"role": "assistant", "content": "3"}, True

    class FakeSamplingClient:
        async def sample_async(self, _prompt, *, sampling_params):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])])

    monkeypatch.setattr(
        training,
        "load_dataset",
        lambda seed: (train_dataset, test_dataset),
    )
    monkeypatch.setattr(training, "get_tokenizer", lambda _name: object())
    monkeypatch.setattr(training, "get_image_processor", lambda _name: object())
    monkeypatch.setattr(
        training,
        "get_renderer",
        lambda *_args, **_kwargs: FakeRenderer(),
    )

    evaluator = training.VisualSpatialAccuracyEvaluator(
        model_name="model",
        renderer_name="renderer",
        seed=0,
        output_dir=str(tmp_path),
        max_examples=1,
        images_per_example=2,
        task_variant="absolute_difference",
        log_examples=True,
    )
    metrics = await evaluator(FakeSamplingClient())

    assert captured_messages[0]["content"][:2] == [
        {"type": "image", "image": "first-image"},
        {"type": "image", "image": "second-image"},
    ]
    assert metrics["test/visual_spatial_accuracy"] == 1.0
    assert metrics["test/visual_spatial_single_integer_answer_rate"] == 1.0
    prediction = json.loads(
        (tmp_path / "visual_spatial_eval_0.jsonl").read_text()
    )
    assert prediction["expected"] == "3"


@pytest.mark.asyncio
async def test_count_category_evaluator_grades_transformed_target(
    monkeypatch, tmp_path
) -> None:
    test_dataset = datasets.Dataset.from_dict(
        {"question": ["How many circles?"], "answer": ["2"], "image": ["image"]}
    )

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return {"role": "assistant", "content": "0"}, True

    class FakeSamplingClient:
        async def sample_async(self, _prompt, *, sampling_params):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])])

    monkeypatch.setattr(
        training,
        "load_dataset",
        lambda seed: (test_dataset, test_dataset),
    )
    monkeypatch.setattr(training, "get_tokenizer", lambda _name: object())
    monkeypatch.setattr(training, "get_image_processor", lambda _name: object())
    monkeypatch.setattr(
        training,
        "get_renderer",
        lambda *_args, **_kwargs: FakeRenderer(),
    )

    evaluator = training.VisualSpatialAccuracyEvaluator(
        model_name="model",
        renderer_name="renderer",
        seed=0,
        output_dir=str(tmp_path),
        task_variant="count_category",
        log_examples=True,
    )
    metrics = await evaluator(FakeSamplingClient())

    assert metrics["test/visual_spatial_accuracy"] == 1.0
    prediction = json.loads(
        (tmp_path / "visual_spatial_eval_0.jsonl").read_text()
    )
    assert prediction["expected"] == "0"


@pytest.mark.asyncio
async def test_evaluator_bounds_sampling_and_builds_prompts_off_event_loop(
    monkeypatch, tmp_path
) -> None:
    test_dataset = datasets.Dataset.from_dict(
        {
            "question": [f"question {index}" for index in range(12)],
            "answer": ["answer"] * 12,
            "image": ["image"] * 12,
        }
    )
    main_thread = threading.get_ident()
    build_threads = []
    active_samples = 0
    max_active_samples = 0

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            build_threads.append(threading.get_ident())
            from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return {"role": "assistant", "content": "answer"}, True

    class FakeSamplingClient:
        async def sample_async(self, _prompt, *, sampling_params):
            nonlocal active_samples, max_active_samples
            active_samples += 1
            max_active_samples = max(max_active_samples, active_samples)
            await asyncio.sleep(0)
            active_samples -= 1
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])])

    monkeypatch.setattr(
        training,
        "load_dataset",
        lambda seed: (test_dataset, test_dataset),
    )
    monkeypatch.setattr(training, "get_tokenizer", lambda _name: object())
    monkeypatch.setattr(training, "get_image_processor", lambda _name: object())
    monkeypatch.setattr(
        training,
        "get_renderer",
        lambda *_args, **_kwargs: FakeRenderer(),
    )

    evaluator = training.VisualSpatialAccuracyEvaluator(
        model_name="model",
        renderer_name="renderer",
        seed=0,
        output_dir=str(tmp_path),
        concurrency=3,
        prompt_cache_max_mb=0,
    )
    metrics = await evaluator(FakeSamplingClient())

    assert metrics["test/visual_spatial_examples"] == 12
    assert max_active_samples <= 3
    assert build_threads
    assert all(thread_id != main_thread for thread_id in build_threads)


def test_evaluator_rejects_empty_evaluation_before_sampling_client_creation(
    monkeypatch,
    tmp_path,
) -> None:
    empty_dataset = datasets.Dataset.from_dict(
        {"question": [], "answer": [], "image": []}
    )
    monkeypatch.setattr(
        training,
        "load_dataset",
        lambda seed: (empty_dataset, empty_dataset),
    )

    with pytest.raises(
        ValueError,
        match="visual_spatial/evaluation must contain at least one example",
    ):
        training.VisualSpatialAccuracyEvaluator(
            model_name="model",
            renderer_name="renderer",
            seed=0,
            output_dir=str(tmp_path),
        )


def test_grouped_evaluator_preflight_rejects_oversized_prompt(
    monkeypatch, tmp_path
) -> None:
    test_dataset = datasets.Dataset.from_dict(
        {"question": ["question"], "answer": ["answer"], "image": ["image"]}
    )

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

            return ModelInput(chunks=[ModelInputChunk(tokens=list(range(31)))])

    monkeypatch.setattr(training, "load_dataset", lambda seed: (test_dataset, test_dataset))
    monkeypatch.setattr(training, "get_tokenizer", lambda model_name: object())
    monkeypatch.setattr(training, "get_image_processor", lambda model_name: object())
    monkeypatch.setattr(
        training,
        "get_renderer",
        lambda renderer_name, tokenizer, image_processor: FakeRenderer(),
    )

    evaluator = training.VisualSpatialAccuracyEvaluator(
        model_name="model",
        renderer_name="renderer",
        seed=0,
        output_dir=str(tmp_path),
        max_examples=1,
        max_tokens=2,
    )

    with pytest.raises(
        ValueError,
        match=r"exceeding the model context length of 32 tokens by 1",
    ) as exc_info:
        evaluator.preflight(model_context_length=32)

    message = str(exc_info.value)
    assert "the prompt may contain at most 30 tokens" in message
    assert "remove at least 1 token" in message
    assert "reduce max_tokens from 2 to at most 1" in message