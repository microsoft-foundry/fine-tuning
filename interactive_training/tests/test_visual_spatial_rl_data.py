import io
import json
from pathlib import Path

import chz
import datasets
import pytest
from azure.ai.finetuningsessions.models import ImageChunk, ModelInput, ModelInputChunk
from PIL import Image

from interactive_training.completers import TokensWithLogprobs
from interactive_training.recipes.visual_spatial import (
    train_azure,
    train_rft_azure,
    train_sft_azure,
)
from interactive_training.recipes.visual_spatial.training import VisualSpatialRLDataset
from interactive_training.rl.data_processing import trajectory_to_data
from interactive_training.rl.types import Trajectory, Transition


def test_visual_spatial_has_a_dedicated_rl_entrypoint() -> None:
    fields = set(chz.chz_fields(train_rft_azure.CLIConfig))

    assert {
        "project_endpoint",
        "data_path",
        "images_per_example",
        "task_variant",
        "model_context_length",
        "preflight_only",
        "prompt_cache_max_mb",
        "freeze_vision_tower",
        "freeze_multi_modal_projector",
        "max_steps_off_policy",
        "num_epochs",
    } <= fields


def test_visual_spatial_has_a_dedicated_sft_entrypoint() -> None:
    fields = set(chz.chz_fields(train_sft_azure.CLIConfig))

    assert {
        "project_endpoint",
        "data_path",
        "images_per_example",
        "task_variant",
        "model_context_length",
        "preflight_only",
        "freeze_vision_tower",
        "freeze_multi_modal_projector",
        "batch_size",
        "num_epochs",
        "max_steps",
    } <= fields


def test_visual_spatial_has_a_staged_entrypoint() -> None:
    fields = set(chz.chz_fields(train_azure.CLIConfig))

    assert {
        "project_endpoint",
        "model_name",
        "sft_learning_rate",
        "sft_max_steps",
        "rft_learning_rate",
        "rft_group_size",
        "rft_max_steps",
    } <= fields


def test_staged_workflow_selects_best_recoverable_sft_checkpoint(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints.jsonl").write_text(
        "\n".join(
            (
                '{"step":20,"state_path":"model_test/20"}',
                '{"step":40,"state_path":"model_test/40"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            (
                '{"step":0,"test/visual_spatial_accuracy":0.9}',
                '{"step":20,"test/visual_spatial_accuracy":0.4}',
                '{"step":40,"test/visual_spatial_accuracy":0.6,'
                '"test/visual_spatial_correct":6,'
                '"test/visual_spatial_examples":10}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    selected = train_azure._select_best_sft_checkpoint(str(tmp_path))

    assert selected == train_azure.CheckpointEvaluation(
        checkpoint_path="model_test/40",
        step=40,
        accuracy=0.6,
        correct=6,
        examples=10,
    )


def test_staged_workflow_pairs_final_checkpoint_with_final_evaluation(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints.jsonl").write_text(
        "\n".join(
            (
                '{"step":10,"state_path":"model_test/10"}',
                '{"step":20,"state_path":"model_test/20"}',
                '{"step":20,"state_path":"model_test/final"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            (
                '{"step":10,"test/visual_spatial_accuracy":0.8}',
                '{"step":20,"test/visual_spatial_accuracy":0.9}',
                '{"step":20,"test/visual_spatial_accuracy":0.4}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_evaluations.jsonl").write_text(
        "\n".join(
            (
                '{"step":20,"checkpoint_path":"model_test/20",'
                '"test/visual_spatial_accuracy":0.9}',
                '{"step":20,"checkpoint_path":"model_test/final",'
                '"test/visual_spatial_accuracy":0.4}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    selected = train_azure._select_best_sft_checkpoint(str(tmp_path))

    assert selected == train_azure.CheckpointEvaluation(
        checkpoint_path="model_test/20",
        step=20,
        accuracy=0.9,
    )


def test_staged_workflow_ignores_unmatched_final_checkpoint(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints.jsonl").write_text(
        "\n".join(
            (
                '{"step":10,"state_path":"model_test/10"}',
                '{"step":20,"state_path":"model_test/20"}',
                '{"step":20,"state_path":"model_test/final"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            (
                '{"step":10,"test/visual_spatial_accuracy":0.8}',
                '{"step":20,"test/visual_spatial_accuracy":0.9}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_evaluations.jsonl").write_text(
        "\n".join(
            (
                '{"step":10,"checkpoint_path":"model_test/10",'
                '"test/visual_spatial_accuracy":0.8}',
                '{"step":20,"checkpoint_path":"model_test/20",'
                '"test/visual_spatial_accuracy":0.9}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    selected = train_azure._select_best_sft_checkpoint(str(tmp_path))

    assert selected == train_azure.CheckpointEvaluation(
        checkpoint_path="model_test/20",
        step=20,
        accuracy=0.9,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "training_type",
    [None, "GlobalStandard"],
)
async def test_staged_workflow_hands_best_sft_checkpoint_to_rft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    training_type: str | None,
) -> None:
    captured: dict = {}

    async def fake_sft_main(config):
        captured["sft"] = config
        sft_path = Path(config.log_path)
        sft_path.mkdir(parents=True)
        (sft_path / "run_meta.json").write_text(
            '{"azure":{"session_id":"session_sft"}}', encoding="utf-8"
        )
        (sft_path / "checkpoints.jsonl").write_text(
            '{"step":20,"state_path":"model_test/20"}\n'
            '{"step":40,"state_path":"model_test/40"}\n',
            encoding="utf-8",
        )
        (sft_path / "metrics.jsonl").write_text(
            '{"step":20,"test/visual_spatial_accuracy":0.25}\n'
            '{"step":40,"test/visual_spatial_accuracy":0.75}\n',
            encoding="utf-8",
        )

    async def fake_rft_main(config):
        captured["rft"] = config
        rft_path = Path(config.log_path)
        rft_path.mkdir(parents=True)
        (rft_path / "run_meta.json").write_text(
            '{"azure":{"session_id":"session_rft"}}', encoding="utf-8"
        )

    monkeypatch.setattr(train_azure.train_sft_azure, "cli_main", fake_sft_main)
    monkeypatch.setattr(train_azure.train_rft_azure, "cli_main", fake_rft_main)

    log_path = tmp_path / "staged"
    await train_azure.cli_main(
        train_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            log_path=str(log_path),
            behavior_if_log_dir_exists="raise",
            training_type=training_type,
        )
    )

    assert captured["sft"].learning_rate == 3e-5
    assert captured["sft"].training_type == training_type
    assert captured["rft"].training_type == training_type
    assert captured["sft"].max_steps == 60
    assert captured["rft"].load_checkpoint_path == "model_test/40"
    assert captured["rft"].learning_rate == 3e-6
    assert captured["rft"].group_size == 16
    assert captured["rft"].max_steps == 10
    summary = json.loads((log_path / "staged_summary.json").read_text())
    assert summary["selected_sft_checkpoint"]["step"] == 40
    metadata = json.loads((log_path / "run_meta.json").read_text())
    assert metadata["azure"]["session_ids"] == {
        "sft": "session_sft", "rft": "session_rft"
    }


@pytest.mark.asyncio
async def test_staged_workflow_records_created_session_on_stage_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    async def failing_sft(config):
        stage = Path(config.log_path)
        stage.mkdir(parents=True)
        (stage / "run_meta.json").write_text(
            '{"azure":{"session_id":"session_sft"}}', encoding="utf-8"
        )
        raise RuntimeError("training failed")

    monkeypatch.setattr(train_azure.train_sft_azure, "cli_main", failing_sft)
    log_path = tmp_path / "staged"
    with pytest.raises(RuntimeError, match="training failed"):
        await train_azure.cli_main(train_azure.CLIConfig(
            project_endpoint="https://example.test",
            log_path=str(log_path),
            behavior_if_log_dir_exists="raise",
        ))
    metadata = json.loads((log_path / "run_meta.json").read_text())
    assert metadata["azure"]["session_ids"] == {"sft": "session_sft", "rft": None}


@pytest.mark.asyncio
async def test_staged_workflow_resumes_existing_child_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "staged"
    sft_path = log_path / "sft"
    rft_path = log_path / "rft"
    sft_path.mkdir(parents=True)
    rft_path.mkdir()
    (sft_path / "checkpoints.jsonl").write_text(
        '{"step":20,"state_path":"model_test/20"}\n',
        encoding="utf-8",
    )
    (sft_path / "metrics.jsonl").write_text(
        '{"step":20,"test/visual_spatial_accuracy":0.75}\n',
        encoding="utf-8",
    )
    captured: dict = {}

    async def fake_sft_main(config):
        captured["sft"] = config

    async def fake_rft_main(config):
        captured["rft"] = config

    monkeypatch.setattr(train_azure.train_sft_azure, "cli_main", fake_sft_main)
    monkeypatch.setattr(train_azure.train_rft_azure, "cli_main", fake_rft_main)

    await train_azure.cli_main(
        train_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            log_path=str(log_path),
            behavior_if_log_dir_exists="resume",
        )
    )

    assert captured["sft"].behavior_if_log_dir_exists == "resume"
    assert captured["rft"].behavior_if_log_dir_exists == "resume"
    assert captured["rft"].load_checkpoint_path == "model_test/20"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "training_type",
    [None, "GlobalStandard"],
)
async def test_staged_preflight_runs_both_stages_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    training_type: str | None,
) -> None:
    captured: list = []

    async def fake_sft_main(config):
        captured.append(("sft", config))

    async def fake_rft_main(config):
        captured.append(("rft", config))

    monkeypatch.setattr(train_azure.train_sft_azure, "cli_main", fake_sft_main)
    monkeypatch.setattr(train_azure.train_rft_azure, "cli_main", fake_rft_main)

    await train_azure.cli_main(
        train_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            log_path=str(tmp_path / "staged"),
            preflight_only=True,
            behavior_if_log_dir_exists="raise",
            training_type=training_type,
        )
    )

    assert [stage for stage, _ in captured] == ["sft", "rft"]
    assert all(config.preflight_only for _, config in captured)
    assert all(config.training_type == training_type for _, config in captured)
    assert captured[1][1].load_checkpoint_path is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_test_examples", 0),
        ("max_train_examples", 0),
        ("eval_concurrency", 0),
        ("images_per_example", 65),
        ("prompt_cache_max_mb", -1),
    ],
)
def test_visual_spatial_rejects_invalid_limits_before_session(
    field_name, value
) -> None:
    config = train_rft_azure.CLIConfig(
        project_endpoint="https://example.test",
        model_name="example/vision-language-model",
        **{field_name: value},
    )

    with pytest.raises(ValueError, match=field_name):
        train_rft_azure._validate_cli_config(config)


def test_absolute_difference_requires_exactly_two_images() -> None:
    config = train_rft_azure.CLIConfig(
        project_endpoint="https://example.test",
        model_name="example/vision-language-model",
        task_variant="absolute_difference",
        images_per_example=1,
    )

    with pytest.raises(
        ValueError,
        match="absolute_difference requires images_per_example=2",
    ):
        train_rft_azure._validate_cli_config(config)


def test_visual_spatial_rejects_unknown_task_variant() -> None:
    config = train_rft_azure.CLIConfig(
        project_endpoint="https://example.test",
        model_name="example/vision-language-model",
        task_variant="unknown",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="task_variant must be one of"):
        train_rft_azure._validate_cli_config(config)


def test_visual_spatial_defaults_to_muse_with_explicit_override() -> None:
    config = train_azure.CLIConfig(project_endpoint="https://example.test")
    assert config.model_name == "meta-models/Muse-Glimmer-30B"
    overridden = train_azure.CLIConfig(
        project_endpoint="https://example.test",
        model_name="example/vision-language-model",
    )
    assert overridden.model_name == "example/vision-language-model"


@pytest.mark.parametrize(
    ("checkpoint_path", "expected"),
    [
        ("model_23aafbc3/final", ("session_23aafbc3", "final")),
        ("model_23aafbc3/final", ("session_23aafbc3", "final")),
        ("session_23aafbc3/final", ("session_23aafbc3", "final")),
    ],
)
def test_visual_spatial_checkpoint_path_uses_source_session_id(
    checkpoint_path: str,
    expected: tuple[str, str],
) -> None:
    assert train_rft_azure._parse_checkpoint_path(checkpoint_path) == expected


def test_math_rl_does_not_own_visual_spatial() -> None:
    math_entrypoint = (
        Path(train_azure.__file__).parents[1] / "math_rl" / "train_azure.py"
    ).read_text(encoding="utf-8")

    assert "visual_spatial" not in math_entrypoint
    assert "VisualSpatial" not in math_entrypoint


def test_visual_spatial_readme_uses_the_dedicated_entrypoint() -> None:
    readme = Path(train_azure.__file__).with_name("README.md").read_text(
        encoding="utf-8"
    )

    assert "interactive_training.recipes.visual_spatial.train_azure" in readme
    assert "interactive_training.recipes.math_rl.train_azure" not in readme
    assert "env=visual_spatial" not in readme
    assert "behavior_if_log_dir_exists=raise" in readme
    assert "## Muse Glimmer reasoning strength" in readme
    assert "muse_glimmer_low_reasoning" in readme
    assert "muse_glimmer_medium_reasoning" in readme
    assert "muse_glimmer_xhigh_reasoning" in readme
    assert "train_rft_azure" in readme
    assert "train_staged_azure" not in readme
    assert "staged_summary.json" in readme
    assert "### Untrusted image limits" in readme
    assert "Production egress policy" in readme
    assert "Remote references must use\nHTTPS" in readme
    assert "Aggregate encoded images per example" in readme
    assert "log_examples=true" in readme
    assert '"messages":[{"role":"user"' in readme
    assert '"url":"https://example.com/part.png"' in readme
    assert '"url":"data:image/png;base64,' in readme
    assert "data_path=\"data/my_images.jsonl\"" in readme
    assert "azure_openai_jsonl.py" in readme


def test_visual_spatial_entrypoints_are_model_agnostic() -> None:
    recipe_dir = Path(train_azure.__file__).parent
    source = "\n".join(
        (recipe_dir / filename).read_text(encoding="utf-8")
        for filename in (
            "__init__.py",
            "sample_azure.py",
            "train_azure.py",
            "train_rft_azure.py",
            "train_sft_azure.py",
        )
    )

    # The default ID is the only model-specific detail; execution stays generic.
    assert "muse" not in source.replace("meta-models/Muse-Glimmer-30B", "").casefold()


@pytest.mark.asyncio
async def test_visual_spatial_sft_preflight_covers_train_and_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict = {}

    class FakeDatasetBuilder:
        def __init__(self, **kwargs):
            captured["builder"] = kwargs

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured["evaluator"] = kwargs

        def preflight(self, *, model_context_length: int):
            captured["evaluation_context"] = model_context_length
            return type(
                "Summary",
                (),
                {"examples": 81, "max_total_tokens": 200},
            )()

    monkeypatch.setattr(train_sft_azure, "VisualSpatialSftBuilder", FakeDatasetBuilder)
    monkeypatch.setattr(
        train_sft_azure,
        "VisualSpatialAccuracyEvaluator",
        FakeEvaluator,
    )
    monkeypatch.setattr(
        train_sft_azure.train,
        "prepare_datasets",
        lambda config: ([object(), object()], None),
    )
    monkeypatch.setattr(
        train_sft_azure.model_info,
        "require_vision_language_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        train_sft_azure.model_info,
        "get_recommended_renderer_name",
        lambda _model_name: "test_renderer",
    )

    await train_sft_azure.cli_main(
        train_sft_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            model_context_length=4096,
            task_variant="count_category",
            max_train_examples=160,
            preflight_only=True,
            log_path=str(tmp_path / "run"),
            behavior_if_log_dir_exists="raise",
        )
    )

    assert captured["builder"]["task_variant"] == "count_category"
    assert captured["builder"]["max_train_examples"] == 160
    assert captured["evaluator"]["task_variant"] == "count_category"
    assert captured["evaluation_context"] == 4096


@pytest.mark.asyncio
async def test_visual_spatial_preflight_only_builds_dedicated_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeSplit:
        async def preflight(self, *, max_tokens: int, model_context_length: int):
            captured.setdefault("preflight", []).append(
                (max_tokens, model_context_length)
            )
            return type(
                "Summary",
                (),
                {
                    "split": "visual_spatial/test",
                    "examples": 1,
                    "max_prompt_tokens": 10,
                    "max_total_tokens": 20,
                },
            )()

    class FakeDatasetBuilder:
        def __init__(self, **kwargs):
            captured["builder"] = kwargs

        async def __call__(self):
            return FakeSplit(), FakeSplit()

    monkeypatch.setattr(
        train_rft_azure,
        "VisualSpatialRLDatasetBuilder",
        FakeDatasetBuilder,
    )
    monkeypatch.setattr(
        train_rft_azure.model_info,
        "require_vision_language_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        train_rft_azure.model_info,
        "get_recommended_renderer_name",
        lambda _model_name: "test_renderer",
    )
    monkeypatch.setattr(
        train_rft_azure.model_info,
        "get_model_context_length",
        lambda _model_name: 4096,
    )

    await train_rft_azure.cli_main(
        train_rft_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            preflight_only=True,
            images_per_example=4,
            max_tokens=64,
            eval_max_tokens=128,
        )
    )

    assert captured["builder"]["images_per_example"] == 4
    assert captured["builder"]["data_path"] is None
    assert captured["builder"]["num_epochs"] == 1
    assert captured["builder"]["prompt_cache_max_mb"] == 128
    assert captured["builder"]["model_name"] == "example/vision-language-model"
    assert captured["builder"]["tokenizer_name"] == "example/vision-language-model"
    assert captured["preflight"] == [(128, 4096), (128, 4096)]


@pytest.mark.asyncio
async def test_jsonl_log_policy_runs_before_cache_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class FakeSplit:
        async def preflight(self, **_kwargs):
            return type(
                "Summary",
                (),
                {
                    "split": "test",
                    "examples": 1,
                    "max_prompt_tokens": 1,
                    "max_total_tokens": 2,
                },
            )()

    class FakeDatasetBuilder:
        def __init__(self, **_kwargs):
            pass

        async def __call__(self):
            events.append("prepare_cache")
            return (FakeSplit(),)

    monkeypatch.setattr(
        train_rft_azure,
        "VisualSpatialRLDatasetBuilder",
        FakeDatasetBuilder,
    )
    monkeypatch.setattr(
        train_rft_azure.cli_utils,
        "check_log_dir",
        lambda *_args, **_kwargs: events.append("check_log_dir"),
    )
    monkeypatch.setattr(
        train_rft_azure.model_info,
        "require_vision_language_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        train_rft_azure.model_info,
        "get_recommended_renderer_name",
        lambda _model_name: "test_renderer",
    )
    monkeypatch.setattr(
        train_rft_azure.model_info,
        "get_model_context_length",
        lambda _model_name: 4096,
    )

    await train_rft_azure.cli_main(
        train_rft_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            data_path="data.jsonl",
            log_path=str(tmp_path / "run"),
            preflight_only=True,
        )
    )

    assert events == ["check_log_dir", "prepare_cache"]


def test_visual_spatial_dataset_repeats_batches_across_epochs() -> None:
    dataset = datasets.Dataset.from_dict(
        {
            "question": ["q0", "q1", "q2"],
            "answer": ["a0", "a1", "a2"],
            "image": ["i0", "i1", "i2"],
        }
    )

    class Renderer:
        def build_generation_prompt(self, _messages):
            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

    rl_dataset = VisualSpatialRLDataset(
        dataset,
        batch_size=2,
        group_size=1,
        renderer=Renderer(),
        dataset_name="visual_spatial/train",
        num_epochs=3,
    )

    assert len(rl_dataset) == 6
    assert [len(rl_dataset.get_batch(index)) for index in range(len(rl_dataset))] == [
        2,
        1,
        2,
        1,
        2,
        1,
    ]


@pytest.mark.asyncio
async def test_absolute_difference_dataset_rewards_one_integer_answer() -> None:
    dataset = datasets.Dataset.from_dict(
        {
            "question": ["How many circles?", "How many squares?"],
            "answer": ["5", "2"],
            "image": ["first-image", "second-image"],
        }
    )
    rendered_messages = []

    class Renderer:
        def build_generation_prompt(self, messages):
            rendered_messages.append(messages)
            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return {"role": "assistant", "content": "3"}, True

    rl_dataset = VisualSpatialRLDataset(
        dataset,
        batch_size=1,
        group_size=1,
        renderer=Renderer(),
        dataset_name="visual_spatial/train",
        images_per_example=2,
        task_variant="absolute_difference",
        max_tokens=16,
    )

    environment = (await rl_dataset.get_batch(0)[0].make_envs())[0]
    await environment.initial_observation()
    result = await environment.step([7])

    assert rendered_messages[0][0]["content"][:2] == [
        {"type": "image", "image": "first-image"},
        {"type": "image", "image": "second-image"},
    ]
    assert result.reward == 1.0
    assert result.metrics == {
        "correct": 1.0,
        "parsed": 1.0,
        "single_integer_answer": 1.0,
        "truncated": 0.0,
    }


@pytest.mark.asyncio
async def test_count_category_dataset_rewards_learned_code_not_count() -> None:
    dataset = datasets.Dataset.from_dict(
        {"question": ["How many circles?"], "answer": ["2"], "image": ["image"]}
    )

    class Renderer:
        def build_generation_prompt(self, _messages):
            return ModelInput(chunks=[ModelInputChunk(tokens=[1])])

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return {"role": "assistant", "content": "0"}, True

    rl_dataset = VisualSpatialRLDataset(
        dataset,
        batch_size=1,
        group_size=1,
        renderer=Renderer(),
        dataset_name="visual_spatial/train",
        task_variant="count_category",
        max_tokens=16,
    )

    environment = (await rl_dataset.get_batch(0)[0].make_envs())[0]
    result = await environment.step([7])

    assert result.reward == 1.0
    assert result.logs["example_id"]


@pytest.mark.asyncio
async def test_prepared_prompt_is_reused_across_rollouts_and_epochs() -> None:
    dataset = datasets.Dataset.from_dict(
        {"question": ["question"], "answer": ["answer"], "image": ["image"]}
    )

    class CountingRenderer:
        def __init__(self):
            self.build_calls = 0

        def build_generation_prompt(self, _messages):
            self.build_calls += 1
            return ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])])

        def get_stop_sequences(self):
            return []

    renderer = CountingRenderer()
    rl_dataset = VisualSpatialRLDataset(
        dataset,
        batch_size=1,
        group_size=5,
        renderer=renderer,
        dataset_name="visual_spatial/train",
        num_epochs=10,
    )

    await rl_dataset.preflight(max_tokens=4, model_context_length=32)
    for batch_index in range(len(rl_dataset)):
        builders = rl_dataset.get_batch(batch_index)
        environments = await builders[0].make_envs()
        observations = [await environment.initial_observation() for environment in environments]
        assert all(observation[0].chunks[0].tokens == [1, 2, 3] for observation in observations)

    assert renderer.build_calls == 1


def test_trajectory_to_data_preserves_image_chunks_and_alignment() -> None:
    prompt = ModelInput(
        chunks=[
            ModelInputChunk(tokens=[1]),
            ImageChunk(data=b"\xff\xd8\xffjpeg", format="jpeg", expected_tokens=3),
            ModelInputChunk(tokens=[2]),
        ]
    )
    trajectory = Trajectory(
        transitions=[
            Transition(
                ob=prompt,
                ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[-0.2, -0.1]),
                reward=1.0,
                episode_done=True,
            )
        ],
        final_ob=ModelInput(chunks=[]),
    )

    datum = trajectory_to_data(trajectory, traj_advantage=0.5)[0]

    assert isinstance(datum.model_input.chunks[1], ImageChunk)
    assert len(datum.loss_fn_inputs["target_tokens"].data) == 6
    assert datum.loss_fn_inputs["mask"].data == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert datum.loss_fn_inputs["advantages"].data[-2:] == [0.5, 0.5]


@pytest.mark.asyncio
async def test_visual_spatial_preflight_rejects_oversized_prompt() -> None:
    dataset = datasets.Dataset.from_dict(
        {"question": ["question"], "answer": ["answer"], "image": ["image"]}
    )

    image_buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(image_buffer, format="PNG")
    image_data = image_buffer.getvalue()

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            return ModelInput(
                chunks=[
                    ImageChunk(data=image_data, format="png", expected_tokens=10),
                    ImageChunk(data=image_data, format="png", expected_tokens=10),
                    ImageChunk(data=image_data, format="png", expected_tokens=10),
                    ModelInputChunk(tokens=list(range(5))),
                ]
            )

        def get_stop_sequences(self):
            return []

    rl_dataset = VisualSpatialRLDataset(
        dataset,
        batch_size=1,
        group_size=1,
        renderer=FakeRenderer(),
        dataset_name="visual_spatial/train",
    )

    with pytest.raises(
        ValueError,
        match=r"exceeding the model context length of 25 tokens by 14",
    ) as exc_info:
        await rl_dataset.preflight(max_tokens=4, model_context_length=25)

    message = str(exc_info.value)
    assert "the prompt may contain at most 21 tokens" in message
    assert "3 images consuming 30 tokens" in message
    assert "remove approximately 2 of the largest context images" in message
    assert "reduce image resolution" in message
    assert "reducing max_tokens alone is insufficient" in message