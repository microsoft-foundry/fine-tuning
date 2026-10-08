"""Training-tier parsing and session forwarding across Azure recipe entrypoints."""

import ast
import sys
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import chz
import pytest

from azure.ai.finetuningsessions.aio import FineTuningSessionClient


def _internal_training_entrypoints(recipes_dir):
    """Keep source-only recipe coverage without naming those recipes in exports."""
    paths = {
        path
        for marker in recipes_dir.rglob(".internal_only")
        for path in marker.parent.rglob("train_azure.py")
    }
    return tuple(
        ".".join(path.relative_to(recipes_dir).with_suffix("").parts)
        for path in sorted(paths)
        if any(
            isinstance(node, ast.ClassDef) and node.name == "CLIConfig"
            for node in ast.parse(path.read_text(encoding="utf-8")).body
        )
    )


ASYNC_ENTRYPOINTS = (
    "code_rl.train_azure",
    "math_rl.train_azure",
    "mcp_image_tool.train_azure",
    "search_tool.train_azure",
    "tool_rl.train_azure",
    "tool_ner_rl.train_azure",
    "distillation.on_policy_distillation_azure",
    "visual_spatial.train_sft_azure",
    "visual_spatial.train_rft_azure",
    *_internal_training_entrypoints(
        Path(__file__).resolve().parents[1] / "interactive_training" / "recipes"
    ),
)
CONFIG_ENTRYPOINTS = (
    *ASYNC_ENTRYPOINTS,
    "preference.dpo.train_azure",
    "tulu3_sft.train_azure",
    "visual_spatial.train_azure",
    "tool_ner_rl.eval_azure",
)
TRAINING_TYPES = (
    (None, None),
    ("GlobalStandard", "GlobalStandard"),
    ("globalstandard", "GlobalStandard"),
    ("GLOBALSTANDARD", "GlobalStandard"),
)
UNAVAILABLE_TRAINING_TYPES = (
    "DeveloperTier", "developertier",
    "DatazoneStandard", "DataZoneStandard", "datazonestandard",
)
INVALID_TRAINING_TYPES = (*UNAVAILABLE_TRAINING_TYPES, "developer-tier", "", "unknown")


@pytest.mark.parametrize("training_type", UNAVAILABLE_TRAINING_TYPES)
def test_unavailable_training_type_raises_before_session_creation(training_type):
    from interactive_training.training_types import normalize_training_type

    with pytest.raises(ValueError, match="currently unavailable"):
        normalize_training_type(training_type)


def test_internal_tier_coverage_uses_export_markers(tmp_path):
    for relative in ("private/train_azure.py", "private/nested/train_azure.py", "public/train_azure.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("class CLIConfig:\n    pass\n", encoding="utf-8")
    (tmp_path / "private" / ".internal_only").touch()
    assert _internal_training_entrypoints(tmp_path) == (
        "private.nested.train_azure", "private.train_azure",
    )


def test_all_recipe_cli_configs_have_training_type_coverage():
    recipes = import_module("interactive_training.recipes")
    recipes_dir = Path(next(iter(recipes.__path__)))
    discovered = set()
    for path in recipes_dir.rglob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.ClassDef) and node.name == "CLIConfig"
            for node in module.body
        ):
            discovered.add(".".join(path.relative_to(recipes_dir).with_suffix("").parts))

    assert discovered == set(CONFIG_ENTRYPOINTS)


@pytest.mark.parametrize("entrypoint", CONFIG_ENTRYPOINTS)
@pytest.mark.parametrize(("training_type", "expected_training_type"), TRAINING_TYPES)
def test_recipe_cli_accepts_optional_training_type(
    entrypoint, training_type, expected_training_type
):
    recipe = import_module(f"interactive_training.recipes.{entrypoint}")
    argv = [
        "project_endpoint=http://localhost:8000",
        "model_name=Qwen/Qwen3-32B",
    ]
    if training_type is not None:
        argv.append(f"training_type={training_type}")

    config = chz.Blueprint(recipe.CLIConfig).make_from_argv(argv)

    assert config.training_type == expected_training_type


@pytest.mark.parametrize("entrypoint", CONFIG_ENTRYPOINTS)
@pytest.mark.parametrize("training_type", INVALID_TRAINING_TYPES)
def test_recipe_cli_rejects_invalid_training_type(entrypoint, training_type):
    recipe = import_module(f"interactive_training.recipes.{entrypoint}")
    argv = [
        "project_endpoint=http://localhost:8000",
        "model_name=Qwen/Qwen3-32B",
        f"training_type={training_type}",
    ]

    with pytest.raises(chz.blueprint.InvalidBlueprintArg):
        chz.Blueprint(recipe.CLIConfig).make_from_argv(argv)


def _mock_visual_preflight(monkeypatch, recipe, *, supervised):
    monkeypatch.setattr(
        recipe.model_info, "require_vision_language_model", lambda *a, **k: None
    )
    summary = SimpleNamespace(
        split="test",
        examples=1,
        max_prompt_tokens=8,
        max_total_tokens=16,
    )
    if supervised:
        monkeypatch.setattr(recipe, "VisualSpatialSftBuilder", MagicMock())
        evaluator = MagicMock()
        evaluator.preflight.return_value = summary
        monkeypatch.setattr(
            recipe, "VisualSpatialAccuracyEvaluator", MagicMock(return_value=evaluator)
        )
        monkeypatch.setattr(
            recipe.train, "prepare_datasets", lambda _config: ([object()], None)
        )
        monkeypatch.setattr(recipe, "get_tokenizer", MagicMock())
        monkeypatch.setattr(
            recipe, "AzureSDKTrainingClient", MagicMock(return_value=AsyncMock())
        )
    else:
        split = SimpleNamespace(preflight=AsyncMock(return_value=summary))
        builder = AsyncMock(return_value=(split, split))
        monkeypatch.setattr(
            recipe, "VisualSpatialRLDatasetBuilder", MagicMock(return_value=builder)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("entrypoint", "checkpoint", "training_overrides"), [
    *((entrypoint, None, {}) for entrypoint in ASYNC_ENTRYPOINTS),
    ("mcp_image_tool.train_azure", "model_source/warmup", {}),
    ("mcp_image_tool.train_azure", None, {
        "loss_fn": "ppo", "loss_fn_config": {"clip_low_threshold": 0.1},
        "num_substeps": 2, "compute_post_kl": True,
        "max_tool_calls": 5,
        "freeze_vision_tower": False, "freeze_multi_modal_projector": False,
    }),
])
@pytest.mark.parametrize(("training_type", "expected_training_type"), TRAINING_TYPES)
async def test_recipe_forwards_training_type(
    monkeypatch, tmp_path, entrypoint, checkpoint, training_overrides,
    training_type, expected_training_type
):
    recipe = import_module(f"interactive_training.recipes.{entrypoint}")
    create_session = AsyncMock(return_value="session_training_type_test")
    monkeypatch.setattr(FineTuningSessionClient, "create_session", create_session)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", AsyncMock())
    monkeypatch.setattr(FineTuningSessionClient, "close", AsyncMock())
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    supervised = entrypoint == "visual_spatial.train_sft_azure"
    training_main = AsyncMock()
    if entrypoint != "mcp_image_tool.train_azure":
        monkeypatch.setattr(
            recipe.train if supervised else recipe.rl_azure, "main", training_main
        )
    extra_config = dict(training_overrides)
    if entrypoint.startswith("visual_spatial."):
        _mock_visual_preflight(monkeypatch, recipe, supervised=supervised)
        extra_config["model_context_length"] = 4096
    if entrypoint == "code_rl.train_azure":
        monkeypatch.setattr(recipe, "assert_sandbox_reachable", AsyncMock())
    if "sandbox_kind" in chz.chz_fields(recipe.CLIConfig):
        extra_config["sandbox_kind"] = "memory"
    if entrypoint == "mcp_image_tool.train_azure":
        extra_config["load_checkpoint_path"] = checkpoint
        monkeypatch.setattr(
            recipe.model_info,
            "require_vision_language_model",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(recipe, "get_tokenizer", MagicMock())
        dataset_builder = AsyncMock(return_value=(MagicMock(), MagicMock()))
        monkeypatch.setattr(
            recipe, "MCPImageToolDatasetBuilder", MagicMock(return_value=dataset_builder)
        )

        async def record_evaluations(config, *args, **kwargs):
            (Path(config.log_path) / "evaluations.jsonl").write_text(
                "", encoding="utf-8"
            )
            return {}

        training_main.side_effect = record_evaluations
        monkeypatch.setattr(recipe, "train_mixed", training_main)
        warmup = AsyncMock()
        monkeypatch.setattr(recipe, "train_warmup", warmup)
        evaluate = AsyncMock(return_value={"test/env/all/total_episodes": 30})
        monkeypatch.setattr(recipe, "evaluate_repeated", evaluate)
        monkeypatch.setattr(
            recipe, "MCPImageTrainingClient", MagicMock(return_value=AsyncMock())
        )

        @asynccontextmanager
        async def mcp_session(_config):
            yield object()

        monkeypatch.setattr(recipe, "_mcp_session", mcp_session)
        monkeypatch.setattr(
            recipe,
            "tools_from_mcp_session",
            AsyncMock(return_value=[SimpleNamespace(name="reveal_alien_color")]),
        )

    user_metadata = {"DeveloperTier": True, "experimentName": "tier-test"}
    supports_metadata = "user_metadata" in chz.chz_fields(recipe.CLIConfig)
    if supports_metadata:
        extra_config["user_metadata"] = user_metadata
    config = recipe.CLIConfig(
        project_endpoint="http://localhost:8000",
        model_name="Qwen/Qwen3-32B",
        tokenizer_name="Qwen/Qwen3-32B",
        renderer_name="qwen3_disable_thinking",
        log_path=str(tmp_path / "run"),
        behavior_if_log_dir_exists="raise",
        training_type=training_type,
        **extra_config,
    )

    await recipe.cli_main(config)

    create_session.assert_awaited_once()
    assert create_session.await_args.kwargs["training_type"] == expected_training_type
    if supports_metadata:
        assert create_session.await_args.kwargs["user_metadata"] == user_metadata
    else:
        assert "user_metadata" not in create_session.await_args.kwargs
    training_main.assert_awaited_once()
    if entrypoint == "mcp_image_tool.train_azure":
        if checkpoint is None:
            warmup.assert_awaited_once()
            assert warmup.await_args.kwargs["epochs"] == config.warmup_epochs
            evaluate.assert_awaited_once()
            assert create_session.await_args.kwargs["from_checkpoint"] is None
        else:
            warmup.assert_not_awaited()
            evaluate.assert_not_awaited()
            source = create_session.await_args.kwargs["from_checkpoint"]
            assert source.source_session_id == "session_source"
            assert source.checkpoint_id == "warmup"
        assert training_main.await_args.kwargs["task_count"] == 60
        training_config = training_main.await_args.args[0]
        for name, default in {
            "loss_fn": "importance_sampling", "loss_fn_config": None,
            "num_substeps": 1, "compute_post_kl": False,
        }.items():
            assert getattr(training_config, name) == training_overrides.get(name, default)
        lora_config = create_session.await_args.kwargs["lora_config"]
        for name in ("freeze_vision_tower", "freeze_multi_modal_projector"):
            assert getattr(lora_config, name) == training_overrides.get(name, True)
        assert dataset_builder.await_count == 1
        builder_kwargs = recipe.MCPImageToolDatasetBuilder.call_args.kwargs
        assert builder_kwargs["max_tool_calls"] == training_overrides.get("max_tool_calls", 20)
        assert builder_kwargs["format_coef"] == 0.0
        FineTuningSessionClient.close_session.assert_awaited_once_with(
            "session_training_type_test"
        )
    elif not supervised:
        assert training_main.await_args.kwargs["training_type"] == training_type


@pytest.mark.parametrize(("training_type", "expected_training_type"), TRAINING_TYPES)
def test_visual_sampling_forwards_training_type(
    monkeypatch, training_type, expected_training_type
):
    from interactive_training.recipes.visual_spatial import sample_azure

    argv = [
        "sample_azure",
        "--project-endpoint",
        "http://localhost:8000",
        "--model-name",
        "Qwen/Qwen3-32B",
        "--renderer-name",
        "qwen3_disable_thinking",
    ]
    if training_type is not None:
        argv.extend(["--training-type", training_type])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    row = {"question": "How many?", "answer": "1"}
    monkeypatch.setattr(sample_azure, "load_dataset", lambda **kwargs: ([], [row]))
    monkeypatch.setattr(sample_azure, "user_message", MagicMock())
    monkeypatch.setattr(
        sample_azure.model_info,
        "require_vision_language_model",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(sample_azure, "get_tokenizer", MagicMock())
    monkeypatch.setattr(sample_azure, "get_image_processor", MagicMock())
    renderer = MagicMock()
    renderer.parse_response.return_value = (
        {"role": "assistant", "content": "1"},
        True,
    )
    monkeypatch.setattr(sample_azure, "get_renderer", lambda *a, **k: renderer)
    monkeypatch.setattr(sample_azure, "FineTuningSessionClient", MagicMock())
    session = MagicMock()
    session.sample.return_value.sequences = [SimpleNamespace(tokens=[1])]
    create_session = MagicMock(return_value=session)
    monkeypatch.setattr(sample_azure.FineTuningSession, "create", create_session)

    sample_azure.main()

    create_session.assert_called_once()
    assert create_session.call_args.kwargs["training_type"] == expected_training_type
    session.close.assert_called_once()
