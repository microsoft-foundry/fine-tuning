"""Public recipes share supported defaults and model-aware rendering."""

import ast
from importlib import import_module
from pathlib import Path
import sys

import chz
import pytest

from interactive_training import model_info


TEXT_ENTRYPOINTS = (
    "code_rl.train_azure",
    "distillation.on_policy_distillation_azure",
    "math_rl.train_azure",
    "preference.dpo.train_azure",
    "search_tool.train_azure",
    "tool_rl.train_azure",
    "tool_ner_rl.train_azure",
    "tool_ner_rl.eval_azure",
    "tulu3_sft.train_azure",
)

VISION_MODEL = "meta-models/Muse-Glimmer-30B"
VISION_ENTRYPOINTS = (
    "mcp_image_tool.train_azure",
    "visual_spatial.train_azure",
    "visual_spatial.train_sft_azure",
    "visual_spatial.train_rft_azure",
)


def _public_cli_entrypoints(recipes_dir):
    # Use the export markers, not a list of internal recipe names. Internal
    # recipe behavior remains covered by the recipe-specific test suites.
    internal_dirs = {marker.parent for marker in recipes_dir.rglob(".internal_only")}
    return {
        ".".join(path.relative_to(recipes_dir).with_suffix("").parts)
        for path in recipes_dir.rglob("*.py")
        if not any(path.is_relative_to(directory) for directory in internal_dirs)
        if any(
            isinstance(node, ast.ClassDef) and node.name == "CLIConfig"
            for node in ast.parse(path.read_text(encoding="utf-8")).body
        )
    }


def test_default_inventory_covers_all_recipe_cli_configs():
    recipes_dir = Path(__file__).resolve().parents[1] / "interactive_training" / "recipes"
    assert _public_cli_entrypoints(recipes_dir) == set(TEXT_ENTRYPOINTS + VISION_ENTRYPOINTS)


def test_default_inventory_excludes_only_marker_scoped_recipes(tmp_path):
    for relative in (
        "public/train_azure.py",
        "private/train_azure.py",
        "private/nested/train_azure.py",
        "private_sibling/train_azure.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("class CLIConfig:\n    pass\n", encoding="utf-8")
    (tmp_path / "private" / ".internal_only").touch()

    assert _public_cli_entrypoints(tmp_path) == {
        "public.train_azure",
        "private_sibling.train_azure",
    }


@pytest.mark.parametrize("entrypoint", TEXT_ENTRYPOINTS)
def test_public_text_recipe_defaults_to_qwen38(entrypoint):
    module = import_module(f"interactive_training.recipes.{entrypoint}")
    config = chz.Blueprint(module.CLIConfig).make_from_argv(
        ["project_endpoint=https://example.invalid/api/projects/test"]
    )
    assert config.model_name == "Qwen/Qwen3.8-27B"
    assert config.tokenizer_name is None
    assert config.renderer_name is None
    assert model_info.get_recommended_renderer_name(config.model_name) == "qwen3_8_low_reasoning"
    if entrypoint.startswith("distillation."):
        assert config.teacher_model is None


@pytest.mark.parametrize("entrypoint", TEXT_ENTRYPOINTS)
@pytest.mark.parametrize(
    "renderer_name",
    ["qwen3_8_low_reasoning", "qwen3_8_medium_reasoning", "qwen3_8", "qwen3_8_disable_thinking"],
)
def test_explicit_qwen38_renderer_override_is_preserved(entrypoint, renderer_name):
    module = import_module(f"interactive_training.recipes.{entrypoint}")
    config = chz.Blueprint(module.CLIConfig).make_from_argv(
        [
            "project_endpoint=https://example.invalid/api/projects/test",
            f"renderer_name={renderer_name}",
        ]
    )
    assert config.renderer_name == renderer_name


@pytest.mark.parametrize("entrypoint", TEXT_ENTRYPOINTS)
def test_explicit_supported_model_override_is_preserved(entrypoint):
    module = import_module(f"interactive_training.recipes.{entrypoint}")
    config = module.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        model_name="openai/gpt-oss-20b",
    )
    assert config.model_name == "openai/gpt-oss-20b"
    assert config.tokenizer_name is None
    assert config.renderer_name is None


@pytest.mark.parametrize(
    "model,renderer",
    [
        ("Qwen/Qwen3.8-27B", "qwen3_8_disable_thinking"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_disable_thinking"),
        ("Qwen/Qwen3-32B", "qwen3_disable_thinking"),
        ("openai/gpt-oss-20b", "gpt_oss_no_sysprompt"),
    ],
)
def test_non_thinking_preference_remains_model_specific(model, renderer):
    assert model_info.get_recommended_renderer_name(
        model, prefer_non_thinking=True
    ) == renderer


@pytest.mark.parametrize("entrypoint", VISION_ENTRYPOINTS)
def test_public_vision_recipe_defaults_to_muse(entrypoint):
    module = import_module(f"interactive_training.recipes.{entrypoint}")
    config = chz.Blueprint(module.CLIConfig).make_from_argv(
        ["project_endpoint=https://example.invalid/api/projects/test"]
    )
    assert config.model_name == VISION_MODEL
    assert config.tokenizer_name is None
    assert config.renderer_name in (None, "muse_glimmer_low_reasoning")
    assert model_info.get_model_attributes(config.model_name).is_vl


@pytest.mark.parametrize("entrypoint", TEXT_ENTRYPOINTS + VISION_ENTRYPOINTS)
def test_explicit_model_tokenizer_and_renderer_overrides_are_preserved(entrypoint):
    module = import_module(f"interactive_training.recipes.{entrypoint}")
    config = chz.Blueprint(module.CLIConfig).make_from_argv([
        "project_endpoint=https://example.invalid/api/projects/test",
        f"model_name={VISION_MODEL}",
        f"tokenizer_name={VISION_MODEL}",
        "renderer_name=muse_glimmer_low_reasoning",
    ])
    assert config.model_name == config.tokenizer_name == VISION_MODEL
    assert config.renderer_name == "muse_glimmer_low_reasoning"


@pytest.mark.parametrize("override", [None, "example/vision-language-model"])
def test_visual_sampling_cli_default_and_override(monkeypatch, override):
    from interactive_training.recipes.visual_spatial import sample_azure

    argv = ["sample_azure", "--project-endpoint", "https://example.invalid"]
    if override is not None:
        argv.extend(["--model-name", override])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sample_azure, "load_dataset", lambda **kwargs: ([], [{}]))

    class CheckedModel(Exception):
        pass

    def check_model(model_name, **kwargs):
        assert model_name == (override or VISION_MODEL)
        raise CheckedModel

    monkeypatch.setattr(sample_azure.model_info, "require_vision_language_model", check_model)
    with pytest.raises(CheckedModel):
        sample_azure.main()


@pytest.mark.parametrize("entrypoint", ["train_azure", "eval_azure"])
@pytest.mark.parametrize(
    "checkpoint", ["session_abc/final", "session_abc/final"]
)
def test_ner_checkpoint_preserves_public_session_id(entrypoint, checkpoint):
    module = import_module(f"interactive_training.recipes.tool_ner_rl.{entrypoint}")
    assert module._parse_checkpoint_path(checkpoint) == ("session_abc", "final")


def test_manual_diagnostic_uses_selected_models_tokenizer(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tests import test_prompt_logprobs_e2e as diagnostic

    monkeypatch.setattr(sys, "argv", ["check-prompt-logprobs"])
    args = diagnostic.parse_args()
    assert args.model == "Qwen/Qwen3.8-27B"
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2]
    get_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(diagnostic.tokenizer_utils, "get_tokenizer", get_tokenizer)
    assert diagnostic.prompt_tokens_for_model(args) == [1, 2]
    get_tokenizer.assert_called_once_with(args.model)
    tokenizer.encode.assert_called_once_with(args.prompt, add_special_tokens=True)
    assert diagnostic.prompt_tokens_for_model(SimpleNamespace(prompt_tokens=[3, 4])) == [3, 4]