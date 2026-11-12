"""Every public training entrypoint enables recovery and exposes an opt-out."""

import ast
from importlib import import_module
from pathlib import Path

import chz
import pytest

from programmed_training.auto_resume import AutoResumeConfig
from tests.test_public_recipe_defaults import (
    QWEN3_32B_ENTRYPOINTS, TEXT_ENTRYPOINTS, VISION_ENTRYPOINTS, _public_cli_entrypoints,
)


TRAINING_ENTRYPOINTS = tuple(
    entrypoint
    for entrypoint in TEXT_ENTRYPOINTS + QWEN3_32B_ENTRYPOINTS + VISION_ENTRYPOINTS
    if entrypoint != "tool_ner_rl.eval_azure"
)


def test_all_public_training_entrypoints_are_in_recovery_inventory():
    recipes = Path(__file__).resolve().parents[1] / "programmed_training/recipes"
    assert set(TRAINING_ENTRYPOINTS) == (
        _public_cli_entrypoints(recipes) - {"tool_ner_rl.eval_azure"}
    )


@pytest.mark.parametrize("entrypoint", TRAINING_ENTRYPOINTS)
def test_public_recipe_recovery_config_and_wiring(entrypoint):
    module = import_module(f"programmed_training.recipes.{entrypoint}")
    assert issubclass(module.CLIConfig, AutoResumeConfig)
    arguments = ["project_endpoint=https://unit.invalid"]
    default = chz.Blueprint(module.CLIConfig).make_from_argv(arguments)
    assert default.auto_resume
    assert default.auto_resume_max_retries == 3
    assert default.resumable_error_codes == ("engine_dead",)
    disabled = chz.Blueprint(module.CLIConfig).make_from_argv(arguments + ["auto_resume=False"])
    assert not disabled.auto_resume
    configured = chz.Blueprint(module.CLIConfig).make_from_argv(
        arguments + ["auto_resume_max_retries=5"]
    )
    assert configured.auto_resume_max_retries == 5
    code = ast.parse(Path(module.__file__).read_text())
    calls = {
        node.func.attr for node in ast.walk(code)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "auto_resume"
    }
    assert calls & {"run", "run_rl", "run_supervised"} or entrypoint == "visual_spatial.train_azure"


def test_error_allowlist_can_be_set_from_cli():
    config = chz.Blueprint(AutoResumeConfig).make_from_argv([
        "auto_resume=True", "resumable_error_codes=engine_dead,worker_crashed",
    ])
    assert config.resumable_error_codes == ("engine_dead", "worker_crashed")


@pytest.mark.parametrize("limit", [-1, True, 1.5, "3", None])
def test_invalid_retry_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="auto_resume_max_retries"):
        AutoResumeConfig(auto_resume_max_retries=limit)
