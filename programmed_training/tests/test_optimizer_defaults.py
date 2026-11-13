"""Recipe defaults preserve effective optimizer updates, not old ignored requests."""

import ast
from importlib import import_module
from pathlib import Path

import chz
import pytest
import torch
from programmed_training.optimizers import AdamConfig

RECIPE_LEARNING_RATES = {
    "code_rl.train_azure": {"learning_rate": 1e-5},
    "distillation.on_policy_distillation_azure": {"learning_rate": 1e-5},
    "llm_judge_rl.train_azure": {"learning_rate": 5e-6},
    "math_rl.train_azure": {"learning_rate": 1e-5},
    "mcp_image_tool.train_azure": {"learning_rate": 5e-6, "warmup_learning_rate": 2e-4},
    "preference.dpo.train_azure": {"learning_rate": 1e-5},
    "rl_numerics_check.train_azure": {"learning_rate": 4e-5},
    "search_tool.train_azure": {"learning_rate": 4e-5},
    "tool_ner_rl.train_azure": {"learning_rate": 1e-5},
    "tool_rl.train_azure": {"learning_rate": 1e-5},
    "tulu3_sft.train_azure": {"learning_rate": 5e-4},
    "visual_spatial.train_azure": {"sft_learning_rate": 3e-5, "rft_learning_rate": 3e-6},
    "visual_spatial.train_rft_azure": {"learning_rate": 1e-5},
    "visual_spatial.train_sft_azure": {"learning_rate": 1e-4},
}
EFFECTIVE_DEFAULTS = {
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "adam_weight_decay": 0.0,
}


def _recipe_config(entrypoint: str, overrides: dict[str, float]) -> AdamConfig:
    module = import_module(f"programmed_training.recipes.{entrypoint}")
    return chz.Blueprint(module.CLIConfig).make_from_argv([
        "project_endpoint=https://example.invalid/api/projects/test",
        *(f"{name}={value}" for name, value in overrides.items()),
    ])


def test_inventory_covers_every_training_recipe():
    root = Path(__file__).resolve().parents[1] / "programmed_training" / "recipes"
    discovered = {
        ".".join(path.relative_to(root).with_suffix("").parts)
        for path in root.rglob("*azure.py")
        if not path.name.startswith(("eval_", "sample_"))
        if any(
            isinstance(node, ast.ClassDef) and node.name == "CLIConfig"
            for node in ast.parse(path.read_text(encoding="utf-8")).body
        )
    }
    assert discovered == set(RECIPE_LEARNING_RATES)


@pytest.mark.parametrize("entrypoint", RECIPE_LEARNING_RATES)
@pytest.mark.parametrize("overrides", [
    {},
    {"adam_beta1": 0.6},
    {"adam_beta2": 0.8},
    {"adam_eps": 1e-4},
    {"adam_weight_decay": 0.15},
    {
        "adam_beta1": 0.6,
        "adam_beta2": 0.8,
        "adam_eps": 1e-4,
        "adam_weight_decay": 0.15,
    },
])
def test_every_recipe_inherits_defaults_and_accepts_overrides(entrypoint, overrides):
    config = _recipe_config(entrypoint, overrides)
    expected = EFFECTIVE_DEFAULTS | overrides
    assert isinstance(config, AdamConfig)
    assert config.adam_kwargs() == expected
    assert config.to_adam_params(2e-4).as_dict() == {
        "learning_rate": 2e-4,
        "beta1": expected["adam_beta1"],
        "beta2": expected["adam_beta2"],
        "eps": expected["adam_eps"],
        "weight_decay": expected["adam_weight_decay"],
    }


@pytest.mark.parametrize("entrypoint,rates", RECIPE_LEARNING_RATES.items())
def test_recipe_defaults_match_effective_optimizer_weights_and_state(entrypoint, rates):
    config = _recipe_config(entrypoint, {})
    assert {
        name: value
        for name, value in chz.asdict(config).items()
        if name.endswith("learning_rate")
    } == rates
    for learning_rate in rates.values():
        params = config.to_adam_params(learning_rate)
        actual = torch.nn.Parameter(torch.tensor([0.3, -0.7], dtype=torch.float64))
        reference = torch.nn.Parameter(actual.detach().clone())
        actual_optimizer = torch.optim.AdamW(
            [actual], lr=params.learning_rate, betas=(params.beta1, params.beta2),
            eps=params.eps, weight_decay=params.weight_decay,
        )
        reference_optimizer = torch.optim.AdamW(
            [reference], lr=learning_rate, betas=(0.9, 0.999),
            eps=1e-8, weight_decay=0.0,
        )
        for gradient in ([1.0, -0.5], [-0.3, 1.7], [0.2, -0.1]):
            actual.grad = torch.tensor(gradient, dtype=torch.float64)
            reference.grad = actual.grad.clone()
            actual_optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_optimizer.state_dict(), reference_optimizer.state_dict(),
                rtol=0, atol=0,
            )


@pytest.mark.parametrize("entrypoint", RECIPE_LEARNING_RATES)
def test_recipe_forwards_adam_overrides_to_each_training_stage(entrypoint):
    module = import_module(f"programmed_training.recipes.{entrypoint}")
    source = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    constructors = [
        node for node in ast.walk(source)
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name) and node.func.id == "Config"
            or isinstance(node.func, ast.Attribute) and (
                node.func.attr == "Config"
                or node.func.attr == "CLIConfig"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"train_sft_azure", "train_rft_azure"}
            )
        )
    ]
    assert constructors, f"No training config found in {entrypoint}"
    for constructor in constructors:
        assert any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Attribute)
            and keyword.value.func.attr == "adam_kwargs"
            for keyword in constructor.keywords
        ), f"{entrypoint}:{constructor.lineno} does not forward Adam overrides"
