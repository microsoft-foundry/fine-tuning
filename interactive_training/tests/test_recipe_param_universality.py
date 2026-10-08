"""Enforce parameter universality across cookbook recipes.

Two independent concerns live here:

1. ``test_all_recipes_are_registered`` — discovers every recipe under
   ``interactive_training/recipes/`` that ships a ``train_azure.py`` entrypoint and
    fails if it is not classified in :data:`RECIPE_KINDS`.  This is the tripwire
    that forces whoever adds a new recipe to declare whether it supports
    ``"rl"``, ``"sft"``, or both so the universality checks below actually cover
    every training entrypoint.

2. ``test_universal_params_present`` / ``test_rl_universal_params_present`` /
   ``test_sft_universal_params_present`` — assert that each recipe's
   ``CLIConfig`` surfaces the agreed-upon universal parameter contract:

     * :data:`UNIVERSAL_PARAMS`     — required in *every* recipe.
     * :data:`RL_ONLY_PARAMS`       — required additionally in every RL recipe.
     * :data:`SFT_ONLY_PARAMS`      — required additionally in every SFT recipe.

   Recipes may expose extra recipe-unique params; these tests only assert the
   required floor (subset), never an exact match.
    The MCP image-tool recipe has a fixed SFT-to-RL workflow: its explicit
    exceptions below must be absent, rather than exposed as ignored options.

The ``CLIConfig`` fields are extracted by importing each recipe's
``train_azure`` module and introspecting the ``chz`` config via
``chz.chz_fields``. Unlike static parsing, this resolves *inherited* fields, so
if universal params are later hoisted into a shared base config the contract
still sees them. The import step doubles as the recipe->config coverage check.

Run with::

    cd interactive_training && uv run pytest tests/test_recipe_param_universality.py -v
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import chz
import pytest

import interactive_training.recipes as _recipes_pkg

# ``recipes`` ships no __init__.py (namespace package), so ``__file__`` is None.
# Namespace packages still expose ``__path__``; use its first entry.
RECIPES_DIR = Path(next(iter(_recipes_pkg.__path__)))


# =============================================================================
# Recipe registry — the source of truth for which recipes exist and their kind.
# Adding a recipe dir with a train_azure.py but NOT listing it here fails
# ``test_all_recipes_are_registered``.
# =============================================================================

RECIPE_KINDS: dict[str, frozenset[str]] = {
    "code_rl": frozenset({"rl"}),
    "math_rl": frozenset({"rl"}),
    "mcp_image_tool": frozenset({"rl"}),
    "search_tool": frozenset({"rl"}),
    "tool_rl": frozenset({"rl"}),
    "tool_ner_rl": frozenset({"rl"}),
    "tulu3_sft": frozenset({"sft"}),
    "visual_spatial": frozenset({"rl", "sft"}),
}


# =============================================================================
# Parameter contract (target spec, post-unification).
#
# These sets are the enforced floor. See the unify-cookbook-params plan:
#   - max_steps, max_wall_clock_seconds, seed, max_train_examples,
#     max_test_examples were promoted to UNIVERSAL.
#   - infrequent_eval_every is intentionally NOT required (being removed from
#     tulu3_sft; the supervised loop keeps the mechanism).
# =============================================================================

UNIVERSAL_PARAMS: frozenset[str] = frozenset({
    # Platform / plumbing
    "project_endpoint",
    "verbose_http",
    "training_type",
    "user_metadata",
    "behavior_if_log_dir_exists",
    "log_path",
    "wandb_project",
    "wandb_name",
    # Core model / training
    "model_name",
    "tokenizer_name",
    "renderer_name",
    "lora_rank",
    "load_checkpoint_path",
    "learning_rate",
    "eval_every",
    "save_every",
    # Promoted to universal
    "seed",
    "max_steps",
    "max_wall_clock_seconds",
    "max_train_examples",
    "max_test_examples",
})

# Required in every RL recipe, on top of UNIVERSAL_PARAMS.
RL_ONLY_PARAMS: frozenset[str] = frozenset({
    "eval_strategy",
    "group_size",
    "groups_per_batch",
    "max_tokens",
    "temperature",
    "kl_penalty_coef",
    "num_substeps",
    "loss_fn",
    "loss_fn_config",
    "compute_post_kl",
    "max_steps_off_policy",
})

# Required in every SFT recipe, on top of UNIVERSAL_PARAMS.
SFT_ONLY_PARAMS: frozenset[str] = frozenset({
    "num_epochs",
    "lr_schedule",
    "batch_size",
    "max_length",
    "train_on_what",
})


# =============================================================================
# Helpers
# =============================================================================


def _discover_recipes() -> list[str]:
    """Recipe = an immediate subdir of recipes/ that ships a train_azure.py.

    Recipes carrying an ``.internal_only`` marker are internal to the team and
    are excluded from the public repo by scripts/sync_finetuning_cookbook.py, so
    they are exempt from the public param-universality contract here too.
    """
    return sorted(
        p.name
        for p in RECIPES_DIR.iterdir()
        if p.is_dir()
        and (p / "train_azure.py").exists()
        and not (p / ".internal_only").exists()
    )


def _discover_specialized_sft_recipes() -> set[str]:
    """Recipes that expose SFT through a dedicated entrypoint module."""
    return {
        path.name
        for path in RECIPES_DIR.iterdir()
        if path.is_dir()
        and (path / "train_sft_azure.py").exists()
        and not (path / ".internal_only").exists()
    }


def _discover_specialized_rl_recipes() -> set[str]:
    """Recipes that expose RFT through a dedicated entrypoint module."""
    return {
        path.name
        for path in RECIPES_DIR.iterdir()
        if path.is_dir()
        and (path / "train_rft_azure.py").exists()
        and not (path / ".internal_only").exists()
    }


def _entrypoint_name(recipe: str, kind: str) -> str:
    """Return the entrypoint module that implements a recipe training kind."""
    if kind == "sft" and (RECIPES_DIR / recipe / "train_sft_azure.py").exists():
        return "train_sft_azure"
    if kind == "rl" and (RECIPES_DIR / recipe / "train_rft_azure.py").exists():
        return "train_rft_azure"
    return "train_azure"


def _import_cliconfig(recipe: str, kind: str):
    """Import a recipe-kind entrypoint and return its ``CLIConfig``.

    Doubles as the recipe->config coverage check: a recipe dir that ships a
    training entrypoint but whose module fails to import, or which exposes no
    ``CLIConfig``, fails here with a clear message.
    """
    entrypoint_name = _entrypoint_name(recipe, kind)
    module_name = f"{_recipes_pkg.__name__}.{recipe}.{entrypoint_name}"
    try:
        module = import_module(module_name)
    except Exception as exc:  # surface the real import error as a test failure
        raise AssertionError(f"Could not import {module_name}: {exc}") from exc
    cli_config = getattr(module, "CLIConfig", None)
    if cli_config is None:
        raise AssertionError(f"{module_name} defines no CLIConfig class")
    return cli_config


def _cliconfig_fields(recipe: str, kind: str) -> set[str]:
    """Field names of the recipe's ``CLIConfig``, including inherited fields."""
    return set(chz.chz_fields(_import_cliconfig(recipe, kind)))


def _recipes_of_kind(kind: str) -> list[str]:
    return sorted(recipe for recipe, kinds in RECIPE_KINDS.items() if kind in kinds)


def _recipe_kind_cases() -> list[tuple[str, str]]:
    return sorted(
        (recipe, kind)
        for recipe, kinds in RECIPE_KINDS.items()
        for kind in kinds
    )


# =============================================================================
# 1. Registration coverage
# =============================================================================


def test_all_recipes_are_registered():
    """Every recipe on disk must be classified, and vice-versa."""
    discovered = set(_discover_recipes())
    registered = set(RECIPE_KINDS)

    unregistered = discovered - registered
    assert not unregistered, (
        f"Recipe(s) {sorted(unregistered)} exist under recipes/ but are not "
        f"classified in RECIPE_KINDS. Add each with 'rl', 'sft', or both so "
        f"the param universality checks cover them."
    )

    stale = registered - discovered
    assert not stale, (
        f"RECIPE_KINDS references non-existent recipe(s): {sorted(stale)}. "
        f"Remove them or restore the recipe directory."
    )

    unclassified_sft = {
        recipe
        for recipe in _discover_specialized_sft_recipes()
        if "sft" not in RECIPE_KINDS.get(recipe, frozenset())
    }
    assert not unclassified_sft, (
        f"Recipe(s) {sorted(unclassified_sft)} ship train_sft_azure.py but are "
        f"not classified as 'sft' in RECIPE_KINDS."
    )

    unclassified_rl = {
        recipe
        for recipe in _discover_specialized_rl_recipes()
        if "rl" not in RECIPE_KINDS.get(recipe, frozenset())
    }
    assert not unclassified_rl, (
        f"Recipe(s) {sorted(unclassified_rl)} ship train_rft_azure.py but are "
        f"not classified as 'rl' in RECIPE_KINDS."
    )


def test_recipe_kinds_are_valid():
    valid_kinds = {"rl", "sft"}
    invalid = {
        recipe: sorted(kinds)
        for recipe, kinds in RECIPE_KINDS.items()
        if not kinds or not kinds <= valid_kinds
    }
    assert not invalid, (
        f"RECIPE_KINDS has empty or invalid kinds (expected rl/sft): {invalid}"
    )


@pytest.mark.parametrize(("recipe", "kind"), _recipe_kind_cases())
def test_recipe_has_cliconfig(recipe: str, kind: str):
    fields = _cliconfig_fields(recipe, kind)
    entrypoint_name = _entrypoint_name(recipe, kind)
    assert fields, f"{recipe}/{entrypoint_name}.py has an empty CLIConfig"


# =============================================================================
# 2. Parameter universality
# =============================================================================


FIXED_WORKFLOW_EXCEPTIONS = {
    "mcp_image_tool": frozenset({
        "wandb_project", "wandb_name", "eval_every", "save_every",
        "eval_strategy", "kl_penalty_coef", "max_steps_off_policy",
    }),
}


@pytest.mark.parametrize("recipe", FIXED_WORKFLOW_EXCEPTIONS)
def test_fixed_workflow_omits_unsupported_options(recipe: str):
    assert not FIXED_WORKFLOW_EXCEPTIONS[recipe] & _cliconfig_fields(recipe, "rl")


@pytest.mark.parametrize(("recipe", "kind"), _recipe_kind_cases())
def test_universal_params_present(recipe: str, kind: str):
    required = UNIVERSAL_PARAMS - FIXED_WORKFLOW_EXCEPTIONS.get(recipe, frozenset())
    missing = required - _cliconfig_fields(recipe, kind)
    assert not missing, (
        f"Recipe '{recipe}' ({kind}) is missing universal params: {sorted(missing)}"
    )


@pytest.mark.parametrize("recipe", _recipes_of_kind("rl"))
def test_rl_universal_params_present(recipe: str):
    required = RL_ONLY_PARAMS - FIXED_WORKFLOW_EXCEPTIONS.get(recipe, frozenset())
    missing = required - _cliconfig_fields(recipe, "rl")
    assert not missing, (
        f"RL recipe '{recipe}' is missing rl-universal params: {sorted(missing)}"
    )


@pytest.mark.parametrize("recipe", [
    recipe for recipe in _recipes_of_kind("rl")
    if "eval_strategy" not in FIXED_WORKFLOW_EXCEPTIONS.get(recipe, frozenset())
])
def test_rl_recipes_default_to_step_evaluation(recipe: str):
    # RL recipe CLIConfigs default to step-based evaluation (master parity).
    cli_config = _import_cliconfig(recipe, "rl")(
        project_endpoint="https://example.test",
        model_name="example/model",
    )
    assert cli_config.eval_strategy == "steps"
    assert cli_config.eval_every == 20


@pytest.mark.parametrize("recipe", _recipes_of_kind("sft"))
def test_sft_universal_params_present(recipe: str):
    missing = SFT_ONLY_PARAMS - _cliconfig_fields(recipe, "sft")
    assert not missing, (
        f"SFT recipe '{recipe}' is missing sft-universal params: {sorted(missing)}"
    )


# =============================================================================
# 3. Wiring — a declared universal param must actually be *consumed*.
#
# ``chz`` does not complain about a field that is declared but never read, so a
# recipe could satisfy the presence checks above while silently dropping the
# param on the floor (declared in ``CLIConfig`` but never threaded into the
# ``Config`` / dataset builder). This guard fails when a universal param is not
# referenced as ``cli_config.<param>`` anywhere in the recipe's entrypoint —
# the cheap, high-signal tripwire for exactly that plumbing mistake.
# =============================================================================


@pytest.mark.parametrize(("recipe", "kind"), _recipe_kind_cases())
def test_universal_params_are_wired(recipe: str, kind: str):
    entrypoint_name = _entrypoint_name(recipe, kind)
    source = (RECIPES_DIR / recipe / f"{entrypoint_name}.py").read_text(
        encoding="utf-8"
    )
    required = UNIVERSAL_PARAMS - FIXED_WORKFLOW_EXCEPTIONS.get(recipe, frozenset())
    unwired = sorted(p for p in required if f"cli_config.{p}" not in source)
    assert not unwired, (
        f"Recipe '{recipe}' ({kind}) declares universal params but never consumes them "
        f"(no `cli_config.<param>` reference): {unwired}. Thread them into the "
        f"Config / dataset builder, or they silently do nothing."
    )


# =============================================================================
# 4. Backend coverage — the shared training Configs / builders must expose the
# promoted fields too. A recipe can add a CLIConfig knob but forget to add the
# corresponding field on the backend Config or dataset builder; ``chz`` would
# then reject the keyword at call time. These assert the backend actually grew
# the fields the recipes now forward.
# =============================================================================


def test_backend_configs_expose_promoted_fields():
    from interactive_training.rl.train import Config as RLConfig
    from interactive_training.supervised.train import Config as SFTConfig
    from interactive_training.recipes.tulu3_sft.chat_datasets import Tulu3Builder

    rl_fields = set(chz.chz_fields(RLConfig))
    assert {"max_steps", "max_wall_clock_seconds"} <= rl_fields, (
        f"rl.train.Config missing budget fields: "
        f"{sorted({'max_steps', 'max_wall_clock_seconds'} - rl_fields)}"
    )

    sft_fields = set(chz.chz_fields(SFTConfig))
    assert {"max_steps", "max_wall_clock_seconds", "seed"} <= sft_fields, (
        f"supervised.train.Config missing budget/seed fields: "
        f"{sorted({'max_steps', 'max_wall_clock_seconds', 'seed'} - sft_fields)}"
    )

    tulu_fields = set(chz.chz_fields(Tulu3Builder))
    assert "seed" in tulu_fields, "Tulu3Builder is missing the seed field"
