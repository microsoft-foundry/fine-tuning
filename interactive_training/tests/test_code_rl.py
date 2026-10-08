"""Unit tests for the code_rl recipe.

These tests intentionally avoid GPU, network, dataset downloads and any
real sandbox execution. They target the small pure-function surface that
is most likely to silently break during a port:

* import wiring (all submodules resolve under interactive_training.recipes.code_rl)
* fenced-code-block extraction
* TACO ↔ LiveCodeBench test-case shape conversion
* metadata / test-list normalization
* DeepcoderReward arithmetic with the sandbox stubbed out
* CLIConfig fields used by experiments/run_all_benchmarks.sh
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from interactive_training.recipes.code_rl import (
    code_env,
    code_grading,
    deepcoder_tool,
    train_azure,
)
from interactive_training.recipes.code_rl.code_env import _ensure_dict, _normalize_tests
from interactive_training.recipes.code_rl.code_grading import (
    extract_code_from_model,
    postprocess_lcb_sample,
    taco_to_lcb_format,
)
from interactive_training.recipes.code_rl.deepcoder_tool import DeepcoderReward, DeepcoderTask
from interactive_training.recipes.code_rl.train_azure import CLIConfig


# ---------------------------------------------------------------------------
# Import wiring — catches missing modules or broken interactive_training.* paths
# ---------------------------------------------------------------------------


def test_recipe_submodules_import():
    """All recipe submodules must import cleanly (no missing names from the port)."""
    assert code_env.DeepcoderDatasetBuilder is not None
    assert code_grading.sandbox_check_correctness is not None
    assert deepcoder_tool.DeepcoderTool is not None
    assert train_azure.CLIConfig is not None


# ---------------------------------------------------------------------------
# extract_code_from_model
# ---------------------------------------------------------------------------


def test_extract_code_returns_none_when_no_fence():
    assert extract_code_from_model("just some prose, no code") is None


def test_extract_code_basic_python_fence():
    response = "Here is the answer:\n```python\nprint('hi')\n```\nThanks."
    assert extract_code_from_model(response) == "print('hi')"


def test_extract_code_unlanguaged_fence():
    response = "```\nx = 1\n```"
    assert extract_code_from_model(response) == "x = 1"


def test_extract_code_returns_last_block_when_multiple():
    response = "```python\nfirst()\n```\nthen\n```python\nlast()\n```"
    assert extract_code_from_model(response) == "last()"


# ---------------------------------------------------------------------------
# postprocess_lcb_sample
# ---------------------------------------------------------------------------


def test_postprocess_stdin_stdout_sample():
    sample = [
        {"input": "1\n", "output": "1\n", "testtype": "stdin_stdout", "metadata": {}},
        {"input": "2\n", "output": "2\n", "testtype": "stdin_stdout", "metadata": {}},
    ]
    out = postprocess_lcb_sample(sample)
    payload = json.loads(out["input_output"])
    assert payload["inputs"] == ["1\n", "2\n"]
    assert payload["outputs"] == ["1\n", "2\n"]
    assert "fn_name" not in payload


def test_postprocess_functional_sample_promotes_fn_name():
    sample = [
        {
            "input": "[1,2]",
            "output": "3",
            "testtype": "functional",
            "metadata": {"func_name": "add"},
        }
    ]
    payload = json.loads(postprocess_lcb_sample(sample)["input_output"])
    assert payload["fn_name"] == "add"


def test_postprocess_functional_sample_requires_fn_name():
    sample = [{"input": "x", "output": "y", "testtype": "functional", "metadata": {}}]
    with pytest.raises(AssertionError):
        postprocess_lcb_sample(sample)


# ---------------------------------------------------------------------------
# taco_to_lcb_format
# ---------------------------------------------------------------------------


def test_taco_to_lcb_stdin_stdout():
    tests = {"inputs": ["a", "b"], "outputs": ["1", "2"]}
    cases = taco_to_lcb_format(tests)
    assert len(cases) == 2
    assert cases[0] == {"input": "a", "output": "1", "metadata": {}, "testtype": "stdin_stdout"}
    assert cases[1]["testtype"] == "stdin_stdout"


def test_taco_to_lcb_functional_sets_fn_name():
    tests = {"inputs": ["[1]"], "outputs": ["1"], "fn_name": "ident"}
    cases = taco_to_lcb_format(tests)
    assert cases[0]["testtype"] == "functional"
    assert cases[0]["metadata"]["func_name"] == "ident"


def test_taco_to_lcb_unwraps_list_outputs():
    tests = {"inputs": ["x"], "outputs": [["wrapped"]]}
    cases = taco_to_lcb_format(tests)
    assert cases[0]["output"] == "wrapped"


# ---------------------------------------------------------------------------
# _ensure_dict / _normalize_tests
# ---------------------------------------------------------------------------


def test_ensure_dict_passes_through_dict():
    assert _ensure_dict({"a": 1}) == {"a": 1}


def test_ensure_dict_parses_json_string():
    assert _ensure_dict('{"a": 1}') == {"a": 1}


def test_ensure_dict_returns_empty_on_garbage():
    assert _ensure_dict("not json") == {}
    assert _ensure_dict(42) == {}
    assert _ensure_dict(None) == {}


def test_normalize_tests_taco_dict_is_converted():
    raw = {"inputs": ["a"], "outputs": ["b"]}
    normalized = _normalize_tests(raw, metadata={})
    assert len(normalized) == 1
    assert normalized[0]["input"] == "a"
    assert normalized[0]["output"] == "b"
    assert normalized[0]["testtype"] == "stdin_stdout"


def test_normalize_tests_promotes_metadata_func_name_into_test():
    raw = [{"input": "x", "output": "y", "testtype": "functional", "metadata": {}}]
    normalized = _normalize_tests(raw, metadata={"func_name": "solve"})
    assert normalized[0]["metadata"]["func_name"] == "solve"


def test_normalize_tests_drops_non_dict_entries():
    normalized = _normalize_tests([{"input": "x", "output": "y"}, "garbage", 5], metadata={})
    assert len(normalized) == 1
    assert normalized[0]["testtype"] == "stdin_stdout"


def test_normalize_tests_handles_json_string_input():
    raw = json.dumps([{"input": "x", "output": "y"}])
    normalized = _normalize_tests(raw, metadata={})
    assert len(normalized) == 1


def test_normalize_tests_returns_empty_on_unparseable_string():
    assert _normalize_tests("not json", metadata={}) == []


# ---------------------------------------------------------------------------
# DeepcoderReward — verify reward arithmetic with sandbox stubbed out.
#
# Formula (from DeepcoderReward.__call__):
#     reward = format_coef * (has_code_block - 1) + correct
# with format_coef defaulting to 0.1.
# ---------------------------------------------------------------------------


def _task() -> DeepcoderTask:
    return DeepcoderTask(problem="P", tests=[{"input": "x", "output": "y"}])


@pytest.mark.asyncio
async def test_code_env_forwards_failed_parse_reward(monkeypatch):
    renderer = MagicMock()
    captured: dict[str, float] = {}

    monkeypatch.setattr(code_env.tokenizer_utils, "get_tokenizer", MagicMock())
    monkeypatch.setattr(code_env, "get_renderer", MagicMock(return_value=renderer))
    monkeypatch.setattr(code_env, "_initial_messages", MagicMock(return_value=[]))

    def fake_build_agent_tool_env(**kwargs):
        captured["failed_parse_reward"] = kwargs["failed_parse_reward"]
        return MagicMock()

    monkeypatch.setattr(code_env, "build_agent_tool_env", fake_build_agent_tool_env)
    builder = code_env.DeepcoderEnvGroupBuilder(
        task=_task(),
        model_name="test-model",
        renderer_name="test-renderer",
        max_turns=2,
        group_size=1,
        sandbox_backend=None,
        failed_parse_reward=0.0,
    )

    await builder.make_envs()

    assert captured["failed_parse_reward"] == 0.0


def _patch_sandbox(monkeypatch, *, passed: bool):
    """Replace sandbox_check_correctness in deepcoder_tool with an async stub."""
    monkeypatch.setattr(
        deepcoder_tool,
        "sandbox_check_correctness",
        AsyncMock(return_value=(passed, {"stub": True})),
    )


@pytest.mark.asyncio
async def test_reward_passing_code_block_is_one(monkeypatch):
    _patch_sandbox(monkeypatch, passed=True)
    reward_fn = DeepcoderReward(task=_task())
    history = [
        {"role": "user", "content": "solve it"},
        {"role": "assistant", "content": "```python\nprint(1)\n```"},
    ]
    reward, metrics = await reward_fn(history)
    # format=1, correct=1 → 0.1*(1-1) + 1.0 = 1.0
    assert reward == pytest.approx(1.0)
    assert metrics == {"format": 1.0, "correct": 1.0}


@pytest.mark.asyncio
async def test_reward_failing_code_block_keeps_format_credit(monkeypatch):
    _patch_sandbox(monkeypatch, passed=False)
    reward_fn = DeepcoderReward(task=_task())
    history = [{"role": "assistant", "content": "```python\nprint(1)\n```"}]
    reward, metrics = await reward_fn(history)
    # format=1, correct=0 → 0.1*(1-1) + 0 = 0.0
    assert reward == pytest.approx(0.0)
    assert metrics == {"format": 1.0, "correct": 0.0}


@pytest.mark.asyncio
async def test_reward_no_code_block_is_negative_format_penalty(monkeypatch):
    _patch_sandbox(monkeypatch, passed=True)  # should NOT be called
    reward_fn = DeepcoderReward(task=_task())
    history = [{"role": "assistant", "content": "I would write some code but I won't."}]
    reward, metrics = await reward_fn(history)
    # format=0, correct=0 → 0.1*(0-1) + 0 = -0.1
    assert reward == pytest.approx(-0.1)
    assert metrics == {"format": 0.0, "correct": 0.0}


@pytest.mark.asyncio
async def test_reward_no_assistant_message_is_zero(monkeypatch):
    _patch_sandbox(monkeypatch, passed=True)  # should NOT be called
    reward_fn = DeepcoderReward(task=_task())
    history = [{"role": "user", "content": "hi"}]
    reward, metrics = await reward_fn(history)
    assert reward == 0.0
    assert metrics == {"format": 0.0, "correct": 0.0}


@pytest.mark.asyncio
async def test_reward_picks_last_assistant_message(monkeypatch):
    _patch_sandbox(monkeypatch, passed=True)
    reward_fn = DeepcoderReward(task=_task())
    history = [
        {"role": "assistant", "content": "no code here"},  # earlier, ignored
        {"role": "user", "content": "try again"},
        {"role": "assistant", "content": "```\nfinal()\n```"},
    ]
    reward, metrics = await reward_fn(history)
    assert metrics["format"] == 1.0
    assert metrics["correct"] == 1.0
    assert reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLIConfig schema — pins the field set that run_all_benchmarks.sh uses.
# If a CLIConfig field is removed or renamed (e.g. the LossFnType union
# stops accepting "importance_sampling"), this test fails before the
# benchmark script does on a real cluster.
# ---------------------------------------------------------------------------


def test_train_azure_cli_config_accepts_run_all_benchmarks_args():
    """Mirror the exact kwargs run_code_rl() in run_all_benchmarks.sh passes."""
    cfg = CLIConfig(
        project_endpoint="https://example.azure.com/api/projects/x",
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3_disable_thinking",
        learning_rate=2e-5,
        temperature=1.0,
        max_tokens=1200,
        lora_rank=32,
        group_size=8,
        groups_per_batch=256,
        loss_fn="importance_sampling",
        remove_constant_reward_groups=True,
        seed=42,
        eval_every=10,
        save_every=10,
        log_path="/tmp/x",
        behavior_if_log_dir_exists="delete",
    )
    assert cfg.model_name == "Qwen/Qwen3-32B"
    assert cfg.groups_per_batch == 256
    assert cfg.lora_rank == 32
    assert cfg.remove_constant_reward_groups is True
    assert cfg.format_coef == 0.1
    assert cfg.failed_parse_reward == -0.1
    # project_endpoint has no default — confirm it is required.
    assert cfg.project_endpoint.startswith("https://")


def test_train_azure_cli_config_requires_project_endpoint():
    with pytest.raises(Exception):
        CLIConfig()  # type: ignore[call-arg]
