"""Unit tests for the code_rl recipe.

These tests intentionally avoid GPU, network, dataset downloads and any
real sandbox execution. They target the small pure-function surface that
is most likely to silently break during a port:

* import wiring (all submodules resolve under programmed_training.recipes.code_rl)
* fenced-code-block extraction
* TACO ↔ LiveCodeBench test-case shape conversion
* metadata / test-list normalization
* DeepcoderReward arithmetic with the sandbox stubbed out
* CLIConfig fields used by experiments/run_all_benchmarks.sh
"""

import ast
import io
import json
import sys
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from azure.ai.finetuningsessions.models import ModelInput
from programmed_training.recipes.code_rl import (
    code_env,
    code_grading,
    deepcoder_tool,
    train_azure,
)
from programmed_training.recipes.code_rl.code_env import _ensure_dict, _normalize_tests
from programmed_training.recipes.code_rl.code_grading import (
    extract_code_from_model,
    postprocess_lcb_sample,
    taco_to_lcb_format,
)
from programmed_training.recipes.code_rl.deepcoder_tool import DeepcoderReward, DeepcoderTask
from programmed_training.recipes.code_rl.lcb_utils import TEST_UTIL
from programmed_training.recipes.code_rl.train_azure import CLIConfig
from programmed_training.utils.misc_utils import dict_mean

# ---------------------------------------------------------------------------
# Import wiring — catches missing modules or broken programmed_training.* paths
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


def test_selected_dataset_normalization_preserves_source_order():
    dataset = code_env.Dataset.from_list([
        {"problem": "first question", "tests": [{"input": "1", "output": "1"}]},
        {"problem": "unjudgeable question", "tests": []},
        {"problem": "second question", "tests": [{"input": "2", "output": "2"}]},
    ])
    tasks = code_env.deepcoder_tasks_from_dataset(dataset)
    assert len(tasks) == 2
    assert "first question" in tasks[0].problem
    assert "second question" in tasks[1].problem


@pytest.fixture
def code_io_harness():
    # Load only trusted I/O helpers, not reliability_guard or model programs.
    definitions = [
        node for node in ast.parse(TEST_UTIL).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ("call_method", "Capturing", "stripped_string_compare")
    ]
    assert len(definitions) == 3
    namespace = {
        "__name__": "trusted_code_io_harness",
        "sys": sys, "patch": patch, "mock_open": mock_open,
        "StringIO": io.StringIO, "BytesIO": io.BytesIO,
        "BufferedReader": io.BufferedReader, "BufferedWriter": io.BufferedWriter,
        "TextIOWrapper": io.TextIOWrapper,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "trusted_code_io_harness", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("reader", "expected"),
    [
        pytest.param(lambda: sys.stdin.read(), "a\nb\n", id="text-read"),
        pytest.param(lambda: sys.stdin.buffer.read(), b"a\nb\n", id="binary-read"),
        pytest.param(lambda: sys.stdin.buffer.peek(), b"a\nb\n", id="binary-peek"),
        pytest.param(lambda: (sys.stdin.read(1), sys.stdin.read()), ("a", "\nb\n"), id="sized-read"),
        pytest.param(
            lambda: (sys.stdin.readline(), sys.stdin.readline(), sys.stdin.readline()),
            ("a\n", "b\n", ""), id="readline-eof",
        ),
        pytest.param(lambda: sys.stdin.readlines(), ["a\n", "b\n"], id="readlines"),
        pytest.param(lambda: list(sys.stdin), ["a\n", "b\n"], id="line-iteration"),
        pytest.param(lambda: (input(), input()), ("a", "b"), id="builtin-input"),
    ],
)
def test_grader_preserves_stdin_stream_semantics(code_io_harness, reader, expected):
    original = sys.stdin
    assert code_io_harness["call_method"](reader, "a\nb\n") == expected
    assert sys.stdin is original


def test_grader_captures_text_and_binary_stdout(code_io_harness):
    original = sys.stdout
    with code_io_harness["Capturing"]() as output:
        print("a", end="")
        sys.stdout.buffer.write(b"b")
        print("c", end="")
    assert output == ["abc"]
    assert sys.stdout is original


def test_grader_captures_utf8_binary_stdout(code_io_harness):
    with code_io_harness["Capturing"]() as output:
        sys.stdout.buffer.write("caf\u00e9".encode())
    assert output == ["caf\u00e9"]


def test_grader_can_read_output_after_solution_closes_stdout(code_io_harness):
    with code_io_harness["Capturing"]() as output:
        sys.stdout.buffer.write(b"done")
        sys.stdout.close()
    assert output == ["done"]


def test_grader_restores_stdout_after_invalid_utf8(code_io_harness):
    original = sys.stdout
    with pytest.raises(UnicodeDecodeError):
        with code_io_harness["Capturing"]():
            sys.stdout.buffer.write(b"\xff")
    assert sys.stdout is original


def test_grader_restores_stdin_after_program_error(code_io_harness):
    original = sys.stdin
    with pytest.raises(ZeroDivisionError):
        code_io_harness["call_method"](lambda: 1 / 0, "a")
    assert sys.stdin is original


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("0\n6\n7\n", "0 6 7", True),
        (" YES \n answer\n", "YES answer", True),
        ("2 1", "1 2", False),
        ("1 1 2", "1 2 2", False),
        ("YES abc", "YES bac", False),
    ],
)
def test_stdout_whitespace_matching_preserves_order_and_multiplicity(
    code_io_harness, actual, expected, matches
):
    assert code_io_harness["stripped_string_compare"](actual, expected) is matches


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


@pytest.mark.asyncio
async def test_code_env_counts_parse_failures_in_correctness(monkeypatch):
    renderer = MagicMock()
    renderer.get_stop_sequences.return_value = None
    renderer.build_generation_prompt.return_value = ModelInput(chunks=[])
    renderer.parse_response_with_tools.side_effect = [
        ({"role": "assistant", "content": ""}, False),
        ({"role": "assistant", "content": "```python\nprint(1)\n```"}, True),
    ]
    monkeypatch.setattr(code_env.tokenizer_utils, "get_tokenizer", MagicMock())
    monkeypatch.setattr(code_env, "get_renderer", MagicMock(return_value=renderer))
    monkeypatch.setattr(code_env, "_initial_messages", MagicMock(return_value=[]))
    grader = AsyncMock(return_value=(True, {"stub": True}))
    monkeypatch.setattr(deepcoder_tool, "sandbox_check_correctness", grader)
    builder = code_env.DeepcoderEnvGroupBuilder(
        task=_task(),
        model_name="test-model",
        renderer_name="test-renderer",
        max_turns=2,
        group_size=2,
        sandbox_backend=None,
        failed_parse_reward=-0.25,
    )

    failed_env, passing_env = await builder.make_envs()
    failed = await failed_env.step([])
    passing = await passing_env.step([])

    assert failed.episode_done
    assert failed.reward == -0.25
    assert failed.metrics == {"parse_error": 1.0, "format": 0.0, "correct": 0.0}
    assert passing.episode_done
    assert passing.reward == 1.0
    assert passing.metrics == {"format": 1.0, "correct": 1.0}
    aggregate = dict_mean([failed.metrics, passing.metrics])
    assert aggregate["correct"] == 0.5
    assert aggregate["format"] == 0.5
    grader.assert_awaited_once()


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
