"""Network-free contracts for the long-context counting recipe."""

import asyncio
import json
import random
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import chz
import pytest
from azure.ai.finetuningsessions import TrainingEngineError
from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk
from azure.core.credentials import AzureKeyCredential

from programmed_training.completers import TokensWithLogprobs
from programmed_training.optimizers import AdamConfig
from programmed_training.recipes.rl_numerics_check import env, train_azure
from programmed_training.renderers import get_renderer
from programmed_training.renderers.base import Message, Renderer, ToolCall, TrainOnWhat
from programmed_training.rl.data_processing import trajectory_to_data
from programmed_training.rl.types import Trajectory, Transition


def tool_call(page, call_id="first", name="read_page"):
    return ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps({"page": page})),
    )


@pytest.fixture
def problem():
    return env.CountingProblem(
        word="time", num_pages=2,
        pages=("Time time's times lifetime", "time-based TIME", "time"),
        prompt="Count time across two pages.", true_count=4,
    )


def renderer():
    result = MagicMock(spec=Renderer)
    result.get_stop_sequences.return_value = [99]
    result.create_conversation_prefix_with_tools.return_value = []
    return result


async def message_env(problem, group_size=1, max_turns=6):
    task = env.CountingTaskConfig(num_pages_min=2, num_pages_max=2)
    builder = env.CountingGroupBuilder(problem, task, renderer(), group_size, 1000, 10)
    environments = await builder.make_envs()
    for environment in environments:
        environment.message_env.max_turns = max_turns
        await environment.message_env.initial_observation()
    return environments


def test_word_matching_and_last_answer():
    assert env.count_word("Time time's time-based times lifetime time2", "time") == 4
    assert env.parse_answer("Answer: 2\nCorrected Answer: **1,234**") == 1234
    assert env.parse_answer("ANSWER: -12") == -12
    assert env.parse_answer("There are 12.") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("close_failure", ["owned_terminal", "wrong_session", "other_code", "ordinary_conflict"])
async def test_launcher_does_not_replay_consumed_terminal_close_after_recovery(
    monkeypatch, tmp_path, close_failure,
):
    from azure.ai.finetuningsessions import TrainingEngineError
    from azure.core.exceptions import HttpResponseError

    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    create = AsyncMock(side_effect=["session_original", "session_replacement"])

    async def close_session(session_id):
        if session_id == "session_original":
            if close_failure == "ordinary_conflict":
                raise HttpResponseError("ordinary conflict")
            raise TrainingEngineError(
                "The training engine stopped",
                error_code="other" if close_failure == "other_code" else "engine_dead",
                session_id="session_unowned" if close_failure == "wrong_session" else session_id,
            )

    async def training(_config, _client, session_id, **_kwargs):
        if session_id == "session_original":
            raise TrainingEngineError(
                "The training engine stopped", error_code="engine_dead", session_id=session_id,
            )

    close = AsyncMock()
    monkeypatch.setattr(FineTuningSessionClient, "create_session", create)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", AsyncMock(side_effect=close_session))
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    monkeypatch.setattr(train_azure.rl_azure, "main", AsyncMock(side_effect=training))
    monkeypatch.setattr(train_azure.auto_resume.random, "uniform", lambda *_args: 0)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"), auto_resume=True,
    )
    if close_failure == "owned_terminal":
        await train_azure.cli_main(cli)
        assert create.await_count == 2
        assert close.await_count == 2
        assert len(train_azure.auto_resume.attempt_log_paths(cli.log_path)) == 2
    else:
        with pytest.raises(TrainingEngineError, match="The training engine stopped"):
            await train_azure.cli_main(cli)
        assert create.await_count == 1


@pytest.mark.parametrize("kwargs", [
    {"num_pages_min": 0}, {"num_pages_min": 3, "num_pages_max": 2},
    {"page_chars": 0}, {"words": ()}, {"words": ("Time",)},
    {"words": ("time", "time")}, {"word_hint": "hidden"},
    {"word_hint": "first_last", "words": ("time", "tame")},
    {"reward_tolerance": 0}, {"reward_tolerance": float("nan")},
    {"secret_num_pages": 99}, {"num_pages_hint": "invalid"},
    {"target_context_tokens": 0}, {"target_context_tokens": -1},
    {"target_context_tokens": 1.5}, {"target_context_tokens": True},
])
def test_task_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        env.CountingTaskConfig(**kwargs)


def test_sampling_is_repeatable_and_keeps_hidden_page_count_fixed():
    corpus = " ".join(random.Random(2).choices(["time", "other", "after"], k=4000))
    task = env.CountingTaskConfig(
        num_pages_min=2, num_pages_max=4, page_chars=100,
        num_pages_hint="range", secret_num_pages=3,
        word_hint="hidden", words=("time",),
    )
    first = env.sample_problem(corpus, task, 1)
    assert first == env.sample_problem(corpus, task, 1)
    assert env.sample_problem(corpus, task, 100).num_pages == first.num_pages == 3
    assert len(first.pages) == task.max_turns
    assert first.true_count == sum(env.count_word(page, "time") for page in first.pages[:3])
    assert "secret word" in first.prompt and '"time"' not in first.prompt
    assert all(not page.startswith(("ime", "ther", "fter")) for page in first.pages)
    with pytest.raises(ValueError, match="Corpus has"):
        env.sample_problem("short", task, 0)


@pytest.mark.parametrize("density", [3, 5])
@pytest.mark.parametrize("seed", [0, 17])
def test_context_calibration_targets_actual_rendered_240k_and_recomputes_count(density, seed):
    corpus = " ".join(random.Random(4).choices(env.DEFAULT_WORDS, k=650_000))
    task = env.CountingTaskConfig(
        num_pages_min=28, num_pages_max=36, secret_num_pages=32,
        page_chars=27200, target_context_tokens=245760,
    )
    render = renderer()
    render.build_generation_prompt.side_effect = lambda history: model_input(
        [0] * (sum(len(message["content"]) for message in history) // density)
    )
    problem = env._fit_problem(corpus, task, render, seed)
    measured = env._completed_context_tokens(problem, render)
    assert 244736 <= measured <= 246784
    assert measured + 8192 <= 262144
    assert problem.calibrated_context_tokens == measured
    assert problem.num_pages == 32 and len(problem.pages) == task.max_turns
    assert problem.true_count == sum(
        env.count_word(page, problem.word) for page in problem.pages[:32]
    )
    assert problem == env._fit_problem(corpus, task, render, seed)
    history = render.build_generation_prompt.call_args.args[0]
    assert history[-1]["role"] == "tool" and "[Page 32]" in history[-1]["content"]


def test_context_calibration_is_opt_in_and_impossible_target_fails_explicitly():
    corpus = "time other after " * 1000
    task = env.CountingTaskConfig(num_pages_min=2, num_pages_max=2, page_chars=40)
    render = renderer()
    assert env._fit_problem(corpus, task, render, 0) == env.sample_problem(corpus, task, 0)
    render.build_generation_prompt.assert_not_called()
    targeted = chz.replace(task, target_context_tokens=245760)
    render.build_generation_prompt.side_effect = lambda history: model_input(
        [0] * sum(len(message["content"]) for message in history)
    )
    with pytest.raises(ValueError, match="Cannot size completed-document context"):
        env._fit_problem(corpus, targeted, render, 0)


def test_context_calibration_rejects_generation_budget_overflow_before_loading():
    with pytest.raises(ValueError, match="Target context"):
        env.CountingDatasetBuilder(
            model_name_for_tokenizer="test", batch_size=1, group_size=2,
            task=env.CountingTaskConfig(target_context_tokens=245760),
            max_trajectory_tokens=253952, max_generation_tokens=8192,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model,revision,renderer_name", [
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "qwen3_8"),
    (
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "77df655d5e9f8362164ed14dd8b48f8bce657498", "nemotron3_ultra_disable_thinking",
    ),
])
async def test_cached_model_document_traversal_reaches_calibrated_240k(
    model, revision, renderer_name,
):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True,
        )
    except OSError:
        pytest.skip("Optional pinned tokenizer is not cached; this test never downloads")
    render = get_renderer(renderer_name, tokenizer)
    task = env.CountingTaskConfig(
        num_pages_min=28, num_pages_max=36, secret_num_pages=32,
        page_chars=27200, target_context_tokens=245760,
    )
    corpus = " ".join(random.Random(4).choices(env.DEFAULT_WORDS, k=650_000))
    dataset = env.CountingDataset(corpus, task, render, 1, 2, 1, 0, 262144, 8192)
    group = dataset.get_batch(0)[0]
    (environment,) = await env.CountingGroupBuilder(
        group.problem, task, render, 1, 262144, 8192,
    ).make_envs()
    try:
        initial, _ = await environment.initial_observation()
        assert len(env.text_tokens(initial)) < 4096
        for page in range(1, 33):
            result = await environment.message_env.step(Message(
                role="assistant", content="",
                tool_calls=[tool_call(page, call_id=f"read_page_{page}")],
            ))
            assert not result.episode_done
        actual = len(env.text_tokens(render.build_generation_prompt(result.next_messages)))
        assert actual == group.problem.calibrated_context_tokens
        assert 244736 <= actual <= 246784
        assert actual + 8192 <= 262144
        final = await environment.message_env.step(Message(
            role="assistant", content=f"Answer: {group.problem.true_count}",
        ))
        assert final.episode_done and final.reward == 1
        assert final.metrics["read_all_pages"] == 1
    finally:
        await environment.close()


@pytest.mark.asyncio
async def test_pages_rejections_and_state_isolation(problem):
    first, second = await message_env(problem, group_size=2)
    state = first.message_env.state
    result = await first.message_env.step(Message(
        role="assistant", content="",
        tool_calls=[tool_call(1), tool_call(2, "second")],
    ))
    assert not result.episode_done
    assert state.pages_read == {1}
    assert state.rejected_tool_calls == 1
    assert second.message_env.state.pages_read == set()
    assert result.next_messages[-1]["tool_call_id"] == "second"
    assert "first tool call" in result.next_messages[-1]["content"]
    await first.message_env.step(Message(
        role="assistant", content="", tool_calls=[tool_call(99)],
    ))
    assert state.invalid_page_requests == 1
    await first.message_env.step(Message(
        role="assistant", content="", tool_calls=[tool_call(3)],
    ))
    final = await first.message_env.step(Message(role="assistant", content="Answer: 4"))
    assert final.episode_done and final.reward == 1
    assert final.metrics["read_past_end"] == 1
    assert final.metrics["read_all_pages"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,reward", [
    ("Answer: 4", 1.0), ("Answer: 3", 0.5), ("Answer: 0", 0.0),
    ("No answer", 0.0),
])
async def test_reward_formula(problem, answer, reward):
    grader = env.CountingReward(problem, env.EpisodeState(), tolerance=0.5)
    actual, metrics = await grader([Message(role="assistant", content=answer)])
    assert actual == reward
    assert metrics["answer_parsed"] == float("Answer:" in answer)


@pytest.mark.asyncio
async def test_turn_limit_does_not_grade_an_answer_from_a_tool_turn(problem):
    (environment,) = await message_env(problem, max_turns=1)
    result = await environment.message_env.step(Message(
        role="assistant", content="Answer: 4", tool_calls=[tool_call(1)],
    ))
    assert result.episode_done and result.reward == 0
    assert result.metrics["max_turns"] == 1
    assert result.metrics["answer_parsed"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("page", [True, 1.5, "1"])
async def test_noninteger_pages_return_explicit_validation_errors(problem, page):
    (environment,) = await message_env(problem)
    result = await environment.message_env.step(Message(
        role="assistant", content="", tool_calls=[tool_call(page)],
    ))
    assert environment.message_env.state.pages_read == set()
    assert "error" in result.next_messages[-1]["content"]


def model_input(tokens):
    return ModelInput(chunks=[ModelInputChunk(tokens=tokens)])


@pytest.mark.asyncio
async def test_token_budget_and_extension_rate(problem):
    (environment,) = await message_env(problem)
    render = environment.renderer
    render.build_generation_prompt.side_effect = [
        model_input([1, 2]), model_input([1, 2, 3, 4]), model_input([1, 2, 3, 4, 5]),
    ]
    render.parse_response_with_tools.side_effect = [
        (Message(role="assistant", content="", tool_calls=[tool_call(1)]), True),
        (Message(role="assistant", content="Answer: 4"), True),
    ]
    await environment.initial_observation()
    first = await environment.step([3])
    assert not first.episode_done
    final = await environment.step([5])
    assert final.reward == 1
    assert final.metrics["extension_rate"] == 1
    assert final.metrics["max_context_tokens"] == 5
    assert final.metrics["max_sampled_prompt_tokens"] == 4


@pytest.mark.asyncio
async def test_overflow_reserves_next_generation_and_clears_success_metrics(problem):
    (environment,) = await message_env(problem)
    environment.token_limit = 15
    render = environment.renderer
    render.build_generation_prompt.side_effect = [model_input([1, 2]), model_input(list(range(6)))]
    render.parse_response_with_tools.return_value = (
        Message(role="assistant", content="", tool_calls=[tool_call(1)]), True,
    )
    await environment.initial_observation()
    result = await environment.step([3])
    assert result.episode_done and result.reward == -0.1
    assert result.metrics["context_overflow"] == 1
    assert result.metrics["max_sampled_prompt_tokens"] == 2
    assert result.metrics["exact_match"] == 0
    assert result.next_observation.chunks == []


@pytest.mark.asyncio
async def test_initial_overflow_and_parse_failure(problem):
    (environment,) = await message_env(problem)
    render = environment.renderer
    render.build_generation_prompt.return_value = model_input([1, 2])
    environment.token_limit = 11
    with pytest.raises(ValueError, match="Initial prompt"):
        await environment.initial_observation()
    environment.token_limit = 100
    await environment.initial_observation()
    render.parse_response_with_tools.return_value = (Message(role="assistant", content=""), False)
    result = await environment.step([3])
    assert result.episode_done and result.reward == -0.1
    assert result.metrics["parse_error"] == 1
    assert result.metrics["answer_parsed"] == result.metrics["exact_match"] == 0


@pytest.mark.asyncio
async def test_exact_context_boundary_accepts_answer_but_overflow_does_not(problem):
    for action, expected in (([3] * 10, 1.0), ([3] * 11, -0.1)):
        (environment,) = await message_env(problem)
        environment.token_limit = 12
        environment.renderer.build_generation_prompt.return_value = model_input([1, 2])
        environment.renderer.parse_response_with_tools.return_value = (
            Message(role="assistant", content="Answer: 4"), True,
        )
        await environment.initial_observation()
        result = await environment.step(action)
        assert result.reward == expected
        assert result.metrics["context_overflow"] == float(expected < 0)
        assert result.metrics["max_context_tokens"] == 2 + len(action)
        if expected < 0:
            assert result.metrics["answer_parsed"] == 0
            assert "abs_error" not in result.metrics


@pytest.mark.asyncio
async def test_zero_gold_count_is_well_defined():
    problem = env.CountingProblem("time", 1, ("other",), "Count.", 0)
    grader = env.CountingReward(problem, env.EpisodeState(), 0.2)
    assert (await grader([Message(role="assistant", content="Answer: 0")]))[0] == 1
    assert (await grader([Message(role="assistant", content="Answer: 1")]))[0] == 0


def test_dataset_batches_are_lazy_deterministic_and_keep_partial_batch():
    dataset = env.CountingDataset(
        "time other after " * 1000,
        env.CountingTaskConfig(num_pages_min=1, num_pages_max=2, page_chars=40),
        renderer(), 2, 2, 3, 12, 1000, 10,
    )
    assert len(dataset) == 2
    assert len(dataset.get_batch(1)) == 1
    assert dataset.get_batch(0)[0].problem == dataset.get_batch(0)[0].problem
    with pytest.raises(IndexError):
        dataset.get_batch(2)


@pytest.mark.asyncio
async def test_dataset_builder_preflights_local_corpus(monkeypatch, tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("time other after " * 1000, encoding="utf-8")
    render = renderer()
    render.build_generation_prompt.return_value = model_input([1, 2])
    monkeypatch.setattr(env, "get_renderer", lambda *_: render)
    monkeypatch.setattr(env, "get_tokenizer", lambda _: object())
    builder = env.CountingDatasetBuilder(
        model_name_for_tokenizer="test", batch_size=1, group_size=2,
        renderer_name="test", corpus_path=str(path),
        task=env.CountingTaskConfig(num_pages_min=1, num_pages_max=1, page_chars=40),
        max_train_examples=2, max_test_examples=1,
        max_trajectory_tokens=100, max_generation_tokens=10,
    )
    train, test = await builder()
    assert len(train) == 2 and len(test) == 1
    assert train.seed != test.seed


@pytest.mark.asyncio
@pytest.mark.parametrize("renderer_name", [
    "qwen3_8_disable_thinking", "qwen3_8",
])
async def test_cached_qwen38_tool_round_trip_merges_action_only_training_data(
    problem, renderer_name,
):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.8-27B", local_files_only=True,
        )
    except OSError:
        pytest.skip("Optional Qwen3.8 tokenizer is not cached; this test never downloads")
    render = get_renderer(renderer_name, tokenizer)
    if renderer_name == "qwen3_8":
        assert not render.strip_thinking_from_history
        assert render.reasoning_effort == "xhigh"
    builder = env.CountingGroupBuilder(
        problem, env.CountingTaskConfig(num_pages_min=2, num_pages_max=2),
        render, 1, 8192, 512,
    )
    (environment,) = await builder.make_envs()
    observation, _ = await environment.initial_observation()
    transitions = []
    actions = [
        Message(role="assistant", content="", tool_calls=[tool_call(1)]),
        Message(role="assistant", content="", tool_calls=[tool_call(2)]),
        Message(role="assistant", content="Answer: 4"),
    ]
    if renderer_name == "qwen3_8":
        for index, message in enumerate(actions):
            message["content"] = [
                {"type": "thinking", "thinking": f"Counting step {index + 1}."},
                {"type": "text", "text": message["content"]},
            ]
    for message in actions:
        full, _ = render.build_supervised_example(
            environment.message_env.history + [message],
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        prompt = env.text_tokens(observation)
        tokens = env.text_tokens(full)
        assert tokens[:len(prompt)] == prompt
        action = tokens[len(prompt):]
        result = await environment.step(action)
        transitions.append(Transition(
            ob=observation,
            ac=TokensWithLogprobs(tokens=action, maybe_logprobs=[-1.0] * len(action)),
            reward=result.reward, episode_done=result.episode_done, metrics=result.metrics,
        ))
        observation = result.next_observation
        if renderer_name == "qwen3_8" and not result.episode_done:
            assert "Counting step 1." in tokenizer.decode(env.text_tokens(observation))
    assert result.reward == result.metrics["read_all_pages"] == 1
    assert result.metrics["extension_rate"] == 1
    data = trajectory_to_data(Trajectory(transitions=transitions, final_ob=observation), 1.0)
    assert len(data) == 1
    assert sum(data[0].loss_fn_inputs["mask"].data) == sum(
        len(transition.ac.tokens) for transition in transitions
    )
    await environment.close()


def test_config_selects_matching_non_thinking_renderer_and_forwards_controls():
    cli = chz.Blueprint(train_azure.CLIConfig).make_from_argv([
        "project_endpoint=https://example.invalid/api/projects/test",
        "model_name=Qwen/Qwen3.6-35B-A3B",
        "task.num_pages_min=2", "task.num_pages_max=3",
        "group_size=2", "groups_per_batch=1",
        "max_steps=2", "max_steps_off_policy=1",
    ])
    config = train_azure.build_config(cli)
    assert config.dataset_builder.renderer_name == "qwen3_5_disable_thinking"
    assert config.model_name == config.tokenizer_name == cli.model_name
    assert config.dataset_builder.task.num_pages_min == 2
    assert config.dataset_builder.max_generation_tokens == cli.max_tokens
    assert config.async_config.max_steps_off_policy == 1
    assert config.max_steps == 2


def test_cli_defaults_match_smaller_thinking_experiment():
    cli = chz.Blueprint(train_azure.CLIConfig).make_from_argv([
        "project_endpoint=https://example.invalid/api/projects/test",
    ])
    config = train_azure.build_config(cli)
    builder = config.dataset_builder
    assert config.model_name == "Qwen/Qwen3.8-27B-256K"
    assert config.tokenizer_name == "Qwen/Qwen3.8-27B"
    assert builder.renderer_name == "qwen3_8"
    assert builder.group_size == 4 and builder.batch_size == 1
    assert config.max_steps == builder.max_train_examples == 10
    assert config.max_wall_clock_seconds == 259200
    assert builder.task.num_pages_min == 7 and builder.task.num_pages_max == 9
    assert builder.task.num_pages_hint == "range"
    assert builder.task.hidden_num_pages == 8
    assert builder.task.page_chars == 108800
    assert builder.task.word_hint == "exact" and builder.seed == 0
    assert config.max_tokens == builder.max_generation_tokens == 8192
    assert builder.max_trajectory_tokens == 262144
    assert config.eval_every == 0 and config.save_every == 1
    assert cli.auto_resume and cli.auto_resume_max_retries == 3
    assert cli.behavior_if_log_dir_exists == "raise"


@pytest.mark.parametrize("model,tokenizer,renderer", [
    ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B", "qwen3_8_disable_thinking"),
    ("Qwen/Qwen3.8-27B-256K", "custom/tokenizer", "qwen3_8_medium_reasoning"),
])
def test_stress_defaults_preserve_explicit_model_and_rendering_overrides(
    model, tokenizer, renderer,
):
    kwargs = {"model_name": model}
    if model.endswith("-256K"):
        kwargs.update(tokenizer_name=tokenizer, renderer_name=renderer)
    cli = train_azure.CLIConfig(project_endpoint="https://example.invalid", **kwargs)
    config = train_azure.build_config(cli)
    assert config.tokenizer_name == tokenizer
    assert config.dataset_builder.renderer_name == renderer


@pytest.mark.parametrize("overrides", [
    {},
    {
        "adam_beta1": 0.85, "adam_beta2": 0.98,
        "adam_eps": 1e-6, "adam_weight_decay": 0.01,
    },
])
def test_config_forwards_shared_adam_settings(overrides):
    cli = chz.Blueprint(train_azure.CLIConfig).make_from_argv([
        "project_endpoint=https://example.invalid/api/projects/test",
        *(f"{name}={value}" for name, value in overrides.items()),
    ])
    config = train_azure.build_config(cli)
    assert config.adam_kwargs() == AdamConfig(**overrides).adam_kwargs()


@pytest.mark.parametrize("model,render", [
    ("Qwen/Qwen3.8-27B", "qwen3_8"),
    ("Qwen/Qwen3.8-27B-256K", "qwen3_8"),
])
def test_full_experiment_cli_preserves_shared_task_and_batch_dimensions(model, render):
    cli = chz.Blueprint(train_azure.CLIConfig).make_from_argv([
        "project_endpoint=https://example.invalid/api/projects/qualified",
        f"model_name={model}", f"renderer_name={render}",
        f"tokenizer_name={model.removesuffix('-256K')}",
        "task.num_pages_min=28", "task.num_pages_max=36", "task.page_chars=27200",
        "task.num_pages_hint=range", "task.secret_num_pages=32", "task.word_hint=exact",
        "seed=0", "group_size=8", "groups_per_batch=8", "max_train_examples=208",
        "max_steps=26", "max_tokens=8192", "max_trajectory_tokens=262144",
        "eval_every=0", "save_every=3", "behavior_if_log_dir_exists=raise",
    ])
    config = train_azure.build_config(cli)
    builder = config.dataset_builder
    assert config.max_steps == 26 and config.save_every == 3
    assert builder.batch_size * builder.group_size == 64
    assert builder.max_train_examples == config.max_steps * builder.batch_size
    assert builder.task.hidden_num_pages == 32
    assert builder.task.page_chars == 27200
    assert builder.task.words == env.DEFAULT_WORDS
    assert builder.renderer_name == render
    assert builder.max_trajectory_tokens == 262144
    assert builder.max_generation_tokens == config.max_tokens == 8192
    assert config.tokenizer_name == model.removesuffix("-256K")


def test_readme_commands_use_native_qwen38_and_explicit_stress_budgets():
    readme = Path(train_azure.__file__).with_name("README.md").read_text(encoding="utf-8")
    configs = []
    for block in readme.split("```powershell\n")[1:]:
        argv = shlex.split(block.split("```", 1)[0].replace("`", ""))
        if argv[:3] == ["python", "-m", train_azure.__name__]:
            configs.append(chz.Blueprint(train_azure.CLIConfig).make_from_argv(argv[3:]))
    assert len(configs) == 3
    for cli in configs:
        assert cli.renderer_name == "qwen3_8"
        assert cli.max_tokens == 8192
        assert cli.behavior_if_log_dir_exists == "raise"
        config = train_azure.build_config(cli)
        assert config.dataset_builder.renderer_name == "qwen3_8"
    smoke, small, full = configs
    assert smoke.model_name == "Qwen/Qwen3.8-27B"
    assert smoke.max_steps == 2 and smoke.max_trajectory_tokens == 32768
    assert small.model_name == "Qwen/Qwen3.8-27B-256K"
    assert small.tokenizer_name == "Qwen/Qwen3.8-27B"
    assert small.max_steps == small.max_train_examples == 10
    assert small.group_size == 4 and small.groups_per_batch == 1
    assert small.task.hidden_num_pages == 8 and small.task.secret_num_pages == 8
    assert small.task.num_pages_min == 7 and small.task.num_pages_max == 9
    assert small.task.page_chars == 108800 and small.task.num_pages_hint == "range"
    assert small.max_trajectory_tokens == 262144 and small.save_every == 1
    assert small.eval_every == 0 and small.max_wall_clock_seconds == 259200
    assert full.model_name == "Qwen/Qwen3.8-27B-256K"
    assert full.tokenizer_name == "Qwen/Qwen3.8-27B"
    assert full.max_steps == 26 and full.max_trajectory_tokens == 262144
    assert full.groups_per_batch * full.group_size == 64
    assert full.max_train_examples == full.groups_per_batch * full.max_steps
    assert full.task.hidden_num_pages == 32


@pytest.mark.parametrize("kwargs", [
    {"group_size": 1}, {"max_train_examples": 0}, {"max_steps": 0},
    {"max_test_examples": -1}, {"max_tokens": 262144},
    {"max_steps_off_policy": -1}, {"max_concurrent_groups": 0},
])
def test_cli_rejects_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        train_azure.CLIConfig(project_endpoint="https://example.invalid", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("auto_resume", [False, True])
async def test_launcher_retains_metadata_and_closes_session_on_failure(
    monkeypatch, tmp_path, fail, auto_resume,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    prepared = (MagicMock(), None)
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=prepared),
    )
    create = AsyncMock(return_value="session_test")
    closed = False

    async def close_session(_session_id):
        if closed:
            raise ValueError("HTTP transport has already been closed")

    async def close_client():
        nonlocal closed
        closed = True

    complete = AsyncMock(side_effect=close_session)
    close = AsyncMock(side_effect=close_client)

    async def shared_training(_config, client, session_id, **_kwargs):
        try:
            if fail:
                raise RuntimeError("training failed")
        finally:
            if train_azure.auto_resume.current_attempt() is None:
                await client.close_session(session_id)
                await client.close()

    training = AsyncMock(side_effect=shared_training)
    monkeypatch.setattr(FineTuningSessionClient, "create_session", create)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    monkeypatch.setattr(train_azure.rl_azure, "main", training)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"), behavior_if_log_dir_exists="raise",
        load_checkpoint_path="session_source/saved", training_type="GlobalStandard",
        auto_resume=auto_resume,
    )
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            await train_azure.cli_main(cli)
    else:
        await train_azure.cli_main(cli)
    assert create.await_args.kwargs["from_checkpoint"].source_session_id == "session_source"
    assert create.await_args.kwargs["training_type"] == "GlobalStandard"
    assert training.await_args.kwargs["prepared_datasets"] == prepared
    complete.assert_awaited_once_with("session_test")
    close.assert_awaited_once()
    metadata = json.loads((tmp_path / "run" / "run_meta.json").read_text(encoding="utf-8"))
    assert metadata["azure"]["session_id"] == "session_test"
    assert metadata["model"]["renderer_name"] == "qwen3_8"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["create_session", "metadata"])
async def test_launcher_closes_resources_before_training_handoff(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    create = AsyncMock(
        side_effect=RuntimeError("create failed") if failure == "create_session" else None,
        return_value="session_test",
    )
    complete = AsyncMock()
    close = AsyncMock()
    training = AsyncMock()
    monkeypatch.setattr(FineTuningSessionClient, "create_session", create)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    monkeypatch.setattr(train_azure.rl_azure, "main", training)
    if failure == "metadata":
        from pathlib import Path

        original_write = Path.write_text
        writes = 0

        def write_metadata(path, *args, **kwargs):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("metadata write failed")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write_metadata)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"), behavior_if_log_dir_exists="raise",
    )
    with pytest.raises((RuntimeError, OSError), match="failed"):
        await train_azure.cli_main(cli)
    training.assert_not_awaited()
    close.assert_awaited_once()
    if failure == "metadata":
        complete.assert_awaited_once_with("session_test")
    else:
        complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [
    "logging", "trace", "checkpoint", "tokenizer", "training_client",
])
@pytest.mark.parametrize("auto_resume", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_launcher_closes_resources_when_real_shared_loop_setup_fails(
    monkeypatch, tmp_path, stage, auto_resume, cancelled,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    failure = asyncio.CancelledError() if cancelled else RuntimeError("shared setup failed")
    setup_logging = MagicMock()
    trace_init = MagicMock()
    tokenizer = MagicMock()
    training_client = MagicMock()
    setup_started = False

    def setup(*_args, **_kwargs):
        nonlocal setup_started
        setup_started = True
        if stage == "logging":
            raise failure
        return MagicMock()

    def checkpoint(*_args, **_kwargs):
        if setup_started and stage == "checkpoint":
            raise failure
        return None

    setup_logging.side_effect = setup
    failing_setup = {
        "trace": trace_init,
        "tokenizer": tokenizer,
        "training_client": training_client,
    }
    if stage in failing_setup:
        failing_setup[stage].side_effect = failure
    build_config = train_azure.build_config
    monkeypatch.setattr(
        train_azure, "build_config",
        lambda config: chz.replace(build_config(config), enable_trace=True),
    )
    monkeypatch.setattr(train_azure.rl_azure.ml_log, "setup_logging", setup_logging)
    monkeypatch.setattr(train_azure.rl_azure, "trace_init", trace_init)
    monkeypatch.setattr(train_azure.checkpoint_utils, "get_last_checkpoint", checkpoint)
    monkeypatch.setattr(train_azure.rl_azure, "get_tokenizer", tokenizer)
    monkeypatch.setattr(train_azure.rl_azure, "AzureSDKTrainingClient", training_client)
    create = AsyncMock(return_value="session_test")
    complete = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(FineTuningSessionClient, "create_session", create)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"), behavior_if_log_dir_exists="raise",
        auto_resume=auto_resume,
    )
    with pytest.raises(type(failure)) as caught:
        await train_azure.cli_main(cli)
    assert caught.value is failure
    complete.assert_awaited_once_with("session_test")
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume", [False, True])
@pytest.mark.parametrize("cleanup_failure", [None, "session", "client", "both"])
async def test_launcher_closes_allocation_when_real_sdk_model_load_fails(
    monkeypatch, tmp_path, auto_resume, cleanup_failure, caplog,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    allocated = MagicMock(status_code=200)
    allocated.json.return_value = {
        "session_id": "model_allocated", "request_id": "load_request",
    }
    failed = MagicMock(status_code=200)
    failed.json.return_value = {
        "status": "failed", "error_code": "engine_dead", "error": "model load failed",
    }
    send = AsyncMock(side_effect=[allocated, failed])
    complete = AsyncMock(
        side_effect=RuntimeError("session cleanup failed")
        if cleanup_failure in ("session", "both") else None,
    )
    close = AsyncMock(
        side_effect=RuntimeError("client cleanup failed")
        if cleanup_failure in ("client", "both") else None,
    )
    training = AsyncMock()
    monkeypatch.setattr(FineTuningSessionClient, "send_request", send)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    monkeypatch.setattr(train_azure.rl_azure, "main", training)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"), auto_resume=auto_resume,
    )
    with pytest.raises(TrainingEngineError, match="model load failed") as caught:
        await train_azure.cli_main(cli)
    assert caught.value.session_id == "session_allocated"
    assert caught.value.error_code == "engine_dead"
    assert [call.args[0].method for call in send.await_args_list] == ["POST", "GET"]
    complete.assert_awaited_once_with("session_allocated")
    close.assert_awaited_once()
    training.assert_not_awaited()
    if cleanup_failure is not None:
        assert "Recipe session/client cleanup failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume", [False, True])
@pytest.mark.parametrize("failure", [
    TrainingEngineError("unknown allocation", error_code="engine_dead"),
    TimeoutError("model load timed out"),
    asyncio.CancelledError(),
])
async def test_creation_failure_without_session_id_still_closes_client(
    monkeypatch, tmp_path, auto_resume, failure,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    monkeypatch.setattr(FineTuningSessionClient, "create_session", AsyncMock(side_effect=failure))
    complete = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        log_path=str(tmp_path / "run"), auto_resume=auto_resume,
    )
    with pytest.raises(type(failure)) as caught:
        await train_azure.cli_main(cli)
    assert caught.value is failure
    complete.assert_not_awaited()
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume", [False, True])
@pytest.mark.parametrize("failure_kind", [
    "timeout", "poll_timeout", "cancellation", "untyped_load_failure",
])
async def test_launcher_closes_real_sdk_allocation_without_exception_session_id(
    monkeypatch, tmp_path, auto_resume, failure_kind,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    allocated = MagicMock(status_code=200)
    allocated.json.return_value = {
        "session_id": "model_allocated", "request_id": "load_request",
    }
    failed = MagicMock(status_code=200)
    failed.json.return_value = {"status": "unexpected"}
    completed = MagicMock(status_code=200)
    poll_started = asyncio.Event()

    async def send(request, **_kwargs):
        path = urlparse(request.url).path
        if request.method == "POST" and path.endswith("/fine_tuning/sessions"):
            return allocated
        if request.method == "GET":
            poll_started.set()
            if failure_kind == "cancellation":
                await asyncio.Event().wait()
            if failure_kind == "poll_timeout":
                raise TimeoutError("model load poll timed out")
            return failed
        if request.method == "POST" and path.endswith(
            "/fine_tuning/sessions/model_allocated/complete"
        ):
            return completed
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    send_request = AsyncMock(side_effect=send)
    close_client = FineTuningSessionClient.close
    clients_closed = []

    async def close(client):
        clients_closed.append(client)
        await close_client(client)

    monkeypatch.setattr(FineTuningSessionClient, "send_request", send_request)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    training = AsyncMock()
    monkeypatch.setattr(train_azure.rl_azure, "main", training)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid", log_path=str(tmp_path / "run"),
        auto_resume=auto_resume,
        create_session_timeout_sec=0.0 if failure_kind == "timeout" else 600.0,
    )
    task = asyncio.create_task(train_azure.cli_main(cli))
    try:
        if failure_kind == "cancellation":
            await asyncio.wait_for(poll_started.wait(), timeout=2)
            task.cancel()
        expected = {
            "timeout": RuntimeError,
            "poll_timeout": TimeoutError,
            "cancellation": asyncio.CancelledError,
            "untyped_load_failure": RuntimeError,
        }[failure_kind]
        with pytest.raises(expected) as caught:
            await asyncio.wait_for(task, timeout=2)
        assert not hasattr(caught.value, "session_id")
        if failure_kind == "timeout":
            assert "Timed out" in str(caught.value)
        elif failure_kind == "untyped_load_failure":
            assert "Unexpected envelope status" in str(caught.value)
        requests = [call.args[0] for call in send_request.await_args_list]
        assert requests[0].method == "POST"
        assert urlparse(requests[-1].url).path.endswith(
            "/fine_tuning/sessions/model_allocated/complete"
        )
        completed.raise_for_status.assert_called_once()
        assert len(clients_closed) == 1
        training.assert_not_awaited()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_close", ["session", "client", "both"])
@pytest.mark.parametrize("failure_kind", ["runtime", "engine_dead", "cancellation"])
@pytest.mark.parametrize("ignore_first_cancellation", [False, True])
async def test_launcher_preserves_real_setup_failure_after_recovery_cleanup_timeout(
    monkeypatch, tmp_path, pending_close, failure_kind, ignore_first_cancellation, caplog,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(MagicMock(), None)),
    )
    monkeypatch.setattr(train_azure.auto_resume, "CLEANUP_TIMEOUT_SECONDS", 0.02)
    failure = {
        "runtime": RuntimeError("shared setup failed"),
        "engine_dead": TrainingEngineError(
            "shared setup failed", error_code="engine_dead", session_id="session_test",
        ),
        "cancellation": asyncio.CancelledError("shared setup cancelled"),
    }[failure_kind]
    monkeypatch.setattr(
        train_azure.rl_azure.ml_log, "setup_logging", MagicMock(side_effect=failure),
    )
    create = AsyncMock(return_value="session_test")
    monkeypatch.setattr(FineTuningSessionClient, "create_session", create)
    cancelled = set()
    release = asyncio.Event()
    cleanup_tasks = []

    async def pending(stage):
        task = asyncio.current_task()
        assert task is not None
        cleanup_tasks.append(task)
        try:
            try:
                await release.wait()
            except asyncio.CancelledError:
                if not ignore_first_cancellation:
                    raise
                await release.wait()
        finally:
            cancelled.add(stage)

    async def close_session(_session_id):
        if pending_close in ("session", "both"):
            await pending("session")

    async def close_client():
        if pending_close in ("client", "both"):
            await pending("client")

    complete = AsyncMock(side_effect=close_session)
    close = AsyncMock(side_effect=close_client)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid", log_path=str(tmp_path / "run"),
        auto_resume=True,
    )
    try:
        with pytest.raises(type(failure)) as caught:
            await asyncio.wait_for(train_azure.cli_main(cli), timeout=2)
        assert caught.value is failure
        complete.assert_awaited_once_with("session_test")
        close.assert_awaited_once()
        create.assert_awaited_once()
        assert "Recovery cleanup failed; preserving the training failure" in caplog.text
        assert "Recipe session/client cleanup failed" in caplog.text
    finally:
        release.set()
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    assert cancelled == ({"session", "client"} if pending_close == "both" else {pending_close})


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_recipe_client_shares_cleanup_results_without_repeating_io(monkeypatch, fails):
    session_error = RuntimeError("session cleanup failed")
    client_error = RuntimeError("client cleanup failed")
    complete = AsyncMock(return_value=None, side_effect=session_error if fails else None)
    close = AsyncMock(return_value=None, side_effect=client_error if fails else None)
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    client = train_azure._RecipeClient(
        endpoint="https://example.invalid", credential=AzureKeyCredential("test-key"),
    )
    for operation, args, error in (
        (client.close_session, ("session_test",), session_error),
        (client.close, (), client_error),
    ):
        results = await asyncio.gather(
            operation(*args), operation(*args), return_exceptions=True,
        )
        if fails:
            assert results[0] is results[1] is error
        else:
            assert results == [None, None]
    complete.assert_awaited_once_with("session_test")
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_launcher_cleanup_does_not_repeat_real_shared_loop_cleanup(
    monkeypatch, tmp_path, auto_resume, fails,
):
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    monkeypatch.setattr(
        env.CountingDatasetBuilder, "__call__", AsyncMock(return_value=(dataset, None)),
    )
    monkeypatch.setattr(train_azure.rl_azure.ml_log, "setup_logging", MagicMock())
    monkeypatch.setattr(train_azure.rl_azure, "get_tokenizer", MagicMock())
    monkeypatch.setattr(train_azure.rl_azure, "AzureSDKTrainingClient", MagicMock())
    failure = RuntimeError("rollout failed")
    training = AsyncMock(return_value=(0, None), side_effect=failure if fails else None)
    monkeypatch.setattr(train_azure.rl_azure, "do_sync_training", training)
    monkeypatch.setattr(
        FineTuningSessionClient, "create_session", AsyncMock(return_value="session_test"),
    )
    monkeypatch.setattr(
        FineTuningSessionClient, "send_request",
        AsyncMock(side_effect=AssertionError("Unexpected network I/O")),
    )
    complete = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(FineTuningSessionClient, "close_session", complete)
    monkeypatch.setattr(FineTuningSessionClient, "close", close)
    cli = train_azure.CLIConfig(
        project_endpoint="https://example.invalid",
        log_path=str(tmp_path / "run"), auto_resume=auto_resume,
    )
    if fails:
        with pytest.raises(RuntimeError, match="rollout failed") as caught:
            await train_azure.cli_main(cli)
        assert caught.value is failure
    else:
        await train_azure.cli_main(cli)
    training.assert_awaited_once()
    complete.assert_awaited_once_with("session_test")
    close.assert_awaited_once()
