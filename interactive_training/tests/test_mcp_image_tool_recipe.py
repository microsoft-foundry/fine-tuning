from __future__ import annotations

import tomllib
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from azure.ai.finetuningsessions.models import ImageChunk, ModelInput, ModelInputChunk

from interactive_training.recipes.mcp_image_tool.data import (
    ALIEN_WORD_TO_COLOR,
    COLORS,
    build_curriculum_tasks,
    build_mixed_tasks,
    build_tasks,
)
from interactive_training.recipes.mcp_image_tool import train_azure
from interactive_training.recipes.mcp_image_tool import mcp_env
from interactive_training.recipes.mcp_image_tool import warmup
from interactive_training.recipes.mcp_image_tool.training_client import MCPImageTrainingClient, split_image_batches
from interactive_training.recipes.mcp_image_tool.learning_report import build_learning_report
from interactive_training.recipes.mcp_image_tool.warmup import demonstrate_task
from interactive_training.tool_use.agent_tool_message_env import AgentToolMessageEnv
from interactive_training.tool_use.types import ToolResult
from interactive_training.recipes.mcp_image_tool.mcp_env import (
    AlienColorAnswerReward,
    MCPImageToolDatasetBuilder,
    SYSTEM_PROMPT,
    _alien_word_orders,
    _task_tool_call_limit,
)
from interactive_training.recipes.mcp_image_tool.server import TOOL_DESCRIPTION
from interactive_training.recipes.mcp_image_tool.train_azure import (
    CLIConfig,
    _mcp_server_parameters,
    _session_client_kwargs,
    _validate_config,
)
from interactive_training.renderers.base import Message, ToolCall
from interactive_training.rl.message_env import _model_input_token_count


def test_mcp_extra_includes_image_processing_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        extras = tomllib.load(pyproject_file)["project"]["optional-dependencies"]

    assert set(extras["image"]) <= set(extras["mcp"])


class StubTool:
    name = "reveal_alien_color"
    description = "Reveal an alien color."
    parameters_schema: ClassVar[dict[str, str]] = {"type": "object"}

    def to_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input):
        raise AssertionError("dataset construction must not invoke tools")


def test_mixed_schedule_is_balanced_reproducible_and_held_out():
    tasks = build_mixed_tasks()
    assert tasks == build_mixed_tasks(seed=0)
    assert tasks != build_mixed_tasks(seed=1)
    assert len({task.question for task in tasks}) == 30
    _, test = build_curriculum_tasks()
    assert not {task.question for task in tasks} & {task.question for task in test}
    assert Counter(len(task.colors) for task in tasks) == {1: 10, 2: 10, 3: 10}
    for offset in range(0, 30, 3):
        assert [len(task.colors) for task in tasks[offset:offset + 3]] == [1, 2, 3]
    assert tasks[0].colors == ("pink",)
    assert "`Answer: <alien-word>`" in tasks[0].question
    assert "pink=<alien-word>" not in tasks[0].question
    assert "white=<alien-word>, yellow=<alien-word>" in tasks[1].question
    assert len(build_mixed_tasks(3)) == 3

    config = CLIConfig()
    production_tasks = build_mixed_tasks(config.mixed_train_examples)
    assert Counter(len(task.colors) for task in production_tasks) == {
        1: 10,
        2: 25,
        3: 25,
    }
    batches = [
        production_tasks[offset:offset + config.groups_per_batch]
        for offset in range(0, len(production_tasks), config.groups_per_batch)
    ]
    assert [Counter(len(task.colors) for task in batch) for batch in batches] == [
        {1: 4, 2: 3, 3: 3},
        {1: 3, 2: 4, 3: 3},
        {1: 3, 2: 3, 3: 4},
        {2: 5, 3: 5},
        {2: 5, 3: 5},
        {2: 5, 3: 5},
    ]

    for count in (0, 181):
        with pytest.raises(ValueError, match="mixed task count"):
            build_mixed_tasks(count)


def test_pixel_batches_bound_decoded_images_and_preserve_order():
    from interactive_training.recipes.mcp_image_tool.server import color_swatch_png

    image = SimpleNamespace(data=color_swatch_png("varkesh"))
    datums = [SimpleNamespace(model_input=SimpleNamespace(chunks=[image] * 3)) for _ in range(160)]
    batches = split_image_batches(datums)
    assert len(batches) > 1
    assert all(len(batch) * 3 * 640 * 360 <= 25_000_000 for batch in batches)
    assert [datum for batch in batches for datum in batch] == datums
    with pytest.raises(ValueError, match="single training example"):
        split_image_batches(datums, max_pixels=640 * 360)


@pytest.mark.asyncio
async def test_pixel_client_submits_all_requests_before_optimizer(monkeypatch):
    from unittest.mock import AsyncMock
    from interactive_training.recipes.mcp_image_tool import training_client

    events = []
    monkeypatch.setattr(training_client, "split_image_batches", lambda batch: [[datum] for datum in batch])

    async def forward(self, batch, loss_fn, loss_fn_config):
        events.append(("submit", batch[0]))
        async def complete():
            events.append(("complete", batch[0]))
            return SimpleNamespace(loss_fn_outputs=batch)
        return SimpleNamespace(result_async=complete)

    monkeypatch.setattr(training_client.AzureSDKTrainingClient, "forward_backward_async", forward)
    sdk = SimpleNamespace(optim_step_async=AsyncMock(return_value=AsyncMock()))
    client = MCPImageTrainingClient(sdk, "session_test", None)
    future = await client.forward_backward_async(["first", "second"], loss_fn="cross_entropy")
    assert events == [("submit", "first"), ("submit", "second")]
    await client.optim_step_async()
    sdk.optim_step_async.assert_awaited_once()
    assert (await future.result_async()).loss_fn_outputs == ["first", "second"]


@pytest.mark.parametrize("initial,post_sft,final,episodes,accepted", [
    (2 / 30, 10 / 30, 11 / 30, 30, True),
    (2 / 30, 11 / 30, 11 / 30, 30, False),
    (2 / 30, 12 / 30, 11 / 30, 30, False),
    (2 / 30, 2 / 30, 2 / 30, 30, False),
    (0.1, 0.4, 0.5, 30, False),
    (0.0, 0.2, 0.3, 30, False),
    (0.0, 0.4, 0.5, 29, False),
])
def test_learning_report_requires_accuracy_gain_on_complete_evaluation(
    initial, post_sft, final, episodes, accepted
):
    metrics = [
        {"stage": "before_sft", "step": -1, "test/env/all/correct": initial,
         "test/env/all/total_episodes": 30},
        {"stage": "before_rl", "step": 0, "test/env/all/correct": post_sft,
         "test/env/all/total_episodes": 30},
        {"stage": "after_rl", "step": 3, "test/env/all/correct": final,
         "test/env/all/total_episodes": episodes},
    ]
    report = build_learning_report(metrics, expected_episodes=30)
    assert report["accepted"] is accepted
    assert report["accuracy_gain"] == final - initial
    assert report["rl_accuracy_gain"] == final - post_sft


def test_learning_report_rejects_missing_evaluation():
    assert not build_learning_report([], expected_episodes=30)["accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task_index", [0, 10, 20])
async def test_demonstrations_execute_each_requested_tool_and_pass_exact_grader(task_index):
    calls = []

    class ImageTool(StubTool):
        async def run(self, input):
            calls.append(input.arguments["alien_word"])
            return ToolResult(messages=[alien_result(input.call_id)])

    task = build_tasks(30)[task_index]
    grader = AlienColorAnswerReward(
        expected=task.answers[0] if len(task.colors) == 1 else tuple(zip(task.colors, task.answers)),
        max_tool_calls=len(task.colors),
    )
    env = AgentToolMessageEnv(
        tools=[ImageTool()], initial_messages=[Message(role="user", content=task.question)],
        max_turns=len(task.colors) + 1, max_tool_calls=len(task.colors), reward_fn=grader,
    )
    history = await demonstrate_task(task, env)
    assert calls == list(task.answers)
    assert (await grader(history))[0] == 1.0
    assert history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_warmup_uses_only_training_builders_and_awaits_each_update(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    from azure.ai.finetuningsessions.models import Datum, TensorData

    task = build_tasks(1)[0]
    env = SimpleNamespace(message_env=SimpleNamespace(close=AsyncMock()), renderer=object())
    builder = SimpleNamespace(task=task, max_trajectory_tokens=16384, make_envs=AsyncMock(return_value=[env]))
    dataset = mcp_env.MCPImageToolDataset([builder], batch_size=10)
    monkeypatch.setattr(warmup.chz, "replace", lambda builder, **kwargs: builder)
    demonstrate = AsyncMock(return_value=[Message(role="assistant", content="Answer: varkesh")])
    monkeypatch.setattr(warmup, "demonstrate_task", demonstrate)
    datum = Datum(model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2])]), loss_fn_inputs={
        "target_tokens": TensorData(data=[2, 3]), "weights": TensorData(data=[0.0, 1.0]),
    })
    monkeypatch.setattr(warmup, "conversation_to_datum", lambda *args, **kwargs: datum)
    events = []

    class TrainingClient:
        async def forward_backward_async(self, batch, loss_fn):
            assert batch == [datum]
            assert loss_fn == "cross_entropy"
            events.append("forward")
            async def complete():
                events.append("forward_complete")
                return SimpleNamespace(loss_fn_outputs=[{"logprobs": TensorData(data=[-2.0, -1.0])}])
            return SimpleNamespace(result_async=complete)

        async def optim_step_async(self, params):
            assert params.learning_rate == 2e-4
            events.append("optim")
            async def complete():
                events.append("optim_complete")
            return SimpleNamespace(result_async=complete)

        async def save_state_async(self, name):
            assert name == "warmup"
            events.append("checkpoint")
            return SimpleNamespace(result_async=AsyncMock())

    await warmup.train_warmup(TrainingClient(), dataset, epochs=2, learning_rate=2e-4, seed=0, log_path=str(tmp_path))
    assert events == ["forward", "forward_complete", "optim", "optim_complete"] * 2 + ["checkpoint"]
    demonstrate.assert_awaited_once_with(task, env.message_env)
    env.message_env.close.assert_awaited_once()
    assert len((tmp_path / "warmup_metrics.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("overrides", [{"warmup_learning_rate": 0.0}])
def test_warmup_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError, match="warmup"):
        _validate_config(CLIConfig(project_endpoint="http://localhost:8000", **overrides))


@pytest.mark.asyncio
async def test_evaluation_is_persisted_before_training(monkeypatch, tmp_path):
    import json
    from unittest.mock import AsyncMock
    from interactive_training.recipes.mcp_image_tool import mixed_rl

    metrics = {"test/env/all/correct": 0.5, "test/env/all/total_episodes": 30}
    evaluate = AsyncMock(return_value=metrics)
    monkeypatch.setattr(mixed_rl.RLTestSetEvaluator, "__call__", evaluate)
    dataset = mcp_env.MCPImageToolDataset([], batch_size=10)
    result = await mixed_rl.evaluate_repeated(
        None, dataset, max_tokens=2048, repetitions=1, expected_episodes=30,
        log_path=str(tmp_path), stage="before_rl", step=0, seed=7,
    )
    assert result == {"stage": "before_rl", "step": 0, "repetitions": 1, **metrics}
    assert json.loads((tmp_path / "evaluations.jsonl").read_text()) == result


def test_multimodal_prompt_token_count_includes_image_tokens():
    model_input = ModelInput(
        chunks=[
            ModelInputChunk(tokens=[1, 2, 3]),
            ImageChunk(
                data=b"\x89PNG\r\n\x1a\n",
                format="png",
                expected_tokens=299,
            ),
        ]
    )

    assert _model_input_token_count(model_input) == 302


@pytest.mark.asyncio
@pytest.mark.parametrize("rewards,updates", [
    ([[1.0, 1.0]], 0), ([[0.0, 0.0]], 0), ([[0.0, 1.0]], 1),
    ([[0.3, 0.3], [0.2, 0.5]], 1),
    ([[0.2, 0.5], [0.3, 0.3], [0.25, 0.5]], 2),
])
async def test_mixed_rl_gates_updates_without_changing_original_datasets(monkeypatch, tmp_path, rewards, updates):
    import json
    from unittest.mock import AsyncMock
    from interactive_training.recipes.mcp_image_tool import mixed_rl

    dataset_builder = MCPImageToolDatasetBuilder(
        model_name="base-model", tokenizer_name="base-model", renderer_name=None,
        tools=[StubTool()], batch_size=10, group_size=2, train_examples=80, test_examples=30,
    )
    train, test = await dataset_builder()
    original_train = list(train.builders)
    original_test = list(test.builders)
    groups = [SimpleNamespace(trajectories_G=[SimpleNamespace(transitions=[])] * 2, metrics_G=[{}, {}],
                              get_total_rewards=lambda values=values: values)
              for values in rewards for _ in range(10)]
    rollout = AsyncMock(side_effect=groups)
    update = AsyncMock(return_value=("after", {"grad_norm": 1.0}))
    evaluation = AsyncMock(return_value={"test/env/all/correct": 0.5,
                                        "test/env/all/tool_calls": 7.0,
                                        "test/env/all/verified_answer": 1.0,
                                        "test/env/all/reward/total": 0.4,
                                        "test/env/all/total_episodes": 30})
    monkeypatch.setattr(mixed_rl, "do_group_rollout_and_filter_constant_reward", rollout)
    monkeypatch.setattr(mixed_rl, "do_train_step_and_get_sampling_client", update)
    monkeypatch.setattr(mixed_rl.RLTestSetEvaluator, "__call__", evaluation)
    checkpoint = AsyncMock()
    client = SimpleNamespace(
        save_weights_and_get_sampling_client_async=AsyncMock(return_value="before"),
        save_state_async=AsyncMock(return_value=SimpleNamespace(result_async=checkpoint)),
    )
    config = train_azure.Config(
        model_name="base-model", dataset_builder=dataset_builder, log_path=str(tmp_path),
        learning_rate=5e-5, max_tokens=2048, sampling_seed=0, max_steps=len(rewards),
    )
    report = await mixed_rl.train_mixed(
        config, client, None, train, test, task_count=30, repetitions=3, expected_episodes=30,
    )
    assert report["rl_updates"] == updates
    assert update.await_count == updates
    assert report["rl_stop_reason"] == "budget_exhausted"
    assert report["rl_skipped_batches"] == len(rewards) - updates
    assert report["final_evaluation_reused"] == (updates == 0)
    assert evaluation.await_count == (6 if updates else 3)
    assert rollout.await_count == 10 * len(rewards)
    for call in rollout.await_args_list:
        assert not call.kwargs["do_remove_constant_reward_groups"]
        assert len(call.args[1].task.colors) in (1, 2, 3)
    assert train.builders == original_train
    assert test.builders == original_test
    assert [call.args[0] for call in client.save_state_async.await_args_list] == [
        *[f"rl_step_{step}" for step in range(1, updates + 1)], "final",
    ]
    assert checkpoint.await_count == updates + 1
    assert report["before_rl_tool_calls"] == report["after_rl_tool_calls"] == 7.0
    assert report["rl_tool_calls_change"] == 0.0
    assert not report["rl_efficiency_improved"]
    probes = [json.loads(line) for line in (tmp_path / "probe.jsonl").read_text().splitlines()]
    assert len(probes) == len(rewards)
    for probe, values in zip(probes, rewards, strict=True):
        assert probe["mixed_groups"] == (10 if max(values) > min(values) else 0)
        assert len(probe["tasks"]) == 10
    rollouts = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    assert len(rollouts) == 20 * len(rewards)
    assert [row["reward"] for row in rollouts] == [value for values in rewards for _ in range(10) for value in values]


@pytest.mark.asyncio
async def test_rollout_budget_returns_control_for_checkpoint_and_evaluation():
    import asyncio
    from interactive_training.recipes.mcp_image_tool.mixed_rl import rollout_with_budget

    async def pending():
        await asyncio.Event().wait()

    assert await rollout_with_budget(pending(), 0.001) is None


@pytest.mark.asyncio
async def test_rollout_budget_preserves_underlying_timeout():
    from interactive_training.recipes.mcp_image_tool.mixed_rl import rollout_with_budget

    async def failed():
        raise TimeoutError("sampling service timeout")

    with pytest.raises(TimeoutError, match="sampling service timeout"):
        await rollout_with_budget(failed(), 1)


@pytest.mark.asyncio
async def test_repeated_evaluation_handles_optional_metrics(monkeypatch, tmp_path):
    import json
    from unittest.mock import AsyncMock
    from interactive_training.recipes.mcp_image_tool import mixed_rl

    common = {"test/env/all/correct": 0.1, "test/env/all/total_episodes": 30,
              "test/env/all/verified_answer": 0.8, "test/env/all/tool_calls": 6.0}
    optional = "test/env/mcp_alien_color/parse_error"
    evaluation = AsyncMock(side_effect=[dict(common, **{optional: 1.0}), common])
    monkeypatch.setattr(mixed_rl.RLTestSetEvaluator, "__call__", evaluation)
    result = await mixed_rl.evaluate_repeated(
        None, mcp_env.MCPImageToolDataset([], 10), max_tokens=2048, repetitions=2,
        expected_episodes=30, log_path=str(tmp_path), stage="before_rl", step=0,
        seed=7,
    )
    assert optional not in result
    assert all(result[key] == value for key, value in common.items())
    passes = [json.loads(line) for line in (tmp_path / "evaluation_passes.jsonl").read_text().splitlines()]
    assert passes[0][optional] == 1.0
    assert optional not in passes[1]


@pytest.mark.asyncio
async def test_repeated_evaluation_persists_failed_pass_but_rejects_comparison(monkeypatch, tmp_path):
    import json
    from unittest.mock import AsyncMock
    from interactive_training.recipes.mcp_image_tool import mixed_rl

    evaluation = AsyncMock(return_value={"test/env/all/correct": 1.0,
                                        "test/env/all/total_episodes": 29})
    monkeypatch.setattr(mixed_rl.RLTestSetEvaluator, "__call__", evaluation)
    with pytest.raises(RuntimeError, match="Incomplete before_rl"):
        await mixed_rl.evaluate_repeated(
            None, mcp_env.MCPImageToolDataset([], 10), max_tokens=2048, repetitions=3,
            expected_episodes=30, log_path=str(tmp_path), stage="before_rl", step=0,
            seed=7,
        )
    assert not (tmp_path / "evaluations.jsonl").exists()
    assert json.loads((tmp_path / "evaluation_passes.jsonl").read_text())["test/env/all/total_episodes"] == 29


@pytest.mark.parametrize("overrides", [
    {"max_steps_off_policy": 1}, {"kl_penalty_coef": 0.1},
    {"remove_constant_reward_groups": True}, {"wandb_project": "test"},
    {"eval_every": 1}, {"rl_schedule": "curriculum"},
    {"format_coef": 1.0}, {"save_every": 1},
    {"eval_strategy": "steps"}, {"wandb_name": "test"},
])
def test_fixed_workflow_does_not_expose_alternate_modes(overrides):
    with pytest.raises(TypeError, match="unexpected keyword"):
        CLIConfig(project_endpoint="http://localhost:8000", **overrides)


@pytest.mark.parametrize("name", ["num_substeps", "max_tool_calls", "warmup_epochs"])
@pytest.mark.parametrize("value", [0, -1])
def test_training_limits_must_be_positive(name, value):
    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        _validate_config(CLIConfig(mcp_smoke_only=True, **{name: value}))


def test_alien_color_tasks_are_deterministic_and_cover_every_color():
    tasks = build_tasks(30)

    assert tasks == build_tasks(30)
    assert all("query" not in task.question.lower() for task in tasks)
    assert all("find" not in task.question.lower() for task in tasks)
    assert {color for task in tasks for color in task.colors} == set(COLORS)
    assert [len(task.colors) for task in tasks].count(1) == 10
    assert [len(task.colors) for task in tasks].count(2) == 10
    assert [len(task.colors) for task in tasks].count(3) == 10
    assert len(ALIEN_WORD_TO_COLOR) == 10
    assert all(
        answer not in task.question
        for task in tasks
        for answer in task.answers
    )
    assert "Answer: red=<alien-word>, green=<alien-word>" in tasks[10].question
    assert (
        "Answer: red=<alien-word>, green=<alien-word>, pink=<alien-word>"
        in tasks[20].question
    )


def test_curriculum_repeats_atomic_recall_before_held_out_composition():
    train, test = build_curriculum_tasks()

    assert len(train) == 40
    assert len(test) == 30
    assert [len(task.colors) for task in train] == [1] * 20 + [2] * 10 + [3] * 10
    assert [task.colors[0] for task in train[:20]].count("red") == 2
    assert {task.colors[0] for task in train[:20]} == set(COLORS)
    assert all(
        count == 7
        for count in Counter(color for task in train for color in task.colors).values()
    )
    assert {task.task_id // 30 for task in train if len(task.colors) == 2} == set(range(5))
    assert {task.task_id // 30 for task in train if len(task.colors) == 3} == set(range(5))
    assert [len(task.colors) for task in test] == [1] * 10 + [2] * 10 + [3] * 10
    assert {task.question for task in train}.isdisjoint(
        task.question for task in test
    )


def test_alien_word_order_is_shared_across_group_members():
    orders = _alien_word_orders(task_id=7, group_size=10, seed=3)
    expected_words = set(ALIEN_WORD_TO_COLOR)

    assert all(set(order) == expected_words for order in orders)
    assert len(set(orders)) == 1
    assert orders == _alien_word_orders(task_id=7, group_size=10, seed=3)


def test_tool_call_limit_honors_configured_budget_for_every_task_size():
    tasks = build_tasks(30)

    assert _task_tool_call_limit(tasks[0], 20) == 2
    assert _task_tool_call_limit(tasks[10], 20) == 3
    assert _task_tool_call_limit(tasks[20], 20) == 4
    assert _task_tool_call_limit(tasks[20], 2) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("task_index,call_limit", [(0, 2), (10, 3), (20, 4)])
async def test_constructed_env_enforces_task_budget_and_zero_parse_reward(
    monkeypatch, task_index, call_limit
):
    monkeypatch.setattr(mcp_env.tokenizer_utils, "get_tokenizer", lambda _: None)
    monkeypatch.setattr(mcp_env, "get_image_processor", lambda _: None)
    monkeypatch.setattr(mcp_env, "get_renderer", lambda *args, **kwargs: SimpleNamespace(
        create_conversation_prefix_with_tools=lambda **kwargs: []
    ))
    monkeypatch.setattr(mcp_env, "build_agent_tool_env", lambda **kwargs: kwargs)
    builder = mcp_env.MCPImageToolEnvGroupBuilder(
        task=build_tasks(30)[task_index], model_name="stub", tokenizer_name="stub",
        renderer_name="stub", group_size=2, tools=[], max_tool_calls=20,
    )
    envs = await builder.make_envs()
    assert len(envs) == 2
    for env in envs:
        assert env["max_tool_calls"] == call_limit
        assert env["max_turns"] == call_limit + 1
        assert env["failed_parse_reward"] == 0.0


def test_system_prompt_requires_recall_then_tool_verification():
    assert "one successful call for each requested color" in SYSTEM_PROMPT
    assert "you may try another word" in SYSTEM_PROMPT
    assert "reward is binary" in SYSTEM_PROMPT
    assert "extra" in SYSTEM_PROMPT and "earns zero" in SYSTEM_PROMPT
    assert "avoid repeated calls" in TOOL_DESCRIPTION


def alien_call(word: str, call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolCall.FunctionBody(
                    name="reveal_alien_color",
                    arguments=f'{{"alien_word":"{word}"}}',
                ),
            )
        ],
    )


def alien_result(call_id: str) -> Message:
    return Message(
        role="tool",
        content=[
            {"type": "text", "text": "Oracle result:"},
            {"type": "image", "image": "data:image/png;base64,iVBORw0KGgo="},
        ],
        name="reveal_alien_color",
        tool_call_id=call_id,
    )


@pytest.mark.asyncio
async def test_group_diagnostics_preserve_reward_and_record_executed_history():
    import json

    builder = mcp_env.MCPImageToolEnvGroupBuilder(
        task=build_tasks(1)[0], model_name="stub", tokenizer_name="stub",
        renderer_name="stub", group_size=1, tools=[], seed=0,
    )
    history = [
        alien_call("nalmek", "wrong"), alien_result("wrong"),
        alien_call("varkesh", "right"), alien_result("right"),
        alien_call("bexith", "unexecuted"),
        Message(role="assistant", content="Answer: varkesh"),
    ]
    transition = SimpleNamespace(reward=1 / 3, metrics={"verified_answer": 1.0}, logs={})
    trajectory = SimpleNamespace(transitions=[transition])
    env = SimpleNamespace(message_env=SimpleNamespace(history=history))
    assert await builder.compute_group_rewards([trajectory], [env]) == [(0.0, {})]
    assert transition.reward == 1 / 3
    assert transition.metrics == {"verified_answer": 1.0}
    serialized = transition.logs["mcp_diagnostics"]
    diagnostics = json.loads(serialized)
    assert diagnostics["executed_words"] == ["nalmek", "varkesh"]
    assert diagnostics["executed_call_count"] == 2
    assert diagnostics["proposed_call_count"] == 3
    assert diagnostics["final_answer"] == "Answer: varkesh"
    assert diagnostics["final_answer_present"]
    assert diagnostics["schema_word_order"] == list(_alien_word_orders(task_id=0, group_size=1, seed=0)[0])
    assert "base64" not in serialized


@pytest.mark.asyncio
async def test_alien_color_reward_requires_executed_tool_evidence():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=20)

    assert await reward([{"role": "assistant", "content": "Answer: varkesh"}]) == (
        0.0,
        {
            "format": 1.0,
            "correct": 0.0,
            "verified_answer": 0.0,
            "verified_fraction": 0.0,
            "color_found": 0.0,
            "efficiency": 0.0,
            "tool_calls": 0.0,
            "mistakes": 0.0,
            "wrong_calls": 0.0,
            "post_discovery_calls": 0.0,
        },
    )
    assert await reward(
        [
            alien_call("varkesh", "unexecuted"),
            Message(role="assistant", content="Answer: varkesh"),
        ]
    ) == (
        0.0,
        {
            "format": 1.0,
            "correct": 0.0,
            "verified_answer": 0.0,
            "verified_fraction": 0.0,
            "color_found": 0.0,
            "efficiency": 0.0,
            "tool_calls": 0.0,
            "mistakes": 0.0,
            "wrong_calls": 0.0,
            "post_discovery_calls": 0.0,
        },
    )


@pytest.mark.asyncio
async def test_reward_charges_search_calls_truncated_by_execution_budget():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=1)
    proposed_search = alien_call("varkesh", "right")
    proposed_search["tool_calls"].extend(
        alien_call(word, word)["tool_calls"][0]
        for word in ALIEN_WORD_TO_COLOR
        if word != "varkesh"
    )
    score, metrics = await reward([
        proposed_search,
        alien_result("right"),
        Message(role="assistant", content="Answer: varkesh"),
    ])

    assert score == 0.0
    assert metrics["correct"] == 0.0


@pytest.mark.asyncio
async def test_alien_color_reward_gives_discounted_exploratory_credit():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=20)
    direct_history = [
        alien_call("varkesh", "direct"),
        alien_result("direct"),
        Message(role="assistant", content="Answer: varkesh"),
    ]
    searched_history = [
        alien_call("nalmek", "wrong"),
        alien_result("wrong"),
        alien_call("varkesh", "right"),
        alien_result("right"),
        Message(role="assistant", content="Answer: varkesh"),
    ]

    direct_reward, direct_metrics = await reward(direct_history)
    searched_reward, searched_metrics = await reward(searched_history)

    assert direct_reward == 1.0
    assert direct_metrics == {
        "format": 1.0,
        "correct": 1.0,
        "verified_answer": 1.0,
        "verified_fraction": 1.0,
        "color_found": 1.0,
        "efficiency": 1.0,
        "tool_calls": 1.0,
        "mistakes": 0.0,
        "wrong_calls": 0.0,
        "post_discovery_calls": 0.0,
    }
    assert searched_reward == 0.0
    assert searched_metrics["efficiency"] == 0.5
    assert searched_metrics["correct"] == 0.0
    assert searched_metrics["verified_answer"] == 1.0
    assert searched_metrics["mistakes"] == 1.0
    assert searched_metrics["wrong_calls"] == 1.0
    assert searched_metrics["post_discovery_calls"] == 0.0
    assert searched_reward < direct_reward


@pytest.mark.asyncio
async def test_multi_color_reward_gives_partial_credit_for_verified_answers():
    reward = AlienColorAnswerReward(
        expected=(("green", "irevax"), ("blue", "nalmek")),
        max_tool_calls=20,
    )
    complete_history = [
        alien_call("irevax", "green"),
        alien_result("green"),
        alien_call("nalmek", "blue"),
        alien_result("blue"),
        Message(
            role="assistant",
            content="Answer: green=irevax, blue=nalmek",
        ),
    ]
    partial_history = [
        alien_call("irevax", "green"),
        alien_result("green"),
        Message(role="assistant", content="Answer: green=irevax"),
    ]

    complete_reward, complete_metrics = await reward(complete_history)
    partial_reward, partial_metrics = await reward(partial_history)

    assert complete_reward == 1.0
    assert complete_metrics["correct"] == 1.0
    assert complete_metrics["verified_answer"] == 1.0
    assert complete_metrics["verified_fraction"] == 1.0
    assert complete_metrics["color_found"] == 1.0
    assert complete_metrics["tool_calls"] == 2.0
    assert partial_reward == 0.0
    assert partial_metrics["correct"] == 0.0
    assert partial_metrics["verified_answer"] == 0.0
    assert partial_metrics["verified_fraction"] == 0.5
    assert partial_metrics["color_found"] == 0.5


@pytest.mark.asyncio
async def test_multi_color_reward_requires_exactly_one_call_per_color():
    reward = AlienColorAnswerReward(
        expected=(
            ("green", "irevax"),
            ("blue", "nalmek"),
            ("red", "varkesh"),
        ),
        max_tool_calls=20,
    )
    exact_history = [
        alien_call("varkesh", "red"),
        alien_result("red"),
        alien_call("irevax", "green"),
        alien_result("green"),
        alien_call("nalmek", "blue"),
        alien_result("blue"),
        Message(
            role="assistant",
            content="Answer: green=irevax, blue=nalmek, red=varkesh",
        ),
    ]
    extra_call_history = [
        alien_call("quorali", "extra"),
        alien_result("extra"),
        *exact_history,
    ]

    exact_reward, exact_metrics = await reward(exact_history)
    extra_reward, extra_metrics = await reward(extra_call_history)

    assert exact_reward == 1.0
    assert exact_metrics["correct"] == 1.0
    assert exact_metrics["tool_calls"] == 3.0
    assert exact_metrics["mistakes"] == 0.0
    assert extra_reward == 0.0
    assert extra_metrics["correct"] == 0.0
    assert extra_metrics["tool_calls"] == 4.0
    assert extra_metrics["mistakes"] == 1.0


@pytest.mark.asyncio
async def test_alien_color_reward_penalizes_calls_after_discovery():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=20)
    immediate_history = [
        alien_call("varkesh", "right"),
        alien_result("right"),
        Message(role="assistant", content="Answer: varkesh"),
    ]
    delayed_history = [
        alien_call("varkesh", "right"),
        alien_result("right"),
        alien_call("nalmek", "unnecessary"),
        alien_result("unnecessary"),
        Message(role="assistant", content="Answer: varkesh"),
    ]

    immediate_reward, _ = await reward(immediate_history)
    delayed_reward, delayed_metrics = await reward(delayed_history)

    assert delayed_reward == 0.0
    assert delayed_metrics["wrong_calls"] == 0.0
    assert delayed_metrics["correct"] == 0.0
    assert delayed_metrics["verified_answer"] == 1.0
    assert delayed_metrics["post_discovery_calls"] == 1.0
    assert delayed_metrics["mistakes"] == 1.0
    assert delayed_reward < immediate_reward


@pytest.mark.asyncio
async def test_alien_color_reward_rejects_answer_not_found_by_tool():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=20)

    assert await reward(
        [
            alien_call("nalmek", "wrong"),
            alien_result("wrong"),
            Message(role="assistant", content="Answer: varkesh"),
        ]
    ) == (
        0.0,
        {
            "format": 1.0,
            "correct": 0.0,
            "verified_answer": 0.0,
            "verified_fraction": 0.0,
            "color_found": 0.0,
            "efficiency": 0.0,
            "tool_calls": 1.0,
            "mistakes": 1.0,
            "wrong_calls": 1.0,
            "post_discovery_calls": 0.0,
        },
    )


@pytest.mark.asyncio
async def test_alien_color_reward_rejects_text_only_tool_error():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=20)
    history = [
        alien_call("varkesh", "failed"),
        Message(
            role="tool",
            content='{"error":"execution failed"}',
            name="reveal_alien_color",
            tool_call_id="failed",
        ),
        Message(role="assistant", content="Answer: varkesh"),
    ]

    result, metrics = await reward(history)

    assert result == 0.0
    assert metrics["correct"] == 0.0
    assert metrics["verified_answer"] == 0.0
    assert metrics["color_found"] == 0.0
    assert metrics["tool_calls"] == 1.0
    assert metrics["wrong_calls"] == 1.0


@pytest.mark.asyncio
async def test_alien_color_reward_gives_lookup_credit_but_requires_final_answer():
    reward = AlienColorAnswerReward(expected="varkesh", max_tool_calls=20)

    result = await reward([alien_call("varkesh", "right"), alien_result("right")])

    assert result[0] == pytest.approx(-0.1)
    assert result[1]["correct"] == 0.0
    assert result[1]["color_found"] == 1.0
    assert result[1]["format"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    "Answer: green=irevax, blue=nalmek, red=varkesh",
    "Answer: blue=nalmek, green=irevax",
    "I think Answer: green=irevax, blue=nalmek",
])
async def test_full_reward_requires_exact_final_answer_contract(answer):
    reward = AlienColorAnswerReward(
        expected=(("green", "irevax"), ("blue", "nalmek")),
        max_tool_calls=2,
        format_coef=0.0,
    )
    score, metrics = await reward([
        alien_call("irevax", "green"), alien_result("green"),
        alien_call("nalmek", "blue"), alien_result("blue"),
        Message(role="assistant", content=answer),
    ])
    assert score == 0.0
    assert metrics["correct"] == 0.0


@pytest.mark.asyncio
async def test_image_tool_dataset_has_disjoint_train_and_test_tasks():
    builder = MCPImageToolDatasetBuilder(
        model_name="base-model",
        tokenizer_name="tokenizer-model",
        tools=[StubTool()],
        batch_size=4,
        group_size=3,
        train_examples=48,
        test_examples=12,
    )

    train, test = await builder()

    assert test is not None
    assert len(train) == 12
    assert len(test) == 3
    train_task_ids = {group.task.task_id for group in train.builders}
    test_task_ids = {group.task.task_id for group in test.builders}
    train_questions = {group.task.question for group in train.builders}
    test_questions = {group.task.question for group in test.builders}
    assert train_task_ids.isdisjoint(test_task_ids)
    assert train_questions.isdisjoint(test_questions)
    assert len(train_questions | test_questions) == 60
    assert all(group.group_size == 3 for group in train.builders)
    assert all(group.group_size == 1 for group in test.builders)
    assert all(group.model_name == "base-model" for group in train.builders)
    assert all(group.tokenizer_name == "tokenizer-model" for group in train.builders)


def test_stdio_defaults_to_bundled_server():
    config = CLIConfig(mcp_smoke_only=True)

    parameters = _mcp_server_parameters(config)

    assert parameters.args == [
        "-m",
        "interactive_training.recipes.mcp_image_tool.server",
        "--server",
    ]


def test_training_defaults_use_compact_warmup_and_bounded_mixed_rl():
    config = CLIConfig(mcp_smoke_only=True)

    assert config.group_size == 8
    assert config.max_tool_calls == 20
    assert config.groups_per_batch == 10
    assert config.max_concurrent_groups == 4
    assert config.max_train_examples == 40
    assert config.max_test_examples == 30
    assert config.learning_rate == 5e-6
    assert config.warmup_learning_rate == 2e-4
    assert config.warmup_epochs == 3
    assert config.lora_seed == 552161550
    assert config.mixed_train_examples == 60
    assert config.evaluation_repetitions == 1
    assert config.max_steps == 6
    assert config.max_wall_clock_seconds is None
    assert config.temperature == 1.0
    assert config.seed == 0
    assert config.load_checkpoint_path is None
    _validate_config(config)


def test_streamable_http_requires_url_and_rejects_stdio_options():
    with pytest.raises(ValueError, match="mcp_server_url is required"):
        _validate_config(
            CLIConfig(mcp_transport="streamable_http", mcp_smoke_only=True)
        )

    with pytest.raises(ValueError, match="only valid for stdio"):
        _validate_config(
            CLIConfig(
                mcp_transport="streamable_http",
                mcp_server_url="https://mcp.example.test",
                mcp_server_command="python",
                mcp_smoke_only=True,
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("authenticated", [True, False])
async def test_streamable_http_sends_configured_bearer_token(monkeypatch, caplog, authenticated):
    import json
    import logging
    import secrets
    import httpx

    token = secrets.token_urlsafe(24)
    azure_token = secrets.token_urlsafe(24)
    monkeypatch.setenv("TEST_MCP_TOKEN", token)
    monkeypatch.setenv("AZURE_AI_API_KEY", azure_token)
    caplog.set_level(logging.DEBUG)
    methods = []

    def handle_request(request):
        authorization = request.headers.get("Authorization")
        if authenticated:
            assert authorization == f"Bearer {token}"
            assert authorization != token
        else:
            assert authorization is None
        assert str(request.url) == "https://mcp.example.test/mcp"
        payload = json.loads(request.content)
        methods.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": payload["id"],
                "result": {
                    "protocolVersion": payload["params"]["protocolVersion"],
                    "capabilities": {},
                    "serverInfo": {"name": "test-mcp", "version": "1.0"},
                },
            })
        return httpx.Response(202)

    client_class = httpx.AsyncClient

    def create_http_client(**kwargs):
        return client_class(transport=httpx.MockTransport(handle_request), **kwargs)

    monkeypatch.setattr(train_azure.httpx, "AsyncClient", create_http_client)

    config = CLIConfig(
        mcp_transport="streamable_http",
        mcp_server_url="https://mcp.example.test/mcp",
        mcp_bearer_token_env="TEST_MCP_TOKEN" if authenticated else None,
        mcp_smoke_only=True,
    )
    async with train_azure._mcp_session(config):
        pass

    assert "initialize" in methods
    credentials_logged = any(value in caplog.text for value in (token, azure_token))
    assert not credentials_logged, "Credentials appeared in logs"


def test_training_requires_project_endpoint_and_remote_mcp_requires_https():
    with pytest.raises(ValueError, match="project_endpoint is required"):
        _validate_config(CLIConfig())

    with pytest.raises(ValueError, match="must use HTTPS"):
        _validate_config(
            CLIConfig(
                mcp_transport="streamable_http",
                mcp_server_url="http://mcp.example.test/mcp",
                mcp_smoke_only=True,
            )
        )

    _validate_config(
        CLIConfig(
            mcp_transport="streamable_http",
            mcp_server_url="http://127.0.0.1:8000/mcp",
            mcp_smoke_only=True,
        )
    )


def test_local_session_client_uses_dummy_key_and_allows_http():
    client_kwargs, credential = _session_client_kwargs(
        "http://localhost:8000",
        azure_api_key=None,
    )

    assert client_kwargs["allow_insecure_http"] is True
    assert client_kwargs["credential"] is credential
    assert "credential_scopes" not in client_kwargs
