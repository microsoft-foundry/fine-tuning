from __future__ import annotations

from typing import ClassVar

import pytest

from interactive_training.renderers.base import Message, ToolCall
from interactive_training.tool_use.agent_tool_message_env import (
    AgentToolMessageEnv,
    build_agent_tool_env,
)
from interactive_training.tool_use.types import ToolInput, ToolResult


class RecordingTool:
    name = "color_oracle"
    description = "Return a color."
    parameters_schema: ClassVar[dict[str, object]] = {"type": "object"}

    def __init__(self) -> None:
        self.inputs: list[ToolInput] = []

    def to_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, tool_input: ToolInput) -> ToolResult:
        self.inputs.append(tool_input)
        return ToolResult(
            messages=[
                Message(
                    role="tool",
                    content="result",
                    tool_call_id=tool_input.call_id or "",
                    name=self.name,
                )
            ]
        )


def tool_message(call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolCall.FunctionBody(name="color_oracle", arguments="{}"),
            )
        ],
    )


@pytest.mark.asyncio
async def test_tool_call_limit_returns_last_result_then_rejects_another_call():
    tool = RecordingTool()

    async def reward_fn(history: list[Message]):
        return 0.5, {"graded": 1.0}

    env = AgentToolMessageEnv(
        tools=[tool],
        initial_messages=[Message(role="user", content="Find red")],
        max_turns=10,
        max_tool_calls=2,
        reward_fn=reward_fn,
    )

    first = await env.step(tool_message("call-1"))
    second = await env.step(tool_message("call-2"))
    over_limit = await env.step(tool_message("call-3"))

    assert not first.episode_done
    assert not second.episode_done
    assert len(tool.inputs) == 2
    assert over_limit.episode_done
    assert over_limit.reward == 0.5
    assert over_limit.metrics == {"max_tool_calls": 1.0, "graded": 1.0}


def test_builder_threads_parse_failure_metrics():
    class Renderer:
        def get_stop_sequences(self):
            return None

    async def reward_fn(history: list[Message]):
        return 0.0, {"correct": 0.0}

    env = build_agent_tool_env(
        renderer=Renderer(),
        tools=[RecordingTool()],
        initial_messages=[Message(role="user", content="Find red")],
        reward_fn=reward_fn,
        parse_failure_metrics={"correct": 0.0},
    )

    assert env.parse_failure_metrics == {"correct": 0.0}
