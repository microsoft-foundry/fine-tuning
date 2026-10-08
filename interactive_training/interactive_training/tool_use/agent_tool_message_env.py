"""Tool-using agent environment."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from interactive_training.renderers import Renderer
from interactive_training.renderers.base import Message, ToolCall
from interactive_training.rl.message_env import (
    EnvFromMessageEnv,
    MessageEnv,
    MessageStepResult,
)
from interactive_training.tool_use.tools import handle_tool_call
from interactive_training.tool_use.types import Tool

RewardResult = tuple[float, dict[str, float]]
RewardFn = Callable[[list[Message]], Awaitable[RewardResult]]
# TODO(tyler): Consider supporting stateful tools that need to grade rollouts based on
# information not contained in the message history (e.g., internal tool state that changes
# during execution).


@dataclass
class AgentToolMessageEnv(MessageEnv):
    """Generic tool-use MessageEnv for agents."""

    tools: list[Tool]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: RewardFn
    max_tool_calls: int | None = None
    history: list[Message] = field(default_factory=list)

    _turn_count: int = 0
    _tool_call_count: int = 0
    _tool_dict: dict[str, Tool] = field(default_factory=dict, init=False)
    _should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Execute tool calls and append results to history.

        Note: Tool metrics are not accumulated in the message history.
        Only messages and should_stop are used from ToolResult.
        """
        tool_results = await asyncio.gather(
            *[handle_tool_call(self._tool_dict, tc) for tc in tool_calls]
        )

        all_messages: list[Message] = []

        for tool_call, tool_result in zip(tool_calls, tool_results, strict=True):
            # Append messages to history
            for msg in tool_result.messages:
                self.history.append(msg)
                all_messages.append(msg)

            # Check if any tool signals to stop
            if tool_result.should_stop:
                self._should_stop = True

        return all_messages

    async def step(self, message: Message) -> MessageStepResult:
        """Execute any tools and return next messages.

        The episode ends when:
        - no tool calls in message (model decided to stop)
        - a tool returns should_stop=True
        - max_turns reached

        reward_fn is called once at episode end to grade the full trajectory.
        """
        self._turn_count += 1
        metrics: dict[str, float] = {}

        # Append the message to history
        self.history.append(message)

        # Extract and execute tool calls if present
        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        tool_call_limit_exceeded = False
        if tool_calls:
            calls_to_execute = tool_calls
            if self.max_tool_calls is not None:
                remaining_calls = max(self.max_tool_calls - self._tool_call_count, 0)
                calls_to_execute = tool_calls[:remaining_calls]
                tool_call_limit_exceeded = len(calls_to_execute) < len(tool_calls)
            if calls_to_execute:
                await self._handle_tool_calls(calls_to_execute)
                self._tool_call_count += len(calls_to_execute)

        # Determine if episode is done
        no_tool_calls = len(tool_calls) == 0
        max_turns_reached = self._turn_count >= self.max_turns
        done = (
            no_tool_calls
            or max_turns_reached
            or self._should_stop
            or tool_call_limit_exceeded
        )

        if max_turns_reached and not no_tool_calls:
            metrics["max_turns"] = 1.0
        if self._should_stop:
            metrics["tool_stopped"] = 1.0
        if tool_call_limit_exceeded:
            metrics["max_tool_calls"] = 1.0

        reward = 0.0
        if done:
            reward, reward_metrics = await self.reward_fn(self.history)
            metrics.update(reward_metrics)

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=metrics,
        )

    async def close(self) -> None:
        close_operations = []
        for tool in self.tools:
            aclose = getattr(tool, "aclose", None)
            if aclose is not None:
                close_operations.append(aclose())
        if close_operations:
            await asyncio.gather(*close_operations)


def build_agent_tool_env(
    renderer: Renderer,
    tools: list[Tool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    *,
    max_turns: int = 5,
    max_tool_calls: int | None = None,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = None,
    parse_failure_metrics: dict[str, float] | None = None,
) -> EnvFromMessageEnv:
    """Convenience method to build an EnvFromMessageEnv for tool-using agents.

    Args:
        renderer: The renderer for tokenizing messages.
        tools: List of tools the agent can call (must implement Tool protocol).
        initial_messages: Initial conversation history (system prompt, user message, etc.).
        reward_fn: Function that grades a completed episode. Takes the full message
            history and returns (reward, metrics). Called once at episode end.
        max_turns: Maximum turns before episode ends.
        max_tool_calls: Maximum tool calls to execute. The final allowed result is
            returned to the model, but a subsequent tool call ends the episode.
        failed_parse_reward: Reward when model output fails to parse.
        max_trajectory_tokens: Maximum tokens in trajectory before terminating episode.
        parse_failure_metrics: Metrics emitted as zero-valued outcomes when parsing fails.

    Returns:
        An EnvFromMessageEnv ready for RL training.
    """
    msg_env = AgentToolMessageEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn,
        max_tool_calls=max_tool_calls,
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        tool_specs=[tool.to_spec() for tool in tools],
        parse_failure_metrics=parse_failure_metrics,
    )
