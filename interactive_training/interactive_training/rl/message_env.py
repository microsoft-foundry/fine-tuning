"""Message-level environment abstraction.

MessageEnv operates at the message level (list[Message]) rather than token level.

EnvFromMessageEnv bridges MessageEnv to the token-level Env interface used by
the RL training loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from azure.ai.finetuningsessions.models import ImageChunk, ModelInput

from interactive_training.completers import StopCondition
from interactive_training.renderers import Renderer
from interactive_training.renderers.base import Message, ToolSpec
from interactive_training.rl import types


def _model_input_token_count(model_input: ModelInput) -> int:
    return sum(
        chunk.expected_tokens if isinstance(chunk, ImageChunk) else len(chunk.tokens)
        for chunk in model_input.chunks
    )


@dataclass
class MessageStepResult:
    """Result of a message-level step."""

    reward: float
    episode_done: bool
    next_messages: list[Message]
    metrics: dict[str, float] = field(default_factory=dict)
    next_stop_condition: StopCondition | None = None


class MessageEnv(ABC):
    """Abstract base class for message-level environments."""

    @abstractmethod
    async def initial_observation(self) -> list[Message]:
        """Return the initial conversation history as renderer messages."""
        ...

    @abstractmethod
    async def step(self, message: Message) -> MessageStepResult:
        """Process an assistant message and return reward/next state."""
        ...

    async def close(self) -> None:
        """Release resources held by the environment."""


class EnvFromMessageEnv(types.Env):
    """Adapter that wraps a MessageEnv to implement the token-level Env interface.

    This bridges the message-level abstraction to the token-level interface
    expected by the RL training loop.
    """

    def __init__(
        self,
        renderer: Renderer,
        message_env: MessageEnv,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
        max_trajectory_tokens: int | None = None,
        tool_specs: list[ToolSpec] | None = None,
        parse_failure_metrics: dict[str, float] | None = None,
    ):
        self.renderer = renderer
        self.message_env = message_env
        self.failed_parse_reward = failed_parse_reward
        self.terminate_on_parse_error = terminate_on_parse_error
        self.max_trajectory_tokens = max_trajectory_tokens
        self.tool_specs = tool_specs
        # Zero-valued versions of whatever per-episode metric keys a
        # successful step() would report (e.g. {"format": 0.0, "correct":
        # 0.0}) -- without these, `dict_mean` (which averages each metric key
        # only over the episodes that reported it) silently drops parse
        # failures from the format/correct denominator instead of counting
        # them as a miss, inflating those aggregates relative to any
        # message_env implementation that reports them unconditionally
        # (e.g. a recipe with no extra metadata). Defaults to {} so existing
        # callers that don't pass this keep their current behavior.
        self.parse_failure_metrics = dict(parse_failure_metrics or {})
        self._base_stop_condition = renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        messages = await self.message_env.initial_observation()
        return self.renderer.build_generation_prompt(
            messages
        ), self._base_stop_condition

    async def step(self, action: types.Action) -> types.StepResult:
        """Parse tokens to a message, delegate to MessageEnv, and render response."""
        if self.tool_specs is None:
            assistant_message, parse_success = self.renderer.parse_response(action)
        else:
            assistant_message, parse_success = self.renderer.parse_response_with_tools(
                action, self.tool_specs
            )

        if not parse_success:
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=ModelInput(chunks=[]),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0, **self.parse_failure_metrics},
            )

        msg_step = await self.message_env.step(assistant_message)
        next_observation = self.renderer.build_generation_prompt(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        # Check if trajectory exceeds max token limit
        if self.max_trajectory_tokens is not None:
            token_count = _model_input_token_count(next_observation)
            if token_count > self.max_trajectory_tokens:
                return types.StepResult(
                    reward=0.0,
                    episode_done=True,
                    next_observation=ModelInput(chunks=[]),
                    next_stop_condition=self._base_stop_condition,
                    metrics={**msg_step.metrics, "context_overflow": 1.0},
                )

        return types.StepResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
        )

    async def close(self) -> None:
        await self.message_env.close()
