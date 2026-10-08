"""Tool-use RL environment for an image-returning alien-color MCP server."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chz

from interactive_training import model_info, tokenizer_utils
from interactive_training.image_processing_utils import get_image_processor
from interactive_training.recipes.mcp_image_tool.data import (
    ALIEN_WORD_TO_COLOR,
    AlienColorTask,
    build_curriculum_tasks,
)
from interactive_training.renderers import get_renderer, get_text_content
from interactive_training.renderers.base import Message, ToolSpec
from interactive_training.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from interactive_training.tool_use import build_agent_tool_env
from interactive_training.tool_use.types import Tool, ToolInput, ToolResult

SYSTEM_PROMPT = """You decode an alien color language using the available image tool.

Use the alien-word mapping you know to choose tool calls, then inspect the returned images.
If a choice is wrong, you may try another word within the call budget. Correct lookups and
verified answers are diagnostic only; reward is binary. Reward requires exactly
one successful call for each requested color and the correct final answer. Any wrong, extra,
or repeated call earns zero. Stop once all requested colors are verified. Do not skip
verification. Follow the user's exact `Answer:` format.
"""


@dataclass(frozen=True)
class _AlienWordOrderedTool:
    tool: Tool
    alien_words: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        schema = deepcopy(self.tool.parameters_schema)
        schema["properties"]["alien_word"]["enum"] = list(self.alien_words)
        return schema

    async def run(self, input: ToolInput) -> ToolResult:
        return await self.tool.run(input)

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


def _alien_word_orders(
    *, task_id: int, group_size: int, seed: int
) -> list[tuple[str, ...]]:
    words = list(ALIEN_WORD_TO_COLOR)
    random.Random(seed * 1_000_003 + task_id).shuffle(words)
    order = tuple(words)
    return [order] * group_size


def _task_tool_call_limit(task: AlienColorTask, max_tool_calls: int) -> int:
    return min(max_tool_calls, len(task.colors) + 1)


@dataclass(frozen=True)
class AlienColorAnswerReward:
    expected: str | tuple[tuple[str, str], ...]
    max_tool_calls: int
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final_text = ""
        for message in reversed(history):
            if message.get("role") == "assistant":
                if not message.get("tool_calls"):
                    final_text = get_text_content(message)
                break

        executed_words = self._executed_words(history)
        proposed_calls = sum(
            len(message.get("tool_calls") or [])
            for message in history
            if message.get("role") == "assistant"
        )
        answer = self._extract_answer(final_text)
        expected_pairs = self._expected_pairs()
        expected_words = tuple(word for _, word in expected_pairs)
        expected_word_set = set(expected_words)
        if isinstance(self.expected, str):
            parsed_answers = (
                (("", answer),) if answer is not None else None
            )
        else:
            parsed_answers = self._parse_mapped_answers(answer)
            if parsed_answers is not None:
                answer_colors = tuple(self._normalize(color) for color, _ in parsed_answers)
                expected_colors = tuple(self._normalize(color) for color, _ in expected_pairs)
                if answer_colors != tuple(color for color in expected_colors if color in answer_colors):
                    parsed_answers = None
        format_score = float(parsed_answers is not None)
        normalized_answers = {
            self._normalize(color): self._normalize(word)
            for color, word in (parsed_answers or ())
        }

        found_words: set[str] = set()
        wrong_calls = 0
        post_discovery_calls = 0
        for word in executed_words:
            if len(found_words) == len(expected_word_set):
                post_discovery_calls += 1
            elif word in expected_word_set and word not in found_words:
                found_words.add(word)
            else:
                wrong_calls += 1

        color_found = len(found_words) / len(expected_pairs)
        verified_count = sum(
            word in found_words
            and normalized_answers.get(self._normalize(color))
            == self._normalize(word)
            for color, word in expected_pairs
        )
        verified_fraction = verified_count / len(expected_pairs)
        verified_answer = float(verified_count == len(expected_pairs))
        exact_expected_calls = (
            len(executed_words) == len(expected_words)
            and proposed_calls == len(executed_words)
            and set(executed_words) == expected_word_set
        )
        correct = float(
            verified_answer and exact_expected_calls
        )
        wasted_calls = wrong_calls + post_discovery_calls
        attempted_calls = max(proposed_calls, len(executed_words))
        efficiency = (
            len(found_words) / attempted_calls
            if attempted_calls
            else 0.0
        )
        reward = correct + self.format_coef * (format_score - 1.0)
        return reward, {
            "format": format_score,
            "correct": correct,
            "verified_answer": verified_answer,
            "verified_fraction": verified_fraction,
            "color_found": color_found,
            "efficiency": efficiency,
            "tool_calls": float(len(executed_words)),
            "mistakes": float(wasted_calls),
            "wrong_calls": float(wrong_calls),
            "post_discovery_calls": float(post_discovery_calls),
        }

    def _expected_pairs(self) -> tuple[tuple[str, str], ...]:
        if isinstance(self.expected, str):
            return (("", self.expected),)
        return self.expected

    @staticmethod
    def _executed_words(history: list[Message]) -> list[str | None]:
        pending_calls: list[tuple[str | None, str | None]] = []
        executed_words: list[str | None] = []
        for message in history:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    alien_word = None
                    if tool_call.function.name == "reveal_alien_color":
                        try:
                            arguments = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        value = arguments.get("alien_word")
                        alien_word = value if isinstance(value, str) else None
                    pending_calls.append((tool_call.id, alien_word))
                continue
            if message.get("role") != "tool" or not pending_calls:
                continue

            result_id = message.get("tool_call_id") or None
            match_index = next(
                (
                    index
                    for index, (call_id, _) in enumerate(pending_calls)
                    if call_id == result_id
                ),
                0 if result_id is None else -1,
            )
            if match_index < 0:
                continue
            _, alien_word = pending_calls.pop(match_index)
            content = message.get("content")
            has_oracle_image = (
                message.get("name") == "reveal_alien_color"
                and isinstance(content, list)
                and any(part["type"] in {"image", "image_url"} for part in content)
            )
            executed_words.append(alien_word if has_oracle_image else None)
        return executed_words

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        if not text.strip().startswith("Answer:") or text.count("Answer:") != 1:
            return None
        answer = text.partition("Answer:")[2].strip()
        return answer or None

    @classmethod
    def _parse_mapped_answers(
        cls, answer: str | None
    ) -> tuple[tuple[str, str], ...] | None:
        if answer is None:
            return None
        pairs: list[tuple[str, str]] = []
        seen_colors: set[str] = set()
        for field in answer.split(","):
            if field.count("=") != 1:
                return None
            color, word = (part.strip() for part in field.split("=", 1))
            normalized_color = cls._normalize(color)
            if not normalized_color or not word or normalized_color in seen_colors:
                return None
            seen_colors.add(normalized_color)
            pairs.append((color, word))
        return tuple(pairs) or None

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.replace(",", "").split()).casefold()


@chz.chz
class MCPImageToolEnvGroupBuilder(EnvGroupBuilder):
    task: AlienColorTask
    model_name: str
    tokenizer_name: str
    renderer_name: str | None
    group_size: int
    tools: list[Tool]
    max_tool_calls: int = 20
    format_coef: float = 0.1
    max_trajectory_tokens: int = 8 * 1024
    seed: int = 0

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.tokenizer_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(
            renderer_name,
            tokenizer,
            image_processor=get_image_processor(self.model_name),
        )
        reward_fn = AlienColorAnswerReward(
            expected=(
                self.task.answers[0]
                if len(self.task.answers) == 1
                else tuple(zip(self.task.colors, self.task.answers))
            ),
            max_tool_calls=self.max_tool_calls,
            format_coef=self.format_coef,
        )
        tool_call_limit = _task_tool_call_limit(self.task, self.max_tool_calls)
        envs = []
        for alien_word_order in _alien_word_orders(
            task_id=self.task.task_id,
            group_size=self.group_size,
            seed=self.seed,
        ):
            ordered_tools = [
                _AlienWordOrderedTool(tool, alien_word_order)
                if tool.name == "reveal_alien_color"
                else tool
                for tool in self.tools
            ]
            tool_specs = [tool.to_spec() for tool in ordered_tools]
            initial_messages = renderer.create_conversation_prefix_with_tools(
                tools=tool_specs,
                system_prompt=SYSTEM_PROMPT + f"\nYour total tool-call budget is {tool_call_limit}.",
            ) + [{"role": "user", "content": self.task.question}]
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=ordered_tools,
                    initial_messages=initial_messages,
                    reward_fn=reward_fn,
                    max_turns=tool_call_limit + 1,
                    max_tool_calls=tool_call_limit,
                    failed_parse_reward=0.0,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    parse_failure_metrics={
                        "format": 0.0,
                        "correct": 0.0,
                        "verified_answer": 0.0,
                        "verified_fraction": 0.0,
                    },
                )
            )
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        for trajectory, env in zip(trajectory_group, env_group, strict=True):
            if not trajectory.transitions:
                continue
            history = env.message_env.history
            final_message = next(
                (message for message in reversed(history) if message.get("role") == "assistant"),
                {},
            )
            executed_words = AlienColorAnswerReward._executed_words(history)
            proposed_calls = [
                {"name": call.function.name, "arguments": call.function.arguments}
                for message in history if message.get("role") == "assistant"
                for call in (message.get("tool_calls") or [])
            ]
            trajectory.transitions[-1].logs["mcp_diagnostics"] = json.dumps({
                "executed_words": executed_words,
                "executed_call_count": len(executed_words),
                "proposed_calls": proposed_calls,
                "proposed_call_count": len(proposed_calls),
                "final_answer": get_text_content(final_message) if final_message else "",
                "final_answer_present": bool(final_message) and not bool(final_message.get("tool_calls")),
                "schema_word_order": _alien_word_orders(
                    task_id=self.task.task_id, group_size=1, seed=self.seed,
                )[0],
            })
        return await super().compute_group_rewards(trajectory_group, env_group)

    def logging_tags(self) -> list[str]:
        return ["mcp_alien_color", f"colors_{len(self.task.colors)}", *self.task.colors]


class MCPImageToolDataset(RLDataset):
    def __init__(self, builders: list[MCPImageToolEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class MCPImageToolDatasetBuilder(RLDatasetBuilder):
    model_name: str
    tokenizer_name: str
    tools: list[Tool]
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_tool_calls: int = 20
    format_coef: float = 0.1
    max_trajectory_tokens: int = 8 * 1024
    seed: int = 0
    train_examples: int = 48
    test_examples: int = 12

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_tasks, test_tasks = build_curriculum_tasks(
            self.train_examples,
            self.test_examples,
        )

        def make_builder(
            task: AlienColorTask, group_size: int
        ) -> MCPImageToolEnvGroupBuilder:
            return MCPImageToolEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                renderer_name=self.renderer_name,
                group_size=group_size,
                tools=self.tools,
                max_tool_calls=self.max_tool_calls,
                format_coef=self.format_coef,
                max_trajectory_tokens=self.max_trajectory_tokens,
                seed=self.seed,
            )

        train = MCPImageToolDataset(
            [make_builder(task, self.group_size) for task in train_tasks],
            self.batch_size,
        )
        test = MCPImageToolDataset(
            [make_builder(task, 1) for task in test_tasks],
            self.batch_size,
        )
        return train, test
