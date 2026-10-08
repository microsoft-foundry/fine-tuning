"""Single-turn function-calling environment for xLAM.

The xLAM dataset has no live tools to execute — we only have ground-truth tool
calls to grade against. This recipe uses a single-turn environment with no tool
dispatch:

  1. observation = system_prompt_with_tool_schemas + user_query
  2. model emits one assistant message (which should contain tool-call blocks)
  3. Response is parsed and graded against gold answers via BFCL AST matching
  4. episode ends

The environment extends ``ProblemEnv`` and overrides ``step()`` to handle
tool-call-specific parsing and grading rather than plain-text answer checking.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Sequence

import chz
from azure.ai.finetuningsessions.models import ModelInput

from interactive_training import model_info, renderers
from interactive_training.recipes.tool_rl.data import XLAMTask, load_xlam_tasks
from interactive_training.recipes.tool_rl.grading import compute_call_match_score
from interactive_training.renderers import ToolCall, get_renderer
from interactive_training.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from interactive_training.rl.types import Action, RLDataset, RLDatasetBuilder, StepResult
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.utils import logtree

logger = logging.getLogger(__name__)


# System prompt taken from BFCL's chat-model evaluation prompt
# (https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
# section "When to function-call (tool-call) and when to prompt?"). xLAM was trained
# and evaluated against BFCL conventions, so this matches the canonical setup.
TOOL_CALL_SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set "
    "of possible functions. Based on the question, you will need to make one or "
    "more function/tool calls to achieve the purpose. If none of the functions "
    "can be used, point it out. If the given question lacks the parameters "
    "required by the function, also point it out."
)


class ToolCallEnv(ProblemEnv):
    """A ProblemEnv that grades tool-call output against xLAM gold answers.

    Overrides ``step()`` to parse the model's response for structured tool calls
    (via the renderer) and grade them using BFCL-style bipartite AST matching,
    instead of the plain-text answer checking that ProblemEnv normally does.

    Episode lifecycle:
        initial_observation() -> [system_with_tools, user_query]  (inherited)
        step(action) -> reward via BFCL matching, episode_done=True
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        task: XLAMTask,
        format_coef: float = 0.1,
    ):
        # Build tool-augmented system prompt as the conversation prefix.
        try:
            initial_prefix = renderer.create_conversation_prefix_with_tools(
                tools=list(task.tool_specs),
                system_prompt=TOOL_CALL_SYSTEM_PROMPT,
            )
        except NotImplementedError as e:
            raise RuntimeError(
                f"Renderer does not support tool calling "
                f"(create_conversation_prefix_with_tools raised NotImplementedError). "
                f"Use a tool-aware renderer (qwen3, gpt_oss, deepseek_v3)."
            ) from e

        super().__init__(renderer, convo_prefix=initial_prefix, format_coef=format_coef)
        self.task = task

    def get_question(self) -> str:
        return self.task.query

    def check_answer(self, sample_str: str) -> bool:
        # Not used — step() is overridden with tool-call-specific grading.
        return False

    def check_format(self, sample_str: str) -> bool:
        # Not used — step() is overridden with tool-call-specific grading.
        return False

    def get_reference_answer(self) -> str:
        return json.dumps(
            [{"name": a.name, "arguments": a.arguments} for a in self.task.gold_answers],
            ensure_ascii=False,
        )

    async def step(self, action: Action) -> StepResult:
        message, _parse_success = self.renderer.parse_response_with_tools(
            action, list(self.task.tool_specs)
        )

        pred_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        unparsed = list(message.get("unparsed_tool_calls") or [])

        format_score = 1.0 if pred_calls and not unparsed else 0.0
        correctness, bfcl_strict = compute_call_match_score(
            pred_calls, list(self.task.gold_answers)
        )

        reward = self.format_coef * (format_score - 1.0) + correctness

        # Trajectory log for inspection.
        logtree.log_text(f"Query: {self.task.query}")
        logtree.log_text(
            "Gold ({} calls): {}".format(
                len(self.task.gold_answers),
                json.dumps(
                    [{"name": a.name, "arguments": a.arguments} for a in self.task.gold_answers],
                    ensure_ascii=False,
                ),
            )
        )
        logtree.log_text(
            "Pred ({} calls): {}".format(
                len(pred_calls),
                json.dumps(
                    [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in pred_calls
                    ],
                    ensure_ascii=False,
                ),
            )
        )
        if unparsed:
            logtree.log_text(f"Unparsed tool-call blocks: {len(unparsed)}")
        logtree.log_text(
            f"Format: {format_score} | Correct: {correctness:.3f} | "
            f"BFCL-strict: {int(bfcl_strict)} | Reward: {reward:.3f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=ModelInput(chunks=[]),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": format_score,
                "correct": correctness,
                "bfcl_strict": float(bfcl_strict),
                "n_pred": float(len(pred_calls)),
                "n_gold": float(len(self.task.gold_answers)),
            },
        )


class ToolCallDataset(RLDataset):
    """Dataset wrapping ProblemGroupBuilders for xLAM tasks into batches."""

    def __init__(
        self,
        env_group_builders: list[ProblemGroupBuilder],
        batch_size: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[ProblemGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        # Ceil division so a partial final batch still runs. Floor division would
        # drop it, and with max_train_examples < batch_size the whole dataset
        # would round to 0 batches and the run would exit without training.
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class XLAMDatasetBuilder(RLDatasetBuilder):
    """Build train + held-out-eval RLDatasets from xLAM-function-calling-60k."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    format_coef: float = 0.1
    seed: int = 0
    max_test_examples: int = 1000
    max_train_examples: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_tasks, test_tasks = load_xlam_tasks(seed=self.seed, test_size=self.max_test_examples)
        if self.max_train_examples is not None and self.max_train_examples >= 0:
            train_tasks = train_tasks[: self.max_train_examples]

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        renderer = get_renderer(renderer_name, tokenizer)

        train_builders = [
            ProblemGroupBuilder(
                env_thunk=partial(
                    ToolCallEnv,
                    renderer=renderer,
                    task=task,
                    format_coef=self.format_coef,
                ),
                num_envs=self.group_size,
                dataset_name="xlam_fc",
            )
            for task in train_tasks
        ]
        train_dataset = ToolCallDataset(
            env_group_builders=train_builders,
            batch_size=self.batch_size,
        )

        test_dataset: RLDataset | None = None
        if test_tasks:
            test_builders = [
                ProblemGroupBuilder(
                    env_thunk=partial(
                        ToolCallEnv,
                        renderer=renderer,
                        task=task,
                        format_coef=self.format_coef,
                    ),
                    num_envs=1,  # single sample per task for eval
                    dataset_name="xlam_fc",
                )
                for task in test_tasks
            ]
            test_dataset = ToolCallDataset(
                env_group_builders=test_builders,
                batch_size=self.batch_size,
            )

        return train_dataset, test_dataset
