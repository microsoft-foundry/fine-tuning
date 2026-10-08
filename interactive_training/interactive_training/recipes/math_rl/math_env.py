import math
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from azure.ai.finetuningsessions.models import ModelInput
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from interactive_training import renderers
from interactive_training.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from interactive_training.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from interactive_training.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.utils import logtree


# Default user-turn prompt for s1-style budget forcing (Muennighoff et al.,
# 2025). When the model exhausts its turn-1 token budget without emitting a
# post-think boxed answer, we open a short turn 2 with this instruction so it
# commits to whatever its best guess is rather than returning nothing.
_DEFAULT_FORCED_COMMIT_PROMPT = (
    "You hit your reasoning budget without giving a final answer. Based on"
    " what you have so far, state only your final answer in \\boxed{} format"
    " on a single line. Do not restart your reasoning."
)

# When forced_commit_prefill_boxed is enabled, this string is prefilled into
# the turn-2 generation so the model cannot deliberate about formatting and
# simply completes the box. The leading ``</think>\n\n`` closes the thinking
# block that Qwen3-family chat templates auto-open after the assistant
# header; non-thinking renderers will simply emit it as literal text, which
# the grader tolerates. If you use a renderer with a different close-thinking
# marker, override forced_commit_prefill via MathEnv kwargs.
_BOXED_PREFILL = "</think>\n\n\\boxed{"
# Prepended to the model's continuation when grading so the parsed text looks
# like ``\boxed{...}`` even though the model only emitted the inside.
_BOXED_PREFILL_GRADE_PREFIX = "\\boxed{"


def _get_thinking_content(message: renderers.Message) -> str:
    """Concatenate all ThinkingPart text in an assistant message.

    Returns an empty string for messages with plain string content or no
    thinking parts. Used to show turn-1 reasoning back to the model on the
    forced-commit turn (the renderer would otherwise strip it from history).
    """
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        p.get("thinking", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "thinking"
    )


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        forced_commit: bool = False,
        forced_commit_penalty: float = 0.0,
        forced_commit_prompt: str | None = None,
        forced_commit_max_tokens: int = 1024,
        forced_commit_prefill_boxed: bool = False,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout
        self.forced_commit = forced_commit
        self.forced_commit_penalty = forced_commit_penalty
        self.forced_commit_prompt = forced_commit_prompt or _DEFAULT_FORCED_COMMIT_PROMPT
        self.forced_commit_max_tokens = forced_commit_max_tokens
        self.forced_commit_prefill_boxed = forced_commit_prefill_boxed
        # Per-episode state. Envs are constructed fresh per group rollout, so
        # using instance attrs here is safe (no cross-rollout contamination).
        self._used_forced_commit = False

    @classmethod
    def question_suffix(cls) -> str:
        return " Write your answer in \\boxed{} format."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    async def step(self, action: Action) -> StepResult:
        # Budget-forcing-aware override of ProblemEnv.step. With
        # forced_commit=False, behavior matches the base implementation
        # exactly (modulo the explicit StepResult construction).
        message, parse_success = self.renderer.parse_response(action)
        text_after = renderers.get_text_content(message)

        # If turn 2 was opened with a ``\boxed{`` prefill, the model only
        # generated the inside of the box. Reconstruct the full ``\boxed{...}``
        # for format/grader checks.
        if self._used_forced_commit and self.forced_commit_prefill_boxed:
            text_after = _BOXED_PREFILL_GRADE_PREFIX + text_after

        has_answer = parse_success and self.check_format(text_after)
        is_turn1 = not self._used_forced_commit

        # Turn-1 forced-commit branch: when enabled and the model produced
        # no boxed answer in its post-think text, open a short second turn
        # with an explicit instruction to commit. Reward is deferred to
        # turn 2.
        if self.forced_commit and is_turn1 and not has_answer:
            self._used_forced_commit = True
            think_text = _get_thinking_content(message)
            # Concatenate thinking + post-think text into a single visible
            # text part. Storing as ``text`` (not ``thinking``) survives the
            # Qwen3-family renderer's strip-thinking-from-history behavior,
            # so the model can see what it had been reasoning about when we
            # ask it to commit on turn 2.
            turn1_visible = think_text
            if text_after:
                turn1_visible = f"{turn1_visible}\n\n{text_after}" if turn1_visible else text_after
            assistant_msg: renderers.Message = {
                "role": "assistant",
                "content": [{"type": "text", "text": turn1_visible}],
            }
            convo: list[renderers.Message] = [
                *self.convo_prefix,
                {"role": "user", "content": self.get_question()},
                assistant_msg,
                {"role": "user", "content": self.forced_commit_prompt},
            ]
            prefill = _BOXED_PREFILL if self.forced_commit_prefill_boxed else None
            next_obs = self.renderer.build_generation_prompt(convo, prefill=prefill)
            logtree.log_text(
                f"[forced_commit] turn1 tokens={len(action)}; opening forced commit turn"
            )
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=next_obs,
                next_stop_condition=self.stop_condition,
                metrics={"forced_commit_opened": 1.0},
                next_max_tokens=self.forced_commit_max_tokens,
            )

        correct_format = 1.0 if has_answer else 0.0
        correct_answer = float(self.check_answer(text_after)) if has_answer else 0.0
        answer_from_forced = 1.0 if (self._used_forced_commit and has_answer) else 0.0
        forced_commit_penalty_value = (
            self.forced_commit_penalty if answer_from_forced else 0.0
        )

        total_reward = (
            self.format_coef * (correct_format - 1)
            + correct_answer
            - forced_commit_penalty_value
        )

        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'Y' if correct_format else 'N'}, "
            f"Correct: {'Y' if correct_answer else 'N'}, "
            f"ForcedCommit: {'Y' if self._used_forced_commit else 'N'}, "
            f"Reward: {total_reward:.3f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=ModelInput(chunks=[]),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
                "answer_from_forced": answer_from_forced,
                "forced_commit_used": 1.0 if self._used_forced_commit else 0.0,
                "forced_commit_penalty_value": forced_commit_penalty_value,
            },
        )

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset


class MathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        max_examples: int | None = None,
        forced_commit: bool = False,
        forced_commit_penalty: float = 0.0,
        forced_commit_prompt: str | None = None,
        forced_commit_max_tokens: int = 1024,
        forced_commit_prefill_boxed: bool = False,
    ):
        if split == "train":
            self.ds = _get_hendrycks_math_train().shuffle(seed=seed)
        elif split == "test":
            self.ds = _get_hendrycks_math_test()
        if max_examples is not None:
            self.ds = self.ds.select(range(min(max_examples, len(self.ds))))
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.forced_commit = forced_commit
        self.forced_commit_penalty = forced_commit_penalty
        self.forced_commit_prompt = forced_commit_prompt
        self.forced_commit_max_tokens = forced_commit_max_tokens
        self.forced_commit_prefill_boxed = forced_commit_prefill_boxed

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                x["problem"],
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                forced_commit=self.forced_commit,
                forced_commit_penalty=self.forced_commit_penalty,
                forced_commit_prompt=self.forced_commit_prompt,
                forced_commit_max_tokens=self.forced_commit_max_tokens,
                forced_commit_prefill_boxed=self.forced_commit_prefill_boxed,
            ),
            num_envs=group_size,
        )


@chz.chz
class MathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    # See MathEnv for budget-forcing semantics. forced_commit=False is a no-op.
    forced_commit: bool = False
    forced_commit_penalty: float = 0.0
    forced_commit_prompt: str | None = None
    forced_commit_max_tokens: int = 1024
    forced_commit_prefill_boxed: bool = False

    async def __call__(self) -> tuple[MathDataset, MathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        max_examples_map = {"train": self.max_train_examples, "test": self.max_test_examples}
        datasets = [
            MathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
                max_examples=max_examples_map[split],
                forced_commit=self.forced_commit,
                forced_commit_penalty=self.forced_commit_penalty,
                forced_commit_prompt=self.forced_commit_prompt,
                forced_commit_max_tokens=self.forced_commit_max_tokens,
                forced_commit_prefill_boxed=self.forced_commit_prefill_boxed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


class PolarisDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        max_examples: int | None = None,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(
            seed=seed
        )
        if max_examples is not None:
            self.ds = self.ds.select(range(min(max_examples, len(self.ds))))
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="polaris",
        )


@chz.chz
class PolarisDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    max_train_examples: int | None = None
    max_test_examples: int | None = None

    async def __call__(self) -> tuple[PolarisDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return PolarisDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
            max_examples=self.max_train_examples,
        ), None


class DeepMathDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        max_examples: int | None = None,
        forced_commit: bool = False,
        forced_commit_penalty: float = 0.0,
        forced_commit_prompt: str | None = None,
        forced_commit_max_tokens: int = 1024,
        forced_commit_prefill_boxed: bool = False,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
        if max_examples is not None:
            self.ds = self.ds.select(range(min(max_examples, len(self.ds))))
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.forced_commit = forced_commit
        self.forced_commit_penalty = forced_commit_penalty
        self.forced_commit_prompt = forced_commit_prompt
        self.forced_commit_max_tokens = forced_commit_max_tokens
        self.forced_commit_prefill_boxed = forced_commit_prefill_boxed

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                forced_commit=self.forced_commit,
                forced_commit_penalty=self.forced_commit_penalty,
                forced_commit_prompt=self.forced_commit_prompt,
                forced_commit_max_tokens=self.forced_commit_max_tokens,
                forced_commit_prefill_boxed=self.forced_commit_prefill_boxed,
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


@chz.chz
class DeepMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    # See MathEnv for budget-forcing semantics. forced_commit=False is a no-op.
    forced_commit: bool = False
    forced_commit_penalty: float = 0.0
    forced_commit_prompt: str | None = None
    forced_commit_max_tokens: int = 1024
    forced_commit_prefill_boxed: bool = False

    async def __call__(self) -> tuple[DeepMathDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return DeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
            max_examples=self.max_train_examples,
            forced_commit=self.forced_commit,
            forced_commit_penalty=self.forced_commit_penalty,
            forced_commit_prompt=self.forced_commit_prompt,
            forced_commit_max_tokens=self.forced_commit_max_tokens,
            forced_commit_prefill_boxed=self.forced_commit_prefill_boxed,
        ), None


class Gsm8kDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        max_examples: int | None = None,
        forced_commit: bool = False,
        forced_commit_penalty: float = 0.0,
        forced_commit_prompt: str | None = None,
        forced_commit_max_tokens: int = 1024,
        forced_commit_prefill_boxed: bool = False,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        if max_examples is not None:
            self.ds = self.ds.select(range(min(max_examples, len(self.ds))))
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.forced_commit = forced_commit
        self.forced_commit_penalty = forced_commit_penalty
        self.forced_commit_prompt = forced_commit_prompt
        self.forced_commit_max_tokens = forced_commit_max_tokens
        self.forced_commit_prefill_boxed = forced_commit_prefill_boxed

    @classmethod
    def question_suffix(cls) -> str:
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                forced_commit=self.forced_commit,
                forced_commit_penalty=self.forced_commit_penalty,
                forced_commit_prompt=self.forced_commit_prompt,
                forced_commit_max_tokens=self.forced_commit_max_tokens,
                forced_commit_prefill_boxed=self.forced_commit_prefill_boxed,
            ),
            num_envs=group_size,
        )


@chz.chz
class Gsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    # See MathEnv for budget-forcing semantics. forced_commit=False is a no-op.
    forced_commit: bool = False
    forced_commit_penalty: float = 0.0
    forced_commit_prompt: str | None = None
    forced_commit_max_tokens: int = 1024
    forced_commit_prefill_boxed: bool = False

    async def __call__(self) -> tuple[Gsm8kDataset, Gsm8kDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        max_examples_map = {"train": self.max_train_examples, "test": self.max_test_examples}
        datasets = [
            Gsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
                max_examples=max_examples_map[split],
                forced_commit=self.forced_commit,
                forced_commit_penalty=self.forced_commit_penalty,
                forced_commit_prompt=self.forced_commit_prompt,
                forced_commit_max_tokens=self.forced_commit_max_tokens,
                forced_commit_prefill_boxed=self.forced_commit_prefill_boxed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "math": MathDatasetBuilder,
    "polaris": PolarisDatasetBuilder,
    "deepmath": DeepMathDatasetBuilder,
    "gsm8k": Gsm8kDatasetBuilder,
}


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
    forced_commit: bool = False,
    forced_commit_penalty: float = 0.0,
    forced_commit_prompt: str | None = None,
    forced_commit_max_tokens: int = 1024,
    forced_commit_prefill_boxed: bool = False,
) -> RLDatasetBuilder:
    """
    Unified function to get any math dataset builder.
    Args:
        dataset_name: One of "math", "polaris", "deepmath", or "gsm8k"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
        max_train_examples: If set, truncate training dataset to this many examples
        max_test_examples: If set, truncate test dataset to this many examples
        forced_commit, forced_commit_penalty, forced_commit_prompt,
        forced_commit_max_tokens, forced_commit_prefill_boxed:
            s1-style budget-forcing knobs (Muennighoff et al., 2025). Wired
            through to ``math``, ``deepmath``, and ``gsm8k``; ignored for
            ``polaris``. forced_commit=False (default) is a no-op.
    Returns:
        The appropriate dataset builder instance
    """
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(
            f"Unknown math dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
        )

    builder_class = DATASET_BUILDER_MAP[dataset_name]

    kwargs: dict = dict(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )
    if builder_class in (MathDatasetBuilder, DeepMathDatasetBuilder, Gsm8kDatasetBuilder):
        kwargs.update(
            forced_commit=forced_commit,
            forced_commit_penalty=forced_commit_penalty,
            forced_commit_prompt=forced_commit_prompt,
            forced_commit_max_tokens=forced_commit_max_tokens,
            forced_commit_prefill_boxed=forced_commit_prefill_boxed,
        )
    return builder_class(**kwargs)
