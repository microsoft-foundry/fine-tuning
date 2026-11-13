"""Document-counting tasks for sampler/trainer consistency checks."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import chz
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk
from pydantic import Field

from programmed_training import model_info
from programmed_training.renderers import get_renderer
from programmed_training.renderers.base import Message, Renderer, ToolCall, get_text_content
from programmed_training.rl.message_env import EnvFromMessageEnv
from programmed_training.rl.types import Action, Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from programmed_training.tokenizer_utils import get_tokenizer
from programmed_training.tool_use import (
    AgentToolMessageEnv,
    ToolResult,
    error_tool_result,
    simple_tool_result,
    tool,
)

logger = logging.getLogger(__name__)
DEFAULT_WORDS = (
    "after", "against", "although", "before", "being", "between", "both",
    "could", "during", "however", "including", "into", "known", "later",
    "many", "most", "only", "over", "several", "such", "their", "then",
    "through", "time", "under", "until", "when", "while",
)
WORD_RE = re.compile(r"[a-z]+")
ANSWER_RE = re.compile(r"answer\s*:\s*\**\s*(-?\d[\d,]*)", re.IGNORECASE)


@chz.chz
class CountingTaskConfig:
    num_pages_min: int = 7
    num_pages_max: int = 9
    page_chars: int = 108_800
    target_context_tokens: int | None = None
    words: tuple[str, ...] = DEFAULT_WORDS
    num_pages_hint: Literal["exact", "range"] = "range"
    word_hint: Literal["exact", "first_last", "hidden"] = "exact"
    secret_num_pages: int | None = None
    reward_tolerance: float = 0.2

    @chz.validate
    def _validate(self) -> None:
        if not 1 <= self.num_pages_min <= self.num_pages_max or self.page_chars < 1:
            raise ValueError("Require 1 <= num_pages_min <= num_pages_max and page_chars > 0")
        if self.target_context_tokens is not None and (
            type(self.target_context_tokens) is not int or self.target_context_tokens < 1
        ):
            raise ValueError("target_context_tokens must be a positive integer when set")
        if not math.isfinite(self.reward_tolerance) or self.reward_tolerance <= 0:
            raise ValueError("reward_tolerance must be finite and positive")
        if not self.words or len(set(self.words)) != len(self.words):
            raise ValueError("words must be nonempty and unique")
        if any(WORD_RE.fullmatch(word) is None for word in self.words):
            raise ValueError("words must contain only lowercase ASCII letters")
        if self.num_pages_hint not in ("exact", "range"):
            raise ValueError("num_pages_hint must be exact or range")
        if self.word_hint not in ("exact", "first_last", "hidden"):
            raise ValueError("word_hint must be exact, first_last, or hidden")
        if self.word_hint == "hidden" and len(self.words) != 1:
            raise ValueError("word_hint=hidden requires exactly one target word")
        if self.word_hint == "first_last":
            hints = {(word[0], word[-1]) for word in self.words}
            if len(hints) != len(self.words):
                raise ValueError("first_last hints must identify a unique target word")
        if self.secret_num_pages is not None and not (
            self.num_pages_min <= self.secret_num_pages <= self.num_pages_max
        ):
            raise ValueError("secret_num_pages must be within the configured page range")

    @property
    def max_turns(self) -> int:
        return self.num_pages_max + 4

    @property
    def hidden_num_pages(self) -> int:
        return (
            self.secret_num_pages if self.secret_num_pages is not None
            else random.Random(0).randint(self.num_pages_min, self.num_pages_max)
        )


def count_word(text: str, word: str) -> int:
    return WORD_RE.findall(text.lower()).count(word)


def parse_answer(text: str) -> int | None:
    matches = ANSWER_RE.findall(text)
    return int(matches[-1].replace(",", "")) if matches else None


@lru_cache(maxsize=2)
def load_corpus(corpus_path: str | None = None) -> str:
    if corpus_path is not None:
        return Path(corpus_path).read_text(encoding="utf-8")
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    lines: list[str] = []
    size = 0
    for row in dataset:
        text = row["text"]
        if not isinstance(text, str):
            raise TypeError("WikiText text must be a string")
        for old, new in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")):
            text = text.replace(old, new)
        text = text.strip()
        if text:
            lines.append(text)
            size += len(text) + 1
        if size >= 100_000_000:
            break
    corpus = "\n".join(lines)[:100_000_000]
    logger.info("Loaded %s WikiText corpus characters", len(corpus))
    return corpus


@dataclass(frozen=True)
class CountingProblem:
    word: str
    num_pages: int
    pages: tuple[str, ...]
    prompt: str
    true_count: int
    calibrated_context_tokens: int | None = None


def sample_problem(
    corpus: str, task: CountingTaskConfig, seed: int, *, start: int | None = None,
) -> CountingProblem:
    rng = random.Random(seed)
    word = rng.choice(task.words)
    n = (
        rng.randint(task.num_pages_min, task.num_pages_max)
        if task.num_pages_hint == "exact" else task.hidden_num_pages
    )
    span = math.ceil(task.max_turns * task.page_chars * 1.1)
    if len(corpus) <= span:
        raise ValueError(f"Corpus has {len(corpus)} characters; need more than {span}")
    if start is None:
        start = rng.randrange(len(corpus) - span)
    elif not 0 <= start < len(corpus) - span:
        raise ValueError("Corpus cannot supply the requested pages at the fixed offset")
    while start < len(corpus) and not corpus[start].isspace():
        start += 1
    pages: list[str] = []
    for _ in range(task.max_turns):
        end = min(start + task.page_chars, len(corpus))
        while end < len(corpus) and not corpus[end].isspace():
            end += 1
        page = corpus[start:end].strip()
        if not page:
            raise ValueError("Corpus cannot supply enough nonempty, word-aligned pages")
        pages.append(page)
        start = end
    target = {
        "exact": f'the word "{word}"',
        "first_last": f'a secret word starting with "{word[0]}" and ending with "{word[-1]}"',
        "hidden": "a secret word",
    }[task.word_hint]
    length = (
        f"The document consists of pages 1 through {n}."
        if task.num_pages_hint == "exact" else
        f"The document has between {task.num_pages_min} and {task.num_pages_max} pages. "
        "Its exact length is hidden. Pages beyond its end return text, but do not count."
    )
    prompt = (
        f"Count occurrences of {target} in this document. {length}\n"
        "Use read_page(page) with page numbers starting at 1. Make only one tool call "
        "in each response; extra calls will be rejected. Match case-insensitively. "
        "Each maximal run of ASCII letters is one word; punctuation separates words. "
        "After reading, finish without a tool call and end with Answer: <count>."
    )
    return CountingProblem(
        word, n, tuple(pages), prompt, sum(count_word(page, word) for page in pages[:n])
    )


@dataclass
class EpisodeState:
    pages_read: set[int] = field(default_factory=set)
    invalid_page_requests: int = 0
    rejected_tool_calls: int = 0


class PageReader:
    def __init__(self, problem: CountingProblem, state: EpisodeState):
        self.problem = problem
        self.state = state

    @tool
    async def read_page(self, page: Annotated[int, Field(strict=True)]) -> ToolResult:
        """Return the requested page. Page numbers start at 1."""
        if not 1 <= page <= len(self.problem.pages):
            self.state.invalid_page_requests += 1
            return error_tool_result(f"Page {page} is unavailable.", name="read_page")
        self.state.pages_read.add(page)
        return simple_tool_result(
            f"[Page {page}]\n{self.problem.pages[page - 1]}", name="read_page"
        )


@dataclass
class CountingReward:
    problem: CountingProblem
    state: EpisodeState
    tolerance: float

    def metrics(self, predicted: int | None) -> dict[str, float]:
        pages = self.state.pages_read
        return {
            "answer_parsed": float(predicted is not None),
            "exact_match": float(predicted == self.problem.true_count),
            "true_count": float(self.problem.true_count),
            "num_pages": float(self.problem.num_pages),
            "pages_read": float(len(pages)),
            "read_all_pages": float(set(range(1, self.problem.num_pages + 1)) <= pages),
            "read_past_end": float(any(page > self.problem.num_pages for page in pages)),
            "rejected_tool_calls": float(self.state.rejected_tool_calls),
            "invalid_page_requests": float(self.state.invalid_page_requests),
        }

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final = next((msg for msg in reversed(history) if msg["role"] == "assistant"), None)
        predicted = (
            parse_answer(get_text_content(final))
            if final is not None and not final.get("tool_calls") else None
        )
        metrics = self.metrics(predicted)
        if predicted is None:
            return 0.0, metrics
        error = abs(predicted - self.problem.true_count)
        metrics["abs_error"] = float(error)
        return max(0.0, 1 - error / (self.tolerance * max(self.problem.true_count, 1))), metrics


@dataclass
class CountingMessageEnv(AgentToolMessageEnv):
    state: EpisodeState = field(default_factory=EpisodeState)

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        messages = await super()._handle_tool_calls(tool_calls[:1])
        for call in tool_calls[1:]:
            self.state.rejected_tool_calls += 1
            rejected = error_tool_result(
                "Only the first tool call in a response is executed.",
                call_id=call.id or "", name=call.function.name, error_type="rejected",
            )
            self.history.extend(rejected.messages)
            messages.extend(rejected.messages)
        return messages


def text_tokens(observation: ModelInput) -> list[int]:
    tokens: list[int] = []
    for chunk in observation.chunks:
        if not isinstance(chunk, ModelInputChunk):
            raise TypeError("The counting recipe requires text-only model inputs")
        tokens.extend(chunk.tokens)
    return tokens


def _initial_messages(
    problem: CountingProblem, renderer: Renderer, reader: PageReader,
) -> list[Message]:
    return renderer.create_conversation_prefix_with_tools(
        tools=[reader.read_page.to_spec()],
        system_prompt="Read the document using tools and count carefully.",
    ) + [Message(role="user", content=problem.prompt)]


def _context_tolerance(target: int) -> int:
    return min(1024, max(1, target // 100))


def _completed_context_tokens(problem: CountingProblem, renderer: Renderer) -> int:
    reader = PageReader(problem, EpisodeState())
    messages = _initial_messages(problem, renderer, reader)
    for page in range(1, problem.num_pages + 1):
        call_id = f"read_page_{page}"
        messages.append(Message(
            role="assistant", content="",
            tool_calls=[ToolCall(
                id=call_id, function=ToolCall.FunctionBody(
                    name="read_page", arguments=json.dumps({"page": page}),
                ),
            )],
        ))
        messages.extend(simple_tool_result(
            f"[Page {page}]\n{problem.pages[page - 1]}",
            call_id=call_id, name="read_page",
        ).messages)
    return len(text_tokens(renderer.build_generation_prompt(messages)))


def _fit_problem(
    corpus: str, task: CountingTaskConfig, renderer: Renderer, seed: int,
) -> CountingProblem:
    target = task.target_context_tokens
    if target is None:
        return sample_problem(corpus, task, seed)
    if len(corpus) < 2:
        raise ValueError("Corpus is too short for context calibration")
    # Hold the corpus offset fixed so resizing extends the same document.
    start = random.Random(seed).randrange(len(corpus) // 2)
    low = 1
    high = (len(corpus) - start - 1) * 10 // (task.max_turns * 11)
    page_chars = min(task.page_chars, high)
    measured = 0
    for _ in range(12):
        if low > high:
            break
        problem = sample_problem(corpus, chz.replace(task, page_chars=page_chars), seed, start=start)
        measured = _completed_context_tokens(problem, renderer)
        if measured < 1:
            raise ValueError("Renderer produced an empty completed-document context")
        if abs(measured - target) <= _context_tolerance(target):
            logger.info(
                "Calibrated document context: target=%s tokens=%s page_chars=%s seed=%s",
                target, measured, page_chars, seed,
            )
            return replace(problem, calibrated_context_tokens=measured)
        if measured < target:
            low = page_chars + 1
        else:
            high = page_chars - 1
        page_chars = min(high, max(low, round(page_chars * target / measured)))
    raise ValueError(
        f"Cannot size completed-document context to {target} tokens "
        f"within {_context_tolerance(target)} tokens; last measured {measured}"
    )


class CountingEnv(EnvFromMessageEnv):
    """Reserve generation space and report exact token-prefix extension rates."""

    def __init__(
        self, renderer: Renderer, message_env: CountingMessageEnv,
        reward: CountingReward, token_limit: int, generation_tokens: int,
        target_context_tokens: int | None = None,
        calibrated_context_tokens: int | None = None,
    ):
        super().__init__(
            renderer, message_env, failed_parse_reward=-0.1,
            tool_specs=[tool.to_spec() for tool in message_env.tools],
        )
        self.reward = reward
        self.token_limit = token_limit
        self.generation_tokens = generation_tokens
        self.checked = self.extended = 0
        self.peak_context_tokens = 0
        self.max_sampled_prompt_tokens = 0
        self.target_context_tokens = target_context_tokens
        self.calibrated_context_tokens = calibrated_context_tokens
        self.previous: list[int] = []

    async def initial_observation(self):
        observation, stop = await super().initial_observation()
        self.previous = text_tokens(observation)
        self.peak_context_tokens = len(self.previous)
        if len(self.previous) + self.generation_tokens > self.token_limit:
            raise ValueError("Initial prompt plus generation budget exceeds max_trajectory_tokens")
        return observation, stop

    async def step(self, action: Action):
        self.max_sampled_prompt_tokens = max(self.max_sampled_prompt_tokens, len(self.previous))
        result = await super().step(action)
        next_tokens = text_tokens(result.next_observation)
        sampled_length = len(self.previous) + len(action)
        self.peak_context_tokens = max(
            self.peak_context_tokens, sampled_length,
            len(next_tokens) if not result.episode_done else 0,
        )
        overflow = sampled_length > self.token_limit or (
            not result.episode_done
            and len(next_tokens) + self.generation_tokens > self.token_limit
        )
        if overflow or result.metrics.get("parse_error"):
            result.reward = -0.1
            result.episode_done = True
            result.next_observation = ModelInput(chunks=[])
            result.metrics.pop("abs_error", None)
            result.metrics.update(self.reward.metrics(None))
            if overflow:
                result.metrics["context_overflow"] = 1.0
        elif not result.episode_done:
            prefix = self.previous + action
            self.checked += 1
            self.extended += next_tokens[:len(prefix)] == prefix
            self.previous = next_tokens
        if result.episode_done and self.checked:
            rate = self.extended / self.checked
            result.metrics["extension_rate"] = rate
            if self.checked >= 5 and rate < 0.5:
                logger.warning(
                    "Only %.1f%% of turns extend sampled tokens; choose a renderer "
                    "that preserves history to avoid separate full-context sequences.",
                    rate * 100,
                )
        if result.episode_done:
            result.metrics["max_context_tokens"] = self.peak_context_tokens
            result.metrics["max_sampled_prompt_tokens"] = self.max_sampled_prompt_tokens
            if self.target_context_tokens is not None:
                result.metrics["target_context_tokens"] = self.target_context_tokens
            if self.calibrated_context_tokens is not None:
                result.metrics["calibrated_context_tokens"] = self.calibrated_context_tokens
            result.metrics.setdefault("context_overflow", 0.0)
            result.metrics.setdefault("parse_error", 0.0)
        return result


@dataclass(frozen=True)
class CountingGroupBuilder(EnvGroupBuilder):
    problem: CountingProblem
    task: CountingTaskConfig
    renderer: Renderer
    group_size: int
    token_limit: int
    generation_tokens: int

    async def make_envs(self) -> Sequence[Env]:
        envs: list[Env] = []
        for _ in range(self.group_size):
            state = EpisodeState()
            reader = PageReader(self.problem, state)
            reward = CountingReward(self.problem, state, self.task.reward_tolerance)
            messages = _initial_messages(self.problem, self.renderer, reader)
            message_env = CountingMessageEnv(
                tools=[reader.read_page], initial_messages=messages,
                max_turns=self.task.max_turns, reward_fn=reward, state=state,
            )
            envs.append(CountingEnv(
                self.renderer, message_env, reward, self.token_limit, self.generation_tokens,
                self.task.target_context_tokens, self.problem.calibrated_context_tokens,
            ))
        return envs

    def logging_tags(self) -> list[str]:
        return ["rl_numerics_check"]


@dataclass
class CountingDataset(RLDataset):
    corpus: str
    task: CountingTaskConfig
    renderer: Renderer
    batch_size: int
    group_size: int
    num_examples: int
    seed: int
    token_limit: int
    generation_tokens: int

    def __len__(self) -> int:
        return (self.num_examples + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[CountingGroupBuilder]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.batch_size
        return [
            CountingGroupBuilder(
                _fit_problem(self.corpus, self.task, self.renderer, self.seed + i),
                self.task, self.renderer, self.group_size,
                self.token_limit, self.generation_tokens,
            )
            for i in range(start, min(start + self.batch_size, self.num_examples))
        ]


@chz.chz
class CountingDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    task: CountingTaskConfig = chz.field(default_factory=CountingTaskConfig)
    corpus_path: str | None = None
    seed: int = 0
    max_train_examples: int = 10_000
    max_test_examples: int = 0
    max_trajectory_tokens: int = 131072
    max_generation_tokens: int = 1024

    @chz.validate
    def _validate(self) -> None:
        if min(self.batch_size, self.group_size, self.max_train_examples) < 1:
            raise ValueError("Batch size, group size, and training example count must be positive")
        if self.max_test_examples < 0:
            raise ValueError("max_test_examples must be nonnegative")
        if not 0 < self.max_generation_tokens < self.max_trajectory_tokens:
            raise ValueError("Require 0 < max_generation_tokens < max_trajectory_tokens")
        target = self.task.target_context_tokens
        if target is not None and (
            target + _context_tolerance(target) + self.max_generation_tokens
            > self.max_trajectory_tokens
        ):
            raise ValueError("Target context plus tolerance and generation budget exceeds max_trajectory_tokens")

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        import asyncio

        corpus = await asyncio.to_thread(load_corpus, self.corpus_path)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer, prefer_non_thinking=True
        )
        renderer = get_renderer(renderer_name, get_tokenizer(self.model_name_for_tokenizer))

        def dataset(examples: int, seed: int) -> CountingDataset:
            return CountingDataset(
                corpus, self.task, renderer, self.batch_size, self.group_size,
                examples, seed, self.max_trajectory_tokens, self.max_generation_tokens,
            )

        train = dataset(self.max_train_examples, self.seed)
        # Fail before creating a paid session when the corpus, tools, or prompt are invalid.
        first_env = (await replace(train.get_batch(0)[0], group_size=1).make_envs())[0]
        try:
            await first_env.initial_observation()
        finally:
            await first_env.close()
        test = dataset(self.max_test_examples, self.seed + 1_000_000_000)
        return train, test if self.max_test_examples else None
