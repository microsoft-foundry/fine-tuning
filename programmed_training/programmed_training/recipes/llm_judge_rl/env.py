"""Single-turn summary environment and group-level LLM judging."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Sequence

import chz
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from programmed_training import model_info, renderers
from programmed_training.recipes.llm_judge_rl.data import (
    BILLSUM_DATASET_NAME,
    BILLSUM_REVISION,
    BillSumExample,
    load_billsum_train,
    prepare_billsum_data,
)
from programmed_training.recipes.llm_judge_rl.grader import (
    AzureOpenAIJudge,
    JudgeGradingError,
)
from programmed_training.rl.rollouts import IncompleteGroupGradingError
from programmed_training.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from programmed_training.tokenizer_utils import get_tokenizer
from programmed_training.utils import logtree

logger = logging.getLogger(__name__)

POLICY_SYSTEM_PROMPT = """\
You summarize proposed legislation for a general audience. Write one accurate,
self-contained summary of at most {word_limit} words. Focus on the bill's main
provisions, affected parties, requirements, funding, enforcement, and effective
dates when present. Do not invent facts. Return only the summary.
"""


def build_policy_messages(
    example: BillSumExample, *, word_limit: int
) -> list[renderers.Message]:
    """Build policy-visible messages; the human reference is intentionally absent."""

    return [
        {
            "role": "system",
            "content": POLICY_SYSTEM_PROMPT.format(word_limit=word_limit),
        },
        {
            "role": "user",
            "content": f"Title: {example.title}\n\nBill text:\n{example.text}",
        },
    ]


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _model_input_length(model_input: ModelInput) -> int:
    return sum(
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in model_input.chunks
    )


class SummaryEnv(Env):
    """One policy completion; authoritative reward is computed by the group builder."""

    def __init__(
        self,
        *,
        renderer: renderers.Renderer,
        example: BillSumExample,
        word_limit: int,
        policy_max_tokens: int,
        dataset_metrics: Metrics,
    ) -> None:
        self.renderer = renderer
        self.example = example
        self.word_limit = word_limit
        self.policy_max_tokens = policy_max_tokens
        self.dataset_metrics = dataset_metrics
        self.candidate: str | None = None
        self.word_count = 0
        self.word_limit_compliant = False
        self.parse_success = False
        self.output_token_count = 0

    async def initial_observation(self):
        messages = build_policy_messages(self.example, word_limit=self.word_limit)
        return (
            self.renderer.build_generation_prompt(messages),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        candidate = renderers.get_text_content(message).strip()
        word_count = count_words(candidate)
        compliant = word_count <= self.word_limit
        self.candidate = candidate
        self.word_count = word_count
        self.word_limit_compliant = compliant
        self.parse_success = bool(parse_success and candidate)
        self.output_token_count = len(action)

        logtree.log_text(f"Example ID: {self.example.example_id}")
        logtree.log_text(f"Candidate summary: {candidate}")
        logtree.log_text(
            f"Words: {word_count}/{self.word_limit} | "
            f"limit compliant: {compliant} | parsed: {bool(parse_success)}"
        )
        metrics: Metrics = {
            "summary/word_count": float(word_count),
            "summary/word_limit": float(self.word_limit),
            "summary/word_limit_compliant": float(compliant),
            "summary/word_limit_excess": float(max(0, word_count - self.word_limit)),
            "policy/parse_success": float(self.parse_success),
            "policy/output_token_count": float(len(action)),
            "policy/output_at_token_limit": float(
                self.policy_max_tokens > 0 and len(action) >= self.policy_max_tokens
            ),
            **self.dataset_metrics,
        }
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=ModelInput(chunks=[]),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
            logs={"example_id": self.example.example_id},
        )


@dataclass(frozen=True)
class LLMJudgeGroupBuilder(EnvGroupBuilder):
    renderer: renderers.Renderer
    example: BillSumExample
    judge: AzureOpenAIJudge
    num_envs: int
    word_limit: int
    policy_max_tokens: int
    dataset_metrics: Metrics

    async def make_envs(self) -> Sequence[Env]:
        return [
            SummaryEnv(
                renderer=self.renderer,
                example=self.example,
                word_limit=self.word_limit,
                policy_max_tokens=self.policy_max_tokens,
                dataset_metrics=self.dataset_metrics,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        summary_envs = [env for env in env_group if isinstance(env, SummaryEnv)]
        if len(summary_envs) != len(env_group):
            raise TypeError("LLMJudgeGroupBuilder received an unexpected environment")
        if any(env.candidate is None for env in summary_envs):
            raise RuntimeError("summary environment completed without a candidate")

        results = await asyncio.gather(
            *[
                self.judge.grade(self.example, env.candidate or "")
                for env in summary_envs
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        unexpected = [
            result
            for result in results
            if isinstance(result, Exception)
            and not isinstance(result, JudgeGradingError)
        ]
        if unexpected:
            raise unexpected[0]
        failures = [
            result for result in results if isinstance(result, JudgeGradingError)
        ]
        if failures:
            categories = sorted(
                {
                    failure.category
                    if isinstance(failure, JudgeGradingError)
                    else type(failure).__name__
                    for failure in failures
                }
            )
            error = IncompleteGroupGradingError(
                "Azure OpenAI judge did not return a complete reward vector; "
                f"failures={len(failures)} categories={','.join(categories)}",
                trajectory_count=len(env_group),
            )
            raise error from failures[0]

        scored: list[tuple[float, Metrics]] = []
        for env, result in zip(summary_envs, results, strict=True):
            if isinstance(result, BaseException):
                raise AssertionError("unreachable")
            reward_eligible = env.parse_success and env.word_limit_compliant
            final_reward = result.reward if reward_eligible else 0.0
            metrics: Metrics = {
                "judge/aggregate_reward": result.reward,
                "policy/eligibility_adjusted_reward": final_reward,
                "judge/factuality_cap_applied": float(
                    result.factuality_cap_applied
                ),
                "judge/transport_retries": float(result.transport_retries),
                "judge/structured_retries": float(result.structured_retries),
                "judge/failures": 0.0,
                "judge/latency_ms": result.latency_ms,
                "judge/prompt_tokens": float(result.prompt_tokens),
                "summary/mechanical_limit_zeroed_reward": float(
                    not env.word_limit_compliant
                ),
                "policy/parse_failure_zeroed_reward": float(
                    not env.parse_success
                ),
                "policy/reward_eligible": float(reward_eligible),
            }
            metrics.update(
                {
                    f"judge/{name}": score
                    for name, score in result.normalized_dimensions.items()
                }
            )
            scored.append((final_reward, metrics))
        return scored

    def logging_tags(self) -> list[str]:
        return ["llm_judge_rl", "billsum"]


class BillSumRLDataset(RLDataset):
    def __init__(
        self, builders: list[LLMJudgeGroupBuilder], batch_size: int
    ) -> None:
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[LLMJudgeGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class BillSumDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    judge: AzureOpenAIJudge
    renderer_name: str | None = None
    dataset_name: str = BILLSUM_DATASET_NAME
    dataset_revision: str = BILLSUM_REVISION
    dataset_path: str | None = None
    validation_size: int = 500
    seed: int = 0
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    policy_prompt_token_budget: int = 12000
    judge_prompt_token_budget: int = 16000
    judge_candidate_token_reserve: int = 512
    word_limit: int = 200
    policy_max_tokens: int = 512

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        examples = load_billsum_train(
            dataset_name=self.dataset_name,
            revision=self.dataset_revision,
            dataset_path=self.dataset_path,
        )

        def policy_tokens(example: BillSumExample) -> int:
            prompt = renderer.build_generation_prompt(
                build_policy_messages(example, word_limit=self.word_limit)
            )
            return _model_input_length(prompt)

        prepared = prepare_billsum_data(
            examples=examples,
            validation_size=self.validation_size,
            seed=self.seed,
            policy_token_count=policy_tokens,
            judge_base_token_count=self.judge.base_prompt_token_count,
            policy_prompt_token_budget=self.policy_prompt_token_budget,
            judge_prompt_token_budget=self.judge_prompt_token_budget,
            judge_candidate_token_reserve=self.judge_candidate_token_reserve,
            max_train_examples=self.max_train_examples,
            max_validation_examples=self.max_test_examples,
        )
        dataset_metrics = prepared.stats.as_metrics()

        def builder(example: BillSumExample, num_envs: int) -> LLMJudgeGroupBuilder:
            return LLMJudgeGroupBuilder(
                renderer=renderer,
                example=example,
                judge=self.judge,
                num_envs=num_envs,
                word_limit=self.word_limit,
                policy_max_tokens=self.policy_max_tokens,
                dataset_metrics=dataset_metrics,
            )

        train_builders = [
            builder(example, self.group_size) for example in prepared.train
        ]
        validation_builders = [
            builder(example, 1) for example in prepared.validation
        ]
        return (
            BillSumRLDataset(train_builders, self.batch_size),
            BillSumRLDataset(validation_builders, self.batch_size),
        )
