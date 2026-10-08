from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import chz
import pandas as pd
from huggingface_hub import hf_hub_download

from interactive_training import model_info, tokenizer_utils
from interactive_training.recipes.search_tool.tools import (
    RetrievalConfig,
    RetrievalTool,
    TextAnswerReward,
)
from interactive_training.renderers import get_renderer
from interactive_training.renderers.base import Message, Renderer
from interactive_training.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from interactive_training.tool_use import build_agent_tool_env

SEARCH_TASK_INSTRUCTIONS = """You are an expert assistant who solves tasks using a Wikipedia search tool.

Here are instructions for how to solve a problem:
1. Think step by step before calling the tool and after you receive the result of the tool call. Decide what queries to call the tool with.
2. Call the tool with the queries you have decided on.
3. Think step by step again after you receive the result of the tool call. If you have the information you need, you can stop here.
4. Otherwise, come up with new queries that combine information from the previous results.
5. Include your final answer after the \"Answer:\" prefix. The answer should be between one to five words.

Here is an example of solving a real question:
\"Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?\"

1. Think step by step: In order to answer this question, I need to know the population of New York City and San Francisco between 2020 and 2025. I will search for the population of New York City in each year
2. Calling search tool: <tool_call>{\"name\": \"search\", \"arguments\": {\"query_list\": [\"Population New York city between 2020 and 2025\"]}}</tool_call> (Output omitted for brevity)
3. Think step by step again: I have the population of New York City in each year, and I see that the population of New York City grew the most in 2024. I need to know the population of San Francisco in 2024. I will search for the population of San Francisco in each year.
<tool_call>{\"name\": \"search\", \"arguments\": {\"query_list\": [\"Population San Francisco between 2023 and 2024\"]}}</tool_call> (Output omitted for brevity)
4. Answer: The population of New York City grew the most in 2024, and the population of San Francisco changed by XXXX in 2024.
"""


class SearchR1Datum(TypedDict):
    question: str
    answer: list[str]
    data_source: str


def process_single_row(row_series: pd.Series) -> SearchR1Datum:
    """Normalize one row from the Search-R1 parquet shape."""
    row = row_series.to_dict()
    question = str(row.get("question", ""))

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth: Any = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", [])

    # Some rows may carry numpy arrays depending on parquet deserialization.
    if hasattr(ground_truth, "tolist"):
        ground_truth = ground_truth.tolist()

    if not isinstance(ground_truth, list):
        ground_truth = []
    answers = [x for x in ground_truth if isinstance(x, str)]

    return {
        "question": question,
        "answer": cast(list[str], answers),
        "data_source": str(row.get("data_source", "unknown")),
    }


def _default_cache_dir(split: Literal["train", "test"], repo_id: str) -> Path:
    user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
    return Path(tempfile.gettempdir()) / user / "data" / repo_id / split


def download_search_r1_dataset(
    split: Literal["train", "test"],
    cache_dir: str | None = None,
) -> list[SearchR1Datum]:
    """Download and parse Search-R1 train/test split from Hugging Face parquet."""
    hf_repo_id = "PeterJinGo/nq_hotpotqa_train"
    parquet_filename = f"{split}.parquet"

    local_dir = Path(cache_dir) if cache_dir else _default_cache_dir(split, hf_repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    local_parquet_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=parquet_filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )

    df_raw = pd.read_parquet(local_parquet_path)
    return df_raw.apply(process_single_row, axis=1).tolist()


def _initial_messages(
    datum: SearchR1Datum,
    renderer: Renderer,
    search_tool: RetrievalTool,
) -> list[Message]:
    tool_schemas = [search_tool.search.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=SEARCH_TASK_INSTRUCTIONS,
    )
    return prefix + [{"role": "user", "content": datum["question"]}]


@chz.chz
class SearchEnvGroupBuilder(EnvGroupBuilder):
    """Build a group of Search-R1 environments sharing one tool instance."""

    datum: SearchR1Datum
    model_name: str
    renderer_name: str | None
    max_turns: int
    group_size: int
    search_tool: RetrievalTool
    format_coef: float = 0.1
    max_trajectory_tokens: int = 32 * 1024

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        initial_messages = _initial_messages(self.datum, renderer, self.search_tool)
        reward_fn = TextAnswerReward(gold_answers=self.datum["answer"], format_coef=self.format_coef)

        return [
            build_agent_tool_env(
                renderer=renderer,
                tools=[self.search_tool.search],
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.datum.get("data_source", "unknown")]


class SearchRLDataset(RLDataset):
    """Batch adapter around SearchEnvGroupBuilder list."""

    def __init__(self, env_group_builders: list[SearchEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class SearchR1DatasetBuilder(RLDatasetBuilder):
    """Build train/eval datasets for Search-R1 tool-use RL."""

    model_name_for_tokenizer: str

    retrieval_host: str
    retrieval_port: int
    retrieval_config: RetrievalConfig = chz.field(default_factory=RetrievalConfig)

    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 5
    format_coef: float = 0.1
    max_trajectory_tokens: int = 32 * 1024
    seed: int = 0

    max_train_examples: int | None = None
    max_test_examples: int | None = 1000
    cache_dir: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        search_tool = await RetrievalTool.build(
            retrieval_host=self.retrieval_host,
            retrieval_port=self.retrieval_port,
            retrieval_config=self.retrieval_config,
        )

        rng = random.Random(self.seed)

        train_rows = download_search_r1_dataset("train", cache_dir=self.cache_dir)
        rng.shuffle(train_rows)
        if self.max_train_examples is not None and self.max_train_examples >= 0:
            train_rows = train_rows[: self.max_train_examples]

        train_builders = [
            SearchEnvGroupBuilder(
                datum=row,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                search_tool=search_tool,
                format_coef=self.format_coef,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            for row in train_rows
        ]
        train_dataset = SearchRLDataset(train_builders, self.batch_size)

        test_dataset: RLDataset | None = None
        if self.max_test_examples is None or self.max_test_examples != 0:
            test_rows = download_search_r1_dataset("test", cache_dir=self.cache_dir)
            random.Random(self.seed + 1).shuffle(test_rows)
            if self.max_test_examples is not None and self.max_test_examples > 0:
                test_rows = test_rows[: self.max_test_examples]

            test_builders = [
                SearchEnvGroupBuilder(
                    datum=row,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    max_turns=self.max_turns,
                    group_size=1,
                    search_tool=search_tool,
                    format_coef=self.format_coef,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
                for row in test_rows
            ]
            test_dataset = SearchRLDataset(test_builders, self.batch_size)

        return train_dataset, test_dataset
