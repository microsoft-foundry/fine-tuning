from __future__ import annotations

import asyncio
import logging
import re
import string
from dataclasses import dataclass
from functools import reduce
from typing import Annotated

import chz
import httpx

from interactive_training.renderers import get_text_content
from interactive_training.renderers.base import Message
from interactive_training.tool_use import ToolResult, simple_tool_result, tool

logger = logging.getLogger(__name__)

# The retrieval server serializes searches (one brute-force FAISS scan over
# ~21M vectors at a time, using all CPU cores), so in-flight requests queue
# server-side. Bound client concurrency so the queue tail latency
# (concurrency x per-search time) stays well under the httpx read timeout
# below; an unbounded fan-out would let the tail exceed the timeout and fail
# every request with an empty ReadTimeout.
_CONNECTION_SEMAPHORE = asyncio.Semaphore(16)


def normalize_answer(s: str) -> str:
    """Normalize answer by lowercasing, removing punctuation/articles, and fixing whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return reduce(lambda text, fn: fn(text), [lower, remove_punc, remove_articles, white_space_fix], s)


@chz.chz
class RetrievalConfig:
    n_results: int = 3


class RetrievalTool:
    """Search tool backed by a FAISS + E5 retrieval server (``retrieval_server.py``).

    The recipe sends raw query strings to the server, which encodes them with
    E5 and returns the top-k wiki-18 passages, so query and document vectors
    share one vector space without any local embedding step.

    The async ``httpx`` client is not picklable. We drop it in ``__getstate__``
    and lazily reconnect on first use after deserialization.
    """

    def __init__(
        self,
        base_url: str,
        retrieval_config: RetrievalConfig,
        max_retries: int,
        initial_retry_delay: int,
    ):
        self._base_url = base_url.rstrip("/")
        self._retrieval_config = retrieval_config
        self._max_retries = max_retries
        self._initial_retry_delay = initial_retry_delay
        self._client: httpx.AsyncClient | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)
        return self._client

    @staticmethod
    async def build(
        retrieval_host: str,
        retrieval_port: int,
        retrieval_config: RetrievalConfig | None = None,
        max_retries: int = 10,
        initial_retry_delay: int = 1,
    ) -> RetrievalTool:
        return RetrievalTool(
            base_url=f"http://{retrieval_host}:{retrieval_port}",
            retrieval_config=retrieval_config or RetrievalConfig(),
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
        )

    async def _retrieve_with_retry(self, query_list: list[str]) -> list[list[str]]:
        client = self._ensure_client()
        payload = {
            "queries": query_list,
            "topk": self._retrieval_config.n_results,
        }
        for attempt in range(self._max_retries):
            try:
                response = await client.post("/retrieve", json=payload)
                response.raise_for_status()
                return response.json().get("result", [])
            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait_time = self._initial_retry_delay * (1.5**attempt)
                    logger.error(
                        "Retrieval attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise RuntimeError("All retrieval attempts failed")

    @tool
    async def search(
        self,
        query_list: Annotated[
            list[str],
            "A list of semantic search queries. Returns top matching Wikipedia snippets for each query.",
        ],
    ) -> ToolResult:
        """Search the indexed Wikipedia corpus for relevant snippets."""
        async with _CONNECTION_SEMAPHORE:
            results = await self._retrieve_with_retry(query_list)

        message_content = ""
        for query, documents in zip(query_list, results):
            message_content += f"Query: {query}\n"
            for doc_i, doc in enumerate(documents):
                message_content += f"Document {doc_i + 1}:\n{doc}\n"

        return simple_tool_result(message_content)


@dataclass
class TextAnswerReward:
    """Reward for final short-text answer with 'Answer:' format.

    Formula: format_coef * (format_score - 1) + correct
    """

    gold_answers: list[str]
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            return 0.0, {"format": 0.0, "correct": 0.0}

        content = get_text_content(final_message)
        format_score = float(self._extract_answer(content) is not None)
        correct = float(self._check_answer(content))
        reward = self.format_coef * (format_score - 1.0) + correct
        return reward, {"format": format_score, "correct": correct}

    def _extract_answer(self, text: str) -> str | None:
        if "Answer:" not in text:
            return None
        parts = text.split("Answer:")
        if len(parts) != 2:
            return None
        return parts[1].strip()

    def _check_answer(self, text: str) -> bool:
        model_answer = self._extract_answer(text)
        if model_answer is None or len(self.gold_answers) == 0:
            return False
        for gold in self.gold_answers:
            if normalize_answer(model_answer) == normalize_answer(gold):
                return True
        return False
