"""Unit tests for the per-sample timeout on SessionTokenCompleter.

A stalled sample must raise TimeoutError (so the rollout layer can treat it as a
retryable failed trajectory) instead of blocking the batch indefinitely.
"""

import asyncio
from types import SimpleNamespace

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.completers import SessionTokenCompleter


def _model_input():
    return ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])])


class _StallingClient:
    """sample_async never resolves within the test's timeout window."""

    def sample_async(self, **kwargs):
        async def _never():
            await asyncio.Event().wait()

        return _never()


class _ImmediateClient:
    """sample_async resolves immediately with one sequence."""

    def __init__(self):
        self.calls = 0

    def sample_async(self, **kwargs):
        self.calls += 1

        async def _done():
            seq = SimpleNamespace(tokens=[7, 8], logprobs=[-0.2, -0.3])
            return SimpleNamespace(sequences=[seq])

        return _done()


def test_sample_timeout_raises_timeout_error():
    completer = SessionTokenCompleter(
        _StallingClient(), max_tokens=16, sample_timeout_sec=0.05
    )

    async def run():
        await completer(_model_input(), stop=[])

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(run())


def test_no_timeout_returns_result():
    client = _ImmediateClient()
    completer = SessionTokenCompleter(client, max_tokens=16)

    result = asyncio.run(completer(_model_input(), stop=[]))

    assert client.calls == 1
    assert result.tokens == [7, 8]
    assert result.logprobs == [-0.2, -0.3]


def test_configured_timeout_allows_fast_result():
    client = _ImmediateClient()
    completer = SessionTokenCompleter(client, max_tokens=16, sample_timeout_sec=5.0)

    result = asyncio.run(completer(_model_input(), stop=[]))

    assert result.tokens == [7, 8]


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_timeout_rejected(bad):
    with pytest.raises(ValueError, match="sample_timeout_sec must be positive"):
        SessionTokenCompleter(_ImmediateClient(), max_tokens=16, sample_timeout_sec=bad)
