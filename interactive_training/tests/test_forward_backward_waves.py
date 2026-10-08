"""Azure SDK chunk-wave integration with the cookbook training step."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from azure.ai.finetuningsessions.aio import _patch as aio_patch
from azure.ai.finetuningsessions.models import (
    Datum,
    ForwardBackwardOperationResult,
    ModelInput,
    ModelInputChunk,
    TensorData,
)
from interactive_training.rl.train import train_step
from interactive_training.rl.train_azure import AzureSDKTrainingClient


def _awaitable(value):
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


def _datum():
    return Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1])]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[1.0]),
            "weights": TensorData(data=[1.0]),
        },
    )


def _fwd_bwd_result():
    return ForwardBackwardOperationResult(
        {
            "total_loss": 1.0,
            "loss_fn_outputs": [{"logprobs": {"data": [-0.1]}}],
        }
    )


async def test_adapter_omits_wave_keyword_by_default():
    client = MagicMock()
    client.forward_backward_async = AsyncMock(
        return_value=_awaitable(_fwd_bwd_result())
    )
    training_client = AzureSDKTrainingClient(client, "model_abc", MagicMock())

    future = await training_client.forward_backward_async([_datum()], loss_fn="cross_entropy")
    await future.result_async()

    _, kwargs = client.forward_backward_async.call_args
    assert "max_chunks_per_wave" not in kwargs


async def test_adapter_passes_configured_wave_size():
    client = MagicMock()
    client.forward_backward_async = AsyncMock(
        return_value=_awaitable(_fwd_bwd_result())
    )
    training_client = AzureSDKTrainingClient(
        client,
        "model_abc",
        MagicMock(),
        forward_backward_chunk_wave_size=4,
    )

    future = await training_client.forward_backward_async([_datum()], loss_fn="cross_entropy")
    await future.result_async()

    _, kwargs = client.forward_backward_async.call_args
    assert kwargs["max_chunks_per_wave"] == 4


async def test_custom_loss_adapter_omits_wave_keyword_by_default():
    import torch

    client = MagicMock()
    client.forward_async = AsyncMock(return_value=_awaitable(_fwd_bwd_result()))
    client.forward_backward_async = AsyncMock(
        return_value=_awaitable(_fwd_bwd_result())
    )
    training_client = AzureSDKTrainingClient(client, "model_abc", MagicMock())

    def loss_fn(data, logprobs):
        return torch.cat(logprobs).sum(), {}

    future = await training_client.forward_backward_custom_async([_datum()], loss_fn=loss_fn)
    await future.result_async()

    # The surrogate forward_backward is the last forward_backward_async call.
    _, kwargs = client.forward_backward_async.call_args
    assert "max_chunks_per_wave" not in kwargs


async def test_custom_loss_adapter_passes_configured_wave_size():
    import torch

    client = MagicMock()
    client.forward_async = AsyncMock(return_value=_awaitable(_fwd_bwd_result()))
    client.forward_backward_async = AsyncMock(
        return_value=_awaitable(_fwd_bwd_result())
    )
    training_client = AzureSDKTrainingClient(
        client,
        "model_abc",
        MagicMock(),
        forward_backward_chunk_wave_size=4,
    )

    def loss_fn(data, logprobs):
        return torch.cat(logprobs).sum(), {}

    future = await training_client.forward_backward_custom_async([_datum()], loss_fn=loss_fn)
    await future.result_async()

    # The surrogate forward_backward must carry the same bounded-wave knob.
    _, kwargs = client.forward_backward_async.call_args
    assert kwargs["max_chunks_per_wave"] == 4


async def test_all_waves_complete_before_single_optimizer_post(monkeypatch):
    events = []
    chunks = [["chunk-1"], ["chunk-2"], ["chunk-3"]]

    class _Pending:
        def __init__(self, wave):
            self._wave = wave

        async def _poll_chunk_results(self):
            events.append(("poll", list(self._wave)))
            return [
                ForwardBackwardOperationResult(
                    {
                        "total_loss": 1.0,
                        "loss_fn_outputs": [{"logprobs": {"data": [-0.1]}}],
                    }
                )
                for _ in self._wave
            ]

    async def _post_wave(
        client,
        session_id,
        wave,
        *,
        loss_fn,
        loss_fn_config,
    ):
        del client, session_id, loss_fn, loss_fn_config
        events.append(("post", list(wave)))
        return _Pending(wave)

    class _Client:
        optim_calls = 0

        async def forward_backward_async(self, session_id, batch, **kwargs):
            return await aio_patch.forward_backward_async(self, session_id, batch, **kwargs)

        async def optim_step_async(self, session_id, adam_params):
            del session_id, adam_params
            self.optim_calls += 1
            events.append(("optim", None))
            return _awaitable(MagicMock(metrics={}))

    monkeypatch.setattr(aio_patch, "_chunk_data", lambda batch: chunks)
    monkeypatch.setattr(aio_patch, "_forward_backward_chunks_post", _post_wave)
    client = _Client()
    training_client = AzureSDKTrainingClient(
        client,
        "model_abc",
        MagicMock(),
        forward_backward_chunk_wave_size=2,
    )

    await train_step(
        [_datum()],
        training_client,
        learning_rate=1e-5,
        num_substeps=1,
        loss_fn="cross_entropy",
    )

    assert events == [
        ("post", [["chunk-1"], ["chunk-2"]]),
        ("poll", [["chunk-1"], ["chunk-2"]]),
        ("post", [["chunk-3"]]),
        ("poll", [["chunk-3"]]),
        ("optim", None),
    ]
    assert client.optim_calls == 1
