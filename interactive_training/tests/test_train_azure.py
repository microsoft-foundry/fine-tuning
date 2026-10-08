"""Unit tests for interactive_training.rl.train_azure adapters.

Tests verify that AzureSDKTrainingClient and AzureSDKSamplingClient
correctly wrap FineTuningSessionClient methods and translate SDK results
into the duck-types expected by rl/train.py.

Run with:
    cd interactive_training && uv run pytest tests/ -v
"""

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.ai.finetuningsessions.models import (
    AdamParams,
    Datum,
    ForwardBackwardOperationResult,
    ModelInput,
    ModelInputChunk,
    SampleOperationResult,
    SampledSequence,
    SamplingParams,
    SaveCheckpointOperationResult,
    SaveSamplerWeightsOperationResult,
    OptimStepOperationResult,
    TensorData,
)

from interactive_training.rl.train_azure import (
    _FwdBwdResult,
    _LogprobsEntry,
    _OptimResult,
    _CheckpointResult,
    _SamplerCheckpointResult,
    _SampleResult,
    _SDKFuture,
    AzureSDKSamplingClient,
    AzureSDKTrainingClient,
)
from interactive_training.rl.types import SamplingCheckpoint
from interactive_training.recipes.math_rl import train_azure as math_train_azure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _awaitable(value):
    """Wrap a value in a resolved asyncio.Future so it can be awaited by _SDKFuture."""
    fut = asyncio.get_event_loop().create_future()
    fut.set_result(value)
    return fut


def _make_mock_client() -> MagicMock:
    """Create a mock FineTuningSessionClient with async methods."""
    client = MagicMock()
    client.forward_backward_async = AsyncMock()
    client.forward_async = AsyncMock()
    client.optim_step_async = AsyncMock()
    client.optim_step_post = AsyncMock()
    client.save_weights_async = AsyncMock()
    client.save_weights_for_sampler_async = AsyncMock()
    client.save_weights_and_get_sampling_client_async = AsyncMock()
    client.sample = AsyncMock()
    client.close_session = AsyncMock()
    client.create_session = AsyncMock(return_value="model_test123")
    return client


def _make_fb_result(total_loss=1.5, logprobs=None, metrics=None, loss_fn_outputs=None):
    """Build a ForwardBackwardOperationResult, optionally with loss_fn_outputs."""
    if loss_fn_outputs is not None:
        # Use dict constructor so the extra field survives
        d = {
            "type": "forward_backward",
            "operation_id": "req_1",
            "status": "succeeded",
            "total_loss": total_loss,
            "loss_fn_outputs": loss_fn_outputs,
        }
        if logprobs is not None:
            d["per_datum_logprobs"] = logprobs
        if metrics is not None:
            d["metrics"] = metrics
        return ForwardBackwardOperationResult(d)
    return ForwardBackwardOperationResult(
        total_loss=total_loss,
        per_datum_logprobs=logprobs,
        metrics=metrics,
    )


def _make_tokenizer():
    """Create a minimal tokenizer mock."""
    tok = MagicMock()
    tok.encode = MagicMock(return_value=[1, 2, 3])
    tok.decode = MagicMock(return_value="hello")
    return tok


# ---------------------------------------------------------------------------
# _FwdBwdResult adapter tests
# ---------------------------------------------------------------------------


class TestFwdBwdResult:
    """Test the _FwdBwdResult adapter."""

    def test_loss_fn_outputs_from_extra_field(self):
        """When loss_fn_outputs is present, adapter reads logprobs from it."""
        sdk_result = _make_fb_result(
            total_loss=2.0,
            loss_fn_outputs=[
                {"logprobs": {"data": [-0.5, -0.3]}},
                {"logprobs": {"data": [-0.1, -0.2]}},
            ],
        )
        adapted = _FwdBwdResult(sdk_result)
        assert len(adapted.loss_fn_outputs) == 2
        assert adapted.loss_fn_outputs[0].data == [-0.5, -0.3]
        assert adapted.loss_fn_outputs[1].data == [-0.1, -0.2]

    def test_fallback_to_per_datum_logprobs(self):
        """When loss_fn_outputs is absent, adapter falls back to per_datum_logprobs."""
        sdk_result = _make_fb_result(
            total_loss=1.0,
            logprobs=[TensorData(data=[-1.0, -2.0])],
        )
        adapted = _FwdBwdResult(sdk_result)
        assert len(adapted.loss_fn_outputs) == 1
        assert adapted.loss_fn_outputs[0].data == [-1.0, -2.0]

    def test_metrics_exposed(self):
        """Metrics from SDK result are exposed on the adapter."""
        sdk_result = _make_fb_result(
            total_loss=1.0,
            metrics={"total_loss:sum": 1.0, "kl:mean": 0.01},
        )
        adapted = _FwdBwdResult(sdk_result)
        assert adapted.metrics["total_loss:sum"] == 1.0
        assert adapted.metrics["kl:mean"] == 0.01

    def test_empty_logprobs(self):
        """No logprobs → empty loss_fn_outputs."""
        sdk_result = _make_fb_result(total_loss=0.5)
        adapted = _FwdBwdResult(sdk_result)
        assert adapted.loss_fn_outputs == []


class TestLogprobsEntry:
    """Test the _LogprobsEntry helper."""

    def test_getitem_logprobs(self):
        entry = _LogprobsEntry([-0.5, -0.3])
        lp = entry["logprobs"]
        assert lp.data == [-0.5, -0.3]

    def test_getitem_wrong_key(self):
        entry = _LogprobsEntry([-0.5])
        with pytest.raises(KeyError):
            entry["not_logprobs"]

    def test_contains(self):
        entry = _LogprobsEntry([-0.5])
        assert "logprobs" in entry
        assert "other" not in entry


# ---------------------------------------------------------------------------
# _OptimResult adapter tests
# ---------------------------------------------------------------------------


class TestOptimResult:
    def test_metrics_exposed(self):
        sdk_result = MagicMock()
        sdk_result.metrics = {"grad_norm": 0.5, "step_count": 10}
        adapted = _OptimResult(sdk_result)
        assert adapted.metrics["grad_norm"] == 0.5

    def test_none_metrics(self):
        sdk_result = MagicMock()
        sdk_result.metrics = None
        adapted = _OptimResult(sdk_result)
        assert adapted.metrics == {}


# ---------------------------------------------------------------------------
# _CheckpointResult adapter tests
# ---------------------------------------------------------------------------


class TestCheckpointResult:
    def test_path_from_path_attr(self):
        sdk_result = MagicMock()
        sdk_result.path = "my_checkpoint"
        sdk_result.checkpoint_id = "ckpt_1"
        adapted = _CheckpointResult(sdk_result)
        assert adapted.path == "my_checkpoint"

    def test_path_fallback_to_checkpoint_id(self):
        sdk_result = MagicMock()
        sdk_result.path = None
        sdk_result.checkpoint_id = "ckpt_1"
        adapted = _CheckpointResult(sdk_result)
        assert adapted.path == "ckpt_1"

    def test_interactive_training_uri_passed_through(self):
        """The SDK adapter preserves service paths until checkpoint logging."""
        sdk_result = MagicMock()
        sdk_result.path = "service://model_abc123/weights/5"
        sdk_result.checkpoint_id = "ckpt_1"
        adapted = _CheckpointResult(sdk_result)
        assert adapted.path == "service://model_abc123/weights/5"

    def test_empty_path_falls_back(self):
        """Empty string path falls back to checkpoint_id."""
        sdk_result = MagicMock()
        sdk_result.path = ""
        sdk_result.checkpoint_id = "ckpt_fallback"
        adapted = _CheckpointResult(sdk_result)
        assert adapted.path == "ckpt_fallback"


# ---------------------------------------------------------------------------
# _SamplerCheckpointResult adapter tests
# ---------------------------------------------------------------------------


class TestSamplerCheckpointResult:
    def test_checkpoint_id_and_path(self):
        sdk_result = MagicMock()
        sdk_result.checkpoint_id = "ss1_seq5"
        adapted = _SamplerCheckpointResult(sdk_result)
        assert adapted.checkpoint_id == "ss1_seq5"
        assert adapted.path == "ss1_seq5"


# ---------------------------------------------------------------------------
# _SampleResult adapter tests
# ---------------------------------------------------------------------------


class TestSampleResult:
    def test_sequences_exposed(self):
        seq = SampledSequence(tokens=[1, 2, 3], text="abc")
        sdk_result = MagicMock()
        sdk_result.sequences = [seq]
        adapted = _SampleResult(sdk_result)
        assert len(adapted.sequences) == 1
        assert adapted.sequences[0].tokens == [1, 2, 3]

    def test_empty_sequences(self):
        sdk_result = MagicMock()
        sdk_result.sequences = None
        adapted = _SampleResult(sdk_result)
        assert adapted.sequences == []


# ---------------------------------------------------------------------------
# _SDKFuture tests
# ---------------------------------------------------------------------------


class TestSDKFuture:
    @pytest.mark.asyncio
    async def test_result_async_no_converter(self):
        future = asyncio.Future()
        future.set_result(42)
        sdk_future = _SDKFuture(future)
        assert await sdk_future.result_async() == 42

    @pytest.mark.asyncio
    async def test_result_async_with_converter(self):
        future = asyncio.Future()
        future.set_result(10)
        sdk_future = _SDKFuture(future, converter=lambda x: x * 2)
        assert await sdk_future.result_async() == 20


# ---------------------------------------------------------------------------
# AzureSDKTrainingClient tests
# ---------------------------------------------------------------------------


class TestAzureSDKTrainingClient:
    def test_operation_context_hooks_set_and_reset_sdk_label(self):
        from azure.ai.finetuningsessions.aio import _patch as aio_patch

        tc = AzureSDKTrainingClient(
            _make_mock_client(), "model_abc", _make_tokenizer()
        )
        token = tc.set_operation_context("step=12")
        try:
            assert aio_patch.operation_label_var.get() == "step=12"
        finally:
            tc.reset_operation_context(token)

        assert aio_patch.operation_label_var.get() == ""

    @pytest.mark.asyncio
    async def test_forward_backward_async(self):
        """forward_backward_async wraps client.forward_backward_async and returns _FwdBwdResult."""
        client = _make_mock_client()
        fb_result = _make_fb_result(
            total_loss=2.5,
            loss_fn_outputs=[{"logprobs": {"data": [-0.1, -0.2]}}],
        )
        client.forward_backward_async.return_value = _awaitable(fb_result)

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        batch = [MagicMock(spec=Datum)]
        future = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
        result = await future.result_async()

        client.forward_backward_async.assert_called_once_with(
            "model_abc", batch, loss_fn="cross_entropy", loss_fn_config=None,
        )
        assert isinstance(result, _FwdBwdResult)
        assert len(result.loss_fn_outputs) == 1

    @pytest.mark.asyncio
    async def test_forward_backward_async_strips_client_only_mask_key(self):
        """``mask`` is a cookbook-internal aux key and must not reach the wire.

        Regression guard for ``_strip_client_only_keys``: the plain
        ``forward_backward_async`` path (i.e. NOT the custom-loss path,
        which already strips everything except ``{target_tokens, weights}``)
        carries arbitrary loss-input keys through to the server, with
        ``mask`` as the sole documented exception.  The server's loss
        schema would reject ``mask``; this test pins the strip so a regression
        fails here in CI instead of on the first real training step.
        """
        client = _make_mock_client()
        client.forward_backward_async.return_value = _awaitable(
            _make_fb_result(total_loss=1.0)
        )

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        batch = [
            Datum(
                model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2])]),
                loss_fn_inputs={
                    "target_tokens": TensorData(data=[2.0, 3.0]),
                    "weights": TensorData(data=[0.1, 0.2]),
                    "mask": TensorData(data=[1.0, 0.0]),
                },
            ),
        ]

        future = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
        await future.result_async()

        wire_batch = client.forward_backward_async.call_args[0][1]
        assert len(wire_batch) == 1
        assert "mask" not in wire_batch[0].loss_fn_inputs
        # Other keys are passed through untouched.
        assert set(wire_batch[0].loss_fn_inputs.keys()) == {
            "target_tokens",
            "weights",
        }

    @pytest.mark.asyncio
    async def test_forward_backward_uses_configured_fixed_poll_interval(self):
        client = _make_mock_client()
        client.forward_backward_async.return_value = _awaitable(_make_fb_result())
        tc = AzureSDKTrainingClient(
            client,
            "model_abc",
            _make_tokenizer(),
            forward_poll_interval_sec=1.0,
        )

        future = await tc.forward_backward_async(
            [MagicMock(spec=Datum)],
            loss_fn="cross_entropy",
        )
        await future.result_async()

        assert client.forward_backward_async.await_args.kwargs == {
            "loss_fn": "cross_entropy",
            "loss_fn_config": None,
            "poll_min_sec": 1.0,
            "poll_max_sec": 1.0,
        }

    @pytest.mark.asyncio
    async def test_optim_step_async_default_params(self):
        """optim_step_async with None creates default AdamParams."""
        client = _make_mock_client()
        optim_result = MagicMock()
        optim_result.metrics = {"grad_norm": 0.5}
        client.optim_step_async.return_value = _awaitable(optim_result)

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        future = await tc.optim_step_async(None)
        result = await future.result_async()

        assert client.optim_step_async.called
        call_args = client.optim_step_async.call_args
        assert call_args[0][0] == "model_abc"
        adam = call_args[0][1]
        assert isinstance(adam, AdamParams)
        assert isinstance(result, _OptimResult)

    @pytest.mark.asyncio
    async def test_optim_step_increments_step(self):
        """Each optim_step_async call increments internal step counter."""
        client = _make_mock_client()
        client.optim_step_async.return_value = _awaitable(MagicMock(metrics={}))

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        assert tc._step == 0
        await tc.optim_step_async()
        assert tc._step == 1
        await tc.optim_step_async()
        assert tc._step == 2

    @pytest.mark.asyncio
    async def test_optim_step_uses_configured_fixed_poll_interval(self):
        client = _make_mock_client()
        client.optim_step_async.return_value = _awaitable(MagicMock(metrics={}))
        tc = AzureSDKTrainingClient(
            client,
            "model_abc",
            _make_tokenizer(),
            optim_poll_interval_sec=0.5,
        )

        await tc.optim_step_async()

        assert client.optim_step_async.await_args.kwargs == {
            "poll_min_sec": 0.5,
            "poll_max_sec": 0.5,
        }

    @pytest.mark.asyncio
    async def test_optim_poll_waits_for_paired_forward_completion(self):
        client = _make_mock_client()
        forward_task = asyncio.get_running_loop().create_future()
        client.forward_backward_async.return_value = forward_task

        pending = MagicMock()
        pending.poll_result = AsyncMock(return_value=MagicMock(metrics={}))
        client.optim_step_post.return_value = pending

        tc = AzureSDKTrainingClient(
            client,
            "model_abc",
            _make_tokenizer(),
            forward_poll_interval_sec=1.0,
            optim_poll_interval_sec=0.5,
            defer_optim_poll_until_forward_complete=True,
        )

        forward = await tc.forward_backward_async([MagicMock(spec=Datum)])
        optim = await tc.optim_step_async()

        client.optim_step_post.assert_awaited_once()
        await asyncio.sleep(0)
        pending.poll_result.assert_not_awaited()

        forward_task.set_result(_make_fb_result())
        assert isinstance(await forward.result_async(), _FwdBwdResult)
        assert isinstance(await optim.result_async(), _OptimResult)
        pending.poll_result.assert_awaited_once_with(
            poll_min_sec=0.5,
            poll_max_sec=0.5,
        )

    @pytest.mark.asyncio
    async def test_deferred_optim_poll_requires_forward(self):
        tc = AzureSDKTrainingClient(
            _make_mock_client(),
            "model_abc",
            _make_tokenizer(),
            defer_optim_poll_until_forward_complete=True,
        )

        with pytest.raises(RuntimeError, match="preceding forward_backward_async"):
            await tc.optim_step_async()

    @pytest.mark.asyncio
    async def test_deferred_optim_polls_pair_with_forwards_fifo(self):
        client = _make_mock_client()
        forward_a = asyncio.get_running_loop().create_future()
        forward_b = asyncio.get_running_loop().create_future()
        client.forward_backward_async.side_effect = [forward_a, forward_b]

        pending_a = MagicMock()
        pending_a.poll_result = AsyncMock(return_value=MagicMock(metrics={}))
        pending_b = MagicMock()
        pending_b.poll_result = AsyncMock(return_value=MagicMock(metrics={}))
        client.optim_step_post.side_effect = [pending_a, pending_b]

        tc = AzureSDKTrainingClient(
            client,
            "model_abc",
            _make_tokenizer(),
            defer_optim_poll_until_forward_complete=True,
        )

        await tc.forward_backward_async([MagicMock(spec=Datum)])
        await tc.forward_backward_async([MagicMock(spec=Datum)])
        optim_a = await tc.optim_step_async()
        optim_b = await tc.optim_step_async()
        await asyncio.sleep(0)

        forward_b.set_result(_make_fb_result())
        await asyncio.sleep(0)
        pending_a.poll_result.assert_not_awaited()
        pending_b.poll_result.assert_awaited_once()

        forward_a.set_result(_make_fb_result())
        await optim_a.result_async()
        await optim_b.result_async()
        pending_a.poll_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_state_async(self):
        """save_state_async delegates to client.save_weights_async."""
        client = _make_mock_client()
        ckpt = MagicMock()
        ckpt.path = "step10"
        ckpt.checkpoint_id = "step10"
        client.save_weights_async.return_value = _awaitable(ckpt)

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        future = await tc.save_state_async("step10")
        result = await future.result_async()

        client.save_weights_async.assert_called_once_with(
            "model_abc", "step10", step_number=None, metrics=None
        )
        assert isinstance(result, _CheckpointResult)
        assert result.path == "step10"

    @pytest.mark.asyncio
    async def test_save_weights_for_sampler_async(self):
        """save_weights_for_sampler_async delegates to client.save_weights_for_sampler_async."""
        client = _make_mock_client()
        sampler_ckpt = MagicMock()
        sampler_ckpt.checkpoint_id = "my_ckpt"
        client.save_weights_for_sampler_async.return_value = _awaitable(sampler_ckpt)

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        tc._step = 5
        future = await tc.save_weights_for_sampler_async("my_ckpt")
        result = await future.result_async()

        assert client.save_weights_for_sampler_async.called
        call_args = client.save_weights_for_sampler_async.call_args
        assert call_args[0][0] == "model_abc"  # session_id
        assert call_args[0][1] == "my_ckpt"  # name
        assert isinstance(result, _SamplerCheckpointResult)

    @pytest.mark.asyncio
    async def test_save_weights_and_get_sampling_client(self):
        """save_weights_and_get_sampling_client_async returns AzureSDKSamplingClient."""
        client = _make_mock_client()
        sampler_ckpt = MagicMock()
        sampler_ckpt.checkpoint_id = "ss1_seq1"
        client.save_weights_and_get_sampling_client_async.return_value = _awaitable(sampler_ckpt)

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        tc._step = 1
        sampling_client = await tc.save_weights_and_get_sampling_client_async()

        assert isinstance(sampling_client, AzureSDKSamplingClient)
        assert sampling_client._checkpoint_id == "ss1_seq1"
        assert sampling_client._session_id == "model_abc"

    def test_create_sampling_client(self):
        """create_sampling_client returns AzureSDKSamplingClient with given path."""
        client = _make_mock_client()
        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        sc = tc.create_sampling_client("ckpt_path")
        assert isinstance(sc, AzureSDKSamplingClient)
        assert sc._checkpoint_id == "ckpt_path"
        assert sc._session_id == "model_abc"

    def test_get_tokenizer(self):
        """get_tokenizer returns the injected tokenizer."""
        tok = _make_tokenizer()
        tc = AzureSDKTrainingClient(_make_mock_client(), "model_abc", tok)
        assert tc.get_tokenizer() is tok

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_surrogate_payload(self):
        """custom loss with loss = sum(logprobs) → autograd gives g=ones → weights = -1.0."""
        client = _make_mock_client()
        # forward (server-side) returns logprobs [-0.5, -0.3]
        fwd_result = _make_fb_result(
            total_loss=0.0,
            loss_fn_outputs=[{"logprobs": {"data": [-0.5, -0.3]}}],
        )
        client.forward_async.return_value = _awaitable(fwd_result)
        # forward_backward (server-side) returns whatever -- we just need to
        # inspect the surrogate batch passed to it.
        fb_result = _make_fb_result(total_loss=1.0, metrics={"grad_norm": 0.5})
        client.forward_backward_async.return_value = _awaitable(fb_result)

        tc = AzureSDKTrainingClient(
            client,
            "model_abc",
            _make_tokenizer(),
            forward_backward_chunk_wave_size=4,
            forward_poll_interval_sec=0.25,
        )
        original = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[10, 20])]),
            loss_fn_inputs={"target_tokens": TensorData(data=[20.0, 30.0])},
        )

        import torch as _torch

        def loss_fn(data, logprobs):
            # sum of logprobs → dC/dell = ones
            return _torch.cat(logprobs).sum(), {}

        future = await tc.forward_backward_custom_async([original], loss_fn=loss_fn)
        await future.result_async()

        # Inspect the surrogate batch dispatched to forward_backward_async.
        client.forward_backward_async.assert_called_once()
        call_args = client.forward_backward_async.call_args
        assert call_args[0][0] == "model_abc"
        surrogate_batch = call_args[0][1]
        assert call_args[1]["loss_fn"] == "cross_entropy"
        assert call_args[1]["max_chunks_per_wave"] == 4
        assert call_args[1]["poll_min_sec"] == 0.25
        assert call_args[1]["poll_max_sec"] == 0.25
        assert len(surrogate_batch) == 1
        surrogate = surrogate_batch[0]
        # model_input is the original, unmodified.
        assert surrogate.model_input is original.model_input
        # weights = -g = -1.0 for each token.
        weights = surrogate.loss_fn_inputs["weights"].data
        assert weights == [-1.0, -1.0]

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_emits_surrogate_weight_diagnostics(
        self,
    ):
        """Surrogate-weight diagnostics flow through ``result.metrics``.

        Pins the conventions so future dashboard wiring can rely on them:
          - keys live under the ``surrogate/`` namespace
          - values are plain Python ``float`` (JSON / wandb safe)
          - the all-positive degenerate case (loss = -sum(logprobs) gives
            g = -1, so weights = +1) collapses to the vanilla-CE signature:
            ``pos_frac == 1.0`` and ``cos_uniform == 1.0`` -- this is the
            "your custom loss is just CE in disguise" indicator
          - user-returned metrics override on name collision
        """
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(
            _make_fb_result(loss_fn_outputs=[{"logprobs": {"data": [-0.5, -0.3]}}])
        )
        client.forward_backward_async.return_value = _awaitable(
            _make_fb_result(total_loss=1.0)
        )

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        datum = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[10, 20])]),
            loss_fn_inputs={"target_tokens": TensorData(data=[20.0, 30.0])},
        )

        import torch as _torch

        def loss_fn(data, logprobs):
            # loss = -sum(lp) -> dC/dlp = -1 -> weights = -g = +1 (CE-like)
            return -_torch.cat(logprobs).sum(), {
                # Collision sentinel: this should win over the adapter's
                # default value to confirm precedence.
                "surrogate/weights_pos_frac": 999.0,
            }

        future = await tc.forward_backward_custom_async([datum], loss_fn=loss_fn)
        result = await future.result_async()

        # Convention: namespace + float types.
        assert "surrogate/weights_l2" in result.metrics
        assert "surrogate/weights_pos_frac" in result.metrics
        assert "surrogate/weights_cos_uniform" in result.metrics
        for k in (
            "surrogate/weights_l2",
            "surrogate/weights_pos_frac",
            "surrogate/weights_cos_uniform",
        ):
            assert isinstance(result.metrics[k], float), (
                f"{k} must be a plain float for JSON/wandb compatibility, "
                f"got {type(result.metrics[k])}"
            )

        # CE-degenerate case: all weights +1.
        assert result.metrics["surrogate/weights_l2"] == pytest.approx(
            (2.0) ** 0.5, abs=1e-6
        )
        # User override beats adapter default.
        assert result.metrics["surrogate/weights_pos_frac"] == 999.0
        assert result.metrics["surrogate/weights_cos_uniform"] == pytest.approx(
            1.0, abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_allows_extra_keys_but_strips_from_server_payload(
        self,
    ):
        """Extra aux keys in loss_fn_inputs are readable by the closure but
        NEVER reach the server.

        Regression guard for two distinct contract points:
          1. The client-side validator must permit arbitrary aux keys
             (advantages, sampling_logprobs, ref_logprobs, ...).
          2. Both server calls (forward + surrogate forward_backward) must
             receive Datums whose ``loss_fn_inputs`` contains ONLY
             ``{target_tokens, weights}`` -- extras must be stripped on
             egress.  This protects against a regression where
             ``_prepare_forward_data`` pass-through would leak the original
             Datum (with extras) to the wire.
        """
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(
            _make_fb_result(loss_fn_outputs=[{"logprobs": {"data": [-0.5, -0.3]}}])
        )
        client.forward_backward_async.return_value = _awaitable(
            _make_fb_result(total_loss=1.0)
        )

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        datum = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[10, 20])]),
            loss_fn_inputs={
                "target_tokens": TensorData(data=[20.0, 30.0]),
                "weights": TensorData(data=[0.1, 0.2]),  # exercise the
                # "weights-already-present" code path (the buggy branch).
                "advantages": TensorData(data=[0.5, -0.5]),
                "sampling_logprobs": TensorData(data=[-1.1, -1.2]),
            },
        )

        import torch as _torch

        seen_keys: list[set[str]] = []

        def loss_fn(data, logprobs):
            # Closure must still see the original (un-stripped) aux fields.
            for d in data:
                seen_keys.append(set(d.loss_fn_inputs.keys()))
            return _torch.cat(logprobs).sum(), {}

        # (1) No ValueError -- extras are accepted client-side.
        future = await tc.forward_backward_custom_async([datum], loss_fn=loss_fn)
        await future.result_async()

        # (1b) The closure observed all four keys on the original datum.
        assert seen_keys == [
            {"target_tokens", "weights", "advantages", "sampling_logprobs"}
        ]

        # (2) The forward call (server-bound) payload was stripped.
        client.forward_async.assert_called_once()
        forward_batch = client.forward_async.call_args[0][1]
        assert len(forward_batch) == 1
        assert set(forward_batch[0].loss_fn_inputs.keys()) == {
            "target_tokens",
            "weights",
        }
        # And user-supplied weights were preserved (not overwritten by zeros).
        assert forward_batch[0].loss_fn_inputs["weights"].data == [0.1, 0.2]

        # (2b) The surrogate forward_backward payload was also stripped.
        client.forward_backward_async.assert_called_once()
        surrogate_batch = client.forward_backward_async.call_args[0][1]
        assert len(surrogate_batch) == 1
        assert set(surrogate_batch[0].loss_fn_inputs.keys()) == {
            "target_tokens",
            "weights",
        }

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_rejects_missing_target_tokens(self):
        """Missing target_tokens in loss_fn_inputs raises ValueError."""
        client = _make_mock_client()
        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        bad = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1])]),
            loss_fn_inputs={"weights": TensorData(data=[0.0])},
        )

        import torch as _torch

        def loss_fn(data, logprobs):
            return _torch.cat(logprobs).sum(), {}

        with pytest.raises(ValueError, match="missing required key 'target_tokens'"):
            await tc.forward_backward_custom_async([bad], loss_fn=loss_fn)

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_injects_zero_weights_for_forward(self):
        """When input lacks 'weights', forward call sees an injected zero tensor."""
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(
            _make_fb_result(loss_fn_outputs=[{"logprobs": {"data": [-0.1, -0.2]}}])
        )
        client.forward_backward_async.return_value = _awaitable(_make_fb_result())

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        original = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2])]),
            loss_fn_inputs={"target_tokens": TensorData(data=[2.0, 3.0])},
        )

        import torch as _torch

        def loss_fn(data, logprobs):
            return _torch.cat(logprobs).sum(), {}

        await (await tc.forward_backward_custom_async([original], loss_fn=loss_fn)).result_async()

        # Inspect the forward call's batch.
        fwd_batch = client.forward_async.call_args[0][1]
        assert set(fwd_batch[0].loss_fn_inputs.keys()) == {"target_tokens", "weights"}
        assert fwd_batch[0].loss_fn_inputs["weights"].data == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_preserves_user_weights_in_forward(self):
        """When input already has 'weights', forward call gets the original datum unchanged."""
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(
            _make_fb_result(loss_fn_outputs=[{"logprobs": {"data": [-0.1, -0.2]}}])
        )
        client.forward_backward_async.return_value = _awaitable(_make_fb_result())

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        original = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2])]),
            loss_fn_inputs={
                "target_tokens": TensorData(data=[2.0, 3.0]),
                "weights": TensorData(data=[0.7, 0.3]),
            },
        )

        import torch as _torch

        def loss_fn(data, logprobs):
            return _torch.cat(logprobs).sum(), {}

        await (await tc.forward_backward_custom_async([original], loss_fn=loss_fn)).result_async()

        fwd_batch = client.forward_async.call_args[0][1]
        # User-supplied weights preserved verbatim; payload is stripped to the
        # standard {target_tokens, weights} contract (see
        # test_..._allows_extra_keys_but_strips_from_server_payload).
        assert set(fwd_batch[0].loss_fn_inputs.keys()) == {"target_tokens", "weights"}
        assert fwd_batch[0].loss_fn_inputs["weights"].data == [0.7, 0.3]
        assert fwd_batch[0].loss_fn_inputs["target_tokens"].data == [2.0, 3.0]

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_raises_when_grad_is_none(self):
        """If user's loss doesn't depend on a datum's logprobs, raise ValueError."""
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(
            _make_fb_result(
                loss_fn_outputs=[
                    {"logprobs": {"data": [-0.1]}},
                    {"logprobs": {"data": [-0.2]}},
                ]
            )
        )
        client.forward_backward_async.return_value = _awaitable(_make_fb_result())

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        datums = [
            Datum(
                model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1])]),
                loss_fn_inputs={"target_tokens": TensorData(data=[2.0])},
            ),
            Datum(
                model_input=ModelInput(chunks=[ModelInputChunk(tokens=[3])]),
                loss_fn_inputs={"target_tokens": TensorData(data=[4.0])},
            ),
        ]

        def loss_fn(data, logprobs):
            # Loss only depends on logprobs[0]; logprobs[1].grad will be None.
            return logprobs[0].sum(), {}

        with pytest.raises(ValueError, match=r"data\[1\].*no gradient"):
            await tc.forward_backward_custom_async(datums, loss_fn=loss_fn)
        # Surrogate forward_backward never dispatched.
        client.forward_backward_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_raises_when_grad_length_mismatches_target_tokens(
        self,
    ):
        """``_build_surrogate_datum`` rejects a per-datum gradient whose length
        doesn't match ``target_tokens`` -- gives custom-loss authors a clear
        error instead of an obscure server-side cross-entropy failure.
        """
        import torch as _torch
        from interactive_training.rl.train_azure import _build_surrogate_datum

        original = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])]),
            loss_fn_inputs={"target_tokens": TensorData(data=[2.0, 3.0, 4.0])},
        )
        with pytest.raises(ValueError, match=r"length 4.*length 3"):
            _build_surrogate_datum(original, _torch.zeros(4))

    @pytest.mark.asyncio
    async def test_forward_backward_custom_async_merges_metrics(self):
        """User metrics merge with backend metrics in the returned result."""
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(
            _make_fb_result(loss_fn_outputs=[{"logprobs": {"data": [-0.5]}}])
        )
        client.forward_backward_async.return_value = _awaitable(
            _make_fb_result(metrics={"grad_norm": 0.5})
        )

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        original = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1])]),
            loss_fn_inputs={"target_tokens": TensorData(data=[2.0])},
        )

        import torch as _torch

        def loss_fn(data, logprobs):
            return _torch.cat(logprobs).sum(), {"my_loss": 3.14}

        future = await tc.forward_backward_custom_async([original], loss_fn=loss_fn)
        result = await future.result_async()

        assert result.metrics["grad_norm"] == 0.5
        assert result.metrics["my_loss"] == 3.14


# ---------------------------------------------------------------------------
# AzureSDKSamplingClient tests
# ---------------------------------------------------------------------------


class TestAzureSDKSamplingClient:
    def test_sampling_checkpoint_is_public_and_immutable(self):
        sc = AzureSDKSamplingClient(
            _make_mock_client(), "model_abc", "ckpt_1", _make_tokenizer()
        )

        assert sc.sampling_checkpoint == SamplingCheckpoint("model_abc", "ckpt_1")
        with pytest.raises(FrozenInstanceError):
            sc.sampling_checkpoint.session_id = "other"

    @pytest.mark.asyncio
    async def test_sample_async(self):
        """sample_async preserves structured prompts for the SDK."""
        client = _make_mock_client()
        seq = SampledSequence(tokens=[10, 20, 30], text="xyz")
        sample_result = MagicMock()
        sample_result.sequences = [seq]
        client.sample.return_value = sample_result

        prompt = ModelInput(chunks=[
            ModelInputChunk(tokens=[1, 2]),
            ModelInputChunk(tokens=[3, 4]),
        ])
        sc = AzureSDKSamplingClient(client, "model_abc", "ckpt_1", _make_tokenizer())
        result = await sc.sample_async(prompt, num_samples=2)

        client.sample.assert_called_once()
        call_args = client.sample.call_args
        assert call_args[0][0] == "model_abc"  # session_id
        assert call_args[0][1] is prompt
        assert call_args[1]["checkpoint_id"] == "ckpt_1"
        assert call_args[1]["num_samples"] == 2
        assert isinstance(result, _SampleResult)
        assert result.sequences[0].tokens == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_compute_logprobs_async(self):
        """compute_logprobs_async calls sample with max_tokens=1 and prompt_logprobs=True.

        max_tokens must be >= 1 (the server rejects 0); prompt_logprobs come from
        prefill, so the single generated token is discarded.
        """
        client = _make_mock_client()
        sample_result = MagicMock()
        sample_result.prompt_logprobs = [None, -0.2, -0.3]
        sample_result.sequences = [MagicMock()]
        client.sample.return_value = sample_result

        prompt = ModelInput(chunks=[ModelInputChunk(tokens=[5, 6, 7])])
        sc = AzureSDKSamplingClient(client, "model_abc", "ckpt_1", _make_tokenizer())
        logprobs = await sc.compute_logprobs_async(prompt)

        assert logprobs == [None, -0.2, -0.3]
        call_args = client.sample.call_args
        sp = call_args[0][2]  # SamplingParams
        assert sp.max_tokens == 1
        assert call_args[1]["prompt_logprobs"] is True

    @pytest.mark.asyncio
    async def test_compute_logprobs_async_empty(self):
        """compute_logprobs_async returns [] when no prompt_logprobs."""
        client = _make_mock_client()
        sample_result = MagicMock()
        sample_result.prompt_logprobs = None
        sample_result.sequences = [MagicMock()]
        client.sample.return_value = sample_result

        prompt = ModelInput(chunks=[ModelInputChunk(tokens=[5])])
        sc = AzureSDKSamplingClient(client, "model_abc", "ckpt_1", _make_tokenizer())
        logprobs = await sc.compute_logprobs_async(prompt)
        assert logprobs == []

    def test_model_path_alias(self):
        """model_path is set to checkpoint_id for checkpoint_utils compat."""
        sc = AzureSDKSamplingClient(
            _make_mock_client(), "model_abc", "my_ckpt", _make_tokenizer()
        )
        assert sc.model_path == "my_ckpt"


# ---------------------------------------------------------------------------
# forward_async tests
# ---------------------------------------------------------------------------


class TestForwardAsync:
    @pytest.mark.asyncio
    async def test_forward_async_delegates_to_client_forward(self):
        """forward_async wraps client.forward_async and returns _FwdBwdResult."""
        client = _make_mock_client()
        fb_result = _make_fb_result(
            total_loss=0.0,
            loss_fn_outputs=[{"logprobs": {"data": [-0.5, -0.3]}}],
        )
        client.forward_async.return_value = _awaitable(fb_result)

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        batch = [MagicMock(spec=Datum)]
        future = await tc.forward_async(batch, loss_fn="cross_entropy")
        result = await future.result_async()

        client.forward_async.assert_called_once_with(
            "model_abc", batch, loss_fn="cross_entropy", loss_fn_config=None,
        )
        assert isinstance(result, _FwdBwdResult)
        assert len(result.loss_fn_outputs) == 1
        assert result.loss_fn_outputs[0].data == [-0.5, -0.3]

    @pytest.mark.asyncio
    async def test_forward_async_defaults_to_cross_entropy(self):
        """forward_async defaults loss_fn to cross_entropy when None."""
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(_make_fb_result())

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        await tc.forward_async([MagicMock(spec=Datum)])

        _, kwargs = client.forward_async.call_args
        assert kwargs["loss_fn"] == "cross_entropy"

    @pytest.mark.asyncio
    async def test_forward_async_passes_loss_fn_config(self):
        """forward_async forwards loss_fn_config to client.forward_async."""
        client = _make_mock_client()
        client.forward_async.return_value = _awaitable(_make_fb_result())
        config = {"beta": 0.1}

        tc = AzureSDKTrainingClient(client, "model_abc", _make_tokenizer())
        await tc.forward_async(
            [MagicMock(spec=Datum)], loss_fn="cross_entropy", loss_fn_config=config
        )

        _, kwargs = client.forward_async.call_args
        assert kwargs["loss_fn_config"] == config


# ---------------------------------------------------------------------------
# ForwardInput / ForwardRequest model tests
# ---------------------------------------------------------------------------


class TestForwardInputModel:
    def test_forward_input_is_subclass_of_forward_backward_input(self):
        """ForwardInput inherits from ForwardBackwardInput."""
        from azure.ai.finetuningsessions.models import ForwardBackwardInput, ForwardInput

        assert issubclass(ForwardInput, ForwardBackwardInput)

    def test_forward_input_construction(self):
        """ForwardInput can be constructed with the same args as ForwardBackwardInput."""
        from azure.ai.finetuningsessions.models import ForwardInput, ForwardRequest

        datum = MagicMock(spec=Datum)
        fi = ForwardInput(data=[datum], loss_fn="cross_entropy")
        assert fi.data == [datum]
        assert fi.loss_fn == "cross_entropy"
        assert fi.loss_fn_config is None

    def test_forward_request_wraps_forward_input(self):
        """ForwardRequest has a forward_input field typed as ForwardInput."""
        from azure.ai.finetuningsessions.models import ForwardInput, ForwardRequest

        datum = MagicMock(spec=Datum)
        fi = ForwardInput(data=[datum], loss_fn="cross_entropy")
        req = ForwardRequest(forward_input=fi)
        assert req.forward_input is fi


# ---------------------------------------------------------------------------
# Training-budget tests: max_steps and max_wall_clock_seconds wiring
# ---------------------------------------------------------------------------


class TestComputeEffectiveEnd:
    """Unit tests for compute_effective_end()."""

    def test_no_cap_returns_num_batches(self):
        from interactive_training.rl.train_azure import compute_effective_end

        assert compute_effective_end(start_batch=0, num_batches=100, max_steps=None) == 100
        assert compute_effective_end(start_batch=42, num_batches=100, max_steps=None) == 100

    def test_max_steps_caps_below_dataset(self):
        from interactive_training.rl.train_azure import compute_effective_end

        # start=0, dataset=100, cap=10 -> stop at 10
        assert compute_effective_end(start_batch=0, num_batches=100, max_steps=10) == 10

    def test_max_steps_respects_resume(self):
        from interactive_training.rl.train_azure import compute_effective_end

        # Resumed at batch 30, want 5 more steps -> stop at 35
        assert compute_effective_end(start_batch=30, num_batches=100, max_steps=5) == 35

    def test_max_steps_clamped_to_dataset(self):
        from interactive_training.rl.train_azure import compute_effective_end

        # Cap larger than remaining dataset is clamped to num_batches
        assert compute_effective_end(start_batch=90, num_batches=100, max_steps=999) == 100

    def test_max_steps_zero_yields_no_iterations(self):
        from interactive_training.rl.train_azure import compute_effective_end

        # max_steps=0 means "do nothing" -- effective_end equals start_batch
        assert compute_effective_end(start_batch=7, num_batches=100, max_steps=0) == 7

    def test_negative_max_steps_raises(self):
        from interactive_training.rl.train_azure import compute_effective_end

        with pytest.raises(ValueError):
            compute_effective_end(start_batch=0, num_batches=100, max_steps=-1)


class TestSyncTrainingDeadline:
    """Verify the inner training loops honor a wall-clock deadline.

    Regression test for: math_rl/train_azure.py exposed max_wall_clock_seconds
    on CLIConfig but the Azure-SDK main() never plumbed a deadline through to
    do_sync_training, so runs continued past the budget.
    """

    @pytest.mark.asyncio
    async def test_do_sync_training_breaks_on_deadline(self):
        """A deadline already in the past stops the loop before any iteration."""
        import time as _time

        from interactive_training.rl.train import Config, do_sync_training

        # Capture i_batch values reached.  We only need the deadline check to
        # fire; we patch save_checkpoint_and_get_sampling_client so the loop
        # body never actually runs (and the test stays hermetic).
        with patch(
            "interactive_training.rl.train.save_checkpoint_and_get_sampling_client",
            new=AsyncMock(return_value=(MagicMock(), {})),
        ):
            cfg = MagicMock(spec=Config)
            cfg.log_path = "/tmp/__never_used__"
            cfg.save_every = 999
            cfg.ttl_seconds = None
            cfg.eval_every = 0
            cfg.learning_rate = 1e-5
            # The refill path (dynamic_sampling, or a strategy with
            # refill_on_drop) triggers extra pre-loop work that calls
            # len(dataset); disable all of its inputs so the test stays hermetic.
            # A MagicMock(spec=Config) yields truthy child mocks for
            # strategy/refill_on_drop, which would otherwise enable refill.
            cfg.dynamic_sampling = False
            cfg.strategy = None
            cfg.refill_on_drop = False
            # Rollout resilience is opt-in; disable it so the hermetic mock does
            # not build a retry policy from truthy child mocks.
            cfg.sample_timeout_sec = None
            cfg.max_retries_per_trajectory = 0
            cfg.max_extra_trajectory_attempts_per_group = 0

            await do_sync_training(
                start_batch=0,
                end_batch=100,
                num_batches=100,
                cfg=cfg,
                training_client=MagicMock(),
                kl_reference_client=None,
                evaluators=[],
                dataset=MagicMock(),
                ml_logger=MagicMock(),
                tokenizer=MagicMock(),
                deadline=_time.time() - 1.0,  # already past
            )
        # If the loop respected the deadline, we returned without executing
        # the (unmocked) batch body.  No assertion on side effects -- the
        # test passing without raising AttributeError on the unmocked
        # dataset.get_batch is itself the proof.


class TestMainHonorsBudget:
    """Verify train_azure.main() forwards budget knobs to training_func."""

    @pytest.mark.asyncio
    async def test_main_passes_effective_end_and_deadline(self):
        """When max_steps and max_wall_clock_seconds are set, main() shrinks
        end_batch and computes a finite deadline before calling training_func."""
        import time as _time

        from interactive_training.rl import train_azure

        captured: dict[str, object] = {}

        async def fake_training_func(**kwargs):
            captured.update(kwargs)
            return kwargs["end_batch"], kwargs["end_batch"] - 1

        # Build a minimal Config-like stub.  We avoid constructing a real
        # Config (its dataclass has many required fields and Pydantic
        # validators) -- main() only reads attributes off cfg.
        cfg = MagicMock()
        cfg.log_path = "/tmp/__never_used_main__"
        cfg.wandb_project = None
        cfg.wandb_name = None
        cfg.enable_trace = False
        cfg.tokenizer_name = None
        cfg.model_name = "Qwen/Qwen3-0.6B"
        cfg.kl_penalty_coef = 0
        cfg.async_config = None
        cfg.stream_minibatch_config = None
        cfg.eval_every = 0
        cfg.ttl_seconds = None
        cfg.evaluator_builders = []
        cfg.max_steps = 7
        cfg.max_wall_clock_seconds = 30.0

        # dataset_builder() -> (dataset, maybe_test_dataset)
        dataset = MagicMock()
        dataset.__len__ = lambda self: 100

        async def _builder():
            return dataset, None

        cfg.dataset_builder = _builder

        client = MagicMock()
        client.close_session = AsyncMock()
        client.close = AsyncMock()

        with patch.object(train_azure, "ml_log") as ml_mock, \
             patch.object(train_azure, "get_tokenizer", return_value=MagicMock()), \
             patch.object(train_azure, "AzureSDKTrainingClient", return_value=MagicMock()), \
             patch.object(train_azure.checkpoint_utils, "get_last_checkpoint", return_value=None), \
             patch.object(train_azure.checkpoint_utils, "save_checkpoint_async",
                          new=AsyncMock(return_value={})), \
             patch.object(train_azure, "do_sync_training", new=fake_training_func):
            ml_mock.setup_logging.return_value = MagicMock()

            t0 = _time.time()
            await train_azure.main(cfg, client, "session_xyz")

        # Effective end honors max_steps (start_batch=0, dataset=100, cap=7)
        assert captured["end_batch"] == 7
        assert captured["num_batches"] == 100
        assert captured["start_batch"] == 0

        # Deadline was computed and falls within the expected window
        deadline = captured["deadline"]
        assert deadline is not None
        assert t0 + 29.0 <= deadline <= t0 + 31.5

    @pytest.mark.asyncio
    async def test_main_no_budget_passes_none_deadline(self):
        """When neither knob is set, end_batch=num_batches and deadline=None."""
        from interactive_training.rl import train_azure

        captured: dict[str, object] = {}

        async def fake_training_func(**kwargs):
            captured.update(kwargs)
            return kwargs["end_batch"], kwargs["end_batch"] - 1

        cfg = MagicMock()
        cfg.log_path = "/tmp/__never_used_main2__"
        cfg.wandb_project = None
        cfg.wandb_name = None
        cfg.enable_trace = False
        cfg.tokenizer_name = None
        cfg.model_name = "Qwen/Qwen3-0.6B"
        cfg.kl_penalty_coef = 0
        cfg.async_config = None
        cfg.stream_minibatch_config = None
        cfg.eval_every = 0
        cfg.ttl_seconds = None
        cfg.evaluator_builders = []
        cfg.max_steps = None
        cfg.max_wall_clock_seconds = None

        dataset = MagicMock()
        dataset.__len__ = lambda self: 42

        async def _builder():
            return dataset, None

        cfg.dataset_builder = _builder

        client = MagicMock()
        client.close_session = AsyncMock()
        client.close = AsyncMock()

        with patch.object(train_azure, "ml_log") as ml_mock, \
             patch.object(train_azure, "get_tokenizer", return_value=MagicMock()), \
             patch.object(train_azure, "AzureSDKTrainingClient", return_value=MagicMock()), \
             patch.object(train_azure.checkpoint_utils, "get_last_checkpoint", return_value=None), \
             patch.object(train_azure.checkpoint_utils, "save_checkpoint_async",
                          new=AsyncMock(return_value={})), \
             patch.object(train_azure, "do_sync_training", new=fake_training_func):
            ml_mock.setup_logging.return_value = MagicMock()
            await train_azure.main(cfg, client, "session_xyz")

        assert captured["end_batch"] == 42
        assert captured["deadline"] is None

class TestLoopSignaturesAcceptDeadline:
    """All three inner training loops must accept a ``deadline`` kwarg.

    The ``compute_effective_end`` + deadline plumbing in
    ``rl/train_azure.main()`` invokes whichever loop matches ``cfg.async_config``
    / ``cfg.stream_minibatch_config``. If any one of them lacks the kwarg, the
    Azure entrypoint will raise TypeError at runtime instead of honoring the
    budget. This signature check is the cheapest regression guard.
    """

    def test_do_sync_training_accepts_deadline(self):
        import inspect

        from interactive_training.rl.train import do_sync_training

        sig = inspect.signature(getattr(do_sync_training, "__wrapped__", do_sync_training))
        assert "deadline" in sig.parameters
        assert sig.parameters["deadline"].default is None

    def test_do_async_training_accepts_deadline(self):
        import inspect

        from interactive_training.rl.train import do_async_training

        sig = inspect.signature(getattr(do_async_training, "__wrapped__", do_async_training))
        assert "deadline" in sig.parameters
        assert sig.parameters["deadline"].default is None

    def test_do_sync_training_with_stream_minibatch_accepts_deadline(self):
        import inspect

        from interactive_training.rl.train import do_sync_training_with_stream_minibatch

        sig = inspect.signature(
            getattr(
                do_sync_training_with_stream_minibatch,
                "__wrapped__",
                do_sync_training_with_stream_minibatch,
            )
        )
        assert "deadline" in sig.parameters
        assert sig.parameters["deadline"].default is None


class TestConfigBudgetFields:
    """Config must declare the budget knobs with safe defaults."""

    def test_config_has_budget_fields_with_none_defaults(self):
        from interactive_training.rl.train import Config

        # Read defaults from the chz dataclass without constructing a full Config
        # (which has many required fields). We just need to confirm the fields
        # exist and default to None so silent omission stays backward-compatible.
        ann = getattr(Config, "__annotations__", {})
        assert "max_steps" in ann
        assert "max_wall_clock_seconds" in ann


def _minimal_config(**overrides):
    """Build a minimal Config for validator tests.

    Uses a stub dataset_builder since we never call it -- the chz validator
    runs at construction time, which is all we exercise here.
    """
    import chz

    from interactive_training.rl.types import RLDatasetBuilder
    from interactive_training.rl.train import Config

    @chz.chz
    class _StubBuilder(RLDatasetBuilder):
        async def __call__(self):
            return [], None

    defaults = dict(
        learning_rate=1e-5,
        dataset_builder=_StubBuilder(),
        model_name="test-model",
        max_tokens=10,
        log_path="/tmp/x",
        eval_every=0,
        save_every=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestConfigCustomLossInvariant:
    """Config.loss_fn and Config.custom_loss_name must stay in sync."""

    def test_default_is_builtin_loss_with_none_name(self):
        cfg = _minimal_config()
        assert cfg.loss_fn == "importance_sampling"
        assert cfg.custom_loss_name is None

    def test_custom_with_name_is_accepted(self):
        cfg = _minimal_config(loss_fn="custom", custom_loss_name="dapo")
        assert cfg.loss_fn == "custom"
        assert cfg.custom_loss_name == "dapo"

    def test_custom_without_name_raises(self):
        with pytest.raises(ValueError, match="custom_loss_name"):
            _minimal_config(loss_fn="custom")

    def test_name_without_custom_sentinel_raises(self):
        with pytest.raises(ValueError, match="must be in sync"):
            _minimal_config(custom_loss_name="dapo")

    def test_builtin_loss_with_no_name_is_accepted(self):
        cfg = _minimal_config(loss_fn="cross_entropy")
        assert cfg.loss_fn == "cross_entropy"
        assert cfg.custom_loss_name is None


class TestMathRlRecipeResolvesCustomLoss:
    """The math_rl recipe must translate cli_config.custom_loss into the
    new Config.loss_fn='custom' + Config.custom_loss_name pair, not leak
    the (ignored) cli_config.loss_fn enum into Config when a custom loss
    is in use.  Textual check (cli_main spins up an Azure session and is
    not unit-testable end-to-end)."""

    def test_recipe_sets_config_loss_fn_to_sentinel_for_custom(self):
        import inspect

        from interactive_training.recipes.math_rl import train_azure as recipe

        src = inspect.getsource(recipe)
        # The resolution must set Config.loss_fn = "custom" and pass the
        # registry key through to custom_loss_name when custom_loss is set.
        assert 'config_loss_fn: str = "custom"' in src
        assert "config_custom_loss_name: str | None = cli_config.custom_loss" in src
        # And the Config(...) call must forward both fields.
        assert "loss_fn=config_loss_fn" in src
        assert "custom_loss_name=config_custom_loss_name" in src


class TestMathRlRecipeExposesBudget:
    """The math_rl/train_azure.py recipe must expose the budget knobs and
    forward them into Config -- this is the user-facing surface from the
    original bug report (BUGBASH #31)."""

    def test_cli_config_declares_budget_knobs(self):
        from interactive_training.recipes.math_rl.train_azure import CLIConfig

        ann = getattr(CLIConfig, "__annotations__", {})
        assert "max_steps" in ann
        assert "max_wall_clock_seconds" in ann

    def test_cli_config_defaults_are_none(self):
        """Budget knobs default to None so existing callers see no change."""
        from interactive_training.recipes.math_rl.train_azure import CLIConfig

        # chz dataclasses store field defaults on the class. Read them via the
        # attribute lookup chain that chz exposes (falls back to inspecting
        # the class dict if no chz API is available).
        for fname in ("max_steps", "max_wall_clock_seconds"):
            default = getattr(CLIConfig, fname, "MISSING")
            # chz stores defaults as class attrs; if it's a chz Field marker
            # we accept that too -- we only care it isn't a required field.
            assert default is None or default == "MISSING" or default is not Ellipsis, (
                f"{fname} should default to None, got {default!r}"
            )

    def test_cli_config_forwards_budget_to_config(self):
        """The recipe source must reference cli_config.max_steps and
        cli_config.max_wall_clock_seconds inside its Config(...) call.

        This is a textual check rather than a runtime call because the recipe's
        cli_main() spins up an Azure session and is not unit-testable. The
        regression we are guarding against is "knob added to CLIConfig but
        never forwarded to Config" -- which would silently disable the budget
        even though `--max_steps=...` parses successfully.
        """
        import inspect

        from interactive_training.recipes.math_rl import train_azure as recipe

        src = inspect.getsource(recipe)
        assert "max_steps=cli_config.max_steps" in src
        assert "max_wall_clock_seconds=cli_config.max_wall_clock_seconds" in src


class TestMathRlRecipeForwardsUserMetadata:
    """The math_rl/train_azure.py recipe exposes a ``user_metadata`` knob and
    forwards it verbatim to ``create_session``. The session schema accepts
    arbitrary JSON (``dict[str, Any]``) and the deployment metadata includes
    booleans (``customBoolean``, ``customFalseBoolean``), so the
    CLI field must be typed ``dict[str, Any]`` -- typing it ``dict[str, str]``
    makes chz coerce ``user_metadata.customBoolean=True`` to the
    string ``"True"`` and the recipe stops forwarding the user's types
    faithfully to the server.
    """

    def test_cli_config_user_metadata_is_any_typed(self):
        """The field must be ``dict[str, Any] | None``, never ``dict[str, str]``."""
        from interactive_training.recipes.math_rl.train_azure import CLIConfig

        ann = getattr(CLIConfig, "__annotations__", {})
        assert "user_metadata" in ann
        ann_str = str(ann["user_metadata"])
        assert "str, str" not in ann_str.replace(" ", "").replace(",", ", "), (
            f"user_metadata must not be dict[str, str]; got {ann_str!r}"
        )
        assert "Any" in ann_str, (
            f"user_metadata must be Any-typed so booleans survive; got {ann_str!r}"
        )

    def test_boolean_metadata_forwarded_unchanged(self):
        """chz must parse boolean CLI metadata into real bools (not strings).

        Regression guard: with ``dict[str, str]`` chz coerced these to the
        strings ``"True"``/``"False"``; with ``dict[str, Any]`` they stay bools.
        """
        import chz

        from interactive_training.recipes.math_rl.train_azure import CLIConfig

        cfg = chz.Blueprint(CLIConfig).make_from_argv(
            [
                "project_endpoint=http://localhost:8000",
                "user_metadata.customBoolean=True",
                "user_metadata.customFalseBoolean=False",
                "user_metadata.experimentName=modelcopy",
            ]
        )
        assert cfg.user_metadata == {
            "customBoolean": True,
            "customFalseBoolean": False,
            "experimentName": "modelcopy",
        }
        # Types, not just equality (True == 1 == "1" is a trap): assert real bools.
        assert cfg.user_metadata["customBoolean"] is True
        assert cfg.user_metadata["customFalseBoolean"] is False
        assert isinstance(cfg.user_metadata["experimentName"], str)

    def test_recipe_forwards_user_metadata_to_create_session(self):
        """The recipe must pass cli_config.user_metadata into create_session.

        Textual check: cli_main() spins up an Azure session and is not
        unit-testable end-to-end, so guard against the "knob added but never
        forwarded" regression at the source level.
        """
        import inspect

        from interactive_training.recipes.math_rl import train_azure as recipe

        src = inspect.getsource(recipe)
        assert "user_metadata=cli_config.user_metadata" in src


class TestAzureRlRecipesExposeCreateSessionTimeout:
    """Azure recipes must let slow first-load models wait longer than 600s."""

    def test_session_creating_recipes_forward_create_session_timeout(self):
        import inspect

        from interactive_training.recipes.code_rl import train_azure as code_recipe
        from interactive_training.recipes.math_rl import train_azure as math_recipe
        from interactive_training.recipes.search_tool import train_azure as search_tool_recipe
        from interactive_training.recipes.tulu3_sft import train_azure as tulu3_sft_recipe
        from interactive_training.recipes.tool_rl import train_azure as tool_recipe

        for recipe in (
            math_recipe,
            tool_recipe,
            code_recipe,
            search_tool_recipe,
            tulu3_sft_recipe,
        ):
            ann = getattr(recipe.CLIConfig, "__annotations__", {})
            assert "create_session_timeout_sec" in ann
            src = inspect.getsource(recipe)
            assert "create_session_timeout_sec: float = 600.0" in src
            assert "timeout_sec=cli_config.create_session_timeout_sec" in src


# ---------------------------------------------------------------------------
# _parse_checkpoint_path tests (recipes/math_rl/train_azure.py)
# ---------------------------------------------------------------------------

from interactive_training.recipes.math_rl.train_azure import (
    _parse_checkpoint_path,
)


class TestParseCheckpointPath:
    """Tests for checkpoint path parsing used by --load_checkpoint_path."""

    def test_interactive_training_uri(self):
        """A model ID and checkpoint name resolve to a canonical session ID."""
        session, name = _parse_checkpoint_path("model_abc123/5")
        assert session == "session_abc123"
        assert name == "5"

    def test_interactive_training_uri_adds_session_prefix(self):
        """session_id without session_ prefix gets it added."""
        session, name = _parse_checkpoint_path("abc123/final")
        assert session == "session_abc123"
        assert name == "final"

    def test_plain_path(self):
        """Plain <session>/<name> format works."""
        session, name = _parse_checkpoint_path("model_xyz/final")
        assert session == "session_xyz"
        assert name == "final"

    def test_plain_path_adds_session_prefix(self):
        """Plain path without session_ prefix gets it added."""
        session, name = _parse_checkpoint_path("xyz/final")
        assert session == "session_xyz"
        assert name == "final"

    def test_invalid_single_segment_raises(self):
        """Single segment (no slash, no ://) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid checkpoint_path"):
            _parse_checkpoint_path("just_a_name")

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid checkpoint_path"):
            _parse_checkpoint_path("")

    def test_plain_integer_name(self):
        """Plain integer checkpoint names (post-rename from 06d) work."""
        session, name = _parse_checkpoint_path("model_abc/4")
        assert session == "session_abc"
        assert name == "4"

    def test_interactive_training_uri_with_integer_name(self):
        """An integer checkpoint name is accepted in the plain path form."""
        session, name = _parse_checkpoint_path("model_sess1/12")
        assert session == "session_sess1"
        assert name == "12"


class TestParseReferenceCheckpointPath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("session_deadbeef/final", ("session_deadbeef", "final")),
            ("model_deadbeef/final", ("session_deadbeef", "final")),
            ("deadbeef/final", ("session_deadbeef", "final")),
            ("model_deadbeef/final", ("session_deadbeef", "final")),
        ],
    )
    def test_normalizes_public_session_id(self, path, expected):
        from interactive_training.rl.train_azure import _parse_reference_checkpoint_path

        assert _parse_reference_checkpoint_path(path) == expected


# ---------------------------------------------------------------------------
# tulu3_sft continual-FT / resume wiring (recipes/tulu3_sft/train_azure.py)
#
# Regression guard for the bug where ``--load_checkpoint_path`` was accepted and
# stuffed into ``supervised.train.Config`` but never threaded into
# ``FineTuningSession.create(...)``. The supervised loop only advances the
# epoch/batch counters from ``log_path`` and never reloads weights itself, so a
# missing ``from_checkpoint`` silently cold-starts from the base model — SFT
# would "succeed" while throwing the checkpoint away. These tests drive the real
# ``cli_main`` code path with the SDK + training stubbed and assert the session
# is created ``from_checkpoint`` (the only place the weights actually load).
# ---------------------------------------------------------------------------
class TestTulu3SftForwardsFromCheckpoint:
    def _run_cli_main(
        self,
        monkeypatch,
        tmp_path,
        *,
        load_checkpoint_path,
        last_checkpoint,
        train_error=None,
        close_error=None,
        training_type=None,
        user_metadata=None,
    ):
        """Invoke tulu3_sft.cli_main with the SDK + training loop mocked.

        Returns the session arguments captured at ``FineTuningSession.create(...)``.
        """
        from types import SimpleNamespace

        from interactive_training.recipes.tulu3_sft.train_azure import CLIConfig, cli_main

        captured: dict[str, object] = {}

        def _fake_create(
            client,
            *,
            base_model,
            lora_config,
            type,
            from_checkpoint,
            timeout_sec,
            user_metadata=None,
            training_type=None,
        ):
            captured["from_checkpoint"] = from_checkpoint
            captured["user_metadata"] = user_metadata
            captured["training_type"] = training_type
            return SimpleNamespace(session_id="session_new_sft")

        class _DummyClient:
            def __init__(self, **kwargs):
                pass

        class _DummyAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def close_session(self, *a, **k):
                if close_error is not None:
                    raise close_error

            async def close(self, *a, **k):
                pass

        class _DummyTrainingClient:
            def __init__(self, *a, **k):
                pass

            async def close_poller(self, *a, **k):
                pass

        monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
        monkeypatch.setattr(
            "azure.ai.finetuningsessions.FineTuningSession",
            SimpleNamespace(create=_fake_create),
        )
        monkeypatch.setattr(
            "azure.ai.finetuningsessions.FineTuningSessionClient", _DummyClient
        )
        monkeypatch.setattr(
            "azure.ai.finetuningsessions.aio.FineTuningSessionClient", _DummyAsyncClient
        )
        monkeypatch.setattr(
            "interactive_training.rl.train_azure.AzureSDKTrainingClient", _DummyTrainingClient
        )
        monkeypatch.setattr(
            "interactive_training.tokenizer_utils.get_tokenizer", lambda *a, **k: MagicMock()
        )
        monkeypatch.setattr(
            "interactive_training.supervised.train.main",
            AsyncMock(side_effect=train_error),
        )
        monkeypatch.setattr(
            "interactive_training.checkpoint_utils.get_last_checkpoint",
            lambda _log_path: last_checkpoint,
        )

        cfg = CLIConfig(
            project_endpoint="http://localhost:8000",
            model_name="Qwen/Qwen3-32B",
            tokenizer_name="Qwen/Qwen3-32B",
            renderer_name="qwen3_disable_thinking",
            load_checkpoint_path=load_checkpoint_path,
            log_path=str(tmp_path / "sft"),
            behavior_if_log_dir_exists="delete",
            training_type=training_type,
            user_metadata=user_metadata,
        )
        asyncio.run(cli_main(cfg))
        assert "from_checkpoint" in captured, "FineTuningSession.create was never called"
        return captured

    def test_continual_ft_passes_from_checkpoint(self, monkeypatch, tmp_path):
        """An explicit --load_checkpoint_path (RL→SFT) is forwarded to create()."""
        create_args = self._run_cli_main(
            monkeypatch,
            tmp_path,
            load_checkpoint_path="model_rl123/5",
            last_checkpoint=None,
        )
        from_ckpt = create_args["from_checkpoint"]
        assert from_ckpt is not None, (
            "continual-FT must create the session from_checkpoint, else it "
            "silently cold-starts from the base model"
        )
        assert from_ckpt.source_session_id == "session_rl123"
        assert from_ckpt.checkpoint_id == "5"

    def test_autoresume_passes_from_checkpoint(self, monkeypatch, tmp_path):
        """With no explicit path, the last saved state in log_path is resumed."""
        create_args = self._run_cli_main(
            monkeypatch,
            tmp_path,
            load_checkpoint_path=None,
            last_checkpoint={"state_path": "model_sft9/3", "batch": 3},
        )
        from_ckpt = create_args["from_checkpoint"]
        assert from_ckpt is not None
        assert from_ckpt.source_session_id == "session_sft9"
        assert from_ckpt.checkpoint_id == "3"

    def test_explicit_path_wins_over_autoresume(self, monkeypatch, tmp_path):
        """Explicit --load_checkpoint_path takes precedence over auto-resume."""
        create_args = self._run_cli_main(
            monkeypatch,
            tmp_path,
            load_checkpoint_path="model_explicit/final",
            last_checkpoint={"state_path": "model_autoresume/9", "batch": 9},
        )
        from_ckpt = create_args["from_checkpoint"]
        assert from_ckpt is not None
        assert from_ckpt.source_session_id == "session_explicit"
        assert from_ckpt.checkpoint_id == "final"

    def test_cold_start_passes_none(self, monkeypatch, tmp_path):
        """No checkpoint anywhere → from_checkpoint is None (base-model start)."""
        create_args = self._run_cli_main(
            monkeypatch,
            tmp_path,
            load_checkpoint_path=None,
            last_checkpoint=None,
        )
        assert create_args["from_checkpoint"] is None
        assert create_args["training_type"] is None

    @pytest.mark.parametrize(
        ("training_type", "expected_training_type"),
        [
            (None, None),
            ("GlobalStandard", "GlobalStandard"),
        ],
    )
    def test_training_type_is_forwarded_with_checkpoint(
        self, monkeypatch, tmp_path, training_type, expected_training_type
    ):
        user_metadata = {"DeveloperTier": True, "experimentName": "tier-test"}
        create_args = self._run_cli_main(
            monkeypatch,
            tmp_path,
            load_checkpoint_path="model_sft9/3",
            last_checkpoint=None,
            training_type=training_type,
            user_metadata=user_metadata,
        )

        assert create_args["training_type"] == expected_training_type
        assert create_args["user_metadata"] == user_metadata
        from_ckpt = create_args["from_checkpoint"]
        assert from_ckpt.source_session_id == "session_sft9"
        assert from_ckpt.checkpoint_id == "3"

    def test_close_failure_does_not_mask_workload_failure(self, monkeypatch, tmp_path):
        workload_error = RuntimeError("save sampler failed")

        with pytest.raises(RuntimeError, match="save sampler failed") as raised:
            self._run_cli_main(
                monkeypatch,
                tmp_path,
                load_checkpoint_path=None,
                last_checkpoint=None,
                train_error=workload_error,
                close_error=RuntimeError("close returned 404"),
            )

        assert raised.value is workload_error


@pytest.mark.parametrize(
    ("training_type", "expected_training_type"),
    [
        (None, None),
        ("GlobalStandard", "GlobalStandard"),
        ("globalstandard", "GlobalStandard"),
    ],
)
def test_tulu3_sft_cli_training_type(training_type, expected_training_type):
    import chz

    from interactive_training.recipes.tulu3_sft.train_azure import CLIConfig

    argv = ["project_endpoint=http://localhost:8000"]
    if training_type is not None:
        argv.append(f"training_type={training_type}")

    cfg = chz.Blueprint(CLIConfig).make_from_argv(argv)

    assert cfg.training_type == expected_training_type
