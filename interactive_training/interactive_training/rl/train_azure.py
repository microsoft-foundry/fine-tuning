"""Azure AI Fine-Tuning Sessions SDK adapters for rl/train.py.

Creates AzureSDKTrainingClient and AzureSDKSamplingClient that wrap
``FineTuningSessionClient`` (from ``azure.ai.finetuningsessions.aio``),
so the unmodified do_sync_training / do_async_training loops from
rl/train.py are reused verbatim.

The only thing that changes vs rl/train.py is the ``main()`` function:
instead of creating a ServiceClient, it accepts an already-created
``FineTuningSessionClient`` and ``session_id`` and wraps them in adapters.

Concurrency architecture
------------------------
The SDK's ``FineTuningSessionClient`` handles all async I/O internally:

1. **POST** (async, fast ~50-200 ms): Submits the job via the azure-core async
   pipeline.  Gated by an internal concurrency limit to prevent TCP
   connection storms.

2. **Poll** (async, non-blocking): Short-polls ``GET /request/{id}`` via the
   same async pipeline.  Adaptive backoff (1 s -> 30 s).

3. **Chunking**: Forward-backward batches exceeding 1024 datums or 5 MB are
   automatically split and submitted in parallel via ``asyncio.gather``.

All HTTP goes through the azure-core async pipeline -- no threads, no custom
httpx client, no urllib3 pool exhaustion warnings.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import (
    AdamParams,
    Datum,
    ModelInput,
    SamplingParams,
    TensorData,
)

from interactive_training import checkpoint_utils
from interactive_training.rl.train import (
    Config,
    _config_for_logging,
    _effective_num_groups_to_log,
    compute_effective_end,
    do_async_training,
    do_sync_training,
    do_sync_training_with_stream_minibatch,
    run_evaluations_parallel,
)
from interactive_training.rl.metric_util import RLTestSetEvaluator
from interactive_training.training_types import TrainingType, normalize_training_type
from interactive_training.rl.types import SamplingCheckpoint
from interactive_training.tokenizer_utils import Tokenizer, get_tokenizer
from interactive_training.utils import ml_log
from interactive_training.utils.misc_utils import timed
from interactive_training.utils.trace import scope, trace_init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Micro-future: wraps an awaitable so it has a ``result_async()`` method.
# ---------------------------------------------------------------------------


class _SDKFuture:
    """Wraps an asyncio Task so it has a ``result_async()`` method.

    ``forward_backward_async`` and ``optim_step_async`` must return an
    object with ``result_async()`` -- this is that object.
    """

    def __init__(self, task: asyncio.Future, converter=None):
        self._task = task
        self._converter = converter  # optional callable: sdk_result -> adapter

    async def result_async(self):
        sdk_result = await self._task
        return self._converter(sdk_result) if self._converter else sdk_result


# ---------------------------------------------------------------------------
# Result adapters -- translate SDK result objects to the interface expected
# by rl/train.py (e.g. _training_logprobs_from_fwd_bwd, checkpoint_utils)
# ---------------------------------------------------------------------------


class _FwdBwdResult:
    """Adapts OperationResult (forward_backward) to the duck-type used by train.py.

    ``_training_logprobs_from_fwd_bwd`` accesses
    ``result.loss_fn_outputs[i]["logprobs"].data``.  The Interactive Training server returns
    this in the ``loss_fn_outputs`` key of the raw JSON. Current SDK versions
    declare that field; older versions preserve it through dict-like access.
    """

    def __init__(self, sdk_result):
        self.metrics = sdk_result.metrics or {}

        # Prefer the declared SDK field, while retaining compatibility with
        # releases that preserve loss_fn_outputs only through dict-like access.
        lfo = getattr(sdk_result, "loss_fn_outputs", None)
        if lfo is None and hasattr(sdk_result, "get"):
            lfo = sdk_result.get("loss_fn_outputs")
        if lfo:
            self.loss_fn_outputs = [_LogprobsEntry(entry["logprobs"]["data"]) for entry in lfo]
        else:
            pdl = sdk_result.per_datum_logprobs or []
            self.loss_fn_outputs = [_LogprobsEntry(td.data) for td in pdl]

        logger.debug(
            "forward_backward returned %d loss_fn_outputs entries",
            len(self.loss_fn_outputs),
        )


class _LogprobsEntry:
    """Entry in loss_fn_outputs: supports ``entry["logprobs"].data``."""

    def __init__(self, data: list[float]):
        self.data = data

    def __getitem__(self, key: str):
        if key == "logprobs":
            return self
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key == "logprobs"


class _OptimResult:
    """Adapts OptimStepOperationResult: exposes ``.metrics``."""

    def __init__(self, sdk_result):
        self.metrics = sdk_result.metrics or {}


class _CheckpointResult:
    """Adapts SaveCheckpointOperationResult: exposes ``.path``."""

    def __init__(self, sdk_result):
        self.path = getattr(sdk_result, "path", None) or sdk_result.checkpoint_id


class _SamplerCheckpointResult:
    """Adapts SaveSamplerWeightsOperationResult: exposes ``.path`` and ``.checkpoint_id``."""

    def __init__(self, sdk_result):
        self.checkpoint_id = sdk_result.checkpoint_id
        self.path = sdk_result.checkpoint_id  # checkpoint_utils uses .path


class _SampleResult:
    """Adapts SampleOperationResult to the duck-type used by completers.py."""

    def __init__(self, sdk_result):
        self.sequences = sdk_result.sequences or []


# ---------------------------------------------------------------------------
# AzureSDKSamplingClient -- wraps FineTuningSessionClient for sampling
# ---------------------------------------------------------------------------


class AzureSDKSamplingClient:
    """Sampling client backed by ``FineTuningSessionClient``.

    Implements the async interface expected by ``SessionTokenCompleter``
    (completers.py), evaluators, and KL-penalty code (metrics.py).
    """

    def __init__(
        self,
        client: FineTuningSessionClient,
        session_id: str,
        checkpoint_id: str,
        tokenizer: Tokenizer,
    ):
        self._client = client
        self._session_id = session_id
        self._checkpoint_id = checkpoint_id
        self._sampling_checkpoint = SamplingCheckpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )
        self._tokenizer = tokenizer
        # Alias for compatibility -- checkpoint_utils reads .model_path
        self.model_path = checkpoint_id

    @property
    def sampling_checkpoint(self) -> SamplingCheckpoint:
        """The immutable session and checkpoint binding used for sampling."""
        return self._sampling_checkpoint

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
    ) -> _SampleResult:
        """Sample from the model via FineTuningSessionClient."""
        sp = sampling_params or SamplingParams()

        sdk_result = await self._client.sample(
            self._session_id,
            prompt,
            sp,
            checkpoint_id=self._checkpoint_id,
            num_samples=num_samples,
            topk_prompt_logprobs=0,
        )
        return _SampleResult(sdk_result)

    async def compute_logprobs_async(self, model_input: ModelInput) -> list[float | None]:
        """Compute per-token log-probabilities for the given input."""
        prompt_tokens: list[int] = []
        for chunk in model_input.chunks:
            if not hasattr(chunk, "tokens"):
                raise ValueError(
                    "Prompt logprobs are not supported for image prompts"
                )
            prompt_tokens.extend(chunk.tokens)

        # max_tokens must be >= 1 (the server rejects 0 with "max_tokens must be
        # a positive number"). We only need prompt_logprobs, so the single
        # generated token is discarded.
        sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1)
        sdk_result = await self._client.sample(
            self._session_id,
            prompt_tokens,
            sp,
            checkpoint_id=self._checkpoint_id,
            num_samples=1,
            prompt_logprobs=True,
            topk_prompt_logprobs=0,
        )

        if sdk_result.prompt_logprobs:
            return sdk_result.prompt_logprobs
        return []


# ---------------------------------------------------------------------------
# Helpers for forward_backward_custom_async (surrogate-gradient algorithm)
# ---------------------------------------------------------------------------


# Aux keys that the cookbook uses purely client-side (e.g. as inputs to
# custom-loss repack functions) and that must never reach the server's
# loss-input schema.
_CLIENT_ONLY_AUX_KEYS = frozenset({"mask"})


def _strip_client_only_keys(datum: Datum) -> Datum:
    """Return ``datum`` with cookbook-internal aux keys removed.

    The server's loss kernels validate ``loss_fn_inputs`` keys against a
    schema; cookbook-only carriers (e.g. ``mask``) would be rejected or
    misinterpreted.  Stripping happens here -- inside the wire adapter --
    so the training loop in ``rl/train.py`` stays generic.
    """
    if not any(k in datum.loss_fn_inputs for k in _CLIENT_ONLY_AUX_KEYS):
        return datum
    return Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            k: v for k, v in datum.loss_fn_inputs.items()
            if k not in _CLIENT_ONLY_AUX_KEYS
        },
    )


_CUSTOM_REQUIRED_KEYS = frozenset({"target_tokens"})


def _validate_custom_loss_inputs(data: list[Datum]) -> None:
    """Validate ``data`` for ``forward_backward_custom_async``.

    Each datum's ``loss_fn_inputs`` must contain ``target_tokens``.  Any
    additional keys are permitted and remain accessible inside the user's
    ``loss_fn`` closure -- they are useful as carriers for per-token
    auxiliary data (e.g. sampling logprobs, advantages, reference logprobs,
    masks).  Only ``{target_tokens, weights}`` are forwarded to the server's
    forward / surrogate-backward calls; extras are stripped on the way out
    (the user's closure still sees the original datum).
    """
    for i, datum in enumerate(data):
        if "target_tokens" not in datum.loss_fn_inputs:
            raise ValueError(
                f"forward_backward_custom_async: data[{i}].loss_fn_inputs is "
                "missing required key 'target_tokens'."
            )


def _prepare_forward_data(data: list[Datum]) -> list[Datum]:
    """Build the server-bound forward payload, stripping aux keys.

    The server-side forward pass uses ``loss_fn="cross_entropy"``, whose
    backend kernel reads ``L = sum(-logprobs * weights)`` and therefore
    requires a ``weights`` tensor in every datum.  We rebuild each Datum
    with **only** ``{target_tokens, weights}`` so any extra aux fields the
    caller packed into ``loss_fn_inputs`` (e.g. ``advantages``,
    ``sampling_logprobs``, ``ref_logprobs``) never reach the wire.  Those
    extras are still readable from the *original* data list inside the
    user's ``loss_fn`` closure.  If ``weights`` is absent we inject zeros
    (custom-loss users typically don't supply it on input -- it only
    matters on the surrogate backward).
    """
    forward_data: list[Datum] = []
    for datum in data:
        target_tokens = datum.loss_fn_inputs["target_tokens"]
        weights = datum.loss_fn_inputs.get(
            "weights", TensorData(data=[0.0] * len(target_tokens.data))
        )
        forward_data.append(
            Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": weights,
                },
            )
        )
    return forward_data


def _logprobs_to_tensor(entry) -> torch.Tensor:
    """Build a leaf tensor from a ``_FwdBwdResult.loss_fn_outputs[i]`` entry.

    The tensor is a leaf with ``requires_grad=True`` so the user's loss can
    backprop into it; the resulting ``.grad`` is the surrogate-gradient
    direction g = dC/dlogprobs.
    """
    data = entry["logprobs"].data
    return torch.tensor(data, dtype=torch.float32, requires_grad=True)


def _build_surrogate_datum(original: Datum, g: torch.Tensor) -> Datum:
    """Build a surrogate Datum carrying weights = -g for the server-side CE pass.

    ``g`` is the gradient of the user's loss w.r.t. the per-token log-probs
    (``g_t = dC/dlogprob_t``), obtained from autograd on a leaf logprobs
    tensor.  The Interactive Training backend's ``cross_entropy`` loss is
    ``L = sum_t -logprob_t * weight_t``, so setting ``weight_t = -g_t`` yields
    ``dL/dtheta = sum_t g_t * dlogprob_t/dtheta = dC/dtheta`` exactly.

    Only ``{target_tokens, weights}`` are forwarded; any other keys in the
    caller's ``loss_fn_inputs`` (advantages, logprobs, masks, ...) are
    intentionally stripped so the backend's CE kernel sees a clean payload.
    """
    target_tokens = original.loss_fn_inputs["target_tokens"]
    expected_len = len(target_tokens.data)
    g_flat = g.reshape(-1)
    if g_flat.numel() != expected_len:
        raise ValueError(
            "forward_backward_custom_async: loss_fn produced a gradient of "
            f"length {g_flat.numel()} for a target_tokens sequence of length "
            f"{expected_len}; these must match per datum."
        )
    weights = TensorData(data=(-g_flat).detach().cpu().tolist())
    return Datum(
        model_input=original.model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


def _surrogate_weight_stats(surrogate_data: list[Datum]) -> dict[str, float]:
    """Compute diagnostic stats on the surrogate ``weights`` vector.

    The surrogate weights ``w = -dC/dlogprobs`` *are* the differential
    between the custom loss and vanilla cross-entropy: under plain CE the
    backend would see ``w = 1`` everywhere (or whatever the user supplied).
    Three batch-level numbers make it easy to confirm at-a-glance that the
    custom-loss path is actually doing something different from CE:

      ``surrogate/weights_l2``           magnitude (step strength in CE units)
      ``surrogate/weights_pos_frac``     fraction with w > 0; CE is 1.0,
                                         advantage-driven losses are ~0.5
      ``surrogate/weights_cos_uniform``  cosine vs the all-ones vector; CE
                                         is exactly 1.0, differential losses
                                         are <<1 (can be negative)

    These flow through the standard ``result.metrics`` pipeline; no extra
    wire traffic, one tensor op on the (already small) concatenated weight
    vector.  Returns an empty dict on empty batches to keep the contract
    boring.
    """
    if not surrogate_data:
        return {}
    flat: list[float] = []
    for d in surrogate_data:
        flat.extend(d.loss_fn_inputs["weights"].data)
    if not flat:
        return {}
    with torch.no_grad():
        w = torch.tensor(flat, dtype=torch.float32)
        n = w.numel()
        l2 = float(w.norm().item())
        pos_frac = float((w > 0).float().mean().item())
        # cos(w, [1,...,1]) = sum(w) / (||w|| * sqrt(n))
        denom = l2 * (n ** 0.5)
        cos_uniform = float(w.sum().item() / denom) if denom > 0.0 else 0.0
    return {
        "surrogate/weights_l2": l2,
        "surrogate/weights_pos_frac": pos_frac,
        "surrogate/weights_cos_uniform": cos_uniform,
    }


# ---------------------------------------------------------------------------
# AzureSDKTrainingClient -- wraps FineTuningSessionClient for training
# ---------------------------------------------------------------------------


class AzureSDKTrainingClient:
    """Training client backed by ``FineTuningSessionClient``.

    Implements the async interface expected by ``rl/train.py`` and
    ``checkpoint_utils.py``.  Each async method delegates to the SDK's
    ``FineTuningSessionClient`` which handles POST, poll, chunking, retry,
    and auth internally. Optional polling controls let recipes tune latency
    without changing the default behavior for other training clients.
    """

    def __init__(
        self,
        client: FineTuningSessionClient,
        session_id: str,
        tokenizer: Tokenizer,
        forward_backward_chunk_wave_size: int | None = None,
        forward_poll_interval_sec: float | None = None,
        optim_poll_interval_sec: float | None = None,
        defer_optim_poll_until_forward_complete: bool = False,
    ):
        self._client = client
        self._session_id = session_id
        self._tokenizer = tokenizer
        self._forward_backward_chunk_wave_size = forward_backward_chunk_wave_size
        self._step: int = 0
        self._forward_poll_interval_sec = forward_poll_interval_sec
        self._optim_poll_interval_sec = optim_poll_interval_sec
        self._defer_optim_poll_until_forward_complete = (
            defer_optim_poll_until_forward_complete
        )
        self._unpaired_forward_tasks: deque[asyncio.Future] = deque()

    def get_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def set_operation_context(self, label: str):
        """Set an optional correlation label for SDK operation timeline logs."""
        from azure.ai.finetuningsessions.aio._patch import set_operation_label

        return set_operation_label(label)

    def reset_operation_context(self, token) -> None:
        """Restore the SDK operation timeline correlation context."""
        from azure.ai.finetuningsessions.aio._patch import reset_operation_label

        reset_operation_label(token)

    async def forward_backward_async(
        self,
        batch: list[Datum],
        loss_fn: str | None = None,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> _SDKFuture:
        """Enqueue a forward+backward pass; returns a future with ``.result_async()``.

        The SDK awaits all POSTs (including chunking) before returning,
        guaranteeing UUID v7 ordering vs any subsequent optim_step call.
        """
        _loss_fn = loss_fn or "importance_sampling"
        logger.debug("forward_backward_async: sending %d datums", len(batch))

        # Strip client-side-only aux keys before the wire call.  ``mask`` is
        # consumed by client-side custom-loss repackers (see
        # CustomLossTrainingClient) and is not part of any server-side loss
        # contract.  Subclasses that route through forward_backward_custom_async
        # override this method and never reach this line, so the mask is still
        # available to their loss closures.
        wire_batch = [_strip_client_only_keys(d) for d in batch]

        kwargs: dict[str, Any] = {
            "loss_fn": _loss_fn,
            "loss_fn_config": loss_fn_config,
        }
        if self._forward_backward_chunk_wave_size is not None:
            kwargs["max_chunks_per_wave"] = self._forward_backward_chunk_wave_size
        if self._forward_poll_interval_sec is not None:
            kwargs["poll_min_sec"] = self._forward_poll_interval_sec
            kwargs["poll_max_sec"] = self._forward_poll_interval_sec
        task = await self._client.forward_backward_async(
            self._session_id,
            wire_batch,
            **kwargs,
        )
        if self._defer_optim_poll_until_forward_complete:
            self._unpaired_forward_tasks.append(task)
        return _SDKFuture(task, _FwdBwdResult)

    async def forward_async(
        self,
        batch: list[Datum],
        loss_fn: str | None = None,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> _SDKFuture:
        """Enqueue a forward-only pass (no gradients); returns a future with ``.result_async()``.

        The SDK awaits all POSTs (including chunking) before returning,
        guaranteeing UUID v7 ordering vs any subsequent request.
        """
        _loss_fn = loss_fn or "cross_entropy"
        logger.debug("forward_async: sending %d datums", len(batch))

        task = await self._client.forward_async(
            self._session_id,
            batch,
            loss_fn=_loss_fn,
            loss_fn_config=loss_fn_config,
        )
        return _SDKFuture(task, _FwdBwdResult)

    async def forward_backward_custom_async(
        self,
        data: list[Datum],
        loss_fn: Callable[
            [list[Datum], list[torch.Tensor]],
            tuple[torch.Tensor, dict[str, float]],
        ],
    ) -> _SDKFuture:
        """Forward-backward with a Python-side custom loss (surrogate-gradient).

        Implements a two-pass surrogate-gradient algorithm:

        1. **Forward** pass (server-side, ``loss_fn="cross_entropy"``) to obtain
           per-token log-probs ``ell``.
        2. **Local autograd**: wrap each ``ell`` as a leaf tensor with
           ``requires_grad=True``, call the user's ``loss_fn(data, logprobs)``
           to get ``(loss, custom_metrics)``, call ``loss.backward()`` to get
           ``g_i = dC/dell_i``.
        3. **Forward-backward** pass (server-side, ``loss_fn="cross_entropy"``)
           with a surrogate ``Datum`` per example whose ``loss_fn_inputs`` are
           trimmed to ``{target_tokens, weights = -g}``.  Because the backend
           CE is ``L = sum -ell * w``, this yields ``dL/dtheta = dC/dtheta``
           exactly -- ``C`` is computed faithfully even though it never
           crosses the wire.

        ``custom_metrics`` returned by the user's ``loss_fn`` are merged into
        the resulting ``_FwdBwdResult.metrics`` alongside backend metrics
        (e.g. ``grad_norm``).

        Cost: 1.5x the FLOPs of a single ``forward_backward_async`` (one extra
        forward pass).  Use the built-in ``forward_backward_async`` loss
        functions when they suffice.

        Memory note: step 2 below runs autograd on the *entire substep
        batch* in host RAM.  Leaf storage is
        ``(groups_per_batch / num_substeps) * group_size * avg_seq_len``
        float32s (~10 MB at 2048 datums * 1200 tokens; ~1 GB at 8192
        datums * 32k tokens, with another 3-5x live for saved
        intermediates during ``backward()``).  If you OOM here, raise
        ``num_substeps`` -- it shrinks this call proportionally without
        changing the math (and is the same knob used for GPU-side
        memory).  Internal chunking inside this function is intentionally
        not provided because it would silently break ``loss_fn``\\ s with
        cross-datum terms (batch-level normalizers, global denominators,
        etc.).
        """
        logger.debug("forward_backward_custom_async: %d datums", len(data))

        # Fail-fast validation: target_tokens is mandatory.  Extra
        # loss_fn_inputs keys are allowed (the user's Python loss_fn sees
        # them); only {target_tokens, weights} are forwarded to the server
        # (see _prepare_forward_data and _build_surrogate_datum).
        _validate_custom_loss_inputs(data)

        # Step 1: server-side forward to obtain logprobs.  The forward call
        # uses cross_entropy, which requires a weights tensor; inject zeros
        # for any datum that doesn't already supply one.
        forward_data = _prepare_forward_data(data)
        fwd_future = await self.forward_async(forward_data, loss_fn="cross_entropy")
        fwd_result = await fwd_future.result_async()

        # Step 2: local autograd on the user's loss.  Operates on the
        # full substep batch (no internal chunking -- see docstring).
        # Cheap in FLOPs since the leaves are 1D per-token logprobs and
        # loss_fn is elementwise; the memory footprint can become the
        # binding constraint at long contexts -- see docstring memory
        # note and raise ``num_substeps`` if you OOM here.
        logprobs_tensors = [
            _logprobs_to_tensor(entry) for entry in fwd_result.loss_fn_outputs
        ]
        loss, custom_metrics = loss_fn(data, logprobs_tensors)
        loss.backward()

        # Step 3: build surrogate batch with weights = -grad.  Fail fast if
        # the user's loss did not depend on any datum's logprobs -- silently
        # substituting zeros would mask indexing / typo bugs in loss_fn.
        surrogate_data: list[Datum] = []
        for i, (original, tensor) in enumerate(
            zip(data, logprobs_tensors, strict=True)
        ):
            if tensor.grad is None:
                raise ValueError(
                    f"forward_backward_custom_async: data[{i}]'s logprobs "
                    "tensor has no gradient. loss_fn must produce a scalar "
                    "loss whose autograd graph depends on every datum's "
                    "logprobs."
                )
            surrogate_data.append(_build_surrogate_datum(original, tensor.grad))

        # Step 3.5: diagnostic stats on the surrogate weights.  Cheap, generic
        # across all custom losses; merged in below with user metrics winning
        # on any name collision.  See ``_surrogate_weight_stats``.
        surrogate_metrics = _surrogate_weight_stats(surrogate_data)

        # Step 4: dispatch the server-side forward_backward; merge metrics.
        # Forward the bounded-wave knob just like forward_backward_async() does,
        # so custom-loss runs also cap chunks-per-wave on this gradient pass
        # before the optimizer step (not just the built-in loss path).
        fb_kwargs: dict[str, Any] = {"loss_fn": "cross_entropy"}
        if self._forward_backward_chunk_wave_size is not None:
            fb_kwargs["max_chunks_per_wave"] = self._forward_backward_chunk_wave_size
        if self._forward_poll_interval_sec is not None:
            fb_kwargs["poll_min_sec"] = self._forward_poll_interval_sec
            fb_kwargs["poll_max_sec"] = self._forward_poll_interval_sec
        task = await self._client.forward_backward_async(
            self._session_id,
            surrogate_data,
            **fb_kwargs,
        )
        if self._defer_optim_poll_until_forward_complete:
            self._unpaired_forward_tasks.append(task)

        def _convert(sdk_result):
            result = _FwdBwdResult(sdk_result)
            # Merge order: backend, then adapter diagnostics, then user.
            # User wins on any collision so a custom loss can override.
            result.metrics = {
                **result.metrics,
                **surrogate_metrics,
                **custom_metrics,
            }
            return result

        return _SDKFuture(task, _convert)

    async def optim_step_async(
        self, adam_params: AdamParams | None = None
    ) -> _SDKFuture:
        """Enqueue an optimizer step; returns a future with ``.result_async()``.

        The SDK awaits the POST before returning, guaranteeing its UUID v7
        is later than all preceding forward_backward_async calls. When deferred
        optimizer polling is configured, the POST still happens immediately but
        polling waits for the corresponding forward task to complete.
        """
        if adam_params is None:
            adam_params = AdamParams(
                learning_rate=3e-4, beta1=0.9, beta2=0.95, eps=1e-8
            )
        self._step += 1

        poll_kwargs: dict[str, float] = {}
        if self._optim_poll_interval_sec is not None:
            poll_kwargs = {
                "poll_min_sec": self._optim_poll_interval_sec,
                "poll_max_sec": self._optim_poll_interval_sec,
            }

        if self._defer_optim_poll_until_forward_complete:
            if not self._unpaired_forward_tasks:
                raise RuntimeError(
                    "optim_step_async requires a preceding forward_backward_async "
                    "when deferred optimizer polling is enabled"
                )
            forward_task = self._unpaired_forward_tasks.popleft()
            pending = await self._client.optim_step_post(
                self._session_id, adam_params
            )

            async def _poll_after_forward():
                await forward_task
                return await pending.poll_result(**poll_kwargs)

            task = asyncio.create_task(
                _poll_after_forward(),
                name="optim_poll_after_forward",
            )
        else:
            task = await self._client.optim_step_async(
                self._session_id,
                adam_params,
                **poll_kwargs,
            )
        return _SDKFuture(task, _OptimResult)

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None,
        *, step_number: int | None = None, metrics: dict[str, Any] | None = None,
    ) -> _SDKFuture:
        """Save training state (LoRA weights + optimizer state).

        The SDK awaits the POST before returning, guaranteeing its UUID v7
        is later than all preceding requests.
        """
        task = await self._client.save_weights_async(
            self._session_id, name, step_number=step_number, metrics=metrics
        )
        return _SDKFuture(task, _CheckpointResult)

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> _SDKFuture:
        """Persist sampler checkpoint via the training engine.

        Called at save_every boundaries via checkpoint_utils.save_checkpoint_async.
        """
        task = await self._client.save_weights_for_sampler_async(
            self._session_id, name,
        )
        return _SDKFuture(task, _SamplerCheckpointResult)

    async def save_weights_and_get_sampling_client_async(
        self, name: str | None = None,
    ) -> "AzureSDKSamplingClient":
        """Ephemeral weight sync — not persisted.

        Returns a sampling client pointed at the synced weights.
        """
        checkpoint_name = name or f"step{self._step}"
        task = await self._client.save_weights_and_get_sampling_client_async(
            self._session_id, checkpoint_name,
        )
        future = _SDKFuture(task, _SamplerCheckpointResult)
        result = await future.result_async()
        return AzureSDKSamplingClient(
            self._client, self._session_id, result.checkpoint_id, self._tokenizer
        )

    def create_sampling_client(
        self, model_path: str | None
    ) -> "AzureSDKSamplingClient":
        """Create a sampling client from an existing checkpoint path."""
        return AzureSDKSamplingClient(
            self._client, self._session_id, model_path or "", self._tokenizer
        )

    async def close_poller(self) -> None:
        """No-op -- kept for API compatibility."""
        pass


def _parse_reference_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Parse a reference/teacher checkpoint into ``(session_id, checkpoint_name)``.

    Accepts the forms written to ``checkpoints.jsonl`` and passed by recipes:
      - ``<session_id>/<name>`` (the ``state_path`` URI, e.g. the
        checkpoint an SFT run saves as ``final``)
      - plain ``<session_id>/<checkpoint_name>``

        Raises ``ValueError`` on any other shape.
    """
    def _canonical_session_id(session_id: str) -> str:
        if session_id.startswith("session_"):
            return session_id
        return f"session_{session_id.removeprefix('model_')}"

    if "://" in checkpoint_path:
        stripped = checkpoint_path.split("://", 1)[1]
        parts = [p for p in stripped.split("/") if p]
        if len(parts) >= 2:
            session, name = parts[0], parts[-1]
            return _canonical_session_id(session), name

    parts = checkpoint_path.split("/", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return _canonical_session_id(parts[0]), parts[1]

    raise ValueError(
        "kl_reference_config.load_checkpoint_path must be "
        "'<session_id>/<name>' or '<session_id>/<checkpoint_name>', "
        f"got: {checkpoint_path!r}"
    )


async def _create_kl_reference_client(
    client: FineTuningSessionClient,
    cfg: Config,
    tokenizer: Tokenizer,
    *,
    training_type: TrainingType | None = None,
    on_session_created: Callable[[str], None] | None = None,
) -> tuple["AzureSDKSamplingClient", str]:
    """Create a reference (teacher) sampling client for the KL penalty.

    Reference-KL (and distillation) needs log-probs from a model that is
    *separate* from the one being trained. The SDK supports this via an
    independent session:

    1. ``create_session`` for the reference base model (optionally resumed from a
       checkpoint), then
    2. snapshot its initial (frozen) weights into a sampler checkpoint and return
       a sampling client pointed at them.

    When ``cfg.kl_reference_config`` is None the student's own base model
    (``cfg.model_name``) is used as the reference — handy for a smoke test where
    the teacher does not need to be good at any task.

    ``training_type`` applies the recipe's tier to the reference session too.
    None leaves the server default unchanged.

    Returns ``(sampling_client, reference_session_id)`` so the caller can tear
    the reference session down.
    """
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    ref_config = cfg.kl_reference_config
    ref_base = ref_config.base_model if ref_config is not None else cfg.model_name
    ref_ckpt_path = (
        ref_config.load_checkpoint_path if ref_config is not None else None
    )

    from_checkpoint: FromCheckpoint | None = None
    if ref_ckpt_path:
        source_session_id, checkpoint_id = _parse_reference_checkpoint_path(ref_ckpt_path)
        from_checkpoint = FromCheckpoint(
            source_session_id=source_session_id, checkpoint_id=checkpoint_id
        )

    logger.info(
        "Creating KL reference session: base_model=%s checkpoint=%s",
        ref_base,
        ref_ckpt_path,
    )
    ref_session_id = await client.create_session(
        base_model=ref_base,
        lora_config=LoRAConfig(rank=cfg.lora_rank),
        type="training",
        from_checkpoint=from_checkpoint,
        timeout_sec=600.0,
        training_type=normalize_training_type(training_type),
    )
    # Snapshot the reference's (frozen) weights into a sampler checkpoint so we
    # can compute teacher log-probs against them. If the snapshot fails the
    # caller never receives ``ref_session_id`` (so its ``finally`` cannot close
    # it), so tear the just-created reference session down here before
    # propagating — otherwise the second model leaks on the service.
    try:
        if on_session_created is not None:
            on_session_created(ref_session_id)
        ref_training_client = AzureSDKTrainingClient(client, ref_session_id, tokenizer)
        kl_reference_client = (
            await ref_training_client.save_weights_and_get_sampling_client_async(
                "reference"
            )
        )
    except BaseException:
        logger.warning(
            "Reference sampler snapshot failed; closing reference session %s",
            ref_session_id,
        )
        try:
            await client.close_session(ref_session_id)
        except Exception as close_exc:
            logger.warning("close_session(reference) cleanup failed: %s", close_exc)
        raise
    logger.info("KL reference session ready: session_id=%s", ref_session_id)
    return kl_reference_client, ref_session_id


# ---------------------------------------------------------------------------
# main() -- mirrors rl/train.py main() but takes FineTuningSessionClient
# ---------------------------------------------------------------------------


@scope
async def main(
    cfg: Config,
    client: FineTuningSessionClient,
    session_id: str,
    *,
    training_type: TrainingType | None = None,
    prepared_datasets: tuple[Any, Any] | None = None,
    forward_backward_chunk_wave_size: int | None = None,
    on_reference_session_created: Callable[[str], None] | None = None,
    training_client_factory: Callable[
        [FineTuningSessionClient, str, Any], "AzureSDKTrainingClient"
    ]
    | None = None,
) -> None:
    """Training loop using the Azure AI Fine-Tuning Sessions SDK backend.

    Identical to ``rl/train.py main()`` except:
      - Accepts ``FineTuningSessionClient`` + ``session_id`` instead of base_url
      - Wraps them in ``AzureSDKTrainingClient`` / ``AzureSDKSamplingClient``

    Args:
        training_type: Optional tier for newly created teacher/KL-reference
            sessions. None leaves the server default unchanged; the caller
            selects the primary session's tier when creating that session.
        forward_backward_chunk_wave_size: Maximum number of SDK chunks to
            register before waiting for that wave to complete. Applies to the
            default training client and preserves one optimizer step.
        training_client_factory: Optional factory ``(client, session_id, tokenizer)
            -> AzureSDKTrainingClient``.  Recipes can pass a subclass (e.g. one
            that routes ``forward_backward_async`` through
            ``forward_backward_custom_async`` for a Python-side loss).  Default:
            the plain :class:`AzureSDKTrainingClient` constructor.
    """
    e2e_start_time = time.time()

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=_config_for_logging(cfg),
        wandb_name=cfg.wandb_name,
    )

    if cfg.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing enabled; saving to {trace_events_path}")
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    # Build tokenizer
    tokenizer = (
        get_tokenizer(cfg.tokenizer_name)
        if cfg.tokenizer_name
        else get_tokenizer(cfg.model_name)
    )

    training_client = (
        training_client_factory(client, session_id, tokenizer)
        if training_client_factory is not None
        else AzureSDKTrainingClient(
            client,
            session_id,
            tokenizer,
            forward_backward_chunk_wave_size=forward_backward_chunk_wave_size,
        )
    )

    # Reference (teacher) session for the KL penalty. Created only when
    # kl_penalty_coef > 0; torn down in the finally below. Declared out here so
    # it is visible to the finally even if session creation fails.
    kl_reference_session_id: str | None = None

    try:
        kl_reference_client = None
        if cfg.kl_penalty_coef > 0:
            kl_reference_client, kl_reference_session_id = (
                await _create_kl_reference_client(
                    client, cfg, tokenizer, training_type=training_type,
                    on_session_created=on_reference_session_created,
                )
            )

        if prepared_datasets is None:
            dataset, maybe_test_dataset = await cfg.dataset_builder()
        else:
            dataset, maybe_test_dataset = prepared_datasets
        evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
        if maybe_test_dataset is not None:
            evaluators.append(
                RLTestSetEvaluator(
                    maybe_test_dataset,
                    max_tokens=cfg.max_tokens,
                    num_groups_to_log=_effective_num_groups_to_log(cfg),
                    sample_timeout_sec=cfg.sample_timeout_sec,
                    max_retries_per_trajectory=cfg.max_retries_per_trajectory,
                    max_extra_trajectory_attempts_per_group=(
                        cfg.max_extra_trajectory_attempts_per_group
                    ),
                    observer=cfg.validation_observer,
                    response_format=cfg.response_format,
                    require_full_validation=cfg.require_full_validation,
                )
            )

        num_batches = len(dataset)
        effective_end = compute_effective_end(
            start_batch=start_batch,
            num_batches=num_batches,
            max_steps=cfg.max_steps,
        )
        if cfg.max_steps is not None:
            logger.info(
                f"Will train on {effective_end - start_batch} batches "
                f"(max_steps={cfg.max_steps}, dataset has {num_batches})"
            )
        else:
            logger.info(f"Will train on {num_batches} batches")

        deadline: float | None = (
            e2e_start_time + cfg.max_wall_clock_seconds
            if cfg.max_wall_clock_seconds is not None
            else None
        )
        if deadline is not None:
            logger.info(
                f"Wall-clock budget: {cfg.max_wall_clock_seconds:.0f}s "
                f"(deadline at +{cfg.max_wall_clock_seconds:.0f}s from start)"
            )

        if cfg.async_config is not None:
            training_func = do_async_training
        elif cfg.stream_minibatch_config is not None:
            training_func = do_sync_training_with_stream_minibatch
        else:
            training_func = do_sync_training

        training_kwargs: dict[str, Any] = {
            "start_batch": start_batch,
            "end_batch": effective_end,
            "num_batches": num_batches,
            "cfg": cfg,
            "training_client": training_client,
            "kl_reference_client": kl_reference_client,
            "evaluators": evaluators,
            "dataset": dataset,
            "ml_logger": ml_logger,
            "tokenizer": tokenizer,
            "deadline": deadline,
        }
        if training_func is do_sync_training and resume_info is not None:
            training_kwargs["resume_prompt_cursor"] = resume_info.get(
                "prompt_cursor"
            )
        final_loop_state: dict[str, Any] = {}
        if training_func is do_async_training and resume_info is not None:
            training_kwargs["resume_group_ordinal"] = resume_info.get("async_group_ordinal")
        if training_func in (do_sync_training, do_async_training):
            training_kwargs["final_loop_state"] = final_loop_state
        actual_next_batch, last_completed_step = await training_func(
            **training_kwargs
        )

        if actual_next_batch > start_batch:
            final_checkpoint_kwargs: dict[str, Any] = {}
            if last_completed_step is not None:
                final_checkpoint_kwargs["step_number"] = last_completed_step
            final_paths = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name="final",
                log_path=cfg.log_path,
                kind="both",
                loop_state={"batch": actual_next_batch, **final_loop_state},
                ttl_seconds=cfg.ttl_seconds,
                **final_checkpoint_kwargs,
            )
        else:
            logger.info("Training was already complete; nothing to do")
            final_paths = None

        if (
            final_paths is not None
            and last_completed_step is not None
            and len(evaluators) > 0
            and cfg.eval_every > 0
        ):
            sampling_client = training_client.create_sampling_client(
                final_paths["sampler_path"]
            )
            metrics: dict[str, Any] = {}
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, actual_next_batch
                )
                metrics.update(eval_metrics)
            ml_logger.log_metrics(metrics, step=actual_next_batch)

    finally:
        # Close the KL reference (teacher) session first, if one was created.
        if kl_reference_session_id is not None:
            logger.info("Closing KL reference session ...")
            try:
                await client.close_session(kl_reference_session_id)
            except Exception as exc:
                logger.warning("close_session(reference) failed: %s", exc)
        # Close the session to release GPU resources on the service
        logger.info("Closing session ...")
        try:
            await client.close_session(session_id)
        except Exception as exc:
            logger.warning("close_session() failed: %s", exc)
        await client.close()
        logger.info("Session closed.")

    e2e_seconds = time.time() - e2e_start_time
    h, rem = divmod(int(e2e_seconds), 3600)
    m, s = divmod(rem, 60)
    logger.info(
        f"Total wall-clock time: {h}h {m}m {s}s ({e2e_seconds / 60:.1f} min)"
    )
    try:
        timing_path = os.path.join(cfg.log_path, "timing.json")
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "e2e_wall_clock_seconds": round(e2e_seconds, 1),
                    "e2e_wall_clock_minutes": round(e2e_seconds / 60, 1),
                    "e2e_wall_clock_formatted": f"{h}h {m}m {s}s",
                },
                f,
                indent=4,
            )
    except Exception:
        pass

    ml_logger.close()
    logger.info("Training completed successfully")
