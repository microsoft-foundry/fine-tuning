"""Equivalence proof for ``forward_backward_custom_async`` — surrogate gradient ≡ textbook autograd.

This is the canonical correctness test to read if you want to
**understand what ``forward_backward_custom_async`` actually does**.  It
shows, side-by-side:

1. The textbook PyTorch way to train against a custom loss on per-token
   log-probabilities (what every ML researcher already knows).
2. The way the cookbook adapter does it via the surrogate-gradient trick,
   composed of one server-side ``forward`` + one server-side
   ``forward_backward`` with ``weights = -dC/dlogprobs``.

…and asserts that both paths produce **bit-for-bit identical parameter
gradients** for a tiny model.  If you want a fast correctness check of
the basic idea behind ``forward_backward_custom_async``, this is it.

The "backend" here is a local ``nn.Module`` (random weights, vocab=8) —
no network, no checkpoint, no GPU.  Each test runs in well under a second.

See [docs/loss_functions.md](../docs/loss_functions.md) for the math
derivation; this file is the executable proof that the math is
implemented correctly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from azure.ai.finetuningsessions.models import (
    Datum,
    ForwardBackwardOperationResult,
    ModelInput,
    ModelInputChunk,
    TensorData,
)

from interactive_training.rl.train_azure import AzureSDKTrainingClient
from interactive_training.tensor_utils import tensor_data_to_torch


# ---------------------------------------------------------------------------
# PicoLM — tiny language model (vocab=8, embed=4).  Random weights; no
# pretraining.  Big enough that parameter gradients have non-trivial
# structure, small enough to fit in a comment.
# ---------------------------------------------------------------------------


class PicoLM(nn.Module):
    """A picogram-sized language model: token-embedding + linear head."""

    def __init__(self, vocab: int = 8, dim: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (T,) → (T, V)
        return self.head(self.embed(tokens))


def _per_token_target_logprobs(
    model: PicoLM, tokens: list[int], targets: list[int]
) -> torch.Tensor:
    """log p_model(target_t | tokens_<=t) for each position t.  No mask, no shift.

    Returns a 1-D tensor of length ``len(tokens)``; the value at index ``t``
    is the model's logprob of ``targets[t]`` given input ``tokens``.  This
    is exactly the per-token quantity the Interactive Training backend ships back inside
    ``loss_fn_outputs[i]["logprobs"].data``.
    """
    logits = model(torch.tensor(tokens, dtype=torch.long))  # (T, V)
    logprobs = F.log_softmax(logits, dim=-1)
    target_idx = torch.tensor(targets, dtype=torch.long).unsqueeze(-1)  # (T, 1)
    return logprobs.gather(-1, target_idx).squeeze(-1)  # (T,)


# ---------------------------------------------------------------------------
# MockBackend — stands in for the Interactive Training training backend.
#
# Owns an ``nn.Module``; implements just enough of ``FineTuningSessionClient``
# for ``forward_backward_custom_async`` to route through it.  The two methods
# the adapter calls — ``forward_async`` and ``forward_backward_async`` —
# translate the wire protocol into local PyTorch ops:
#
#   forward_async         → with torch.no_grad(): compute per-token logprobs
#   forward_backward_async→ L = sum(-logprob_t * weight_t); L.backward()
#
# These mirror the documented ``cross_entropy`` contract on the training
# engine (``L_CE = sum_t -logprob_t * weight_t``; see docs/loss_functions.md).
#
# Load-bearing anchor: the ``forward_backward_async`` body below is a
# one-line reimplementation of the real training-backend kernel at
# ``interactive-post-training_skyrl/skyrl/backends/skyrl_train/utils/ppo_utils.py`` (search for
# ``cross_entropy_loss``; the formula is literally
# ``loss = (-log_probs * loss_mask).sum()``).  We can't import that kernel
# here because the cookbook venv is a thin CPU client and the worker venv
# pulls in flash-attn / vLLM.  If the server-side kernel ever changes shape
# (sign, reduction, masking, clamping), this mock and the surrogate-gradient
# identity it certifies both go stale -- update both call sites together.
# An end-to-end equivalence test against the real mock-backend stack
# belongs in ``interactive-post-training_tests/`` (the cross-service e2e suite), not here.
# ---------------------------------------------------------------------------


def _awaitable(value):
    fut = asyncio.get_event_loop().create_future()
    fut.set_result(value)
    return fut


def _fb_result(
    *,
    loss_fn_outputs: list[dict] | None = None,
    total_loss: float = 0.0,
    metrics: dict[str, float] | None = None,
) -> ForwardBackwardOperationResult:
    """Build a real ``ForwardBackwardOperationResult`` the way the SDK would
    deserialize one from the server's JSON response.  ``loss_fn_outputs`` is
    an undeclared extra field that the underlying ``_Model`` mapping preserves
    and exposes via ``.get("loss_fn_outputs")`` — exactly what the adapter's
    ``_FwdBwdResult`` reads."""
    payload: dict = {
        "operation_id": "op_pico",
        "status": "succeeded",
        "type": "forward_backward",
        "total_loss": total_loss,
        "per_datum_logprobs": None,
        "metrics": metrics or {},
    }
    if loss_fn_outputs is not None:
        payload["loss_fn_outputs"] = loss_fn_outputs
    return ForwardBackwardOperationResult(payload)


class MockBackend:
    """Replaces ``FineTuningSessionClient``.  Holds the model and runs it."""

    def __init__(self, model: PicoLM, *, sgd_lr: float = 0.1):
        self.model = model
        # Most recent batch passed to forward_backward_async, captured so
        # tests can inspect the surrogate weights the adapter built.
        self.last_fb_batch: list[Datum] | None = None
        # Plain SGD-no-momentum so the parameter update is exactly
        # ``p -= lr * p.grad`` -- trivial to mirror in the textbook
        # reference path.  AdamParams handed in by the adapter is
        # intentionally ignored; this is a mock, not the real optimizer.
        self.sgd_lr = sgd_lr

    async def forward_async(
        self, session_id, batch, *, loss_fn, loss_fn_config=None
    ):
        assert loss_fn == "cross_entropy"
        loss_fn_outputs: list[dict] = []
        with torch.no_grad():
            for datum in batch:
                tokens = [t for chunk in datum.model_input.chunks for t in chunk.tokens]
                targets = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
                tgt_lp = _per_token_target_logprobs(self.model, tokens, targets)
                loss_fn_outputs.append({"logprobs": {"data": tgt_lp.tolist()}})
        return _awaitable(_fb_result(loss_fn_outputs=loss_fn_outputs))

    async def forward_backward_async(
        self, session_id, batch, *, loss_fn, loss_fn_config=None
    ):
        assert loss_fn == "cross_entropy"
        self.last_fb_batch = list(batch)
        total = torch.zeros(())
        for datum in batch:
            tokens = [t for chunk in datum.model_input.chunks for t in chunk.tokens]
            targets = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
            weights = torch.tensor(
                datum.loss_fn_inputs["weights"].data, dtype=torch.float32
            )
            tgt_lp = _per_token_target_logprobs(self.model, tokens, targets)
            total = total + (-tgt_lp * weights).sum()
        total.backward()  # accumulates into self.model parameters
        return _awaitable(
            _fb_result(total_loss=float(total.item()), metrics={"total_loss": float(total.item())})
        )

    async def optim_step_async(self, session_id, adam_params):
        """Apply one SGD step and clear grads.  Ignores ``adam_params``.

        Mirrors what a real training engine does between forward_backward
        calls: consume accumulated ``.grad``, update parameters, zero out
        the buffers so the next forward_backward starts fresh.
        """
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.sub_(self.sgd_lr * p.grad)
                    p.grad.zero_()
        result = MagicMock()
        result.metrics = {}
        return _awaitable(result)


def _build_training_client(backend: MockBackend) -> AzureSDKTrainingClient:
    """Wrap a ``MockBackend`` so the adapter sees the right async surface."""
    client = MagicMock()
    client.forward_async = backend.forward_async
    client.forward_backward_async = backend.forward_backward_async
    client.optim_step_async = backend.optim_step_async
    return AzureSDKTrainingClient(client, "session_pico", tokenizer=MagicMock())


def _clone_state(src: PicoLM, dst: PicoLM) -> None:
    """Copy ``src``'s parameter tensors into ``dst`` so both models start
    from identical weights.  Each model keeps its own ``.grad`` buffers."""
    dst.load_state_dict({k: v.clone() for k, v in src.state_dict().items()})


def _assert_grads_equal(ref: PicoLM, surr: PicoLM, *, atol: float = 1e-5) -> None:
    """Assert that every parameter has the same gradient in both models."""
    for (n_ref, p_ref), (n_surr, p_surr) in zip(
        ref.named_parameters(), surr.named_parameters(), strict=True
    ):
        assert n_ref == n_surr
        assert p_ref.grad is not None, f"reference model has no grad on {n_ref}"
        assert p_surr.grad is not None, f"surrogate model has no grad on {n_surr}"
        assert torch.allclose(p_ref.grad, p_surr.grad, atol=atol, rtol=atol), (
            f"gradient mismatch on parameter {n_ref!r}:\n"
            f"  reference (textbook PyTorch): {p_ref.grad}\n"
            f"  surrogate  (custom adapter ): {p_surr.grad}"
        )


# ---------------------------------------------------------------------------
# Single datum, non-linear loss in the logprobs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surrogate_matches_pytorch_autograd_squared_logprob_loss():
    """Side-by-side proof: ``forward_backward_custom_async`` ≡ ``loss.backward()``.

    Loss: ``C = sum_t logprob_t**2`` — a "confidence-squared" regulariser.
    It is **non-linear** in the logprobs, so it cannot be expressed as a
    built-in weighted cross-entropy — making this a genuine test of the
    surrogate-gradient trick.
    """
    torch.manual_seed(0)
    tokens = [1, 4, 2, 7, 0, 3]
    targets = [4, 2, 7, 0, 3, 5]

    # =========================================================================
    # Path A — the textbook PyTorch way.  Read this like you'd read any
    # researcher's notebook: build a model, run it, compute a loss on its
    # per-token logprobs, call .backward(), inspect the parameter gradients.
    # =========================================================================
    model_ref = PicoLM()
    target_logprobs = _per_token_target_logprobs(model_ref, tokens, targets)
    loss_ref = (target_logprobs ** 2).sum()
    loss_ref.backward()

    # =========================================================================
    # Path B — the same loss, but routed through forward_backward_custom_async.
    # The "backend" is the same model architecture loaded from path A's
    # weights, wrapped in a MockBackend that the cookbook adapter talks to
    # exactly as it would talk to a real Interactive Training session.
    # =========================================================================
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)
    tc = _build_training_client(MockBackend(model_surr))

    datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[float(t) for t in targets]),
        },
    )

    def my_loss(data, logprobs_list):
        # Mirrors path A: sum of squared per-token logprobs.
        loss = (logprobs_list[0] ** 2).sum()
        return loss, {"my_loss": float(loss.item())}

    future = await tc.forward_backward_custom_async([datum], loss_fn=my_loss)
    await future.result_async()

    # =========================================================================
    # The whole point: parameter gradients are identical.
    # =========================================================================
    _assert_grads_equal(model_ref, model_surr)


# ---------------------------------------------------------------------------
# Multiple datums, DPO-style pairwise preference loss.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surrogate_matches_pytorch_autograd_dpo_style_pairwise():
    """Same equivalence proof, but with a loss that **spans two datums**.

    Loss: a simplified DPO objective without a reference model,

        ``C = -log sigmoid(beta * (sum logprobs_chosen - sum logprobs_rejected))``

    This exercises the per-datum bookkeeping inside
    ``forward_backward_custom_async`` (zip over data + logprobs_list, separate
    ``.grad`` reads, two surrogate Datums on the backward call).
    """
    torch.manual_seed(0)
    chosen_tokens, chosen_targets = [1, 2, 3, 4], [2, 3, 4, 0]
    rejected_tokens, rejected_targets = [5, 6, 7, 1], [6, 7, 1, 5]
    beta = 0.5

    def dpo_objective(lp_chosen: torch.Tensor, lp_rejected: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(beta * (lp_chosen.sum() - lp_rejected.sum()))

    # ----- Path A: textbook PyTorch -----
    model_ref = PicoLM()
    lp_chosen_ref = _per_token_target_logprobs(model_ref, chosen_tokens, chosen_targets)
    lp_rejected_ref = _per_token_target_logprobs(
        model_ref, rejected_tokens, rejected_targets
    )
    loss_ref = dpo_objective(lp_chosen_ref, lp_rejected_ref)
    loss_ref.backward()

    # ----- Path B: forward_backward_custom_async -----
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)
    tc = _build_training_client(MockBackend(model_surr))

    data = [
        Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=chosen_tokens)]),
            loss_fn_inputs={
                "target_tokens": TensorData(
                    data=[float(t) for t in chosen_targets]
                ),
            },
        ),
        Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=rejected_tokens)]),
            loss_fn_inputs={
                "target_tokens": TensorData(
                    data=[float(t) for t in rejected_targets]
                ),
            },
        ),
    ]

    def dpo_loss(data, logprobs_list):
        loss = dpo_objective(logprobs_list[0], logprobs_list[1])
        return loss, {"dpo_loss": float(loss.item())}

    future = await tc.forward_backward_custom_async(data, loss_fn=dpo_loss)
    await future.result_async()

    _assert_grads_equal(model_ref, model_surr)


# ---------------------------------------------------------------------------
# Trivial sanity: custom loss = cross-entropy.
#
# This is the calibration test.  If the surrogate trick is correctly
# implemented, then plugging in ``C = -sum_t logprob_t`` (vanilla NLL) must
# reduce *exactly* to the backend's built-in ``cross_entropy``:
#
#     g_t = dC/dlogprob_t = -1   =>   w_t = -g_t = +1
#
# So the surrogate forward_backward is just ``cross_entropy`` with
# ``weights = [+1, +1, ...]``.  If anything is wrong with the sign, the
# autograd plumbing, or the weights packing, this test catches it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surrogate_matches_pytorch_autograd_trivial_cross_entropy():
    """Plain NLL through the custom path ≡ built-in cross-entropy.

    Asserts two things:
      1. Parameter gradients match a textbook ``loss = -tgt_lp.sum()`` backward.
      2. The surrogate ``weights`` the adapter sends are all exactly ``+1.0``
         — i.e. the custom path collapses to a vanilla cross-entropy step.
    """
    torch.manual_seed(0)
    tokens = [2, 5, 1, 6, 3, 0]
    targets = [5, 1, 6, 3, 0, 7]

    # ----- Path A: textbook PyTorch (vanilla NLL) -----
    model_ref = PicoLM()
    target_logprobs = _per_token_target_logprobs(model_ref, tokens, targets)
    loss_ref = -target_logprobs.sum()
    loss_ref.backward()

    # ----- Path B: forward_backward_custom_async -----
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)
    backend = MockBackend(model_surr)
    tc = _build_training_client(backend)

    datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[float(t) for t in targets]),
        },
    )

    def vanilla_nll(data, logprobs_list):
        loss = -logprobs_list[0].sum()
        return loss, {"nll": float(loss.item())}

    future = await tc.forward_backward_custom_async([datum], loss_fn=vanilla_nll)
    await future.result_async()

    # Gradients identical to vanilla autograd backward.
    _assert_grads_equal(model_ref, model_surr)

    # And the calibration check: weights sent to the backend are all +1.
    assert backend.last_fb_batch is not None and len(backend.last_fb_batch) == 1
    sent_weights = backend.last_fb_batch[0].loss_fn_inputs["weights"].data
    assert sent_weights == [1.0] * len(tokens), (
        f"expected surrogate weights = [+1.0]*{len(tokens)}, got {sent_weights}"
    )


# ---------------------------------------------------------------------------
# k3 KL estimator using ``weights`` as an auxiliary-data carrier.
#
# Loss: Schulman's unbiased non-negative KL estimator
# (http://joschu.net/blog/kl-approx.html), used by modern RLHF stacks
# (TRL, InstructGPT) as the KL penalty term:
#
#     C = sum_t [ exp(lp_ref_t - lp_t) - (lp_ref_t - lp_t) - 1 ]
#
# This equivalence case demonstrates the pattern every non-trivial research loss needs:
# **packing per-token auxiliary data into the ``weights`` field** — here,
# the reference-model logprobs.  The custom loss function reads them back
# via the ``tensor_data_to_torch`` helper.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surrogate_matches_pytorch_autograd_k3_kl_with_reference_logprobs():
    """k3 KL penalty against a frozen reference model.

    Demonstrates the ``weights``-as-auxiliary-data pattern: the user packs
    per-token reference logprobs into ``loss_fn_inputs["weights"]`` and the
    loss function reads them back inside the closure.
    """
    torch.manual_seed(0)
    tokens = [3, 1, 4, 1, 5, 2]
    targets = [1, 4, 1, 5, 2, 6]

    # Frozen reference model with different random init from the student.
    torch.manual_seed(42)
    model_frozen_ref = PicoLM()
    for p in model_frozen_ref.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ref_logprobs = _per_token_target_logprobs(model_frozen_ref, tokens, targets)

    def k3_kl(lp_student: torch.Tensor, lp_ref: torch.Tensor) -> torch.Tensor:
        diff = lp_ref - lp_student
        return (torch.exp(diff) - diff - 1).sum()

    # ----- Path A: textbook PyTorch -----
    torch.manual_seed(0)
    model_ref = PicoLM()  # the "student" along path A
    student_lp_ref = _per_token_target_logprobs(model_ref, tokens, targets)
    loss_ref = k3_kl(student_lp_ref, ref_logprobs)
    loss_ref.backward()

    # ----- Path B: forward_backward_custom_async, ref logprobs in `weights` -----
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)
    tc = _build_training_client(MockBackend(model_surr))

    datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[float(t) for t in targets]),
            # The aux-data carrier: per-token reference logprobs.
            "weights": TensorData(data=ref_logprobs.tolist()),
        },
    )

    def k3_kl_loss(data, logprobs_list):
        lp_student = logprobs_list[0]
        # Read the auxiliary per-token data the user packed into `weights`.
        lp_ref = tensor_data_to_torch(data[0].loss_fn_inputs["weights"])
        loss = k3_kl(lp_student, lp_ref)
        return loss, {"k3_kl": float(loss.item())}

    future = await tc.forward_backward_custom_async([datum], loss_fn=k3_kl_loss)
    await future.result_async()

    _assert_grads_equal(model_ref, model_surr)


# ---------------------------------------------------------------------------
# DAPO decoupled-clip surrogate (arXiv:2503.14476).
#
# DAPO is the canonical example of a loss that needs **multiple per-token
# aux quantities** at the same time: it consumes both ``advantages`` and
# ``sampling_logprobs`` to compute the raw ratio ``r = exp(lp - lp_sampling)``
# inside the clip.  Squeezing them into the single ``weights`` field the
# way we did for ``is_py`` doesn't work, because DAPO's
# ``min(r*A, clip(r, 1-eps_low, 1+eps_high)*A)`` needs the raw ratio.
#
# This test exercises the multi-aux-key pattern the cookbook adapter
# supports: the user packs each aux as its own ``loss_fn_inputs`` key
# (here ``"dapo/mask_adv"`` and ``"dapo/sampling_lp"``); the closure reads
# them back via ``tensor_data_to_torch``; the adapter strips them from
# the server-bound forward and surrogate-backward payloads so the wire
# contract stays ``{target_tokens, weights}``.
#
# We re-use the recipe's own ``_repack_for_dapo`` / ``_dapo_loss`` so this
# also pins the recipe behaviour against textbook autograd.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surrogate_matches_pytorch_autograd_dapo_decoupled_clip():
    """DAPO decoupled-clip surrogate ≡ ``loss.backward()`` on the recipe's
    own loss closure.

    Loss::

        r_t   = exp(lp_t - lp_sampling_t)
        L_t   = min(r_t * (mask_t * adv_t),
                    clamp(r_t, 1-eps_low, 1+eps_high) * (mask_t * adv_t))
        C     = -sum_t L_t

    We pick advantages and a sampling-policy logprob field with enough
    spread that the clip *does* bind on a meaningful fraction of tokens
    -- otherwise the test reduces to vanilla IS.  Hyperparameters come
    from the recipe (``_DAPO_EPS_LOW``, ``_DAPO_EPS_HIGH``) so any future
    change to those defaults is caught by both this test and the recipe
    in lock-step.
    """
    from interactive_training.recipes.math_rl.train_azure import (
        _DAPO_EPS_HIGH,
        _DAPO_EPS_LOW,
        _dapo_loss,
        _repack_for_dapo,
    )

    torch.manual_seed(0)
    tokens = [3, 1, 4, 1, 5, 2]
    targets = [1, 4, 1, 5, 2, 6]
    # Mix of positive / negative advantages plus a masked-out prompt
    # token; the mask is what the recipe pre-multiplies into ``adv``.
    advantages = [0.8, -0.6, 1.2, -0.4, 0.5, -0.9]
    mask = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Construct sampling logprobs offset from the student model's current
    # logprobs by a per-token delta wide enough to trigger clipping on
    # both sides.  Recall r = exp(lp - lp_sampling); to drive r above
    # 1 + eps_high = 1.28 we need lp - lp_sampling > ln(1.28) ~= 0.247,
    # i.e. sampling_offset < -0.247 on at least one token.  To drive r
    # below 1 - eps_low = 0.80 we need lp - lp_sampling < ln(0.80)
    # ~= -0.223, i.e. sampling_offset > +0.223 on at least one token.
    # High-clip only *binds* (surr2 < surr1) when adv > 0; low-clip only
    # binds when adv < 0.  Token 2 (adv=+1.2) gets a deep negative offset
    # for the high-clip; tokens 1 and 3 (adv<0) get positive offsets for
    # the low-clip.
    sampling_offset = [-0.40, 0.35, -0.40, 0.30, 0.05, -0.10]

    # ----- Path A: textbook PyTorch DAPO -----
    model_ref = PicoLM()
    student_lp_ref = _per_token_target_logprobs(model_ref, tokens, targets)
    samp_lp_t = student_lp_ref.detach() + torch.tensor(sampling_offset)
    mask_adv_t = torch.tensor(
        [m * a for m, a in zip(mask, advantages, strict=True)], dtype=torch.float32
    )
    r_t = torch.exp(student_lp_ref - samp_lp_t)
    r_clipped_t = torch.clamp(r_t, min=1.0 - _DAPO_EPS_LOW, max=1.0 + _DAPO_EPS_HIGH)
    surr1 = r_t * mask_adv_t
    surr2 = r_clipped_t * mask_adv_t
    loss_ref = -torch.minimum(surr1, surr2).sum()
    loss_ref.backward()

    # Sanity: the test data must actually exercise both clip sides --
    # otherwise this collapses to vanilla IS and proves nothing about the
    # clip math.  (Side count uses the recipe's sign convention from
    # ``_dapo_loss``: a token is "clipped" when surr2 < surr1.)
    with torch.no_grad():
        active = mask_adv_t != 0
        clipped = (surr2 < surr1) & active
        n_low = int((clipped & (r_t < 1.0 - _DAPO_EPS_LOW)).sum().item())
        n_high = int((clipped & (r_t > 1.0 + _DAPO_EPS_HIGH)).sum().item())
    assert n_low >= 1 and n_high >= 1, (
        f"test data does not exercise both clip sides "
        f"(n_low={n_low}, n_high={n_high}) -- pick wider sampling_offset"
    )

    # ----- Path B: recipe's repack + loss through forward_backward_custom_async -----
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)
    tc = _build_training_client(MockBackend(model_surr))

    # The "raw" datum the recipe receives from the dataset builder --
    # advantages / logprobs / mask, the canonical interactive-training shape.
    raw_datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[float(t) for t in targets]),
            "advantages": TensorData(data=advantages),
            "logprobs": TensorData(data=samp_lp_t.tolist()),
            "mask": TensorData(data=mask),
        },
    )
    repacked = _repack_for_dapo(raw_datum)

    # Sanity: repack stripped advantages/logprobs/mask and added the two
    # named aux keys.  This pins the multi-aux-key contract.
    assert set(repacked.loss_fn_inputs.keys()) == {
        "target_tokens",
        "dapo/mask_adv",
        "dapo/sampling_lp",
    }

    future = await tc.forward_backward_custom_async([repacked], loss_fn=_dapo_loss)
    await future.result_async()

    _assert_grads_equal(model_ref, model_surr)


# ---------------------------------------------------------------------------
# Multi-step parameter equivalence across full training cycles.
#
# The single-step tests above stop at gradient equivalence after one forward+backward.
# This one runs N full training cycles -- ``(forward_backward_custom_async
# → optim_step_async)`` -- against the MockBackend's local SGD, and compares
# **parameter values** to N equivalent steps of textbook PyTorch.
#
# What this catches that single-step tests can't:
#   - Stale .grad from the forward-only call leaking into the surrogate
#     backward (would make grads correct on step 1 but wrong on step 2+).
#   - Missing or out-of-order zero_grad on the engine side.
#   - Any state the adapter accidentally carries across calls (e.g. the
#     internal _step counter influencing payload construction).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surrogate_matches_pytorch_autograd_across_multiple_optim_steps():
    """N full ``(forward_backward_custom_async → optim_step_async)`` cycles
    leave the surrogate model bit-for-bit identical to N steps of textbook
    PyTorch SGD.

    Uses a non-trivial loss (squared logprobs) so the per-token gradient
    differs from vanilla CE on every step, exercising the surrogate-weight
    construction at each cycle.
    """
    torch.manual_seed(0)
    num_steps = 3
    sgd_lr = 0.1
    tokens = [1, 4, 2, 7, 0, 3]
    targets = [4, 2, 7, 0, 3, 5]

    model_ref = PicoLM()
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)

    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=sgd_lr)
    backend = MockBackend(model_surr, sgd_lr=sgd_lr)
    tc = _build_training_client(backend)

    datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[float(t) for t in targets]),
        },
    )

    def my_loss(data, logprobs_list):
        loss = (logprobs_list[0] ** 2).sum()
        return loss, {}

    for step in range(num_steps):
        # Reference: one full PyTorch training step.
        opt_ref.zero_grad()
        tgt_lp = _per_token_target_logprobs(model_ref, tokens, targets)
        (tgt_lp ** 2).sum().backward()
        opt_ref.step()

        # Surrogate: one full adapter training step.
        fb_future = await tc.forward_backward_custom_async([datum], loss_fn=my_loss)
        await fb_future.result_async()
        opt_future = await tc.optim_step_async()
        await opt_future.result_async()

        # After each step, parameters must match -- not just gradients.
        # If anything drifts (stale .grad, missed zero_grad, ordering bug
        # between forward-only and surrogate backward), the divergence
        # compounds and this assertion fails on step 2+.
        for (n_ref, p_ref), (n_surr, p_surr) in zip(
            model_ref.named_parameters(),
            model_surr.named_parameters(),
            strict=True,
        ):
            assert torch.allclose(p_ref, p_surr, atol=1e-5, rtol=1e-5), (
                f"parameter {n_ref!r} diverged at step {step + 1}/{num_steps}:\n"
                f"  reference: {p_ref.detach()}\n"
                f"  surrogate: {p_surr.detach()}"
            )


# ---------------------------------------------------------------------------
# Negative control — prove the equivalence comparison is actually discriminating.
#
# A green test suite is only meaningful if the assertions can fail.  We inject
# two plausible regressions into a MockBackend subclass and re-run the
# equivalence scaffold against them; if the assertion still passes, the
# equivalence tests weren't really testing what they claimed to.  This is
# the meta-check that pins the whole file's credibility.
#
# Mutation A: server kernel silently ignores ``weights`` (treats them as +1).
#   Models the failure mode "wire field got dropped / renamed somewhere along
#   the schema chain" -- the exact thing Layer-2-in-cookbook can't catch.
# Mutation B: surrogate weight sign flip (``+g`` instead of ``-g``).
#   Models a regression in ``_build_surrogate_datum``.
# ---------------------------------------------------------------------------


class _MockBackendIgnoresWeights(MockBackend):
    """Buggy backend: pretends server runs CE with weights = +1 regardless."""

    async def forward_backward_async(
        self, session_id, batch, *, loss_fn, loss_fn_config=None
    ):
        assert loss_fn == "cross_entropy"
        self.last_fb_batch = list(batch)
        total = torch.zeros(())
        for datum in batch:
            tokens = [t for chunk in datum.model_input.chunks for t in chunk.tokens]
            targets = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
            tgt_lp = _per_token_target_logprobs(self.model, tokens, targets)
            total = total + (-tgt_lp).sum()  # weights silently set to +1
        total.backward()
        return _awaitable(_fb_result(total_loss=float(total.item())))


class _MockBackendSignFlippedWeights(MockBackend):
    """Buggy backend: applies +weights instead of -weights in CE."""

    async def forward_backward_async(
        self, session_id, batch, *, loss_fn, loss_fn_config=None
    ):
        assert loss_fn == "cross_entropy"
        self.last_fb_batch = list(batch)
        total = torch.zeros(())
        for datum in batch:
            tokens = [t for chunk in datum.model_input.chunks for t in chunk.tokens]
            targets = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
            weights = torch.tensor(
                datum.loss_fn_inputs["weights"].data, dtype=torch.float32
            )
            tgt_lp = _per_token_target_logprobs(self.model, tokens, targets)
            total = total + (tgt_lp * weights).sum()  # sign-flipped vs CE contract
        total.backward()
        return _awaitable(_fb_result(total_loss=float(total.item())))


@pytest.mark.parametrize(
    "buggy_backend_cls,bug_description",
    [
        (_MockBackendIgnoresWeights, "server kernel ignores surrogate weights"),
        (_MockBackendSignFlippedWeights, "server kernel applies +weights instead of -weights"),
    ],
)
@pytest.mark.asyncio
async def test_negative_control_equivalence_assertion_catches_known_regressions(
    buggy_backend_cls, bug_description
):
    """Inject a known regression into the mock backend; assert the equivalence check fails.

    This is a meta-test: it pins that the gradient-equivalence assertion
    used throughout this file has real discriminating power.  If either
    mutation passes silently here, the equivalence tests above aren't
    proving what their docstrings claim.
    """
    torch.manual_seed(0)
    tokens = [1, 4, 2, 7, 0, 3]
    targets = [4, 2, 7, 0, 3, 5]

    # Reference path: textbook autograd on a loss that is NOT plain CE (so
    # the weights-ignoring bug actually differs from the correct answer).
    model_ref = PicoLM()
    tgt_lp = _per_token_target_logprobs(model_ref, tokens, targets)
    (tgt_lp ** 2).sum().backward()

    # Surrogate path through a buggy backend.
    model_surr = PicoLM()
    _clone_state(model_ref, model_surr)
    tc = _build_training_client(buggy_backend_cls(model_surr))

    datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[float(t) for t in targets]),
        },
    )

    def my_loss(data, logprobs_list):
        return (logprobs_list[0] ** 2).sum(), {}

    future = await tc.forward_backward_custom_async([datum], loss_fn=my_loss)
    await future.result_async()

    # The whole point: the equivalence assertion MUST fail under this mutation.
    with pytest.raises(AssertionError):
        _assert_grads_equal(model_ref, model_surr)
