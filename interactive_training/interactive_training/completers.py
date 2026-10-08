"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, TypeAlias

from azure.ai.finetuningsessions import FineTuningSession, RequestValidationError
from azure.ai.finetuningsessions.models import ModelInput, SamplingParams
from azure.core.exceptions import HttpResponseError

from interactive_training import renderers

logger = logging.getLogger(__name__)

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


# --- Context-window overflow handling -------------------------------------
#
# The sampling server rejects a request with HTTP 400 or a polled
# RequestValidationError (invalid_request) when
#   prompt_tokens + max_tokens > model_context_length
# The error body looks like:
#   Requested prompt tokens (31361) + max_tokens (4096) = 35457 exceeds the
#   maximum context length (32768)
# We parse the authoritative numbers straight out of that message so the
# behaviour self-configures to whatever model / context length is deployed
# (no client-side context-length parameter required).
_CTX_LEN_RE = re.compile(r"maximum context length \((\d+)\)")
_PROMPT_TOKENS_RE = re.compile(r"prompt tokens \((\d+)\)")

# Below this generation budget the prompt has effectively filled the window and
# the turn cannot produce a useful action, so we stop rather than retry.
_MIN_CLAMPED_MAX_TOKENS = 16
# Safety margin subtracted from the remaining window when clamping, to absorb
# any small discrepancy between the client- and server-side token counts.
_CTX_CLAMP_MARGIN = 8

# Rate-limit the (per-process) clamp warning so a long trajectory that clamps
# every turn does not flood the log. One line per this many seconds.
_CLAMP_LOG_MIN_INTERVAL_SEC = 30.0
_last_clamp_log_at = 0.0


class ContextLengthExceededError(RuntimeError):
    """Raised when the prompt alone meets or exceeds the model context window,
    so no tokens can be generated for this turn. Callers should end the
    trajectory gracefully rather than crash the run."""


def _maybe_log_clamp(prompt_tokens: int, requested: int, clamped: int, context_limit: int) -> None:
    """Emit a rate-limited WARNING so operators can see when/how often the
    generation budget is being clamped to fit the context window."""
    global _last_clamp_log_at
    now = time.monotonic()
    if now - _last_clamp_log_at < _CLAMP_LOG_MIN_INTERVAL_SEC:
        return
    _last_clamp_log_at = now
    logger.warning(
        "[context] prompt=%d + max_tokens=%d exceeds context length %d; "
        "clamped max_tokens to %d and retried (rate-limited; further clamps suppressed for %.0fs)",
        prompt_tokens,
        requested,
        context_limit,
        clamped,
        _CLAMP_LOG_MIN_INTERVAL_SEC,
    )


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    async def __call__(
        self, model_input: ModelInput, stop: StopCondition, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        raise NotImplementedError


# Implementations


@dataclass
class SessionTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a FineTuningSession to sample actions.
    """

    sampling_client: FineTuningSession
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0   # nucleus sampling threshold (1.0 = disabled)
    top_k: int = -1      # top-k filtering (-1 = disabled)
    seed: int | None = None  # base RNG seed for reproducible sampling
    response_format: dict[str, Any] | None = None
    sample_timeout_sec: float | None = None

    def __post_init__(self):
        if self.sample_timeout_sec is not None and self.sample_timeout_sec <= 0:
            raise ValueError("sample_timeout_sec must be positive when set")
        self._call_counter = 0
        # Number of times this completer clamped max_tokens to fit the context
        # window and retried. Surfaced by the rollout layer as a metric.
        self.context_clamp_count = 0

    async def _sample_once(
        self,
        model_input: ModelInput,
        stop: StopCondition,
        max_tokens: int,
        call_seed: int | None,
    ):
        """Issue a single sample call, honouring the optional per-call timeout.

        A fresh operation/coroutine is built on every invocation because
        ``asyncio.wait_for`` consumes the awaitable, so a retry cannot reuse it.
        """
        sample_operation = self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=SamplingParams(
                stop_criteria=stop,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                seed=call_seed,
                response_format=self.response_format,
            ),
        )
        if self.sample_timeout_sec is None:
            return await sample_operation
        return await asyncio.wait_for(sample_operation, timeout=self.sample_timeout_sec)

    def _parse_context_overflow(
        self, exc: HttpResponseError, model_input: ModelInput
    ) -> tuple[int, int] | None:
        """If ``exc`` is an HTTP or polled context-window rejection, return
        ``(context_limit, prompt_tokens)``; otherwise ``None`` so the caller
        re-raises the original error.
        """
        status = getattr(exc, "status_code", None)
        if status is None and getattr(exc, "response", None) is not None:
            status = getattr(exc.response, "status_code", None)
        text = str(exc)
        # Poll failures have no HTTP error response: the poll itself succeeded.
        # Accept only the SDK's terminal validation type/code, not arbitrary
        # status-less errors or retryable failures with similar wording.
        polled_validation = (
            status is None
            and isinstance(exc, RequestValidationError)
            and exc.error_code == "invalid_request"
        )
        if (status != 400 and not polled_validation) or "context length" not in text:
            return None

        ctx_match = _CTX_LEN_RE.search(text)
        if ctx_match is None:
            return None  # cannot determine the limit; nothing safe to clamp to
        context_limit = int(ctx_match.group(1))
        if context_limit <= 0:
            return None

        prompt_match = _PROMPT_TOKENS_RE.search(text)
        if prompt_match is not None:
            prompt_tokens = int(prompt_match.group(1))
        else:
            # Fall back to a client-side count of the prompt tokens.
            prompt_tokens = sum(len(chunk.tokens) for chunk in model_input.chunks)
        return context_limit, prompt_tokens

    async def __call__(
        self, model_input: ModelInput, stop: StopCondition, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        # Derive a deterministic per-call seed when a base seed is configured.
        # Each call increments the counter so concurrent rollouts within the
        # same group get different (but reproducible) seeds.
        call_seed = None
        if self.seed is not None:
            call_seed = (self.seed + self._call_counter) % (2**31)
            self._call_counter += 1

        requested = max_tokens if max_tokens is not None else self.max_tokens

        # Sample from the model. On a context-window rejection the
        # server tells us the exact limit; clamp the generation budget to fit
        # and retry once. This keeps the trajectory alive without any
        # client-side context-length configuration or server deployment.
        try:
            sample_result = await self._sample_once(model_input, stop, requested, call_seed)
        except HttpResponseError as exc:
            overflow = self._parse_context_overflow(exc, model_input)
            if overflow is None:
                raise
            context_limit, prompt_tokens = overflow
            clamped = min(requested, context_limit - prompt_tokens - _CTX_CLAMP_MARGIN)
            if clamped < _MIN_CLAMPED_MAX_TOKENS:
                # The prompt alone fills the window: no useful action can be
                # sampled. Signal a graceful trajectory end rather than crash.
                logger.warning(
                    "[context] prompt tokens (%d) fill the context window (%d); "
                    "cannot generate an action for this turn \u2014 ending trajectory",
                    prompt_tokens,
                    context_limit,
                )
                raise ContextLengthExceededError(
                    f"prompt tokens ({prompt_tokens}) leave no room to generate "
                    f"within context length ({context_limit})"
                ) from exc
            self.context_clamp_count += 1
            _maybe_log_clamp(prompt_tokens, requested, clamped, context_limit)
            try:
                sample_result = await self._sample_once(
                    model_input, stop, clamped, call_seed
                )
            except HttpResponseError as retry_exc:
                # The clamped budget still overflowed (e.g. the server counts
                # more prompt tokens than reported, beyond our margin). Don't
                # loop or crash: end the trajectory gracefully.
                if self._parse_context_overflow(retry_exc, model_input) is None:
                    raise
                logger.warning(
                    "[context] clamped retry (max_tokens=%d) still exceeded "
                    "context length %d \u2014 ending trajectory",
                    clamped,
                    context_limit,
                )
                raise ContextLengthExceededError(
                    "sample still exceeded the context window after clamping"
                ) from retry_exc

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None

        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


class SessionMessageCompleter(MessageCompleter):
    """A completer that uses the actual model to generate responses."""

    def __init__(
        self,
        sampling_client: FineTuningSession,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
        temperature: float = 1.0,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_criteria=self.stop_condition,
            ),
        )

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)

        return {"role": "assistant", "content": parsed_message["content"]}
