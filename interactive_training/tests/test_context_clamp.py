"""Unit tests for context-window overflow handling in SessionTokenCompleter.

When the sampling server rejects a request because
``prompt_tokens + max_tokens > context_length`` (HTTP 400 or a polled
``RequestValidationError``), the completer must
clamp ``max_tokens`` to fit the remaining window and retry once, keeping the
trajectory alive without any client-side context-length configuration or server
deployment. When the prompt alone fills the window it must raise
``ContextLengthExceededError`` so the rollout can end the trajectory gracefully.
"""

import asyncio
from types import SimpleNamespace

import pytest
from azure.ai.finetuningsessions import RequestValidationError
from azure.ai.finetuningsessions._exceptions import _classify_poll_failure
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk
from azure.core.exceptions import HttpResponseError

from interactive_training.completers import (
    ContextLengthExceededError,
    SessionTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from interactive_training.rl.rollouts import do_single_rollout
from interactive_training.rl.types import Env, StepResult


def _model_input(n_tokens: int = 3):
    return ModelInput(chunks=[ModelInputChunk(tokens=list(range(n_tokens)))])


def _overflow_error(prompt: int, max_tokens: int, context_limit: int, *, polled: bool = False) -> HttpResponseError:
    total = prompt + max_tokens
    message = (
        f"Requested prompt tokens ({prompt}) + max_tokens ({max_tokens}) = {total} "
        f"exceeds the maximum context length ({context_limit})"
    )
    if polled:
        exc = _classify_poll_failure({
            "status": "failed", "error_code": "invalid_request", "error": message,
        })
        assert isinstance(exc, RequestValidationError)
        assert exc.status_code is None
        return exc
    exc = HttpResponseError(message=message)
    exc.status_code = 400
    return exc


class _ClampThenSucceedClient:
    """Raises a context-overflow 400 on the first call, succeeds afterwards.

    Records the max_tokens seen on each call so the test can assert the clamp.
    """

    def __init__(self, prompt: int, context_limit: int, *, polled: bool = False):
        self.prompt = prompt
        self.context_limit = context_limit
        self.polled = polled
        self.max_tokens_seen: list[int] = []

    def sample_async(self, **kwargs):
        max_tokens = kwargs["sampling_params"].max_tokens
        self.max_tokens_seen.append(max_tokens)
        first_call = len(self.max_tokens_seen) == 1

        async def _run():
            if first_call:
                raise _overflow_error(self.prompt, max_tokens, self.context_limit, polled=self.polled)
            seq = SimpleNamespace(tokens=[7, 8], logprobs=[-0.2, -0.3])
            return SimpleNamespace(sequences=[seq])

        return _run()


class _AlwaysOverflowClient:
    """Every call reports the prompt alone exceeds the window."""

    def __init__(self, prompt: int, context_limit: int, *, polled: bool = False):
        self.prompt = prompt
        self.context_limit = context_limit
        self.polled = polled
        self.calls = 0

    def sample_async(self, **kwargs):
        self.calls += 1
        max_tokens = kwargs["sampling_params"].max_tokens

        async def _run():
            raise _overflow_error(self.prompt, max_tokens, self.context_limit, polled=self.polled)

        return _run()


class _NonContextBadRequestClient:
    """Raises a 400 that is NOT a context-overflow (must not be clamped)."""

    def __init__(self):
        self.calls = 0

    def sample_async(self, **kwargs):
        self.calls += 1

        async def _run():
            exc = HttpResponseError(message="invalid value for temperature")
            exc.status_code = 400
            raise exc

        return _run()


@pytest.mark.parametrize("polled", [False, True])
def test_clamps_and_retries_on_overflow(polled):
    context_limit = 32768
    prompt = 31361
    client = _ClampThenSucceedClient(prompt=prompt, context_limit=context_limit, polled=polled)
    completer = SessionTokenCompleter(client, max_tokens=4096)

    result = asyncio.run(completer(_model_input(), stop=[]))

    # First request used the full budget, retry used the clamped budget.
    assert len(client.max_tokens_seen) == 2
    assert client.max_tokens_seen[0] == 4096
    # Clamped to context_limit - prompt - margin(8) = 1399, capped at request.
    assert client.max_tokens_seen[1] == context_limit - prompt - 8
    assert client.max_tokens_seen[1] < 4096
    assert result.tokens == [7, 8]
    assert completer.context_clamp_count == 1


@pytest.mark.parametrize("polled", [False, True])
def test_clamp_never_exceeds_requested(polled):
    # Plenty of room, but the (unlikely) reported prompt is tiny; the clamp must
    # never raise max_tokens above what was requested.
    context_limit = 32768
    prompt = 10
    client = _ClampThenSucceedClient(prompt=prompt, context_limit=context_limit, polled=polled)
    completer = SessionTokenCompleter(client, max_tokens=256)

    asyncio.run(completer(_model_input(), stop=[]))

    assert client.max_tokens_seen[1] == 256


@pytest.mark.parametrize("polled", [False, True])
def test_prompt_alone_exceeds_raises_sentinel(polled):
    context_limit = 32768
    prompt = 32768  # no room to generate anything useful
    client = _AlwaysOverflowClient(prompt=prompt, context_limit=context_limit, polled=polled)
    completer = SessionTokenCompleter(client, max_tokens=4096)

    with pytest.raises(ContextLengthExceededError):
        asyncio.run(completer(_model_input(), stop=[]))

    # Only the initial attempt is made; no retry when the prompt fills the window.
    assert client.calls == 1
    assert completer.context_clamp_count == 0


def test_non_context_bad_request_is_reraised():
    client = _NonContextBadRequestClient()
    completer = SessionTokenCompleter(client, max_tokens=256)

    with pytest.raises(HttpResponseError):
        asyncio.run(completer(_model_input(), stop=[]))

    # Not retried: a non-context 400 is not clamped.
    assert client.calls == 1
    assert completer.context_clamp_count == 0


class _DoubleOverflowClient:
    """Reports room on the first call, but the clamped retry ALSO overflows.

    Simulates the server counting more prompt tokens than it reported, so the
    clamped budget still does not fit. The completer must degrade to a graceful
    trajectory end rather than surfacing a raw 400.
    """

    def __init__(self, context_limit: int, *, polled: bool = False):
        self.context_limit = context_limit
        self.polled = polled
        self.calls = 0

    def sample_async(self, **kwargs):
        self.calls += 1
        max_tokens = kwargs["sampling_params"].max_tokens
        # First call: report a prompt that leaves ample room (so we clamp+retry).
        # Retry: report a prompt that fills the window (still overflows).
        prompt = 30000 if self.calls == 1 else self.context_limit

        async def _run():
            raise _overflow_error(prompt, max_tokens, self.context_limit, polled=self.polled)

        return _run()


@pytest.mark.parametrize("polled", [False, True])
def test_clamped_retry_still_overflowing_ends_gracefully(polled):
    client = _DoubleOverflowClient(context_limit=32768, polled=polled)
    completer = SessionTokenCompleter(client, max_tokens=4096)

    with pytest.raises(ContextLengthExceededError):
        asyncio.run(completer(_model_input(), stop=[]))
    assert client.calls == 2


class _ScriptedClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.budgets = []

    async def sample_async(self, **kwargs):
        self.budgets.append(kwargs["sampling_params"].max_tokens)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1], logprobs=[-0.1])])


@pytest.mark.parametrize("context_limit", [2048, 32768, 33792, 131072])
def test_polled_recovery_uses_reported_model_limit(context_limit):
    client = _ClampThenSucceedClient(prompt=context_limit - 100, context_limit=context_limit, polled=True)
    completer = SessionTokenCompleter(client, max_tokens=512)
    result = asyncio.run(completer(_model_input(), stop=[]))
    assert result.tokens == [7, 8]
    assert client.max_tokens_seen == [512, 92]


@pytest.mark.parametrize("first_polled", [False, True])
def test_retry_can_fail_through_other_transport(first_polled):
    client = _ScriptedClient([
        _overflow_error(100, 40, 128, polled=first_polled),
        _overflow_error(100, 20, 110, polled=not first_polled),
    ])
    completer = SessionTokenCompleter(client, max_tokens=40)
    with pytest.raises(ContextLengthExceededError):
        asyncio.run(completer(_model_input(), stop=[]))
    assert client.budgets == [40, 20]


@pytest.mark.parametrize("error", [
    RequestValidationError("Invalid temperature", error_code="invalid_request"),
    RequestValidationError("maximum context length (128)", error_code="internal_error"),
    RequestValidationError("maximum context length exceeded", error_code="invalid_request"),
    RequestValidationError("maximum context length (0)", error_code="invalid_request"),
    HttpResponseError(message="maximum context length (128)"),
])
def test_unrelated_or_unusable_polled_error_is_not_retried(error):
    client = _ScriptedClient([error])
    completer = SessionTokenCompleter(client, max_tokens=40)
    with pytest.raises(HttpResponseError) as exc_info:
        asyncio.run(completer(_model_input(), stop=[]))
    assert exc_info.value is error
    assert client.budgets == [40]


@pytest.mark.parametrize("status", [401, 429, 500])
def test_real_http_failure_is_not_reinterpreted_as_polled_validation(status):
    error = RequestValidationError("maximum context length (128)", error_code="invalid_request")
    error.status_code = status
    client = _ScriptedClient([error])
    with pytest.raises(RequestValidationError) as exc_info:
        asyncio.run(SessionTokenCompleter(client, max_tokens=40)(_model_input(), stop=[]))
    assert exc_info.value is error
    assert client.budgets == [40]


# --- Graceful trajectory end when the prompt fills the window --------------


class _OverflowAfterNPolicy(TokenCompleter):
    """Succeeds for ``n_ok`` calls, then raises ContextLengthExceededError."""

    def __init__(self, n_ok: int):
        self.n_ok = n_ok
        self.calls = 0

    async def __call__(self, model_input, stop, max_tokens=None):
        self.calls += 1
        if self.calls > self.n_ok:
            raise ContextLengthExceededError("prompt fills the window")
        return TokensWithLogprobs(tokens=[1], maybe_logprobs=[-0.1])


class _MultiStepEnv(Env):
    """Runs until the policy stops feeding it actions."""

    def __init__(self):
        self.closed = 0
        self.observation = ModelInput(chunks=[ModelInputChunk(tokens=[0])])

    async def initial_observation(self):
        return self.observation, []

    async def step(self, action):
        return StepResult(
            reward=1.0,
            episode_done=False,
            next_observation=self.observation,
            next_stop_condition=[],
        )

    async def close(self):
        self.closed += 1


def test_rollout_ends_gracefully_when_prompt_fills_window():
    policy = _OverflowAfterNPolicy(n_ok=2)
    env = _MultiStepEnv()

    trajectory = asyncio.run(do_single_rollout(policy, env))

    # The two successful steps are kept; the run does not crash.
    assert len(trajectory.transitions) == 2
    # The env is left OPEN on the graceful-overflow path — like the normal
    # success path — so do_group_rollout can read final state in
    # compute_group_rewards before its finally closes every env. Closing here
    # would tear down the sandbox and corrupt terminal rewards.
    assert env.closed == 0


def test_rollout_reraises_when_first_turn_overflows():
    policy = _OverflowAfterNPolicy(n_ok=0)
    env = _MultiStepEnv()

    with pytest.raises(ContextLengthExceededError):
        asyncio.run(do_single_rollout(policy, env))

    # Env is still cleaned up on the propagated failure.
    assert env.closed == 1


@pytest.mark.parametrize("successful_turns", [0, 2])
def test_polled_context_error_integrates_with_rollout_lifecycle(successful_turns):
    client = _ScriptedClient([
        *([None] * successful_turns), _overflow_error(128, 40, 128, polled=True),
    ])
    policy = SessionTokenCompleter(client, max_tokens=40)
    env = _MultiStepEnv()
    if successful_turns:
        trajectory = asyncio.run(do_single_rollout(policy, env))
        assert len(trajectory.transitions) == successful_turns
        assert env.closed == 0  # Reward calculation still needs this environment.
    else:
        with pytest.raises(ContextLengthExceededError):
            asyncio.run(do_single_rollout(policy, env))
        assert env.closed == 1
    assert len(client.budgets) == successful_turns + 1
