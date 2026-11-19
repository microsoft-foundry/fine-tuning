from __future__ import annotations

import pytest
from azure.core.exceptions import (
    AzureError,
    HttpResponseError,
    ODataV4Format,
    ServiceRequestError,
    ServiceResponseError,
)

from shared.retry import RetryError, RetryPolicy, is_transient_error, retry_call


class ServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_retries_transient_error_then_succeeds() -> None:
    attempts = 0
    delays = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ServiceError("temporary", 503)
        return "done"

    result = retry_call(
        operation,
        policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=1,
            jitter_ratio=0,
            timeout_seconds=10,
        ),
        sleep=delays.append,
        random_values=[0.5, 0.5],
    )
    assert result == "done"
    assert delays == [1, 2]


def test_does_not_retry_invalid_input() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ServiceError("invalid request", 500)

    with pytest.raises(ServiceError):
        retry_call(operation, sleep=lambda _: None)
    assert attempts == 1


def test_treats_sdk_timed_out_message_as_transient() -> None:
    assert is_transient_error(RuntimeError("Request timed out."))


def test_treats_sdk_connection_error_as_transient() -> None:
    assert is_transient_error(RuntimeError("Connection error."))


def test_raises_explicit_error_when_exhausted() -> None:
    with pytest.raises(RetryError, match="failed after 2"):
        retry_call(
            lambda: (_ for _ in ()).throw(ServiceError("temporary", 503)),
            policy=RetryPolicy(
                max_attempts=2,
                initial_delay_seconds=0,
                jitter_ratio=0,
            ),
            sleep=lambda _: None,
        )


@pytest.mark.parametrize("elapsed_after_sleep", [1.0, 1.5])
def test_does_not_start_another_attempt_after_retry_deadline(
    elapsed_after_sleep: float,
) -> None:
    attempts = 0
    now = 0.0
    transient = ServiceError("temporary", 503)

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise transient
        return "must not run after the deadline"

    def sleep(delay: float) -> None:
        nonlocal now
        assert delay == 1.0
        now = elapsed_after_sleep

    with pytest.raises(RetryError, match="cannot retry within") as error:
        retry_call(
            operation,
            policy=RetryPolicy(
                max_attempts=2,
                initial_delay_seconds=1,
                jitter_ratio=0,
                timeout_seconds=1,
            ),
            sleep=sleep,
            clock=lambda: now,
            random_values=[0.5],
        )
    assert attempts == 1
    assert error.value.__cause__ is transient


@pytest.mark.parametrize("elapsed", [1.0, 1.5])
@pytest.mark.parametrize("initial_failure", [False, True])
def test_rejects_success_returned_at_or_after_retry_deadline(elapsed, initial_failure) -> None:
    now = 0.0
    attempts = 0
    transient = ServiceError("temporary", 503)

    def operation():
        nonlocal now, attempts
        attempts += 1
        if initial_failure and attempts == 1:
            raise transient
        now = elapsed
        return "late result"

    with pytest.raises(RetryError, match="did not complete within") as error:
        retry_call(
            operation,
            policy=RetryPolicy(initial_delay_seconds=0, jitter_ratio=0, timeout_seconds=1),
            sleep=lambda _: None,
            clock=lambda: now,
        )
    assert attempts == (2 if initial_failure else 1)
    assert error.value.__cause__ is (transient if initial_failure else None)


def test_checks_budget_before_the_first_operation() -> None:
    times = iter([0.0, 1.0])
    attempts = []
    with pytest.raises(RetryError, match="cannot retry within"):
        retry_call(
            lambda: attempts.append(True),
            policy=RetryPolicy(timeout_seconds=1),
            clock=lambda: next(times),
        )
    assert attempts == []


def test_retry_callback_cannot_extend_the_budget_or_start_another_attempt() -> None:
    now = 0.0
    attempts = []
    delays = []
    transient = ServiceError("temporary", 503)

    def operation():
        attempts.append(True)
        raise transient

    def on_retry(attempt, delay, error):
        nonlocal now
        assert (attempt, delay, error) == (1, 0.5, transient)
        now = 1.0

    with pytest.raises(RetryError, match="cannot retry within") as error:
        retry_call(
            operation,
            policy=RetryPolicy(initial_delay_seconds=0.5, jitter_ratio=0, timeout_seconds=1),
            on_retry=on_retry,
            sleep=delays.append,
            clock=lambda: now,
        )
    assert attempts == [True]
    assert delays == []
    assert error.value.__cause__ is transient


def test_absolute_deadline_caps_retry_sleep_without_renewing_the_budget() -> None:
    now = 0.0
    attempts = []
    delays = []
    transient = ServiceError("temporary", 503)

    def operation():
        attempts.append(True)
        raise transient

    def sleep(delay):
        nonlocal now
        delays.append(delay)
        now += delay

    with pytest.raises(RetryError, match="cannot retry within") as error:
        retry_call(
            operation,
            policy=RetryPolicy(initial_delay_seconds=4, jitter_ratio=0, timeout_seconds=10),
            deadline=2,
            sleep=sleep,
            clock=lambda: now,
        )
    assert attempts == [True]
    assert delays == [2]
    assert now == 2
    assert error.value.__cause__ is transient


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "parameter",
    ["initial_delay_seconds", "max_delay_seconds", "multiplier", "jitter_ratio", "timeout_seconds"],
)
def test_retry_policy_rejects_nonfinite_parameters(parameter, value) -> None:
    with pytest.raises(ValueError, match="finite"):
        RetryPolicy(**{parameter: value})


@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("nan"), float("inf")])
def test_retry_policy_requires_a_positive_integer_attempt_count(value) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_retry_call_rejects_nonfinite_absolute_deadline(value) -> None:
    attempts = []
    with pytest.raises(ValueError, match="deadline"):
        retry_call(lambda: attempts.append(True), deadline=value)
    assert attempts == []


@pytest.mark.parametrize("code", ["insufficient_quota", "InsufficientQuota"])
@pytest.mark.parametrize("shape", ["code", "error_code", "body", "nested-body", "error"])
def test_structured_terminal_quota_overrides_429_and_transient_text(code, shape) -> None:
    terminal = ServiceError("RateLimit: connection error; request timed out", 429)
    if shape in {"code", "error_code"}:
        setattr(terminal, shape, code)
    elif shape == "nested-body":
        terminal.body = {"error": {"details": [{"error": {"code": code}}]}}
    else:
        setattr(terminal, shape, {"error": {"code": code}})
    attempts = []
    delays = []

    def operation():
        attempts.append(True)
        raise terminal

    assert not is_transient_error(terminal)
    with pytest.raises(ServiceError) as error:
        retry_call(operation, sleep=delays.append)
    assert error.value is terminal
    assert attempts == [True]
    assert delays == []


@pytest.mark.parametrize("attribute", ["error", "model"])
@pytest.mark.parametrize("error_type", [HttpResponseError, ServiceRequestError, ServiceResponseError])
def test_native_sdk_error_model_quota_code_overrides_429(attribute, error_type) -> None:
    terminal = error_type("RateLimit: connection error; request timed out")
    terminal.status_code = 429
    setattr(terminal, attribute, ODataV4Format({
        "code": "RateLimit",
        "message": "Temporarily unavailable",
        "details": [{"code": "insufficient_quota", "message": "Quota exhausted"}],
    }))
    assert not is_transient_error(terminal)


def test_structured_rate_limit_without_terminal_quota_remains_transient() -> None:
    transient = ServiceError("Rate limited", 429)
    transient.body = {"error": {"code": "rate_limit_exceeded"}}
    assert is_transient_error(transient)


@pytest.mark.parametrize("error_type", [ServiceRequestError, ServiceResponseError])
def test_explicit_azure_transport_failures_are_retried_without_text_markers(error_type) -> None:
    transient = error_type("Transport failed")
    attempts = []
    delays = []

    def operation():
        attempts.append(True)
        if len(attempts) == 1:
            raise transient
        return "recovered"

    assert is_transient_error(transient)
    assert retry_call(
        operation,
        policy=RetryPolicy(initial_delay_seconds=0, jitter_ratio=0),
        sleep=delays.append,
        clock=lambda: 0,
    ) == "recovered"
    assert attempts == [True, True]
    assert delays == [0]


@pytest.mark.parametrize("error_type", [AzureError, HttpResponseError, RuntimeError, ValueError])
def test_unclassified_errors_are_not_blanket_transport_retries(error_type) -> None:
    assert not is_transient_error(error_type("Transport failed"))
