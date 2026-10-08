from __future__ import annotations

import pytest

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
