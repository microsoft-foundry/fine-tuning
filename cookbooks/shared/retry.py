"""Bounded retries for explicitly transient service failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_NON_RETRYABLE_MARKERS = (
    "authentication",
    "authorization",
    "invalid",
    "content filter",
    "content_filter",
    "safety",
    "insufficient quota",
    "quota exceeded",
)
_TRANSIENT_CODE_MARKERS = (
    "timeout",
    "temporarilyunavailable",
    "serviceunavailable",
    "toomanyrequests",
    "ratelimit",
    "badrequestfordependentservice",
)


class RetryError(RuntimeError):
    """Raised when a bounded transient retry policy is exhausted."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2
    timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


def _exception_text(error: BaseException) -> str:
    parts = [str(error)]
    for attribute in ("code", "error_code", "body"):
        value = getattr(error, attribute, None)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).casefold()


def is_transient_error(error: BaseException) -> bool:
    """Classify common SDK/HTTP failures without retrying terminal conditions."""
    text = _exception_text(error)
    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return False
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status in _TRANSIENT_STATUS_CODES:
        return True
    return any(marker in text for marker in _TRANSIENT_CODE_MARKERS)


def retry_call(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy = RetryPolicy(),
    operation_name: str = "operation",
    retry_if: Callable[[BaseException], bool] = is_transient_error,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    random_values: Iterable[float] | None = None,
) -> T:
    """Execute an operation with bounded exponential backoff and jitter."""
    started = clock()
    random_iterator = iter(random_values) if random_values is not None else None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if not retry_if(error):
                raise
            elapsed = clock() - started
            if attempt == policy.max_attempts or elapsed >= policy.timeout_seconds:
                raise RetryError(
                    f"{operation_name} failed after {attempt} transient attempt(s)"
                ) from error
            base_delay = min(
                policy.initial_delay_seconds * policy.multiplier ** (attempt - 1),
                policy.max_delay_seconds,
            )
            sample = (
                next(random_iterator)
                if random_iterator is not None
                else random.random()
            )
            jitter = base_delay * policy.jitter_ratio * ((sample * 2) - 1)
            delay = max(0.0, base_delay + jitter)
            remaining = policy.timeout_seconds - elapsed
            if delay > remaining:
                raise RetryError(
                    f"{operation_name} cannot retry within its "
                    f"{policy.timeout_seconds:g}-second timeout"
                ) from error
            if on_retry:
                on_retry(attempt, delay, error)
            sleep(delay)
    raise AssertionError("retry loop ended unexpectedly")
