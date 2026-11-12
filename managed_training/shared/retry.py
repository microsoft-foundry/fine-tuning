"""Bounded retries for explicitly transient service failures."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from azure.core.exceptions import ODataV4Format, ServiceRequestError, ServiceResponseError

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
    "insufficient_quota",
    "insufficientquota",
    "quota exceeded",
    "quota_exceeded",
    "quotaexceeded",
)
_TRANSIENT_CODE_MARKERS = (
    "connection error",
    "dns",
    "getaddrinfo",
    "handshake",
    "name resolution",
    "timeout",
    "timed out",
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
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be an integer of at least 1")
        for delay in (self.initial_delay_seconds, self.max_delay_seconds):
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("retry delays must be finite and non-negative")
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("multiplier must be finite and at least 1")
        if not math.isfinite(self.jitter_ratio) or not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be finite and between 0 and 1")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")


def _exception_text(error: BaseException) -> str:
    parts = [str(error)]
    pending: list[Any] = [error]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        else:
            for attribute in (
                "code", "error_code", "message", "body", "model", "details",
                "innererror", "inner_error",
            ):
                pending.append(getattr(value, attribute, None))
            # ODataV4Format.error is a deprecated alias for itself, not a child.
            if not isinstance(value, ODataV4Format):
                pending.append(getattr(value, "error", None))
    return " ".join(parts).casefold()


def is_transient_error(error: BaseException) -> bool:
    """Classify common SDK/HTTP failures without retrying terminal conditions."""
    text = _exception_text(error)
    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return False
    if isinstance(error, (ServiceRequestError, ServiceResponseError)):
        return True
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
    deadline: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    random_values: Iterable[float] | None = None,
) -> T:
    """Reject results past the budget without interrupting an in-flight operation.

    An optional absolute monotonic ``deadline`` can only shorten the policy's
    budget, allowing a polling loop to share one deadline across all retries.
    """
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    started = clock()
    expires = started + policy.timeout_seconds
    if deadline is not None:
        expires = min(expires, deadline)
    budget = max(0.0, expires - started)
    timeout_message = (
        f"{operation_name} cannot retry within its {budget:g}-second timeout"
    )
    last_error: Exception | None = None
    random_iterator = iter(random_values) if random_values is not None else None
    for attempt in range(1, policy.max_attempts + 1):
        if clock() >= expires:
            raise RetryError(timeout_message) from last_error
        try:
            result = operation()
        except Exception as error:
            if not retry_if(error):
                raise
            last_error = error
            if attempt == policy.max_attempts:
                raise RetryError(
                    f"{operation_name} failed after {attempt} transient attempt(s)"
                ) from error
            remaining = expires - clock()
            if remaining <= 0:
                raise RetryError(timeout_message) from error
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
            delay = min(max(0.0, base_delay + jitter), remaining)
            if on_retry:
                on_retry(attempt, delay, error)
            remaining = expires - clock()
            if remaining <= 0:
                raise RetryError(timeout_message) from error
            sleep(min(delay, remaining))
        else:
            if clock() >= expires:
                raise RetryError(
                    f"{operation_name} did not complete within its "
                    f"{budget:g}-second timeout"
                ) from last_error
            return result
    raise AssertionError("retry loop ended unexpectedly")
