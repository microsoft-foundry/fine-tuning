"""Bounded runtime waits shared by the Stage 05 notebooks."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

FILE_TERMINAL_STATES = {"processed", "error", "expired", "failed", "cancelled", "canceled"}
JOB_TERMINAL_STATES = {"succeeded", "failed", "cancelled", "canceled"}
TRANSIENT_EXCEPTION_NAMES = {"APIConnectionError", "APITimeoutError"}


def _status(resource: Any) -> str:
    return str(getattr(resource, "status", "")).split(".")[-1].casefold()


def _require_id(resource: Any, resource_name: str) -> str:
    resource_id = getattr(resource, "id", None)
    if not resource_id:
        raise RuntimeError(f"{resource_name} response did not contain an id.")
    return str(resource_id)


def _validate_wait(poll_interval_seconds: float, timeout_seconds: float) -> None:
    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be finite and greater than zero.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero.")


def _is_transient(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (status_code or 0) >= 500:
        return True
    if status_code == 404:
        message = str(exc).casefold()
        return "project not found" in message or "no such file object" in message
    return type(exc).__name__ in TRANSIENT_EXCEPTION_NAMES


def _retrieve_with_retry(
    operation: Any,
    *,
    resource_name: str,
    deadline: float,
    attempts: int = 5,
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts or not _is_transient(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{resource_name} polling exceeded its timeout after a transient error."
                ) from exc
            delay = min(float(2 ** (attempt - 1)), remaining)
            print(
                f"{resource_name} polling hit a transient error; retrying in "
                f"{delay:g}s ({attempt}/{attempts}): {exc}"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def wait_for_file_processed(
    client: Any,
    file_id: str,
    *,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 600,
) -> Any:
    """Wait until an uploaded file is processed or fail on timeout/terminal error."""
    _validate_wait(poll_interval_seconds, timeout_seconds)
    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        uploaded = _retrieve_with_retry(
            lambda: client.files.retrieve(file_id),
            resource_name=f"File {file_id}",
            deadline=deadline,
        )
        current = _status(uploaded)
        if current in FILE_TERMINAL_STATES:
            if current != "processed":
                details = (
                    getattr(uploaded, "status_details", None)
                    or getattr(uploaded, "error", None)
                )
                raise RuntimeError(
                    f"Uploaded file ended in terminal state {current}: {details}"
                )
            return uploaded
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Uploaded file did not reach processed within {timeout_seconds:g} seconds."
            )
        time.sleep(min(poll_interval_seconds, timeout_seconds - elapsed))


def upload_file_and_wait(
    client: Any,
    path: str | Path,
    *,
    purpose: str,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 600,
) -> Any:
    """Upload a non-empty file and wait for its processed terminal state."""
    selected = Path(path)
    if not selected.is_file() or selected.stat().st_size == 0:
        raise ValueError(f"Upload source must be a non-empty file: {selected}")
    with selected.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose=purpose)
    return wait_for_file_processed(
        client,
        _require_id(uploaded, "File upload"),
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def monitor_fine_tuning_job(
    client: Any,
    job_id: str,
    *,
    poll_interval_seconds: float = 60,
    timeout_seconds: float = 86_400,
) -> Any:
    """Wait for a fine-tuning job to succeed or fail explicitly."""
    _validate_wait(poll_interval_seconds, timeout_seconds)
    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        job = _retrieve_with_retry(
            lambda: client.fine_tuning.jobs.retrieve(job_id),
            resource_name=f"Job {job_id}",
            deadline=deadline,
        )
        current = _status(job)
        if current in JOB_TERMINAL_STATES:
            if current != "succeeded":
                raise RuntimeError(
                    f"Fine-tuning job ended in terminal state {current}: "
                    f"{getattr(job, 'error', None)}"
                )
            return job
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Fine-tuning job did not reach a terminal state within "
                f"{timeout_seconds:g} seconds."
            )
        time.sleep(min(poll_interval_seconds, timeout_seconds - elapsed))
