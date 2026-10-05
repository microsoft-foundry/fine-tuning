"""Bounded runtime waits shared by the Stage 05 notebooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

FILE_TERMINAL_STATES = {"processed", "error", "expired", "failed", "cancelled"}
JOB_TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def _status(resource: Any) -> str:
    return str(getattr(resource, "status", "")).casefold()


def _require_id(resource: Any, resource_name: str) -> str:
    resource_id = getattr(resource, "id", None)
    if not resource_id:
        raise RuntimeError(f"{resource_name} response did not contain an id.")
    return str(resource_id)


def _validate_wait(poll_interval_seconds: float, timeout_seconds: float) -> None:
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")


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
    while True:
        uploaded = client.files.retrieve(file_id)
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
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
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
