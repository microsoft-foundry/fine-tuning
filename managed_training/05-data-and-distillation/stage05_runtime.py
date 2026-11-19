"""Deadline-bounded reads and single-writer submission journals for Stage 05."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FILE_TERMINAL_STATES = {"processed", "error", "expired", "failed", "cancelled", "canceled"}
JOB_TERMINAL_STATES = {"succeeded", "failed", "cancelled", "canceled"}
TRANSIENT_EXCEPTION_NAMES = {"APIConnectionError", "APITimeoutError"}


def _status(resource: Any) -> str:
    value = resource.get("status", "") if isinstance(resource, Mapping) else getattr(resource, "status", "")
    return str(value).split(".")[-1].casefold()


def _require_id(resource: Any, resource_name: str) -> str:
    resource_id = resource.get("id") if isinstance(resource, Mapping) else getattr(resource, "id", None)
    if not isinstance(resource_id, str) or not resource_id.strip() or resource_id != resource_id.strip():
        raise RuntimeError(
            f"{resource_name} response did not contain a valid id; retain pending state for reconciliation."
        )
    return resource_id


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


def _remaining_seconds(deadline: float, resource_name: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{resource_name} polling exceeded its timeout.")
    return remaining


def _retrieve_with_retry(
    operation: Callable[[float], Any],
    *,
    resource_name: str,
    deadline: float,
    attempts: int = 5,
) -> Any:
    for attempt in range(1, attempts + 1):
        remaining = _remaining_seconds(deadline, resource_name)
        try:
            result = operation(remaining)
        except Exception as exc:
            try:
                remaining = _remaining_seconds(deadline, resource_name)
            except TimeoutError as timeout:
                raise timeout from exc
            if attempt == attempts or not _is_transient(exc):
                raise
            delay = min(float(2 ** (attempt - 1)), remaining)
            print(
                f"{resource_name} polling hit a transient error; retrying in "
                f"{delay:g}s ({attempt}/{attempts}): {exc}"
            )
            # Logging and an oversleep consume the same budget as requests.
            try:
                time.sleep(min(delay, _remaining_seconds(deadline, resource_name)))
                _remaining_seconds(deadline, resource_name)
            except TimeoutError as timeout:
                raise timeout from exc
        else:
            # A terminal response received at/after expiry is not an on-time result.
            _remaining_seconds(deadline, resource_name)
            return result
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
    deadline = time.monotonic() + timeout_seconds
    resource_name = f"File {file_id}"
    polling_client = client.with_options(max_retries=0)
    while True:
        uploaded = _retrieve_with_retry(
            lambda remaining: polling_client.files.retrieve(file_id, timeout=min(remaining, timeout_seconds)),
            resource_name=resource_name,
            deadline=deadline,
        )
        current = _status(uploaded)
        _remaining_seconds(deadline, resource_name)
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
        time.sleep(min(poll_interval_seconds, _remaining_seconds(deadline, resource_name)))


def upload_file_and_wait(
    client: Any,
    path: str | Path,
    *,
    purpose: str,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 600,
) -> Any:
    """Upload a non-empty file and wait for its processed terminal state."""
    _validate_wait(poll_interval_seconds, timeout_seconds)
    selected = Path(path)
    if not selected.is_file() or selected.stat().st_size == 0:
        raise ValueError(f"Upload source must be a non-empty file: {selected}")
    submission_client = client.with_options(max_retries=0)
    with selected.open("rb") as handle:
        uploaded = submission_client.files.create(file=handle, purpose=purpose)
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
    deadline = time.monotonic() + timeout_seconds
    resource_name = f"Job {job_id}"
    polling_client = client.with_options(max_retries=0)
    while True:
        job = _retrieve_with_retry(
            lambda remaining: polling_client.fine_tuning.jobs.retrieve(
                job_id, timeout=min(remaining, timeout_seconds),
            ),
            resource_name=resource_name,
            deadline=deadline,
        )
        current = _status(job)
        _remaining_seconds(deadline, resource_name)
        if current in JOB_TERMINAL_STATES:
            if current != "succeeded":
                raise RuntimeError(
                    f"Fine-tuning job ended in terminal state {current}: "
                    f"{getattr(job, 'error', None)}"
                )
            return job
        time.sleep(min(poll_interval_seconds, _remaining_seconds(deadline, resource_name)))


def write_json_atomic(
    path: str | Path, value: dict[str, Any], *, outputs_root: str | Path,
) -> Path:
    """Reuse the shared same-directory replace, file fsync and directory fsync."""
    from shared.experiment_manifest import _write_json_atomic, runtime_output_path

    selected = runtime_output_path(path, outputs_root=outputs_root)
    return _write_json_atomic(selected, value)


@contextmanager
def submission_lock(path: str | Path, *, outputs_root: str | Path) -> Iterator[None]:
    """Hold the shared exclusive per-run lock before loading any submission state.

    A leftover lock is never removed automatically: it requires reconciliation.
    Imports are lazy so the polling helpers remain usable without SDK imports.
    """
    from shared.experiment_manifest import runtime_output_path
    from shared.sft_submission import _submission_lock

    selected = runtime_output_path(path, outputs_root=outputs_root)
    with _submission_lock(selected):
        yield


def _require_clear_pending(state: dict[str, Any]) -> None:
    if state.get("pending_operation") is not None or state.get("submission_pending"):
        raise RuntimeError(
            "A previous POST has an ambiguous outcome. Retain pending state and reconcile "
            "the upload or job in Foundry; no mutation will be repeated."
        )
    if state.get("schema_version") != 1 or "pending_operation" not in state:
        raise RuntimeError(
            "Saved submission state uses an older format without an upload journal. "
            "Retain all known IDs and reconcile it; do not clear state to repeat a submission."
        )


def load_run_state(
    path: str | Path,
    signature: dict[str, Any],
    *,
    file_names: tuple[str, str],
) -> dict[str, Any]:
    """Load only while holding submission_lock; unknown acceptance fails closed."""
    selected = Path(path)
    if not selected.exists():
        return {
            "schema_version": 1,
            "signature": json.loads(json.dumps(signature, allow_nan=False)),
            "pending_operation": None,
        }
    try:
        state = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("Saved submission state is unreadable; retain it for reconciliation.") from None
    if not isinstance(state, dict) or state.get("signature") != signature:
        raise RuntimeError(
            "This run ID belongs to different project/model/data settings or invalid state. "
            "Retain it for reconciliation rather than overwriting it."
        )
    _require_clear_pending(state)
    keys = tuple(f"{name}_file_id" for name in file_names)
    if set(state) - {"schema_version", "signature", "pending_operation", "job_id", *keys}:
        raise RuntimeError("Saved submission state has unknown fields; retain it for reconciliation.")
    for key in (*keys, "job_id"):
        if key in state:
            _require_id({"id": state[key]}, "Saved resource")
    if (state.get(keys[1]) and not state.get(keys[0])) or (
        state.get("job_id") and not all(state.get(key) for key in keys)
    ):
        raise RuntimeError("Saved submission progress is inconsistent; retain all known IDs.")
    return state


def _checkpoint_run_state(
    state: dict[str, Any],
    path: str | Path,
    outputs_root: str | Path,
    **updates: Any,
) -> None:
    candidate = {**state, **updates}
    write_json_atomic(path, candidate, outputs_root=outputs_root)
    # A failed persistence must not appear successful to this in-memory caller.
    state.clear()
    state.update(candidate)


def upload_file_once(
    client: Any,
    state: dict[str, Any],
    name: str,
    *,
    data_dir: str | Path,
    state_path: str | Path,
    outputs_root: str | Path,
    expected_sha256: str,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 600,
) -> Any:
    """Journal one upload and save its ID before polling, under submission_lock.

    Upload the immutable bytes whose hash was checked, never reopen a path after
    validation. Any exception, interruption, missing ID or failed ID checkpoint
    leaves durable pending intent, so a later run cannot blindly replay the POST.
    """
    _validate_wait(poll_interval_seconds, timeout_seconds)
    _require_clear_pending(state)
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Upload names must be plain filenames.")
    content = (Path(data_dir) / name).read_bytes()
    if not content or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{name}: dataset bytes changed after validation; no upload was attempted.")
    key = f"{name}_file_id"
    if not state.get(key):
        # Never fall back to a default client that might automatically replay a POST.
        submission_client = client.with_options(max_retries=0)
        _checkpoint_run_state(
            state, state_path, outputs_root, pending_operation=f"upload:{name}",
        )
        try:
            uploaded = submission_client.files.create(
                file=(name, content, "application/jsonl"), purpose="fine-tune",
            )
        except Exception:
            raise RuntimeError(
                "Upload response was not recorded. Retain pending state and reconcile "
                "in Foundry; do not repeat the upload POST."
            ) from None
        _checkpoint_run_state(
            state, state_path, outputs_root,
            **{key: _require_id(uploaded, "File upload"), "pending_operation": None},
        )
    return wait_for_file_processed(
        client, state[key], poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def create_job_once(
    client: Any,
    state: dict[str, Any],
    *,
    state_path: str | Path,
    outputs_root: str | Path,
    **payload: Any,
) -> str:
    """Journal a one-shot job POST and persist its ID before any monitoring read."""
    _require_clear_pending(state)
    if state.get("job_id"):
        return _require_id({"id": state["job_id"]}, "Saved job")
    submission_client = client.with_options(max_retries=0)
    _checkpoint_run_state(state, state_path, outputs_root, pending_operation="job_create")
    try:
        job = submission_client.fine_tuning.jobs.create(**payload)
    except Exception:
        raise RuntimeError(
            "Job response was not recorded. Retain pending state and reconcile "
            "in Foundry; do not repeat the job POST."
        ) from None
    job_id = _require_id(job, "Fine-tuning submission")
    _checkpoint_run_state(state, state_path, outputs_root, job_id=job_id, pending_operation=None)
    return job_id
