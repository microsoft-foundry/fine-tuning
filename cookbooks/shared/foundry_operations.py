"""Transparent wrappers around the OpenAI-compatible Foundry operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset_validation import hash_file
from .logging import get_logger, log_event, safe_reference
from .naming import slugify
from .retry import RetryPolicy, retry_call

FILE_TERMINAL_STATUSES = {"processed", "error", "expired", "failed", "cancelled"}
JOB_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
REUSABLE_JOB_STATUSES = {
    "validating_files",
    "queued",
    "running",
    "succeeded",
}


@dataclass(frozen=True, slots=True)
class OperationResult:
    operation: str
    resource_id: str
    name: str
    status: str
    reused: bool
    elapsed_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


def _status(resource: Any) -> str:
    return str(getattr(resource, "status", "")).casefold()


def _resource_id(resource: Any, resource_name: str) -> str:
    value = getattr(resource, "id", None)
    if not value:
        raise RuntimeError(f"{resource_name} response did not contain an id")
    return str(value)


def _iter_page(page: Any) -> list[Any]:
    data = getattr(page, "data", page)
    return list(data)


def _remote_filename(demo_slug: str, purpose: str, path: Path) -> str:
    digest = hash_file(path)[:12]
    suffix = path.suffix.casefold() or ".jsonl"
    return f"{slugify(demo_slug, max_length=30)}-{slugify(purpose, max_length=20)}-{digest}{suffix}"


def wait_for_file(
    client: Any,
    file_id: str,
    *,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 600,
) -> Any:
    started = time.monotonic()
    last_status: str | None = None
    logger = get_logger()
    while True:
        uploaded = retry_call(
            lambda: client.files.retrieve(file_id),
            policy=RetryPolicy(timeout_seconds=min(timeout_seconds, 90)),
            operation_name="retrieve uploaded file",
        )
        current = _status(uploaded)
        if current != last_status:
            log_event(
                logger,
                operation="file-upload",
                state=current or "unknown",
                message="Uploaded file state changed",
                fields={"reference": safe_reference(file_id)},
            )
            last_status = current
        if current in FILE_TERMINAL_STATUSES:
            if current != "processed":
                details = getattr(uploaded, "status_details", None)
                raise RuntimeError(
                    f"Uploaded file ended in terminal state {current}: {details}"
                )
            return uploaded
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Uploaded file {safe_reference(file_id)} was not processed within "
                f"{timeout_seconds:g} seconds"
            )
        time.sleep(poll_interval_seconds)


def upload_or_reuse_file(
    client: Any,
    *,
    demo_slug: str,
    purpose_name: str,
    path: str | Path,
    api_purpose: str = "fine-tune",
    reuse: bool = True,
    wait: bool = True,
    timeout_seconds: float = 600,
) -> OperationResult:
    """Reuse an exact content-addressed filename or upload the local file."""
    selected = Path(path)
    if not selected.is_file() or selected.stat().st_size == 0:
        raise ValueError(f"Upload source must be a non-empty file: {selected}")
    remote_name = _remote_filename(demo_slug, purpose_name, selected)
    started = time.monotonic()

    if reuse:
        candidates = _iter_page(
            retry_call(
                client.files.list,
                operation_name="list uploaded files",
            )
        )
        matching = [
            item
            for item in candidates
            if getattr(item, "filename", None) == remote_name
            and getattr(item, "purpose", api_purpose) == api_purpose
        ]
        if len(matching) > 1:
            raise RuntimeError(
                f"Multiple remote files match content-addressed name {remote_name}"
            )
        if matching:
            uploaded = matching[0]
            if wait and _status(uploaded) != "processed":
                uploaded = wait_for_file(
                    client,
                    _resource_id(uploaded, "File"),
                    timeout_seconds=timeout_seconds,
                )
            return OperationResult(
                operation="file-upload",
                resource_id=_resource_id(uploaded, "File"),
                name=remote_name,
                status=_status(uploaded) or "available",
                reused=True,
                elapsed_seconds=time.monotonic() - started,
                details={"sha256": hash_file(selected), "bytes": selected.stat().st_size},
            )

    def upload() -> Any:
        with selected.open("rb") as stream:
            return client.files.create(
                file=(remote_name, stream, "application/jsonl"),
                purpose=api_purpose,
            )

    uploaded = retry_call(
        upload,
        operation_name="upload file",
    )
    file_id = _resource_id(uploaded, "File")
    if wait:
        uploaded = wait_for_file(
            client,
            file_id,
            timeout_seconds=timeout_seconds,
        )
    return OperationResult(
        operation="file-upload",
        resource_id=file_id,
        name=remote_name,
        status=_status(uploaded) or "submitted",
        reused=False,
        elapsed_seconds=time.monotonic() - started,
        details={"sha256": hash_file(selected), "bytes": selected.stat().st_size},
    )


def _jobs(client: Any) -> list[Any]:
    return _iter_page(
        retry_call(
            client.fine_tuning.jobs.list,
            operation_name="list fine-tuning jobs",
        )
    )


def create_or_reuse_fine_tuning_job(
    client: Any,
    *,
    model: str,
    training_file_id: str,
    validation_file_id: str | None = None,
    suffix: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    method: dict[str, Any] | None = None,
    reuse: bool = True,
) -> OperationResult:
    """Create a job, or reuse one with the same immutable inputs and suffix."""
    if not model.strip() or not training_file_id.strip():
        raise ValueError("model and training_file_id are required")
    started = time.monotonic()
    normalized_suffix = slugify(suffix, max_length=40) if suffix else None
    if reuse:
        matches = [
            job
            for job in _jobs(client)
            if getattr(job, "model", None) == model
            and getattr(job, "training_file", None) == training_file_id
            and getattr(job, "validation_file", None) == validation_file_id
            and (
                normalized_suffix is None
                or getattr(job, "suffix", None) == normalized_suffix
            )
            and _status(job) in REUSABLE_JOB_STATUSES
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple reusable fine-tuning jobs match the requested inputs; "
                "provide a unique suffix or disable reuse"
            )
        if matches:
            job = matches[0]
            return OperationResult(
                operation="fine-tuning-job",
                resource_id=_resource_id(job, "Fine-tuning job"),
                name=normalized_suffix or model,
                status=_status(job),
                reused=True,
                elapsed_seconds=time.monotonic() - started,
            )

    payload: dict[str, Any] = {
        "model": model,
        "training_file": training_file_id,
    }
    if validation_file_id:
        payload["validation_file"] = validation_file_id
    if normalized_suffix:
        payload["suffix"] = normalized_suffix
    if hyperparameters is not None:
        payload["hyperparameters"] = hyperparameters
    if method is not None:
        payload["method"] = method
    job = retry_call(
        lambda: client.fine_tuning.jobs.create(**payload),
        operation_name="create fine-tuning job",
    )
    return OperationResult(
        operation="fine-tuning-job",
        resource_id=_resource_id(job, "Fine-tuning job"),
        name=normalized_suffix or model,
        status=_status(job) or "submitted",
        reused=False,
        elapsed_seconds=time.monotonic() - started,
    )


def monitor_fine_tuning_job(
    client: Any,
    job_id: str,
    *,
    poll_interval_seconds: float = 60,
    timeout_seconds: float = 24 * 60 * 60,
    require_success: bool = True,
) -> Any:
    """Poll a fine-tuning job to a terminal state with an explicit timeout."""
    started = time.monotonic()
    last_status: str | None = None
    logger = get_logger()
    while True:
        job = retry_call(
            lambda: client.fine_tuning.jobs.retrieve(job_id),
            operation_name="retrieve fine-tuning job",
        )
        current = _status(job)
        if current != last_status:
            log_event(
                logger,
                operation="fine-tuning-job",
                state=current or "unknown",
                message="Fine-tuning job state changed",
                fields={"reference": safe_reference(job_id)},
            )
            last_status = current
        if current in JOB_TERMINAL_STATUSES:
            if require_success and current != "succeeded":
                error = getattr(job, "error", None)
                raise RuntimeError(
                    f"Fine-tuning job ended in terminal state {current}: {error}"
                )
            return job
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Fine-tuning job {safe_reference(job_id)} did not finish within "
                f"{timeout_seconds:g} seconds"
            )
        time.sleep(poll_interval_seconds)
