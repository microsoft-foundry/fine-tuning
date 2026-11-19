"""Transparent wrappers around the OpenAI-compatible Foundry operations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset_validation import hash_file
from .logging import get_logger, log_event, safe_reference
from .naming import slugify
from .retry import RetryError, RetryPolicy, retry_call

FILE_TERMINAL_STATUSES = {"processed", "error", "expired", "failed", "cancelled"}
JOB_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
REUSABLE_JOB_STATUSES = {
    "pending",
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
    if hasattr(page, "has_next_page"):
        return list(page)
    return list(getattr(page, "data", page))


def _matches_configuration(actual: Any, requested: Any) -> bool:
    if hasattr(actual, "model_dump"):
        actual = actual.model_dump(exclude_none=True)
    if isinstance(requested, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_configuration(actual[key], value)
            for key, value in requested.items()
        ) and not any(
            key not in requested and value is not None
            for key, value in actual.items()
        )
    if isinstance(requested, list):
        return isinstance(actual, list) and len(actual) == len(requested) and all(
            _matches_configuration(value, expected)
            for value, expected in zip(actual, requested)
        )
    return actual == requested


def _submission_client(client: Any) -> Any:
    if callable(getattr(client, "with_options", None)):
        return client.with_options(max_retries=0)
    return client


def _remote_filename(demo_slug: str, purpose: str, path: Path) -> str:
    digest = hash_file(path)[:12]
    suffix = path.suffix.casefold() or ".jsonl"
    return f"{slugify(demo_slug, max_length=30)}-{slugify(purpose, max_length=20)}-{digest}{suffix}"


def _polling_deadline(poll_interval_seconds: float, timeout_seconds: float) -> float:
    for name, value in (
        ("poll_interval_seconds", poll_interval_seconds),
        ("timeout_seconds", timeout_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return time.monotonic() + timeout_seconds


def _remaining_seconds(deadline: float, timeout_message: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    return remaining


def _retrieve_with_deadline(
    retrieve: Callable[[float], Any],
    *,
    deadline: float,
    timeout_message: str,
    operation_name: str,
    policy: RetryPolicy,
) -> Any:
    def request() -> Any:
        remaining = _remaining_seconds(deadline, timeout_message)
        try:
            result = retrieve(remaining)
        except Exception as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(timeout_message) from error
            raise
        _remaining_seconds(deadline, timeout_message)
        return result

    try:
        return retry_call(
            request,
            policy=policy,
            deadline=deadline,
            operation_name=operation_name,
            sleep=time.sleep,
            clock=time.monotonic,
        )
    except RetryError as error:
        if time.monotonic() >= deadline:
            raise TimeoutError(timeout_message) from error
        raise


def wait_for_file(
    client: Any,
    file_id: str,
    *,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 600,
) -> Any:
    """Wait within one request/retry/polling budget; expiry does not delete files."""
    deadline = _polling_deadline(poll_interval_seconds, timeout_seconds)
    timeout_message = (
        f"Uploaded file {safe_reference(file_id)} was not processed within "
        f"{timeout_seconds:g} seconds"
    )
    polling_client = _submission_client(client)
    retrieve = polling_client.files.retrieve
    policy = RetryPolicy(timeout_seconds=timeout_seconds)
    last_status: str | None = None
    logger = get_logger()
    while True:
        uploaded = _retrieve_with_deadline(
            lambda remaining: retrieve(file_id, timeout=remaining),
            deadline=deadline,
            timeout_message=timeout_message,
            policy=policy,
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
        remaining = _remaining_seconds(deadline, timeout_message)
        if current in FILE_TERMINAL_STATUSES:
            if current != "processed":
                details = getattr(uploaded, "status_details", None)
                raise RuntimeError(
                    f"Uploaded file ended in terminal state {current}: {details}"
                )
            return uploaded
        time.sleep(min(poll_interval_seconds, remaining))


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
            return _submission_client(client).files.create(
                file=(remote_name, stream, "application/jsonl"),
                purpose=api_purpose,
            )

    uploaded = upload()
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


def _job_suffix(job: Any) -> str | None:
    suffix = getattr(job, "suffix", None)
    metadata = getattr(job, "metadata", None)
    if suffix is None and isinstance(metadata, dict):
        suffix = metadata.get("cookbook_suffix")
    return suffix


def _job_training_type(job: Any) -> str | None:
    value = getattr(job, "trainingType", None) or getattr(job, "training_type", None)
    return str(value).casefold() if value is not None else None


def _matches_job_recipe(
    job: Any,
    *,
    suffix: str | None,
    hyperparameters: dict[str, Any] | None,
    method: dict[str, Any] | None,
) -> bool:
    # An omitted method requests the service's default SFT recipe, not any
    # method. Resolved/default hyperparameters cannot be inferred reliably
    # from a list response, so decline reuse without an explicit method.
    if method is None or not method.get("type"):
        return False
    if hyperparameters is None:
        settings = method.get(method["type"])
        if not isinstance(settings, dict) or not settings.get("hyperparameters"):
            return False
    elif not hyperparameters:
        return False
    if not _matches_configuration(getattr(job, "method", None), method):
        return False
    if suffix is None:
        # A missing response field is not proof that no custom suffix was used.
        if not hasattr(job, "suffix") or _job_suffix(job) is not None:
            return False
    elif _job_suffix(job) != suffix:
        return False
    actual_hyperparameters = getattr(job, "hyperparameters", None)
    if hasattr(actual_hyperparameters, "model_dump"):
        actual_hyperparameters = actual_hyperparameters.model_dump(exclude_none=True)
    if hyperparameters is None:
        # Only the exact explicit method may supply settings in this case.
        # Materialized top-level defaults are uncertain, never wildcards.
        return actual_hyperparameters is None or actual_hyperparameters == {}
    return _matches_configuration(actual_hyperparameters, hyperparameters)


def create_or_reuse_fine_tuning_job(
    client: Any,
    *,
    model: str,
    training_file_id: str,
    validation_file_id: str | None = None,
    existing_job_id: str | None = None,
    suffix: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    method: dict[str, Any] | None = None,
    training_type: str | None = None,
    reuse: bool = True,
) -> OperationResult:
    """Reuse only an observable, exact recipe; uncertain defaults mean nonreuse.

    Omitting ``method`` still submits the service-default SFT request, but does
    not establish equivalence to a listed job (including a materialized SFT
    default). Resume an already recorded job by ID rather than guessing its
    defaults. Extra resolved settings also conservatively prevent list reuse.
    """
    if not model.strip() or not training_file_id.strip():
        raise ValueError("model and training_file_id are required")
    if training_type is not None and not training_type.strip():
        raise ValueError("training_type must be non-empty when supplied")
    started = time.monotonic()
    normalized_suffix = slugify(suffix, max_length=40) if suffix else None
    if existing_job_id:
        job = retry_call(
            lambda: client.fine_tuning.jobs.retrieve(existing_job_id),
            operation_name="retrieve recorded fine-tuning job",
        )
        expected = {
            "model": model,
            "training_file": training_file_id,
            "validation_file": validation_file_id,
        }
        if hyperparameters is not None:
            expected["hyperparameters"] = hyperparameters
        if method is not None:
            expected["method"] = method
        mismatches = [
            key for key, value in expected.items()
            if not _matches_configuration(getattr(job, key, None), value)
        ]
        if training_type is not None and _job_training_type(job) != training_type.casefold():
            mismatches.append("trainingType")
        if normalized_suffix is not None and _job_suffix(job) != normalized_suffix:
            mismatches.append("suffix")
        if mismatches:
            raise RuntimeError(
                "Recorded fine-tuning job does not match the requested recipe: "
                + ", ".join(mismatches)
            )
        if _status(job) not in REUSABLE_JOB_STATUSES:
            raise RuntimeError(
                f"Recorded fine-tuning job cannot be reused in state {_status(job)}"
            )
        return OperationResult(
            operation="fine-tuning-job",
            resource_id=_resource_id(job, "Fine-tuning job"),
            name=normalized_suffix or model,
            status=_status(job),
            reused=True,
            elapsed_seconds=time.monotonic() - started,
        )
    if reuse:
        matches = [
            job
            for job in _jobs(client)
            if getattr(job, "model", None) == model
            and getattr(job, "training_file", None) == training_file_id
            and getattr(job, "validation_file", None) == validation_file_id
            and _status(job) in REUSABLE_JOB_STATUSES
            and _matches_job_recipe(
                job,
                suffix=normalized_suffix,
                hyperparameters=hyperparameters,
                method=method,
            )
            and (
                training_type is None
                or _job_training_type(job) == training_type.casefold()
            )
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
        payload["metadata"] = {"cookbook_suffix": normalized_suffix}
    if hyperparameters is not None:
        payload["hyperparameters"] = hyperparameters
    if method is not None:
        payload["method"] = method
    if training_type is not None:
        payload["extra_body"] = {"trainingType": training_type}
    # An ambiguous POST failure must be reconciled, not automatically resubmitted.
    job = _submission_client(client).fine_tuning.jobs.create(**payload)
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
    """Poll within one request/retry/polling budget without cancelling the job."""
    deadline = _polling_deadline(poll_interval_seconds, timeout_seconds)
    timeout_message = (
        f"Fine-tuning job {safe_reference(job_id)} did not finish within "
        f"{timeout_seconds:g} seconds"
    )
    polling_client = _submission_client(client)
    retrieve = polling_client.fine_tuning.jobs.retrieve
    policy = RetryPolicy(max_attempts=10, timeout_seconds=timeout_seconds)
    last_status: str | None = None
    logger = get_logger()
    while True:
        job = _retrieve_with_deadline(
            lambda remaining: retrieve(job_id, timeout=remaining),
            deadline=deadline,
            timeout_message=timeout_message,
            operation_name="retrieve fine-tuning job",
            policy=policy,
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
        remaining = _remaining_seconds(deadline, timeout_message)
        if current in JOB_TERMINAL_STATUSES:
            if require_success and current != "succeeded":
                error = getattr(job, "error", None)
                raise RuntimeError(
                    f"Fine-tuning job ended in terminal state {current}: {error}"
                )
            return job
        time.sleep(min(poll_interval_seconds, remaining))
