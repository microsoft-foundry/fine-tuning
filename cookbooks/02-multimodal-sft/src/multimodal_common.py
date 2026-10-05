from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

from PIL import Image, ImageOps

T = TypeVar("T")

FILE_TERMINAL_STATUSES = {"processed", "error", "expired", "failed", "cancelled"}
JOB_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
TRANSIENT_MARKERS = (
    "connection",
    "rate limit",
    "temporarily unavailable",
    "timeout",
    "too many requests",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def assert_disjoint(parts: dict[str, set[str]]) -> None:
    names = list(parts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = parts[left] & parts[right]
            if overlap:
                raise ValueError(
                    f"Split leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )


def image_uri(
    image: Image.Image,
    max_size: int | None,
    quality: int,
) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_size:
        image.thumbnail((max_size, max_size))
    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=quality,
        optimize=max_size is not None,
    )
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _status(value: Any) -> str:
    return str(getattr(value, "status", "")).strip().lower()


def _resource_id(value: Any, resource_name: str) -> str:
    resource_id = str(getattr(value, "id", "")).strip()
    if not resource_id:
        raise RuntimeError(f"{resource_name} response did not include an id.")
    return resource_id


def _is_transient(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in TRANSIENT_STATUS_CODES:
        return True
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) in TRANSIENT_STATUS_CODES:
        return True
    text = f"{type(error).__name__}: {error}".casefold()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def retry_call(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 5,
    timeout_seconds: float = 90,
) -> T:
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            elapsed = time.monotonic() - started
            if (
                not _is_transient(error)
                or attempt == max_attempts
                or elapsed >= timeout_seconds
            ):
                raise RuntimeError(
                    f"{operation_name} failed after {attempt} attempt(s)."
                ) from error
            delay = min(2 ** (attempt - 1), 20) * random.uniform(0.8, 1.2)
            if elapsed + delay >= timeout_seconds:
                raise TimeoutError(
                    f"{operation_name} could not retry within "
                    f"{timeout_seconds:g} seconds."
                ) from error
            print(
                {
                    "operation": operation_name,
                    "retry": attempt,
                    "delay_seconds": round(delay, 1),
                }
            )
            time.sleep(delay)
    raise AssertionError("Retry loop ended unexpectedly.")


def get_clients(project_endpoint: str) -> tuple[Any, Any]:
    if not project_endpoint:
        raise RuntimeError("Set FOUNDRY_PROJECT_ENDPOINT first.")
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    project = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )
    return project, project.get_openai_client()


def close_clients(project: Any, client: Any) -> None:
    for resource in (client, project):
        close = getattr(resource, "close", None)
        if callable(close):
            close()


@contextmanager
def project_clients(project_endpoint: str) -> Iterator[tuple[Any, Any]]:
    project, client = get_clients(project_endpoint)
    try:
        yield project, client
    finally:
        close_clients(project, client)


def wait_for_file(
    client: Any,
    file_id: str,
    *,
    poll_interval_seconds: float = 10,
    timeout_seconds: float = 600,
) -> Any:
    started = time.monotonic()
    while True:
        uploaded = retry_call(
            lambda: client.files.retrieve(file_id),
            operation_name="retrieve uploaded file",
        )
        status = _status(uploaded)
        print({"file": file_id, "status": status})
        if status in FILE_TERMINAL_STATUSES:
            if status != "processed":
                details = getattr(uploaded, "status_details", None)
                raise RuntimeError(
                    f"File {file_id} ended in terminal state {status}: {details}"
                )
            return uploaded
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"File {file_id} was not processed within "
                f"{timeout_seconds:g} seconds."
            )
        time.sleep(poll_interval_seconds)


def wait_for_job(
    client: Any,
    job_id: str,
    *,
    poll_interval_seconds: float = 60,
    timeout_seconds: float = 24 * 60 * 60,
) -> Any:
    started = time.monotonic()
    while True:
        job = retry_call(
            lambda: client.fine_tuning.jobs.retrieve(job_id),
            operation_name="retrieve fine-tuning job",
        )
        status = _status(job)
        print({"job": _resource_id(job, "Fine-tuning job"), "status": status})
        if status in JOB_TERMINAL_STATUSES:
            if status != "succeeded":
                raise RuntimeError(
                    f"Fine-tuning job {job_id} ended in terminal state "
                    f"{status}: {getattr(job, 'error', None)}"
                )
            return job
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Fine-tuning job {job_id} did not finish within "
                f"{timeout_seconds:g} seconds."
            )
        time.sleep(poll_interval_seconds)


def _iter_page(page: Any) -> list[Any]:
    return list(getattr(page, "data", page))


def upload_or_reuse_file(
    client: Any,
    path: Path,
    *,
    timeout_seconds: float = 600,
) -> Any:
    selected = Path(path)
    if not selected.is_file() or not selected.stat().st_size:
        raise FileNotFoundError(selected)
    remote_name = (
        f"{selected.stem}-{sha256(selected)[:16]}"
        f"{selected.suffix or '.jsonl'}"
    )
    existing = _iter_page(
        retry_call(client.files.list, operation_name="list uploaded files")
    )
    for candidate in existing:
        if str(getattr(candidate, "filename", "")) == remote_name:
            print({"file": remote_name, "action": "reuse"})
            return wait_for_file(
                client,
                _resource_id(candidate, "File"),
                timeout_seconds=timeout_seconds,
            )

    def upload() -> Any:
        with selected.open("rb") as stream:
            return client.files.create(
                file=(remote_name, stream, "application/jsonl"),
                purpose="fine-tune",
            )

    print({"file": remote_name, "action": "upload"})
    uploaded = retry_call(upload, operation_name="upload fine-tuning file")
    return wait_for_file(
        client,
        _resource_id(uploaded, "File"),
        timeout_seconds=timeout_seconds,
    )


def resolve_or_create_job(
    *,
    client: Any,
    reuse_id: str,
    model: str,
    train: Path,
    validation: Path,
    suffix: str,
    seed: int,
    training_type: str,
    run_paid_jobs: bool,
    file_timeout_seconds: float = 600,
    job_timeout_seconds: float = 24 * 60 * 60,
) -> Any | None:
    if reuse_id:
        return wait_for_job(
            client,
            reuse_id,
            timeout_seconds=job_timeout_seconds,
        )
    if not run_paid_jobs:
        print("Paid training disabled; exact inputs:", train, validation)
        return None
    training = upload_or_reuse_file(
        client,
        train,
        timeout_seconds=file_timeout_seconds,
    )
    validation_file = upload_or_reuse_file(
        client,
        validation,
        timeout_seconds=file_timeout_seconds,
    )
    job = retry_call(
        lambda: client.fine_tuning.jobs.create(
            model=model,
            training_file=_resource_id(training, "Training file"),
            validation_file=_resource_id(validation_file, "Validation file"),
            seed=seed,
            suffix=suffix,
            method={
                "type": "supervised",
                "supervised": {"hyperparameters": {"n_epochs": 1}},
            },
            extra_body={"trainingType": training_type},
        ),
        operation_name="create fine-tuning job",
    )
    return wait_for_job(
        client,
        _resource_id(job, "Fine-tuning job"),
        timeout_seconds=job_timeout_seconds,
    )


def vision_chat(
    client: Any,
    deployment: str,
    system: str,
    text: str,
    images: list[str],
    max_tokens: int = 80,
) -> str:
    content = [{"type": "text", "text": text}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": uri, "detail": "low"},
        }
        for uri in images
    )
    response = retry_call(
        lambda: client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=max_tokens,
        ),
        operation_name=f"invoke deployment {deployment}",
    )
    return (response.choices[0].message.content or "").strip()


def paid_jobs_enabled() -> bool:
    return os.getenv("FOUNDRY_RUN_PAID_JOBS", "false").strip().lower() == "true"


def live_evaluation_enabled() -> bool:
    return (
        os.getenv("FOUNDRY_RUN_LIVE_EVALUATION", "false").strip().lower()
        == "true"
    )
