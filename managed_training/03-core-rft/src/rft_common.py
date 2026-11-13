from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
_REWARD_PATTERN = re.compile(
    r"(?:validation|training)(?: mean)? reward\s*=\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOSS_PATTERN = re.compile(
    r"(?:validation|training)(?: mean)? loss\s*=\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def require_env(*names: str) -> dict[str, str]:
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Set the required environment variables: " + ", ".join(missing)
        )
    return values


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value, received {value!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_jsonl_bytes(content: bytes) -> bytes:
    """Match the manifests' CRLF-to-LF rule, leaving JSON string escapes intact."""
    return content.replace(b"\r\n", b"\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def validate_hash_manifest(manifest_path: str | Path, base_dir: str | Path) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(base_dir)
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        content = path.read_bytes()
        if manifest.get("normalization", "").startswith("CRLF-to-LF"):
            content = canonical_jsonl_bytes(content)
        actual = hashlib.sha256(content).hexdigest()
        expected = artifact["sha256"]
        if actual != expected:
            raise AssertionError(
                f"Hash mismatch for {path}: expected {expected}, received {actual}"
            )
        if "bytes" in artifact and len(content) != artifact["bytes"]:
            raise AssertionError(
                f"Byte count mismatch for {path}: expected {artifact['bytes']}, "
                f"received {len(content)}"
            )


def content_addressed_name(prefix: str, path: str | Path) -> str:
    source = Path(path)
    content = canonical_jsonl_bytes(source.read_bytes())
    return _content_addressed_name(prefix, source, content)


def _content_addressed_name(prefix: str, source: Path, content: bytes) -> str:
    suffix = "".join(source.suffixes) or ".bin"
    digest = hashlib.sha256(content).hexdigest()
    return f"{prefix}-{digest[:12]}{suffix}"


def retry(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int = 3,
    initial_delay_seconds: float = 5,
    retryable: Callable[[Exception], bool] | None = None,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    retryable = retryable or _is_transient_service_error
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts or not retryable(exc):
                raise RuntimeError(
                    f"{description} failed on attempt {attempt}/{attempts}: {exc}"
                ) from exc
            delay = initial_delay_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, min(1.0, delay / 10))
            print(
                f"{description} hit a transient error; retrying in {delay:.1f}s "
                f"({attempt}/{attempts}): {exc}"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _is_transient_service_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (status_code or 0) >= 500:
        return True
    if status_code == 404:
        message = str(exc).casefold()
        return "project not found" in message or "no such file object" in message
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def upload_file_once(
    client: Any,
    path: str | Path,
    *,
    prefix: str,
    purpose: str,
    existing_file_id: str | None = None,
    reconcile_only: bool = False,
    before_create: Callable[[], None] | None = None,
    on_file: Callable[[Any], None] | None = None,
) -> Any:
    """Upload canonical bytes once; checkpoint IDs before processing or other reads.

    A caller recovering an interrupted POST must set ``reconcile_only`` until it
    has reconciled the outcome. An empty list is not proof that a POST failed.
    ``before_create`` must durably record intent, and ``on_file`` must durably
    record the returned ID. Callback failures propagate without another POST.
    """
    source = Path(path)
    content = canonical_jsonl_bytes(source.read_bytes())
    if not content:
        raise ValueError(f"Upload source is empty: {source}")
    remote_name = _content_addressed_name(prefix, source, content)
    if existing_file_id:
        uploaded = retry(
            lambda: client.files.retrieve(existing_file_id),
            description="retrieve saved RFT file",
        )
        if getattr(uploaded, "id", None) != existing_file_id:
            raise RuntimeError("Retrieved file does not match the saved file ID")
    else:
        existing = retry(
            lambda: _page_items(client.files.list()),
            description=f"list files before uploading {remote_name}",
        )
        matches = [item for item in existing if item.filename == remote_name]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple remote files match {remote_name}")
        if matches:
            uploaded = matches[0]
        else:
            if reconcile_only:
                raise RuntimeError(
                    "Upload outcome is ambiguous and no unique file is visible. "
                    "Reconcile in Foundry; no upload POST was repeated."
                )
            submission_client = client.with_options(max_retries=0)
            if before_create is not None:
                before_create()
            # Non-idempotent POST: neither this helper nor the SDK may retry it.
            uploaded = submission_client.files.create(
                file=(remote_name, content, "application/jsonl"), purpose=purpose,
            )
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("Upload response has no file ID; reconcile before resubmitting")
    if on_file is not None:
        on_file(uploaded)
    retry(
        lambda: client.files.wait_for_processing(file_id),
        description=f"process {remote_name}",
    )
    remote = retry(
        lambda: client.files.content(file_id), description="verify RFT file content",
    )
    downloaded = getattr(remote, "content", remote)
    if not isinstance(downloaded, bytes):
        raise TypeError("File content must be returned as bytes")
    if getattr(uploaded, "purpose", None) != purpose:
        raise RuntimeError("Remote file purpose does not match the requested purpose")
    if getattr(uploaded, "bytes", None) != len(downloaded):
        raise RuntimeError("Remote file byte count does not match its downloaded content")
    if canonical_jsonl_bytes(downloaded) != content:
        raise RuntimeError("Remote file content does not match the canonical local data")
    return uploaded


def _page_items(page: Any) -> list[Any]:
    return list(page) if hasattr(page, "has_next_page") else list(getattr(page, "data", page))


def _matches_recipe(actual: Any, expected: Any) -> bool:
    if hasattr(actual, "model_dump"):
        actual = actual.model_dump()
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_recipe(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def create_job_once(
    client: Any,
    *,
    existing_job_id: str | None,
    expected: dict[str, Any],
    create: Callable[[], Any],
) -> Any:
    if existing_job_id:
        job = retry(
            lambda: client.fine_tuning.jobs.retrieve(existing_job_id),
            description=f"retrieve existing job {existing_job_id}",
        )
        if job.status not in {
            "pending", "validating_files", "queued", "running", "succeeded"
        }:
            raise RuntimeError(
                f"Existing job {existing_job_id} cannot be reused ({job.status}). "
                "Clear the existing-job setting only to intentionally submit a new job."
            )
        mismatches = [
            key for key, value in expected.items()
            if not _matches_recipe(getattr(job, key, None), value)
        ]
        if mismatches:
            raise RuntimeError(
                f"Existing job {existing_job_id} does not match the requested recipe: "
                + ", ".join(mismatches)
            )
        print(f"Reusing job {existing_job_id} ({job.status})")
        return job
    return create()


def wait_for_job(
    client: Any,
    job_id: str,
    *,
    timeout_seconds: int = 14_400,
    poll_seconds: int = 60,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    previous_status = None
    while time.monotonic() < deadline:
        job = retry(
            lambda: client.fine_tuning.jobs.retrieve(job_id),
            description=f"poll job {job_id}",
        )
        if job.status != previous_status:
            print(f"{job_id}: {job.status}")
            previous_status = job.status
        if job.status in TERMINAL_JOB_STATUSES:
            if job.status != "succeeded":
                raise RuntimeError(f"RFT job {job_id} ended with {job.status}: {job.error}")
            return job
        time.sleep(poll_seconds)
    raise TimeoutError(f"RFT job {job_id} exceeded {timeout_seconds} seconds")


def collect_new_events(client: Any, job_id: str, seen_ids: set[str]) -> list[Any]:
    events = retry(
        lambda: list(
            client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id,
                limit=100,
            ).data
        ),
        description=f"list events for {job_id}",
    )
    new_events = [event for event in reversed(events) if event.id not in seen_ids]
    seen_ids.update(event.id for event in new_events)
    return new_events


def extract_training_metrics(messages: Iterable[str]) -> dict[str, list[float]]:
    rewards: list[float] = []
    losses: list[float] = []
    for message in messages:
        rewards.extend(float(value) for value in _REWARD_PATTERN.findall(message))
        losses.extend(float(value) for value in _LOSS_PATTERN.findall(message))
    return {"reward": rewards, "loss": losses}
