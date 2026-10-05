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
            content = content.replace(b"\r\n", b"\n")
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
    suffix = "".join(source.suffixes) or ".bin"
    return f"{prefix}-{sha256_file(source)[:12]}{suffix}"


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
    retryable = retryable or (
        lambda exc: getattr(exc, "status_code", None) == 429
        or (getattr(exc, "status_code", 0) or 0) >= 500
    )
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


def upload_file_once(client: Any, path: str | Path, *, prefix: str, purpose: str) -> Any:
    source = Path(path)
    remote_name = content_addressed_name(prefix, source)
    existing = retry(
        lambda: _page_items(client.files.list()),
        description=f"list files before uploading {remote_name}",
    )
    matches = [item for item in existing if item.filename == remote_name]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple remote files match {remote_name}")
    if matches:
        match = matches[0]
        retry(
            lambda: client.files.wait_for_processing(match.id),
            description=f"process reused {remote_name}",
        )
        print(f"Reusing remote file {remote_name}")
        return match

    def create() -> Any:
        with source.open("rb") as handle:
            return client.files.create(
                file=(remote_name, handle, "application/jsonl"),
                purpose=purpose,
            )

    uploaded = retry(create, description=f"upload {remote_name}")
    retry(
        lambda: client.files.wait_for_processing(uploaded.id),
        description=f"process {remote_name}",
    )
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
