"""Single-writer, fail-closed submission journal for the three introductory SFTs.

Only reads may be repeated automatically. A pending POST with no recorded ID
requires operator reconciliation, not another POST or a list-based guess.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import load_foundry_config
from .experiment_manifest import _write_json_atomic, runtime_output_path

_ID_KEYS = ("training_file_id", "validation_file_id", "job_id")
_PENDING = {None, "training_upload", "validation_upload", "job_create"}
_STATE_KEYS = {
    "schema_version", "equivalence", "version_evidence", "pending_operation",
    "submitted_at_utc", *_ID_KEYS,
}


def model_version_evidence(expected_version: str) -> dict[str, Any]:
    """Describe an expectation, never turn request metadata into version proof.

    These OpenAI-compatible requests select a model name, not a version. No
    documented base-model-version response field is assumed by this helper.
    In particular, caller metadata and the fine-tuned artifact ID are not proof.
    """
    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError("An expected model version is required")
    return {
        "expected_model_version": expected_version,
        "actual_model_version": None,
        "model_version_verification": "unverified",
    }


def _field(resource: Any, name: str) -> Any:
    return resource.get(name) if isinstance(resource, Mapping) else getattr(resource, name, None)


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _resource_id(resource: Any) -> str:
    value = _field(resource, "id")
    if not _valid_id(value):
        raise RuntimeError("No valid response ID was recorded; retain and reconcile pending state")
    return value


@contextmanager
def _submission_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(
            "Submission is locked. Confirm no writer is active and reconcile local/remote "
            "state before removing a stale submission-state.json.lock."
        ) from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def _load_state(path: Path, equivalence: dict[str, Any]) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("Saved submission state is unreadable; retain it for reconciliation") from None
    if (
        not isinstance(state, dict) or set(state) != _STATE_KEYS
        or state.get("schema_version") != 1 or state.get("equivalence") != equivalence
        or state.get("version_evidence") != model_version_evidence(equivalence["expected_model_version"])
    ):
        raise RuntimeError(
            "Saved state uses another recipe/project/data or an older format. Retain and "
            "reconcile it; do not overwrite or archive it merely to repeat a submission."
        )
    if any(state[key] is not None and not _valid_id(state[key]) for key in _ID_KEYS):
        raise RuntimeError("Saved submission state contains an invalid resource ID")
    pending = state["pending_operation"]
    if not isinstance(pending, (str, type(None))) or pending not in _PENDING:
        raise RuntimeError("Saved submission state contains an unknown pending operation")
    if (
        (state["validation_file_id"] and not state["training_file_id"])
        or (state["job_id"] and not all(state[key] for key in _ID_KEYS[:2]))
        or (pending == "training_upload" and any(state[key] for key in _ID_KEYS))
        or (pending == "validation_upload" and (
            not state["training_file_id"] or state["validation_file_id"] or state["job_id"]
        ))
        or (pending == "job_create" and (
            not all(state[key] for key in _ID_KEYS[:2]) or state["job_id"]
        ))
        or (state["submitted_at_utc"] is not None and not isinstance(state["submitted_at_utc"], str))
    ):
        raise RuntimeError("Saved submission state contains inconsistent progress")
    return state


def _verify_file(
    client: Any, file_id: str, contract: dict[str, Any], content: bytes,
    *, poll_interval_seconds: float, timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        uploaded = client.files.retrieve(file_id, extra_headers={"Accept-Encoding": "identity"})
        if (
            _resource_id(uploaded) != file_id
            or _field(uploaded, "filename") != contract["filename"]
            or _field(uploaded, "purpose") != "fine-tune"
        ):
            raise RuntimeError("Saved file does not match this upload contract; retain state")
        status = str(_field(uploaded, "status")).casefold()
        if status == "processed":
            break
        if status in {"error", "failed", "expired", "cancelled"}:
            raise RuntimeError("File processing failed; its ID remains saved for cleanup")
        if time.monotonic() >= deadline:
            raise TimeoutError("File processing timed out; its ID remains saved for resume/cleanup")
        time.sleep(poll_interval_seconds)
    response = client.files.content(file_id, extra_headers={"Accept-Encoding": "identity"})
    downloaded = response if isinstance(response, bytes) else response.read()
    if (
        not isinstance(downloaded, bytes)
        or _field(uploaded, "bytes") != len(downloaded)
        or downloaded.replace(b"\r\n", b"\n") != content
    ):
        raise RuntimeError("Saved file bytes do not match this dataset; retain state for reconciliation")


def _verify_job(job: Any, state: dict[str, Any]) -> None:
    if any(
        _field(job, name) != value
        for name, value in {
            "id": state["job_id"], "model": state["equivalence"]["model"],
            "training_file": state["training_file_id"],
            "validation_file": state["validation_file_id"],
        }.items()
    ):
        raise RuntimeError("Retrieved job does not match the saved model and file IDs")
    method = _field(job, "method")
    if hasattr(method, "model_dump"):
        method = method.model_dump(exclude_none=True)
    supervised = method.get("supervised") if isinstance(method, dict) else None
    actual = supervised.get("hyperparameters") if isinstance(supervised, dict) else None
    expected = state["equivalence"]["hyperparameters"]
    if (
        not isinstance(method, dict) or method.get("type") != "supervised"
        or not isinstance(actual, dict)
        or any(actual.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError("Retrieved job does not expose the requested supervised recipe")


def submit_or_resume_sft_job(
    client: Any,
    *,
    state_path: str | Path,
    outputs_root: str | Path,
    project_endpoint: str,
    model: str,
    expected_model_version: str,
    reference_region: str,
    training_type: str,
    hyperparameters: dict[str, Any],
    suffix: str,
    training_path: str | Path,
    validation_path: str | Path,
    expected_training_sha256: str,
    expected_validation_sha256: str,
    method_in_extra_body: bool = False,
    poll_interval_seconds: float = 10,
    file_timeout_seconds: float = 600,
) -> Any:
    """Record intent before every one-shot POST, and each ID before any read.

    Known IDs are reused and checked, including file bytes and the job recipe.
    Unknown acceptance blocks all subsequent creates until operator
    reconciliation. The top-level IDs and equivalence.project_endpoint retain
    the cancellation/cleanup shape documented in POST_TRAINING.md. This journal
    binds the requested recipe; it does not establish a pinned model version.
    """
    path = runtime_output_path(state_path, outputs_root=outputs_root)
    if path.name != "submission-state.json":
        raise ValueError("Use the demo's ignored outputs/submission-state.json")
    if poll_interval_seconds <= 0 or file_timeout_seconds <= 0:
        raise ValueError("File polling interval and timeout must be positive")
    endpoint = load_foundry_config(environ={"FOUNDRY_PROJECT_ENDPOINT": project_endpoint}).project_endpoint
    version = model_version_evidence(expected_model_version)
    if not all(isinstance(value, str) and value.strip() for value in (model, training_type, suffix)):
        raise ValueError("model, training_type, and suffix are required")
    if not hyperparameters or any(
        key not in {"n_epochs", "batch_size", "learning_rate_multiplier"}
        or isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0
        for key, value in hyperparameters.items()
    ):
        raise ValueError("Use only the introductory SFT numeric hyperparameters; never persist credentials")
    contents = {}
    contracts = {}
    for kind, source, expected_hash in (
        ("training", Path(training_path), expected_training_sha256),
        ("validation", Path(validation_path), expected_validation_sha256),
    ):
        content = source.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content).hexdigest()
        if not content or digest != expected_hash:
            raise ValueError("Dataset bytes do not match the pinned canonical hash")
        contents[kind] = content
        contracts[kind] = {"filename": source.name, "sha256": digest, "bytes": len(content)}
    method = {"type": "supervised", "supervised": {"hyperparameters": dict(hyperparameters)}}
    equivalence = {
        "project_endpoint": endpoint, "reference_region": reference_region,
        "model": model, "expected_model_version": expected_model_version,
        "model_version_verification": "unverified", "training_type": training_type,
        "hyperparameters": dict(hyperparameters), "suffix": suffix,
        "method_in_extra_body": method_in_extra_body, "files": contracts,
        "dataset_sha256": {kind: contract["sha256"] for kind, contract in contracts.items()},
    }
    with _submission_lock(path):
        if path.exists():
            state = _load_state(path, equivalence)
        else:
            state = {
                "schema_version": 1, "equivalence": equivalence, "version_evidence": version,
                "training_file_id": None, "validation_file_id": None, "job_id": None,
                "pending_operation": None, "submitted_at_utc": None,
            }
            _write_json_atomic(path, state)

        def checkpoint(**updates: Any) -> None:
            nonlocal state
            candidate = {**state, **updates}
            destination = runtime_output_path(path, outputs_root=outputs_root)
            _write_json_atomic(destination, candidate)
            state = candidate

        if state["pending_operation"] is not None:
            raise RuntimeError(
                "A previous POST has an ambiguous outcome. Retain pending state and reconcile "
                "in Foundry; no upload or job POST was repeated. Do not clear state to retry."
            )
        if state["job_id"]:
            job = client.fine_tuning.jobs.retrieve(state["job_id"])
            _verify_job(job, state)
            return job

        # Require the real SDK retry override; never silently fall back to a
        # retrying client for a mutation.
        submission_client = client.with_options(max_retries=0)
        for kind, content in contents.items():
            key = f"{kind}_file_id"
            if not state[key]:
                checkpoint(pending_operation=f"{kind}_upload")
                try:
                    uploaded = submission_client.files.create(
                        file=(contracts[kind]["filename"], content, "application/jsonl"),
                        purpose="fine-tune", extra_headers={"Accept-Encoding": "identity"},
                    )
                except Exception:
                    raise RuntimeError(
                        "Upload response was not recorded. Retain pending state and reconcile "
                        "in Foundry before any further submission."
                    ) from None
                checkpoint(**{key: _resource_id(uploaded), "pending_operation": None})
            _verify_file(
                client, state[key], contracts[kind], content,
                poll_interval_seconds=poll_interval_seconds, timeout_seconds=file_timeout_seconds,
            )

        checkpoint(pending_operation="job_create")
        payload = {
            "model": model, "training_file": state["training_file_id"],
            "validation_file": state["validation_file_id"], "suffix": suffix,
            "extra_body": {"trainingType": training_type},
        }
        if method_in_extra_body:
            payload["extra_body"]["method"] = method
        else:
            payload["method"] = method
        try:
            job = submission_client.fine_tuning.jobs.create(**payload)
        except Exception:
            raise RuntimeError(
                "Job response was not recorded. Retain pending state and reconcile in Foundry; "
                "do not repeat the job POST."
            ) from None
        checkpoint(
            job_id=_resource_id(job), pending_operation=None,
            submitted_at_utc=datetime.now(UTC).isoformat(),
        )
        _verify_job(job, state)
        return job