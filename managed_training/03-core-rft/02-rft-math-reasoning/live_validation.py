from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rft_common import canonical_jsonl_bytes, create_job_once, upload_file_once

EXPECTED_HYPERPARAMETERS = {
    "batch_size": 128,
    "eval_interval": 25,
    "eval_samples": 2,
    "learning_rate_multiplier": 2.0,
    "n_epochs": 2,
    "reasoning_effort": "medium",
}

FILE_PREFIXES = {
    "training": "core-rft-openr1-math-training",
    "validation": "core-rft-openr1-math-validation",
}


def _write_submission_state(path: Path, state: dict[str, Any]) -> None:
    """Flush a same-directory temporary file before atomically replacing state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Windows supports fsync on the file, not on an opened directory.
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _submission_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "Math submission is locked. After a crash, reconcile saved state and "
            "remote operations and confirm no writer is active before removing the lock."
        ) from exc
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def _file_contract(prefix: str, path: Path) -> dict[str, Any]:
    content = canonical_jsonl_bytes(path.read_bytes())
    digest = hashlib.sha256(content).hexdigest()
    suffix = "".join(path.suffixes) or ".bin"
    return {
        "name": f"{prefix}-{digest[:12]}{suffix}",
        "sha256": digest,
        "bytes": len(content),
    }


def _resource_id(resource: Any) -> str:
    value = getattr(resource, "id", None)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Response has no resource ID; reconcile before resubmitting")
    return value


def submit_or_resume_math_job(
    client: Any,
    *,
    state_path: str | Path,
    project_endpoint: str,
    model: str,
    method: dict[str, Any],
    training_path: str | Path,
    validation_path: str | Path,
    existing_job_id: str | None = None,
) -> Any:
    """Persist intent before each POST and each returned ID before further work.

    This is a single-writer, one-job state file, compatible with POST_TRAINING's
    cancellation and file-cleanup examples. An interrupted upload is reconciled
    by unique canonical name and downloaded content. A job POST with no saved ID
    requires an explicitly verified existing_job_id; it is never sent again.
    """
    path = Path(state_path)
    paths = {"training": Path(training_path), "validation": Path(validation_path)}
    equivalence = {
        "project_endpoint": project_endpoint.strip(),
        "model": model,
        "method": method,
        "training_type": "GlobalStandard",
        "suffix": "core-rft-openr1-math",
        "files": {
            kind: _file_contract(FILE_PREFIXES[kind], source)
            for kind, source in paths.items()
        },
    }
    if not equivalence["project_endpoint"]:
        raise ValueError("project_endpoint is required")
    requested_job_id = (existing_job_id or "").strip() or None

    with _submission_lock(path):
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(state, dict)
                or state.get("schema_version") != 1
                or state.get("equivalence") != equivalence
            ):
                raise RuntimeError(
                    "Saved math submission does not match this project/recipe/data. "
                    "Retain it for reconciliation; do not overwrite it to rerun."
                )
            for key in ("training_file_id", "validation_file_id", "job_id"):
                if key not in state or (
                    state[key] is not None
                    and (not isinstance(state[key], str) or not state[key].strip())
                ):
                    raise RuntimeError("Saved math submission contains an invalid ID")
            if state.get("pending_operation") not in {
                None, "training_upload", "validation_upload", "job_create",
            }:
                raise RuntimeError("Saved math submission has an unknown operation")
        else:
            state = {
                "schema_version": 1,
                "equivalence": equivalence,
                "training_file_id": None,
                "validation_file_id": None,
                "job_id": None,
                "pending_operation": None,
            }
            _write_submission_state(path, state)

        def checkpoint(**updates: Any) -> None:
            nonlocal state
            candidate = {**state, **updates}
            _write_submission_state(path, candidate)
            state = candidate

        if requested_job_id and state["job_id"] not in {None, requested_job_id}:
            raise RuntimeError("Explicit job ID conflicts with the saved math job")
        job_id = state["job_id"] or requested_job_id
        expected = {"model": model, "method": method}
        for kind in paths:
            if state[f"{kind}_file_id"]:
                expected[f"{kind}_file"] = state[f"{kind}_file_id"]

        job = None
        if job_id:
            def never_create() -> Any:
                raise AssertionError("Existing-job recovery must not create a job")

            job = create_job_once(
                client, existing_job_id=job_id, expected=expected, create=never_create,
            )
            if _resource_id(job) != job_id:
                raise RuntimeError("Retrieved job does not match the requested job ID")
            identifiers = {
                f"{kind}_file_id": getattr(job, f"{kind}_file", None)
                for kind in paths
            }
            if any(not isinstance(value, str) or not value for value in identifiers.values()):
                raise RuntimeError("Existing job has no training/validation file IDs")
            checkpoint(job_id=job_id, pending_operation=None, **identifiers)
        elif state["pending_operation"] == "job_create":
            raise RuntimeError(
                "Job submission outcome is ambiguous. Reconcile in Foundry and set "
                "FOUNDRY_EXISTING_MATH_RFT_JOB_ID to the verified job ID. "
                "No job POST was repeated."
            )

        for kind, source in paths.items():
            pending = f"{kind}_upload"

            def before_create(pending: str = pending) -> None:
                if state["pending_operation"] is not None:
                    raise RuntimeError("Reconcile the pending operation before uploading")
                checkpoint(pending_operation=pending)

            def on_file(resource: Any, kind: str = kind, pending: str = pending) -> None:
                updates: dict[str, Any] = {f"{kind}_file_id": _resource_id(resource)}
                if state["pending_operation"] == pending:
                    updates["pending_operation"] = None
                checkpoint(**updates)

            upload_file_once(
                client,
                source,
                prefix=FILE_PREFIXES[kind],
                purpose="fine-tune",
                existing_file_id=state[f"{kind}_file_id"],
                reconcile_only=state["pending_operation"] == pending,
                before_create=before_create,
                on_file=on_file,
            )
            if _file_contract(FILE_PREFIXES[kind], source) != equivalence["files"][kind]:
                raise RuntimeError("Local dataset changed during submission; reconcile state")

        if job is not None:
            return job
        submission_client = client.with_options(max_retries=0)
        checkpoint(pending_operation="job_create")
        job = submission_client.fine_tuning.jobs.create(
            training_file=state["training_file_id"],
            validation_file=state["validation_file_id"],
            model=model,
            method=method,
            extra_body={"trainingType": "GlobalStandard"},
            suffix="core-rft-openr1-math",
        )
        checkpoint(job_id=_resource_id(job), pending_operation=None)
        return job


def assess_completed_job_reuse(
    job: Any,
    *,
    base_model: str,
    retrieve_file: Callable[[str], Any],
    retrieve_content: Callable[[str], bytes],
    training_sha256: str,
    validation_sha256: str,
    local_regrade_matches_service: bool,
) -> dict[str, Any]:
    reinforcement = getattr(getattr(job, "method", None), "reinforcement", None)
    hyperparameters = getattr(reinforcement, "hyperparameters", None)
    actual_hyperparameters = {
        name: getattr(hyperparameters, name, None)
        for name in EXPECTED_HYPERPARAMETERS
    }
    training_file = retrieve_file(job.training_file)
    validation_file = retrieve_file(job.validation_file)
    training_content = retrieve_content(job.training_file)
    validation_content = retrieve_content(job.validation_file)

    checks = {
        "terminal_success": job.status == "succeeded",
        "base_model": job.model == base_model,
        "method": getattr(getattr(job, "method", None), "type", None)
        == "reinforcement",
        "hyperparameters": actual_hyperparameters == EXPECTED_HYPERPARAMETERS,
        "training_bytes": training_file.bytes == len(training_content),
        "validation_bytes": validation_file.bytes == len(validation_content),
        "training_sha256": hashlib.sha256(canonical_jsonl_bytes(training_content)).hexdigest()
        == training_sha256,
        "validation_sha256": hashlib.sha256(canonical_jsonl_bytes(validation_content)).hexdigest()
        == validation_sha256,
        "local_regrade_matches_service": local_regrade_matches_service,
    }
    return {
        "reusable": all(checks.values()),
        "checks": checks,
        "actual_hyperparameters": actual_hyperparameters,
    }


def summarize_result_csv(content: str) -> dict[str, list[float]]:
    rows = list(csv.DictReader(io.StringIO(content)))

    def values(column: str) -> list[float]:
        return [
            float(row[column])
            for row in rows
            if row.get(column) not in {None, ""}
        ]

    return {
        "training_reward": values("train_mean_reward"),
        "validation_reward": values("full_valid_mean_reward"),
        "policy_gradient_loss_sum": values("pg_loss_sum"),
        "total_loss_sum": values("total_loss_sum"),
        "entropy_loss_sum": values("entropy_loss_sum"),
    }
