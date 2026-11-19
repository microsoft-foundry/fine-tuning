"""Countdown-only cancellation safeguards; no authentication or implicit actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


_RECONCILE = (
    "Cancellation identity is incomplete or mismatched; no cancellation was sent. "
    "Retain the original state. Independently inspect the project, job ID, input "
    "files, suffix, and full method/grader in Foundry and service request/audit "
    "evidence. Do not fill missing identity from this state, .env, a deployment "
    "name, or the retrieved job itself. For legacy/incomplete records, use the "
    "separate, explicitly confirmed Foundry cancellation workflow only after "
    "independently proving the intended owned job; otherwise contact the operator."
)


def _metadata(value: Any) -> Any:
    if callable(getattr(value, "model_dump", None)):
        value = value.model_dump()
    elif not isinstance(value, (Mapping, list, tuple)) and hasattr(value, "__dict__"):
        value = {key: item for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, Mapping):
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    return value


def _matches_metadata(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and all(key in actual and _matches_metadata(actual[key], item)
                    for key, item in expected.items())
            # SDK optional fields may be absent from the request, but a different
            # observable non-null recipe field must not be silently ignored.
            and all(key in expected or item is None for key, item in actual.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches_metadata(left, right)
                    for left, right in zip(actual, expected))
        )
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return actual == expected


def _exact_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def validate_countdown_cancellation_target(
    current: Any,
    *,
    state: dict[str, Any],
    label: str,
    project_endpoint: str,
    confirmed_project_endpoint: str,
    confirmed_job_id: str,
    confirmed_label: str,
) -> str:
    """Require independent confirmation AND complete saved/live recipe identity.

    Confirmation arguments must be obtained independently by the caller, not
    copied from state. This function never repairs or upgrades old evidence.
    """
    try:
        if label not in {"model-grader", "python-grader"} or confirmed_label != label:
            raise ValueError("Select and confirm exactly one Countdown variant")
        if (
            not _exact_text(project_endpoint)
            or not _exact_text(confirmed_project_endpoint)
            or project_endpoint.rstrip("/") != confirmed_project_endpoint.rstrip("/")
            or not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != 2
        ):
            raise ValueError("Independent project confirmation and current intent are required")
        job_id = state["jobs"][label]
        if not _exact_text(job_id) or not _exact_text(confirmed_job_id) or job_id != confirmed_job_id:
            raise ValueError("Independently confirmed job ID does not match the selected label")
        intent = state["pending"][label]
        if (
            type(intent["schema_version"]) is not int
            or intent["schema_version"] != 1
            or intent["record_type"] != "job_creation_intent"
            or intent["label"] != label
            or not _exact_text(intent["project_endpoint"])
            or intent["project_endpoint"].rstrip("/") != project_endpoint.rstrip("/")
            or not _exact_text(state["fingerprint"])
            or intent["run_fingerprint"] != state["fingerprint"]
        ):
            raise ValueError("Saved intent identity does not match the selected job/project")
        saved_endpoint = state.get("job_endpoints", {}).get(label)
        if saved_endpoint is not None and saved_endpoint != intent["project_endpoint"]:
            raise ValueError("Saved endpoint evidence conflicts")
        payload = intent["payload"]
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if intent["payload_sha256"] != digest:
            raise ValueError("Saved payload digest does not match")
        if payload["model"] != "qwen3.6-35b-a3b":
            raise ValueError("Saved model is not the Countdown model")
        for field in ("training_file", "validation_file", "suffix"):
            if not _exact_text(payload[field]):
                raise ValueError("Saved file/suffix identity is incomplete")
        for kind in ("training", "validation"):
            evidence = intent["files"][kind]
            if (
                evidence["file_id"] != payload[f"{kind}_file"]
                or type(evidence["bytes"]) is not int or evidence["bytes"] <= 0
                or type(evidence["rows"]) is not int or evidence["rows"] <= 0
                or not isinstance(evidence["sha256"], str)
                or len(evidence["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in evidence["sha256"])
            ):
                raise ValueError("Saved file evidence is incomplete or inconsistent")
        method = payload["method"]
        reinforcement = method["reinforcement"]
        grader = reinforcement["grader"]
        expected_name, expected_type = {
            "model-grader": ("countdown_model_grader", "score_model"),
            "python-grader": ("countdown_python_grader", "python"),
        }[label]
        if (
            method["type"] != "reinforcement"
            or grader["name"] != expected_name or grader["type"] != expected_type
            or not isinstance(reinforcement["hyperparameters"], dict)
            or not reinforcement["hyperparameters"]
            or not isinstance(reinforcement["response_format"], dict)
            or not reinforcement["response_format"]
        ):
            raise ValueError("Saved method/grader identity is incomplete")
        if label == "python-grader":
            if not isinstance(grader["source"], str) or not grader["source"].strip():
                raise ValueError("Saved Python grader source is missing")
        elif (
            not _exact_text(grader["model"])
            or not isinstance(grader["input"], list) or not grader["input"]
            or grader["range"] != [0.0, 5.0] or grader["pass_threshold"] != 5
        ):
            raise ValueError("Saved model-grader contract is incomplete")
        actual = _metadata(current)
        if not isinstance(actual, dict) or actual.get("id") != job_id:
            raise ValueError("Retrieved job ID does not match")
        for field in ("model", "training_file", "validation_file", "method"):
            if field not in actual or not _matches_metadata(actual[field], payload[field]):
                raise ValueError(f"Retrieved {field} does not match the saved payload")
        suffixes = [actual[key] for key in ("suffix", "user_provided_suffix")
                    if actual.get(key) is not None]
        if not suffixes or any(value != payload["suffix"] for value in suffixes):
            raise ValueError("Retrieved suffix is absent or does not match")
        training_type = payload["extra_body"]["trainingType"]
        if training_type != "globalStandard":
            raise ValueError("Saved training type is not the Countdown recipe")
        for field in ("training_type", "trainingType"):
            if actual.get(field) is not None and actual[field] != training_type:
                raise ValueError("Retrieved training type does not match")
        if actual.get("extra_body") is not None and not _matches_metadata(
            actual["extra_body"], payload["extra_body"],
        ):
            raise ValueError("Retrieved extra request metadata does not match")
        if str(actual.get("status", "")).casefold() not in {
            "pending", "validating_files", "queued", "running",
        }:
            raise ValueError("Job is terminal, cancelling, or has an unknown status")
        return job_id
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError) as exc:
        raise RuntimeError(_RECONCILE) from exc


def cancel_countdown_job_once(client: Any, **confirmation: Any) -> Any:
    """Retrieve and verify one target before one non-retried cancellation POST."""
    state = confirmation.get("state", {})
    label = confirmation.get("label")
    jobs = state.get("jobs", {}) if isinstance(state, dict) else {}
    job_id = jobs.get(label) if isinstance(jobs, dict) else None
    if not _exact_text(job_id):
        raise RuntimeError(_RECONCILE)
    current = client.fine_tuning.jobs.retrieve(job_id)
    verified_id = validate_countdown_cancellation_target(current, **confirmation)
    return client.with_options(max_retries=0).fine_tuning.jobs.cancel(verified_id)