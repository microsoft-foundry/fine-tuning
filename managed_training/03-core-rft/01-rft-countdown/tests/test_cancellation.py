"""AST-selected local safeguards and strict fakes; no real cancellation/client."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guard():
    path = ROOT / "cancellation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_metadata", "_matches_metadata", "_exact_text",
        "validate_countdown_cancellation_target", "cancel_countdown_job_once",
    }
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_RECONCILE"
            for target in node.targets
        ))
    ]
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names
    namespace = {"Mapping": Mapping, "Any": Any, "hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["cancel_countdown_job_once"]


@pytest.fixture(params=["model-grader", "python-grader"])
def target(request):
    label = request.param
    grader = ({
        "name": "countdown_python_grader", "type": "python", "source": "def grade(s, i):\n    return 0\n",
    } if label == "python-grader" else {
        "name": "countdown_model_grader", "type": "score_model", "model": "offline-deployment",
        "input": [{"role": "user", "content": "offline scoring contract"}],
        "range": [0.0, 5.0], "pass_threshold": 5,
    })
    payload = {
        "model": "qwen3.6-35b-a3b", "training_file": "offline-training",
        "validation_file": "offline-validation", "suffix": f"countdown-{label}",
        "method": {"type": "reinforcement", "reinforcement": {
            "grader": grader, "hyperparameters": {"n_epochs": 1},
            "response_format": {"type": "json_schema", "json_schema": {"name": "offline"}},
        }},
        "extra_body": {"trainingType": "globalStandard"},
    }
    endpoint = "https://offline.services.ai.azure.com/api/projects/countdown"
    intent = {
        "schema_version": 1, "record_type": "job_creation_intent", "label": label,
        "project_endpoint": endpoint, "run_fingerprint": "offline-recipe",
        "payload": copy.deepcopy(payload),
        "payload_sha256": hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "files": {
            kind: {"file_id": payload[f"{kind}_file"], "bytes": 10, "rows": 1, "sha256": "a" * 64}
            for kind in ("training", "validation")
        },
    }
    state = {
        "schema_version": 2, "fingerprint": "offline-recipe",
        "jobs": {label: "offline-intended-job"}, "pending": {label: intent},
    }
    current = {key: copy.deepcopy(payload[key]) for key in (
        "model", "training_file", "validation_file", "suffix", "method",
    )}
    current.update(id="offline-intended-job", status="running")
    cancel = Mock(return_value=SimpleNamespace(status="cancelling"))
    retrieve = Mock(side_effect=lambda _job_id: current)
    client = SimpleNamespace(
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(
            retrieve=retrieve,
            cancel=Mock(side_effect=AssertionError("Cancellation must disable SDK retries")),
        )),
        with_options=Mock(return_value=SimpleNamespace(
            fine_tuning=SimpleNamespace(jobs=SimpleNamespace(cancel=cancel)),
        )),
    )
    confirmation = {
        "state": state, "label": label, "project_endpoint": endpoint,
        "confirmed_project_endpoint": endpoint, "confirmed_job_id": "offline-intended-job",
        "confirmed_label": label,
    }
    return SimpleNamespace(
        current=current, client=client, cancel=cancel, retrieve=retrieve,
        confirmation=confirmation, state=state, intent=intent, label=label,
    )


def test_exact_intent_and_independent_confirmations_allow_one_nonretried_cancel(guard, target):
    original = copy.deepcopy(target.state)
    result = guard(target.client, **target.confirmation)
    assert result.status == "cancelling"
    target.retrieve.assert_called_once_with("offline-intended-job")
    target.client.with_options.assert_called_once_with(max_retries=0)
    target.cancel.assert_called_once_with("offline-intended-job")
    assert target.state == original


def test_sdk_style_model_dump_and_nested_optional_fields_preserve_exact_identity(guard, target):
    actual = copy.deepcopy(target.current)
    actual["user_provided_suffix"] = actual.pop("suffix")
    actual["method"]["supervised"] = None
    actual["method"]["reinforcement"]["grader"]["optional_service_field"] = None
    target.retrieve.side_effect = lambda _job_id: SimpleNamespace(model_dump=lambda: actual)
    guard(target.client, **target.confirmation)
    target.cancel.assert_called_once_with("offline-intended-job")


@pytest.mark.parametrize("change", [
    "id", "model", "training_file", "validation_file", "suffix", "missing-suffix",
    "method", "grader-name", "grader-content", "hyperparameters", "response-format",
    "extra-method-field", "training-type", "terminal", "cancelling", "unknown-status",
])
def test_live_identity_mismatch_never_requests_cancellation(guard, target, change):
    if change in {"id", "model", "training_file", "validation_file", "suffix"}:
        target.current[change] = "offline-unrelated"
    elif change == "missing-suffix":
        target.current.pop("suffix")
    elif change == "method":
        target.current.pop("method")
    elif change == "grader-name":
        target.current["method"]["reinforcement"]["grader"]["name"] = "unrelated-grader"
    elif change == "grader-content":
        field = "source" if target.label == "python-grader" else "model"
        target.current["method"]["reinforcement"]["grader"][field] = "unrelated-content"
    elif change == "hyperparameters":
        target.current["method"]["reinforcement"]["hyperparameters"]["n_epochs"] = 2
    elif change == "response-format":
        target.current["method"]["reinforcement"].pop("response_format")
    elif change == "extra-method-field":
        target.current["method"]["reinforcement"]["other_recipe"] = "different"
    elif change == "training-type":
        target.current["trainingType"] = "different"
    else:
        target.current["status"] = {"terminal": "succeeded", "cancelling": "cancelling",
                                    "unknown-status": "unknown"}[change]
    original = copy.deepcopy(target.state)
    with pytest.raises(RuntimeError, match="Independently inspect"):
        guard(target.client, **target.confirmation)
    target.client.with_options.assert_not_called()
    target.cancel.assert_not_called()
    assert target.state == original


@pytest.mark.parametrize("field", [
    "confirmed_project_endpoint", "confirmed_job_id", "confirmed_label",
])
def test_confirmation_is_separate_from_saved_identity(guard, target, field):
    target.confirmation[field] = "independently-selected-different-target"
    with pytest.raises(RuntimeError, match="no cancellation was sent"):
        guard(target.client, **target.confirmation)
    target.cancel.assert_not_called()


def test_swapped_state_cannot_replace_independently_selected_job(guard, target):
    target.state["jobs"][target.label] = "offline-unrelated-job"
    target.current["id"] = "offline-unrelated-job"
    # The independently confirmed ID remains the intended job, not the swapped ID.
    with pytest.raises(RuntimeError, match="no cancellation was sent"):
        guard(target.client, **target.confirmation)
    target.cancel.assert_not_called()


def test_swapped_job_with_confirmed_id_still_cannot_bypass_saved_payload(guard, target):
    target.current["id"] = target.state["jobs"][target.label] = "offline-other-job"
    target.confirmation["confirmed_job_id"] = "offline-other-job"
    target.current["training_file"] = "unrelated-training-file"
    with pytest.raises(RuntimeError, match="no cancellation was sent"):
        guard(target.client, **target.confirmation)
    target.cancel.assert_not_called()


@pytest.mark.parametrize("change", ["legacy", "missing-pending", "missing-payload", "digest", "file-id"])
def test_incomplete_or_tampered_intent_requires_independent_workflow(guard, target, change):
    if change == "legacy":
        target.state.pop("schema_version")
        target.state.pop("pending")
    elif change == "missing-pending":
        target.state["pending"].clear()
    elif change == "missing-payload":
        target.intent.pop("payload")
    elif change == "digest":
        target.intent["payload_sha256"] = "0" * 64
    else:
        target.intent["files"]["training"]["file_id"] = "unrelated-file"
    original = copy.deepcopy(target.state)
    with pytest.raises(RuntimeError, match="legacy/incomplete"):
        guard(target.client, **target.confirmation)
    assert target.state == original
    target.cancel.assert_not_called()


def test_cancellation_failure_is_not_replayed(guard, target):
    target.cancel.side_effect = TimeoutError("offline ambiguous cancellation")
    with pytest.raises(TimeoutError):
        guard(target.client, **target.confirmation)
    target.cancel.assert_called_once_with("offline-intended-job")


def test_guide_uses_guard_and_independent_input_without_running_its_auth_imports():
    text = (ROOT / "CANCELLATION_AND_CLEANUP.md").read_text(encoding="utf-8")
    source = text.split("```python\n", 1)[1].split("```", 1)[0]
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    guarded = [node for node in calls if isinstance(node.func, ast.Name)
               and node.func.id == "cancel_countdown_job_once"]
    assert len(guarded) == 1
    arguments = {item.arg: item.value for item in guarded[0].keywords}
    assert isinstance(arguments["confirmed_job_id"], ast.Name)
    assert arguments["confirmed_job_id"].id == "verified_job_id"
    assert isinstance(arguments["confirmed_project_endpoint"], ast.Name)
    assert arguments["confirmed_project_endpoint"].id == "verified_endpoint"
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "cancel" for node in calls)
    assert "not** from the local state" in text
    assert "Do not synthesize" in text