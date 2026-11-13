"""Offline checks for the Countdown cancellation operator guide and job state."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shared import auth, config


DEMO_ROOT = Path(__file__).resolve().parents[2] / "03-core-rft" / "01-rft-countdown"
ENDPOINT = "https://countdown-offline.services.ai.azure.com/api/projects/offline"
JOB_ID = "offline-job-1"


def load_exact_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path.resolve()
    return module


@pytest.fixture
def countdown_contracts(tmp_path: Path):
    # Reuse the allowlisted actual notebook definitions, including the durable
    # pre-POST intent and lock; never execute a notebook cell or cloud import.
    contracts = load_exact_module(
        "countdown_state_contracts", DEMO_ROOT / "tests/test_countdown.py",
    )
    return contracts, contracts.load_contracts(tmp_path)


def test_countdown_submissions_keep_their_original_state_contract(
    tmp_path: Path, countdown_contracts,
) -> None:
    contracts, values = countdown_contracts
    jobs = contracts.attach_jobs(values)
    state_path = tmp_path / "submission-state.json"
    contracts.submit_variant(values)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["jobs"] == {"model-grader": "offline-job-1"}
    assert state["pending"]["model-grader"]["project_endpoint"] == values["PROJECT_ENDPOINT"]
    assert "job_endpoints" not in state

    # Reuse only independently verified legacy provenance, without rewriting it.
    legacy = {
        "fingerprint": values["RUN_FINGERPRINT"],
        "jobs": {"model-grader": JOB_ID},
        "job_endpoints": {"model-grader": values["PROJECT_ENDPOINT"]},
    }
    original = json.dumps(legacy, indent=4).encode("utf-8")
    state_path.write_bytes(original)
    assert contracts.submit_variant(values).id == JOB_ID
    assert state_path.read_bytes() == original
    assert len(jobs.payloads) == 1
    assert jobs.retrieved == [JOB_ID]


@pytest.fixture
def guide_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, countdown_contracts):
    contracts, values = countdown_contracts
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    state_path = outputs / "submission-state.json"
    values["STATE_PATH"] = state_path
    contracts.attach_jobs(values)
    for label in ("model-grader", "python-grader"):
        contracts.submit_variant(values, label)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["jobs"] == {"model-grader": JOB_ID, "python-grader": "offline-job-2"}
    assert values["PROJECT_ENDPOINT"] == ENDPOINT
    current = copy.deepcopy(state["pending"]["model-grader"]["payload"])
    current.update(id=JOB_ID, status="running")

    # Import only this demo's stdlib-only guard, never a cwd/sys.path namesake.
    guard = load_exact_module("countdown_guide_guard", DEMO_ROOT / "cancellation.py")
    guide_path = DEMO_ROOT / "CANCELLATION_AND_CLEANUP.md"
    guide = guide_path.read_text(encoding="utf-8")
    code = guide.split("```python\n", 1)[1].split("\n```", 1)[0]
    tree = ast.parse(code, filename=str(guide_path))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "cancellation"]
    assert len(imports) == 1
    assert [(item.name, item.asname) for item in imports[0].names] == [
        ("cancel_countdown_job_once", None),
    ]
    path_changes = [node for node in tree.body if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "sys.path.insert"]
    assert len(path_changes) == 1
    # Replace only guide-local import resolution with the exact-path import;
    # all confirmations, auth/config calls, and guard invocation stay intact.
    tree.body = [node for node in tree.body if node not in imports + path_changes]
    compiled = compile(tree, str(guide_path), "exec")

    def retrieve(job_id: str):
        assert job_id == JOB_ID
        return SimpleNamespace(**current)

    cancel = Mock(return_value=SimpleNamespace(status="cancelling"))
    jobs = SimpleNamespace(
        retrieve=Mock(side_effect=retrieve),
        cancel=Mock(side_effect=AssertionError("Cancellation must disable SDK retries")),
    )
    client = SimpleNamespace(
        fine_tuning=SimpleNamespace(jobs=jobs),
        with_options=Mock(return_value=SimpleNamespace(
            fine_tuning=SimpleNamespace(jobs=SimpleNamespace(cancel=cancel)),
        )),
    )

    class Context:
        def __init__(self) -> None:
            self.openai_client = client
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

    context = Context()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "load_foundry_config", lambda path: config.FoundryConfig(ENDPOINT))
    monkeypatch.setattr(auth, "create_project_context", lambda selected: context)

    def run(verified_endpoint=ENDPOINT, confirmed_label="model-grader", confirmed_id=JOB_ID):
        # Values model independently reviewed Foundry evidence, not state readback.
        answers = iter(["model-grader", verified_endpoint, confirmed_id, confirmed_label])
        prompts = []

        def answer(prompt):
            prompts.append(prompt)
            return next(answers)

        monkeypatch.setattr("builtins.input", answer)
        original = (json.dumps(state, indent=4) + "\n").encode("utf-8")
        state_path.write_bytes(original)
        try:
            exec(compiled, {"cancel_countdown_job_once": guard.cancel_countdown_job_once})
        finally:
            assert state_path.read_bytes() == original
            jobs.retrieve.assert_called_once_with(JOB_ID)
            jobs.cancel.assert_not_called()
            assert context.closed
            assert prompts == [
                "Select one job (model-grader or python-grader): ",
                "Paste the project endpoint from Foundry: ",
                "Paste the independently verified intended job ID: ",
                "Type the selected job label to confirm: ",
            ]

    return SimpleNamespace(
        state=state, current=current, client=client, cancel=cancel, run=run,
    )


@pytest.mark.parametrize(
    ("legacy", "saved_endpoint", "verified_endpoint", "status", "confirmed_label", "confirmed_id", "cancels"),
    [
        (False, ENDPOINT, ENDPOINT, "running", "model-grader", JOB_ID, True),
        (False, "https://other.services.ai.azure.com/api/projects/other", ENDPOINT, "running", "model-grader", JOB_ID, False),
        (False, None, "https://other.services.ai.azure.com/api/projects/other", "running", "model-grader", JOB_ID, False),
        (False, ENDPOINT, ENDPOINT, "succeeded", "model-grader", JOB_ID, False),
        (False, ENDPOINT, ENDPOINT, "cancelling", "model-grader", JOB_ID, False),
        (False, ENDPOINT, ENDPOINT, "running", "python-grader", JOB_ID, False),
        (False, ENDPOINT, ENDPOINT, "running", "model-grader", "wrong-job", False),
        (False, None, ENDPOINT, "running", "model-grader", JOB_ID, True),
        (True, ENDPOINT, ENDPOINT, "running", "model-grader", JOB_ID, False),
        (True, None, ENDPOINT, "running", "model-grader", JOB_ID, False),
    ],
)
def test_countdown_guide_cancels_only_verified_selected_job(
    guide_target,
    legacy: bool,
    saved_endpoint: str | None,
    verified_endpoint: str,
    status: str,
    confirmed_label: str,
    confirmed_id: str,
    cancels: bool,
) -> None:
    state = guide_target.state
    if saved_endpoint is not None:
        state["job_endpoints"] = {"model-grader": saved_endpoint}
    if legacy:
        state.pop("schema_version")
        state.pop("pending")
    guide_target.current["status"] = status
    if cancels:
        guide_target.run(verified_endpoint, confirmed_label, confirmed_id)
        guide_target.client.with_options.assert_called_once_with(max_retries=0)
        guide_target.cancel.assert_called_once_with(JOB_ID)
    else:
        with pytest.raises(RuntimeError, match="no cancellation was sent"):
            guide_target.run(verified_endpoint, confirmed_label, confirmed_id)
        guide_target.client.with_options.assert_not_called()
        guide_target.cancel.assert_not_called()


@pytest.mark.parametrize("field", [
    "id", "model", "training_file", "validation_file", "suffix", "method",
    "grader", "hyperparameters", "response_format", "trainingType",
])
def test_countdown_guide_remote_mismatch_never_cancels(guide_target, field):
    current = guide_target.current
    if field in {"grader", "hyperparameters", "response_format"}:
        current["method"]["reinforcement"].pop(field)
    elif field == "method":
        current.pop(field)
    else:
        current[field] = "unrelated-remote-value"
    with pytest.raises(RuntimeError, match="no cancellation was sent"):
        guide_target.run()
    guide_target.client.with_options.assert_not_called()
    guide_target.cancel.assert_not_called()