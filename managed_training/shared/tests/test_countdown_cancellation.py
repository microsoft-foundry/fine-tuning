"""Offline checks for the Countdown cancellation operator guide and job state."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared import auth, config


DEMO_ROOT = Path(__file__).parents[2] / "03-core-rft" / "01-rft-countdown"
ENDPOINT = "https://example.services.ai.azure.com/api/projects/example"
JOB_ID = "example-job"


def test_countdown_submissions_keep_their_original_state_contract(tmp_path: Path) -> None:
    notebook = json.loads((DEMO_ROOT / "notebooks/demo.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("\n".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    selected = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"load_state", "save_state", "submit_once"}
    ]
    assert len(selected) == 3

    class Jobs:
        def __init__(self) -> None:
            self.created = 0

        def create(self, **kwargs):
            self.created += 1
            return SimpleNamespace(id=JOB_ID)

        def retrieve(self, job_id):
            return SimpleNamespace(id=job_id, status="running")

    class Client:
        def __init__(self) -> None:
            self.jobs = Jobs()
            self.fine_tuning = SimpleNamespace(jobs=self.jobs)

        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            return self

    client = Client()
    state_path = tmp_path / "submission-state.json"
    values = {
        "json": json,
        "OUTPUT_ROOT": tmp_path,
        "STATE_PATH": state_path,
        "RUN_FINGERPRINT": "expected-fingerprint",
        "PROJECT_ENDPOINT": ENDPOINT,
        "openai_client": client,
        "HYPERPARAMETERS": {},
        "RESPONSE_FORMAT": {},
        "MODEL": "qwen3.6-35b-a3b",
        "TRAINING_TYPE": "globalStandard",
        "training_file_id": "train",
        "validation_file_id": "validation",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "countdown", "exec"), values)
    values["submit_once"]("model-grader", {"type": "score_model"}, "suffix")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["jobs"] == {"model-grader": JOB_ID}
    assert "job_endpoints" not in state

    # Reusing an older recorded job must not invent provenance for that job.
    legacy = {"fingerprint": "expected-fingerprint", "jobs": {"model-grader": JOB_ID}}
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    values["submit_once"]("model-grader", {"type": "score_model"}, "suffix")
    assert json.loads(state_path.read_text(encoding="utf-8")) == legacy
    assert client.jobs.created == 1


@pytest.mark.parametrize(
    ("saved_endpoint", "verified_endpoint", "status", "confirmed_label", "confirmed_id", "cancels"),
    [
        (ENDPOINT, ENDPOINT, "running", "model-grader", JOB_ID, True),
        ("https://other.services.ai.azure.com/api/projects/other", ENDPOINT, "running", "model-grader", JOB_ID, False),
        (None, "https://other.services.ai.azure.com/api/projects/other", "running", "model-grader", JOB_ID, False),
        (ENDPOINT, ENDPOINT, "succeeded", "model-grader", JOB_ID, False),
        (ENDPOINT, ENDPOINT, "cancelling", "model-grader", JOB_ID, False),
        (ENDPOINT, ENDPOINT, "running", "python-grader", JOB_ID, False),
        (ENDPOINT, ENDPOINT, "running", "model-grader", "wrong-job", False),
        (None, ENDPOINT, "running", "model-grader", JOB_ID, True),
    ],
)
def test_countdown_guide_cancels_only_verified_selected_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_endpoint: str | None,
    verified_endpoint: str,
    status: str,
    confirmed_label: str,
    confirmed_id: str,
    cancels: bool,
) -> None:
    guide = (DEMO_ROOT / "CANCELLATION_AND_CLEANUP.md").read_text(encoding="utf-8")
    code = guide.split("```python\n", 1)[1].split("\n```", 1)[0]
    compile(code, "countdown-guide", "exec")

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    state = {"jobs": {"model-grader": JOB_ID, "python-grader": "other-job"}}
    if saved_endpoint is not None:
        state["job_endpoints"] = {"model-grader": saved_endpoint}
    (outputs / "submission-state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "load_foundry_config", lambda path: config.FoundryConfig(ENDPOINT))

    class Jobs:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def retrieve(self, job_id: str):
            assert job_id == JOB_ID
            return SimpleNamespace(id=job_id, model="qwen3.6-35b-a3b", status=status)

        def cancel(self, job_id: str):
            self.cancelled.append(job_id)
            return SimpleNamespace(status="cancelling")

    class Context:
        def __init__(self) -> None:
            self.jobs = Jobs()
            self.openai_client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=self.jobs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    context = Context()
    monkeypatch.setattr(auth, "create_project_context", lambda selected: context)
    inputs = iter(["model-grader", verified_endpoint, confirmed_label, confirmed_id])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    if cancels:
        exec(compile(code, "countdown-guide", "exec"), {})
        assert context.jobs.cancelled == [JOB_ID]
    else:
        with pytest.raises(RuntimeError):
            exec(compile(code, "countdown-guide", "exec"), {})
        assert context.jobs.cancelled == []