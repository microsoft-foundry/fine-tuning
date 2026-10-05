from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys

TRACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK_ROOT / "src"))

import multimodal_common as common


class FakeFiles:
    def __init__(self) -> None:
        self.created_name = ""
        self.retrieve_status = "processed"

    def list(self) -> list[object]:
        return []

    def create(self, *, file: tuple[str, object, str], purpose: str) -> object:
        self.created_name = file[0]
        assert purpose == "fine-tune"
        return SimpleNamespace(id="file-created")

    def retrieve(self, file_id: str) -> object:
        return SimpleNamespace(id=file_id, status=self.retrieve_status)


def test_live_evaluation_defaults_false_for_empty_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDRY_RUN_LIVE_EVALUATION", "")
    assert common.live_evaluation_enabled() is False


def test_upload_uses_content_addressed_filename(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    files = FakeFiles()
    client = SimpleNamespace(files=files)

    uploaded = common.upload_or_reuse_file(client, source)

    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    assert files.created_name == f"training-{digest}.jsonl"
    assert uploaded.id == "file-created"


def test_reused_failed_job_raises_even_when_paid_jobs_disabled(
    tmp_path: Path,
) -> None:
    jobs = SimpleNamespace(
        retrieve=lambda _: SimpleNamespace(
            id="job-reused",
            status="failed",
            error={"message": "training failed"},
        )
    )
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))

    with pytest.raises(RuntimeError, match="terminal state failed"):
        common.resolve_or_create_job(
            client=client,
            reuse_id="job-reused",
            model="model",
            train=tmp_path / "train.jsonl",
            validation=tmp_path / "validation.jsonl",
            suffix="suffix",
            seed=42,
            training_type="standard",
            run_paid_jobs=False,
        )


def test_transient_calls_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class TransientError(RuntimeError):
        status_code = 503

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientError("temporarily unavailable")
        return "ok"

    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    monkeypatch.setattr(common.random, "uniform", lambda _a, _b: 1.0)

    assert common.retry_call(operation, operation_name="test") == "ok"
    assert attempts == 2


def test_project_clients_close_both_clients_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = SimpleNamespace(closed=False)
    client = SimpleNamespace(closed=False)
    project.close = lambda: setattr(project, "closed", True)
    client.close = lambda: setattr(client, "closed", True)
    monkeypatch.setattr(common, "get_clients", lambda _: (project, client))

    with pytest.raises(RuntimeError, match="boom"):
        with common.project_clients("configured"):
            raise RuntimeError("boom")

    assert project.closed is True
    assert client.closed is True
