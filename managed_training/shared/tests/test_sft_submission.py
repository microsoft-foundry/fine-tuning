from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from shared import experiment_manifest, sft_submission
from shared.sft_submission import model_version_evidence, submit_or_resume_sft_job


class FakeService:
    """Offline SDK boundary that checks the journal at every POST and read."""

    def __init__(self, state_path):
        self.state_path = state_path
        self.calls = []
        self.options = []
        self.file_records = {}
        self.file_bytes = {}
        self.job = None
        self.job_payload = None
        self.fail_post = None
        self.post_error = RuntimeError("Response lost after possible acceptance")
        self.missing_id = None
        self.fail_read = None
        self.status = "processed"
        self.authorization = "credential-must-never-be-persisted"
        self.files = SimpleNamespace(
            create=self.unexpected_create, retrieve=self.retrieve_file, content=self.file_content,
        )
        self.fine_tuning = SimpleNamespace(jobs=SimpleNamespace(
            create=self.unexpected_create, retrieve=self.retrieve_job,
        ))

    def state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def unexpected_create(self, **_kwargs):
        raise AssertionError("A POST used the default retrying client")

    def with_options(self, **kwargs):
        assert kwargs == {"max_retries": 0}
        self.options.append(kwargs)
        return SimpleNamespace(
            files=SimpleNamespace(create=self.create_file),
            fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=self.create_job)),
        )

    def create_file(self, *, file, purpose, extra_headers):
        state = self.state()
        pending = state["pending_operation"]
        assert pending in {"training_upload", "validation_upload"}
        kind = pending.removesuffix("_upload")
        assert state[f"{kind}_file_id"] is None
        if kind == "validation":
            assert state["training_file_id"] == "file-training"
        assert purpose == "fine-tune"
        assert extra_headers == {"Accept-Encoding": "identity"}
        self.calls.append(("POST", pending))
        if self.fail_post == pending:
            raise self.post_error
        if self.missing_id == pending:
            return SimpleNamespace(id=None)
        filename, content, mime_type = file
        assert mime_type == "application/jsonl"
        assert isinstance(content, bytes) and b"\r\n" not in content
        file_id = f"file-{kind}"
        resource = SimpleNamespace(
            id=file_id, filename=filename, purpose=purpose,
            status=self.status, bytes=len(content),
        )
        self.file_records[file_id] = resource
        self.file_bytes[file_id] = content
        return resource

    def retrieve_file(self, file_id, **_kwargs):
        kind = file_id.removeprefix("file-")
        state = self.state()
        assert state["pending_operation"] is None
        assert state[f"{kind}_file_id"] == file_id  # ID must precede the first poll.
        self.calls.append(("GET", file_id))
        if self.fail_read == kind:
            raise TimeoutError("Synthetic read-only timeout")
        return self.file_records[file_id]

    def file_content(self, file_id, **_kwargs):
        self.calls.append(("CONTENT", file_id))
        return SimpleNamespace(read=lambda: self.file_bytes[file_id])

    def create_job(self, **payload):
        state = self.state()
        assert state["pending_operation"] == "job_create"
        assert state["job_id"] is None
        assert state["training_file_id"] == payload["training_file"] == "file-training"
        assert state["validation_file_id"] == payload["validation_file"] == "file-validation"
        self.calls.append(("POST", "job_create"))
        self.job_payload = payload
        if self.fail_post == "job_create":
            raise self.post_error
        if self.missing_id == "job_create":
            return SimpleNamespace(id=None)
        self.job = SimpleNamespace(
            id="ftjob-sft", status="queued", model=payload["model"],
            training_file=payload["training_file"], validation_file=payload["validation_file"],
            method=payload.get("method", payload["extra_body"].get("method")),
            # These undocumented/caller-controlled fields must not become proof.
            model_version="9", metadata={"model_version": "1"},
        )
        return self.job

    def retrieve_job(self, job_id):
        assert self.state()["job_id"] == job_id
        self.calls.append(("GET_JOB", job_id))
        return self.job


@pytest.fixture
def submission(tmp_path):
    root = tmp_path / "outputs"
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    contents = {}
    for kind, path in (("training", training), ("validation", validation)):
        row = {"messages": [
            {"role": "user", "content": kind}, {"role": "assistant", "content": "ok"},
        ]}
        contents[kind] = (json.dumps(row) + "\n").encode("utf-8")
        path.write_bytes(contents[kind].replace(b"\n", b"\r\n"))
    state_path = root / "submission-state.json"
    client = FakeService(state_path)
    kwargs = {
        "state_path": state_path, "outputs_root": root,
        "project_endpoint": "https://unit-test.invalid/api/projects/sft",
        "model": "test-model", "expected_model_version": "1", "reference_region": "reference-only",
        "training_type": "GlobalStandard", "hyperparameters": {"n_epochs": 2},
        "suffix": "test-sft", "training_path": training, "validation_path": validation,
        "expected_training_sha256": hashlib.sha256(contents["training"]).hexdigest(),
        "expected_validation_sha256": hashlib.sha256(contents["validation"]).hexdigest(),
    }
    return SimpleNamespace(client=client, kwargs=kwargs, state_path=state_path)


def test_journal_intent_ids_and_retry_override_precede_every_operation(submission, monkeypatch) -> None:
    writes = []
    original = sft_submission._write_json_atomic

    def record(path, payload):
        writes.append(deepcopy(payload))
        return original(path, payload)

    monkeypatch.setattr(sft_submission, "_write_json_atomic", record)
    client = submission.client
    job = submit_or_resume_sft_job(client, **submission.kwargs)
    assert job.id == "ftjob-sft"
    assert [state["pending_operation"] for state in writes] == [
        None, "training_upload", None, "validation_upload", None, "job_create", None,
    ]
    assert writes[2]["training_file_id"] == "file-training"
    assert writes[4]["validation_file_id"] == "file-validation"
    assert writes[6]["job_id"] == "ftjob-sft"
    assert client.options == [{"max_retries": 0}]
    assert [call for call in client.calls if call[0] == "POST"] == [
        ("POST", "training_upload"), ("POST", "validation_upload"), ("POST", "job_create"),
    ]
    state = client.state()
    assert state["equivalence"]["project_endpoint"] == submission.kwargs["project_endpoint"]
    assert state["submitted_at_utc"]
    assert state["version_evidence"] == model_version_evidence("1")
    assert state["version_evidence"]["actual_model_version"] is None
    assert "model_version" not in state["equivalence"]
    assert client.authorization not in submission.state_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("method_in_extra_body", [False, True])
def test_wire_shape_and_recorded_job_resume_do_not_repeat_mutations(submission, method_in_extra_body) -> None:
    kwargs = {**submission.kwargs, "method_in_extra_body": method_in_extra_body}
    client = submission.client
    first = submit_or_resume_sft_job(client, **kwargs)
    payload = client.job_payload
    method = {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 2}}}
    if method_in_extra_body:
        assert "method" not in payload
        assert payload["extra_body"] == {"trainingType": "GlobalStandard", "method": method}
    else:
        assert payload["method"] == method
        assert payload["extra_body"] == {"trainingType": "GlobalStandard"}
    assert "model_version" not in payload
    before = list(client.calls)
    second = submit_or_resume_sft_job(client, **kwargs)
    assert second is first
    assert client.calls == before + [("GET_JOB", "ftjob-sft")]
    assert client.options == [{"max_retries": 0}]
    assert client.state()["version_evidence"]["model_version_verification"] == "unverified"


@pytest.mark.parametrize("kind", ["training", "validation"])
def test_immediate_saved_file_id_survives_read_failure_and_is_reused(submission, kind) -> None:
    client = submission.client
    client.fail_read = kind
    with pytest.raises(TimeoutError, match="read-only"):
        submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.state()[f"{kind}_file_id"] == f"file-{kind}"
    assert client.state()["pending_operation"] is None
    assert client.state()["job_id"] is None
    client.fail_read = None
    submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.calls.count(("POST", "training_upload")) == 1
    assert client.calls.count(("POST", "validation_upload")) == 1
    assert client.calls.count(("POST", "job_create")) == 1


@pytest.mark.parametrize("pending", ["training_upload", "validation_upload", "job_create"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
def test_unknown_acceptance_or_cancellation_never_replays_a_post(submission, pending, error_type) -> None:
    client = submission.client
    client.fail_post = pending
    client.post_error = error_type("credential-must-not-appear-in-error-or-state")
    with pytest.raises(RuntimeError if error_type is RuntimeError else error_type) as failure:
        submit_or_resume_sft_job(client, **submission.kwargs)
    if error_type is RuntimeError:
        assert "credential-must-not-appear" not in str(failure.value)
    assert client.state()["pending_operation"] == pending
    before = list(client.calls)
    saved = submission.state_path.read_bytes()
    client.fail_post = None
    with pytest.raises(RuntimeError, match="ambiguous"):
        submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.calls == before
    assert submission.state_path.read_bytes() == saved


@pytest.mark.parametrize("pending", ["training_upload", "validation_upload", "job_create"])
def test_missing_response_id_retains_unknown_acceptance(submission, pending) -> None:
    client = submission.client
    client.missing_id = pending
    with pytest.raises(RuntimeError, match="response ID"):
        submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.state()["pending_operation"] == pending
    before = list(client.calls)
    with pytest.raises(RuntimeError, match="ambiguous"):
        submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.calls == before


@pytest.mark.parametrize("pending", ["training_upload", "validation_upload", "job_create"])
def test_failed_pending_checkpoint_prevents_that_post(submission, monkeypatch, pending) -> None:
    original = sft_submission._write_json_atomic

    def fail(path, payload):
        if payload["pending_operation"] == pending:
            raise OSError("Synthetic journal failure before POST")
        return original(path, payload)

    monkeypatch.setattr(sft_submission, "_write_json_atomic", fail)
    with pytest.raises(OSError, match="before POST"):
        submit_or_resume_sft_job(submission.client, **submission.kwargs)
    assert ("POST", pending) not in submission.client.calls


@pytest.mark.parametrize(
    "pending,key", [("training_upload", "training_file_id"),
                    ("validation_upload", "validation_file_id"), ("job_create", "job_id")],
)
def test_id_checkpoint_failure_after_acceptance_leaves_pending_not_replayable(submission, monkeypatch, pending, key) -> None:
    original = sft_submission._write_json_atomic

    def fail(path, payload):
        if payload["pending_operation"] is None and payload[key] is not None:
            raise OSError("Synthetic journal failure after acceptance")
        return original(path, payload)

    monkeypatch.setattr(sft_submission, "_write_json_atomic", fail)
    with pytest.raises(OSError, match="after acceptance"):
        submit_or_resume_sft_job(submission.client, **submission.kwargs)
    assert submission.client.state()["pending_operation"] == pending
    before = list(submission.client.calls)
    monkeypatch.setattr(sft_submission, "_write_json_atomic", original)
    with pytest.raises(RuntimeError, match="ambiguous"):
        submit_or_resume_sft_job(submission.client, **submission.kwargs)
    assert submission.client.calls == before


def test_atomic_replace_failure_keeps_previous_journal_and_removes_temporary_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    experiment_manifest._write_json_atomic(path, {"previous": True})
    original = path.read_bytes()

    def fail(_source, _destination):
        raise OSError("Synthetic replace failure")

    monkeypatch.setattr(experiment_manifest.os, "replace", fail)
    with pytest.raises(OSError, match="replace failure"):
        experiment_manifest._write_json_atomic(path, {"pending": "training_upload"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("state_text", ["{", "[]", '{"job_id": "ftjob-old", "equivalence": {}}'])
def test_corrupt_or_legacy_state_is_retained_without_network_or_mutations(submission, state_text) -> None:
    submission.state_path.parent.mkdir()
    submission.state_path.write_text(state_text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable|older format"):
        submit_or_resume_sft_job(submission.client, **submission.kwargs)
    assert submission.client.calls == []
    assert submission.state_path.read_text(encoding="utf-8") == state_text


@pytest.mark.parametrize(
    "changed", [{"model": "another-model"}, {"expected_model_version": "2"},
                {"hyperparameters": {"n_epochs": 3}},
                {"project_endpoint": "https://unit-test.invalid/api/projects/other"}],
)
def test_changed_request_is_not_mistaken_for_equivalent_saved_job(submission, changed) -> None:
    client = submission.client
    submit_or_resume_sft_job(client, **submission.kwargs)
    before = list(client.calls)
    saved = submission.state_path.read_bytes()
    with pytest.raises(RuntimeError, match="another recipe"):
        submit_or_resume_sft_job(client, **{**submission.kwargs, **changed})
    assert client.calls == before
    assert submission.state_path.read_bytes() == saved


@pytest.mark.parametrize(
    "field,value", [("id", "ftjob-other"), ("model", "other-model"),
                    ("training_file", "file-other"), ("validation_file", "file-other"),
                    ("method", {"type": "reinforcement"}),
                    ("method", {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 9}}})],
)
def test_retrieved_job_mismatch_fails_without_replacement(submission, field, value) -> None:
    client = submission.client
    submit_or_resume_sft_job(client, **submission.kwargs)
    setattr(client.job, field, value)
    post_count = sum(call[0] == "POST" for call in client.calls)
    with pytest.raises(RuntimeError, match="Retrieved job"):
        submit_or_resume_sft_job(client, **submission.kwargs)
    assert sum(call[0] == "POST" for call in client.calls) == post_count
    assert client.state()["job_id"] == "ftjob-sft"


def test_stale_or_active_lock_blocks_all_remote_operations(submission) -> None:
    submission.state_path.parent.mkdir()
    lock = submission.state_path.with_suffix(".json.lock")
    lock.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        submit_or_resume_sft_job(submission.client, **submission.kwargs)
    assert submission.client.calls == []
    assert lock.exists()
    assert not submission.state_path.exists()


def test_processing_timeout_saves_id_before_timeout_and_never_sleeps(submission, monkeypatch) -> None:
    submission.client.status = "processing"
    clock = iter([0.0, 1.0])

    def never_sleep(_seconds):
        raise AssertionError("The bounded timeout should fire before sleeping")

    monkeypatch.setattr(sft_submission, "time", SimpleNamespace(
        monotonic=lambda: next(clock), sleep=never_sleep,
    ))
    with pytest.raises(TimeoutError, match="remains saved"):
        submit_or_resume_sft_job(
            submission.client, **{**submission.kwargs, "file_timeout_seconds": 0.5},
        )
    assert submission.client.state()["training_file_id"] == "file-training"
    assert submission.client.state()["pending_operation"] is None
    assert submission.client.calls.count(("POST", "training_upload")) == 1


def test_terminal_file_failure_keeps_cleanup_id_and_does_not_replace_it(submission) -> None:
    client = submission.client
    client.status = "failed"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="processing failed"):
            submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.state()["training_file_id"] == "file-training"
    assert client.calls.count(("POST", "training_upload")) == 1
    assert ("POST", "validation_upload") not in client.calls
    assert ("POST", "job_create") not in client.calls


def test_saved_file_content_mismatch_blocks_job_without_reupload(submission) -> None:
    client = submission.client
    client.fail_read = "training"
    with pytest.raises(TimeoutError):
        submit_or_resume_sft_job(client, **submission.kwargs)
    client.fail_read = None
    client.file_bytes["file-training"] = b"different remote bytes"
    client.file_records["file-training"].bytes = len(client.file_bytes["file-training"])
    with pytest.raises(RuntimeError, match="bytes do not match"):
        submit_or_resume_sft_job(client, **submission.kwargs)
    assert client.state()["training_file_id"] == "file-training"
    assert client.calls.count(("POST", "training_upload")) == 1
    assert ("POST", "validation_upload") not in client.calls
    assert ("POST", "job_create") not in client.calls


def test_mutations_require_sdk_retry_override_before_any_pending_post(submission) -> None:
    client = submission.client
    delattr(client, "authorization")  # No credential object is needed by the journal.
    without_override = SimpleNamespace(files=client.files, fine_tuning=client.fine_tuning)
    with pytest.raises(AttributeError, match="with_options"):
        submit_or_resume_sft_job(without_override, **submission.kwargs)
    assert client.calls == []
    assert client.state()["pending_operation"] is None
    assert client.state()["training_file_id"] is None


@pytest.mark.parametrize(
    "changed", [{"expected_training_sha256": "0" * 64},
                {"hyperparameters": {"api_key": "must-not-be-written"}},
                {"project_endpoint": "https://unit-test.invalid/api/projects/sft?token=must-not-be-written"}],
)
def test_invalid_data_or_credential_shaped_configuration_is_not_journaled(submission, changed) -> None:
    with pytest.raises(ValueError):
        submit_or_resume_sft_job(submission.client, **{**submission.kwargs, **changed})
    assert submission.client.calls == []
    assert not submission.state_path.exists()