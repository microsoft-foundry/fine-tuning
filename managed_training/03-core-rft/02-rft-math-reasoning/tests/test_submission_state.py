from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import live_validation as live


class ResponseLost(Exception):
    status_code = 500


class OfflineService:
    """A strict fake: every POST checks the already-persisted intent."""

    def __init__(self, state_path):
        self.state_path = state_path
        self.remote_files = {}
        self.contents = {}
        self.remote_jobs = {}
        self.uploads = []
        self.submissions = []
        self.options = []
        self.lose_upload_response = False
        self.lose_job_response = False
        self.fail_processing = False
        self.files = SimpleNamespace(
            list=self.list_files,
            retrieve=self.remote_files.__getitem__,
            create=self.forbidden_post,
            wait_for_processing=self.wait_for_processing,
            content=lambda file_id: SimpleNamespace(content=self.contents[file_id]),
        )
        self.fine_tuning = SimpleNamespace(jobs=SimpleNamespace(
            retrieve=self.remote_jobs.__getitem__, create=self.forbidden_post,
        ))

    @staticmethod
    def forbidden_post(**_kwargs):
        raise AssertionError("POST must use with_options(max_retries=0)")

    def state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def with_options(self, **options):
        assert options == {"max_retries": 0}
        self.options.append(options)
        return SimpleNamespace(
            files=SimpleNamespace(create=self.create_file),
            fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=self.create_job)),
        )

    def list_files(self):
        return SimpleNamespace(data=list(self.remote_files.values()))

    def create_file(self, *, file, purpose):
        name, content, media_type = file
        kind = "training" if "-training-" in name else "validation"
        assert self.state()["pending_operation"] == f"{kind}_upload"
        assert isinstance(content, bytes) and b"\r\n" not in content
        assert media_type == "application/jsonl"
        uploaded = SimpleNamespace(
            id=f"{kind}-file", filename=name, bytes=len(content), purpose=purpose,
        )
        self.uploads.append(uploaded)
        self.remote_files[uploaded.id] = uploaded
        self.contents[uploaded.id] = content
        if self.lose_upload_response:
            raise ResponseLost("upload response lost after server acceptance")
        return uploaded

    def wait_for_processing(self, file_id):
        kind = "training" if file_id == "training-file" else "validation"
        assert self.state()[f"{kind}_file_id"] == file_id
        if self.fail_processing:
            raise ValueError("processing failed")

    def create_job(self, **payload):
        state = self.state()
        assert state["pending_operation"] == "job_create"
        assert state["training_file_id"] == payload["training_file"]
        assert state["validation_file_id"] == payload["validation_file"]
        job = SimpleNamespace(id="math-job", status="queued", **payload)
        self.submissions.append(job)
        self.remote_jobs[job.id] = job
        if self.lose_job_response:
            raise ResponseLost("job response lost after server acceptance")
        return job


@pytest.fixture
def submission(tmp_path):
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_bytes(b'{"answer":"1"}\r\n')
    validation.write_bytes(b'{"answer":"2"}\r\n')
    state_path = tmp_path / "outputs" / "submission-state.json"
    service = OfflineService(state_path)
    arguments = {
        "state_path": state_path,
        "project_endpoint": "project-endpoint",
        "model": "math-model",
        "method": {"type": "reinforcement", "reinforcement": {"grader": {"source": "fixed"}}},
        "training_path": training,
        "validation_path": validation,
    }

    def run(**overrides):
        return live.submit_or_resume_math_job(service, **{**arguments, **overrides})

    return SimpleNamespace(service=service, arguments=arguments, run=run, path=state_path)


def test_success_is_cleanup_compatible_and_kernel_restart_sends_no_posts(submission):
    job = submission.run()
    state = submission.service.state()
    assert state["equivalence"]["project_endpoint"] == "project-endpoint"
    assert state["job_id"] == job.id
    assert state["training_file_id"] == job.training_file
    assert state["validation_file_id"] == job.validation_file
    assert state["pending_operation"] is None
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1
    assert submission.service.options == [{"max_retries": 0}] * 3
    # Only disk state survives; LF and CRLF checkouts have the same equivalence.
    for key in ("training_path", "validation_path"):
        path = submission.arguments[key]
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    assert submission.run() is job
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1
    assert not submission.path.with_suffix(".json.lock").exists()


@pytest.mark.parametrize("phase", ["initial", "training_upload", "validation_upload", "job_create"])
def test_state_write_failure_prevents_the_next_post(submission, monkeypatch, phase):
    original = live._write_submission_state

    def write(path, state):
        if phase == "initial" or state["pending_operation"] == phase:
            raise OSError("state storage unavailable")
        original(path, state)

    monkeypatch.setattr(live, "_write_submission_state", write)
    with pytest.raises(OSError, match="state storage unavailable"):
        submission.run()
    assert len(submission.service.uploads) == {
        "initial": 0, "training_upload": 0, "validation_upload": 1, "job_create": 2,
    }[phase]
    assert not submission.service.submissions


@pytest.mark.parametrize("kind", ["training", "validation"])
def test_file_id_checkpoint_failure_reconciles_without_duplicate_upload(submission, monkeypatch, kind):
    original = live._write_submission_state

    def write(path, state):
        if state[f"{kind}_file_id"]:
            raise OSError("file ID checkpoint failed")
        original(path, state)

    monkeypatch.setattr(live, "_write_submission_state", write)
    with pytest.raises(OSError, match="file ID checkpoint failed"):
        submission.run()
    assert submission.service.state()["pending_operation"] == f"{kind}_upload"
    assert submission.service.state()[f"{kind}_file_id"] is None
    assert not submission.service.submissions
    monkeypatch.setattr(live, "_write_submission_state", original)
    assert submission.run().id == "math-job"
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1


def test_lost_upload_response_recovers_unique_verified_file_without_reposting(submission):
    submission.service.lose_upload_response = True
    with pytest.raises(ResponseLost):
        submission.run()
    assert len(submission.service.uploads) == 1
    assert submission.service.state()["pending_operation"] == "training_upload"
    submission.service.lose_upload_response = False
    assert submission.run().id == "math-job"
    assert len(submission.service.uploads) == 2


@pytest.mark.parametrize("visible", [0, 2])
def test_ambiguous_upload_never_posts_when_absent_or_nonunique(submission, visible):
    submission.service.lose_upload_response = True
    with pytest.raises(ResponseLost):
        submission.run()
    uploaded = submission.service.uploads[0]
    if visible == 0:
        submission.service.remote_files.clear()
    else:
        submission.service.remote_files["duplicate"] = SimpleNamespace(
            **{**vars(uploaded), "id": "duplicate"},
        )
    with pytest.raises(RuntimeError, match="ambiguous|Multiple remote files"):
        submission.run()
    assert len(submission.service.uploads) == 1
    assert not submission.service.submissions


def test_processing_failure_retains_file_id_and_resumes_without_reupload(submission):
    submission.service.fail_processing = True
    with pytest.raises(RuntimeError, match="processing failed"):
        submission.run()
    assert submission.service.state()["training_file_id"] == "training-file"
    assert submission.service.state()["pending_operation"] is None
    submission.service.fail_processing = False
    assert submission.run().id == "math-job"
    assert len(submission.service.uploads) == 2


@pytest.mark.parametrize("failure", ["response", "checkpoint"])
def test_ambiguous_job_requires_explicit_reconciliation_not_another_post(submission, monkeypatch, failure):
    original = live._write_submission_state
    if failure == "response":
        submission.service.lose_job_response = True
        error = ResponseLost
    else:
        def write(path, state):
            if state["job_id"]:
                raise OSError("job ID checkpoint failed")
            original(path, state)

        monkeypatch.setattr(live, "_write_submission_state", write)
        error = OSError
    with pytest.raises(error):
        submission.run()
    assert submission.service.state()["pending_operation"] == "job_create"
    assert submission.service.state()["job_id"] is None
    with pytest.raises(RuntimeError, match="Job submission outcome is ambiguous"):
        submission.run()
    assert len(submission.service.submissions) == 1
    monkeypatch.setattr(live, "_write_submission_state", original)
    job = submission.run(existing_job_id="math-job")
    assert job.id == submission.service.state()["job_id"]
    assert submission.service.state()["pending_operation"] is None
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1


@pytest.mark.parametrize("change", ["endpoint", "model", "method", "data"])
def test_changed_contract_fails_before_any_additional_post(submission, change):
    submission.run()
    overrides = {}
    if change == "endpoint":
        overrides["project_endpoint"] = "different-project"
    elif change == "model":
        overrides["model"] = "different-model"
    elif change == "method":
        overrides["method"] = {"type": "supervised"}
    else:
        submission.arguments["training_path"].write_bytes(b'{"answer":"changed"}\n')
    before = submission.path.read_bytes()
    with pytest.raises(RuntimeError, match="does not match"):
        submission.run(**overrides)
    assert submission.path.read_bytes() == before
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_unsuccessful_job_is_not_replaced(submission, status):
    job = submission.run()
    job.status = status
    with pytest.raises(RuntimeError, match="cannot be reused"):
        submission.run()
    assert len(submission.service.submissions) == 1


def test_saved_job_id_cannot_be_silently_replaced(submission):
    submission.run()
    with pytest.raises(RuntimeError, match="conflicts"):
        submission.run(existing_job_id="another-job")
    assert len(submission.service.submissions) == 1


def test_verified_external_job_adopts_its_ids_without_uploads(submission):
    job = submission.run()
    # Model a kernel/state loss with an independently supplied, verified job ID.
    restored_path = submission.path.parent / "restored-state.json"
    submission.service.state_path = restored_path
    assert submission.run(state_path=restored_path, existing_job_id=job.id) is job
    restored = json.loads(restored_path.read_text(encoding="utf-8"))
    assert restored["job_id"] == job.id
    assert restored["training_file_id"] == job.training_file
    assert restored["validation_file_id"] == job.validation_file
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1


def test_missing_saved_file_never_falls_back_to_reupload(submission):
    submission.run()
    submission.service.remote_files.pop("training-file")
    with pytest.raises(RuntimeError, match="retrieve saved RFT file"):
        submission.run()
    assert len(submission.service.uploads) == 2
    assert len(submission.service.submissions) == 1


def test_corrupt_state_is_preserved_and_does_not_submit(submission):
    submission.path.parent.mkdir(parents=True)
    submission.path.write_text("incomplete", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        submission.run()
    assert submission.path.read_text(encoding="utf-8") == "incomplete"
    assert not submission.service.uploads
    assert not submission.service.submissions


def test_missing_job_id_keeps_intent_and_prevents_resubmission(submission):
    original = submission.service.create_job

    def create(**payload):
        job = original(**payload)
        job.id = None
        return job

    submission.service.create_job = create
    with pytest.raises(RuntimeError, match="no resource ID"):
        submission.run()
    assert submission.service.state()["pending_operation"] == "job_create"
    assert submission.service.state()["job_id"] is None
    with pytest.raises(RuntimeError, match="ambiguous"):
        submission.run()
    assert len(submission.service.submissions) == 1


def test_atomic_writer_fsyncs_before_replace_and_retains_prior_state_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "submission-state.json"
    live._write_submission_state(path, {"version": "prior"})
    original_fsync = live.os.fsync
    original_replace = live.os.replace
    events = []

    def fsync(descriptor):
        events.append("fsync")
        original_fsync(descriptor)

    def replace(source, target):
        assert Path(source).parent == path.parent
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"version": "next"}
        assert events == ["fsync"]
        original_replace(source, target)  # Also exercises native Windows replace.

    monkeypatch.setattr(live.os, "fsync", fsync)
    monkeypatch.setattr(live.os, "replace", replace)
    live._write_submission_state(path, {"version": "next"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "next"}

    def fail_replace(_source, _target):
        raise PermissionError("replace blocked")

    monkeypatch.setattr(live.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="replace blocked"):
        live._write_submission_state(path, {"version": "uncommitted"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "next"}
    assert list(tmp_path.iterdir()) == [path]


def test_writer_lock_rejects_concurrent_submission(submission):
    with live._submission_lock(submission.path):
        with pytest.raises(RuntimeError, match="locked"):
            submission.run()
        assert submission.path.with_suffix(".json.lock").exists()
    assert not submission.path.with_suffix(".json.lock").exists()
    assert not submission.service.uploads
    assert not submission.service.submissions