"""Run saved submission cells with fake services, never notebooks or paid APIs."""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


TRACK = Path(__file__).resolve().parents[1]
DEMOS = (
    "01-chart-reasoning", "02-image-classification", "03-video-action-recognition",
)


def cell_source(cell):
    source = cell["source"]
    if isinstance(source, str):
        return source
    return "".join(source) if any(line.endswith(("\n", "\r")) for line in source) else "\n".join(source)


def training_source(demo):
    document = json.loads((TRACK / demo / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))
    return next(cell_source(cell) for cell in document["cells"]
                if cell["cell_type"] == "code" and ".fine_tuning.jobs.create(" in cell_source(cell))


def lock_for(path):
    return path.with_suffix(path.suffix + ".lock")


def forbidden(*_args, **_kwargs):
    raise AssertionError("An unconfigured POST or competing writer reached the service")


class Service:
    """Inspect real on-disk intent/IDs and pause accepted mutations deterministically."""

    def __init__(self, demo, outputs):
        self.demo = demo
        self.path = outputs / "run-state.json"
        self.file_path = outputs / "file-state.json" if demo.startswith("02") else self.path
        self.lock = lock_for(self.path)
        self.posts = []
        self.block_at = None
        self.fail_at = None
        self.failure = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.job = SimpleNamespace(
            id="job-known", status="succeeded",
            to_dict=lambda: {"id": "job-known", "status": "succeeded", "error": None},
        )
        self.files = SimpleNamespace(create=forbidden, retrieve=self.retrieve_file)
        self.fine_tuning = SimpleNamespace(jobs=SimpleNamespace(
            create=forbidden, retrieve=self.retrieve_job, list=lambda **_kwargs: [],
            list_events=lambda **_kwargs: SimpleNamespace(data=[]),
        ))
        self.configured = SimpleNamespace(
            files=SimpleNamespace(create=self.create_file, retrieve=self.retrieve_file),
            fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=self.create_job, retrieve=self.retrieve_job)),
        )

    def state(self, path):
        return json.loads(path.read_bytes())

    def with_options(self, **kwargs):
        assert kwargs["max_retries"] == 0
        if "timeout" in kwargs:
            assert 0 < kwargs["timeout"] <= 60
        return self.configured

    def accepted(self, phase):
        if self.block_at == phase:
            self.entered.set()
            assert self.release.wait(15), "Test writer was not released"
        if self.fail_at == phase:
            raise self.failure

    def create_file(self, *, file, purpose):
        assert self.lock.is_file(), "The mutation must hold the cross-process run lock"
        stream = file[1] if isinstance(file, tuple) else file
        label = Path(stream.name).stem
        assert label in {"training", "validation"} and purpose == "fine-tune"
        saved = self.state(self.file_path)
        assert saved[f"{label}_upload_pending"] is True and not saved.get(f"{label}_file_id")
        self.posts.append(f"upload:{label}")
        self.accepted(f"upload:{label}")
        return SimpleNamespace(id=f"file-{label}")

    def retrieve_file(self, file_id):
        assert self.lock.is_file(), "Persist the returned ID before releasing the lock"
        label = file_id.removeprefix("file-")
        saved = self.state(self.file_path)
        assert saved[f"{label}_file_id"] == file_id
        assert saved[f"{label}_upload_pending"] is False
        return SimpleNamespace(id=file_id, status="processed")

    def create_job(self, **payload):
        assert self.lock.is_file()
        saved = self.state(self.path)
        assert saved["submission_pending"] is True and not saved.get("job_id")
        assert payload["training_file"] == "file-training"
        assert payload["validation_file"] == "file-validation"
        self.posts.append("job")
        self.accepted("job")
        return self.job

    def retrieve_job(self, job_id):
        assert self.lock.is_file()
        saved = self.state(self.path)
        assert saved["job_id"] == job_id == "job-known"
        assert saved["submission_pending"] is False
        return self.job


def namespace_for(root, service):
    outputs = service.path.parent
    artifacts = {}
    for label in ("training", "validation"):
        path = root / f"{label}.jsonl"
        artifacts[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "STATE_PATH": service.path, "RUN_STATE_PATH": service.path,
        "OUTPUTS": outputs, "OUTPUT_DIR": outputs, "RUN_ID": "shared-run",
        "PROJECT_ENDPOINT": "https://offline.invalid/api/projects/unit", "MODEL": "unit-model",
        "TRAINING_TYPE": "GlobalStandard", "SEED": 42, "EPOCHS": 2,
        "LEARNING_RATE_MULTIPLIER": 0.5, "dataset_evidence": [],
        "DATASETS": {"train": {"path": root / "training.jsonl"}, "validation": {"path": root / "validation.jsonl"}},
        "artifacts": artifacts, "input_summary": artifacts,
        "train_path": root / "training.jsonl", "validation_path": root / "validation.jsonl",
        "client": service, "time": time, "json": json, "hashlib": hashlib,
        "Path": Path, "Any": object, "datetime": datetime, "timezone": timezone,
        "pd": SimpleNamespace(Timestamp=SimpleNamespace(now=lambda _zone: datetime.now(timezone.utc))),
        "TERMINAL_STATES": {"succeeded", "failed", "cancelled"},
        # Deliberately stale per-kernel dictionaries must never drive a new POST.
        "state": {}, "run_state": {}, "file_state": {},
    }


def make_case(demo, root):
    root.mkdir(parents=True, exist_ok=True)
    outputs = root / "outputs" / "unit-run"
    outputs.mkdir(parents=True)
    for label in ("training", "validation"):
        (root / f"{label}.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    service = Service(demo, outputs)
    return SimpleNamespace(
        service=service, root=root, source=compile(training_source(demo), demo, "exec"),
        path=service.path, lock=service.lock,
    )


def run(case):
    namespace = namespace_for(case.root, case.service)
    exec(case.source, namespace)
    return namespace


@pytest.fixture(autouse=True)
def shared_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(TRACK.parent))


@pytest.mark.parametrize("demo", DEMOS)
def test_state_access_and_mutations_are_inside_one_real_submission_lock(demo):
    tree = ast.parse(training_source(demo))
    locks = [node for node in tree.body if isinstance(node, ast.With)
             and isinstance(node.items[0].context_expr, ast.Call)
             and isinstance(node.items[0].context_expr.func, ast.Name)
             and node.items[0].context_expr.func.id == "submission_lock"]
    assert len(locks) == 1, "Lock the actual cell before loading any journal, not only the POST"
    assert any(isinstance(node, ast.ImportFrom) and node.module == "shared.sft_submission"
               and any(alias.name == "_submission_lock" and alias.asname == "submission_lock"
                       for alias in node.names) for node in tree.body)
    locked_calls = [node for node in ast.walk(locks[0]) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "create" for call in locked_calls)
    journals = {"STATE_PATH", "state_path", "file_state_path", "RUN_STATE_PATH"}
    for node in tree.body:
        if node is locks[0] or isinstance(node, ast.FunctionDef):
            continue
        assert not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                       and isinstance(call.func.value, ast.Name) and call.func.value.id in journals
                       and call.func.attr in {"exists", "read_text", "write_text", "replace"}
                       for call in ast.walk(node)), "Journal access before lock acquisition"


def guard_journal_access(monkeypatch, case):
    paths = {case.path, case.service.file_path}
    paths |= {path.with_suffix(".tmp") for path in paths}
    events = []
    for name in ("exists", "read_text", "write_text", "replace"):
        original = getattr(Path, name)

        def checked(path, *args, _name=name, _original=original, **kwargs):
            if path in paths:
                assert case.lock.is_file(), f"{_name} must use the same run lock for both journals"
                events.append((_name, path.name))
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, name, checked)
    return events


@pytest.mark.parametrize("demo", DEMOS)
@pytest.mark.parametrize("phase", ["upload:training", "job"])
def test_two_kernel_namespaces_share_one_upload_and_job_sequence(tmp_path, monkeypatch, demo, phase):
    case = make_case(demo, tmp_path / demo)
    events = guard_journal_access(monkeypatch, case)
    case.service.block_at = phase
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(run, case)
        try:
            assert case.service.entered.wait(15), "Writer did not reach its pending mutation"
            before = {path: path.read_bytes() for path in {case.path, case.service.file_path} if path.is_file()}
            event_count = len(events)
            loser = pool.submit(run, case)
            with pytest.raises(RuntimeError, match="Submission is locked"):
                loser.result(timeout=15)
            assert len(events) == event_count, "The loser must not even inspect stale state"
            assert all(path.read_bytes() == content for path, content in before.items())
            assert case.lock.is_file()
        finally:
            case.service.release.set()
        # The first video monitor imports the real SDK; cold Windows imports
        # can outlast the short inter-thread handshake, without a lock failure.
        winner.result(timeout=60)
    assert not case.lock.is_file()
    assert case.service.posts == ["upload:training", "upload:validation", "job"]
    saved = json.loads(case.path.read_bytes())
    assert saved["job_id"] == "job-known" and saved["submission_pending"] is False
    run(case)  # A fresh kernel reloads the winning IDs, never its own stale dict.
    assert case.service.posts == ["upload:training", "upload:validation", "job"]
    assert not case.lock.is_file()


def process_contender(demo, root, results):
    """Spawn-safe child runs the actual saved cell; no copied locking algorithm."""
    sys.path.insert(0, str(TRACK.parent))
    root = Path(root)
    service = Service(demo, root / "outputs" / "unit-run")
    namespace = namespace_for(root, service)
    original_exists = Path.exists
    original_read = Path.read_text
    journals = {service.path, service.file_path}

    def no_journal_read(path, *args, **kwargs):
        if path in journals:
            raise AssertionError("Competing process inspected state before acquiring the lock")
        return original_read(path, *args, **kwargs)

    def no_journal_exists(path):
        if path in journals:
            raise AssertionError("Competing process inspected state before acquiring the lock")
        return original_exists(path)

    try:
        with patch.object(Path, "exists", no_journal_exists), patch.object(Path, "read_text", no_journal_read):
            exec(compile(training_source(demo), demo, "exec"), namespace)
    except RuntimeError as error:
        results.put(("blocked", str(error)))
    except BaseException as error:
        results.put(("unexpected", type(error).__name__ + ": " + str(error)))
    else:
        results.put(("unexpected", "A second process entered the notebook submission"))


@pytest.mark.parametrize("demo", DEMOS)
def test_spawned_kernel_is_blocked_before_any_state_load(tmp_path, demo):
    from shared.sft_submission import _submission_lock

    case = make_case(demo, tmp_path / demo)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=process_contender, args=(demo, str(case.root), results))
    try:
        with _submission_lock(case.path):
            case.path.write_text('{"submission_pending":true}', encoding="utf-8")
            original = case.path.read_bytes()
            process.start()
            outcome, detail = results.get(timeout=30)
            assert outcome == "blocked" and "Submission is locked" in detail, (outcome, detail)
            process.join(timeout=15)
            assert not process.is_alive() and process.exitcode == 0
            assert case.path.read_bytes() == original and case.lock.is_file()
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=15)
        results.close()
        results.join_thread()
    assert not case.lock.is_file() and not case.service.posts


@pytest.mark.parametrize("demo", DEMOS)
@pytest.mark.parametrize("failure", [ConnectionError("accepted then response lost"), KeyboardInterrupt()])
def test_ambiguous_job_or_interrupt_releases_lock_but_never_reposts(tmp_path, demo, failure):
    case = make_case(demo, tmp_path / demo)
    case.service.fail_at = "job"
    case.service.failure = failure
    with pytest.raises(type(failure)):
        run(case)
    assert not case.lock.is_file()
    saved = json.loads(case.path.read_bytes())
    assert saved["submission_pending"] is True and not saved.get("job_id")
    with pytest.raises(RuntimeError, match="Previous submission outcome is unknown"):
        run(case)
    assert case.service.posts == ["upload:training", "upload:validation", "job"]
    assert not case.lock.is_file()


@pytest.mark.parametrize("demo", DEMOS)
def test_stale_lock_is_not_removed_or_bypassed(tmp_path, demo):
    case = make_case(demo, tmp_path / demo)
    case.lock.write_text("operator reconciliation required", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Submission is locked.*reconcile"):
        run(case)
    assert case.lock.read_text(encoding="utf-8") == "operator reconciliation required"
    assert not case.path.is_file() and not case.service.posts


@pytest.mark.parametrize("case_name", ["newer-state", "other-job", "locked"])
def test_chart_later_metrics_writer_locks_and_reloads_instead_of_using_stale_state(tmp_path, case_name):
    from shared.sft_submission import _submission_lock

    path = tmp_path / "run-state.json"
    current = {
        "job_id": "job-other" if case_name == "other-job" else "job-known",
        "newer_evidence": "must survive the older kernel",
    }
    path.write_text(json.dumps(current), encoding="utf-8")
    document = json.loads((TRACK / DEMOS[0] / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))
    source = next(cell_source(cell) for cell in document["cells"]
                  if cell["cell_type"] == "code" and "def last_non_null" in cell_source(cell))
    # Execute only the actual later journal-update block, excluding metric downloads.
    locked = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.With)
              and isinstance(node.items[0].context_expr, ast.Call)
              and isinstance(node.items[0].context_expr.func, ast.Name)
              and node.items[0].context_expr.func.id == "submission_lock"]
    assert len(locked) == 1
    events = []

    def load():
        assert lock_for(path).is_file()
        events.append("load")
        return json.loads(path.read_bytes())

    def save(state):
        assert lock_for(path).is_file()
        events.append("save")
        path.write_text(json.dumps(state), encoding="utf-8")

    namespace = {
        "STATE_PATH": path, "submission_lock": _submission_lock,
        "load_state": load, "save_state": save,
        "ft_job": SimpleNamespace(id="job-known"), "summary": {"job_id": "job-known"},
        "state": {"job_id": "job-known", "newer_evidence": "stale"},
    }
    code = compile(ast.fix_missing_locations(ast.Module(body=locked, type_ignores=[])), "chart-metrics-state", "exec")
    if case_name == "locked":
        with _submission_lock(path), pytest.raises(RuntimeError, match="Submission is locked"):
            exec(code, namespace)
        assert events == [] and json.loads(path.read_bytes()) == current
    elif case_name == "other-job":
        with pytest.raises(RuntimeError, match="Saved job changed"):
            exec(code, namespace)
        assert events == ["load"] and json.loads(path.read_bytes()) == current
    else:
        exec(code, namespace)
        assert events == ["load", "save"]
        assert json.loads(path.read_bytes()) == {**current, "metrics_summary": {"job_id": "job-known"}}
    assert not lock_for(path).is_file()