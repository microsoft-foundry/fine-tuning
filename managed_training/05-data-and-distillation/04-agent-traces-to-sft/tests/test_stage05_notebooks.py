"""Exercise real notebook functions and AST fragments, never whole notebooks or Azure."""

from __future__ import annotations

import ast
import asyncio
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest


STAGE = Path(__file__).resolve().parents[2]
MODULES = (
    "01-teacher-student-basics", "02-code-distillation",
    "03-synthetic-tool-use", "04-agent-traces-to-sft",
)
CONSTANTS = {
    "MODULE_NAME", "HYPERPARAMETERS", "TRAIN_FILE", "VALIDATION_FILE",
    "TEST_FILE", "REQUIRE_TOOLS", "JOB_SUFFIX", "EXPECTED", "CHAT_FILES",
}
SPEC = importlib.util.spec_from_file_location("stage05_runtime_notebook_tests", STAGE / "stage05_runtime.py")
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def source_text(cell):
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source) if any(line.endswith(("\n", "\r")) for line in source) else "\n".join(source)


def notebook(module):
    path = STAGE / module / "notebooks" / "demo.ipynb"
    return path, json.loads(path.read_text(encoding="utf-8"))


def notebook_namespace(module, root, data):
    """Compile actual top-level function definitions, without executing setup/cells."""
    path, document = notebook(module)
    tree = ast.parse("\n".join(source_text(cell) for cell in document["cells"] if cell["cell_type"] == "code"))
    setup = ast.parse(next(source_text(cell) for cell in document["cells"] if cell["cell_type"] == "code"))
    namespace = {
        "csv": csv, "hashlib": hashlib, "json": json, "math": math, "os": os,
        "Path": Path, "Random": Random,
        "ROOT": root, "DATA": data, "RUN_DIR": root / "outputs" / "unit-run",
        "STATE_PATH": root / "outputs" / "unit-run" / "run-state.json",
        "MODEL": "unit-model", "EXPECTED_MODEL_VERSION": "", "TRAINING_TYPE": "GlobalStandard",
        "FILE_POLL_SECONDS": 1, "FILE_TIMEOUT_SECONDS": 5,
        "JOB_POLL_SECONDS": 1, "JOB_TIMEOUT_SECONDS": 5,
    }
    # The trace notebook derives EXPECTED in a later cell; it is not a literal
    # setup constant and must not be evaluated as one by this AST harness.
    for node in setup.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in CONSTANTS):
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
    for name in (
        "create_job_once", "load_run_state", "submission_lock", "upload_file_once",
        "write_json_atomic", "wait_for_file_processed", "monitor_fine_tuning_job",
    ):
        namespace[name] = getattr(runtime, name)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def hosted_tree(module):
    path, document = notebook(module)
    source = next(source_text(cell) for cell in document["cells"]
                  if cell["cell_type"] == "code" and "with submission_lock(" in source_text(cell))
    return path, ast.parse(source)


def call_name(node):
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def submission_fragment(module):
    """Keep the real enclosing lock, state load, preflight, and context through submit.

    Only the later job-monitoring/metrics statements are omitted. No handwritten
    replacement of the locking or submission algorithm can mask a notebook race.
    """
    path, tree = hosted_tree(module)
    locked = deepcopy(next(node for node in tree.body if isinstance(node, ast.With)
                           and call_name(node.items[0].context_expr) == "submission_lock"))
    project_index = next(index for index, node in enumerate(locked.body)
                         if isinstance(node, ast.With)
                         and call_name(node.items[0].context_expr) == "create_project_context")
    project = locked.body[project_index]
    submitted_index = next(index for index, node in enumerate(project.body)
                           if isinstance(node, ast.Assign) and call_name(node.value) == "submit_or_resume_job")
    project.body = project.body[:submitted_index + 1]
    locked.body = locked.body[:project_index + 1]
    fragment = ast.fix_missing_locations(ast.Module(body=[locked], type_ignores=[]))
    return compile(fragment, str(path), "exec")


def forbidden(*_args, **_kwargs):
    raise AssertionError("A default-client POST, second writer, or hosted call was reached unexpectedly")


class Service:
    """Accept mutations before losing responses; inspect durable intent/IDs on disk."""

    def __init__(self, case):
        self.case = case
        self.calls = []
        self.options = []
        self.remote_files = {}
        self.remote_jobs = []
        self.fail_at = None
        self.failure = RuntimeError("Synthetic accepted-then-response-lost")
        self.missing_id_at = None
        self.fail_read_name = None
        self.on_context_enter = None
        self.on_file_retrieve = None
        self.block_at = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.contexts = 0
        self.closed_contexts = 0
        self.files = SimpleNamespace(create=forbidden, retrieve=forbidden)
        self.fine_tuning = SimpleNamespace(jobs=SimpleNamespace(create=forbidden, retrieve=forbidden))
        self.configured = SimpleNamespace(
            files=SimpleNamespace(create=self.create_file, retrieve=self.retrieve_file),
            fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=self.create_job, retrieve=self.retrieve_job)),
        )

    def state(self):
        return json.loads(self.case.path.read_text(encoding="utf-8"))

    def with_options(self, **kwargs):
        assert kwargs == {"max_retries": 0}
        self.options.append(kwargs)
        return self.configured

    @contextmanager
    def project_context(self, _config):
        assert self.case.lock.is_file()
        self.contexts += 1
        try:
            if self.on_context_enter is not None:
                self.on_context_enter()
            yield SimpleNamespace(openai_client=self)
        finally:
            self.closed_contexts += 1

    def after_acceptance(self, operation):
        if self.block_at == operation:
            self.entered.set()
            assert self.release.wait(15), "Test writer was not released"
        if self.fail_at == operation:
            raise self.failure

    def create_file(self, *, file, purpose):
        name, content, mime_type = file
        operation = f"upload:{name}"
        assert self.case.lock.is_file()
        saved = self.state()
        assert saved["pending_operation"] == operation
        assert not saved.get(f"{name}_file_id")
        assert purpose == "fine-tune" and mime_type == "application/jsonl"
        assert isinstance(content, bytes) and content == self.case.contents[name]
        file_id = f"unit-file-{len(self.remote_files) + 1}"
        resource = SimpleNamespace(id=file_id, status="processed", filename=name)
        self.remote_files[file_id] = resource
        self.calls.append(("POST", operation))
        self.after_acceptance(operation)
        return SimpleNamespace(id=None) if self.missing_id_at == operation else resource

    def retrieve_file(self, file_id, *, timeout):
        assert self.case.lock.is_file()
        assert math.isfinite(timeout) and 0 < timeout <= 5
        resource = self.remote_files[file_id]
        saved = self.state()
        assert saved["pending_operation"] is None
        assert saved[f"{resource.filename}_file_id"] == file_id, "Persist ID before any wait"
        self.calls.append(("GET_FILE", file_id))
        if self.on_file_retrieve is not None:
            self.on_file_retrieve(resource.filename)
        if self.fail_read_name == resource.filename:
            raise ValueError("Synthetic read-only failure")
        return resource

    def create_job(self, **payload):
        assert self.case.lock.is_file()
        saved = self.state()
        assert saved["pending_operation"] == "job_create" and not saved.get("job_id")
        for field, name in (("training_file", self.case.train), ("validation_file", self.case.validation)):
            assert payload[field] == saved[f"{name}_file_id"]
        assert payload["model"] == "unit-model"
        assert payload["method"] == {
            "type": "supervised", "supervised": {"hyperparameters": self.case.namespace["HYPERPARAMETERS"]},
        }
        assert payload["extra_body"] == {"trainingType": "GlobalStandard"}
        self.calls.append(("POST", "job_create"))
        job = SimpleNamespace(id="unit-job", status="succeeded")
        self.remote_jobs.append(job)
        self.after_acceptance("job_create")
        return SimpleNamespace(id=None) if self.missing_id_at == "job_create" else job

    def retrieve_job(self, job_id, *, timeout):
        assert self.case.lock.is_file()
        assert math.isfinite(timeout) and 0 < timeout <= 5
        assert self.state()["job_id"] == job_id and self.state()["pending_operation"] is None
        self.calls.append(("GET_JOB", job_id))
        return self.remote_jobs[-1]


@pytest.fixture(autouse=True)
def shared_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(STAGE.parent))


def prepared_trace_fixture(namespace):
    """Exercise the real preparation contract against three hash-pinned unit traces."""
    module = STAGE / MODULES[-1]
    spec = importlib.util.spec_from_file_location(
        "stage05_trace_integrity_notebook_tests", module / "trace_integrity.py",
    )
    assert spec is not None and spec.loader is not None
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    raw = {name: (module / "data" / name).read_bytes().replace(b"\r\n", b"\n")
           for name in trace.TEXT_SOURCE_FIXTURES}
    prompt = raw["zava_system_prompt.md"].decode("utf-8")
    tools = trace.loads(raw["zava_tools.json"].decode("utf-8"))
    originals = {}
    for index, name in enumerate(trace.SPLITS):
        call_id = f"unit-call-{index}"
        row = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Independent unit conversation {index}"},
                {"role": "assistant", "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "check_inventory", "arguments": '{"sku":"unit-sku"}'},
                }]},
                {"role": "tool", "tool_call_id": call_id, "content": '{"available":true}'},
                {"role": "assistant", "content": "Unit answer"},
            ],
            "tools": deepcopy(tools), "parallel_tool_calls": True,
        }
        originals[name] = [row]
        raw[name] = trace.jsonl_bytes([row])
    raw["traces-clean.jsonl"] = trace.jsonl_bytes(
        [row for rows in originals.values() for row in rows],
    )
    # This private module instance pins only the synthetic inputs. Production
    # source pins and all source/schema/policy/group/manifest validators stay intact.
    trace.SOURCE_HASHES = {name: trace.sha256(content) for name, content in raw.items()}
    trace.SOURCE_ROWS = {"traces-clean.jsonl": 3, **{name: 1 for name in trace.SPLITS}}
    source = namespace["ROOT"] / "data"
    source.mkdir()
    for name, content in raw.items():
        (source / name).write_bytes(content)
    payloads, _ = trace.build_prepared(source)
    for name, content in payloads.items():
        # Hashes and uploads pin bytes, not platform-dependent text newlines.
        (namespace["DATA"] / name).write_bytes(content)

    def validate_prepared(directory):
        return trace.validate_prepared(directory, source=source)

    namespace["validate_prepared"] = validate_prepared
    namespace["VALIDATED_PREPARATION_SHA256"] = hashlib.sha256(
        (namespace["DATA"] / "preparation-manifest.json").read_bytes(),
    ).hexdigest()
    return validate_prepared(namespace["DATA"])


def make_case(module, tmp_path):
    root = tmp_path / module
    data = root / "data" if module != MODULES[-1] else root / "outputs" / "prepared"
    data.mkdir(parents=True)
    namespace = notebook_namespace(module, root, data)
    train, validation = namespace["TRAIN_FILE"], namespace["VALIDATION_FILE"]
    if module == MODULES[-1]:
        datasets, lineage = prepared_trace_fixture(namespace)
        namespace["EXPECTED"] = {
            name: {"rows": info["rows"], "sha256": info["sha256"]}
            for name, info in lineage.items()
        }
    else:
        # Submission regressions must pass the actual schema, split and recipe
        # guards, including the teacher's source_recipe (which is not a file).
        # Do not replace the real notebook validator with a permissive fake.
        for name in namespace["EXPECTED"]:
            (data / name).write_bytes((STAGE / module / "data" / name).read_bytes())
        datasets, lineage = namespace["validate_data_contract"]()
    contents = {name: (data / name).read_bytes() for name in (train, validation)}
    endpoint = "https://unit-test.invalid/api/projects/stage05"
    namespace.update({
        "lineage": deepcopy(lineage), "datasets": deepcopy(datasets),
        "signature": {
            "project_endpoint": endpoint, "model": namespace["MODEL"],
            "expected_model_version": namespace["EXPECTED_MODEL_VERSION"],
            "training_type": namespace["TRAINING_TYPE"],
            "hyperparameters": deepcopy(namespace["HYPERPARAMETERS"]),
            "data": {name: info["sha256"] for name, info in namespace["EXPECTED"].items()},
        },
        "config": SimpleNamespace(project_endpoint=endpoint),
    })
    if module == MODULES[-1]:
        namespace["signature"]["preparation_sha256"] = namespace["VALIDATED_PREPARATION_SHA256"]
    path = namespace["STATE_PATH"]
    case = SimpleNamespace(
        namespace=namespace, fragment=submission_fragment(module), contents=contents, data=data,
        path=path, outputs=root / "outputs", lock=path.with_suffix(path.suffix + ".lock"),
        train=train, validation=validation,
    )
    case.service = Service(case)
    namespace["create_project_context"] = case.service.project_context
    return case


def run(case):
    exec(case.fragment, case.namespace)


def phase(case, kind):
    return "job_create" if kind == "job" else f"upload:{getattr(case, kind)}"


@pytest.mark.parametrize("module", MODULES[:3])
def test_submission_fixture_preserves_real_schema_lineage_and_file_only_signature(tmp_path, module):
    case = make_case(module, tmp_path)
    namespace = case.namespace
    datasets, lineage = namespace["validate_data_contract"]()
    assert datasets == namespace["datasets"] and lineage == namespace["lineage"]
    assert namespace["signature"]["data"] == {
        name: hashlib.sha256((case.data / name).read_bytes()).hexdigest()
        for name in namespace["EXPECTED"]
    }
    if module == MODULES[0]:
        assert lineage["source_recipe"] == {
            "seed": 42, "selected": 50, "teacher_candidates": 40,
            "accepted": 26, "held_out": 10,
        }
        assert "source_recipe" not in namespace["signature"]["data"]
    for name in namespace["CHAT_FILES"]:
        assert lineage[name]["assistant_tool_call_turns"] == namespace["validate_chat_rows"](
            name, datasets[name], require_tools=namespace["REQUIRE_TOOLS"],
        )
    assert not case.service.calls and case.service.contexts == 0 and not case.path.exists()


def test_trace_fixture_matches_real_prepared_getters_and_manifest_bytes(tmp_path):
    case = make_case(MODULES[-1], tmp_path)
    namespace = case.namespace
    datasets, lineage = namespace["validate_prepared"](case.data)
    manifest_bytes = (case.data / "preparation-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert datasets == namespace["datasets"] and lineage == namespace["lineage"]
    assert set(datasets) == {"traces-clean.jsonl", case.train, case.validation, namespace["TEST_FILE"]}
    assert len(manifest["records"]) == len(datasets["traces-clean.jsonl"]) == 3
    assert len(manifest["groups"]) == 3
    assert namespace["VALIDATED_PREPARATION_SHA256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert namespace["signature"]["preparation_sha256"] == namespace["VALIDATED_PREPARATION_SHA256"]
    for name, details in manifest["artifacts"].items():
        assert details["sha256"] == hashlib.sha256((case.data / name).read_bytes()).hexdigest()
    for name, rows in datasets.items():
        assert lineage[name] == manifest["artifacts"][name] == namespace["EXPECTED"][name]
        assert lineage[name]["rows"] == len(rows)
    assert not case.service.calls and case.service.contexts == 0 and not case.path.exists()


@pytest.mark.parametrize("snapshot", ["datasets", "lineage", "VALIDATED_PREPARATION_SHA256"])
def test_trace_preflight_compares_all_original_prepared_snapshots_before_hosted_work(tmp_path, snapshot):
    case = make_case(MODULES[-1], tmp_path)
    if snapshot == "datasets":
        case.namespace[snapshot][case.train][0]["messages"][-1]["content"] = "Changed snapshot"
    elif snapshot == "lineage":
        case.namespace[snapshot][case.train]["rows"] += 1
    else:
        case.namespace[snapshot] = "not-the-validated-manifest-digest"
    with pytest.raises(RuntimeError, match="Prepared data changed after validation"):
        run(case)
    assert not case.service.calls and case.service.contexts == 0
    assert not case.path.exists() and not case.lock.exists()


@pytest.mark.parametrize("artifact", [
    "traces-clean.jsonl", "train.jsonl", "val.jsonl", "test.jsonl",
    "zava_system_prompt.md", "zava_tools.json", "preparation-manifest.json",
])
def test_trace_preflight_independently_rebuilds_preparation_despite_replaced_expected_hash(tmp_path, artifact):
    case = make_case(MODULES[-1], tmp_path)
    path = case.data / artifact
    content = path.read_bytes() + b" "
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    if artifact == "preparation-manifest.json":
        case.namespace["VALIDATED_PREPARATION_SHA256"] = digest
    elif artifact in case.namespace["EXPECTED"]:
        case.namespace["EXPECTED"][artifact]["sha256"] = digest
        case.namespace["lineage"][artifact]["sha256"] = digest
    with pytest.raises(ValueError, match="independently rebuilt source-derived preparation"):
        run(case)
    assert not case.service.calls and case.service.contexts == 0
    assert not case.path.exists() and not case.lock.exists()


@pytest.mark.parametrize("module", MODULES)
def test_all_four_notebooks_parse_and_lock_before_loading_state_or_hosted_operations(module):
    _, document = notebook(module)
    identifiers = [cell.get("id") or cell.get("metadata", {}).get("id") for cell in document["cells"]]
    assert all(identifiers) and len(set(identifiers)) == len(identifiers)
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(source_text(cell))
            assert cell.get("execution_count") is None and not cell.get("outputs")
    _, tree = hosted_tree(module)
    locked = next(node for node in tree.body if isinstance(node, ast.With)
                  and call_name(node.items[0].context_expr) == "submission_lock")
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and call_name(node) == "load_state"]
    assert len(loads) == 1 and loads[0] in list(ast.walk(locked))
    context = next(node for node in locked.body if isinstance(node, ast.With)
                   and call_name(node.items[0].context_expr) == "create_project_context")
    assert loads[0].lineno < context.lineno
    assert any(isinstance(node, ast.Call) and call_name(node) == "submit_or_resume_job"
               for node in ast.walk(context))
    assert any(isinstance(node, ast.Call) and call_name(node) == "monitor_fine_tuning_job"
               for node in ast.walk(context))


@pytest.mark.parametrize("module", MODULES)
def test_real_notebook_submission_journals_all_ids_and_rerun_reuses_the_job(tmp_path, module):
    case = make_case(module, tmp_path)
    run(case)
    service = case.service
    assert [call for call in service.calls if call[0] == "POST"] == [
        ("POST", phase(case, "train")), ("POST", phase(case, "validation")), ("POST", "job_create"),
    ]
    saved = service.state()
    assert saved["pending_operation"] is None and saved["job_id"] == "unit-job"
    assert saved["signature"] == case.namespace["signature"]
    original = case.path.read_bytes()
    calls = list(service.calls)
    run(case)
    assert service.calls == calls and case.path.read_bytes() == original
    assert len(service.remote_files) == 2 and len(service.remote_jobs) == 1
    assert service.options and all(option == {"max_retries": 0} for option in service.options)
    assert service.contexts == service.closed_contexts == 2 and not case.lock.exists()
    with runtime.submission_lock(case.path, outputs_root=case.outputs):
        assert runtime.monitor_fine_tuning_job(service, saved["job_id"], timeout_seconds=5).id == "unit-job"


@pytest.mark.parametrize("module", MODULES)
def test_run_lock_covers_state_load_and_every_durable_pending_and_id_write(tmp_path, monkeypatch, module):
    case = make_case(module, tmp_path)
    load_state = case.namespace["load_state"]
    write_state = runtime.write_json_atomic
    events = []

    def assert_exclusive_owner():
        assert case.lock.is_file()
        with pytest.raises(RuntimeError, match="locked"):
            with runtime.submission_lock(case.path, outputs_root=case.outputs):
                forbidden()
        assert case.lock.is_file(), "A rejected contender must not remove the owner's lock"

    def checked_load(signature):
        assert_exclusive_owner()
        events.append("load")
        return load_state(signature)

    def checked_write(path, value, **kwargs):
        assert Path(path) == case.path
        assert_exclusive_owner()
        result = write_state(path, value, **kwargs)
        assert_exclusive_owner()
        assert json.loads(case.path.read_text(encoding="utf-8")) == value
        events.append(value["pending_operation"])
        return result

    monkeypatch.setitem(case.namespace, "load_state", checked_load)
    monkeypatch.setattr(runtime, "write_json_atomic", checked_write)
    run(case)
    assert events == [
        "load", phase(case, "train"), None,
        phase(case, "validation"), None, "job_create", None,
    ]
    assert case.service.state()["job_id"] == "unit-job" and not case.lock.exists()


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("kind", ["train", "validation", "job"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
def test_accepted_then_response_lost_or_interrupted_is_never_replayed(tmp_path, module, kind, error_type):
    case = make_case(module, tmp_path)
    service = case.service
    service.fail_at = phase(case, kind)
    service.failure = error_type("secret-sentinel-must-not-be-written")
    with pytest.raises(error_type) as caught:
        run(case)
    if error_type is RuntimeError:
        assert "secret-sentinel" not in str(caught.value)
    assert service.state()["pending_operation"] == service.fail_at
    assert service.calls.count(("POST", service.fail_at)) == 1
    calls, saved, contexts = list(service.calls), case.path.read_bytes(), service.contexts
    service.fail_at = None
    with pytest.raises(RuntimeError, match="ambiguous"):
        run(case)
    assert service.calls == calls and case.path.read_bytes() == saved
    assert service.contexts == contexts == service.closed_contexts
    assert b"secret-sentinel" not in saved and not case.lock.exists()


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("kind", ["train", "validation", "job"])
def test_missing_id_keeps_pending_intent_and_blocks_every_later_mutation(tmp_path, module, kind):
    case = make_case(module, tmp_path)
    case.service.missing_id_at = phase(case, kind)
    with pytest.raises(RuntimeError, match="valid id"):
        run(case)
    assert case.service.state()["pending_operation"] == phase(case, kind)
    before = list(case.service.calls)
    with pytest.raises(RuntimeError, match="ambiguous"):
        run(case)
    assert case.service.calls == before


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("kind", ["train", "validation", "job"])
def test_failed_pending_checkpoint_prevents_the_corresponding_post(tmp_path, monkeypatch, module, kind):
    case = make_case(module, tmp_path)
    original = runtime.write_json_atomic

    def fail_pending(path, value, **kwargs):
        if value.get("pending_operation") == phase(case, kind):
            raise OSError("Synthetic failure before POST")
        return original(path, value, **kwargs)

    monkeypatch.setattr(runtime, "write_json_atomic", fail_pending)
    with pytest.raises(OSError, match="before POST"):
        run(case)
    assert ("POST", phase(case, kind)) not in case.service.calls
    assert not case.lock.exists()


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("kind", ["train", "validation", "job"])
def test_failed_id_checkpoint_retains_pending_intent_without_a_wait_or_replay(tmp_path, monkeypatch, module, kind):
    case = make_case(module, tmp_path)
    original = runtime.write_json_atomic
    key = "job_id" if kind == "job" else f"{getattr(case, kind)}_file_id"

    def fail_id(path, value, **kwargs):
        if value.get(key) and value.get("pending_operation") is None:
            raise OSError("Synthetic ID checkpoint failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(runtime, "write_json_atomic", fail_id)
    with pytest.raises(OSError, match="ID checkpoint"):
        run(case)
    saved = case.service.state()
    assert saved["pending_operation"] == phase(case, kind) and not saved.get(key)
    if kind != "job":
        last_id = next(reversed(case.service.remote_files))
        assert ("GET_FILE", last_id) not in case.service.calls
    before = list(case.service.calls)
    with pytest.raises(RuntimeError, match="ambiguous"):
        run(case)
    assert case.service.calls == before


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("kind", ["train", "validation"])
def test_saved_upload_ids_survive_poll_failure_and_are_reused(tmp_path, module, kind):
    case = make_case(module, tmp_path)
    case.service.fail_read_name = getattr(case, kind)
    with pytest.raises(ValueError, match="read-only"):
        run(case)
    saved = case.service.state()
    assert saved[f"{getattr(case, kind)}_file_id"] and saved["pending_operation"] is None
    case.service.fail_read_name = None
    run(case)
    for operation in (phase(case, "train"), phase(case, "validation"), "job_create"):
        assert case.service.calls.count(("POST", operation)) == 1


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("pending", [False, True])
def test_legacy_state_without_upload_intent_is_not_silently_retried(tmp_path, module, pending):
    case = make_case(module, tmp_path)
    with runtime.submission_lock(case.path, outputs_root=case.outputs):
        runtime.write_json_atomic(case.path, {
            "signature": case.namespace["signature"], "submission_pending": pending,
        }, outputs_root=case.outputs)
    original = case.path.read_bytes()
    with pytest.raises(RuntimeError, match="ambiguous|older format"):
        run(case)
    assert not case.service.calls and case.service.contexts == 0
    assert case.path.read_bytes() == original


@pytest.mark.parametrize("module", MODULES)
def test_upload_hash_is_rechecked_after_preflight_and_context_creation(tmp_path, module):
    case = make_case(module, tmp_path)
    case.service.on_context_enter = lambda: (case.data / case.train).write_bytes(b"changed after preflight\n")
    with pytest.raises(ValueError, match="changed after validation"):
        run(case)
    assert not case.service.calls and not case.path.exists()
    assert case.service.contexts == case.service.closed_contexts == 1


@pytest.mark.parametrize("module", MODULES)
def test_validation_drift_during_training_poll_cannot_upload_or_create_a_job(tmp_path, module):
    case = make_case(module, tmp_path)

    def mutate_after_training_read(name):
        if name == case.train:
            (case.data / case.validation).write_bytes(b"changed between uploads\n")

    case.service.on_file_retrieve = mutate_after_training_read
    with pytest.raises(ValueError, match="changed after validation"):
        run(case)
    assert [call for call in case.service.calls if call[0] == "POST"] == [("POST", phase(case, "train"))]
    assert case.service.state()[f"{case.train}_file_id"]
    assert case.service.state()["pending_operation"] is None


@pytest.mark.parametrize("module", MODULES)
def test_upload_sends_the_same_immutable_bytes_that_were_hash_checked(tmp_path, module):
    case = make_case(module, tmp_path)
    configure = case.service.with_options
    changed = False

    def mutate_after_snapshot(**kwargs):
        nonlocal changed
        if not changed:
            changed = True
            (case.data / case.train).write_bytes(b"changed after upload buffer validation\n")
        return configure(**kwargs)

    case.service.with_options = mutate_after_snapshot
    run(case)
    # Service.create_file asserts the payload is the original hash-pinned bytes,
    # not a newly opened path whose contents changed after the check.
    assert changed and len(case.service.remote_files) == 2 and len(case.service.remote_jobs) == 1


@pytest.mark.parametrize("module", MODULES)
def test_concurrent_notebook_threads_exclude_second_load_and_keep_one_submission(tmp_path, module):
    case = make_case(module, tmp_path)
    case.service.block_at = phase(case, "train")
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(run, case)
        try:
            assert case.service.entered.wait(15), "Writer did not reach its journaled upload"
            saved = case.path.read_bytes()
            loser = pool.submit(run, case)
            with pytest.raises(RuntimeError, match="locked"):
                loser.result(timeout=15)
            assert case.path.read_bytes() == saved and case.lock.is_file()
            assert case.service.contexts == 1
        finally:
            case.service.release.set()
        winner.result(timeout=15)
    assert [call for call in case.service.calls if call[0] == "POST"] == [
        ("POST", phase(case, "train")), ("POST", phase(case, "validation")), ("POST", "job_create"),
    ]
    assert not case.lock.exists()
    run(case)
    assert len(case.service.remote_jobs) == 1 and len(case.service.remote_files) == 2


def process_contender(module, root, path, results):
    """Spawn-safe child: the actual notebook lock must fail before load_state."""
    sys.path.insert(0, str(STAGE.parent))
    namespace = {
        "ROOT": Path(root), "STATE_PATH": Path(path), "signature": {},
        "submission_lock": runtime.submission_lock, "load_state": forbidden,
    }
    try:
        exec(submission_fragment(module), namespace)
    except RuntimeError as error:
        results.put(("blocked", str(error)))
    except BaseException as error:
        results.put(("unexpected", type(error).__name__))
    else:
        results.put(("unexpected", "Writer entered the locked notebook"))


@pytest.mark.parametrize("module", MODULES)
def test_spawned_process_cannot_load_or_submit_while_the_run_is_locked(tmp_path, module):
    case = make_case(module, tmp_path)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=process_contender, args=(
        module, str(case.namespace["ROOT"]), str(case.path), results,
    ))
    try:
        with runtime.submission_lock(case.path, outputs_root=case.outputs):
            runtime.write_json_atomic(case.path, {
                "schema_version": 1, "signature": case.namespace["signature"],
                "pending_operation": phase(case, "train"),
            }, outputs_root=case.outputs)
            saved = case.path.read_bytes()
            process.start()
            outcome, detail = results.get(timeout=30)
            assert outcome == "blocked" and "Submission is locked" in detail
            assert case.path.read_bytes() == saved and case.lock.is_file()
            process.join(timeout=15)
            assert not process.is_alive() and process.exitcode == 0
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=15)
        results.close()
        results.join_thread()
    assert not case.lock.exists() and not case.service.calls


def real_data_namespace(module, tmp_path):
    root = tmp_path / module
    data = root / "data"
    data.mkdir(parents=True)
    namespace = notebook_namespace(module, root, data)
    for name in namespace["EXPECTED"]:
        (data / name).write_bytes((STAGE / module / "data" / name).read_bytes())
    datasets, lineage = namespace["validate_data_contract"]()
    namespace.update({
        "datasets": datasets, "lineage": lineage,
        "signature": {"model": "unit-model", "data": {name: info["sha256"]
                       for name, info in namespace["EXPECTED"].items()}},
        "config": object(), "create_project_context": forbidden,
    })
    return namespace


@pytest.mark.parametrize("module", MODULES[:3])
def test_real_preflight_rejects_file_changes_before_any_hosted_call(tmp_path, module):
    namespace = real_data_namespace(module, tmp_path)
    changed = namespace["DATA"] / namespace["TRAIN_FILE"]
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        exec(submission_fragment(module), namespace)
    assert not namespace["STATE_PATH"].exists()


@pytest.mark.parametrize("module,name", [
    (MODULES[0], "qa.jsonl"),
    (MODULES[0], "sarcasm-validation.jsonl"),
    (MODULES[1], "validation_data.jsonl"),
    (MODULES[2], "validation.jsonl"),
    (MODULES[2], "test.jsonl"),
])
def test_real_preflight_rechecks_validation_heldout_and_source_inputs(tmp_path, module, name):
    namespace = real_data_namespace(module, tmp_path)
    changed = namespace["DATA"] / name
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        exec(submission_fragment(module), namespace)
    assert not namespace["STATE_PATH"].exists()
    assert not namespace["STATE_PATH"].with_suffix(".json.lock").exists()


@pytest.mark.parametrize("module", MODULES[:3])
def test_real_preflight_rechecks_chat_schema_even_if_expected_hash_is_replaced(tmp_path, module):
    namespace = real_data_namespace(module, tmp_path)
    name = namespace["TRAIN_FILE"]
    rows = deepcopy(namespace["datasets"][name])
    rows[0]["messages"] = []
    content = ("".join(json.dumps(row) + "\n" for row in rows)).encode("utf-8")
    (namespace["DATA"] / name).write_bytes(content)
    namespace["EXPECTED"][name]["sha256"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="non-empty messages"):
        exec(submission_fragment(module), namespace)
    assert not namespace["STATE_PATH"].exists()


@pytest.mark.parametrize("module", MODULES[:3])
def test_real_revalidation_compares_hash_lineage_to_the_original_snapshot(tmp_path, module):
    namespace = real_data_namespace(module, tmp_path)
    name = namespace["TRAIN_FILE"]
    content = (namespace["DATA"] / name).read_bytes() + b" "
    (namespace["DATA"] / name).write_bytes(content)
    namespace["EXPECTED"][name]["sha256"] = hashlib.sha256(content).hexdigest()
    # Whitespace preserves rows/schema, but the original recorded hash must win.
    with pytest.raises(RuntimeError, match="Data changed after validation"):
        exec(submission_fragment(module), namespace)
    assert not namespace["STATE_PATH"].exists()


@pytest.mark.parametrize("module", MODULES[:3])
def test_preflight_compares_fresh_lineage_with_the_original_validated_lineage(tmp_path, module):
    namespace = real_data_namespace(module, tmp_path)
    changed_lineage = deepcopy(namespace["lineage"])
    changed_lineage[namespace["TRAIN_FILE"]]["assistant_tool_call_turns"] += 1
    namespace["validate_data_contract"] = lambda: (deepcopy(namespace["datasets"]), changed_lineage)
    with pytest.raises(RuntimeError, match="Data changed after validation"):
        exec(submission_fragment(module), namespace)
    assert not namespace["STATE_PATH"].exists()


def test_teacher_preflight_repeats_its_original_source_recipe_and_system_policy(tmp_path):
    namespace = real_data_namespace(MODULES[0], tmp_path)
    name = namespace["TRAIN_FILE"]
    rows = deepcopy(namespace["datasets"][name])
    rows[0]["messages"][0]["content"] = "Changed unit-test policy"
    content = ("".join(json.dumps(row) + "\n" for row in rows)).encode("utf-8")
    (namespace["DATA"] / name).write_bytes(content)
    namespace["EXPECTED"][name]["sha256"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="Clippy system prompt"):
        exec(submission_fragment(MODULES[0]), namespace)