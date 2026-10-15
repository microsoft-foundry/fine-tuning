from __future__ import annotations

import ast
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from shared.config import FoundryConfig
from shared.dataset_validation import DatasetValidationError, validate_jsonl


ROOT = Path(__file__).parents[2]
DEMO = ROOT / "03-core-rft" / "03-rft-code-repair"
NOTEBOOK = json.loads((DEMO / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))
SOURCES = {
    cell["id"]: "".join(cell["source"])
    for cell in NOTEBOOK["cells"]
    if cell["cell_type"] == "code"
}
OFFLINE_CELLS = ("mai-prepare", "mai-data", "mai-grader", "mai-recipe")
EXPECTED_INPUTS = {
    "training": (50, 47678, "00398c62ebc0bf7ab9b4d4abe6231452a6ca387af56ae6bc9558c483bf7ab28c"),
    "validation": (15, 15177, "2944c6a1e73f3a32a98ea8dc957d4dcdc06f3fdf32c5b219a9592f9a46c437bc"),
}
GRADER_HASH = "57ec9f07ff12f001f9cdbf8a1aaa9a2e9d5e2e080d9dea1a000e0ae1fc7a64d7"


def execute(cell_id, namespace):
    exec(compile(SOURCES[cell_id], f"code-repair:{cell_id}", "exec"), namespace)


def prepare(monkeypatch, demo, *, notebooks_cwd=False):
    monkeypatch.chdir(demo / "notebooks" if notebooks_cwd else demo)

    def no_cloud(*_args, **_kwargs):
        raise AssertionError("Offline preparation attempted a cloud client")

    monkeypatch.setattr("shared.auth.create_project_context", no_cloud)
    namespace = {}
    for cell_id in OFFLINE_CELLS:
        execute(cell_id, namespace)
    return namespace


@pytest.fixture
def demo_copy(tmp_path):
    target = tmp_path / "code-repair"
    shutil.copytree(DEMO, target, ignore=shutil.ignore_patterns("outputs", ".env", "__pycache__"))
    return target


@pytest.fixture
def configured(monkeypatch, demo_copy):
    namespace = prepare(monkeypatch, demo_copy)
    namespace["load_foundry_config"] = lambda _path: FoundryConfig(
        project_endpoint="https://example.test/api/projects/offline"
    )
    namespace["version"] = lambda _name: "offline-test"
    monkeypatch.setenv("FOUNDRY_RESOURCE_LOCATION", "offline-region")
    monkeypatch.delenv("FOUNDRY_EXISTING_CODE_REPAIR_RFT_JOB_ID", raising=False)
    execute("mai-config-state", namespace)
    return namespace


@pytest.mark.parametrize("notebooks_cwd", [False, True])
def test_offline_preparation_uses_exact_inputs_without_cloud_or_outputs(
    monkeypatch, demo_copy, notebooks_cwd
):
    namespace = prepare(monkeypatch, demo_copy, notebooks_cwd=notebooks_cwd)
    assert namespace["MODEL"] == "MAI-Code-1.1-Flash-2026-09-15"
    assert namespace["TRAINING_TYPE"] == "globalStandard"
    assert namespace["DEMO_ROOT"] == demo_copy
    assert not namespace["OUTPUT_ROOT"].exists()
    assert "config" not in namespace
    assert namespace["METHOD"] == {
        "type": "reinforcement",
        "reinforcement": {"grader": namespace["GRADER"]},
    }
    assert len(namespace["JOB_SUFFIX"]) <= 18
    for label, (rows, byte_count, digest) in EXPECTED_INPUTS.items():
        contract = namespace["DATASETS"][label]
        assert len(namespace["dataset_rows"][label]) == rows
        assert contract["path"].stat().st_size == byte_count
        assert namespace["dataset_hashes"][label] == digest
    assert hashlib.sha256(namespace["GRADER_PATH"].read_bytes()).hexdigest() == GRADER_HASH


def test_metadata_and_navigation_include_the_new_model():
    catalog = yaml.safe_load((ROOT / "catalog.yml").read_text(encoding="utf-8"))
    demo = next(item for item in catalog["demos"] if item["id"] == "rft-mai-code-repair")
    assert demo["training"]["base_models"] == ["MAI-Code-1.1-Flash-2026-09-15"]
    assert demo["training"]["evaluator_models"] == ["o3-mini-2025-01-31"]
    assert demo["techniques"] == ["rft"]
    assert demo["status"] == "preview"
    assert demo["validation"]["status"] == "partial"
    for path in (
        ROOT / "03-core-rft" / "README.md",
        ROOT / "LEARNING_PATH.md",
        ROOT / "TASK_INDEX.md",
        ROOT / "MODEL_AND_NOTEBOOK_INDEX.md",
    ):
        assert "03-rft-code-repair" in path.read_text(encoding="utf-8")


def test_byte_pinned_inputs_disable_git_text_conversion():
    paths = list((DEMO / "data" / "preserved").glob("*.jsonl"))
    paths.append(DEMO / "graders" / "code_repair_quality.json")
    result = subprocess.run(
        ["git", "check-attr", "-z", "text", "--", *[str(path.relative_to(ROOT.parent)) for path in paths]],
        cwd=ROOT.parent, check=True, capture_output=True,
    )
    fields = result.stdout.rstrip(b"\0").split(b"\0")
    assert len(fields) == len(paths) * 3
    for index in range(0, len(fields), 3):
        assert fields[index + 1:index + 3] == [b"text", b"unset"]


@pytest.mark.parametrize("change", [
    lambda row: row.pop("requirements"),
    lambda row: row.update(reference_patch=""),
    lambda row: row.update(test_expectations=123),
    lambda row: row.update(messages=[]),
    lambda row: row["messages"][0].update(role="assistant"),
    lambda row: row["messages"][1].update(content=""),
    lambda row: row["messages"].append({"role": "assistant", "content": "answer"}),
    lambda row: row.update(extra_field="unexpected"),
    lambda row: row["messages"][1].update(content="api_key=not-a-real-secret"),
])
def test_rft_validator_does_not_bypass_message_reference_or_secret_checks(
    monkeypatch, demo_copy, tmp_path, change
):
    namespace = prepare(monkeypatch, demo_copy)
    row = deepcopy(namespace["dataset_rows"]["training"][0])
    change(row)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        validate_jsonl(
            path, required_keys=("messages", *namespace["GRADER_FIELDS"]),
            validate_messages=False, row_validator=namespace["validate_rft_row"],
        ).require_valid()


def test_prompt_overlap_is_rejected_even_when_reference_metadata_differs(monkeypatch, demo_copy):
    monkeypatch.chdir(demo_copy)
    namespace = {}
    execute("mai-prepare", namespace)
    train = demo_copy / "data" / "preserved" / "code-repair-rft-training.jsonl"
    valid = demo_copy / "data" / "preserved" / "code-repair-rft-validation.jsonl"
    rows = [json.loads(line) for line in valid.read_text(encoding="utf-8").splitlines()]
    rows[0]["messages"] = json.loads(train.read_text(encoding="utf-8").splitlines()[0])["messages"]
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    valid.write_bytes(raw)
    manifest_path = demo_copy / "data" / "hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][1].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="user prompts overlap"):
        execute("mai-data", namespace)


@pytest.mark.parametrize("field", ["project_endpoint", "region", "recipe_sha256"])
def test_saved_state_mismatch_stops_before_remote_operations(configured, field):
    namespace = configured
    namespace["runtime"].resource_context[field] = "different"
    namespace["runtime"].write(namespace["STATE_PATH"])
    with pytest.raises(RuntimeError, match="different project, region, or recipe"):
        execute("mai-config-state", namespace)


@pytest.mark.parametrize("change", ["dataset", "grader", "recipe", "training_type"])
def test_changed_inputs_are_rejected_before_upload_or_client_creation(configured, change):
    namespace = configured
    if change == "dataset":
        path = namespace["DATASETS"]["training"]["path"]
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "grader":
        path = namespace["GRADER_PATH"]
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "recipe":
        namespace["METHOD"]["reinforcement"]["grader"]["model"] = "other-model"
    else:
        namespace["TRAINING_TYPE"] = "standard"
    with pytest.raises(ValueError, match="changed after preparation"):
        execute("mai-upload", namespace)


class FakeFiles:
    def __init__(self):
        self.data = []
        self.contents = {}
        self.created = 0
        self.ambiguous = False

    def list(self):
        return SimpleNamespace(data=self.data)

    def create(self, *, file, purpose):
        self.created += 1
        name, stream, _ = file
        raw = stream.read()
        item = SimpleNamespace(
            id=f"file-offline-{self.created}", filename=name, purpose=purpose,
            status="processed", bytes=len(raw),
        )
        self.data.append(item)
        self.contents[item.id] = raw
        if self.ambiguous:
            raise RuntimeError("Ambiguous upload response")
        return item

    def retrieve(self, file_id):
        return next(item for item in self.data if item.id == file_id)

    def content(self, file_id):
        return io.BytesIO(self.contents[file_id])


class FakeJobs:
    def __init__(self):
        self.data = []
        self.created = 0
        self.ambiguous = False

    def list(self):
        return SimpleNamespace(data=self.data)

    def retrieve(self, job_id):
        return next(job for job in self.data if job.id == job_id)

    def create(self, **payload):
        self.created += 1
        extra_body = payload.pop("extra_body", {})
        job = SimpleNamespace(
            id="job-offline-recorded", status="succeeded", error=None,
            fine_tuned_model="artifact-offline-result", result_files=["file-offline-metrics"],
            **deepcopy(payload), **extra_body,
        )
        effective_method = deepcopy(payload["method"])
        effective_method["reinforcement"]["hyperparameters"] = {"n_epochs": 1}
        job.model_dump = lambda **_kwargs: {"method": effective_method}
        self.data.append(job)
        if self.ambiguous:
            raise RuntimeError("Ambiguous submission response")
        return job


class FakeClient:
    def __init__(self):
        self.files = FakeFiles()
        self.fine_tuning = SimpleNamespace(jobs=FakeJobs())
        self.options = []
        self.contexts = 0
        self.files.contents["file-offline-metrics"] = (
            b"step,train_mean_reward,full_valid_mean_reward,pg_loss_sum\n"
            b"1,0.3,0.4,1.2\n2,0.6,0.5,0.9\n"
        )

    def with_options(self, **options):
        self.options.append(options)
        return self

    def context(self, _config):
        self.contexts += 1
        return nullcontext(SimpleNamespace(openai_client=self))


def attach_client(namespace):
    client = FakeClient()
    namespace["create_project_context"] = client.context
    return client


def test_mocked_workflow_persists_ids_before_waiting_and_resumes_without_duplicates(configured, capsys):
    namespace = configured
    client = attach_client(namespace)
    wait = namespace["wait_for_file"]

    def recorded_before_wait(client, file_id, **kwargs):
        saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
        assert file_id in saved["operation_ids"].values()
        return wait(client, file_id, **kwargs)

    namespace["wait_for_file"] = recorded_before_wait
    for cell_id in ("mai-upload", "mai-submit", "mai-monitor", "mai-metrics"):
        execute(cell_id, namespace)
    assert client.files.created == 2
    assert client.fine_tuning.jobs.created == 1
    assert all(options == {"max_retries": 0} for options in client.options)
    assert client.fine_tuning.jobs.data[0].method == namespace["METHOD"]
    assert client.fine_tuning.jobs.data[0].trainingType == namespace["TRAINING_TYPE"]
    saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
    assert saved["operation_ids"]["job"] == "job-offline-recorded"
    assert saved["operation_ids"]["fine_tuned_model"] == "artifact-offline-result"
    assert saved["resource_context"]["submission_state"] == "recorded"
    manifest = json.loads((namespace["OUTPUT_ROOT"] / "experiment-manifest.json").read_text(encoding="utf-8"))
    assert manifest["hyperparameters"]["effective"] == {"n_epochs": 1}
    assert manifest["metrics"]["service-result-1"]["metrics"]["full_valid_mean_reward"] == {
        "observations": 2, "first": 0.4, "last": 0.5, "min": 0.4, "max": 0.5,
    }

    execute("mai-config-state", namespace)
    execute("mai-upload", namespace)
    execute("mai-submit", namespace)
    assert client.files.created == 2
    assert client.fine_tuning.jobs.created == 1
    assert namespace["submission"].reused
    output = capsys.readouterr().out
    assert "https://example.test" not in output
    assert "offline-region" not in output
    for value in saved["operation_ids"].values():
        assert value not in output


def test_ambiguous_upload_is_not_blindly_retried(configured):
    namespace = configured
    client = attach_client(namespace)
    client.files.ambiguous = True
    with pytest.raises(RuntimeError, match="Ambiguous upload"):
        execute("mai-upload", namespace)
    saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
    assert saved["resource_context"]["training_upload_state"] == "in-flight"
    assert "training" not in saved["operation_ids"]
    execute("mai-config-state", namespace)
    with pytest.raises(RuntimeError, match="upload outcome is unknown"):
        execute("mai-upload", namespace)
    assert client.files.created == 1
    assert client.contexts == 1


def test_ambiguous_submission_recovers_only_a_confirmed_matching_job(configured, monkeypatch):
    namespace = configured
    client = attach_client(namespace)
    execute("mai-upload", namespace)
    client.fine_tuning.jobs.ambiguous = True
    with pytest.raises(RuntimeError, match="Ambiguous submission"):
        execute("mai-submit", namespace)
    saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
    assert saved["resource_context"]["submission_state"] == "in-flight"
    assert "job" not in saved["operation_ids"]
    execute("mai-config-state", namespace)
    with pytest.raises(RuntimeError, match="Submission outcome is unknown"):
        execute("mai-submit", namespace)
    assert client.fine_tuning.jobs.created == 1

    monkeypatch.setenv("FOUNDRY_EXISTING_CODE_REPAIR_RFT_JOB_ID", client.fine_tuning.jobs.data[0].id)
    execute("mai-config-state", namespace)
    execute("mai-submit", namespace)
    assert namespace["submission"].reused
    assert client.fine_tuning.jobs.created == 1


def test_recorded_remote_bytes_are_checked_before_submission(configured):
    namespace = configured
    client = attach_client(namespace)
    execute("mai-upload", namespace)
    file_id = namespace["runtime"].operation_ids["training"]
    raw = client.files.contents[file_id]
    client.files.contents[file_id] = b" " + raw[1:]
    with pytest.raises(RuntimeError, match="bytes do not match"):
        execute("mai-upload", namespace)
    assert client.files.created == 2
    assert client.fine_tuning.jobs.created == 0
    with pytest.raises(RuntimeError, match="require successful byte verification"):
        execute("mai-submit", namespace)
    assert client.fine_tuning.jobs.created == 0


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_or_cancelled_job_is_recorded_and_not_replaced(configured, status):
    namespace = configured
    client = attach_client(namespace)
    execute("mai-upload", namespace)
    execute("mai-submit", namespace)
    job = client.fine_tuning.jobs.data[0]
    job.status = status
    job.error = {"message": "Terminal service failure"}
    with pytest.raises(RuntimeError, match=f"RFT ended in {status}"):
        execute("mai-monitor", namespace)
    saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
    assert saved["resource_context"]["job_status"] == status
    with pytest.raises(RuntimeError, match="cannot be reused"):
        execute("mai-submit", namespace)
    assert client.fine_tuning.jobs.created == 1


@pytest.mark.parametrize("content", [
    "step,accuracy\n1,0.5\n",
    "step,full_valid_mean_reward\n1,\n",
    "step,full_valid_mean_reward\n1,nan\n",
    "step,full_valid_mean_reward\n1,inf\n",
    "step,full_valid_mean_reward\n1,not-a-score\n",
])
def test_metrics_do_not_fabricate_success_for_missing_or_invalid_values(content):
    tree = ast.parse(SOURCES["mai-metrics"])
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    import csv
    import math

    namespace = {"csv": csv, "io": io, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "metric-parser", "exec"), namespace)
    with pytest.raises(ValueError):
        namespace["summarize_metrics"](content)
