from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from shared.dataset_validation import DatasetIssue, DatasetValidationError, validate_jsonl


ROOT = Path(__file__).parents[2]
DEMO = ROOT / "03-core-rft" / "03-rft-code-repair"
NOTEBOOK = json.loads((DEMO / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))
SOURCES = {
    cell["id"]: "".join(cell["source"])
    for cell in NOTEBOOK["cells"]
    if cell["cell_type"] == "code"
}
OFFLINE_CELLS = ("mai-prepare", "mai-data", "mai-grader")
GRADER_FIELDS = ("reference_patch", "test_expectations", "requirements")
EXPECTED_INPUTS = {
    "training": (50, 47678, "00398c62ebc0bf7ab9b4d4abe6231452a6ca387af56ae6bc9558c483bf7ab28c"),
    "validation": (15, 15177, "2944c6a1e73f3a32a98ea8dc957d4dcdc06f3fdf32c5b219a9592f9a46c437bc"),
}
GRADER_HASH = "57ec9f07ff12f001f9cdbf8a1aaa9a2e9d5e2e080d9dea1a000e0ae1fc7a64d7"
METRICS_CSV = b"step,train_mean_reward,full_valid_mean_reward\n1,0.3,0.4\n2,0.6,0.5\n"
COMPLETION_TEXT = json.dumps({
    "patch": "--- a/helpers.py\n+++ b/helpers.py\n@@ -1 +1 @@\n-return items\n+return items[:count]\n",
    "explanation": "Return the requested prefix.",
    "tests": ["A zero count returns an empty list."],
})
COMPLETION = {
    "id": "completion-offline", "object": "chat.completion", "created": 1,
    "model": "offline-deployment",
    "choices": [{
        "index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": COMPLETION_TEXT},
    }],
}
MANAGEMENT_DEPLOYMENT = {
    "name": "offline-deployment",
    "properties": {
        "provisioningState": "Succeeded",
        "model": {"format": "Microsoft", "name": "artifact-offline-result", "version": "1"},
    },
    "sku": {"name": "GlobalStandard", "capacity": 10},
}


def execute(cell_id, namespace):
    exec(compile(SOURCES[cell_id], f"code-repair:{cell_id}", "exec"), namespace)


def prepare(monkeypatch, demo, *, notebooks_cwd=False):
    monkeypatch.chdir(demo / "notebooks" if notebooks_cwd else demo)
    namespace = {}
    for cell_id in OFFLINE_CELLS:
        execute(cell_id, namespace)
    return namespace


def validate_rft_row(row, line_number):
    if set(row) != {"messages", *GRADER_FIELDS}:
        yield DatasetIssue("error", "rft.keys", "Unexpected code-repair row fields", line_number)
    for field in GRADER_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            yield DatasetIssue("error", "rft.reference", f"{field} must be non-empty text", line_number)
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        yield DatasetIssue("error", "rft.messages", "Expected developer and user messages", line_number)
        return
    for message, role in zip(messages, ("developer", "user")):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            yield DatasetIssue("error", "rft.message", f"Invalid {role} message", line_number)


@pytest.fixture
def demo_copy(tmp_path):
    target = tmp_path / "code-repair"
    shutil.copytree(DEMO, target, ignore=shutil.ignore_patterns("outputs", ".env", "__pycache__"))
    return target


def patch_project_sdk(monkeypatch, client):
    credential = Mock()
    project = Mock()
    project.get_openai_client.return_value = client
    project_factory = Mock(return_value=project)
    monkeypatch.setitem(
        sys.modules, "azure.ai.projects", SimpleNamespace(AIProjectClient=project_factory)
    )
    monkeypatch.setitem(
        sys.modules, "azure.identity",
        SimpleNamespace(DefaultAzureCredential=Mock(return_value=credential)),
    )
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://example.test/api/projects/offline")
    monkeypatch.setenv("FOUNDRY_CODE_REPAIR_DEPLOYMENT", "offline-deployment")
    return project_factory, project, credential


def configure_serving(monkeypatch):
    for name, value in {
        "FOUNDRY_SUBSCRIPTION_ID": "offline-subscription",
        "FOUNDRY_RESOURCE_GROUP": "offline-group",
        "FOUNDRY_ACCOUNT_NAME": "offline-account",
        "FOUNDRY_CODE_REPAIR_CAPACITY": "10",
    }.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def management(monkeypatch):
    sdk = pytest.importorskip("azure.mgmt.cognitiveservices")
    from azure.mgmt.cognitiveservices.models import Deployment

    configure_serving(monkeypatch)
    client = MagicMock()
    client.__enter__.return_value = client
    client.projects.get.return_value = SimpleNamespace(properties=SimpleNamespace(
        endpoints={"AI Foundry API": "https://example.test/api/projects/offline"}
    ))
    client.deployments.list.return_value = []
    client.deployments.begin_create_or_update.return_value.result.return_value = (
        Deployment.deserialize(MANAGEMENT_DEPLOYMENT)
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(sdk, "CognitiveServicesManagementClient", factory)
    return factory, client


def completed_job(status="succeeded"):
    receipt = {
        "id": "job-offline-recorded", "status": status,
        "fine_tuned_model": "artifact-offline-result",
        "result_files": ["file-offline-metrics"],
        "error": None if status == "succeeded" else {"message": "Terminal service failure"},
    }
    return SimpleNamespace(
        **receipt, model_dump_json=lambda **kwargs: json.dumps(receipt, **kwargs)
    )


@pytest.fixture
def connected(monkeypatch, demo_copy):
    namespace = prepare(monkeypatch, demo_copy)
    client = Mock()
    project_factory, project, credential = patch_project_sdk(monkeypatch, client)
    execute("mai-connect", namespace)
    namespace["time"] = SimpleNamespace(monotonic=Mock(return_value=0), sleep=Mock())
    return namespace, client, project_factory, project, credential


@pytest.mark.parametrize("notebooks_cwd", [False, True])
def test_offline_preparation_uses_exact_inputs_without_cloud_or_outputs(
    monkeypatch, demo_copy, notebooks_cwd
):
    client = Mock()
    project_factory, _, _ = patch_project_sdk(monkeypatch, client)
    namespace = prepare(monkeypatch, demo_copy, notebooks_cwd=notebooks_cwd)
    project_factory.assert_not_called()
    assert namespace["MODEL"] == "MAI-Code-1.1-Flash-2026-09-15"
    assert namespace["DEMO_ROOT"] == demo_copy
    assert not namespace["OUTPUT_DIR"].exists()
    assert "client" not in namespace
    assert len(namespace["training_rows"]) == 50
    assert len(namespace["validation_rows"]) == 15


def test_notebook_teaches_sdk_calls_without_workflow_helpers():
    tree = ast.parse("\n".join(SOURCES.values()))
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assert)) for node in ast.walk(tree))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "azure.ai.projects" in imports
    assert "azure.identity" in imports
    assert not any(name and name.startswith("shared") for name in imports)


def test_notebook_tables_use_explicit_left_alignment():
    tables = 0
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            if not line.startswith("|"):
                continue
            columns = [column.strip() for column in line.strip("|").split("|")]
            if columns and all(re.fullmatch(r":?-{3,}:?", column) for column in columns):
                tables += 1
                assert all(column.startswith(":") and not column.endswith(":") for column in columns)
    assert tables > 0
    dataset = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == "mai-data-guide")
    assert "| :--- | :--- |" in "".join(dataset["source"])


def test_metadata_and_navigation_include_the_new_model():
    catalog = yaml.safe_load((ROOT / "catalog.yml").read_text(encoding="utf-8"))
    demo = next(item for item in catalog["demos"] if item["id"] == "rft-mai-code-repair")
    assert demo["training"]["base_models"] == ["MAI-Code-1.1-Flash-2026-09-15"]
    assert demo["training"]["evaluator_models"] == ["o3-mini-2025-01-31"]
    assert demo["techniques"] == ["rft"]
    assert demo["status"] == "preview"
    assert demo["validation"]["status"] == "partial"
    assert {"deploy", "inference"} <= set(demo["notebooks"][0]["sections"])
    assert {"deployment", "inference"} <= set(demo["capabilities"])
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


def test_dataset_manifest_contains_only_public_integrity_metadata():
    manifest = json.loads((DEMO / "data" / "hashes.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"algorithm", "normalization", "artifacts"}
    assert manifest["algorithm"] == "sha256"
    assert manifest["normalization"] == "none"
    for artifact in manifest["artifacts"]:
        assert set(artifact) == {"path", "bytes", "rows", "sha256"}


def test_sample_integrity_schema_and_split_isolation():
    manifest = json.loads((DEMO / "data" / "hashes.json").read_text(encoding="utf-8"))
    prompt_sets = {}
    for label, artifact in zip(("training", "validation"), manifest["artifacts"], strict=True):
        path = DEMO / artifact["path"]
        report = validate_jsonl(
            path, required_keys=("messages", *GRADER_FIELDS),
            validate_messages=False, row_validator=validate_rft_row,
        ).require_valid()
        assert (report.rows, report.bytes, report.sha256) == EXPECTED_INPUTS[label]
        assert (artifact["rows"], artifact["bytes"], artifact["sha256"]) == EXPECTED_INPUTS[label]
        assert report.duplicate_rows == 0
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        prompt_sets[label] = {row["messages"][-1]["content"] for row in rows}
        assert len(prompt_sets[label]) == len(rows)
    assert prompt_sets["training"].isdisjoint(prompt_sets["validation"])


def test_grader_definition_and_dataset_bindings():
    path = DEMO / "graders" / "code_repair_quality.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == GRADER_HASH
    grader = json.loads(path.read_text(encoding="utf-8"))
    assert grader["type"] == "score_model"
    assert grader["model"] == "o3-mini-2025-01-31"
    assert grader["range"] == [0, 1]
    assert grader["pass_threshold"] == 0.7
    assert grader["sampling_params"] == {"seed": 42}
    bindings = {
        match
        for message in grader["input"]
        for match in re.findall(r"\{\{(item|sample)\.([A-Za-z_][A-Za-z_0-9]*)\}\}", message["content"])
    }
    assert bindings == {*(("item", field) for field in GRADER_FIELDS), ("sample", "output_text")}


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
def test_rft_data_checks_reject_invalid_rows(tmp_path, change):
    path = DEMO / "data" / "preserved" / "code-repair-rft-training.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    change(row)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        validate_jsonl(
            invalid, required_keys=("messages", *GRADER_FIELDS),
            validate_messages=False, row_validator=validate_rft_row,
        ).require_valid()


def test_direct_sdk_workflow_saves_ids_and_displays_service_metrics(
    connected, management, monkeypatch, capsys
):
    namespace, client, project_factory, project, credential = connected
    output = namespace["OUTPUT_DIR"]
    uploaded_bytes = []

    def upload(*, file, purpose):
        assert purpose == "fine-tune"
        uploaded_bytes.append(file.read())
        return SimpleNamespace(id=f"file-offline-{len(uploaded_bytes)}")

    def wait(file_id, **kwargs):
        label = "training" if file_id.endswith("1") else "validation"
        assert (output / f"{label}-file-id.txt").read_text(encoding="utf-8") == file_id
        assert kwargs == {"max_wait_seconds": 600}

    client.files.create.side_effect = upload
    client.files.wait_for_processing.side_effect = wait
    client.fine_tuning.jobs.create.return_value = SimpleNamespace(
        id="job-offline-recorded", status="queued"
    )
    client.fine_tuning.jobs.retrieve.side_effect = [
        SimpleNamespace(status="running"), completed_job(),
    ]
    client.files.content.return_value.write_to_file.side_effect = (
        lambda path: path.write_bytes(METRICS_CSV)
    )
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=COMPLETION_TEXT))],
        model_dump_json=lambda **kwargs: json.dumps(COMPLETION, **kwargs),
    )
    tables = []
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(
        read_csv=lambda path: list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    ))
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=tables.append))

    for cell_id in (
        "mai-upload", "mai-submit", "mai-monitor", "mai-metrics",
        "mai-deploy", "mai-infer", "mai-close",
    ):
        execute(cell_id, namespace)

    assert uploaded_bytes == [namespace["TRAIN_PATH"].read_bytes(), namespace["VALIDATION_PATH"].read_bytes()]
    project_factory.assert_called_once_with(
        endpoint="https://example.test/api/projects/offline", credential=credential
    )
    project.get_openai_client.assert_called_once_with(max_retries=0)
    client.fine_tuning.jobs.create.assert_called_once_with(
        model=namespace["MODEL"], training_file="file-offline-1", validation_file="file-offline-2",
        method={"type": "reinforcement", "reinforcement": {"grader": namespace["GRADER"]}},
        extra_body={"trainingType": "globalStandard"}, suffix="mai-code-repair",
    )
    assert (output / "job-id.txt").read_text(encoding="utf-8") == "job-offline-recorded"
    assert (output / "project-endpoint.txt").read_text(encoding="utf-8") == (
        "https://example.test/api/projects/offline"
    )
    receipt = json.loads((output / "completed-job.json").read_text(encoding="utf-8"))
    assert receipt["fine_tuned_model"] == "artifact-offline-result"
    assert (output / "training-metrics-1.csv").read_bytes() == METRICS_CSV
    assert tables[0][-1]["full_valid_mean_reward"] == "0.5"
    management_factory, management_client = management
    management_factory.assert_called_once_with(
        credential, "offline-subscription"
    )
    management_client.projects.get.assert_called_once_with("offline-group", "offline-account", "offline")
    create = management_client.deployments.begin_create_or_update
    assert create.call_args.args[:3] == ("offline-group", "offline-account", "offline-deployment")
    assert create.call_args.args[3].serialize() == {
        "sku": {"name": "GlobalStandard", "capacity": 10},
        "properties": {"model": {"format": "Microsoft", "name": "artifact-offline-result", "version": "1"}},
    }
    create.return_value.result.assert_called_once_with()
    management_client.__exit__.assert_called_once_with(None, None, None)
    project.deployments.get.assert_not_called()
    assert (output / "deployment-name.txt").read_text(encoding="utf-8") == "offline-deployment"
    deployment_receipt = json.loads((output / "completed-deployment.json").read_text(encoding="utf-8"))
    assert deployment_receipt["properties"]["provisioning_state"] == "Succeeded"
    assert deployment_receipt["properties"]["model"]["name"] == receipt["fine_tuned_model"]
    client.chat.completions.create.assert_called_once_with(
        model="offline-deployment", messages=namespace["inference_messages"], max_completion_tokens=2048
    )
    assert [message["role"] for message in namespace["inference_messages"]] == ["developer", "user"]
    assert set(namespace["inference_messages"][1]) == {"role", "content"}
    assert "take_prefix" in namespace["inference_messages"][1]["content"]
    assert json.loads((output / "inference-response.json").read_text(encoding="utf-8")) == COMPLETION
    namespace["time"].sleep.assert_called_once_with(60)
    client.close.assert_called_once()
    project.close.assert_called_once()
    credential.close.assert_called_once()
    text = capsys.readouterr().out
    assert "https://example.test" not in text
    assert "job-offline-recorded" not in text
    assert "artifact-offline-result" not in text


def test_deployment_mismatch_stops_before_inference(connected, management):
    namespace, client, *_ = connected
    namespace["job"] = completed_job()
    result = management[1].deployments.begin_create_or_update.return_value.result.return_value
    result.properties.model.name = namespace["MODEL"]
    with pytest.raises(RuntimeError, match="this job's fine-tuned model"):
        execute("mai-deploy", namespace)
    assert (namespace["OUTPUT_DIR"] / "deployment-name.txt").exists()
    assert not (namespace["OUTPUT_DIR"] / "completed-deployment.json").exists()
    client.chat.completions.create.assert_not_called()


def test_deployment_failure_preserves_name_and_stops_before_inference(connected, management):
    from azure.core.exceptions import HttpResponseError

    namespace, client, *_ = connected
    namespace["job"] = completed_job()
    poller = management[1].deployments.begin_create_or_update.return_value

    def fail():
        assert (namespace["OUTPUT_DIR"] / "deployment-name.txt").read_text(encoding="utf-8") == (
            "offline-deployment"
        )
        raise HttpResponseError("Deployment provisioning failed")

    poller.result.side_effect = fail
    with pytest.raises(HttpResponseError, match="Deployment provisioning failed"):
        execute("mai-deploy", namespace)
    assert not (namespace["OUTPUT_DIR"] / "completed-deployment.json").exists()
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("setting", [
    "FOUNDRY_SUBSCRIPTION_ID", "FOUNDRY_RESOURCE_GROUP", "FOUNDRY_ACCOUNT_NAME",
    "FOUNDRY_CODE_REPAIR_DEPLOYMENT", "FOUNDRY_CODE_REPAIR_CAPACITY",
])
def test_serving_configuration_is_only_required_for_optional_sections(
    connected, management, monkeypatch, setting
):
    namespace, *_ = connected
    monkeypatch.delenv(setting)
    with pytest.raises(KeyError, match=setting):
        execute("mai-deploy", namespace)
    management[0].assert_not_called()


def test_deployment_name_can_be_added_to_env_after_training(connected, management, monkeypatch):
    namespace, *_ = connected
    namespace["job"] = completed_job()
    monkeypatch.setenv("FOUNDRY_CODE_REPAIR_DEPLOYMENT", "")
    (namespace["DEMO_ROOT"] / ".env").write_text(
        "FOUNDRY_CODE_REPAIR_DEPLOYMENT=offline-new-deployment\n", encoding="utf-8"
    )
    execute("mai-deploy", namespace)
    assert management[1].deployments.begin_create_or_update.call_args.args[2] == (
        "offline-new-deployment"
    )


def test_deployment_checks_target_project_before_creating(connected, management):
    namespace, *_ = connected
    namespace["job"] = completed_job()
    management[1].projects.get.return_value.properties.endpoints = {
        "AI Foundry API": "https://other.example.test/api/projects/offline"
    }
    with pytest.raises(ValueError, match="parent resource"):
        execute("mai-deploy", namespace)
    management[1].deployments.begin_create_or_update.assert_not_called()


def test_deployment_keeps_the_training_client_project_after_env_reload(connected, management):
    namespace, *_ = connected
    namespace["job"] = completed_job()
    (namespace["DEMO_ROOT"] / ".env").write_text(
        "FOUNDRY_PROJECT_ENDPOINT=https://other.example.test/api/projects/other\n", encoding="utf-8"
    )
    execute("mai-deploy", namespace)
    assert namespace["PROJECT_ENDPOINT"] == "https://example.test/api/projects/offline"
    management[1].projects.get.assert_called_once_with("offline-group", "offline-account", "offline")


@pytest.mark.parametrize("existing_name", ["offline-deployment", "OFFLINE-DEPLOYMENT"])
def test_deployment_does_not_replace_an_existing_deployment(connected, management, existing_name):
    namespace, *_ = connected
    namespace["job"] = completed_job()
    management[1].deployments.list.return_value = [SimpleNamespace(name=existing_name)]
    with pytest.raises(ValueError, match="does not update existing deployments"):
        execute("mai-deploy", namespace)
    management[1].deployments.begin_create_or_update.assert_not_called()
    assert not (namespace["OUTPUT_DIR"] / "deployment-name.txt").exists()


@pytest.mark.parametrize("capacity", ["0", "-1"])
def test_deployment_rejects_nonpositive_capacity(connected, management, monkeypatch, capacity):
    namespace, *_ = connected
    monkeypatch.setenv("FOUNDRY_CODE_REPAIR_CAPACITY", capacity)
    with pytest.raises(ValueError, match="positive deployment capacity"):
        execute("mai-deploy", namespace)
    management[0].assert_not_called()


@pytest.mark.parametrize("status,artifact", [("running", None), ("succeeded", None)])
def test_deployment_requires_a_completed_training_artifact(connected, management, status, artifact):
    namespace, *_ = connected
    namespace["job"] = SimpleNamespace(status=status, fine_tuned_model=artifact)
    with pytest.raises(RuntimeError, match="produce a model artifact"):
        execute("mai-deploy", namespace)
    management[0].assert_not_called()


def test_monitor_recovers_saved_job_without_upload_or_submission(connected):
    namespace, client, *_ = connected
    (namespace["OUTPUT_DIR"] / "job-id.txt").write_text("job-offline-recorded", encoding="utf-8")
    client.fine_tuning.jobs.retrieve.return_value = completed_job()
    execute("mai-monitor", namespace)
    client.fine_tuning.jobs.retrieve.assert_called_once_with("job-offline-recorded")
    client.files.create.assert_not_called()
    client.fine_tuning.jobs.create.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_monitor_surfaces_terminal_errors(connected, status):
    namespace, client, *_ = connected
    (namespace["OUTPUT_DIR"] / "job-id.txt").write_text("job-offline-recorded", encoding="utf-8")
    client.fine_tuning.jobs.retrieve.return_value = completed_job(status)
    with pytest.raises(RuntimeError, match=f"Training {status}"):
        execute("mai-monitor", namespace)
    assert not (namespace["OUTPUT_DIR"] / "completed-job.json").exists()
    client.fine_tuning.jobs.create.assert_not_called()


def test_monitor_timeout_does_not_cancel_or_resubmit(connected):
    namespace, client, *_ = connected
    (namespace["OUTPUT_DIR"] / "job-id.txt").write_text("job-offline-recorded", encoding="utf-8")
    client.fine_tuning.jobs.retrieve.return_value = SimpleNamespace(status="running")
    namespace["time"].monotonic.side_effect = [0, 21_601]
    with pytest.raises(TimeoutError, match="remote job may still be running"):
        execute("mai-monitor", namespace)
    client.fine_tuning.jobs.cancel.assert_not_called()
    client.fine_tuning.jobs.create.assert_not_called()


def test_file_processing_failure_stops_before_submission(connected):
    namespace, client, *_ = connected
    client.files.create.side_effect = [
        SimpleNamespace(id="file-offline-1"), SimpleNamespace(id="file-offline-2")
    ]
    client.files.wait_for_processing.side_effect = RuntimeError("File processing failed")
    with pytest.raises(RuntimeError, match="File processing failed"):
        execute("mai-upload", namespace)
    assert (namespace["OUTPUT_DIR"] / "training-file-id.txt").exists()
    assert (namespace["OUTPUT_DIR"] / "validation-file-id.txt").exists()
    client.fine_tuning.jobs.create.assert_not_called()


@pytest.mark.parametrize("result_files", [None, []])
def test_missing_metrics_are_not_reported_as_success(connected, monkeypatch, result_files):
    namespace, client, *_ = connected
    namespace["job"] = SimpleNamespace(result_files=result_files)
    display = Mock()
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=display))
    with pytest.raises(RuntimeError, match="no training metrics"):
        execute("mai-metrics", namespace)
    client.files.content.assert_not_called()
    display.assert_not_called()


def test_notebook_calls_serialize_with_installed_sdk(monkeypatch, demo_copy, management):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("pandas")
    pytest.importorskip("IPython.display")
    namespace = prepare(monkeypatch, demo_copy)
    file_ids = []
    submitted = []

    def handle(request):
        path = request.url.path
        if request.method == "POST" and path == "/files":
            label = "training" if not file_ids else "validation"
            raw = namespace["TRAIN_PATH" if label == "training" else "VALIDATION_PATH"].read_bytes()
            assert raw in request.read()
            file_ids.append(f"file-offline-{label}")
            return httpx.Response(200, json={
                "id": file_ids[-1], "object": "file", "created_at": 1,
                "filename": f"{label}.jsonl", "purpose": "fine-tune",
                "status": "uploaded", "bytes": len(raw),
            })
        if request.method == "POST" and path == "/fine_tuning/jobs":
            payload = json.loads(request.content)
            submitted.append(payload)
            assert payload == {
                "model": namespace["MODEL"], "training_file": file_ids[0],
                "validation_file": file_ids[1], "suffix": "mai-code-repair",
                "method": {"type": "reinforcement", "reinforcement": {"grader": namespace["GRADER"]}},
                "trainingType": "globalStandard",
            }
            return httpx.Response(200, json={"id": "job-offline-recorded", "status": "queued"})
        if request.method == "GET" and path == "/fine_tuning/jobs/job-offline-recorded":
            receipt = json.loads(completed_job().model_dump_json())
            return httpx.Response(200, json=receipt)
        if request.method == "GET" and path == "/files/file-offline-metrics/content":
            return httpx.Response(200, content=METRICS_CSV, headers={"content-type": "text/csv"})
        if request.method == "GET" and path in {f"/files/{file_id}" for file_id in file_ids}:
            return httpx.Response(200, json={"id": path.rsplit("/", 1)[1], "status": "processed"})
        if request.method == "POST" and path == "/chat/completions":
            assert json.loads(request.content) == {
                "model": "offline-deployment", "messages": namespace["inference_messages"],
                "max_completion_tokens": 2048,
            }
            return httpx.Response(200, json=COMPLETION)
        raise AssertionError(f"Unexpected SDK request: {request.method} {path}")

    with openai.OpenAI(
        api_key="offline-test", base_url="https://example.test",
        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        patch_project_sdk(monkeypatch, client)
        for cell_id in (
            "mai-connect", "mai-upload", "mai-submit", "mai-monitor", "mai-metrics",
            "mai-deploy", "mai-infer", "mai-close",
        ):
            execute(cell_id, namespace)
    assert len(submitted) == 1
    assert (demo_copy / "outputs" / "training-metrics-1.csv").read_bytes() == METRICS_CSV
    saved_response = json.loads((demo_copy / "outputs" / "inference-response.json").read_text(encoding="utf-8"))
    assert saved_response["choices"][0]["message"]["content"] == COMPLETION_TEXT


def test_deployment_serializes_and_polls_with_installed_management_sdk(monkeypatch, demo_copy):
    sdk = pytest.importorskip("azure.mgmt.cognitiveservices")
    from time import time
    from urllib.parse import urlsplit

    from azure.core.credentials import AccessTokenInfo
    from azure.core.pipeline.transport import HttpResponse, HttpTransport
    from azure.core.utils import case_insensitive_dict

    namespace = prepare(monkeypatch, demo_copy)
    _, _, credential = patch_project_sdk(monkeypatch, Mock())
    credential.get_token_info.return_value = AccessTokenInfo("offline-token", int(time()) + 3600)
    configure_serving(monkeypatch)
    execute("mai-connect", namespace)
    namespace["job"] = completed_job()
    account_path = (
        "/subscriptions/offline-subscription/resourceGroups/offline-group"
        "/providers/Microsoft.CognitiveServices/accounts/offline-account"
    )
    calls = []

    class Response(HttpResponse):
        def __init__(self, request, status, body, headers=None):
            super().__init__(request, None)
            self.status_code = status
            self.headers = case_insensitive_dict({"Content-Type": "application/json", **(headers or {})})
            self.content_type = "application/json"
            self._body = json.dumps(body).encode()

        def body(self):
            return self._body

        def read(self):
            return self._body

        def close(self):
            pass

        def stream_download(self, pipeline, **kwargs):
            return iter((self._body,))

    class Transport(HttpTransport):
        def open(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def send(self, request, **kwargs):
            path = urlsplit(request.url).path
            calls.append((request.method, path))
            if request.method == "GET" and path == account_path + "/projects/offline":
                return Response(request, 200, {"properties": {"endpoints": {
                    "AI Foundry API": "https://example.test/api/projects/offline"
                }}})
            if request.method == "GET" and path == account_path + "/deployments":
                return Response(request, 200, {"value": []})
            if request.method == "PUT" and path == account_path + "/deployments/offline-deployment":
                assert json.loads(request.body) == {
                    "sku": {"name": "GlobalStandard", "capacity": 10},
                    "properties": {"model": {
                        "format": "Microsoft", "name": "artifact-offline-result", "version": "1"
                    }},
                }
                assert (namespace["OUTPUT_DIR"] / "deployment-name.txt").exists()
                creating = {**MANAGEMENT_DEPLOYMENT, "properties": {
                    **MANAGEMENT_DEPLOYMENT["properties"], "provisioningState": "Creating"
                }}
                return Response(request, 201, creating, {
                    "Azure-AsyncOperation": "https://management.azure.com/operations/offline-deployment",
                    "Retry-After": "0",
                })
            if request.method == "GET" and path == "/operations/offline-deployment":
                return Response(request, 200, {"status": "Succeeded"})
            if request.method == "GET" and path == account_path + "/deployments/offline-deployment":
                return Response(request, 200, MANAGEMENT_DEPLOYMENT)
            raise AssertionError(f"Unexpected management SDK request: {request.method} {path}")

    native_factory = sdk.CognitiveServicesManagementClient
    factory = Mock(side_effect=lambda credential, subscription_id: native_factory(
        credential, subscription_id, transport=Transport(), polling_interval=0, retry_total=0
    ))
    monkeypatch.setattr(sdk, "CognitiveServicesManagementClient", factory)
    execute("mai-deploy", namespace)
    execute("mai-close", namespace)
    assert ("GET", "/operations/offline-deployment") in calls
    assert calls[-1] == ("GET", account_path + "/deployments/offline-deployment")
    receipt = json.loads((namespace["OUTPUT_DIR"] / "completed-deployment.json").read_text(encoding="utf-8"))
    assert receipt["properties"]["provisioning_state"] == "Succeeded"
    assert receipt["properties"]["model"]["name"] == "artifact-offline-result"
