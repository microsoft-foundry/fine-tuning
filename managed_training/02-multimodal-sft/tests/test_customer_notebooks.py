from __future__ import annotations

import ast
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def notebook_cells(demo: str) -> list[dict]:
    path = ROOT / demo / "notebooks" / "demo.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def source_containing(demo: str, marker: str) -> str:
    return next(
        "".join(cell["source"])
        for cell in notebook_cells(demo)
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


@pytest.mark.parametrize(
    "demo", ["01-chart-reasoning", "02-image-classification", "03-video-action-recognition"]
)
def test_notebook_is_output_free_and_compiles(demo: str) -> None:
    code = []
    for cell in notebook_cells(demo):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            code.append("".join(cell["source"]))
    compile("\n\n".join(code), demo, "exec")


def test_chart_preserved_source_hashes() -> None:
    demo_root = ROOT / "01-chart-reasoning"
    with (demo_root / "data" / "hash-manifest.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            data = (demo_root / record["preserved_path"]).read_bytes()
            assert len(data) == int(record["bytes"])
            assert len(data.splitlines()) == int(record["rows"])
            assert hashlib.sha256(data).hexdigest() == record["source_sha256"]
            assert record["source_sha256"] == record["preserved_sha256"]


def test_chart_client_cell_creates_clients_without_execution_switch() -> None:
    source = source_containing("01-chart-reasoning", "credential = None")
    credential = object()
    client = object()
    project = SimpleNamespace(get_openai_client=lambda **_kwargs: client)
    namespace = {
        "PROJECT_ENDPOINT": "https://example.services.ai.azure.com/api/projects/example",
        "DefaultAzureCredential": lambda **_kwargs: credential,
        "AIProjectClient": lambda **_kwargs: project,
    }
    exec(compile(source, "chart-client", "exec"), namespace)
    assert namespace["client"] is client
    assert namespace["project_client"] is project


def test_chart_training_submits_and_resumes_without_execution_switch(tmp_path: Path) -> None:
    source = source_containing("01-chart-reasoning", "state = load_state()")
    uploaded = []
    created = []
    options = {}
    job = SimpleNamespace(id="job-created", status="succeeded", to_dict=lambda: {"status": "succeeded"})

    def create_file(**kwargs):
        file_id = f"file-{len(uploaded)}"
        uploaded.append(kwargs)
        return SimpleNamespace(id=file_id)

    client = SimpleNamespace(
        files=SimpleNamespace(
            list=lambda **_kwargs: [],
            create=create_file,
            retrieve=lambda file_id: SimpleNamespace(id=file_id, status="processed"),
        ),
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(
            list=lambda **_kwargs: [],
            create=lambda **kwargs: created.append(kwargs) or job,
            retrieve=lambda _job_id: job,
        )),
    )
    client.with_options = lambda **kwargs: options.update(kwargs) or client
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"messages":[]}\n')
    validation.write_text('{"messages":[]}\n')
    namespace = {
        "STATE_PATH": tmp_path / "run-state.json",
        "RUN_ID": "customer-test",
        "PROJECT_ENDPOINT": "",
        "MODEL": "gpt-4.1-2025-04-14",
        "TRAINING_TYPE": "GlobalStandard",
        "DATASETS": {"train": {"path": train}, "validation": {"path": validation}},
        "dataset_evidence": [],
        "client": client,
        "time": time,
        "TERMINAL_STATES": {"succeeded", "failed", "cancelled"},
        "datetime": datetime,
        "timezone": timezone,
        "json": json,
        "Path": Path,
    }
    code = compile(source, "chart-training", "exec")
    exec(code, namespace)
    exec(code, namespace)
    assert namespace["ft_job"] is job
    assert len(uploaded) == 2
    assert len(created) == 1
    assert options == {"max_retries": 0}


@pytest.mark.parametrize(
    "demo", ["01-chart-reasoning", "02-image-classification", "03-video-action-recognition"]
)
def test_setup_does_not_create_foundry_clients(demo: str) -> None:
    source = "".join(notebook_cells(demo)[2]["source"])
    calls = {
        node.func.id for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not calls & {"DefaultAzureCredential", "AIProjectClient"}


def test_image_original_split_and_raw_jpeg_encoding() -> None:
    source = source_containing("02-image-classification", "def clean_breed")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "clean_breed")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "clean-breed", "exec"), namespace)
    assert namespace["clean_breed"]("n02085620-Chihuahua") == "Chihuahua"
    assert namespace["clean_breed"]("n02088094-Afghan_hound") == "Afghan Hound"
    assert 'head(50)' in source
    assert '"train" if value < 40 else "validation" if value < 45 else "unused"' in source
    serialization = source_containing("02-image-classification", "def write_jsonl")
    assert "Path(row.image_path).read_bytes()" in serialization
    assert '"detail": "low"' in serialization
    assert "thumbnail" not in serialization


def test_image_split_counts_match_both_guides() -> None:
    demo_root = ROOT / "02-image-classification"
    primary = (demo_root / "README.md").read_text(encoding="utf-8")
    provenance = (demo_root / "data" / "README.md").read_text(encoding="utf-8")
    for text in (primary, provenance):
        assert "4,800" in text
        assert "600" in text
        assert "40/5/5" in primary
        assert "positions 0-39" in provenance


def test_image_client_cell_runs_for_configured_project() -> None:
    source = source_containing("02-image-classification", "client = project_client.get_openai_client(")
    credential = object()
    client = object()
    options = {}
    project = SimpleNamespace(
        get_openai_client=lambda **kwargs: options.update(kwargs) or client,
    )
    namespace = {
        "PROJECT_ENDPOINT": "https://example.services.ai.azure.com/api/projects/example",
        "DefaultAzureCredential": lambda: credential,
        "AIProjectClient": lambda **_kwargs: project,
    }
    exec(compile(source, "image-client", "exec"), namespace)
    assert namespace["client"] is client
    assert namespace["credential"] is credential
    assert options == {"max_retries": 0}


def test_image_missing_project_fails_before_authentication() -> None:
    source = source_containing("02-image-classification", "client = project_client.get_openai_client(")
    calls = []
    namespace = {
        "PROJECT_ENDPOINT": "",
        "DefaultAzureCredential": lambda: calls.append("credential"),
    }
    with pytest.raises(RuntimeError, match="Set FOUNDRY_PROJECT_ENDPOINT"):
        exec(compile(source, "image-client", "exec"), namespace)
    assert calls == []


@pytest.mark.parametrize("demo", ["02-image-classification", "03-video-action-recognition"])
def test_customer_content_does_not_include_internal_run_history(demo: str) -> None:
    demo_root = ROOT / demo
    documents = [
        path for path in demo_root.rglob("*.md") if "outputs" not in path.parts
    ] + [demo_root / ".env.template", demo_root / "demo.yaml"]
    text = "\n".join(path.read_text().casefold() for path in documents)
    text += "\n" + "\n".join(
        "".join(cell["source"]).casefold() for cell in notebook_cells(demo)
    )
    for phrase in (
        "corrective", "known source-data blocker", "training is blocked",
        "recent failure", "test runs", "deviation_reason", "captcha",
    ):
        assert phrase not in text


def test_video_metrics_skip_when_no_job_is_available() -> None:
    source = source_containing("03-video-action-recognition", 'metrics_text =')
    exec(
        compile(source, "video-metrics", "exec"),
        {"job": None, "client": None, "project_client": None, "credential": None},
    )


@pytest.mark.parametrize(
    "demo", ["01-chart-reasoning", "02-image-classification", "03-video-action-recognition"]
)
def test_environment_template_is_blank_and_safe(demo: str) -> None:
    template = (ROOT / demo / ".env.template").read_text()
    values = dict(
        line.split("=", 1)
        for line in template.splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert "FOUNDRY_PROJECT_ENDPOINT" in values
    assert all(value == "" for value in values.values())


@pytest.mark.parametrize(
    "demo", ["01-chart-reasoning", "02-image-classification", "03-video-action-recognition"]
)
def test_assets_and_exact_variability_sentence(demo: str) -> None:
    sentence = (
        "Actual results may vary by model version, data, configuration, "
        "region availability, and service conditions."
    )
    assert (ROOT / demo / "assets" / "README.md").is_file()
    assert sentence in (ROOT / demo / "README.md").read_text()
    assert sentence in "\n".join("".join(cell["source"]) for cell in notebook_cells(demo))


@pytest.mark.parametrize(
    "demo", ["01-chart-reasoning", "02-image-classification", "03-video-action-recognition"]
)
def test_job_creation_disables_sdk_retries(demo: str) -> None:
    source = source_containing(demo, ".fine_tuning.jobs.create(")
    assert "client.with_options(max_retries=0).fine_tuning.jobs.create(" in source


@pytest.mark.parametrize("case", ["pending-unknown", "wrong-recipe", "second-page"])
def test_chart_reconciles_pending_submission_without_duplicate_post(
    tmp_path: Path, case: str,
) -> None:
    source = source_containing("01-chart-reasoning", "state = load_state()")
    candidate = SimpleNamespace(
        id="job-test", suffix="customer-test", model="model",
        training_file="train-test", validation_file="validation-test",
        seed=42, status="pending",
        method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 2 if case == "wrong-recipe" else 1}}},
    )

    class Page:
        data = []

        def __iter__(self):
            return iter([] if case == "pending-unknown" else [candidate])

    class Reconciled(Exception):
        pass

    def retrieve(_job_id: str) -> object:
        raise Reconciled("matched pending job was reused")

    client = SimpleNamespace(
        files=SimpleNamespace(retrieve=lambda file_id: SimpleNamespace(id=file_id, status="processed")),
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(list=lambda **kwargs: Page(), retrieve=retrieve)),
    )
    state = {
        "run_id": "customer-test", "project_endpoint": "",
        "model": "model", "training_type": "GlobalStandard", "datasets": [],
        "submission_pending": True, "training_file_id": "train-test",
        "validation_file_id": "validation-test",
    }
    state_path = tmp_path / "run-state.json"
    state_path.write_text(json.dumps(state))
    namespace = {
        "STATE_PATH": state_path, "RUN_ID": "customer-test", "PROJECT_ENDPOINT": "",
        "MODEL": "model", "TRAINING_TYPE": "GlobalStandard", "dataset_evidence": [],
        "datetime": datetime, "timezone": timezone,
        "json": json, "Path": Path,
        "time": SimpleNamespace(monotonic=time.monotonic, sleep=lambda _delay: None),
        "client": client,
        "DATASETS": {"train": {"path": tmp_path / "train"}, "validation": {"path": tmp_path / "validation"}},
        "TERMINAL_STATES": {"succeeded", "failed", "cancelled"},
    }
    expected = Reconciled if case == "second-page" else RuntimeError
    match = {
        "pending-unknown": "Previous submission outcome is unknown",
        "wrong-recipe": "different seed or epoch",
        "second-page": "pending job was reused",
    }[case]
    with pytest.raises(expected, match=match):
        exec(compile(source, "chart-reconciliation", "exec"), namespace)


@pytest.mark.parametrize("demo", ["02-image-classification", "03-video-action-recognition"])
def test_ambiguous_creation_persists_state_and_blocks_resubmission(
    tmp_path: Path, demo: str,
) -> None:
    marker = "def status_value" if demo.startswith("02") else "def retry_service_call"
    source = source_containing(demo, marker)
    attempts = 0
    options = {}

    class AmbiguousFailure(RuntimeError):
        pass

    def create_job(**kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        pending = json.loads((tmp_path / "run-state.json").read_text())
        assert pending["submission_pending"] is True
        assert "job_id" not in pending
        raise AmbiguousFailure("submission outcome unknown")

    client = SimpleNamespace(
        files=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(id="file-test"),
            retrieve=lambda file_id: SimpleNamespace(id=file_id, status="processed"),
        ),
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=create_job)),
    )

    def with_options(**kwargs: object) -> object:
        options.update(kwargs)
        return client

    client.with_options = with_options
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"messages":[]}\n')
    validation.write_text('{"messages":[]}\n')
    artifacts = {
        "training": {"path": str(train), "sha256": "training-hash"},
        "validation": {"path": str(validation), "sha256": "validation-hash"},
    }
    namespace = {
        "PROJECT_ENDPOINT": "", "MODEL": "model",
        "TRAINING_TYPE": "GlobalStandard", "SEED": 42, "EPOCHS": 2,
        "LEARNING_RATE_MULTIPLIER": 0.5, "OUTPUTS": tmp_path,
        "RUN_STATE_PATH": tmp_path / "run-state.json", "artifacts": artifacts,
        "input_summary": artifacts, "train_path": train, "validation_path": validation,
        "hashlib": hashlib, "json": json, "Path": Path, "Any": object,
        "time": time, "client": client,
    }
    code = compile(source, "ambiguous-submission", "exec")
    with pytest.raises(AmbiguousFailure, match="outcome unknown"):
        exec(code, namespace)
    with pytest.raises(RuntimeError, match="Previous submission outcome is unknown"):
        exec(code, namespace)
    assert attempts == 1
    assert options == {"max_retries": 0}
