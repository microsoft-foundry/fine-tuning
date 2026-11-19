from __future__ import annotations

import ast
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def notebook_cells(demo: str) -> list[dict]:
    path = ROOT / demo / "notebooks" / "demo.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def cell_source(cell: dict) -> str:
    source = cell["source"]
    if isinstance(source, str):
        return source
    if any(line.endswith(("\n", "\r")) for line in source):
        return "".join(source)
    return "\n".join(source)


def source_containing(demo: str, marker: str) -> str:
    return next(
        cell_source(cell)
        for cell in notebook_cells(demo)
        if cell["cell_type"] == "code" and marker in cell_source(cell)
    )


@pytest.mark.parametrize(
    "demo", ["01-chart-reasoning", "02-image-classification", "03-video-action-recognition"]
)
def test_notebook_is_output_free_and_compiles(demo: str) -> None:
    code = []
    for cell in notebook_cells(demo):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
            code.append(cell_source(cell))
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
    source = cell_source(notebook_cells(demo)[2])
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
    assert "read_verified_image(Path(row.image_path), row.source_sha256)" in serialization
    assert "payload = path.read_bytes()" in serialization
    assert "return payload" in serialization
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


def notebook_function(demo: str, name: str, **namespace):
    source = source_containing(demo, f"def {name}(")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), namespace)
    return namespace[name]


def test_image_download_uses_verified_immutable_version(tmp_path: Path) -> None:
    source = source_containing("02-image-classification", "def clean_breed")
    assert 'DOG_DATASET_VERSION = "2"' in source
    dataset = tmp_path / "versions" / "2"
    images = dataset / "images" / "Images"
    images.mkdir(parents=True)
    requested = []
    namespace = {
        "Path": Path,
        "kagglehub": SimpleNamespace(
            dataset_download=lambda handle: requested.append(handle) or str(dataset),
        ),
    }
    names = {"DOG_DATASET", "DOG_DATASET_VERSION", "DOG_DATASET_HANDLE"}
    definitions = [
        node for node in ast.parse(source).body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name == "get_images_root")
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "image-source", "exec"), namespace)
    assert namespace["get_images_root"]() == images
    assert requested == ["jessicali9530/stanford-dogs-dataset/versions/2"]


def test_image_manifest_records_version_and_selected_source_hashes() -> None:
    source = source_containing("02-image-classification", "def clean_breed")
    assert '"dataset_version": DOG_DATASET_VERSION' in source
    assert 'selected["source_sha256"]' in source
    assert '"relative_path"' in source
    serialization = source_containing("02-image-classification", "def write_jsonl")
    assert '"source": source_provenance' in serialization


@pytest.mark.parametrize("changed", [False, True])
def test_image_encoding_verifies_selected_source_bytes(tmp_path: Path, changed: bool) -> None:
    image = tmp_path / "image.jpg"
    original = b"selected-source-jpeg"
    image.write_bytes(b"modified-source-jpeg" if changed else original)
    read_verified_image = notebook_function(
        "02-image-classification", "read_verified_image", Path=Path, hashlib=hashlib,
    )
    expected = hashlib.sha256(original).hexdigest()
    if changed:
        with pytest.raises(ValueError, match="Source image changed"):
            read_verified_image(image, expected)
    else:
        assert read_verified_image(image, expected) == original


def test_video_download_and_source_csvs_are_pinned() -> None:
    source = source_containing("03-video-action-recognition", "dataset_download")
    assert 'UCF101_VERSION = "4"' in source
    assert 'f"{UCF101_DATASET}/versions/{UCF101_VERSION}"' in source
    assert "kagglehub.dataset_download(UCF101_HANDLE)" in source
    # Independently retrieved from the two small version-4 CSV downloads.
    assert "e67f43569701064a568a1e56bcfb2cce5646caa017c1f45171bcb347aeed6ea4" in source
    assert "df8125e4170c3567ee89db96a860be524c6e031fb35aa7c96925e3dd8f4fcabf" in source


@pytest.mark.parametrize("changed", [False, True])
def test_video_verifies_csv_bytes_before_parsing(tmp_path: Path, changed: bool) -> None:
    original = b"clip_name,clip_path,label\nclip,train/clip.avi,Action\n"
    (tmp_path / "train.csv").write_bytes(original + b"extra" if changed else original)
    checksums = {}
    parsed = []
    load_source_csv = notebook_function(
        "03-video-action-recognition", "load_source_csv",
        dataset_path=tmp_path, hashlib=hashlib, StringIO=StringIO,
        UCF101_VERSION="4",
        SOURCE_CSV_SHA256={"train.csv": hashlib.sha256(original).hexdigest()},
        source_csv_sha256=checksums,
        pd=SimpleNamespace(DataFrame=object, read_csv=lambda stream: parsed.append(stream.getvalue())),
    )
    if changed:
        with pytest.raises(ValueError, match="Source CSV hash mismatch"):
            load_source_csv("train.csv")
        assert not parsed and not checksums
    else:
        load_source_csv("train.csv")
        assert parsed == [original.decode("utf-8")]
        assert checksums == {"train.csv": hashlib.sha256(original).hexdigest()}


def test_video_lineage_records_verified_sources_and_selected_clips() -> None:
    source = source_containing("03-video-action-recognition", '"lineage.json"')
    assert '"dataset_version": UCF101_VERSION' in source
    assert '"source_csv_sha256": source_csv_sha256' in source
    assert '"selected_clips"' in source
    for split in ("train", "validation"):
        assert f'{split}_frame[["clip_path", "label"]].to_dict(orient="records")' in source


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
        cell_source(cell).casefold() for cell in notebook_cells(demo)
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
    assert sentence in "\n".join(cell_source(cell) for cell in notebook_cells(demo))


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


def test_chart_request_serializes_only_supported_hyperparameters(tmp_path: Path) -> None:
    import httpx
    from openai import OpenAI

    source = source_containing("01-chart-reasoning", "state = load_state()")
    state_path = tmp_path / "run-state.json"
    state_path.write_text(json.dumps({
        "run_id": "offline-chart", "project_endpoint": "",
        "model": "gpt-4.1-2025-04-14", "training_type": "GlobalStandard",
        "datasets": [], "training_file_id": "file-training",
        "validation_file_id": "file-validation",
    }), encoding="utf-8")
    requests = []
    payloads = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.startswith("/v1/files/"):
            return httpx.Response(200, json={
                "id": request.url.path.rsplit("/", 1)[-1], "object": "file",
                "bytes": 16, "created_at": 1, "filename": "offline.jsonl",
                "purpose": "fine-tune", "status": "processed",
            })
        if (request.method, request.url.path) == ("GET", "/v1/fine_tuning/jobs"):
            return httpx.Response(200, json={
                "object": "list", "data": [], "has_more": False,
            })
        assert (request.method, request.url.path) == ("POST", "/v1/fine_tuning/jobs")
        payloads.append(json.loads(request.content))
        assert request.headers["content-type"].startswith("application/json")
        return httpx.Response(200, json={
            "id": "ftjob-offline-chart", "object": "fine_tuning.job",
            "created_at": 1, "finished_at": 2, "status": "succeeded",
            "model": "gpt-4.1-2025-04-14", "organization_id": "offline",
            "training_file": "file-training", "validation_file": "file-validation",
            "seed": 42, "result_files": [], "trained_tokens": 1,
            "fine_tuned_model": "offline-model", "error": None,
            "hyperparameters": {"n_epochs": 1}, "method": payloads[-1]["method"],
        })

    with OpenAI(
        api_key="offline-test", base_url="https://offline.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        namespace = {
            "STATE_PATH": state_path, "RUN_ID": "offline-chart",
            "PROJECT_ENDPOINT": "", "MODEL": "gpt-4.1-2025-04-14",
            "TRAINING_TYPE": "GlobalStandard", "dataset_evidence": [],
            "DATASETS": {"train": {"path": tmp_path / "train"}, "validation": {"path": tmp_path / "validation"}},
            "client": client, "json": json, "Path": Path, "time": time,
            "datetime": datetime, "timezone": timezone,
            "TERMINAL_STATES": {"succeeded", "failed", "cancelled"},
        }
        exec(compile(source, "chart-sdk-request", "exec"), namespace)

    assert requests.count(("POST", "/v1/fine_tuning/jobs")) == 1
    assert len(payloads) == 1
    assert payloads[0] == {
        "suffix": "offline-chart", "training_file": "file-training",
        "validation_file": "file-validation", "model": "gpt-4.1-2025-04-14",
        "seed": 42, "trainingType": "GlobalStandard",
        "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 1}}},
    }
    hyperparameters = payloads[0]["method"]["supervised"]["hyperparameters"]
    assert "batch_size" not in hyperparameters
    assert "learning_rate_multiplier" not in hyperparameters
    assert None not in hyperparameters.values()


@pytest.mark.parametrize("outcome", ["timeout", "accepted-response-lost", "missing-id"])
def test_image_upload_journals_before_polling_and_blocks_ambiguous_repost(
    tmp_path: Path, outcome: str,
) -> None:
    path = tmp_path / "training.jsonl"
    path.write_text('{"messages":[]}\n', encoding="utf-8")
    state_path = tmp_path / "file-state.json"
    state = {"fingerprint": "offline", "project_endpoint": "https://offline.invalid"}
    posted = []
    waited = []
    options = []

    def create(**kwargs):
        pending = json.loads(state_path.read_text(encoding="utf-8"))
        assert pending["training_upload_pending"] is True
        assert "training_file_id" not in pending
        posted.append(kwargs)
        if outcome == "accepted-response-lost":
            raise ConnectionError("accepted then response lost")
        return SimpleNamespace(id=None if outcome == "missing-id" else "file-known")

    def wait(file_id):
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted["training_file_id"] == file_id == "file-known"
        assert persisted["training_upload_pending"] is False
        waited.append(file_id)
        raise TimeoutError("processing timeout")

    client = SimpleNamespace(files=SimpleNamespace(create=create))
    client.with_options = lambda **kwargs: options.append(kwargs) or client
    namespace = {
        "Path": Path, "Any": object, "json": json, "file_state": state,
        "file_state_path": state_path, "client": client, "wait_for_file": wait,
    }
    namespace["resource_id"] = notebook_function(
        "02-image-classification", "resource_id", Any=object,
    )
    namespace["save_file_state"] = notebook_function(
        "02-image-classification", "save_file_state", **namespace,
    )
    upload = notebook_function("02-image-classification", "upload_file", **namespace)
    expected = TimeoutError if outcome == "timeout" else RuntimeError
    match = "processing timeout" if outcome == "timeout" else "not retried.*Reconcile"
    with pytest.raises(expected, match=match):
        upload(path, "offline-digest", split="training")
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    state.clear()
    state.update(persisted)
    with pytest.raises(expected, match="processing timeout" if outcome == "timeout" else "Previous training upload outcome is unknown"):
        upload(path, "offline-digest", split="training")
    assert len(posted) == 1
    assert options == [{"max_retries": 0}]
    assert waited == (["file-known", "file-known"] if outcome == "timeout" else [])


@pytest.mark.parametrize("demo", ["01-chart-reasoning", "03-video-action-recognition"])
@pytest.mark.parametrize("outcome", ["accepted-response-lost", "terminal-http", "missing-id"])
def test_notebook_upload_failure_blocks_repost_without_filename_reuse(
    tmp_path: Path, demo: str, outcome: str,
) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    state_path = tmp_path / "run-state.json"
    state = {"created_at": datetime.now(timezone.utc).isoformat()}
    created = []
    options = []

    class UploadFailure(RuntimeError):
        status_code = 400 if outcome == "terminal-http" else 503

    def create(**kwargs):
        pending = json.loads(state_path.read_text(encoding="utf-8"))
        assert pending["training_upload_pending"] is True
        assert "training_file_id" not in pending
        created.append(kwargs)
        if outcome == "missing-id":
            return SimpleNamespace(id=None)
        raise UploadFailure("accepted, then response lost" if outcome == "accepted-response-lost" else "invalid upload")

    client = SimpleNamespace(files=SimpleNamespace(
        create=create,
        list=lambda **_kwargs: pytest.fail("Do not select a filename after an ambiguous POST"),
    ))
    client.with_options = lambda **kwargs: options.append(kwargs) or client

    def save(*_args):
        state_path.write_text(json.dumps(state), encoding="utf-8")

    ensure_uploaded = notebook_function(
        demo, "ensure_uploaded", Path=Path, state=state, run_state=state,
        STATE_PATH=state_path, RUN_STATE_PATH=state_path,
        save_state=save, save_run_state=save, client=client,
        wait_for_file=lambda *_args: pytest.fail("Must not poll an unknown file ID"),
    )
    with pytest.raises(RuntimeError, match="not retried.*Reconcile") as caught:
        ensure_uploaded("training", source)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "filename alone" in str(caught.value)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    state.clear()
    state.update(persisted)
    with pytest.raises(RuntimeError, match="Previous training upload outcome is unknown"):
        ensure_uploaded("training", source)
    assert len(created) == 1
    assert options == [{"max_retries": 0}]


@pytest.mark.parametrize("demo", ["01-chart-reasoning", "03-video-action-recognition"])
@pytest.mark.parametrize("outcome", ["timeout", "error", "failed", "cancelled", "expired"])
def test_notebook_saves_file_id_before_wait_and_resumes_it(
    tmp_path: Path, demo: str, outcome: str,
) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    state_path = tmp_path / "run-state.json"
    state = {"created_at": datetime.now(timezone.utc).isoformat()}
    created = []
    waited = []
    client = SimpleNamespace(files=SimpleNamespace(
        create=lambda **kwargs: created.append(kwargs) or SimpleNamespace(id="file-known"),
        list=lambda **_kwargs: pytest.fail("Use the recorded ID, not a filename"),
    ))
    client.with_options = lambda **_kwargs: client

    def save(*_args):
        state_path.write_text(json.dumps(state), encoding="utf-8")

    def wait(file_id, *_args):
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted["training_file_id"] == file_id == "file-known"
        assert persisted["training_upload_pending"] is False
        waited.append(file_id)
        if outcome == "timeout":
            raise TimeoutError("file processing timeout")
        wait_for_file = notebook_function(
            demo, "wait_for_file", client=client, time=time,
            retry_service_call=lambda operation, _name: operation(),
        )
        return wait_for_file(file_id, "training") if demo.startswith("01") else wait_for_file(file_id)

    client.files.retrieve = lambda file_id: SimpleNamespace(
        id=file_id, status=outcome, status_details="Invalid source data",
    )

    ensure_uploaded = notebook_function(
        demo, "ensure_uploaded", Path=Path, state=state, run_state=state,
        STATE_PATH=state_path, RUN_STATE_PATH=state_path,
        save_state=save, save_run_state=save, client=client, wait_for_file=wait,
    )
    expected = TimeoutError if outcome == "timeout" else RuntimeError
    for _ in range(2):
        match = "file processing timeout" if outcome == "timeout" else "(file import failed|ended as)"
        with pytest.raises(expected, match=match):
            ensure_uploaded("training", source)
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        state.clear()
        state.update(persisted)
    assert len(created) == 1
    assert waited == ["file-known", "file-known"]


class MonitorClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert 0 < seconds
        self.sleeps.append(seconds)
        self.now += seconds

    def time(self) -> float:
        pytest.fail("Monitoring deadlines must not use wall-clock time")


def video_monitor(tmp_path: Path, clock: MonitorClock, retrieve):
    state_path = tmp_path / "run-state.json"
    state = {"job_id": "job-known", "submission_pending": False}
    options = []
    jobs = SimpleNamespace(
        retrieve=retrieve,
        cancel=lambda *_args, **_kwargs: pytest.fail("Monitoring must not cancel a remote job"),
        create=lambda **_kwargs: pytest.fail("Monitoring must not create another job"),
    )
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    client.with_options = lambda **kwargs: options.append(kwargs) or client
    namespace = {
        "RUN_STATE_PATH": state_path, "run_state": state, "json": json,
    }
    save_run_state = notebook_function("03-video-action-recognition", "save_run_state", **namespace)
    save_run_state()
    monitor = notebook_function(
        "03-video-action-recognition", "monitor_job", **namespace,
        client=client, save_run_state=save_run_state, time=clock,
        pd=SimpleNamespace(Timestamp=SimpleNamespace(now=lambda _zone: datetime.now(timezone.utc))),
        TERMINAL_STATES={"succeeded", "failed", "cancelled"},
    )
    return monitor, state, state_path, options


def monitor_job_value(status: str):
    return SimpleNamespace(
        id="job-known", status=status,
        to_dict=lambda: {"id": "job-known", "status": status, "error": None},
    )


def test_video_monitor_deadline_persists_latest_state_and_resumes(tmp_path: Path) -> None:
    clock = MonitorClock()
    requests = []
    completed = False

    def retrieve(job_id):
        requests.append(job_id)
        clock.now += 2
        status = "succeeded" if completed else "queued" if len(requests) == 1 else "running"
        return monitor_job_value(status)

    monitor, state, state_path, options = video_monitor(tmp_path, clock, retrieve)
    with pytest.raises(TimeoutError, match="Resume by rerunning this cell.*not cancelled"):
        monitor("job-known", timeout_seconds=10, poll_interval_seconds=3, request_timeout_seconds=60)

    assert clock.now == 10
    assert clock.sleeps == [3, 3]
    assert requests == ["job-known", "job-known"]
    assert options == [{"max_retries": 0, "timeout": 10}, {"max_retries": 0, "timeout": 5}]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["job_id"] == "job-known"
    assert persisted["status"] == persisted["job"]["status"] == "running"
    assert persisted["monitor_timed_out"] is True
    assert [entry["status"] for entry in persisted["status_history"]] == ["queued", "running"]

    state.clear()
    state.update(persisted)
    completed = True
    assert monitor("job-known", timeout_seconds=10).status == "succeeded"
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert requests == ["job-known"] * 3
    assert resumed["job_id"] == "job-known"
    assert resumed["status"] == "succeeded"
    assert "monitor_timed_out" not in resumed
    assert [entry["status"] for entry in resumed["status_history"]] == ["queued", "running", "succeeded"]


@pytest.mark.parametrize("failure_kind", ["http", "connection", "sdk-timeout"])
def test_video_monitor_retries_share_request_and_sleep_deadline(
    tmp_path: Path, failure_kind: str,
) -> None:
    import httpx
    from openai import APITimeoutError

    clock = MonitorClock()
    calls = []

    class Busy(RuntimeError):
        status_code = 503

    def retrieve(job_id):
        calls.append(job_id)
        clock.now += 1
        if failure_kind == "http":
            raise Busy("temporarily unavailable")
        if failure_kind == "connection":
            raise ConnectionError("connection lost")
        raise APITimeoutError(request=httpx.Request("GET", "https://offline.invalid/jobs/job-known"))

    monitor, _state, state_path, options = video_monitor(tmp_path, clock, retrieve)
    with pytest.raises(TimeoutError, match="job-known.*deadline.*Resume"):
        monitor("job-known", timeout_seconds=7, request_timeout_seconds=2)

    assert clock.now == 7
    assert clock.sleeps == [2, 3]
    assert calls == ["job-known", "job-known"]
    assert options == [{"max_retries": 0, "timeout": 2}, {"max_retries": 0, "timeout": 2}]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["job_id"] == "job-known"
    assert persisted["monitor_timed_out"] is True


@pytest.mark.parametrize("late_status", ["running", "succeeded"])
def test_video_monitor_request_time_counts_and_preserves_late_state(
    tmp_path: Path, late_status: str,
) -> None:
    clock = MonitorClock()
    calls = []

    def retrieve(job_id):
        calls.append(job_id)
        clock.now += 5
        return monitor_job_value(late_status)

    monitor, _state, state_path, options = video_monitor(tmp_path, clock, retrieve)
    with pytest.raises(TimeoutError, match="job-known.*deadline"):
        monitor("job-known", timeout_seconds=5)
    assert calls == ["job-known"]
    assert not clock.sleeps
    assert options == [{"max_retries": 0, "timeout": 5}]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["job"]["status"] == late_status
    assert persisted["job_id"] == "job-known"


@pytest.mark.parametrize("status_code", [400, 401, 403, 422])
def test_video_monitor_terminal_request_error_is_not_retried(
    tmp_path: Path, status_code: int,
) -> None:
    clock = MonitorClock()
    calls = []

    class Rejected(RuntimeError):
        pass

    error = Rejected("invalid request")
    error.status_code = status_code

    def retrieve(job_id):
        calls.append(job_id)
        raise error

    monitor, _state, state_path, options = video_monitor(tmp_path, clock, retrieve)
    with pytest.raises(Rejected) as caught:
        monitor("job-known", timeout_seconds=10)
    assert caught.value is error
    assert calls == ["job-known"]
    assert not clock.sleeps
    assert options == [{"max_retries": 0, "timeout": 10}]
    assert json.loads(state_path.read_text(encoding="utf-8"))["job_id"] == "job-known"


def test_video_monitor_retry_exhaustion_preserves_cause_and_job_id(tmp_path: Path) -> None:
    clock = MonitorClock()
    calls = []
    failure = ConnectionError("unavailable")

    def retrieve(job_id):
        calls.append(job_id)
        clock.now += 1
        raise failure

    monitor, _state, state_path, options = video_monitor(tmp_path, clock, retrieve)
    with pytest.raises(RuntimeError, match="failed after 2 attempts.*Resume") as caught:
        monitor("job-known", timeout_seconds=20, request_timeout_seconds=3, max_attempts=2)
    assert caught.value.__cause__ is failure
    assert calls == ["job-known", "job-known"]
    assert clock.now == 4
    assert clock.sleeps == [2]
    assert options == [{"max_retries": 0, "timeout": 3}, {"max_retries": 0, "timeout": 3}]
    assert json.loads(state_path.read_text(encoding="utf-8"))["job_id"] == "job-known"


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_video_monitor_terminal_job_returns_and_persists_without_sleep(
    tmp_path: Path, status: str,
) -> None:
    clock = MonitorClock()
    job = monitor_job_value(status)
    monitor, _state, state_path, _options = video_monitor(tmp_path, clock, lambda _job_id: job)
    assert monitor("job-known", timeout_seconds=10) is job
    assert not clock.sleeps
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == status
    assert persisted["job_id"] == "job-known"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("parameter", ["timeout_seconds", "poll_interval_seconds", "request_timeout_seconds"])
def test_video_monitor_rejects_nonfinite_or_nonpositive_limits(
    tmp_path: Path, parameter: str, value: float,
) -> None:
    clock = MonitorClock()
    monitor, _state, _path, options = video_monitor(
        tmp_path, clock, lambda _job_id: pytest.fail("Invalid limits must not issue a request"),
    )
    with pytest.raises(ValueError, match=f"{parameter} must be positive and finite"):
        monitor("job-known", **{parameter: value})
    assert not options and not clock.sleeps


def test_video_monitor_sdk_request_uses_remaining_timeout_and_no_hidden_retries(tmp_path: Path) -> None:
    import httpx
    from openai import OpenAI

    clock = MonitorClock()
    monitor, _state, _path, _options = video_monitor(
        tmp_path, clock, lambda _job_id: pytest.fail("The real SDK must use the mock transport"),
    )
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("GET", "/v1/fine_tuning/jobs/job-known")
        requests.append(request.extensions["timeout"])
        clock.now += 1
        return httpx.Response(503, json={
            "error": {"message": "Temporarily unavailable", "type": "server_error"},
        })

    with OpenAI(
        api_key="offline-test", base_url="https://offline.invalid/v1",
        max_retries=2, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monitor.__globals__["client"] = client
        with pytest.raises(TimeoutError, match="job-known.*deadline.*Resume"):
            monitor("job-known", timeout_seconds=7, request_timeout_seconds=5)

    assert requests == [
        {"connect": 5, "read": 5, "write": 5, "pool": 5},
        {"connect": 4, "read": 4, "write": 4, "pool": 4},
    ]
    assert clock.sleeps == [2, 3]
    assert clock.now == 7


@pytest.mark.parametrize("outcome", ["terminal", "timeout"])
def test_video_training_cell_resumes_recorded_job_through_bounded_monitor(
    tmp_path: Path, outcome: str,
) -> None:
    source = source_containing("03-video-action-recognition", "def retry_service_call")
    state_path = tmp_path / "run-state.json"
    state_path.write_text(json.dumps({
        "project_endpoint": "", "model": "model",
        "training_sha256": "train-hash", "validation_sha256": "validation-hash",
        "job_id": "job-known", "submission_pending": False,
        "training_file_id": "file-training", "validation_file_id": "file-validation",
    }), encoding="utf-8")
    clock = MonitorClock()
    retrieved = []
    options = []

    def retrieve(job_id):
        retrieved.append(job_id)
        if outcome == "timeout":
            clock.now += 24 * 60 * 60
        return monitor_job_value("running" if outcome == "timeout" else "succeeded")

    client = SimpleNamespace(
        files=SimpleNamespace(create=lambda **_kwargs: pytest.fail("Must not upload on resume")),
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(
            retrieve=retrieve,
            create=lambda **_kwargs: pytest.fail("Must not submit on resume"),
            cancel=lambda *_args, **_kwargs: pytest.fail("Must not cancel on timeout"),
            list_events=lambda **_kwargs: SimpleNamespace(data=[]),
        )),
    )
    client.with_options = lambda **kwargs: options.append(kwargs) or client
    namespace = {
        "PROJECT_ENDPOINT": "", "MODEL": "model", "RUN_STATE_PATH": state_path,
        "input_summary": {"training": {"sha256": "train-hash"}, "validation": {"sha256": "validation-hash"}},
        "OUTPUT_DIR": tmp_path, "client": client, "json": json,
        "Path": Path, "time": clock,
        "pd": SimpleNamespace(Timestamp=SimpleNamespace(now=lambda _zone: datetime.now(timezone.utc))),
    }
    code = compile(source, "video-resume-monitor", "exec")
    if outcome == "timeout":
        with pytest.raises(TimeoutError, match="Resume by rerunning this cell.*not cancelled"):
            exec(code, namespace)
    else:
        exec(code, namespace)
        assert namespace["job"].status == "succeeded"
    assert retrieved == ["job-known"]
    assert options == [{"max_retries": 0, "timeout": 60}]
    assert not clock.sleeps
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["job_id"] == "job-known"
    assert persisted["status"] == ("running" if outcome == "timeout" else "succeeded")
