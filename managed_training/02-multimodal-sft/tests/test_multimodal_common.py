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
        self.create_calls = 0
        self.retrieve_calls = []

    def list(self) -> list[object]:
        return []

    def create(self, *, file: tuple[str, object, str], purpose: str) -> object:
        self.create_calls += 1
        self.created_name = file[0]
        assert purpose == "fine-tune"
        return SimpleNamespace(id="file-created")

    def retrieve(self, file_id: str) -> object:
        self.retrieve_calls.append(file_id)
        return SimpleNamespace(id=file_id, status=self.retrieve_status)


def test_image_preprocessing_and_split_guards(tmp_path: Path) -> None:
    import base64
    import io

    from PIL import Image

    uri = common.image_uri(Image.new("RGB", (32, 16), "red"), 16, 95)
    prefix, encoded = uri.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.size == (16, 8)
        assert decoded.mode == "RGB"
    common.assert_disjoint({"train": {"one"}, "validation": {"two"}})
    with pytest.raises(ValueError, match="Split leakage"):
        common.assert_disjoint({"train": {"one"}, "validation": {"one"}})


def test_upload_uses_content_addressed_filename(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    files = FakeFiles()
    client = SimpleNamespace(files=files)
    options = {}
    client.with_options = lambda **kwargs: options.update(kwargs) or client

    uploaded = common.upload_or_reuse_file(client, source)

    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    assert files.created_name == f"training-{digest}.jsonl"
    assert uploaded.id == "file-created"
    assert files.create_calls == 1
    assert options == {"max_retries": 0}


def test_upload_accepted_then_response_lost_posts_once(tmp_path: Path) -> None:
    import httpx
    from openai import APIConnectionError, OpenAI

    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    requests = []
    accepted = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"data": [], "has_more": False})
        if request.method == "POST" and request.url.path == "/v1/files":
            accepted.append("file-accepted")
            raise httpx.ReadError("accepted, then response lost", request=request)
        pytest.fail(f"Unexpected request: {request.method} {request.url.path}")

    with OpenAI(
        api_key="offline-test", base_url="https://offline.invalid/v1",
        max_retries=2, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        with pytest.raises(RuntimeError, match="not retried.*Reconcile") as caught:
            common.upload_or_reuse_file(client, source)

    assert isinstance(caught.value.__cause__, APIConnectionError)
    assert "filename alone" in str(caught.value)
    assert accepted == ["file-accepted"]
    assert requests == [("GET", "/v1/files"), ("POST", "/v1/files")]


@pytest.mark.parametrize("status_code", [400, 401, 403, 422, 429, 503])
def test_upload_request_errors_never_repeat_post(
    tmp_path: Path, status_code: int,
) -> None:
    import httpx
    from openai import APIStatusError, OpenAI

    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"data": [], "has_more": False})
        assert (request.method, request.url.path) == ("POST", "/v1/files")
        return httpx.Response(
            status_code,
            json={"error": {"message": "Upload rejected", "type": "upload_error"}},
        )

    with OpenAI(
        api_key="offline-test", base_url="https://offline.invalid/v1",
        max_retries=2, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        with pytest.raises(RuntimeError, match="not retried") as caught:
            common.upload_or_reuse_file(client, source)

    assert isinstance(caught.value.__cause__, APIStatusError)
    assert caught.value.__cause__.status_code == status_code
    assert requests == [("GET", "/v1/files"), ("POST", "/v1/files")]


@pytest.mark.parametrize("status", ["error", "failed", "cancelled", "expired"])
def test_upload_terminal_file_error_does_not_repeat_post(
    tmp_path: Path, status: str,
) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    files = FakeFiles()
    files.retrieve_status = status
    client = SimpleNamespace(files=files)
    client.with_options = lambda **_kwargs: client

    with pytest.raises(RuntimeError, match=f"File file-created ended in terminal state {status}"):
        common.upload_or_reuse_file(client, source)

    assert files.create_calls == 1
    assert files.retrieve_calls == ["file-created"]


@pytest.mark.parametrize("missing_id", [None, "", "  "])
def test_upload_missing_id_fails_closed_without_waiting(
    tmp_path: Path, missing_id: str | None,
) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    created = []
    client = SimpleNamespace(files=SimpleNamespace(
        list=lambda: [],
        create=lambda **kwargs: created.append(kwargs) or SimpleNamespace(id=missing_id),
        retrieve=lambda *_args: pytest.fail("Must not poll an unknown file ID"),
    ))
    client.with_options = lambda **_kwargs: client

    with pytest.raises(RuntimeError, match="returned no file ID.*Reconcile"):
        common.upload_or_reuse_file(client, source)
    assert len(created) == 1


@pytest.mark.parametrize("case", ["duplicate", "wrong-purpose", "wrong-size", "missing-metadata"])
def test_upload_refuses_ambiguous_filename_reuse(tmp_path: Path, case: str) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    candidate = SimpleNamespace(
        id="file-existing", filename=f"training-{digest}.jsonl",
        purpose="fine-tune", bytes=source.stat().st_size,
    )
    if case == "wrong-purpose":
        candidate.purpose = "assistants"
    elif case == "wrong-size":
        candidate.bytes += 1
    elif case == "missing-metadata":
        del candidate.purpose
    matches = [candidate, candidate] if case == "duplicate" else [candidate]
    client = SimpleNamespace(files=SimpleNamespace(
        list=lambda: matches,
        retrieve=lambda *_args: pytest.fail("Must not reuse ambiguous matches"),
    ))
    client.with_options = lambda **_kwargs: pytest.fail("Must not create another file")

    with pytest.raises(RuntimeError, match="reconcile"):
        common.upload_or_reuse_file(client, source)


def test_upload_reuses_unique_file_with_matching_metadata(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    candidate = SimpleNamespace(
        id="file-existing", filename=f"training-{digest}.jsonl",
        purpose="fine-tune", bytes=source.stat().st_size,
    )
    retrieved = []
    client = SimpleNamespace(files=SimpleNamespace(
        list=lambda: [candidate],
        retrieve=lambda file_id: retrieved.append(file_id) or SimpleNamespace(
            id=file_id, status="processed",
        ),
    ))
    client.with_options = lambda **_kwargs: pytest.fail("Must not upload an existing file")

    assert common.upload_or_reuse_file(client, source).id == "file-existing"
    assert retrieved == ["file-existing"]


def test_reused_failed_job_raises(
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


def test_creation_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attempts = 0
    options = {}

    class AmbiguousFailure(RuntimeError):
        status_code = 503

    def create(**kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        raise AmbiguousFailure("submission outcome unknown")

    jobs = SimpleNamespace(create=create)
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))

    def with_options(**kwargs: object) -> object:
        options.update(kwargs)
        return client

    client.with_options = with_options
    monkeypatch.setattr(
        common, "upload_or_reuse_file",
        lambda *_args, **_kwargs: SimpleNamespace(id="file-test"),
    )
    with pytest.raises(AmbiguousFailure, match="outcome unknown"):
        common.resolve_or_create_job(
            client=client,
            reuse_id="",
            model="model",
            train=tmp_path / "train.jsonl",
            validation=tmp_path / "validation.jsonl",
            suffix="customer-test",
            seed=42,
            training_type="GlobalStandard",
        )
    assert attempts == 1
    assert options == {"max_retries": 0}


def test_new_job_submits_without_execution_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    created = []
    job = SimpleNamespace(id="job-created", status="succeeded")
    jobs = SimpleNamespace(
        create=lambda **kwargs: created.append(kwargs) or job,
        retrieve=lambda _job_id: job,
    )
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    client.with_options = lambda **_kwargs: client
    monkeypatch.setattr(
        common, "upload_or_reuse_file",
        lambda *_args, **_kwargs: SimpleNamespace(id="file-test"),
    )
    result = common.resolve_or_create_job(
        client=client, reuse_id="", model="model",
        train=tmp_path / "train.jsonl", validation=tmp_path / "validation.jsonl",
        suffix="customer-test", seed=42, training_type="GlobalStandard",
    )
    assert result is job
    assert len(created) == 1
    assert created[0]["method"]["supervised"]["hyperparameters"]["n_epochs"] == 1


def test_file_lookup_iterates_sdk_pages() -> None:
    class Page:
        data = ["first"]

        def has_next_page(self) -> bool:
            return True

        def __iter__(self):
            return iter(["first", "second"])

    assert common._iter_page(Page()) == ["first", "second"]
