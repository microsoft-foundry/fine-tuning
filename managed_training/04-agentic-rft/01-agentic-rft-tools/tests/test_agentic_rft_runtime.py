"""Offline regressions: in-process tool HTTP and fault-injected SDK objects only."""

from __future__ import annotations

import ast
import copy
import errno
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "demo.ipynb"
PROJECT = "https://foundry.example.test/api/projects/offline-project"
TOOL_URL = "https://tool.example.test/tool/search_catalog"
TOKEN = "offline-only-catalog-credential"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load_module("agentic_rft_runtime_regression", ROOT / "src" / "agentic_rft_runtime.py")
catalog = _load_module("agentic_rft_catalog_regression", ROOT / "tools" / "catalog_server.py")


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    if isinstance(value, str):
        return value
    # Retained notebooks may use unterminated-line arrays; editor saves use LF lines.
    return "".join(value) if any(line.endswith("\n") for line in value) else "\n".join(value)


def _notebook_recipe(token: str = TOKEN) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    tree = ast.parse(_source(cells[4]))
    grader = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MULTI_GRADER" for target in node.targets)
    )
    namespace: dict[str, Any] = {
        "TOOL_SERVER_URL": TOOL_URL, "MODEL": "qwen3.6-35b-a3b", "MODEL_VERSION": "1",
        "load_tool_headers": lambda: runtime.load_tool_headers({"FOUNDRY_TOOL_AUTH_TOKEN": token}),
    }
    exec(compile(ast.Module(body=[grader], type_ignores=[]), "offline-grader-constant", "exec"), namespace)
    # This is only the offline recipe-definition cell, never upload/monitoring cells.
    exec(compile(_source(cells[6]), "offline-tool-recipe", "exec"), namespace)
    return namespace["METHOD"], namespace


def _state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeFiles:
    def __init__(self, owner: "FakeClient") -> None:
        self.owner = owner
        self.resources: dict[str, Any] = {}
        self.contents: dict[str, bytes] = {}
        self.created = 0
        self.return_missing_id = False
        self.fail_after_accept = False
        self.read_failure: Exception | None = None
        self.before_read: Any = None
        self.list_override: Any = None
        self.status = "processed"
        self.busy = False

    def list(self) -> Any:
        if self.list_override is not None:
            return self.list_override()
        return SimpleNamespace(data=list(self.resources.values()))

    def create(self, *, file: Any, purpose: str) -> Any:
        current = _state(self.owner.state_path)
        expected = "training_upload" if self.created == 0 else "validation_upload"
        assert current["pending_operation"] == expected
        assert current["job_id"] is None
        if expected == "validation_upload":
            assert current["training_file_id"] == "file-offline-1"
        assert self.owner.options[-1] == {"max_retries": 0}
        self.created += 1
        name, content, content_type = file
        assert content_type == "application/jsonl" and isinstance(content, bytes)
        assert b"\r\n" not in content and purpose == "fine-tune"
        item = SimpleNamespace(
            id=f"file-offline-{self.created}", filename=name, purpose=purpose,
            status=self.status, bytes=len(content),
        )
        self.resources[item.id] = item
        self.contents[item.id] = content
        if self.fail_after_accept:
            raise RuntimeError(f"Accepted upload; untrusted diagnostic {TOKEN}")
        return SimpleNamespace(status="uploaded") if self.return_missing_id else item

    def retrieve(self, file_id: str) -> Any:
        current = _state(self.owner.state_path)
        assert file_id in {current["training_file_id"], current["validation_file_id"]}
        if self.before_read is not None:
            self.before_read(current)
        if self.read_failure is not None:
            raise self.read_failure
        item = self.resources[file_id]
        return SimpleNamespace(**{**vars(item), "status": "uploaded"}) if self.busy else item

    def content(self, file_id: str) -> Any:
        return SimpleNamespace(content=self.contents[file_id])


class FakeJobs:
    def __init__(self, owner: "FakeClient") -> None:
        self.owner = owner
        self.resources: dict[str, Any] = {}
        self.created = 0
        self.fail_after_accept = False
        self.return_missing_id = False
        self.raise_on_read = False
        self.last_payload: dict[str, Any] | None = None

    def create(self, **payload: Any) -> Any:
        current = _state(self.owner.state_path)
        assert current["pending_operation"] == "job_create"
        assert current["training_file_id"] == payload["training_file"]
        assert current["validation_file_id"] == payload["validation_file"]
        assert current["job_id"] is None
        assert self.owner.options[-1] == {"max_retries": 0}
        self.created += 1
        self.last_payload = copy.deepcopy(payload)
        job = SimpleNamespace(id="job-offline", status="queued", **copy.deepcopy(payload))
        self.resources[job.id] = job
        if self.fail_after_accept:
            raise RuntimeError(f"Accepted job; diagnostic payload {json.dumps(payload)}")
        return SimpleNamespace(status="queued") if self.return_missing_id else job

    def retrieve(self, job_id: str) -> Any:
        assert _state(self.owner.state_path)["pending_operation"] in {None, "job_create"}
        if self.raise_on_read:
            raise RuntimeError(f"Untrusted read diagnostic {TOKEN}")
        return self.resources[job_id]


class FakeClient:
    def __init__(self, path: Path) -> None:
        self.state_path = path
        self.options: list[dict[str, Any]] = []
        self.files = FakeFiles(self)
        self.jobs = FakeJobs(self)
        self.fine_tuning = SimpleNamespace(jobs=self.jobs)

    def with_options(self, **options: Any) -> "FakeClient":
        self.options.append(options)
        return self


@pytest.fixture
def submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeClient, dict[str, Any]]:
    # Model a relocated demo; never write runtime state into the checkout's outputs.
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "src" / "agentic_rft_runtime.py"))
    method, _ = _notebook_recipe()
    path = tmp_path / "outputs" / "submission-state.json"
    options = {
        "state_path": path, "project_endpoint": PROJECT,
        "model": "qwen3.6-35b-a3b", "model_version": "1", "training_type": "globalStandard",
        "method": method,
        "training_path": ROOT / "data" / "tool-calling-train.jsonl",
        "validation_path": ROOT / "data" / "tool-calling-validation.jsonl",
        "expected_training_sha256": "11acebfc227c48fee9dd608734ac81b45544bec76c7b814c01e905783b0d6bc4",
        "expected_validation_sha256": "7835cef9a161a1a0311d6b140c782ba60e644aec54641a74746ceab1cdc23517",
    }
    return FakeClient(path), options


@pytest.mark.parametrize("tail", [
    "../submission-state.json", "../other/outputs/submission-state.json",
    "runs/../../submission-state.json",
])
def test_journal_rejects_traversal_before_writes_or_sdk_calls(submission: Any, tail: str) -> None:
    client, options = submission
    root = client.state_path.parent
    options["state_path"] = root / tail

    def never_list() -> Any:
        pytest.fail("A refused journal path must not reach the SDK")

    client.files.list_override = never_list
    with pytest.raises(ValueError, match="outputs"):
        runtime.submit_or_resume_job(client, **options)

    assert list(root.parent.iterdir()) == []
    assert client.options == []
    assert client.files.created == client.jobs.created == 0


def test_journal_rejects_another_outputs_tree_without_changing_existing_state(submission: Any) -> None:
    client, options = submission
    other = client.state_path.parent.parent / "other" / "outputs" / "submission-state.json"
    other.parent.mkdir(parents=True)
    original = b"Retain the other demo's state.\n"
    other.write_bytes(original)
    options["state_path"] = other

    with pytest.raises(ValueError, match="outputs"):
        runtime.submit_or_resume_job(client, **options)

    assert other.read_bytes() == original
    assert list(other.parent.iterdir()) == [other]
    assert not client.state_path.parent.exists()
    assert client.options == []
    assert client.files.created == client.jobs.created == 0


def _journal_symlink(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except NotImplementedError:
        pytest.skip("Symlink creation is not implemented on this platform")
    except OSError as error:
        if error.errno in {errno.ENOSYS, errno.EOPNOTSUPP} or (
            os.name == "nt" and getattr(error, "winerror", None) in {1, 50, 1314}
        ):
            pytest.skip("Symlink creation is unsupported or lacks the Windows symlink privilege")
        raise


@pytest.mark.parametrize("redirect", ["root", "directory", "file"])
def test_journal_rejects_symlink_escape_without_changing_existing_state(
    submission: Any, redirect: str,
) -> None:
    client, options = submission
    root = client.state_path.parent
    outside = root.parent / "outside"
    outside.mkdir()
    target = outside / "submission-state.json"
    original = b"Retain state outside the intended outputs.\n"
    target.write_bytes(original)
    if redirect == "root":
        _journal_symlink(root, outside, directory=True)
    else:
        root.mkdir()
        if redirect == "directory":
            _journal_symlink(root / "alias", outside, directory=True)
            options["state_path"] = root / "alias" / "submission-state.json"
        else:
            _journal_symlink(client.state_path, target, directory=False)

    with pytest.raises(ValueError, match="outputs"):
        runtime.submit_or_resume_job(client, **options)

    assert target.read_bytes() == original
    assert list(outside.iterdir()) == [target]
    assert not client.state_path.with_suffix(".json.lock").exists()
    assert client.options == []
    assert client.files.created == client.jobs.created == 0


@pytest.mark.parametrize("path_style", ["absolute", "demo-relative", "notebook-relative", "contained-parent"])
def test_journal_accepts_demo_and_notebook_paths_and_resumes(
    submission: Any, monkeypatch: pytest.MonkeyPatch, path_style: str,
) -> None:
    client, options = submission
    demo = client.state_path.parent.parent
    if path_style == "demo-relative":
        monkeypatch.chdir(demo)
        options["state_path"] = Path("outputs/submission-state.json")
    elif path_style == "notebook-relative":
        notebooks = demo / "notebooks"
        notebooks.mkdir()
        monkeypatch.chdir(notebooks)
        options["state_path"] = Path("../outputs/submission-state.json")
    elif path_style == "contained-parent":
        options["state_path"] = demo / "outputs" / "run" / ".." / "submission-state.json"

    job = runtime.submit_or_resume_job(client, **options)
    assert _state(client.state_path)["job_id"] == job.id == "job-offline"
    options["state_path"] = client.state_path
    assert runtime.submit_or_resume_job(client, **options).id == job.id
    assert client.files.created == 2 and client.jobs.created == 1


@pytest.mark.parametrize("alias_kind", ["demo", "directory", "file"])
def test_journal_resolves_safe_aliases_and_reuses_saved_state(submission: Any, alias_kind: str) -> None:
    client, options = submission
    root = client.state_path.parent
    job = runtime.submit_or_resume_job(client, **options)
    original = client.state_path.read_bytes()
    if alias_kind == "demo":
        alias = root.parent / "demo-alias"
        _journal_symlink(alias, root.parent, directory=True)
        options["state_path"] = alias / "outputs" / "submission-state.json"
    elif alias_kind == "directory":
        alias = root / "alias"
        _journal_symlink(alias, root, directory=True)
        options["state_path"] = alias / "submission-state.json"
    else:
        alias = root.parent / "journal-alias"
        alias.mkdir()
        target = alias / "submission-state.json"
        _journal_symlink(target, client.state_path, directory=False)
        options["state_path"] = target

    assert runtime.submit_or_resume_job(client, **options).id == job.id
    assert client.state_path.read_bytes() == original
    assert client.files.created == 2 and client.jobs.created == 1
    assert not client.state_path.with_suffix(".json.lock").exists()


def test_notebook_headers_reach_both_protected_tool_callers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FOUNDRY_TOOL_AUTH_TOKEN", TOKEN)
    method, namespace = _notebook_recipe()
    headers = method["reinforcement"]["tools"][0]["headers"]
    assert headers == namespace["TOOL_HEADERS"] == {"Authorization": f"Bearer {TOKEN}"}
    calls = []
    with TestClient(catalog.app, base_url="https://tool.example.test") as client:
        def post(url: str, **kwargs: Any) -> Any:
            assert url == TOOL_URL and kwargs.pop("timeout") == 60
            assert kwargs.pop("allow_redirects") is False
            calls.append(kwargs["headers"].copy())
            return client.post(url, follow_redirects=False, **kwargs)

        runtime.validate_remote_tool(TOOL_URL, headers=headers, post=post)
        # The service's remote-tool definition sends exactly the same header map.
        response = client.post(
            method["reinforcement"]["tools"][0]["server_url"], headers=headers,
            json={"arguments": json.dumps({"query": "Urban Puffer", "top_k": 1})},
        )
    assert response.status_code == 200
    assert calls == [headers]
    assert TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_remote_contract_check_does_not_bypass_protected_tool(
    headers: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDRY_TOOL_AUTH_TOKEN", TOKEN)
    with TestClient(catalog.app, base_url="https://tool.example.test") as client:
        def post(url: str, **kwargs: Any) -> Any:
            kwargs.pop("timeout")
            kwargs.pop("allow_redirects")
            return client.post(url, follow_redirects=False, **kwargs)

        with pytest.raises(RuntimeError, match="Remote tool validation failed"):
            runtime.validate_remote_tool(TOOL_URL, headers=headers, post=post)


def test_unprotected_tool_and_recipe_remain_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOUNDRY_TOOL_AUTH_TOKEN", raising=False)
    method, namespace = _notebook_recipe("")
    assert namespace["TOOL_HEADERS"] == {}
    assert "headers" not in method["reinforcement"]["tools"][0]
    assert runtime.load_tool_headers({}) == {}
    with TestClient(catalog.app, base_url="https://tool.example.test") as client:
        def post(url: str, **kwargs: Any) -> Any:
            kwargs.pop("timeout")
            kwargs.pop("allow_redirects")
            return client.post(url, follow_redirects=False, **kwargs)

        runtime.validate_remote_tool(TOOL_URL, headers={}, post=post)


@pytest.mark.parametrize("token", ["a\nb", "a\rb", "two words", "nonascii-\u00e9"])
def test_header_validation_does_not_echo_invalid_secret(token: str) -> None:
    with pytest.raises(ValueError) as caught:
        runtime.load_tool_headers({"FOUNDRY_TOOL_AUTH_TOKEN": token})
    assert token not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    ["http://tool.example.test", "https://user:secret@tool.example.test/tool", "https://tool.example.test?key=secret"],
)
def test_remote_tool_rejects_url_credentials_and_non_https_without_sending(url: str) -> None:
    def never_post(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid endpoints must not receive a request")

    with pytest.raises(ValueError):
        runtime.validate_remote_tool(url, headers={}, post=never_post)


def test_remote_tool_does_not_follow_redirects_or_echo_transport_errors() -> None:
    def redirect(url: str, **kwargs: Any) -> Any:
        assert kwargs["allow_redirects"] is False
        return SimpleNamespace(status_code=307)

    def fail(url: str, **kwargs: Any) -> Any:
        raise RuntimeError(f"Headers contain {TOKEN}")

    for post in (redirect, fail):
        with pytest.raises(RuntimeError) as caught:
            runtime.validate_remote_tool(TOOL_URL, headers={"Authorization": TOKEN}, post=post)
        assert TOKEN not in str(caught.value)
        assert caught.value.__suppress_context__


def test_per_id_checkpoints_precede_reads_and_state_has_no_secret(submission: Any) -> None:
    client, options = submission
    job = runtime.submit_or_resume_job(client, **options)
    current = _state(client.state_path)
    assert current["training_file_id"] == "file-offline-1"
    assert current["validation_file_id"] == "file-offline-2"
    assert current["job_id"] == job.id == "job-offline"
    assert current["pending_operation"] is None
    assert current["equivalence"]["project_endpoint"] == PROJECT
    assert current["equivalence"]["method"]["reinforcement"]["tools"][0]["headers"] == {
        "authorization": "<redacted>",
    }
    assert len(current["equivalence"]["method_fingerprint"]) == 64
    assert TOKEN not in client.state_path.read_text(encoding="utf-8")
    assert client.jobs.last_payload["method"] == options["method"]
    assert client.jobs.last_payload["extra_body"] == {"trainingType": "globalStandard"}
    assert client.jobs.last_payload["suffix"] == "tool-calling-rft"
    assert client.options == [{"max_retries": 0}] * 3
    runtime.submit_or_resume_job(client, **options)
    assert client.files.created == 2 and client.jobs.created == 1


def test_recipe_semantics_are_unchanged() -> None:
    method, _ = _notebook_recipe()
    reinforcement = method["reinforcement"]
    assert method["type"] == "reinforcement" and reinforcement["max_episode_steps"] == 10
    assert reinforcement["hyperparameters"] == {
        "eval_interval": 3, "eval_samples": 5, "compute_multiplier": 1.0, "reasoning_effort": "medium",
    }
    assert reinforcement["grader"]["calculate_output"] == "0.9 * exact + 0.1 * fuzzy"
    assert reinforcement["grader"]["invalid_grade"] == 0.0
    assert reinforcement["tools"][0]["name"] == "search_catalog"


@pytest.mark.parametrize("failure_kind", ["read", "processing", "timeout"])
def test_first_file_failure_retains_id_and_resumes_without_upload(
    submission: Any, failure_kind: str,
) -> None:
    client, options = submission
    if failure_kind == "read":
        client.files.read_failure = RuntimeError(f"Read failed {TOKEN}")
    elif failure_kind == "processing":
        client.files.status = "failed"
    else:
        client.files.busy = True
        options["file_timeout_seconds"] = 0.000000001
    with pytest.raises((RuntimeError, TimeoutError)) as caught:
        runtime.submit_or_resume_job(client, **options)
    assert TOKEN not in str(caught.value)
    assert _state(client.state_path)["training_file_id"] == "file-offline-1"
    assert _state(client.state_path)["validation_file_id"] is None
    assert client.files.created == 1 and client.jobs.created == 0
    client.files.read_failure = None
    client.files.busy = False
    client.files.resources["file-offline-1"].status = "processed"
    client.files.status = "processed"
    options["file_timeout_seconds"] = 600
    runtime.submit_or_resume_job(client, **options)
    assert client.files.created == 2 and client.jobs.created == 1


@pytest.mark.parametrize("missing_id", [False, True])
def test_accepted_unidentified_upload_reconciles_read_only(submission: Any, missing_id: bool) -> None:
    client, options = submission
    client.files.fail_after_accept = not missing_id
    client.files.return_missing_id = missing_id
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    assert _state(client.state_path)["pending_operation"] == "training_upload"
    assert _state(client.state_path)["training_file_id"] is None
    client.files.fail_after_accept = False
    client.files.return_missing_id = False

    class LaterPage:
        data: list[Any] = []

        def has_next_page(self) -> bool:
            return True

        def __iter__(self):
            return iter(client.files.resources.values())

    client.files.list_override = LaterPage
    runtime.submit_or_resume_job(client, **options)
    assert client.files.created == 2 and client.jobs.created == 1


@pytest.mark.parametrize("matching_count", [0, 2])
def test_pending_upload_does_not_repeat_post_on_absent_or_duplicate_match(
    submission: Any, matching_count: int,
) -> None:
    client, options = submission
    client.files.fail_after_accept = True
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    item = next(iter(client.files.resources.values()))
    client.files.list_override = lambda: SimpleNamespace(data=[item] * matching_count)
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    assert client.files.created == 1 and client.jobs.created == 0
    assert _state(client.state_path)["pending_operation"] == "training_upload"


def test_second_upload_failure_preserves_first_id_and_never_repeats_pending_post(
    submission: Any,
) -> None:
    client, options = submission

    def fail_second(current: dict[str, Any]) -> None:
        client.files.fail_after_accept = True

    client.files.before_read = fail_second
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    current = _state(client.state_path)
    assert current["training_file_id"] == "file-offline-1"
    assert current["validation_file_id"] is None and current["pending_operation"] == "validation_upload"
    assert client.files.created == 2
    client.files.before_read = None
    client.files.fail_after_accept = False
    client.files.list_override = lambda: SimpleNamespace(data=[])
    with pytest.raises(RuntimeError, match="ambiguous"):
        runtime.submit_or_resume_job(client, **options)
    assert client.files.created == 2 and client.jobs.created == 0


@pytest.mark.parametrize("missing_id", [False, True])
def test_accepted_unidentified_job_is_never_automatically_reposted(submission: Any, missing_id: bool) -> None:
    client, options = submission
    client.jobs.fail_after_accept = not missing_id
    client.jobs.return_missing_id = missing_id
    with pytest.raises(RuntimeError) as caught:
        runtime.submit_or_resume_job(client, **options)
    assert TOKEN not in str(caught.value)
    current = _state(client.state_path)
    assert current["training_file_id"] and current["validation_file_id"]
    assert current["job_id"] is None and current["pending_operation"] == "job_create"
    with pytest.raises(RuntimeError, match="ambiguous"):
        runtime.submit_or_resume_job(client, **options)
    assert client.jobs.created == 1 and client.files.created == 2
    job = runtime.submit_or_resume_job(client, reconciled_job_id="job-offline", **options)
    assert job.id == _state(client.state_path)["job_id"] == "job-offline"
    assert _state(client.state_path)["pending_operation"] is None
    assert client.jobs.created == 1


@pytest.mark.parametrize("mismatch", ["model", "training_file", "method"])
def test_reconciled_job_must_match_saved_inputs_and_recipe(submission: Any, mismatch: str) -> None:
    client, options = submission
    client.jobs.fail_after_accept = True
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    job = client.jobs.resources["job-offline"]
    if mismatch == "method":
        job.method["reinforcement"]["hyperparameters"]["eval_samples"] = 6
    else:
        setattr(job, mismatch, "wrong")
    with pytest.raises(RuntimeError, match="match"):
        runtime.submit_or_resume_job(client, reconciled_job_id="job-offline", **options)
    assert _state(client.state_path)["pending_operation"] == "job_create"
    assert client.jobs.created == 1


def test_recovery_can_compare_service_masked_headers_without_persisting_secrets(submission: Any) -> None:
    client, options = submission
    client.jobs.return_missing_id = True
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    client.jobs.resources["job-offline"].method["reinforcement"]["tools"][0]["headers"] = {
        "Authorization": "***",
    }
    runtime.submit_or_resume_job(client, reconciled_job_id="job-offline", **options)
    assert TOKEN not in client.state_path.read_text(encoding="utf-8")
    assert client.jobs.created == 1


def test_saved_job_read_failure_keeps_id_and_never_posts_again(submission: Any) -> None:
    client, options = submission
    runtime.submit_or_resume_job(client, **options)
    client.jobs.raise_on_read = True
    with pytest.raises(RuntimeError) as caught:
        runtime.submit_or_resume_job(client, **options)
    assert TOKEN not in str(caught.value)
    assert _state(client.state_path)["job_id"] == "job-offline"
    assert client.files.created == 2 and client.jobs.created == 1


@pytest.mark.parametrize("mismatch", ["credential", "model", "endpoint", "data"])
def test_changed_equivalence_never_overwrites_existing_state(submission: Any, mismatch: str) -> None:
    client, options = submission
    runtime.submit_or_resume_job(client, **options)
    original = client.state_path.read_bytes()
    if mismatch == "credential":
        options["method"] = _notebook_recipe("rotated-offline-credential")[0]
    elif mismatch == "model":
        options["model"] = "different-model"
    elif mismatch == "endpoint":
        options["project_endpoint"] = "https://foundry.example.test/api/projects/different"
    else:
        options["expected_training_sha256"] = "0" * 64
    with pytest.raises((RuntimeError, ValueError)):
        runtime.submit_or_resume_job(client, **options)
    assert client.state_path.read_bytes() == original
    assert client.files.created == 2 and client.jobs.created == 1


@pytest.mark.parametrize("operation", ["training_upload", "validation_upload", "job_create"])
def test_pending_state_write_failure_prevents_the_corresponding_post(
    submission: Any, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    client, options = submission
    write = runtime._write_submission_state

    def fail(path: Path, current: dict[str, Any]) -> None:
        if current["pending_operation"] == operation:
            raise OSError("Offline injected disk failure")
        write(path, current)

    monkeypatch.setattr(runtime, "_write_submission_state", fail)
    with pytest.raises(OSError):
        runtime.submit_or_resume_job(client, **options)
    assert client.files.created == {"training_upload": 0, "validation_upload": 1, "job_create": 2}[operation]
    assert client.jobs.created == 0


@pytest.mark.parametrize("id_key", ["training_file_id", "validation_file_id", "job_id"])
def test_id_checkpoint_failure_leaves_pending_state_and_stops_further_work(
    submission: Any, monkeypatch: pytest.MonkeyPatch, id_key: str,
) -> None:
    client, options = submission
    write = runtime._write_submission_state

    def fail(path: Path, current: dict[str, Any]) -> None:
        if current[id_key] is not None:
            raise OSError("Offline injected checkpoint failure")
        write(path, current)

    monkeypatch.setattr(runtime, "_write_submission_state", fail)
    with pytest.raises(OSError):
        runtime.submit_or_resume_job(client, **options)
    assert _state(client.state_path)[id_key] is None
    assert _state(client.state_path)["pending_operation"] == {
        "training_file_id": "training_upload", "validation_file_id": "validation_upload", "job_id": "job_create",
    }[id_key]
    assert client.files.created == (1 if id_key == "training_file_id" else 2)
    assert client.jobs.created == (1 if id_key == "job_id" else 0)


def test_atomic_replace_failure_preserves_previous_state_and_removes_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "outputs" / "submission-state.json"
    runtime._write_submission_state(path, {"old": True})
    original = path.read_bytes()

    def fail(source: Any, target: Any) -> None:
        assert Path(source).parent == Path(target).parent
        raise OSError("Offline injected replace failure")

    monkeypatch.setattr(runtime.os, "replace", fail)
    with pytest.raises(OSError):
        runtime._write_submission_state(path, {"new": True})
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_single_writer_lock_prevents_submission(submission: Any) -> None:
    client, options = submission
    lock = client.state_path.with_suffix(".json.lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        runtime.submit_or_resume_job(client, **options)
    assert lock.exists() and client.files.created == 0 and client.jobs.created == 0


@pytest.mark.parametrize("bad_state", ["{", "[]", '{"equivalence": {"legacy": true}}'])
def test_corrupt_or_legacy_state_fails_closed_without_overwrite(submission: Any, bad_state: str) -> None:
    client, options = submission
    client.state_path.parent.mkdir(parents=True)
    client.state_path.write_text(bad_state, encoding="utf-8")
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    assert client.state_path.read_text(encoding="utf-8") == bad_state
    assert client.files.created == client.jobs.created == 0


def test_saved_file_bytes_are_verified_on_resume(submission: Any) -> None:
    client, options = submission
    client.files.read_failure = RuntimeError("Offline read interruption")
    with pytest.raises(RuntimeError):
        runtime.submit_or_resume_job(client, **options)
    client.files.read_failure = None
    client.files.contents["file-offline-1"] = b"wrong data\n"
    with pytest.raises(RuntimeError, match="content or byte count"):
        runtime.submit_or_resume_job(client, **options)
    assert client.files.created == 1 and client.jobs.created == 0


def test_event_redaction_keeps_metrics_but_removes_header_secrets() -> None:
    method, _ = _notebook_recipe()
    event = {
        "message": f"Tool headers Authorization: Bearer {TOKEN}; token={TOKEN}",
        "request": {"headers": {"Authorization": f"Bearer {TOKEN}"}},
        "nested": [{"authorization": TOKEN}], "reward": 1.0,
    }
    redacted = runtime.redact_tool_secrets(event, method)
    assert TOKEN not in json.dumps(redacted)
    assert redacted["reward"] == 1.0 and "<redacted>" in redacted["message"]
    assert event["nested"][0]["authorization"] == TOKEN


def test_event_redaction_handles_json_escaped_credentials() -> None:
    token = 'offline-quoted-"-and-backslash-\\-credential'
    method, _ = _notebook_recipe(token)
    event = {"message": json.dumps(method)}
    redacted = runtime.redact_tool_secrets(event, method)
    assert token not in redacted["message"]
    assert json.dumps(token)[1:-1] not in redacted["message"]


def test_notebook_event_wrapper_redacts_before_persistence_or_printing() -> None:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    tree = ast.parse(_source(cells[10]))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "event_dict"
    )
    method, _ = _notebook_recipe()
    namespace = {
        "Any": Any, "json": json, "METHOD": method,
        "redact_tool_secrets": runtime.redact_tool_secrets,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "offline-event-wrapper", "exec"), namespace)
    event = SimpleNamespace(to_dict=lambda: {"message": f"Untrusted service echo {TOKEN}"})
    assert TOKEN not in json.dumps(namespace["event_dict"](event))


def test_notebook_remains_output_free_and_uses_only_demo_local_runtime() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        assert not cell.get("outputs") and cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            ast.parse(_source(cell))
    code = "\n".join(_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert 'ROOT / "src" / "agentic_rft_runtime.py"' in code
    assert "live_validation" not in code and "rft_common" not in code
    assert "submit_or_resume_job(" in code and "max_retries=0" in code
    assert "validate_remote_tool(TOOL_SERVER_URL, headers=TOOL_HEADERS)" in code
    assert "STATE_PATH.write_text" not in code
    assert "files.create(" not in code and "jobs.create(" not in code


def test_uploaded_bytes_preserve_canonical_ten_ten_contract(submission: Any) -> None:
    client, options = submission
    runtime.submit_or_resume_job(client, **options)
    for file_id, expected_key in (
        ("file-offline-1", "expected_training_sha256"),
        ("file-offline-2", "expected_validation_sha256"),
    ):
        content = client.files.contents[file_id]
        assert len(content.splitlines()) == 10
        assert hashlib.sha256(content).hexdigest() == options[expected_key]