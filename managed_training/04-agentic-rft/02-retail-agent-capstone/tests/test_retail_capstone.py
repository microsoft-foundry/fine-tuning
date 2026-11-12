"""Offline capstone regressions using extracted notebook code and mock clients.

TestClient below is an envelope/authentication fixture, NOT a retail server or
an implementation of the historical return-policy labels. Offline preparation,
saved-state cells, and selected notebook definitions use temporary files/mocks;
service-stage cells are never executed and no cloud client is constructed.
The read-only Git-ignore check runs only when the parent runs this test suite.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import hmac
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "demo.ipynb"
TOKEN = "test-only-retail-bearer-not-a-real-credential"
OTHER_TOKEN = "test-only-rotated-retail-bearer-not-a-real-credential"
URL = "https://retail.example.invalid/tool"
TRACE = "retail-test-only-conversation"


@pytest.fixture(scope="module")
def retail():
    specification = importlib.util.spec_from_file_location(
        "capstone_retail_tools_under_test", ROOT / "src" / "retail_tools.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepared(retail, tmp_path_factory):
    specification = importlib.util.spec_from_file_location(
        "capstone_retail_curation_under_test", ROOT / "src" / "retail_curation.py"
    )
    assert specification is not None and specification.loader is not None
    curation = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "retail_tools", retail)
        specification.loader.exec_module(curation)
    root = tmp_path_factory.mktemp("retail-prepared-capstone")
    for directory in ("data", "configs", "graders"):
        shutil.copytree(ROOT / directory, root / directory)
    before = {path.relative_to(root): path.read_bytes()
              for directory in ("data", "configs", "graders")
              for path in (root / directory).rglob("*") if path.is_file()}
    paths, report = curation.prepare_active_datasets(root)
    assert all((root / relative).read_bytes() == content for relative, content in before.items())
    return SimpleNamespace(root=root, paths=paths, report=report, curation=curation)


@pytest.fixture(autouse=True)
def deny_real_requests(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline capstone tests must not contact a remote server.")

    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    # Notebook editor arrays may retain a final unterminated line.
    return "".join(source) if any(line.endswith("\n") for line in source) else "\n".join(source)


def _notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))


def _cell(marker: str) -> str:
    return next(
        _source(cell) for cell in _notebook()["cells"]
        if cell["cell_type"] == "code" and marker in _source(cell)
    )


def _functions(names: set[str], **values: Any) -> dict[str, Any]:
    source = "\n".join(
        _source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "code"
    )
    nodes = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    namespace = dict(values)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _rft_rows(prepared):
    for name in ("rft-train.jsonl", "rft-validation.jsonl"):
        with prepared.paths[name].open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _headers(retail, token: str = TOKEN):
    return retail.load_tool_headers({"FOUNDRY_TOOL_AUTH_TOKEN": token}, required=True)


def _envelope(payload, output):
    return {
        "type": "function_call_output", "id": payload["id"],
        "call_id": payload["call_id"], "output": json.dumps(output),
    }


def _response(payload, output=None, *, status=200, redirect=False):
    return SimpleNamespace(
        status_code=status, is_redirect=redirect,
        json=lambda: _envelope(payload, {"transport_fixture": True} if output is None else output),
    )


def _state_namespace(tmp_path: Path, retail, token: str = TOKEN, *, prepared_namespace=None):
    headers = _headers(retail, token)
    namespace = {
        "OUTPUTS": tmp_path / "outputs",
        "STATE_PATH": tmp_path / "outputs" / "notebook-state.json",
        "SFT_ENDPOINT": "https://sft.example.invalid/api/projects/test",
        "RFT_ENDPOINT": "https://rft.example.invalid/api/projects/test",
        "SFT_MODEL": "sft-model", "RFT_MODEL": "rft-model",
        "TRAINING_TYPE": "globalStandard", "validated_inputs": {"outputs/prepared-data/fixture": "test-hash"},
        "sft_method": {"type": "supervised"},
        "rft_method": {
            "type": "reinforcement",
            "reinforcement": {
                "tools": retail.build_remote_tools(ROOT / "configs" / "rft-tools.json", URL, headers),
                "hyperparameters": {"n_epochs": 1}, "grader": {"model": "o3-mini"},
            },
        },
        "json": json, "uuid": uuid, "hashlib": hashlib, "io": io,
    }
    if prepared_namespace is not None:
        namespace.update({name: prepared_namespace[name] for name in (
            "ROOT", "validated_inputs", "checked_prepared_path", "checked_prepared_bytes",
        )})
    exec(compile(_cell("def save_state():"), "retail-local-state", "exec"), namespace)
    return namespace


@pytest.fixture
def notebook_upload(tmp_path, retail, prepared):
    """Extract the real preparation and state/upload cells into isolated copies."""
    shutil.copytree(prepared.root, tmp_path, dirs_exist_ok=True)
    preparation = {
        "ROOT": tmp_path, "OUTPUTS": tmp_path / "outputs", "hashlib": hashlib,
        "prepare_active_datasets": prepared.curation.prepare_active_datasets,
        "validated_prepared_paths": prepared.curation.validated_prepared_paths,
        "validate_local_tool_contract": retail.validate_local_tool_contract,
    }
    exec(compile(_cell("prepared_paths, preparation_report ="), str(NOTEBOOK), "exec"), preparation)
    namespace = _state_namespace(tmp_path, retail, prepared_namespace=preparation)
    waits = []
    namespace["wait_for_file"] = lambda client, identifier, **options: waits.append((client, identifier, options))
    return SimpleNamespace(root=tmp_path, preparation=preparation, namespace=namespace, waits=waits)


@pytest.mark.parametrize("name,label", [
    ("sft-train.jsonl", "sft_train"), ("sft-validation.jsonl", "sft_validation"),
    ("rft-train.jsonl", "rft_train"), ("rft-validation.jsonl", "rft_validation"),
])
def test_extracted_notebook_upload_submits_exact_curated_bytes_and_reuses_bound_file(notebook_upload, name, label):
    namespace = notebook_upload.namespace
    root = notebook_upload.root
    path = notebook_upload.preparation["prepared_paths"][name]
    expected = path.read_bytes()
    source_bytes = {source.name: source.read_bytes() for source in (root / "data").iterdir() if source.is_file()}
    digest = hashlib.sha256(expected).hexdigest()
    descriptor = notebook_upload.preparation["preparation_report"]["files"][f"data/{name}"]["output"]
    assert digest == descriptor["raw_sha256"]
    assert namespace["validated_inputs"][path.relative_to(root).as_posix()] == digest
    if name.startswith("sft-"):
        assert expected != source_bytes[name].replace(b"\r\n", b"\n")
    submitted = []

    def create(*, file, purpose):
        filename, stream, content_type = file
        pending = namespace["state"]["files"][label]
        assert pending["pending"] is True and pending["input_sha256"] == digest
        assert pending["input_path"] == path.relative_to(root).as_posix()
        assert filename == pending["filename"] and filename.endswith(f"-{name}")
        assert purpose == "fine-tune" and content_type == "application/jsonl"
        assert isinstance(stream, io.BytesIO)
        payload = stream.read()
        assert payload == expected and hashlib.sha256(payload).hexdigest() == digest
        submitted.append((payload, stream))
        return SimpleNamespace(id="file-test-prepared-upload")

    client = SimpleNamespace(files=SimpleNamespace(create=create))
    assert namespace["upload_once"](client, label, name) == "file-test-prepared-upload"
    assert namespace["upload_once"](client, label, name) == "file-test-prepared-upload"
    assert len(submitted) == 1 and submitted[0][1].closed
    assert path.read_bytes() == expected
    assert len(notebook_upload.waits) == 2
    assert all(identifier == "file-test-prepared-upload" for _, identifier, _ in notebook_upload.waits)
    assert namespace["state"]["files"][label] == {
        "id": "file-test-prepared-upload", "input_path": path.relative_to(root).as_posix(), "input_sha256": digest,
    }
    assert {source.name: source.read_bytes() for source in (root / "data").iterdir() if source.is_file()} == source_bytes


@pytest.mark.parametrize("name", [
    "sft-train.jsonl", "sft-validation.jsonl", "next-tool-evaluation.jsonl",
    "rft-train.jsonl", "rft-validation.jsonl", "manifest.json",
])
@pytest.mark.parametrize("existing_upload", [False, True])
def test_extracted_notebook_upload_blocks_tamper_before_new_or_reused_post(notebook_upload, name, existing_upload):
    namespace = notebook_upload.namespace
    posts = []

    def create(**kwargs):
        posts.append(kwargs["file"][1].read())
        return SimpleNamespace(id="file-test-before-tamper")

    client = SimpleNamespace(files=SimpleNamespace(create=create))
    if existing_upload:
        assert namespace["upload_once"](client, "sft_train", "sft-train.jsonl") == "file-test-before-tamper"
    state_before = copy.deepcopy(namespace["state"])
    saved_before = namespace["STATE_PATH"].read_bytes()
    path = notebook_upload.root / "outputs" / "prepared-data" / name
    corrupted = path.read_bytes() + b" "
    path.write_bytes(corrupted)
    # Check both reuse and a fresh label: neither may bypass the real helpers.
    for label in ("sft_train", "sft_train_new"):
        with pytest.raises(ValueError, match="Prepared"):
            namespace["upload_once"](client, label, "sft-train.jsonl")
    assert len(posts) == int(existing_upload) and len(notebook_upload.waits) == int(existing_upload)
    assert namespace["state"] == state_before
    assert namespace["STATE_PATH"].read_bytes() == saved_before
    assert path.read_bytes() == corrupted


def test_extracted_notebook_upload_uses_checked_snapshot_not_a_later_path_reopen(notebook_upload):
    namespace = notebook_upload.namespace
    path = notebook_upload.preparation["prepared_paths"]["sft-train.jsonl"]
    expected = path.read_bytes()
    assert expected != (notebook_upload.root / "data" / path.name).read_bytes().replace(b"\r\n", b"\n")
    submitted = []

    def create(*, file, purpose):
        # Simulate path mutation after the helpers returned but before the SDK
        # consumes the stream. The immutable checked snapshot must be sent.
        path.write_bytes(b"{changed-after-validation}\n")
        submitted.append(file[1].read())
        assert submitted[-1] == expected and purpose == "fine-tune"
        return SimpleNamespace(id="file-test-snapshot")

    client = SimpleNamespace(files=SimpleNamespace(create=create))
    assert namespace["upload_once"](client, "sft_train", path.name) == "file-test-snapshot"
    assert submitted == [expected]
    assert namespace["state"]["files"]["sft_train"]["input_sha256"] == hashlib.sha256(expected).hexdigest()
    with pytest.raises(ValueError, match="Prepared"):
        namespace["upload_once"](client, "sft_validation", "sft-validation.jsonl")
    assert submitted == [expected]


def test_every_configured_and_dataset_tool_has_a_published_operation(retail, prepared):
    operations = retail.validate_local_tool_contract(prepared.root, dataset_paths=prepared.paths)
    configured = json.loads((ROOT / "configs" / "rft-tools.json").read_text(encoding="utf-8"))
    assert len(operations) == 17
    assert set(operations) == {tool["name"] for tool in configured}
    rows = list(_rft_rows(prepared))
    assert len(rows) == 20
    assert all(any(tool["function"]["name"] == "policy_verify_return" for tool in row["tools"]) for row in rows)
    assert all(row["reference_policy_args"] for row in rows)
    assert sum("policy_verify_return" in row["reference_tool_calls"] for row in rows) == 19
    assert sum("list_user_orders" in row["reference_tool_calls"] for row in rows) == 14
    for row in rows:
        for function in (tool["function"] for tool in row["tools"]):
            if function["name"] in {"list_user_orders", "policy_verify_return"}:
                actual = operations[function["name"]]["requestBody"]["content"]["application/json"]["schema"]
                assert actual == function["parameters"]


def test_new_contract_keeps_strict_reason_and_customer_auth_semantics(retail, prepared):
    operations = retail.validate_local_tool_contract(prepared.root, dataset_paths=prepared.paths)
    schema = operations["policy_verify_return"]["requestBody"]["content"]["application/json"]["schema"]
    assert schema["properties"]["reason"]["enum"] == [
        "unwanted", "wrong_item", "size_issue", "defective", "other",
    ]
    assert schema["required"] == ["order_id", "item_ids", "reason"]
    assert "ownership" in operations["policy_verify_return"]["description"]
    assert "explicit customer confirmation" in operations["policy_verify_return"]["description"]
    assert "original payment method or an existing gift card" in operations["policy_verify_return"]["description"]
    assert "user_id as authentication" in operations["list_user_orders"]["description"]
    specification = json.loads((ROOT / "configs" / "retail-openapi-with-policy.json").read_text(encoding="utf-8"))
    assert specification["security"] == [{"bearerAuth": []}]
    assert specification["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"


@pytest.mark.parametrize("change", ["missing", "parameters", "dataset_reference"])
def test_contract_validator_rejects_missing_or_drifted_operations(tmp_path, retail, prepared, change):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "data").mkdir()
    row = next(_rft_rows(prepared))
    path = tmp_path / "configs" / "retail-openapi-with-policy.json"
    specification = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        del specification["paths"]["/policy_verify_return"]
    elif change == "parameters":
        schema = specification["paths"]["/policy_verify_return"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        schema["properties"]["reason"]["enum"].append("exchange")
    else:
        row["messages"].append({
            "role": "assistant", "tool_calls": [{"function": {"name": "undeclared_retail_operation"}}],
        })
    path.write_text(json.dumps(specification), encoding="utf-8")
    directory = tmp_path / "outputs" / "prepared-data"
    directory.mkdir(parents=True)
    paths = {name: directory / name for name in retail.ACTIVE_RETAIL_DATASETS}
    for name, active in paths.items():
        value = row if name == "rft-train.jsonl" else {"messages": [], "tools": []}
        active.write_bytes((json.dumps(value) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="Missing retail operation|Retail parameter drift"):
        retail.validate_local_tool_contract(tmp_path, dataset_paths=paths)


def test_active_api_lineage_and_original_operations_are_independently_pinned():
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        canonical = (ROOT / entry["path"]).read_bytes().replace(b"\r\n", b"\n")
        assert len(canonical) == entry["bytes"], entry["path"]
        assert hashlib.sha256(canonical).hexdigest() == entry["sha256"], entry["path"]
        if "records" in entry:
            assert len([line for line in canonical.splitlines() if line.strip()]) == entry["records"]
    entry = next(item for item in manifest["files"] if item["path"] == "configs/retail-openapi-with-policy.json")
    original = json.loads((ROOT / entry["path"]).read_text(encoding="utf-8"))
    del original["security"]
    del original["components"]
    del original["paths"]["/list_user_orders"]
    del original["paths"]["/policy_verify_return"]
    reconstructed = json.dumps(original, ensure_ascii=False, indent=2).encode("utf-8")
    assert len(reconstructed) == entry["source_bytes"] == 32503
    assert hashlib.sha256(reconstructed).hexdigest() == entry["source_sha256"]


def test_live_headers_are_injected_into_all_tools_without_rewriting_sources(retail):
    path = ROOT / "configs" / "rft-tools.json"
    before = path.read_bytes()
    original = json.loads(before)
    assert all(tool["headers"] == {} for tool in original)
    headers = _headers(retail)
    live = retail.build_remote_tools(path, URL, headers)
    assert len(live) == 17
    assert all(tool["server_url"] == URL and tool["headers"] == headers for tool in live)
    live[0]["headers"]["Authorization"] = "Bearer test-only-changed"
    assert all(tool["headers"] == headers for tool in live[1:])
    assert path.read_bytes() == before
    assert TOKEN not in (ROOT / "data" / "manifest.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("token", ["bad token", "bad\nheader", "bad\ttoken", "non-ascii-\u00e9"])
def test_bearer_setting_rejects_header_injection_without_echoing_secret(retail, token):
    with pytest.raises(ValueError, match="one printable ASCII credential") as captured:
        retail.load_tool_headers({"FOUNDRY_TOOL_AUTH_TOKEN": token}, required=True)
    assert token not in str(captured.value)


def test_local_preparation_can_be_offline_but_live_auth_is_required(retail):
    assert retail.load_tool_headers({}) == {}
    with pytest.raises(ValueError, match="Set FOUNDRY_TOOL_AUTH_TOKEN"):
        retail.load_tool_headers({}, required=True)


@pytest.mark.parametrize("url", [
    "http://retail.example.invalid/tool", "https://user:secret@retail.example.invalid/tool",
    "https://retail.example.invalid/tool?key=test", "https://retail.example.invalid/tool#secret",
])
def test_tool_urls_reject_credential_channels_before_transport(retail, url):
    with pytest.raises(ValueError, match="HTTPS tool URL"):
        retail.call_read_only_tool(url, "calculate", '{"expression":"2+2"}', "call-test",
            trace_id=TRACE, headers=_headers(retail), sandbox=True)


def test_testclient_preflight_uses_same_bearer_for_all_read_only_probes(retail, prepared):
    database = json.loads((ROOT / "data" / "retail-database.json").read_text(encoding="utf-8"))
    first = next(_rft_rows(prepared))["reference_policy_args"][0]
    user_id = database["orders"][first["order_id"]]["user_id"]
    app = FastAPI()
    seen = []
    authenticated_conversations = set()

    @app.post("/tool")
    def envelope_fixture(payload: dict, authorization: str | None = Header(default=None)):
        if not hmac.compare_digest(authorization or "", f"Bearer {TOKEN}"):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if payload["name"] == "find_user_id_by_email":
            authenticated_conversations.add(payload["trace_id"])
        elif payload["trace_id"] not in authenticated_conversations:
            raise HTTPException(status_code=403, detail="No fixture customer in this conversation")
        seen.append(payload)
        # Deliberately not a policy outcome: this fixture proves transport only.
        output = {"user_id": user_id} if payload["name"] == "find_user_id_by_email" else {"transport_fixture": True}
        return _envelope(payload, output)

    with TestClient(app) as client:
        def post(url, **kwargs):
            assert url == URL
            assert kwargs["allow_redirects"] is False
            assert kwargs["timeout"] == 60
            assert kwargs["headers"] == _headers(retail)
            return client.post("/tool", json=kwargs["json"], headers=kwargs["headers"], follow_redirects=False)

        retail.preflight_retail_tools(prepared.root, URL, dataset_paths=prepared.paths,
            headers=_headers(retail), sandbox=True, post=post)
        retail.preflight_retail_tools(prepared.root, URL, dataset_paths=prepared.paths,
            headers=_headers(retail), sandbox=True, post=post)
        unauthorized = client.post("/tool", json={}, headers={"Authorization": "Bearer test-only-wrong"})
        assert unauthorized.status_code == 401
    assert [call["name"] for call in seen] == [
        "find_user_id_by_email", "list_user_orders", "get_order_details", "policy_verify_return",
    ] * 2
    assert len({call["trace_id"] for call in seen[:4]}) == 1
    assert len({call["trace_id"] for call in seen[4:]}) == 1
    assert seen[0]["trace_id"] != seen[4]["trace_id"]
    assert len({call["call_id"] for call in seen}) == 8
    assert all(call["type"] == "function_call" and call["id"] == call["call_id"] for call in seen)
    assert json.loads(seen[-1]["arguments"]) == first


@pytest.mark.parametrize("bad_result", ["wrong_identity", "tool_error", "empty", "not_json"])
def test_preflight_rejects_unsupported_or_wrong_fixture_results(retail, prepared, bad_result):
    seen = []
    database = json.loads((ROOT / "data" / "retail-database.json").read_text(encoding="utf-8"))
    order_id = next(_rft_rows(prepared))["reference_policy_args"][0]["order_id"]
    user_id = database["orders"][order_id]["user_id"]

    def post(_url, **kwargs):
        payload = kwargs["json"]
        seen.append(payload["name"])
        if payload["name"] == "find_user_id_by_email":
            return _response(payload, {"user_id": "another-user" if bad_result == "wrong_identity" else user_id})
        output = {"error": "Not implemented"} if bad_result == "tool_error" else [] if bad_result == "empty" else "not-json"
        return _response(payload, output)

    with pytest.raises(retail.RetailToolError, match="preflight"):
        retail.preflight_retail_tools(prepared.root, URL, dataset_paths=prepared.paths,
            headers=_headers(retail), sandbox=True, post=post)
    assert len(seen) <= 2


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 404, 500])
def test_auth_errors_redirects_and_http_failures_are_safe_and_not_followed(retail, status):
    seen = []

    def post(_url, **kwargs):
        assert kwargs["allow_redirects"] is False
        seen.append(kwargs)
        return _response(kwargs["json"], {"echoed": TOKEN}, status=status, redirect=300 <= status < 400)

    with pytest.raises(retail.RetailToolError) as captured:
        retail.call_read_only_tool(URL, "calculate", '{"expression":"2+2"}', "call-test",
            trace_id=TRACE, headers=_headers(retail), sandbox=True, post=post)
    assert captured.value.status_code == status
    assert TOKEN not in "".join(traceback.format_exception(captured.value))
    assert len(seen) == 1


@pytest.mark.parametrize("mode", ["transport_exception", "wrong_type", "wrong_id", "missing_output", "echo_secret"])
def test_invalid_or_secret_echoing_tool_responses_do_not_leak(retail, mode):
    def post(_url, **kwargs):
        if mode == "transport_exception":
            raise requests.ConnectionError(f"Request carried Bearer {TOKEN}")
        payload = kwargs["json"]
        body = _envelope(payload, {"echo": TOKEN} if mode == "echo_secret" else {"result": 4})
        if mode == "wrong_type":
            body["type"] = "other"
        elif mode == "wrong_id":
            body["call_id"] = "different-call"
        elif mode == "missing_output":
            del body["output"]
        return SimpleNamespace(status_code=200, is_redirect=False, json=lambda: body)

    with pytest.raises(retail.RetailToolError) as captured:
        retail.call_read_only_tool(URL, "calculate", '{"expression":"2+2"}', "call-test",
            trace_id=TRACE, headers=_headers(retail), sandbox=True, post=post)
    assert TOKEN not in "".join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize("name,sandbox,headers", [
    ("cancel_pending_order", True, {"Authorization": f"Bearer {TOKEN}"}),
    ("unknown_retail_tool", True, {"Authorization": f"Bearer {TOKEN}"}),
    ("calculate", False, {"Authorization": f"Bearer {TOKEN}"}),
    ("calculate", True, {}),
])
def test_blocked_mutations_sandbox_and_missing_auth_never_contact_server(retail, name, sandbox, headers):
    with pytest.raises((ValueError, RuntimeError)):
        retail.call_read_only_tool(URL, name, "{}", "call-test",
            trace_id=TRACE, headers=headers, sandbox=sandbox)


@pytest.mark.parametrize("trace_id", [None, 1, "", " ", "bad\ntrace"])
def test_conversation_trace_must_be_explicit_and_valid_before_transport(retail, trace_id):
    with pytest.raises(ValueError, match="conversation trace"):
        retail.call_read_only_tool(URL, "calculate", "{}", "call-test",
            trace_id=trace_id, headers=_headers(retail), sandbox=True)


def test_notebook_comparison_sends_same_bearer_as_remote_training(retail, monkeypatch):
    headers = _headers(retail)
    tools = retail.build_remote_tools(ROOT / "configs" / "rft-tools.json", URL, headers)
    seen = []

    def post(url, **kwargs):
        seen.append((url, kwargs))
        return _response(kwargs["json"], {"result": 4})

    monkeypatch.setattr(requests, "post", post)
    namespace = _functions({"execute_remote_tool"},
        READ_ONLY_TOOLS=retail.READ_ONLY_TOOLS, env_flag=lambda _name: True,
        TOOL_SERVER_URL=URL, TOOL_HEADERS=headers, call_read_only_tool=retail.call_read_only_tool,
        retry_call=lambda operation, **_kwargs: operation())
    call = SimpleNamespace(id="call-test", function=SimpleNamespace(name="calculate", arguments='{"expression":"2+2"}'))
    assert json.loads(namespace["execute_remote_tool"](call, trace_id=TRACE)) == {"result": 4}
    assert len(seen) == 1
    assert seen[0][0] == URL
    assert all(tool["headers"] == seen[0][1]["headers"] == headers for tool in tools)
    assert seen[0][1]["allow_redirects"] is False
    assert seen[0][1]["json"]["trace_id"] == TRACE


def test_read_retries_preserve_the_same_conversation_and_call(retail, monkeypatch):
    seen = []

    def post(_url, **kwargs):
        seen.append(kwargs["json"])
        if len(seen) == 1:
            raise requests.ConnectionError("test-only interrupted read")
        return _response(kwargs["json"], {"result": 4})

    def retry(operation, **_kwargs):
        with pytest.raises(retail.RetailToolError):
            operation()
        return operation()

    monkeypatch.setattr(requests, "post", post)
    namespace = _functions({"execute_remote_tool"},
        READ_ONLY_TOOLS=retail.READ_ONLY_TOOLS, env_flag=lambda _name: True,
        TOOL_SERVER_URL=URL, TOOL_HEADERS=_headers(retail),
        call_read_only_tool=retail.call_read_only_tool, retry_call=retry)
    call = SimpleNamespace(id="call-test", function=SimpleNamespace(name="calculate", arguments="{}"))
    assert json.loads(namespace["execute_remote_tool"](call, trace_id=TRACE)) == {"result": 4}
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0]["trace_id"] == TRACE


def test_model_tool_turns_share_customer_context_without_reusing_another_conversation(retail, monkeypatch):
    seen = []
    customers = set()

    def post(_url, **kwargs):
        payload = kwargs["json"]
        trace = payload["trace_id"]
        if payload["name"] == "find_user_id_by_email":
            customers.add(trace)
        else:
            assert trace in customers, "Customer lookup context was lost between model turns"
        seen.append(payload)
        return _response(payload, {"fixture": True})

    def tool_call(name, number):
        call_id = f"call-test-{number}"
        function = SimpleNamespace(name=name, arguments="{}")
        return SimpleNamespace(id=call_id, function=function,
            model_dump=lambda **_kwargs: {"id": call_id, "type": "function",
                "function": {"name": name, "arguments": "{}"}})

    monkeypatch.setattr(requests, "post", post)
    namespace = _functions({"invoke_rft", "execute_remote_tool"},
        uuid=uuid, READ_ONLY_TOOLS=retail.READ_ONLY_TOOLS,
        TOOL_SERVER_URL=URL, TOOL_HEADERS=_headers(retail),
        call_read_only_tool=retail.call_read_only_tool, env_flag=lambda _name: True,
        COMPARISON_PACING_SECONDS=0, COMPARISON_RETRY_POLICY=None,
        retry_call=lambda operation, **_kwargs: operation())
    row = {"messages": [{"role": "user", "content": "look up my order"}], "tools": []}
    for _ in range(2):
        turns = iter([
            SimpleNamespace(content=None, tool_calls=[tool_call("find_user_id_by_email", 0)]),
            SimpleNamespace(content=None, tool_calls=[tool_call("list_user_orders", 1),
                tool_call("get_order_details", 2), tool_call("policy_verify_return", 3)]),
            SimpleNamespace(content="done", tool_calls=[]),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(choices=[SimpleNamespace(message=next(turns))]))))
        outcome = namespace["invoke_rft"](client, "test-deployment", row)
        assert outcome["output"] == "done"
        assert len(outcome["executed_tools"]) == 4
    assert len({payload["trace_id"] for payload in seen[:4]}) == 1
    assert len({payload["trace_id"] for payload in seen[4:]}) == 1
    assert seen[0]["trace_id"] != seen[4]["trace_id"]
    assert len(row["messages"]) == 1


def test_blocked_model_batch_executes_nothing_and_skips_policy_grader(retail):
    executed = []
    policy_calls = []
    calls = [SimpleNamespace(function=SimpleNamespace(name=name)) for name in ("calculate", "cancel_pending_order")]
    message = SimpleNamespace(content=None, tool_calls=calls)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)]))))
    namespace = _functions({"invoke_rft", "score_rft_outcome"},
        uuid=uuid, READ_ONLY_TOOLS=retail.READ_ONLY_TOOLS, COMPARISON_PACING_SECONDS=0,
        COMPARISON_RETRY_POLICY=None, retry_call=lambda operation, **_kwargs: operation(),
        execute_remote_tool=lambda call, **_kwargs: executed.append(call),
        policy_score=lambda *args: policy_calls.append(args))
    row = {"messages": [], "tools": [], "reference_tool_calls": ["calculate"]}
    outcome = namespace["invoke_rft"](client, "test-deployment", row)
    assert outcome["blocked_tools"] == ["cancel_pending_order"]
    assert outcome["executed_tools"] == []
    result = namespace["score_rft_outcome"](row, outcome)
    assert result["score"] == 0.0 and result["status"] == "blocked"
    assert executed == [] and policy_calls == []


def test_saved_state_has_only_redacted_methods_and_auth_fingerprint(tmp_path, retail):
    namespace = _state_namespace(tmp_path, retail)
    live = copy.deepcopy(namespace["rft_method"])
    namespace["state"]["service_echo"] = f"Bearer {TOKEN}"
    namespace["save_state"]()
    text = namespace["STATE_PATH"].read_text(encoding="utf-8")
    assert TOKEN not in text
    assert "<redacted>" in text
    assert namespace["rft_method"] == live
    saved = json.loads(text)
    assert len(saved["context"]["tool_auth_fingerprint"]) == 64
    for tool in saved["context"]["methods"][1]["reinforcement"]["tools"]:
        assert tool["headers"] == {"Authorization": "<redacted>"}
    assert not namespace["STATE_PATH"].with_suffix(".tmp").exists()
    namespace["state"]["jobs"]["rft"] = {"id": "job-test", "status": "succeeded", "fine_tuned_model": "model-test"}
    namespace["save_state"]()
    exec(compile(_cell("def save_state():"), "retail-local-state", "exec"), namespace)
    assert namespace["state"]["jobs"]["rft"]["fine_tuned_model"] == "model-test"
    assert "rft_client" not in namespace


@pytest.mark.parametrize("change", ["rotate", "legacy"])
def test_changed_credentials_and_old_state_fail_closed_without_rewriting(tmp_path, retail, change):
    namespace = _state_namespace(tmp_path, retail)
    if change == "rotate":
        for tool in namespace["rft_method"]["reinforcement"]["tools"]:
            tool["headers"] = _headers(retail, OTHER_TOKEN)
    else:
        saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
        del saved["context"]["schema_version"]
        namespace["STATE_PATH"].write_text(json.dumps(saved), encoding="utf-8")
    before = namespace["STATE_PATH"].read_bytes()
    with pytest.raises(RuntimeError, match="Reconcile it explicitly"):
        exec(compile(_cell("def save_state():"), "retail-local-state", "exec"), namespace)
    assert namespace["STATE_PATH"].read_bytes() == before


def test_submission_failure_keeps_pending_marker_and_never_echoes_or_reposts_secret(tmp_path, retail):
    namespace = _state_namespace(tmp_path, retail)
    namespace["state"]["files"] = {"rft_train": {"id": "file-train"}, "rft_validation": {"id": "file-validation"}}
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(f"Server echoed Bearer {TOKEN}")

    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="outcome is uncertain") as captured:
        namespace["submit_once"](client, "rft", "rft-model", namespace["rft_method"])
    assert TOKEN not in "".join(traceback.format_exception(captured.value))
    assert captured.value.__context__ is None
    saved = json.loads(namespace["STATE_PATH"].read_text(encoding="utf-8"))
    assert saved["jobs"]["rft"]["pending"] is True
    assert TOKEN not in namespace["STATE_PATH"].read_text(encoding="utf-8")
    assert calls[0]["method"] == namespace["rft_method"]
    assert calls[0]["extra_body"] == {"trainingType": "globalStandard"}
    with pytest.raises(RuntimeError, match="Uncertain submission"):
        namespace["submit_once"](client, "rft", "rft-model", namespace["rft_method"])
    assert len(calls) == 1


def test_comparison_equivalence_invalidates_server_and_credentials_without_saving_token(tmp_path, retail):
    state = _state_namespace(tmp_path, retail)
    namespace = _functions({"tool_auth_fingerprint", "rft_tool_equivalence", "file_sha256",
        "load_comparison_results", "save_comparison_result"},
        json=json, hashlib=hashlib, rft_method=state["rft_method"], TOOL_SERVER_URL=URL,
        CONFIGS=ROOT / "configs")
    context = {"tool_equivalence": namespace["rft_tool_equivalence"]()}
    path = tmp_path / "outputs" / "rft-comparison-state.json"
    namespace["save_comparison_result"](path, context, "base", {"score": 0.5})
    assert namespace["load_comparison_results"](path, context) == {"base": {"score": 0.5}}
    assert TOKEN not in path.read_text(encoding="utf-8")
    assert context["tool_equivalence"]["server_url"] == URL
    assert len(context["tool_equivalence"]["auth_fingerprint"]) == 64
    assert context["tool_equivalence"]["trace_scope"] == "per-conversation-v1"
    legacy_context = copy.deepcopy(context)
    del legacy_context["tool_equivalence"]["trace_scope"]
    namespace["save_comparison_result"](path, legacy_context, "base", {"score": 0.5})
    assert namespace["load_comparison_results"](path, context) == {}
    for tool in namespace["rft_method"]["reinforcement"]["tools"]:
        tool["headers"] = _headers(retail, OTHER_TOKEN)
    rotated = {"tool_equivalence": namespace["rft_tool_equivalence"]()}
    assert namespace["load_comparison_results"](path, rotated) == {}
    namespace["TOOL_SERVER_URL"] = "https://other-retail.example.invalid/tool"
    moved = {"tool_equivalence": namespace["rft_tool_equivalence"]()}
    assert moved != rotated
    assert namespace["load_comparison_results"](path, moved) == {}


def test_warn_then_proceed_is_one_warning_without_an_extra_budget_gate(capsys):
    namespace = _functions({"warn_paid_once"}, _paid_warning_emitted=False)
    namespace["warn_paid_once"]()
    namespace["warn_paid_once"]()
    assert capsys.readouterr().out.splitlines() == [
        "This workflow uses paid cloud resources and may incur charges.",
    ]
    source = _cell("preflight_retail_tools(ROOT")
    assert source.index("warn_paid_once()") < source.index("preflight_retail_tools(ROOT") < source.index("upload_once(")
    assert "input(" not in source


def test_notebook_is_unexecuted_and_wires_secret_safe_contract_helpers():
    notebook = _notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
            ast.parse(_source(cell))
    setup = _cell("TOOL_HEADERS = load_tool_headers()")
    assert "from retail_tools import" in setup
    assert "from retail_curation import prepare_active_datasets, validated_prepared_paths" in setup
    preparation = _cell("prepared_paths, preparation_report =")
    assert "validate_local_tool_contract(ROOT, dataset_paths=prepared_paths)" in preparation
    assert "validated_prepared_paths(ROOT, preparation_report)" in preparation
    build = _cell("tools = build_remote_tools(")
    assert "TOOL_HEADERS" in build
    assert "print(tools" not in build and "print(rft_method" not in build
    rft = _cell("def execute_remote_tool(")
    assert "'tool_equivalence': rft_tool_equivalence()" in rft
    assert "headers=TOOL_HEADERS" in rft
    assert "preflight_retail_tools(ROOT" in rft
    assert "dataset_paths=prepared_paths" in rft
    for marker in ("sft_input_path =", "rft_input_path ="):
        comparison = _cell(marker)
        assert "checked_prepared_path(" in comparison and "checked_prepared_bytes(" in comparison
        assert "'preparation_manifest_sha256':" in comparison
        assert "= DATA /" not in comparison
    upload = _cell("def upload_once(")
    assert "payload = checked_prepared_bytes(name)" in upload and "io.BytesIO(payload)" in upload
    assert "'input_sha256': digest" in upload and "with path.open(" not in upload
    assert "DATA /" not in _cell("preflight_retail_tools(ROOT")


def test_git_ignores_all_runtime_outputs_but_not_source_or_tests():
    # Parent-only execution: no repo mutation, no temporary files, no Git config change.
    repo = ROOT.parents[2]
    prefix = ROOT.relative_to(repo).as_posix()
    ignored = [f"{prefix}/{path}" for path in (
        "outputs/notebook-state.json", "outputs/notebook-state.tmp",
        "outputs/deployment-state.json", "outputs/comparison-state.json",
        "outputs/sft-comparison-state.json", "outputs/rft-comparison-state.json",
        "outputs/rft-results.csv", "outputs/nested/customer-result.json", ".env",
        "outputs/prepared-data/manifest.json", "outputs/prepared-data/sft-train.jsonl",
        "outputs/prepared-data/next-tool-evaluation.jsonl",
        "data/curation-review.json",
    )]
    not_ignored = [f"{prefix}/{path}" for path in (
        ".env.template", "src/retail_tools.py", "src/retail_curation.py",
        "tests/test_retail_capstone.py", "tests/test_retail_trace_integrity.py",
    )]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"], cwd=repo,
        input=("\0".join(ignored + not_ignored) + "\0").encode("utf-8"),
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path.decode("utf-8") for path in result.stdout.split(b"\0") if path} == set(ignored)
    tracked = subprocess.run(
        ["git", "ls-files", "--", f"{prefix}/outputs/", f"{prefix}/.env"],
        cwd=repo, text=True, encoding="utf-8", capture_output=True, check=True,
    )
    assert not tracked.stdout.strip(), "Runtime state or populated environment files must not be tracked."
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/outputs/" in rules and "!.env.template" in rules