"""Demo-local tool authentication and crash-safe, single-job submission.

Only read operations may be repeated. A pending POST is never evidence that it
failed, even if a subsequent list request cannot find its result.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SUFFIX = "tool-calling-rft"
_ID_KEYS = ("training_file_id", "validation_file_id", "job_id")
_PENDING_OPERATIONS = {None, "training_upload", "validation_upload", "job_create"}
_STATE_KEYS = {"schema_version", "equivalence", "pending_operation", *_ID_KEYS}
_SENSITIVE_KEYS = {
    "authorization", "x-functions-key", "api-key", "api_key", "access_token", "token",
}


def _https_url(value: str, *, project: bool = False) -> str:
    selected = value.strip().rstrip("/") if project else value.strip()
    parsed = urlparse(selected)
    if (
        parsed.scheme != "https" or not parsed.netloc
        or parsed.username or parsed.password or parsed.query or parsed.fragment
    ):
        raise ValueError("Use an HTTPS endpoint without URL credentials, query, or fragment.")
    if project:
        parts = parsed.path.split("/")
        if len(parts) != 4 or parts[:3] != ["", "api", "projects"] or not parts[3].strip():
            raise ValueError("The project endpoint must include /api/projects/<project>.")
    return selected


def load_tool_headers(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read the same secret-backed setting used by the hosted catalog server."""
    values = os.environ if environ is None else environ
    token = values.get("FOUNDRY_TOOL_AUTH_TOKEN", "").strip()
    if not token:
        return {}
    if any(not 33 <= ord(character) <= 126 for character in token):
        raise ValueError("FOUNDRY_TOOL_AUTH_TOKEN must be one printable ASCII credential.")
    return {"Authorization": f"Bearer {token}"}


def _header_secrets(value: Any) -> set[str]:
    secrets: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() == "headers" and isinstance(item, Mapping):
                for header in item.values():
                    if isinstance(header, str) and header:
                        secrets.add(header)
                        if header.lower().startswith("bearer ") and header[7:].strip():
                            secrets.add(header[7:].strip())
            secrets.update(_header_secrets(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            secrets.update(_header_secrets(item))
    return secrets


def redact_tool_secrets(value: Any, method: Mapping[str, Any]) -> Any:
    """Redact header values and their occurrences in service events/error text."""
    raw_secrets = _header_secrets(method)
    secrets = sorted(
        raw_secrets | {json.dumps(secret)[1:-1] for secret in raw_secrets},
        key=len, reverse=True,
    )

    def text(item: str) -> str:
        for secret in secrets:
            item = item.replace(secret, "<redacted>")
        return item

    def scrub(item: Any) -> Any:
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                name = str(key)
                if name.casefold() == "headers":
                    result[text(name)] = (
                        {text(str(header)).casefold(): "<redacted>" for header in child}
                        if isinstance(child, Mapping) else "<redacted>"
                    )
                elif name.casefold() in _SENSITIVE_KEYS:
                    result[text(name)] = "<redacted>"
                else:
                    result[text(name)] = scrub(child)
            return result
        if isinstance(item, (list, tuple)):
            return [scrub(child) for child in item]
        return text(item) if isinstance(item, str) else item

    return scrub(value)


def validate_remote_tool(
    tool_server_url: str,
    *,
    headers: Mapping[str, str],
    post: Callable[..., Any] | None = None,
) -> None:
    """Check the authenticated envelope without following credential-bearing redirects.

    ``post`` is injectable for an in-process TestClient; the default is requests.
    Response bodies, request headers, and underlying exception text are not logged.
    """
    url = _https_url(tool_server_url)
    request = {
        "type": "function_call",
        "id": "fc_contract_check",
        "call_id": "call_contract_check",
        "name": "search_catalog",
        "arguments": json.dumps({"query": "Urban Puffer", "top_k": 1}),
        "trace_id": "cookbook-contract-check",
        "item": {},
    }
    try:
        if post is None:
            import requests

            post = requests.post
        response = post(
            url, json=request, headers=dict(headers), timeout=60, allow_redirects=False,
        )
        if response.status_code != 200:
            raise ValueError("The tool did not return HTTP 200.")
        payload = response.json()
        if (
            payload["type"] != "function_call_output"
            or payload["id"] != request["id"]
            or payload["call_id"] != request["call_id"]
            or json.loads(payload["output"])["items"][0] != {
                "sku": "JKT-URB-009", "name": "Urban Puffer", "price": 149.5,
            }
        ):
            raise ValueError("The tool response does not match the catalog contract.")
    except Exception:
        raise RuntimeError(
            "Remote tool validation failed; verify the endpoint, authentication, and contract."
        ) from None


def _submission_state_path(path: str | Path) -> Path:
    """Resolve aliases inside this demo's outputs, never adopt a redirected root."""
    try:
        root = Path(__file__).resolve().parents[1] / "outputs"
        resolved = Path(path).resolve()
        redirected_root = root.resolve() != root
    except (OSError, RuntimeError):
        raise ValueError("Cannot resolve the demo's outputs boundary.") from None
    if (
        redirected_root or resolved.name != "submission-state.json"
        or not resolved.is_relative_to(root)
    ):
        raise ValueError("Keep runtime state within the demo's outputs/submission-state.json.")
    return resolved


def _write_submission_state(path: Path, state: dict[str, Any]) -> None:
    """Flush a same-directory temporary file, then atomically replace state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _submission_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(
            "Submission is locked. Confirm no writer is active and reconcile local/remote "
            "state before removing a stale submission-state.json.lock."
        ) from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def _field(resource: Any, key: str) -> Any:
    return resource.get(key) if isinstance(resource, Mapping) else getattr(resource, key, None)


def _resource_id(resource: Any) -> str:
    value = _field(resource, "id")
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RuntimeError("Response has no valid resource ID; reconcile the pending POST.")
    return value


def _read(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except Exception:
        raise RuntimeError(
            "Read-only resource verification failed; retain state and reconcile in Foundry."
        ) from None


def _page_items(page: Any) -> list[Any]:
    return list(page) if hasattr(page, "has_next_page") else list(getattr(page, "data", page))


def _matches_recipe(actual: Any, expected: Any) -> bool:
    if hasattr(actual, "model_dump"):
        actual = actual.model_dump(exclude_none=True)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_recipe(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _matches_recipe(item, value) for item, value in zip(actual, expected)
        )
    return actual == expected


def _load_state(path: Path, equivalence: dict[str, Any]) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        raise RuntimeError("Saved state is unreadable; retain it for reconciliation.") from None
    if (
        not isinstance(state, dict) or set(state) != _STATE_KEYS
        or state.get("schema_version") != 1 or state.get("equivalence") != equivalence
    ):
        raise RuntimeError(
            "Saved state does not match this project/recipe/data/credential or uses an older "
            "format. Retain and reconcile it; do not overwrite it to rerun."
        )
    if any(
        state[key] is not None and (
            not isinstance(state[key], str) or not state[key].strip()
            or state[key] != state[key].strip()
        ) for key in _ID_KEYS
    ):
        raise RuntimeError("Saved state contains an invalid resource ID.")
    pending = state["pending_operation"]
    if not isinstance(pending, (str, type(None))) or pending not in _PENDING_OPERATIONS:
        raise RuntimeError("Saved state contains an unknown pending operation.")
    if (
        (state["validation_file_id"] and not state["training_file_id"])
        or (state["job_id"] and not all(state[key] for key in _ID_KEYS[:2]))
        or (pending == "training_upload" and any(state[key] for key in _ID_KEYS))
        or (pending == "validation_upload" and (
            not state["training_file_id"] or state["validation_file_id"] or state["job_id"]
        ))
        or (pending == "job_create" and (
            not all(state[key] for key in _ID_KEYS[:2]) or state["job_id"]
        ))
    ):
        raise RuntimeError("Saved state contains inconsistent submission progress.")
    return state


def _verify_file(
    client: Any,
    file_id: str,
    contract: dict[str, Any],
    content: bytes,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        uploaded = _read(lambda: client.files.retrieve(file_id))
        if (
            _resource_id(uploaded) != file_id
            or _field(uploaded, "filename") != contract["filename"]
            or _field(uploaded, "purpose") != "fine-tune"
        ):
            raise RuntimeError("Saved/reconciled file does not match the upload contract.")
        status = str(_field(uploaded, "status")).casefold()
        if status in {"processed", "completed"}:
            break
        if status in {"error", "failed", "expired", "cancelled"}:
            raise RuntimeError("Uploaded file failed processing; its ID remains saved for cleanup.")
        if time.monotonic() >= deadline:
            raise TimeoutError("File processing timed out; its ID remains saved for resume/cleanup.")
        time.sleep(poll_seconds)
    response = _read(lambda: client.files.content(file_id))
    downloaded = response if isinstance(response, bytes) else _field(response, "content")
    if downloaded is None and callable(getattr(response, "read", None)):
        downloaded = _read(response.read)
    if (
        not isinstance(downloaded, bytes)
        or _field(uploaded, "bytes") != len(downloaded)
        or downloaded.replace(b"\r\n", b"\n") != content
    ):
        raise RuntimeError("Saved/reconciled file content or byte count does not match the data.")
    return uploaded


def _verify_job(
    job: Any, job_id: str, state: dict[str, Any], method: dict[str, Any], *, recovered: bool,
) -> None:
    expected = {
        "model": state["equivalence"]["model"],
        "training_file": state["training_file_id"],
        "validation_file": state["validation_file_id"],
    }
    if _resource_id(job) != job_id or any(
        _field(job, key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Retrieved job does not match the saved project/model/file IDs.")
    if recovered:
        actual = _field(job, "method")
        if hasattr(actual, "model_dump"):
            actual = actual.model_dump(exclude_none=True)
        if not _matches_recipe(
            redact_tool_secrets(actual, method), redact_tool_secrets(method, method),
        ):
            raise RuntimeError("Reconciled job does not expose the matching grader/tool/recipe.")
        suffix = _field(job, "suffix")
        if suffix is not None and suffix != SUFFIX:
            raise RuntimeError("Reconciled job has a different submission suffix.")


def submit_or_resume_job(
    client: Any,
    *,
    state_path: str | Path,
    project_endpoint: str,
    model: str,
    model_version: str,
    training_type: str,
    method: dict[str, Any],
    training_path: str | Path,
    validation_path: str | Path,
    expected_training_sha256: str,
    expected_validation_sha256: str,
    reconciled_job_id: str | None = None,
    poll_seconds: float = 5,
    file_timeout_seconds: float = 600,
) -> Any:
    """Persist intent before every POST and every ID before polling/other work.

    Recover pending uploads only through a unique filename plus verified content.
    An unidentified job POST requires an operator-reconciled ID, never another
    POST. Its redacted recipe and saved file IDs must match; credential equality
    is bound locally by a fingerprint, not claimed from masked service headers.
    """
    path = _submission_state_path(state_path)
    if poll_seconds <= 0 or file_timeout_seconds <= 0:
        raise ValueError("File polling interval and timeout must be positive.")
    tools = method.get("reinforcement", {}).get("tools", [])
    for tool in tools:
        _https_url(tool["server_url"])
    contents = {
        "training": Path(training_path).read_bytes().replace(b"\r\n", b"\n"),
        "validation": Path(validation_path).read_bytes().replace(b"\r\n", b"\n"),
    }
    expected_hashes = {
        "training": expected_training_sha256, "validation": expected_validation_sha256,
    }
    contracts = {}
    for kind, content in contents.items():
        digest = hashlib.sha256(content).hexdigest()
        if not content or digest != expected_hashes[kind]:
            raise ValueError("Dataset bytes do not match the pinned canonical hash.")
        contracts[kind] = {
            "filename": f"{SUFFIX}-{kind}-{digest[:12]}.jsonl",
            "sha256": digest, "bytes": len(content),
        }
    fingerprint = hashlib.sha256(
        json.dumps(method, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    equivalence = {
        "project_endpoint": _https_url(project_endpoint, project=True),
        "model": model, "model_version": model_version, "training_type": training_type,
        "suffix": SUFFIX, "method": redact_tool_secrets(method, method),
        "method_fingerprint": fingerprint,
        "training_sha256": contracts["training"]["sha256"],
        "validation_sha256": contracts["validation"]["sha256"], "files": contracts,
    }
    requested_id = (reconciled_job_id or "").strip() or None
    with _submission_lock(path):
        if path.exists():
            state = _load_state(path, equivalence)
        else:
            if requested_id:
                raise RuntimeError("Job reconciliation requires the original pending state.")
            state = {
                "schema_version": 1, "equivalence": equivalence,
                "training_file_id": None, "validation_file_id": None,
                "job_id": None, "pending_operation": None,
            }
            _write_submission_state(path, state)

        def checkpoint(**updates: Any) -> None:
            nonlocal state
            candidate = {**state, **updates}
            _write_submission_state(_submission_state_path(path), candidate)
            state = candidate

        if requested_id and state["job_id"] not in {None, requested_id}:
            raise RuntimeError("The reconciled job ID conflicts with the saved job ID.")
        job_id = state["job_id"]
        if job_id:
            job = _read(lambda: client.fine_tuning.jobs.retrieve(job_id))
            _verify_job(job, job_id, state, method, recovered=False)
            return job
        if state["pending_operation"] == "job_create":
            if not requested_id:
                raise RuntimeError(
                    "Job submission outcome is ambiguous. Reconcile in Foundry and set "
                    "FOUNDRY_RECONCILED_TOOL_RFT_JOB_ID to the verified job ID. "
                    "No job POST was repeated."
                )
            job = _read(lambda: client.fine_tuning.jobs.retrieve(requested_id))
            _verify_job(job, requested_id, state, method, recovered=True)
            checkpoint(job_id=requested_id, pending_operation=None)
            return job
        if requested_id:
            raise RuntimeError("An explicit job ID is only accepted for a pending job POST.")

        for kind, content in contents.items():
            key = f"{kind}_file_id"
            pending = f"{kind}_upload"
            contract = contracts[kind]
            if not state[key]:
                if state["pending_operation"] not in {None, pending}:
                    raise RuntimeError("Reconcile the pending operation before another upload.")
                candidates = _read(lambda: _page_items(client.files.list()))
                matches = [
                    item for item in candidates
                    if _field(item, "filename") == contract["filename"]
                    and _field(item, "purpose") == "fine-tune"
                ]
                if len(matches) > 1:
                    raise RuntimeError("Multiple files match the upload contract; reconcile without a POST.")
                if matches:
                    uploaded = matches[0]
                elif state["pending_operation"] == pending:
                    raise RuntimeError(
                        "Upload outcome is ambiguous and no unique file is visible. "
                        "Reconcile in Foundry; no upload POST was repeated."
                    )
                else:
                    submission_client = client.with_options(max_retries=0)
                    checkpoint(pending_operation=pending)
                    try:
                        uploaded = submission_client.files.create(
                            file=(contract["filename"], content, "application/jsonl"),
                            purpose="fine-tune",
                        )
                    except Exception:
                        raise RuntimeError(
                            "Upload response was not recorded. Retain pending state and "
                            "reconcile in Foundry before any further submission."
                        ) from None
                checkpoint(**{key: _resource_id(uploaded), "pending_operation": None})
            _verify_file(
                client, state[key], contract, content,
                poll_seconds=poll_seconds, timeout_seconds=file_timeout_seconds,
            )

        submission_client = client.with_options(max_retries=0)
        checkpoint(pending_operation="job_create")
        try:
            job = submission_client.fine_tuning.jobs.create(
                model=model, training_file=state["training_file_id"],
                validation_file=state["validation_file_id"], suffix=SUFFIX, method=method,
                extra_body={"trainingType": training_type},
            )
        except Exception:
            raise RuntimeError(
                "Job response was not recorded. Retain pending state and reconcile in Foundry. "
                "Do not repeat the job POST."
            ) from None
        checkpoint(job_id=_resource_id(job), pending_operation=None)
        return job