"""Capstone-local contract checks and authenticated, read-only tool transport.

This is a client, not a retail server or a return-policy implementation. The
customer's isolated server must implement the published operations and policy.
No import or local-contract check contacts that server.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

READ_ONLY_TOOLS = frozenset({
    "calculate", "find_user_id_by_email", "find_user_id_by_name_zip",
    "get_order_details", "get_product_details", "get_user_details",
    "list_all_product_types", "list_user_orders", "policy_verify_return",
})
CUSTOMER_LOOKUPS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
RETAIL_DATASETS = (
    "sft-train.jsonl", "sft-validation.jsonl", "next-tool-evaluation.jsonl",
    "rft-train.jsonl", "rft-validation.jsonl",
    "synthetic-source-train.jsonl", "synthetic-source-validation.jsonl",
)
PRESERVED_SOURCE_DATASETS = frozenset({
    "synthetic-source-train.jsonl", "synthetic-source-validation.jsonl",
})
ACTIVE_RETAIL_DATASETS = tuple(
    name for name in RETAIL_DATASETS if name not in PRESERVED_SOURCE_DATASETS
)


def _active_dataset_paths(
    root: Path, dataset_paths: Mapping[str, Path] | None,
) -> dict[str, Path]:
    """Never infer active inputs from the immutable data directory."""
    if dataset_paths is None or set(dataset_paths) != set(ACTIVE_RETAIL_DATASETS):
        raise ValueError("Explicit prepared active dataset paths are required for all five splits.")
    directory = root.resolve() / "outputs" / "prepared-data"
    paths: dict[str, Path] = {}
    for name in ACTIVE_RETAIL_DATASETS:
        path = Path(dataset_paths[name])
        expected = directory / name
        if (
            path != expected or path.resolve() != expected or path.is_symlink()
            or not path.is_file() or path.stat().st_nlink != 1
        ):
            raise ValueError("Active datasets must be regular, unlinked files under outputs/prepared-data.")
        paths[name] = path
    return paths


def _tool_result(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except ValueError:
        return content


def _customer_id(result: Any) -> str | None:
    if isinstance(result, dict):
        if "error" in result or result.get("success") is False:
            return None
        result = result.get("user_id")
    if (
        isinstance(result, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", result)
        and result.casefold() not in {
            "error", "none", "null", "false", "true", "not_found", "notfound",
            "ok", "success", "failed", "failure", "unauthorized",
        }
    ):
        return result
    return None


def _lookup_matches(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> bool:
    """Reject contradictory identity fields without requiring invented fields."""
    if name == "find_user_id_by_email":
        email = result.get("email")
        return email is None or (
            isinstance(email, str) and email.casefold() == arguments["email"].casefold()
        )
    identity = result.get("name", {})
    if isinstance(identity, str):
        expected_name = f"{arguments['first_name']} {arguments['last_name']}"
        if identity.casefold() != expected_name.casefold():
            return False
        identity = {}
    if not isinstance(identity, dict):
        return False
    for field in ("first_name", "last_name"):
        observed = identity.get(field, result.get(field))
        if observed is not None and (
            not isinstance(observed, str) or observed.casefold() != arguments[field].casefold()
        ):
            return False
    address = result.get("address", {})
    if not isinstance(address, dict):
        return False
    zip_code = result.get("zip", address.get("zip"))
    return zip_code is None or zip_code == arguments["zip"]


def retail_trace_diagnostics(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Audit observed customer context, authentication order, and payment IDs.

    A bearer header, claimed user ID, assistant prose, declaration, or pending
    lookup is not customer authentication. Only a matching successful lookup
    reply establishes it. Payment evidence comes from earlier, owned profile or
    order replies, never a later result or the global fixture's unobserved data.
    Final supervised calls and the separate one-call expected_output are targets,
    not completed dialogues; no reply is invented for either. RFT reference names
    establish sequence order only, not successful authentication or policy truth.
    """
    if not isinstance(row, Mapping):
        return [{"code": "messages.invalid", "message_index": None}]
    item = row.get("item", row)
    if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
        return [{"code": "messages.invalid", "message_index": None}]
    problems: list[dict[str, Any]] = []
    customer: str | None = None
    credentials: list[tuple[str, dict[str, Any]]] = []
    methods: set[str] = set()
    owned_orders: set[str] = set()
    pending: dict[str, tuple[str, dict[str, Any], bool]] = {}

    def issue(code: str, index: int, name: str | None = None) -> None:
        problems.append({"code": code, "message_index": index, "tool": name})

    messages = item["messages"]
    expected = item.get("expected_output")
    if "expected_output" in item:
        if (
            not isinstance(expected, dict) or expected.get("role") != "assistant"
            or not isinstance(expected.get("tool_calls"), list)
            or len(expected["tool_calls"]) != 1
        ):
            return [{"code": "prediction.one-call-required", "message_index": len(messages)}]
        messages = [*messages, expected]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            issue("message.invalid", index)
            continue
        role = message.get("role")
        if pending and role != "tool":
            issue("tool.missing-reply", index)
            pending.clear()
        if role == "tool":
            reply_id = message.get("tool_call_id")
            call = pending.pop(reply_id, None) if isinstance(reply_id, str) else None
            if call is None:
                issue("tool.unmatched-reply", index)
                continue
            name, arguments, authorized = call
            result = _tool_result(message.get("content"))
            if name in CUSTOMER_LOOKUPS:
                identity = _customer_id(result)
                if identity is None:
                    # Failed authentication can legitimately be retried or refused.
                    continue
                if isinstance(result, dict) and not _lookup_matches(name, arguments, result):
                    issue("authentication.identity-mismatch", index, name)
                    continue
                if customer is not None and identity != customer:
                    issue("authentication.customer-switch", index, name)
                    continue
                customer = identity
                credentials.append((name, arguments))
            elif authorized and isinstance(result, dict) and "error" not in result and result.get("success") is not False:
                owner = result.get("user_id")
                if owner is not None and owner != customer:
                    issue("authentication.foreign-result", index, name)
                    continue
                if name == "get_user_details" and customer is not None:
                    if arguments.get("user_id") != customer:
                        continue
                    if any(not _lookup_matches(lookup, values, result) for lookup, values in credentials):
                        issue("authentication.identity-mismatch", index, name)
                        continue
                    observed = result.get("payment_methods", {})
                    if isinstance(observed, dict):
                        methods.update(key for key in observed if isinstance(key, str))
                    elif isinstance(observed, list):
                        for method in observed:
                            if not isinstance(method, dict):
                                continue
                            identifier = method.get("id", method.get("payment_method_id"))
                            if isinstance(identifier, str):
                                methods.add(identifier)
                    orders = result.get("orders", [])
                    if isinstance(orders, list):
                        owned_orders.update(order for order in orders if isinstance(order, str))
                elif name == "get_order_details" and customer is not None:
                    order_id = arguments.get("order_id")
                    if not isinstance(order_id, str) or result.get("order_id", order_id) != order_id:
                        issue("authentication.order-mismatch", index, name)
                        continue
                    if owner != customer and order_id not in owned_orders:
                        issue("authentication.unproven-order-owner", index, name)
                        continue
                    owned_orders.add(order_id)
                    history = result.get("payment_history", [])
                    if isinstance(history, list):
                        methods.update(
                            payment["payment_method_id"] for payment in history
                            if isinstance(payment, dict) and isinstance(payment.get("payment_method_id"), str)
                        )
            continue
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list) or (calls and role != "assistant"):
            issue("tool.invalid-calls", index)
            continue
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                issue("tool.invalid-call", index)
                continue
            function = call["function"]
            name = function.get("name")
            identifier = call.get("id")
            if call.get("type") != "function" or not isinstance(name, str):
                issue("tool.invalid-call", index)
                continue
            if not isinstance(identifier, str) or not identifier or identifier in pending:
                issue("tool.invalid-id", index, name)
                continue
            try:
                arguments = json.loads(function["arguments"])
            except (KeyError, TypeError, ValueError):
                issue("tool.invalid-arguments", index, name)
                continue
            if not isinstance(arguments, dict):
                issue("tool.invalid-arguments", index, name)
                continue
            authorized = customer is not None
            if name in CUSTOMER_LOOKUPS:
                fields = ("email",) if name == "find_user_id_by_email" else ("first_name", "last_name", "zip")
                if any(not isinstance(arguments.get(field), str) or not arguments[field].strip() for field in fields):
                    issue("authentication.invalid-lookup", index, name)
                    continue
            elif customer is None:
                issue("authentication.required-first", index, name)
            elif "user_id" in arguments and arguments["user_id"] != customer:
                issue("authentication.foreign-user", index, name)
                authorized = False
            if "payment_method_id" in arguments:
                payment_id = arguments["payment_method_id"]
                if customer is None or not isinstance(payment_id, str) or payment_id not in methods:
                    issue("payment.unobserved-method", index, name)
                order_id = arguments.get("order_id")
                if not isinstance(order_id, str) or order_id not in owned_orders:
                    issue("authentication.unproven-action-owner", index, name)
            pending[identifier] = (name, arguments, authorized)
    if pending and (
        not messages or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "assistant"
    ):
        issue("tool.missing-reply", len(messages) - 1)
    references = item.get("reference_tool_calls")
    if references is not None:
        if not isinstance(references, list) or not references or not all(isinstance(name, str) for name in references):
            issue("reference.invalid-sequence", -1)
        elif references[0] not in CUSTOMER_LOOKUPS:
            issue("authentication.reference-required-first", -1, references[0])
    return problems


def audit_retail_datasets(
    root: Path, *, dataset_paths: Mapping[str, Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Report all seven raw sources, or audit only explicit prepared active copies.

    Reporting a raw source's findings never approves it for training/evaluation.
    The two synthetic source splits cannot enter the prepared-active mapping.
    """
    paths = (
        {name: root / "data" / name for name in RETAIL_DATASETS}
        if dataset_paths is None else _active_dataset_paths(root, dataset_paths)
    )
    report: dict[str, list[dict[str, Any]]] = {}
    for name, path in paths.items():
        failures = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Retail dataset contains a blank record: {name}, row {number}.")
                row = json.loads(line)
                for problem in retail_trace_diagnostics(row):
                    failures.append({"row": number, **problem})
        report[name] = failures
    return report


def load_tool_headers(
    environ: Mapping[str, str] | None = None, *, required: bool = False,
) -> dict[str, str]:
    """Use the catalog sample's bearer scheme, without an anonymous live fallback."""
    values = os.environ if environ is None else environ
    token = values.get("FOUNDRY_TOOL_AUTH_TOKEN", "").strip()
    if not token:
        if required:
            raise ValueError("Set FOUNDRY_TOOL_AUTH_TOKEN before a remote-tool stage.")
        return {}
    if any(not 33 <= ord(character) <= 126 for character in token):
        raise ValueError("FOUNDRY_TOOL_AUTH_TOKEN must be one printable ASCII credential.")
    return {"Authorization": f"Bearer {token}"}


def _https_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https" or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
    ):
        raise ValueError("Use an HTTPS tool URL without credentials, query, or fragment.")
    return value.rstrip("/")


def build_remote_tools(
    config_path: Path, server_url: str, headers: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Copy the preserved configuration; never write live headers back to disk."""
    tools = json.loads(config_path.read_text(encoding="utf-8"))
    selected_url = _https_url(server_url) if server_url else None
    for tool in tools:
        if selected_url:
            tool["server_url"] = selected_url
        tool["headers"] = dict(headers)
    return tools


def _referenced_tools(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        references = value.get("reference_tool_calls", [])
        if isinstance(references, list):
            names.update(name for name in references if isinstance(name, str))
        for child in value.values():
            names.update(_referenced_tools(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_referenced_tools(child))
    return names


def validate_local_tool_contract(
    root: Path, *, dataset_paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Require every configured/dataset tool to have a published POST operation.

    The two RFT-only parameter schemas must exactly match every preserved RFT
    declaration. Reference policy arguments are grading data, not permission to
    broaden an operation's reason enum or invent a server-side policy.
    Only explicit prepared copies can be active. Raw artifacts contribute tool
    references/provenance, not authorization to upload their unsafe traces.
    Every prepared row must pass authentication and payment grounding; raw
    findings remain separately available from audit_retail_datasets.
    """
    active_paths = _active_dataset_paths(root, dataset_paths)
    specification = json.loads(
        (root / "configs" / "retail-openapi-with-policy.json").read_text(encoding="utf-8")
    )
    operations: dict[str, dict[str, Any]] = {}
    for path, methods in specification["paths"].items():
        operation = methods.get("post")
        if operation is None:
            continue
        name = operation["operationId"]
        if name in operations or path != f"/{name}":
            raise ValueError("Retail operation identifiers must be unique and match their paths.")
        operations[name] = operation
    configured = json.loads((root / "configs" / "rft-tools.json").read_text(encoding="utf-8"))
    configured_names = [tool["name"] for tool in configured]
    if len(configured_names) != len(set(configured_names)):
        raise ValueError("Duplicate configured retail tool.")
    referenced = set(configured_names)
    evaluation = json.loads(
        (root / "configs" / "sft-evaluation-tools.json").read_text(encoding="utf-8")
    )
    referenced.update(_referenced_tools(evaluation))
    trace_failures: list[str] = []
    # Source declarations still need operation coverage, even though these
    # immutable transcripts are never selected as active training inputs.
    for path in sorted((root / "data").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Retail dataset contains a blank record: {path.name}, row {number}.")
                referenced.update(_referenced_tools(json.loads(line)))
    for name, path in active_paths.items():
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Retail dataset contains a blank record: {name}, row {number}.")
                row = json.loads(line)
                referenced.update(_referenced_tools(row))
                problems = retail_trace_diagnostics(row)
                if problems:
                    codes = ", ".join(sorted({problem["code"] for problem in problems}))
                    trace_failures.append(f"{name}, row {number}: {codes}")
                if name not in {"rft-train.jsonl", "rft-validation.jsonl"}:
                    continue
                for tool in row["tools"]:
                    function = tool["function"]
                    function_name = function["name"]
                    if function_name not in {"list_user_orders", "policy_verify_return"}:
                        continue
                    if function_name not in operations:
                        raise ValueError(f"Missing retail operation: {function_name}.")
                    schema = operations[function_name]["requestBody"]["content"]["application/json"]["schema"]
                    if schema != function["parameters"]:
                        raise ValueError(f"Retail parameter drift: {name}, row {number}, {function_name}.")
    missing = referenced - operations.keys()
    if missing:
        raise ValueError("Missing retail operations: " + ", ".join(sorted(missing)))
    if set(configured_names) != set(operations):
        raise ValueError("Configured retail tools and published operations must match.")
    if trace_failures:
        raise ValueError("Retail trace validation failed: " + "; ".join(trace_failures))
    return operations


class RetailToolError(RuntimeError):
    """Safe diagnostic with only an HTTP status available to bounded read retries."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def call_read_only_tool(
    server_url: str,
    name: str,
    arguments: str,
    call_id: str,
    *,
    trace_id: str,
    headers: Mapping[str, str],
    sandbox: bool,
    post: Callable[..., Any] | None = None,
) -> str:
    """Send the catalog-style envelope; never follow a credential-bearing redirect.

    Error diagnostics deliberately omit response bodies, request headers, and
    underlying exception text, any of which can contain an echoed credential.
    The caller owns one trace per conversation, shared across calls and retries;
    this correlation identifier is not a customer authorization grant.
    """
    if name not in READ_ONLY_TOOLS:
        raise RuntimeError("Blocked mutating or unknown retail tool.")
    if not sandbox:
        raise RuntimeError("Remote tools require an acknowledged isolated sandbox.")
    url = _https_url(server_url)
    authorization = headers.get("Authorization", "")
    if (
        set(headers) != {"Authorization"} or not authorization.startswith("Bearer ")
        or not authorization[7:]
        or any(not 33 <= ord(character) <= 126 for character in authorization[7:])
    ):
        raise ValueError("Remote retail tools require a valid bearer credential.")
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        raise ValueError("Tool arguments must be a JSON object.") from None
    if not isinstance(parsed, dict) or not isinstance(call_id, str) or not call_id:
        raise ValueError("Tool arguments must be an object with a non-empty call identifier.")
    if (
        not isinstance(trace_id, str) or not trace_id
        or any(not 33 <= ord(character) <= 126 for character in trace_id)
    ):
        raise ValueError("An explicit printable conversation trace identifier is required.")
    if post is None:
        import requests

        post = requests.post
    payload = {
        "type": "function_call", "id": call_id, "call_id": call_id,
        "name": name, "arguments": arguments, "trace_id": trace_id,
    }
    failure: RetailToolError | None = None
    try:
        response = post(url, json=payload, headers=dict(headers), timeout=60, allow_redirects=False)
        status = response.status_code
        if response.is_redirect or 300 <= status < 400:
            raise RetailToolError("Retail tool redirects are not permitted.", status)
        if not 200 <= status < 300:
            raise RetailToolError("Retail tool HTTP request failed.", status)
        body = response.json()
        if (
            not isinstance(body, dict) or body.get("type") != "function_call_output"
            or body.get("call_id") != call_id or body.get("id") != call_id
            or "output" not in body
        ):
            raise RetailToolError("Retail tool returned an invalid function-call envelope.")
        output = body["output"]
        # A misconfigured server may echo an Authorization value even on success.
        serialized = json.dumps(output, ensure_ascii=True, allow_nan=False)
        token = authorization[7:]
        if token in serialized or json.dumps(token)[1:-1] in serialized:
            raise RetailToolError("Retail tool response contained credential material.")
        return output if isinstance(output, str) else serialized
    except RetailToolError as error:
        failure = error
    except RequestsTimeout:
        failure = RetailToolError("Retail tool request timed out.", 408)
    except RequestsConnectionError:
        failure = RetailToolError("Retail tool connection error.", 503)
    except Exception:
        failure = RetailToolError("Retail tool transport or response decoding failed.")
    # Raise outside the handler: no secret-bearing HTTP exception is chained.
    raise failure from None


def preflight_retail_tools(
    root: Path,
    server_url: str,
    *,
    dataset_paths: Mapping[str, Path],
    headers: Mapping[str, str],
    sandbox: bool,
    post: Callable[..., Any] | None = None,
) -> None:
    """Probe both missing operations using one owned synthetic fixture identity.

    This checks authenticated transport/dispatch and obvious tool errors, not the
    correctness of the customer's return-policy engine. No mutation is probed.
    """
    validate_local_tool_contract(root, dataset_paths=dataset_paths)
    with dataset_paths["rft-train.jsonl"].open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    policy_arguments = first["reference_policy_args"][0]
    database = json.loads((root / "data" / "retail-database.json").read_text(encoding="utf-8"))
    order = database["orders"][policy_arguments["order_id"]]
    user = database["users"][order["user_id"]]
    probes = [
        ("find_user_id_by_email", {"email": user["email"]}),
        ("list_user_orders", {"user_id": user["user_id"]}),
        ("get_order_details", {"order_id": order["order_id"]}),
        ("policy_verify_return", policy_arguments),
    ]
    trace_id = f"retail-{uuid.uuid4().hex}"
    for name, arguments in probes:
        result = call_read_only_tool(
            server_url, name, json.dumps(arguments), f"call_retail_preflight_{uuid.uuid4().hex}",
            trace_id=trace_id, headers=headers, sandbox=sandbox, post=post,
        )
        try:
            decoded = json.loads(result)
        except (TypeError, ValueError):
            raise RetailToolError("Retail preflight requires JSON tool outputs.") from None
        if name == "find_user_id_by_email":
            identity = decoded.get("user_id") if isinstance(decoded, dict) else decoded
            if identity != user["user_id"]:
                raise RetailToolError("Retail preflight identity does not match the isolated fixture.")
        elif not isinstance(decoded, (dict, list)) or not decoded:
            raise RetailToolError("Retail preflight returned an empty or invalid tool result.")
        if isinstance(decoded, dict) and ("error" in decoded or decoded.get("success") is False):
            raise RetailToolError("Retail preflight reported a tool error; verify the external implementation.")