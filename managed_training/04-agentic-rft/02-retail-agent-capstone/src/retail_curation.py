"""Deterministic offline preparation from immutable retail source artifacts.

Only five validated active copies and their computed manifest may be written,
under outputs/prepared-data. No source, service, or retail tool is modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from retail_tools import (
    ACTIVE_RETAIL_DATASETS, PRESERVED_SOURCE_DATASETS, RETAIL_DATASETS,
    retail_trace_diagnostics,
)

PREPARATION_VERSION = 2
PREPARED_DIRECTORY = "outputs/prepared-data"


AUTHENTICATION_REPAIRS = (
    ("call_3fb27ba67f3240c48a982c95", "call_5805cd701ac84b60a7c7d2a1",
     {"first_name": "Harper", "last_name": "Brown", "zip": "76112"}, "harper_brown_7363"),
    ("call_349fb7e2c7bf40eea73e227b", "call_1d2d61c31ff64373b418873b",
     {"first_name": "Mei", "last_name": "Silva", "zip": "91147"}, "mei_silva_6882"),
    ("call_786d9ae8758549e9abc4d0bc", "call_69f39243f05e4b7db95902bd",
     {"first_name": "Mei", "last_name": "Silva", "zip": "91147"}, "mei_silva_6882"),
)
AUTHENTICATION_PROMPT = (
    "Before I access product information, I must authenticate your identity. "
    "You may provide your email address or your first name, last name, and zip code."
)
OBSOLETE_AUTHENTICATION_REQUESTS = (
    "If you would like to proceed with the exchange, please provide your email address "
    "so I can authenticate your identity and assist you further.",
    "Please provide your email address so I can authenticate your identity and assist "
    "you with the exchange process.",
    "Now, let's proceed with authenticating your identity. Could you please provide your email address?",
)
PAYMENT_REFUSALS = {
    "call_9e5a869f9b624b98ab119d32": (
        "Your retrieved profile lists only gift_card_5701566 with a $37 balance and no stored credit card. "
        "The calculated price difference is $329.40, so that gift card cannot cover it. "
        "The exchange tool accepts one payment method, not a split payment. "
        "I cannot invent a card or charge an unlisted method, and I have not submitted the exchange."
    ),
    "call_7b03739a5a9d4602bc3cdd3a": (
        "Your retrieved profile lists gift_card_1888303 and no stored credit card. "
        "I cannot modify your order using an unlisted credit card. I have not submitted this change. "
        "Please choose an existing payment method from your profile before we proceed."
    ),
}

# These four held-out histories already contain the real profile and a grounded
# next-call target. Remove only the identified failed mutation/reply pair; do
# not turn a later profile into evidence for a mutation that has already run.
EVALUATION_PAYMENT_REPAIRS = (
    {
        "failed_call_id": "call_82056c5fb8984cb58d28e4d2",
        "profile_call_id": "call_f051f4d829d14c4b80f5295e",
        "target_call_id": "call_ba22d12991184a00a2aa3e3c",
        "customer": "sophia_thomas_5301", "name": "modify_pending_order_items",
        "arguments": {"order_id": "#W4862767", "item_ids": ["8323284863"],
                      "new_item_ids": ["4648362606"], "payment_method_id": "credit_card_2378"},
        "payment_method": {"id": "credit_card_1034663", "source": "credit_card",
                           "brand": "mastercard", "last_four": "2378"},
    },
    {
        "failed_call_id": "call_b41864d72a774a818b4dbdc3",
        "profile_call_id": "call_585bd161ed704eae8f635736",
        "target_call_id": "call_11f500e6c9d4440ba29aa494",
        "customer": "sophia_smith_8223", "name": "modify_pending_order_items",
        "arguments": {"order_id": "#W6760641", "item_ids": ["6942241102"],
                      "new_item_ids": ["7381052709"], "payment_method_id": "paypal_0000000"},
        "payment_method": {"id": "paypal_6651356", "source": "paypal"},
    },
    {
        "failed_call_id": "call_379037066c31444486ee6c30",
        "profile_call_id": "call_2276c4bc41a24d5b9ca7f657",
        "target_call_id": "call_28c931bfdf964b06855dd7da",
        "customer": "omar_kim_3528", "name": "modify_pending_order_items",
        "arguments": {"order_id": "#W8557584", "item_ids": ["9747045638"],
                      "new_item_ids": ["3761330360"], "payment_method_id": "credit_card_9843"},
        "payment_method": {"id": "credit_card_3577130", "source": "credit_card",
                           "brand": "mastercard", "last_four": "9843"},
    },
    {
        "failed_call_id": "call_4efdf36d7ffc4c8ab1b2b12f",
        "profile_call_id": "call_6d4771a4e703464890f0abc6",
        "target_call_id": "call_6353932dbcb34c99b2ebf36c",
        "customer": "anya_sanchez_9707", "name": "exchange_delivered_order_items",
        "arguments": {"order_id": "#W4442043", "item_ids": ["5917587651"],
                      "new_item_ids": ["8084436579"], "payment_method_id": "credit_card_0000000"},
        "payment_method": {"id": "paypal_1191071", "source": "paypal"},
    },
)


def _call_index(messages: list[dict[str, Any]], identifier: str) -> int | None:
    matches = [index for index, message in enumerate(messages)
               if any(call.get("id") == identifier for call in message.get("tool_calls", []))]
    if len(matches) > 1:
        raise ValueError(f"Reviewed call identifier is not unique: {identifier}.")
    return matches[0] if matches else None


def _matched_result(messages: list[dict[str, Any]], index: int, identifier: str) -> Any:
    if index + 1 >= len(messages):
        raise ValueError(f"Reviewed lookup has no observed reply: {identifier}.")
    message = messages[index + 1]
    if message.get("role") != "tool" or message.get("tool_call_id") != identifier:
        raise ValueError(f"Reviewed call/reply ordering changed: {identifier}.")
    result = message.get("content")
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        return result


def _single_function(messages: list[dict[str, Any]], index: int, identifier: str) -> dict[str, Any]:
    message = messages[index]
    calls = message.get("tool_calls", [])
    if (
        message.get("role") != "assistant" or len(calls) != 1
        or calls[0].get("id") != identifier or calls[0].get("type") != "function"
        or not isinstance(calls[0].get("function"), dict)
    ):
        raise ValueError("Reviewed repair requires one unchanged function call per assistant message.")
    return calls[0]["function"]


def _profile(
    messages: list[dict[str, Any]], identifier: str, customer: str, *, before: int,
) -> dict[str, Any]:
    index = _call_index(messages, identifier)
    if index is None or index + 1 >= before:
        raise ValueError("Reviewed profile evidence is missing or was not observed before the action.")
    function = _single_function(messages, index, identifier)
    if function["name"] != "get_user_details" or json.loads(function["arguments"]) != {"user_id": customer}:
        raise ValueError("Reviewed profile is not for the authenticated customer.")
    result = _matched_result(messages, index, identifier)
    if (
        not isinstance(result, dict) or result.get("user_id", customer) != customer
        or "error" in result or result.get("success") is False
    ):
        raise ValueError("Reviewed profile requires a successful observed result for the authenticated customer.")
    return result


def curate_reviewed_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply only identified repairs, preserving actual inputs/replies and targets.

    Authentication repair moves the existing credential message and its observed
    successful lookup before the first catalog access; it never invents a result.
    Payment repair ends the trace before the unsupported mutation and supervises
    a refusal, rather than training the failed mutation as a desirable action.
    Reviewed evaluation histories lose only an explicitly failed mutation/reply
    pair, retaining their actual profile replies and original one-call targets.
    """
    curated = copy.deepcopy(row)
    item = curated.get("item", curated)
    messages = item["messages"]
    decisions = []
    for lookup_id, catalog_id, arguments, customer in AUTHENTICATION_REPAIRS:
        lookup_index = _call_index(messages, lookup_id)
        catalog_index = _call_index(messages, catalog_id)
        if lookup_index is None or catalog_index is None or lookup_index < catalog_index:
            continue
        function = _single_function(messages, lookup_index, lookup_id)
        catalog = _single_function(messages, catalog_index, catalog_id)
        if catalog.get("name") not in {"list_all_product_types", "get_product_details"}:
            raise ValueError("Reviewed catalog operation changed.")
        if function["name"] != "find_user_id_by_name_zip" or json.loads(function["arguments"]) != arguments:
            raise ValueError("Reviewed authentication arguments changed.")
        if _matched_result(messages, lookup_index, lookup_id) != customer:
            raise ValueError("Reviewed authentication result changed; do not fabricate a replacement.")
        credential_message = messages[lookup_index - 1]
        if credential_message.get("role") != "user" or not all(
            value.casefold() in credential_message.get("content", "").casefold()
            for value in arguments.values()
        ):
            raise ValueError("Reviewed user-provided credentials are missing.")
        # Move these three real messages together; do not execute the lookup.
        block = messages[lookup_index - 1:lookup_index + 2]
        del messages[lookup_index - 1:lookup_index + 2]
        messages[catalog_index:catalog_index] = [
            {"role": "assistant", "content": AUTHENTICATION_PROMPT}, *block,
        ]
        for message in messages[catalog_index + 4:]:
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                for obsolete in OBSOLETE_AUTHENTICATION_REQUESTS:
                    message["content"] = message["content"].replace(obsolete, "").rstrip()
                if not message["content"] and not message.get("tool_calls"):
                    message["content"] = "Your identity is already authenticated; I can continue with the exchange details."
        decisions.append("authentication-before-product-access")

    for identifier, refusal in PAYMENT_REFUSALS.items():
        index = _call_index(messages, identifier)
        if index is None:
            continue
        if "expected_output" in item:
            raise ValueError("A text refusal cannot replace a one-tool evaluation target.")
        function = _single_function(messages, index, identifier)
        arguments = json.loads(function["arguments"])
        if (
            function.get("name") not in {"exchange_delivered_order_items", "modify_pending_order_items"}
            or arguments.get("payment_method_id") != "credit_card_0000000"
        ):
            raise ValueError("Reviewed ungrounded payment target changed.")
        if identifier == "call_9e5a869f9b624b98ab119d32":
            profile = _profile(messages, "call_b7d75f8e3de74d6591f62b30", "ethan_kim_8860", before=index)
            methods = profile.get("payment_methods")
            if (
                not isinstance(methods, dict) or set(methods) != {"gift_card_5701566"}
                or not isinstance(methods["gift_card_5701566"], dict)
                or methods["gift_card_5701566"].get("balance") != 37
            ):
                raise ValueError("Reviewed one-gift-card profile evidence changed.")
            calculator = _call_index(messages, "call_543dd16e45f844efac6bccdf")
            if calculator is None or calculator >= index:
                raise ValueError("Reviewed price calculation is missing.")
            calculation = _single_function(messages, calculator, "call_543dd16e45f844efac6bccdf")
            if calculation.get("name") != "calculate":
                raise ValueError("Reviewed price calculation operation changed.")
            if _matched_result(messages, calculator, "call_543dd16e45f844efac6bccdf") != 329.4:
                raise ValueError("Reviewed price difference changed.")
            # Also remove the insufficient-balance attempt and the false split-payment offer.
            index = calculator + 2
        else:
            profile = _profile(messages, "call_fad448d0cb44480990b0d876", "mei_wilson_1792", before=index)
            methods = profile.get("payment_methods")
            if (
                not isinstance(methods, dict) or set(methods) != {"gift_card_1888303"}
                or not isinstance(methods["gift_card_1888303"], dict)
            ):
                raise ValueError("Reviewed profile does not establish the listed existing gift card.")
        messages[index:] = [{"role": "assistant", "content": refusal}]
        decisions.append("refuse-unobserved-payment-without-submitting-mutation")

    for repair in EVALUATION_PAYMENT_REPAIRS:
        identifier = repair["failed_call_id"]
        index = _call_index(messages, identifier)
        if index is None:
            continue
        function = _single_function(messages, index, identifier)
        if function.get("name") != repair["name"] or json.loads(function["arguments"]) != repair["arguments"]:
            raise ValueError("Reviewed failed evaluation payment arguments changed.")
        if _matched_result(messages, index, identifier) != "Error: payment method not found":
            raise ValueError("Reviewed evaluation mutation is not the observed payment-method failure.")
        if _call_index(messages, repair["profile_call_id"]) != index + 2:
            raise ValueError("Reviewed evaluation profile must immediately follow the failed call/reply pair.")
        profile = _profile(messages, repair["profile_call_id"], repair["customer"], before=len(messages))
        methods = profile.get("payment_methods")
        selected = repair["payment_method"]
        method = methods.get(selected["id"]) if isinstance(methods, dict) else None
        if (
            not isinstance(method, dict)
            or any(method.get(key) != value for key, value in selected.items())
            or repair["arguments"]["payment_method_id"] in methods
            or not isinstance(profile.get("orders"), list)
            or repair["arguments"]["order_id"] not in profile["orders"]
        ):
            raise ValueError("Reviewed evaluation profile no longer establishes the existing owned payment method.")
        target = item.get("expected_output")
        if not isinstance(target, dict):
            raise ValueError("Reviewed evaluation curation requires the original one-call target.")
        target_function = _single_function([target], 0, repair["target_call_id"])
        if target_function.get("name") != repair["name"] or json.loads(target_function["arguments"]) != {
            **repair["arguments"], "payment_method_id": selected["id"],
        }:
            raise ValueError("Reviewed evaluation target changed; do not invent a replacement.")
        del messages[index:index + 2]
        decisions.append("remove-reviewed-failed-payment-attempt-preserve-observed-profile-and-target")
    if decisions and retail_trace_diagnostics(curated):
        raise ValueError("Reviewed curation still fails the retail trace audit; investigate before adoption.")
    return curated, decisions


def _fingerprint(content: bytes) -> dict[str, Any]:
    canonical = content.replace(b"\r\n", b"\n")
    return {
        "bytes": len(canonical), "sha256": hashlib.sha256(canonical).hexdigest(),
        "raw_bytes": len(content), "raw_sha256": hashlib.sha256(content).hexdigest(),
    }


def _source_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative or "\\" in relative or ":" in relative or path.is_absolute()
        or ".." in path.parts or path.as_posix() != relative
        or len(path.parts) < 2 or path.parts[0] not in {"data", "configs", "graders"}
    ):
        raise ValueError("Source manifest paths must stay in the cookbook's immutable input directories.")
    candidate = root / relative
    if candidate.resolve() != candidate or not candidate.is_file():
        raise ValueError("Source manifest paths must identify regular, contained input files.")
    return candidate


def _jsonl_records(content: bytes, name: str) -> list[bytes]:
    # Split physical LF records only, not escaped text or arbitrary Unicode separators.
    lines = content.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    records = [line + (b"\n" if index < len(lines) - 1 or content.endswith(b"\n") else b"")
               for index, line in enumerate(lines)]
    if not records or any(not line.strip() for line in records):
        raise ValueError(f"Retail dataset contains a blank or missing record: {name}.")
    return records


def _verified_sources(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, bytes]]:
    raw_manifest = _source_path(root, "data/manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("algorithm") != "sha256" or manifest.get("text_normalization") != "crlf-to-lf":
        raise ValueError("Source manifest must declare SHA-256 with CRLF-to-LF normalization.")
    fingerprints: dict[str, dict[str, Any]] = {}
    sources: dict[str, bytes] = {}
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative in fingerprints:
            raise ValueError("Source manifest contains duplicate paths.")
        raw = _source_path(root, relative).read_bytes()
        actual = _fingerprint(raw)
        if actual["bytes"] != entry["bytes"] or actual["sha256"] != entry["sha256"]:
            raise ValueError(f"Source provenance differs from its manifest: {relative}.")
        if relative.endswith(".jsonl"):
            actual["records"] = len(_jsonl_records(raw, relative))
            if actual["records"] != entry["records"]:
                raise ValueError(f"Source record count differs from its manifest: {relative}.")
        fingerprints[relative] = actual
        sources[relative] = raw
    if not {f"data/{name}" for name in RETAIL_DATASETS} <= fingerprints.keys():
        raise ValueError("Source manifest must include all seven immutable retail datasets.")
    return {"path": "data/manifest.json", **_fingerprint(raw_manifest)}, fingerprints, sources


def prepare_reviewed_datasets(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build five validated LF payloads in memory; never write or waive a raw row.

    All seven originals are hashed and audited. Only identified active records
    are reserialized; other active records change at most their physical CRLF.
    Synthetic originals have diagnostics but NO output artifact or active path.
    """
    root = root.resolve()
    source_manifest, source_inputs, sources = _verified_sources(root)
    payloads: dict[str, bytes] = {}
    report: dict[str, Any] = {
        "schema_version": 1, "preparation_version": PREPARATION_VERSION,
        "status": "validated-for-prepared-use", "algorithm": "sha256",
        "text_normalization": "crlf-to-lf", "output_line_endings": "lf",
        "source_manifest": source_manifest, "source_inputs": source_inputs,
        "validation_scope": "authentication-first-and-observed-customer-owned-payments-not-complete-retail-policy",
        "files": {},
    }
    for name in RETAIL_DATASETS:
        relative = f"data/{name}"
        preserved = name in PRESERVED_SOURCE_DATASETS
        output: list[bytes] = []
        records = []
        source_diagnostics = []
        for number, line in enumerate(_jsonl_records(sources[relative], name), 1):
            original = json.loads(line)
            problems = retail_trace_diagnostics(original)
            source_diagnostics.extend({"row": number, **problem} for problem in problems)
            if preserved:
                continue
            if not isinstance(original, dict) or not isinstance(original.get("item", original), dict) or not isinstance(
                original.get("item", original).get("messages"), list
            ) or any(problem["code"] in {
                "message.invalid", "tool.invalid-calls", "tool.invalid-call", "tool.invalid-id",
                "tool.invalid-arguments", "prediction.one-call-required",
            } for problem in problems):
                raise ValueError(f"Unreviewed active trace failures remain: {name}, row {number}: {problems}.")
            curated, decisions = curate_reviewed_row(original)
            remaining = retail_trace_diagnostics(curated)
            if remaining:
                raise ValueError(f"Unreviewed active trace failures remain: {name}, row {number}: {remaining}.")
            if decisions:
                ending = b"\n" if line.endswith(b"\n") else b""
                result = json.dumps(curated, ensure_ascii=True, allow_nan=False).encode("utf-8") + ending
                records.append({
                    "row": number, "decisions": decisions,
                    "source_diagnostics": problems, "output_diagnostics": remaining,
                    "source_raw_line_sha256": hashlib.sha256(line).hexdigest(),
                    "source_lf_line_sha256": hashlib.sha256(line.replace(b"\r\n", b"\n")).hexdigest(),
                    "output_raw_line_sha256": hashlib.sha256(result).hexdigest(),
                    "source_raw_line": line.decode("utf-8"),
                })
            else:
                result = line.replace(b"\r\n", b"\n")
            output.append(result)
        descriptor = None
        if not preserved:
            content = b"".join(output)
            payloads[relative] = content
            descriptor = {"path": f"{PREPARED_DIRECTORY}/{name}", **_fingerprint(content), "records": len(output)}
        report["files"][relative] = {
            "role": "preserved-source-evidence-not-active" if preserved else "curated-active-input",
            "source": {"path": relative, **source_inputs[relative]}, "output": descriptor,
            "source_diagnostics": source_diagnostics,
            "output_diagnostics": None if preserved else [], "curation": records,
        }
    return payloads, report


def _prepared_directory(root: Path, output_dir: Path | None = None) -> Path:
    expected = root / PREPARED_DIRECTORY
    candidate = expected if output_dir is None else Path(output_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate != expected or candidate.resolve() != expected:
        raise ValueError("Prepared output path must be contained in outputs/prepared-data, without links or traversal.")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("Prepared output path must be a directory.")
    return candidate


def _check_output_file(root: Path, path: Path) -> None:
    directory = _prepared_directory(root)
    if path.parent != directory or path.resolve() != path or path.is_symlink():
        raise ValueError("Prepared output files must be contained, without symbolic links.")
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError("Prepared output files must be regular files without hard links.")


def _report_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validated_prepared_paths(root: Path, report: dict[str, Any]) -> dict[str, Path]:
    """Recheck source binding, persisted manifest, output hashes, and every active row.

    The expected report comes from preparation, never from trusting a saved
    manifest alone. Consumers must recheck before uploads or evaluation.
    """
    root = root.resolve()
    directory = _prepared_directory(root)
    if (
        report.get("schema_version") != 1 or report.get("preparation_version") != PREPARATION_VERSION
        or report.get("status") != "validated-for-prepared-use"
        or set(report.get("files", {})) != {f"data/{name}" for name in RETAIL_DATASETS}
    ):
        raise ValueError("Invalid prepared-data manifest; rerun offline preparation.")
    source_manifest, source_inputs, _ = _verified_sources(root)
    if report["source_manifest"] != source_manifest or report["source_inputs"] != source_inputs:
        raise ValueError("Prepared data belongs to different source inputs; reconcile it explicitly.")
    manifest_path = directory / "manifest.json"
    _check_output_file(root, manifest_path)
    if manifest_path.read_bytes() != _report_bytes(report):
        raise ValueError("Prepared manifest differs from the computed preparation report.")
    paths = {}
    for name in RETAIL_DATASETS:
        entry = report["files"][f"data/{name}"]
        if entry["source"] != {"path": f"data/{name}", **source_inputs[f"data/{name}"]}:
            raise ValueError("Prepared source provenance differs from the computed source hashes.")
        if name in PRESERVED_SOURCE_DATASETS:
            if entry["output"] is not None or entry["role"] != "preserved-source-evidence-not-active":
                raise ValueError("Synthetic source evidence cannot be promoted to active inputs.")
            continue
        path = directory / name
        _check_output_file(root, path)
        content = path.read_bytes()
        descriptor = {"path": path.relative_to(root).as_posix(), **_fingerprint(content)}
        expected = entry["output"]
        # Reject changed bytes before record splitting or JSON parsing. Corrupt
        # prepared data is an integrity failure, never a new dataset to consume.
        if (
            b"\r\n" in content or entry["role"] != "curated-active-input"
            or not isinstance(expected, dict)
            or any(expected.get(key) != value for key, value in descriptor.items())
            or entry["output_diagnostics"] != []
        ):
            raise ValueError(f"Prepared active artifact differs from its validated hashes: {name}.")
        lines = _jsonl_records(content, name)
        if expected != {**descriptor, "records": len(lines)}:
            raise ValueError(f"Prepared active artifact differs from its validated counts: {name}.")
        for number, line in enumerate(lines, 1):
            if retail_trace_diagnostics(json.loads(line)):
                raise ValueError(f"Prepared active trace validation failed: {name}, row {number}.")
        paths[name] = path
    return paths


def prepare_active_datasets(
    root: Path, *, output_dir: Path | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Create immutable-to-this-helper prepared copies, with the manifest last.

    Repetition verifies and reuses identical files without rewriting them.
    Different/stale outputs, links, and input paths fail before any writes; no
    force/overwrite switch exists. An interrupted identical preparation can
    complete its missing files, but a conflicting artifact needs reconciliation.
    """
    root = root.resolve()
    directory = _prepared_directory(root, output_dir)
    payloads, report = prepare_reviewed_datasets(root)
    artifacts = {directory / name: payloads[f"data/{name}"] for name in ACTIVE_RETAIL_DATASETS}
    artifacts[directory / "manifest.json"] = _report_bytes(report)
    if directory.exists() and any(path not in artifacts for path in directory.iterdir()):
        raise ValueError("Unexpected prepared artifacts require explicit reconciliation.")
    for path, content in artifacts.items():
        _check_output_file(root, path)
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Prepared artifact already exists with different bytes; will not overwrite: {path.name}.")
    directory.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        _check_output_file(root, path)
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            _check_output_file(root, path)
            if path.read_bytes() != content:
                raise ValueError(f"Prepared artifact changed concurrently; will not overwrite: {path.name}.") from None
    return validated_prepared_paths(root, report), report