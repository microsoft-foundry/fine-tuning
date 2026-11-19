"""Offline source audits and explicit, provenance-preserving preparation.

No network, model calls, generated-code execution, or in-place dataset repairs.
Run this file from the repository root; use --help for the command interface.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SAMPLES = Path(__file__).resolve().parent
REPOSITORY = SAMPLES.parent
CODE_SOURCES = {
    "train": {
        "sample": "Supervised_Fine_Tuning/Text-NL-to-Python/chat_train.jsonl",
        "maintained": "managed_training/05-data-and-distillation/02-code-distillation/data/training_data.jsonl",
        "rows": 1576,
        "sha256": "af16c47387056f9e597deeef495d444702f125d4dd7ab91dee4e4836296cc913",
        "archive_raw_sha256": "18950b69bc0d63a4f56b7194b7eafa76b34aa6acecf2aff7321e18f488280708",
    },
    "validation": {
        "sample": "Supervised_Fine_Tuning/Text-NL-to-Python/chat_validation.jsonl",
        "maintained": "managed_training/05-data-and-distillation/02-code-distillation/data/validation_data.jsonl",
        "rows": 83,
        "sha256": "558f93d59fe1d9c063f0922aec264aad3f26e96561c9ab5bcc514afb30265756",
        "archive_raw_sha256": "821a800a749279b540f9562e2ae86fca7649d088f5238f8a23256ec439b0c3cc",
    },
}
CODE_RULE = "single-assistant-target; Python-3.11-AST; enclosing-backtick-fence-only; keep-source-bytes"
FENCE = re.compile(r"\A```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\Z", re.DOTALL)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_source_bytes(data: bytes) -> bytes:
    """Normalize only physical CRLF line endings, not JSON or target content."""
    return data.replace(b"\r\n", b"\n")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_records(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    """Raw-line hashes include the original line terminator, if present."""
    raw = path.read_bytes()
    entries = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number}: blank JSONL record")
        row = loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
        entries.append({"line": line_number, "raw": line, "sha256": digest(line), "row": row})
    if not entries:
        raise ValueError(f"{path.name}: empty JSONL file")
    return raw, entries


def unwrap_code(text: str) -> str:
    """Remove only one complete conventional wrapper, never interior prose/code."""
    match = FENCE.fullmatch(text.strip())
    return match.group(1) if match else text


def code_diagnostics(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return [{"kind": "chat_shape", "message": "messages must be an array of objects"}]
    answers = [message for message in messages if message.get("role") == "assistant"]
    if (len(answers) != 1 or not isinstance(answers[0].get("content"), str)
            or not any(message.get("role") == "user" for message in messages)):
        return [{"kind": "chat_shape", "message": "expected one text assistant target and a user prompt"}]
    code = unwrap_code(answers[0]["content"])
    if not code.strip():
        return [{"kind": "empty_code", "message": "empty Python target"}]
    try:
        # Parsing only. Never compile for execution, import, eval, or exec this target.
        ast.parse(code, filename="<dataset-target>", mode="exec", feature_version=(3, 11))
    except SyntaxError as error:
        return [{"kind": "syntax", "message": error.msg, "line": error.lineno,
                 "offset": error.offset, "end_line": error.end_lineno, "end_offset": error.end_offset}]
    except ValueError as error:
        return [{"kind": "syntax", "message": str(error)}]
    return []


def audit_code_sources(repository: Path = REPOSITORY) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {
        "schema_version": 1, "rule": CODE_RULE, "parser": sys.version,
        "syntax_target": "3.11", "semantic_or_runtime_validation": False,
        "source_hash_rule": "CRLF-to-LF only; raw file/line hashes retained separately", "sources": {},
    }
    records = {}
    for split, expected in CODE_SOURCES.items():
        sample = repository / "Sample_Datasets" / expected["sample"]
        maintained = repository / expected["maintained"]
        raw, entries = read_records(sample)
        canonical = canonical_source_bytes(raw)
        if digest(canonical) != expected["sha256"] or len(entries) != expected["rows"]:
            raise ValueError(f"{split}: source bytes/count differ from the pinned provenance")
        maintained_raw = maintained.read_bytes()
        if canonical_source_bytes(maintained_raw) != canonical:
            raise ValueError(f"{split}: sample and maintained code-distillation source differ")
        decisions = []
        output_line = 0
        for entry in entries:
            diagnostics = code_diagnostics(entry["row"])
            if not diagnostics:
                output_line += 1
            decisions.append({"source_line": entry["line"], "raw_line_sha256": entry["sha256"],
                              "decision": "reject" if diagnostics else "accept",
                              "output_line": None if diagnostics else output_line,
                              "diagnostics": diagnostics})
        rejected = sum(decision["decision"] == "reject" for decision in decisions)
        report["sources"][split] = {
            "sample": sample.relative_to(repository).as_posix(),
            "maintained": expected["maintained"], "rows": len(entries), "sha256": digest(canonical),
            "raw_sha256": digest(raw), "maintained_raw_sha256": digest(maintained_raw),
            "archive_raw_sha256": expected["archive_raw_sha256"],
            "accepted": len(entries) - rejected, "rejected": rejected, "records": decisions,
        }
        records[split] = entries
    return report, records


def write_bundle(output: Path, payloads: dict[str, bytes]) -> None:
    """Create a new explicit output directory; never overwrite an existing one."""
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory already exists; choose a new versioned directory")
    if any(Path(name).name != name for name in payloads):
        raise ValueError("Bundle filenames must be simple relative filenames")
    output.mkdir(parents=True, exist_ok=False)
    # Manifests are written last. Interrupted output is not a completed contract.
    for name, content in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(content)


def curate_code(output: Path, repository: Path = REPOSITORY) -> dict[str, Any]:
    report, records = audit_code_sources(repository)
    payloads = {}
    report["status"] = "prepared_syntax_only_not_task_evaluated"
    report["artifacts"] = {}
    for split, entries in records.items():
        accepted = [entry for entry, decision in zip(entries, report["sources"][split]["records"])
                    if decision["decision"] == "accept"]
        if not accepted:
            raise ValueError(f"{split}: no syntax-accepted source records")
        name = f"chat_{split}.curated.jsonl"
        content = b"".join(entry["raw"] for entry in accepted)
        payloads[name] = content
        report["artifacts"][name] = {"rows": len(accepted), "sha256": digest(content)}
    payloads["curation-manifest.json"] = json_bytes(report)
    write_bundle(output, payloads)
    return report


def medmcq_prompt_diagnostics(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages")
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip()):
        return ["expected exactly one nonempty user prompt"]
    return []


def medmcq_diagnostics(row: dict[str, Any]) -> list[str]:
    problems = medmcq_prompt_diagnostics(row)
    if row.get("reference_answer") not in ("A", "B", "C", "D"):
        problems.append("missing/invalid reference_answer; labelled data required by grader")
    return problems


def audit_medmcq(directory: Path) -> dict[str, Any]:
    report = {"schema_version": 1, "files": {}}
    for name in ("medmcq_training.jsonl", "medmcq_validation.jsonl"):
        raw, entries = read_records(directory / name)
        failures = [{"line": entry["line"], "raw_line_sha256": entry["sha256"], "errors": errors}
                    for entry in entries if (errors := medmcq_diagnostics(entry["row"]))]
        report["files"][name] = {"rows": len(entries), "sha256": digest(raw), "failures": failures}
    report["usable_with_bundled_grader"] = not any(item["failures"] for item in report["files"].values())
    return report


def original_medmcqa_prompt(row: dict[str, Any]) -> str:
    fields = ("question", "opa", "opb", "opc", "opd")
    if any(not isinstance(row.get(field), str) for field in fields):
        raise ValueError("Original source must contain question and all four option strings")
    return row["question"] + "".join(f"\n- {letter}. {row[field]}"
                                      for letter, field in zip("ABCD", fields[1:]))


def prepare_medmcq(target: Path, labels: Path, labels_sha256: str, source_url: str,
                   source_revision: str, output: Path) -> dict[str, Any]:
    """Match original-format labels exactly; supplied provenance needs human authentication."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", labels_sha256):
        raise ValueError("An independently recorded 64-hex label-source digest is required")
    locator = urlsplit(source_url)
    if (locator.scheme != "https" or not locator.netloc or locator.username or locator.password
            or locator.query or locator.fragment):
        raise ValueError("Provide a public HTTPS source locator without credentials, query or fragment")
    if not source_revision.strip() or source_revision.strip().lower() in {"main", "master", "latest"}:
        raise ValueError("Record an immutable source revision/archive identifier, not a moving branch")
    label_raw, label_entries = read_records(labels)
    if digest(label_raw) != labels_sha256.lower():
        raise ValueError("Label-source SHA-256 mismatch")
    index: dict[str, list[dict[str, Any]]] = {}
    source_ids: set[str] = set()
    for entry in label_entries:
        row = entry["row"]
        if (row.get("choice_type") != "single" or not isinstance(row.get("id"), str) or not row["id"]
                or type(row.get("cop")) is not int or row["cop"] not in (1, 2, 3, 4)):
            raise ValueError(f"Label source line {entry['line']}: not original single-choice 1-based data")
        if row["id"] in source_ids:
            raise ValueError(f"Label source line {entry['line']}: duplicate source ID")
        source_ids.add(row["id"])
        index.setdefault(original_medmcqa_prompt(row), []).append(entry)
    target_raw, targets = read_records(target)
    prepared = []
    mappings = []
    for entry in targets:
        row = entry["row"]
        # Shape is checked independently of the label which is to be restored.
        shape_errors = medmcq_prompt_diagnostics(row)
        if shape_errors:
            raise ValueError(f"Target line {entry['line']}: {shape_errors}")
        matches = index.get(row["messages"][0]["content"], [])
        if len(matches) != 1:
            raise ValueError(f"Target line {entry['line']}: expected exactly one exact question/options match; got {len(matches)}")
        source = matches[0]
        answer = "ABCD"[source["row"]["cop"] - 1]
        if "reference_answer" in row and row["reference_answer"] != answer:
            raise ValueError(f"Target line {entry['line']}: existing label conflicts with source")
        labelled = {**row, "reference_answer": answer}
        prepared.append((json.dumps(labelled, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        mappings.append({"target_line": entry["line"], "target_raw_line_sha256": entry["sha256"],
                         "source_line": source["line"], "source_id": source["row"]["id"],
                         "source_raw_line_sha256": source["sha256"], "reference_answer": answer,
                         "output_line": len(prepared), "output_raw_line_sha256": digest(prepared[-1])})
    content = b"".join(prepared)
    report = {
        "schema_version": 1, "status": "prepared_exact_source_labels_not_independently_authenticated",
        "matching_rule": "exact rendered question and ordered options; unique original single-choice 1-based cop",
        "source": {"url": source_url, "revision": source_revision, "sha256": digest(label_raw),
                   "rows": len(label_entries)},
        "target": {"name": target.name, "sha256": digest(target_raw), "rows": len(targets)},
        "artifact": {"name": "medmcq_validation.labelled.jsonl", "sha256": digest(content), "rows": len(prepared)},
        "records": mappings,
    }
    # All matching succeeds before any output directory or artifact is created.
    write_bundle(output, {"medmcq_validation.labelled.jsonl": content,
                          "preparation-manifest.json": json_bytes(report)})
    return report


def _validate_sft_messages(row: dict[str, Any]) -> set[str]:
    """Check the local content union and reply ordering, returning pending IDs."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    roles = set()
    pending: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid message/role")
        role = message["role"]
        roles.add(role)
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list) or ("tool_calls" in message and (role != "assistant" or not calls)):
            raise ValueError("tool_calls must be a nonempty assistant array when present")
        content = message.get("content")
        if content is None:
            if role != "assistant" or not calls:
                raise ValueError("only an assistant with tool_calls may omit/null content")
        elif isinstance(content, list):
            if role not in {"user", "assistant"} or not content:
                raise ValueError("content parts require a nonempty user/assistant array")
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("invalid content part")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    continue
                image = part.get("image_url")
                if (role == "user" and part.get("type") == "image_url" and isinstance(image, dict)
                        and isinstance(image.get("url"), str) and image["url"]
                        and image.get("detail", "auto") in {"auto", "low", "high"}):
                    continue
                raise ValueError("unsupported sample content part; assistant images are not supported")
        elif not isinstance(content, str):
            raise ValueError("content must be text, supported parts, or assistant tool-call null")
        if calls:
            if pending:
                raise ValueError("tool calls before pending replies")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("invalid tool call")
                function = call.get("function")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    raise ValueError("missing/duplicate tool-call ID")
                if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                        or not function["name"] or not isinstance(function.get("arguments"), str)
                        or not isinstance(loads(function["arguments"]), dict)):
                    raise ValueError("function arguments must be an encoded JSON object")
                pending.add(call_id)
        elif role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("unmatched tool response")
            pending.remove(message["tool_call_id"])
        elif pending:
            raise ValueError("missing tool replies before next message")
    if not {"user", "assistant"}.issubset(roles):
        raise ValueError("a user prompt and assistant supervision are required")
    return pending


def validate_sft_row(row: dict[str, Any]) -> None:
    """Validate a dialogue; every tool call, including the final one, needs a reply.

    This is not the next-call prediction schema. Use validate_sft_target_row for
    the documented stock examples whose supervised answer is itself a tool call.
    Neither validator establishes service/model eligibility.
    """
    if _validate_sft_messages(row):
        raise ValueError("incomplete tool replies")


def validate_sft_target_row(row: dict[str, Any]) -> None:
    """Allow an unanswered final assistant call batch only as an explicit target.

    Earlier batches must be complete. Once replies to the final batch begin,
    all of them are required; a partially answered batch is not a prediction.
    No tool result is invented to make a target look like a completed dialogue.
    """
    pending = _validate_sft_messages(row)
    if pending and not row["messages"][-1].get("tool_calls"):
        raise ValueError("incomplete tool replies")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-code", help="Report AST decisions; exit 1 if any source target is rejected")
    curate = commands.add_parser("curate-code", help="Write unchanged accepted records plus a complete decision manifest")
    curate.add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("check-medmcq", help="Fail closed on missing/invalid labels in either bundled split")
    prepare = commands.add_parser("prepare-medmcq", help="Restore labels only from an exact authenticated original-format local source")
    prepare.add_argument("--labels", type=Path, required=True)
    prepare.add_argument("--labels-sha256", required=True)
    prepare.add_argument("--source-url", required=True)
    prepare.add_argument("--source-revision", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-code":
            report, _ = audit_code_sources()
            failed = any(source["rejected"] for source in report["sources"].values())
        elif args.command == "curate-code":
            report = curate_code(args.output_dir)
            failed = False
        elif args.command == "check-medmcq":
            report = audit_medmcq(SAMPLES / "Reinforcement_Fine_Tuning" / "MedMCQ")
            failed = not report["usable_with_bundled_grader"]
        else:
            report = prepare_medmcq(SAMPLES / "Reinforcement_Fine_Tuning" / "MedMCQ" / "medmcq_validation.jsonl",
                                    args.labels, args.labels_sha256, args.source_url, args.source_revision, args.output_dir)
            failed = False
    except (ValueError, OSError) as error:
        print(f"Dataset preparation blocked: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())