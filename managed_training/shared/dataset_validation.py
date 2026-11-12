"""Dataset validation, hashing, and split-isolation checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NoReturn

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[-_ ]?key|client[-_ ]?secret)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-=]{12,}"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"),
)
_ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool"}


class DatasetValidationError(ValueError):
    """Raised when dataset validation cannot produce a usable report."""


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    line_number: int | None = None


@dataclass(slots=True)
class DatasetValidationReport:
    path: str
    sha256: str
    bytes: int
    rows: int = 0
    unique_rows: int = 0
    duplicate_rows: int = 0
    roles: dict[str, int] = field(default_factory=dict)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def require_valid(self) -> "DatasetValidationReport":
        if not self.valid:
            details = "; ".join(
                f"line {issue.line_number}: {issue.message}"
                if issue.line_number
                else issue.message
                for issue in self.issues
                if issue.severity == "error"
            )
            raise DatasetValidationError(f"{self.path} is invalid: {details}")
        return self

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    selected = Path(path)
    if not selected.is_file():
        raise DatasetValidationError(f"Dataset file does not exist: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_hash(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_content(content: Any) -> bool:
    if isinstance(content, str):
        return _nonempty_string(content)
    if not isinstance(content, list) or not content:
        return False
    for part in content:
        if not isinstance(part, dict):
            return False
        if part.get("type") == "text":
            if not _nonempty_string(part.get("text")):
                return False
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            if not isinstance(image, dict) or not _nonempty_string(image.get("url")):
                return False
        else:
            return False
    return True


def _valid_tool_calls(calls: Any) -> bool:
    if not isinstance(calls, list) or not calls:
        return False
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            return False
        function = call.get("function")
        if (
            not _nonempty_string(call.get("id"))
            or not isinstance(function, dict)
            or not _nonempty_string(function.get("name"))
            or not isinstance(function.get("arguments"), str)
        ):
            return False
    return True


def _messages_issues(
    messages: Any,
    *,
    line_number: int,
    max_content_characters: int,
) -> tuple[list[DatasetIssue], Counter[str]]:
    issues: list[DatasetIssue] = []
    roles: Counter[str] = Counter()
    if not isinstance(messages, list) or not messages:
        return (
            [
                DatasetIssue(
                    "error",
                    "messages.invalid",
                    "'messages' must be a non-empty list",
                    line_number,
                )
            ],
            roles,
        )
    pending_tool_calls: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            issues.append(
                DatasetIssue(
                    "error",
                    "message.invalid",
                    f"messages[{index}] must be an object",
                    line_number,
                )
            )
            continue
        role = message.get("role")
        if not isinstance(role, str) or role not in _ALLOWED_ROLES:
            issues.append(
                DatasetIssue(
                    "error",
                    "message.role",
                    f"messages[{index}].role is not supported: {role!r}",
                    line_number,
                )
            )
        else:
            roles[role] += 1
        if index == 0 and role == "assistant":
            issues.append(
                DatasetIssue(
                    "error",
                    "message.order",
                    "A conversation cannot start with an assistant message",
                    line_number,
                )
            )
        if pending_tool_calls and role != "tool":
            issues.append(
                DatasetIssue(
                    "error",
                    "message.tool-results",
                    f"messages[{index}] interrupts assistant tool calls without matching responses",
                    line_number,
                )
            )
            pending_tool_calls.clear()
        if role == "tool":
            identifier = message.get("tool_call_id")
            if not _nonempty_string(identifier) or identifier not in pending_tool_calls:
                issues.append(
                    DatasetIssue(
                        "error",
                        "message.tool-call-id",
                        f"messages[{index}].tool_call_id must match an outstanding assistant tool call",
                        line_number,
                    )
                )
            else:
                pending_tool_calls.remove(identifier)
        content = message.get("content")
        valid_calls = role == "assistant" and _valid_tool_calls(message.get("tool_calls"))
        if message.get("tool_calls") is not None and not valid_calls:
            issues.append(
                DatasetIssue(
                    "error",
                    "message.tool-calls",
                    f"messages[{index}].tool_calls must be non-empty assistant function calls",
                    line_number,
                )
            )
        if valid_calls:
            identifiers = [call["id"] for call in message["tool_calls"]]
            pending_tool_calls.update(identifiers)
            if len(pending_tool_calls) != len(identifiers):
                issues.append(
                    DatasetIssue(
                        "error",
                        "message.tool-call-id",
                        f"messages[{index}].tool_calls must have unique identifiers",
                        line_number,
                    )
                )
        tool_call_only = valid_calls and (
            content is None or isinstance(content, str) and not content.strip()
        )
        if not _valid_content(content) and not tool_call_only:
            issues.append(
                DatasetIssue(
                    "error",
                    "message.content",
                    f"messages[{index}].content must be a non-empty string or a non-empty "
                    "list of supported text/image_url parts (or an assistant tool call)",
                    line_number,
                )
            )
        elif len(json.dumps(content, ensure_ascii=False)) > max_content_characters:
            issues.append(
                DatasetIssue(
                    "warning",
                    "message.length",
                    f"messages[{index}] exceeds {max_content_characters} characters",
                    line_number,
                )
            )
    # An assistant-only final tool-call batch is a valid supervised target.
    # Once responses begin, however, every call in that batch must be consumed.
    if pending_tool_calls and isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant":
        issues.append(
            DatasetIssue(
                "error",
                "message.tool-results",
                "The final tool response batch is missing assistant tool-call results",
                line_number,
            )
        )
    if roles["assistant"] == 0:
        issues.append(
            DatasetIssue(
                "error",
                "messages.assistant",
                "Supervised examples require at least one assistant message",
                line_number,
            )
        )
    return issues, roles


def _reject_json_constant(value: str) -> NoReturn:
    raise json.JSONDecodeError(f"Non-finite constant {value} is not valid JSON", value, 0)


def validate_jsonl(
    path: str | Path,
    *,
    required_keys: Iterable[str] = ("messages",),
    validate_messages: bool = True,
    max_content_characters: int = 200_000,
    row_validator: Callable[[Mapping[str, Any], int], Iterable[DatasetIssue]] | None = None,
) -> DatasetValidationReport:
    """Validate JSONL syntax, schema basics, duplicates, secrets, and messages."""
    selected = Path(path)
    if not selected.is_file():
        raise DatasetValidationError(f"Dataset file does not exist: {selected}")
    if selected.stat().st_size == 0:
        raise DatasetValidationError(f"Dataset file is empty: {selected}")

    report = DatasetValidationReport(
        path=str(selected),
        sha256=hash_file(selected),
        bytes=selected.stat().st_size,
    )
    seen: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    required = tuple(required_keys)

    with selected.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                report.issues.append(
                    DatasetIssue(
                        "error",
                        "jsonl.blank-line",
                        "Blank lines are not valid JSONL records",
                        line_number,
                    )
                )
                continue
            report.rows += 1
            try:
                record = json.loads(raw_line, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as error:
                report.issues.append(
                    DatasetIssue(
                        "error",
                        "json.invalid",
                        f"Invalid JSON: {error.msg}",
                        line_number,
                    )
                )
                continue
            if not isinstance(record, dict):
                report.issues.append(
                    DatasetIssue(
                        "error",
                        "record.invalid",
                        "Each JSONL row must contain an object",
                        line_number,
                    )
                )
                continue
            missing = [key for key in required if key not in record]
            if missing:
                report.issues.append(
                    DatasetIssue(
                        "error",
                        "record.missing-keys",
                        f"Missing required key(s): {', '.join(missing)}",
                        line_number,
                    )
                )
            serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
            for pattern in _SECRET_PATTERNS:
                if pattern.search(serialized):
                    report.issues.append(
                        DatasetIssue(
                            "error",
                            "record.secret",
                            "A possible credential or secret was detected",
                            line_number,
                        )
                    )
                    break
            fingerprint = canonical_record_hash(record)
            seen[fingerprint] += 1
            if validate_messages and "messages" in record:
                issues, roles = _messages_issues(
                    record["messages"],
                    line_number=line_number,
                    max_content_characters=max_content_characters,
                )
                report.issues.extend(issues)
                role_counts.update(roles)
            if row_validator:
                report.issues.extend(row_validator(record, line_number))

    report.unique_rows = len(seen)
    report.duplicate_rows = sum(count - 1 for count in seen.values() if count > 1)
    report.roles = dict(sorted(role_counts.items()))
    if report.duplicate_rows:
        report.issues.append(
            DatasetIssue(
                "warning",
                "records.duplicate",
                f"Detected {report.duplicate_rows} duplicate row(s)",
            )
        )
    if report.rows == 0:
        report.issues.append(
            DatasetIssue("error", "records.empty", "No JSONL records were found")
        )
    return report


def _record_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for raw_line in stream:
            if raw_line.strip():
                record = json.loads(raw_line, parse_constant=_reject_json_constant)
                if isinstance(record, dict):
                    hashes.add(canonical_record_hash(record))
    return hashes


def validate_split_isolation(splits: Mapping[str, str | Path]) -> dict[str, Any]:
    """Fail when exact canonical records overlap between dataset splits."""
    if len(splits) < 2:
        raise DatasetValidationError("At least two splits are required")
    split_hashes: dict[str, set[str]] = {}
    split_files: dict[str, str] = {}
    for name, path in splits.items():
        selected = Path(path)
        validate_jsonl(selected, validate_messages=False).require_valid()
        split_hashes[name] = _record_hashes(selected)
        split_files[name] = str(selected)

    overlaps: list[dict[str, Any]] = []
    names = list(split_hashes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            count = len(split_hashes[left] & split_hashes[right])
            if count:
                overlaps.append({"left": left, "right": right, "records": count})
    if overlaps:
        details = ", ".join(
            f"{item['left']}/{item['right']}: {item['records']}"
            for item in overlaps
        )
        raise DatasetValidationError(f"Dataset split contamination detected: {details}")
    return {
        "splits": split_files,
        "record_counts": {
            name: len(values) for name, values in split_hashes.items()
        },
        "overlaps": [],
    }
