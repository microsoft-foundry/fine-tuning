from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
LOGGER = logging.getLogger("agentic_rft")


@dataclass(frozen=True)
class ValidatedSplits:
    train_path: Path
    validation_path: Path
    train_rows: int
    validation_rows: int


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def validate_conversation_rows(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
    required_fields: tuple[str, ...] = ("messages", "tools"),
) -> None:
    for index, row in enumerate(rows):
        for field in required_fields:
            if field not in row:
                raise ValueError(f"{label}[{index}] is missing {field}")
        if not isinstance(row["messages"], list) or not row["messages"]:
            raise ValueError(f"{label}[{index}].messages must be non-empty")
        if not isinstance(row["tools"], list) or not row["tools"]:
            raise ValueError(f"{label}[{index}].tools must be non-empty")


def validate_exact_generated_output(
    train_path: Path,
    validation_path: Path,
) -> ValidatedSplits:
    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    validate_conversation_rows(train_rows, label="generated train")
    validate_conversation_rows(validation_rows, label="generated validation")
    return ValidatedSplits(
        train_path=train_path.resolve(),
        validation_path=validation_path.resolve(),
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value, received {value!r}")


def require_env(*names: str) -> dict[str, str]:
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Set the required environment variables: " + ", ".join(missing)
        )
    return values


def normalize_tool_call(message: Any) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    if not tool_calls:
        raise RuntimeError("The model response did not contain a tool call")
    call = tool_calls[0]
    function = getattr(call, "function", None)
    if function is None and isinstance(call, dict):
        function = call.get("function", {})
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if isinstance(function, dict):
        name = name or function.get("name")
        arguments = arguments or function.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, str):
        raise RuntimeError("The tool call is missing a function name or arguments")
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The model returned invalid JSON tool arguments") from exc
    return {"name": name, "arguments": parsed_arguments}


def expected_tool_call(row: dict[str, Any]) -> dict[str, Any]:
    expected = row["item"]["expected_output"]["tool_calls"][0]["function"]
    return {
        "name": expected["name"],
        "arguments": json.loads(expected["arguments"]),
    }


def evaluate_next_tool_calls(
    rows: Iterable[dict[str, Any]],
    predict: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    results = []
    for index, row in enumerate(rows):
        actual = predict(row)
        expected = expected_tool_call(row)
        results.append(
            {
                "index": index,
                "actual": actual,
                "expected": expected,
                "exact": actual == expected,
                "tool_name_match": actual.get("name") == expected["name"],
            }
        )
    return {
        "exact": sum(result["exact"] for result in results),
        "tool_name_match": sum(result["tool_name_match"] for result in results),
        "total": len(results),
        "results": results,
    }


def resolve_tool_endpoints(config_path: Path, server_url: str) -> list[dict[str, Any]]:
    if not server_url.startswith("https://"):
        raise ValueError("Remote tool endpoints must use HTTPS")
    tools = load_json(config_path)
    if not isinstance(tools, list):
        raise ValueError("Tool configuration must be a list")
    rendered = json.loads(json.dumps(tools).replace("#TOOLS_SERVER_URL#", server_url.rstrip("/")))
    if any(not tool.get("server_url", "").startswith("https://") for tool in rendered):
        raise ValueError("Every resolved tool must have an HTTPS server_url")
    return rendered


def retry(
    operation: Callable[[], T],
    *,
    attempts: int = 4,
    initial_delay_seconds: float = 1.0,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
) -> T:
    delay = initial_delay_seconds
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retry_on:
            if attempt == attempts:
                raise
            LOGGER.warning("Attempt %s/%s failed; retrying in %.1fs", attempt, attempts, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")
