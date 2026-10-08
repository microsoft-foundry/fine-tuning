from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
LOGGER = logging.getLogger("agentic_rft")
SKU_PATTERN = re.compile(r"^[A-Z]{3}-[A-Z]{3}-\d{3}$")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def sha256_file(path: Path, *, normalize_line_endings: bool = False) -> str:
    if normalize_line_endings:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_tool_calling_rows(rows: Iterable[dict[str, Any]], label: str) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        tools = row.get("tools")
        reference = row.get("reference_answer")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{label}[{index}] requires non-empty messages")
        if not isinstance(tools, list) or len(tools) != 1:
            raise ValueError(f"{label}[{index}] must expose exactly one tool")
        function = tools[0].get("function", {})
        if function.get("name") != "search_catalog":
            raise ValueError(f"{label}[{index}] has an unexpected tool schema")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"{label}[{index}] requires reference_answer")


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


def response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
    return ""


def response_function_calls(response: Any) -> list[dict[str, Any]]:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output", [])
    calls: list[dict[str, Any]] = []
    for item in output or []:
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type != "function_call":
            continue
        name = getattr(item, "name", None)
        arguments = getattr(item, "arguments", None)
        call_id = getattr(item, "call_id", None)
        if isinstance(item, dict):
            name = name or item.get("name")
            arguments = arguments or item.get("arguments")
            call_id = call_id or item.get("call_id")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise RuntimeError("Function-call response is missing name or arguments")
        calls.append({"name": name, "arguments": arguments, "call_id": call_id})
    return calls


def responses_tools(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Expected a chat-style function tool")
        function = tool["function"]
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError("Function tool requires name and parameters")
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": function.get("description", ""),
                "parameters": parameters,
            }
        )
    return converted


def responses_input(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"developer", "system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("Message content must be a string")
        converted.append(
            {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }
        )
    return converted


def grade_sku_answer(actual: str, reference: str) -> dict[str, float]:
    actual = actual.strip()
    reference = reference.strip()
    exact = float(actual == reference and bool(reference))
    fuzzy = SequenceMatcher(None, actual.casefold(), reference.casefold()).ratio()
    score = 0.9 * exact + 0.1 * fuzzy
    if not SKU_PATTERN.fullmatch(actual):
        score = 0.0
    return {"exact": exact, "fuzzy": round(fuzzy, 4), "score": round(score, 4)}


def evaluate_sku_answers(
    rows: Iterable[dict[str, Any]],
    predict: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    results = []
    for index, row in enumerate(rows):
        actual = predict(row).strip()
        expected = row["reference_answer"].strip()
        results.append(
            {
                "index": index,
                "actual": actual,
                "expected": expected,
                "exact": actual == expected,
            }
        )
    return {
        "exact": sum(result["exact"] for result in results),
        "total": len(results),
        "results": results,
    }


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
