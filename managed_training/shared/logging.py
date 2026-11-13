"""Customer-readable structured logging with defensive redaction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

_GUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https://[^\s\"']+", re.IGNORECASE)
_SECRET_NAMES = (
    r"api[-_ ]?keys?|(?:proxy[-_ ]?)?authorization|client[-_ ]?secrets?|"
    r"(?:(?:access|refresh|id|auth|bearer)[-_ ]?)?tokens?"
)
_SECRET_KEY = re.compile(rf"(?:{_SECRET_NAMES})", re.IGNORECASE)
_SECRET = re.compile(
    rf"\b({_SECRET_NAMES})\b([\"']?\s*[:=]\s*|\s+)(?:bearer\s+)?"
    r'''("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}\]]+)''',
    re.IGNORECASE,
)
_RESOURCE_ID = re.compile(
    r"\b(?:ftjob|file|eval|run|deploy|resp|response)-[A-Za-z0-9_-]{6,}\b",
    re.IGNORECASE,
)


def safe_reference(value: str) -> str:
    """Return a stable short reference suitable for customer-visible logs."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"ref-{digest}"


def redact_text(value: str, *, sensitive_values: tuple[str, ...] = ()) -> str:
    """Redact endpoints, secrets, GUIDs, and service identifiers from text."""
    redacted = value
    # Whole JSON messages can carry object/list-valued secrets too. The regex
    # below also handles quoted keys embedded in ordinary exception prose.
    try:
        structured = json.loads(value)
    except (ValueError, TypeError):
        structured = None
    if isinstance(structured, (dict, list)):
        redacted = json.dumps(_redact_fields(structured), ensure_ascii=False)
    for sensitive in sensitive_values:
        if sensitive:
            redacted = redacted.replace(sensitive, "<redacted>")

    def hide_secret(match: re.Match[str]) -> str:
        value = match.group(3)
        quote = value[0] if value[0] in {"'", '"'} else ""
        return f"{match.group(1)}{match.group(2)}{quote}<redacted>{quote}"

    redacted = _SECRET.sub(hide_secret, redacted)
    redacted = _URL.sub("<redacted-url>", redacted)
    redacted = _GUID.sub("<redacted-guid>", redacted)
    return _RESOURCE_ID.sub("<redacted-id>", redacted)


def _redact_fields(value: Any) -> Any:
    """Scrub structured secrets before JSON serialization, without mutation."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if _SECRET_KEY.fullmatch(str(key).strip())
                else _redact_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_fields(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


class RedactingFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> None:
        super().__init__(fmt)
        self.sensitive_values = sensitive_values

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(
            super().format(record),
            sensitive_values=self.sensitive_values,
        )


def get_logger(
    name: str = "fine-tuning-cookbook",
    *,
    level: int = logging.INFO,
    sensitive_values: tuple[str, ...] = (),
) -> logging.Logger:
    """Return a configured logger without adding duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(RedactingFormatter(sensitive_values=sensitive_values))
        logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    *,
    operation: str,
    state: str,
    message: str,
    fields: Mapping[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    """Log a consistent operation transition with JSON-compatible fields."""
    payload = {
        "operation": operation,
        "state": state,
        "message": message,
    }
    if fields:
        payload["details"] = dict(fields)
    # Do not rely on a particular handler/formatter: callers can supply any
    # logger, and secret values can themselves be dictionaries or lists.
    logger.log(
        level,
        redact_text(json.dumps(_redact_fields(payload), default=str, sort_keys=True)),
    )
