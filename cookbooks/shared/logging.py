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
_SECRET = re.compile(
    r"(?i)\b(api[-_ ]?key|authorization|client[-_ ]?secret|token)\b"
    r"(\s*[:=]\s*|\s+)(bearer\s+)?[^\s,;]+"
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
    for sensitive in sensitive_values:
        if sensitive:
            redacted = redacted.replace(sensitive, "<redacted>")
    redacted = _SECRET.sub(r"\1\2<redacted>", redacted)
    redacted = _URL.sub("<redacted-url>", redacted)
    redacted = _GUID.sub("<redacted-guid>", redacted)
    return _RESOURCE_ID.sub("<redacted-id>", redacted)


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
    logger.log(level, json.dumps(payload, default=str, sort_keys=True))
