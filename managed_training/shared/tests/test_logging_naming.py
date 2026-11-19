from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime

import pytest

from shared.logging import RedactingFormatter, log_event, redact_text, safe_reference
from shared.naming import NameFactory, slugify


def test_redacts_customer_identifiers() -> None:
    text = (
        "Authorization: Bearer secret-token "
        "https://unit-test.invalid/api/projects/private "
        "ftjob-abcdefghijkl"
    )
    redacted = redact_text(text)
    assert "secret-token" not in redacted
    assert "unit-test.invalid" not in redacted
    assert "ftjob-" not in redacted
    assert safe_reference("ftjob-abcdefghijkl").startswith("ref-")


def test_meaningful_names_are_deterministic() -> None:
    factory = NameFactory.for_demo(
        "News Summarization",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    assert factory.remote("training job") == (
        "news-summarization-training-job-20260102-030405"
    )
    assert factory.local("representative metrics", "JSON") == (
        "news-summarization-representative-metrics-20260102-030405.json"
    )
    with pytest.raises(ValueError):
        slugify("***")


@pytest.mark.parametrize(
    "key",
    ["authorization", "Authorization", "tokens", "token", "api_key", "api-key",
     "client_secret", "client-secret", "access_token", "refresh_token"],
)
@pytest.mark.parametrize("quote", ['"', "'"])
def test_text_redacts_quoted_secret_keys_and_space_containing_values(key, quote) -> None:
    text = f"{{{quote}{key}{quote}: {quote}Bearer private secret value{quote}, {quote}safe{quote}: 7}}"
    redacted = redact_text(text)
    assert "private" not in redacted
    assert "secret value" not in redacted
    assert f"{quote}<redacted>{quote}" in redacted
    assert f"{quote}safe{quote}: 7" in redacted


def test_text_redacts_json_escaped_quotes_without_leaking_tail() -> None:
    text = json.dumps({"api_key": 'private "quoted" secret', "safe": "preserved"})
    result = json.loads(redact_text(text))
    assert result == {"api_key": "<redacted>", "safe": "preserved"}


@pytest.mark.parametrize("formatter", [logging.Formatter("%(message)s"), RedactingFormatter("%(message)s")])
def test_log_event_redacts_nested_structures_before_any_handler(formatter) -> None:
    stream = io.StringIO()
    logger = logging.Logger("isolated-redaction-test", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    fields = {
        "authorization": "Bearer auth-private",
        "tokens": ["list-private", {"safe": "also-private"}],
        "nested": [
            {"api_key": "key-private", "safe_count": 3},
            {"client_secret": {"nested": "object-private"}, "message": 'failed {"token": "text-private"}'},
        ],
    }
    log_event(
        logger, operation="upload", state="failed",
        message='response {"authorization": "Bearer message-private"}', fields=fields,
    )
    text = stream.getvalue()
    assert "private" not in text
    payload = json.loads(text)
    assert payload["details"]["authorization"] == "<redacted>"
    assert payload["details"]["tokens"] == "<redacted>"
    assert payload["details"]["nested"][0] == {"api_key": "<redacted>", "safe_count": 3}
    assert payload["details"]["nested"][1]["client_secret"] == "<redacted>"
    assert fields["authorization"] == "Bearer auth-private"  # Never mutate caller data.


def test_formatter_alone_redacts_json_secret_fields() -> None:
    record = logging.LogRecord(
        "isolated", logging.ERROR, __file__, 1,
        json.dumps({"authorization": "Bearer auth-private", "client_secret": "client-private"}),
        (), None,
    )
    result = RedactingFormatter("%(message)s").format(record)
    assert "private" not in result
    assert json.loads(result)["authorization"] == "<redacted>"


def test_raw_json_redaction_covers_object_and_list_valued_secrets() -> None:
    payload = {
        "tokens": ["first-private", {"token": "second-private"}],
        "nested": [{"client_secret": {"key": "third-private"}, "safe": 7}],
    }
    result = redact_text(json.dumps(payload))
    assert "private" not in result
    assert json.loads(result) == {
        "tokens": "<redacted>", "nested": [{"client_secret": "<redacted>", "safe": 7}],
    }
