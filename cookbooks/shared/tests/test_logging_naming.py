from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shared.logging import redact_text, safe_reference
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
