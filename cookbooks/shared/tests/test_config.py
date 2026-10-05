from __future__ import annotations

import pytest

from shared.config import ConfigError, load_foundry_config


def test_loads_valid_environment() -> None:
    config = load_foundry_config(
        environ={
            "FOUNDRY_PROJECT_ENDPOINT": (
                "https://unit-test.invalid/api/projects/sample"
            ),
            "FOUNDRY_ALLOW_PREVIEW": "true",
        }
    )
    assert config.allow_preview is True
    assert "<redacted>" in str(config.diagnostics())
    assert "unit-test.invalid" not in str(config.diagnostics())


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://example.invalid/api/projects/sample",
        "https://example.invalid/not-a-project",
        "https://user:secret@example.invalid/api/projects/sample",
    ],
)
def test_rejects_invalid_endpoint(endpoint: str) -> None:
    with pytest.raises(ConfigError):
        load_foundry_config(environ={"FOUNDRY_PROJECT_ENDPOINT": endpoint})
