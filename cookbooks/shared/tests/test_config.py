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


def test_blank_optional_template_values_use_documented_defaults() -> None:
    config = load_foundry_config(
        environ={
            "FOUNDRY_PROJECT_ENDPOINT": "https://example.invalid/api/projects/sample",
            "FOUNDRY_ALLOW_PREVIEW": "",
            "FOUNDRY_EXCLUDE_INTERACTIVE_BROWSER_CREDENTIAL": "",
            "FOUNDRY_CREDENTIAL_PROCESS_TIMEOUT_SECONDS": "",
        }
    )
    assert not config.allow_preview
    assert config.exclude_interactive_browser_credential
    assert config.credential_process_timeout_seconds == 90


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://example.invalid/api/projects/sample",
        "https://example.invalid/not-a-project",
        "https://example.invalid/api/projects/",
        "https://example.invalid/api/projects/sample/openai/v1",
        "https://example.invalid/other/api/projects/sample",
        "https://example.invalid/api/projects/sample?key=private",
        "https://user:password@example.invalid/api/projects/sample",
        "https://user:secret@example.invalid/api/projects/sample",
    ],
)
def test_rejects_invalid_endpoint(endpoint: str) -> None:
    with pytest.raises(ConfigError):
        load_foundry_config(environ={"FOUNDRY_PROJECT_ENDPOINT": endpoint})
