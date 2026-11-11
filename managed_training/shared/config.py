"""Validated environment configuration for Foundry cookbook notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Raised when cookbook configuration is missing or invalid."""


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, not {value!r}")


def _project_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigError(
            "FOUNDRY_PROJECT_ENDPOINT must be an HTTPS Foundry project endpoint"
        )
    path_parts = parsed.path.split("/")
    if (
        len(path_parts) != 4
        or path_parts[:3] != ["", "api", "projects"]
        or not path_parts[3].strip()
    ):
        raise ConfigError(
            "FOUNDRY_PROJECT_ENDPOINT must include '/api/projects/<project-name>'"
        )
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ConfigError(
            "FOUNDRY_PROJECT_ENDPOINT must not contain credentials, a query, or a fragment"
        )
    return endpoint


@dataclass(frozen=True, slots=True)
class FoundryConfig:
    """Configuration required to create an Entra-authenticated project context."""

    project_endpoint: str
    allow_preview: bool = False
    exclude_interactive_browser_credential: bool = True
    credential_process_timeout_seconds: int = 90

    def diagnostics(self) -> dict[str, object]:
        """Return customer-safe settings without exposing the endpoint."""
        parsed = urlparse(self.project_endpoint)
        return {
            "project_endpoint": f"{parsed.scheme}://<redacted>{parsed.path.rsplit('/', 1)[0]}/<redacted>",
            "allow_preview": self.allow_preview,
            "exclude_interactive_browser_credential": (
                self.exclude_interactive_browser_credential
            ),
            "credential_process_timeout_seconds": (
                self.credential_process_timeout_seconds
            ),
        }


def load_foundry_config(
    env_file: str | Path | None = None,
    *,
    override: bool = False,
    environ: dict[str, str] | None = None,
) -> FoundryConfig:
    """Load `.env` values and validate the common Foundry configuration.

    If ``env_file`` is supplied it must exist. Without an explicit path, a local
    ``.env`` file is optional and process environment variables remain valid.
    """

    if environ is not None and env_file is not None:
        raise ConfigError("env_file and environ cannot be supplied together")

    if environ is None:
        selected = Path(env_file) if env_file is not None else Path(".env")
        if env_file is not None and not selected.is_file():
            raise ConfigError(f"Environment file does not exist: {selected}")
        if selected.is_file():
            try:
                from dotenv import load_dotenv
            except ImportError as error:
                raise ConfigError(
                    "python-dotenv is required to load an environment file"
                ) from error
            load_dotenv(selected, override=override)
        values = os.environ
    else:
        values = environ

    endpoint = values.get("FOUNDRY_PROJECT_ENDPOINT", "")
    if not endpoint.strip():
        raise ConfigError(
            "FOUNDRY_PROJECT_ENDPOINT is required. Copy shared/templates/.env.template "
            "to .env and set the Foundry project endpoint."
        )

    timeout_raw = values.get("FOUNDRY_CREDENTIAL_PROCESS_TIMEOUT_SECONDS") or "90"
    try:
        timeout = int(timeout_raw)
    except ValueError as error:
        raise ConfigError(
            "FOUNDRY_CREDENTIAL_PROCESS_TIMEOUT_SECONDS must be an integer"
        ) from error
    if not 1 <= timeout <= 600:
        raise ConfigError(
            "FOUNDRY_CREDENTIAL_PROCESS_TIMEOUT_SECONDS must be between 1 and 600"
        )

    return FoundryConfig(
        project_endpoint=_project_endpoint(endpoint),
        allow_preview=_parse_bool(
            "FOUNDRY_ALLOW_PREVIEW", values.get("FOUNDRY_ALLOW_PREVIEW") or "false"
        ),
        exclude_interactive_browser_credential=_parse_bool(
            "FOUNDRY_EXCLUDE_INTERACTIVE_BROWSER_CREDENTIAL",
            values.get("FOUNDRY_EXCLUDE_INTERACTIVE_BROWSER_CREDENTIAL") or "true",
        ),
        credential_process_timeout_seconds=timeout,
    )
