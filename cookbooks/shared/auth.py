"""Entra ID and Foundry project-client lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FoundryConfig


@dataclass(slots=True)
class ProjectContext:
    """Own a credential, project client, and documented model-operations child."""

    config: FoundryConfig
    credential: Any
    project_client: Any
    openai_client: Any
    _owns_credential: bool = True

    def close(self) -> None:
        errors: list[Exception] = []
        for resource in (self.openai_client, self.project_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        if self._owns_credential:
            close = getattr(self.credential, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise RuntimeError(
                "One or more Foundry context resources failed to close: "
                + "; ".join(str(error) for error in errors)
            )

    def __enter__(self) -> "ProjectContext":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_project_context(
    config: FoundryConfig,
    *,
    credential: Any | None = None,
    agent_name: str | None = None,
) -> ProjectContext:
    """Create the current SDK 2.x Foundry project and OpenAI-compatible clients."""

    try:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
    except ImportError as error:
        raise RuntimeError(
            "Install the central cookbook environment before creating a project context"
        ) from error

    owns_credential = credential is None
    if credential is None:
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=(
                config.exclude_interactive_browser_credential
            ),
            process_timeout=config.credential_process_timeout_seconds,
        )
    try:
        project_client = AIProjectClient(
            endpoint=config.project_endpoint,
            credential=credential,
            allow_preview=config.allow_preview,
        )
        openai_client = project_client.get_openai_client(agent_name=agent_name)
    except Exception:
        if owns_credential:
            close = getattr(credential, "close", None)
            if callable(close):
                close()
        raise
    return ProjectContext(
        config=config,
        credential=credential,
        project_client=project_client,
        openai_client=openai_client,
        _owns_credential=owns_credential,
    )
