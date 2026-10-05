"""Microsoft Foundry SDK 2.x client helpers."""

import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def create_project_client(*, allow_preview: bool = False) -> AIProjectClient:
    """Create an Entra-authenticated Foundry project client."""
    return AIProjectClient(
        endpoint=required_env("FOUNDRY_PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
        allow_preview=allow_preview,
    )


def create_openai_client(project_client: AIProjectClient, *, agent_name: str | None = None):
    """Return the project client's documented child for model operations."""
    return project_client.get_openai_client(agent_name=agent_name)
