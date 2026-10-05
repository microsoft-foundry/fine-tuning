import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential


PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").rstrip("/")
if not PROJECT_ENDPOINT:
    raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT is required")

_credential = DefaultAzureCredential()
_project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=_credential,
    allow_preview=True,
)


class AsyncFoundryChildClient:
    def __init__(self, **kwargs):
        self._credential = AsyncDefaultAzureCredential()
        self._project_client = AsyncAIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=self._credential,
            allow_preview=True,
        )
        self._child_client = self._project_client.get_openai_client(**kwargs)

    def __getattr__(self, name):
        return getattr(self._child_client, name)

    async def close(self) -> None:
        await self._child_client.close()
        await self._project_client.close()
        await self._credential.close()


def get_project_client() -> AIProjectClient:
    return _project_client


def get_openai_client(**kwargs):
    return _project_client.get_openai_client(**kwargs)


def get_async_openai_client(**kwargs):
    return AsyncFoundryChildClient(**kwargs)
