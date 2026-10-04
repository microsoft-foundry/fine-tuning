import asyncio
import os

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AsyncOpenAI, OpenAI


PROJECT_ENDPOINT = os.getenv(
    "AZURE_AI_PROJECT_ENDPOINT",
    "https://eastus2-prakharg-demo-2026.services.ai.azure.com/api/projects/eastus2-prakharg-demo-2026",
).rstrip("/")
TOKEN_SCOPE = "https://ai.azure.com/.default"

_credential = AzureCliCredential(process_timeout=120)
_token_provider = get_bearer_token_provider(_credential, TOKEN_SCOPE)


async def _async_token_provider() -> str:
    return await asyncio.to_thread(_token_provider)


def get_project_client() -> AIProjectClient:
    return AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=_credential, allow_preview=True)


def get_openai_client(**kwargs) -> OpenAI:
    return get_project_client().get_openai_client(**kwargs)


def get_async_openai_client(**kwargs) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=f"{PROJECT_ENDPOINT}/openai/v1",
        api_key=_async_token_provider,
        **kwargs,
    )
