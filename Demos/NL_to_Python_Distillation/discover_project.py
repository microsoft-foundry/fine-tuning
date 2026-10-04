import json
from dataclasses import asdict, is_dataclass

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


PROJECT_ENDPOINT = "https://swec-prakharg-demo-2026.services.ai.azure.com/api/projects/swec-prakharg-demo-2026"


def serialize(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {key: serialize(item) for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value


credential = DefaultAzureCredential()
project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)

result = {
    "project_endpoint": PROJECT_ENDPOINT,
    "connections": [],
    "deployments": [],
}

try:
    result["connections"] = [serialize(item) for item in project.connections.list()]
except Exception as exc:
    result["connections_error"] = f"{type(exc).__name__}: {exc}"

try:
    result["deployments"] = [serialize(item) for item in project.deployments.list()]
except Exception as exc:
    result["deployments_error"] = f"{type(exc).__name__}: {exc}"

try:
    openai_client = project.get_openai_client()
    result["openai_models"] = [serialize(item) for item in openai_client.models.list()]
except Exception as exc:
    result["openai_models_error"] = f"{type(exc).__name__}: {exc}"

with open("project_discovery.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, default=str)

print(json.dumps(result, indent=2, default=str))
