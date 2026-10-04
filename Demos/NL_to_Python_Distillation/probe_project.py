import json

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


PROJECT_ENDPOINT = "https://swec-prakharg-demo-2026.services.ai.azure.com/api/projects/swec-prakharg-demo-2026"
DEPLOYMENTS = ("gpt-5-4", "gpt-4.1-mini")

project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
client = project.get_openai_client()

result = {"base_url": str(client.base_url), "deployments": {}, "fine_tuning_jobs": []}
for deployment in DEPLOYMENTS:
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_completion_tokens=16,
        )
        result["deployments"][deployment] = {
            "status": "succeeded",
            "response": response.choices[0].message.content,
        }
    except Exception as exc:
        result["deployments"][deployment] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

try:
    result["fine_tuning_jobs"] = [
        {
            "id": job.id,
            "model": job.model,
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "created_at": job.created_at,
        }
        for job in client.fine_tuning.jobs.list(limit=20).data
    ]
except Exception as exc:
    result["fine_tuning_error"] = f"{type(exc).__name__}: {exc}"

with open("project_probe.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, default=str)

print(json.dumps(result, indent=2, default=str))
