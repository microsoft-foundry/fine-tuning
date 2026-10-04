import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


PROJECT_ENDPOINT = "https://eastus2-prakharg-demo-2026.services.ai.azure.com/api/projects/eastus2-prakharg-demo-2026"
SUBSCRIPTION_ID = "ba7979f7-d040-49c9-af1a-7414402bf622"
RESOURCE_GROUP = "prakharg-demo-2026"
ACCOUNT_NAME = "eastus2-prakharg-demo-2026"
BASE_DEPLOYMENT = "image-breed-gpt-41-mini-base"
TEACHER_DEPLOYMENT = "gpt-5.4-mini"
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
TRAINING_FILE = Path("training_data.jsonl")
VALIDATION_FILE = Path("validation_data.jsonl")
RESULT_PATH = Path("eastus2_result.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_result(result):
    RESULT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


credential = DefaultAzureCredential()
project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)
client = project.get_openai_client()

deployments = {item.name: item for item in project.deployments.list()}
for required in (BASE_DEPLOYMENT, TEACHER_DEPLOYMENT):
    if required not in deployments:
        raise RuntimeError(f"Required deployment {required!r} is unavailable. Found: {sorted(deployments)}")

result = {
    "started_at": utc_now(),
    "completed_at": None,
    "project_endpoint": PROJECT_ENDPOINT,
    "resource": {
        "subscription_id": SUBSCRIPTION_ID,
        "resource_group": RESOURCE_GROUP,
        "account_name": ACCOUNT_NAME,
        "location": "eastus2",
    },
    "authentication": {"type": "DefaultAzureCredential", "secrets_used": False},
    "models": {
        "student_deployment": BASE_DEPLOYMENT,
        "student_model": BASE_MODEL,
        "teacher_deployment": TEACHER_DEPLOYMENT,
    },
    "files": {},
    "job": None,
    "deployment": None,
    "evaluation": None,
    "errors": [],
    "fixes": [
        "Reused the final reduced Sweden-validated 11-record training file and 1-record validation file.",
        "Selected the East US 2 GlobalStandard gpt-4.1-mini deployment as the supported student.",
    ],
}
write_result(result)

for deployment_name in (BASE_DEPLOYMENT, TEACHER_DEPLOYMENT):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_completion_tokens=16,
    )
    result["models"][f"{deployment_name}_connectivity"] = response.choices[0].message.content

with TRAINING_FILE.open("rb") as handle:
    training_upload = client.files.create(file=handle, purpose="fine-tune")
with VALIDATION_FILE.open("rb") as handle:
    validation_upload = client.files.create(file=handle, purpose="fine-tune")

result["files"] = {
    "training": {
        "path": str(TRAINING_FILE),
        "records": sum(1 for line in TRAINING_FILE.read_text(encoding="utf-8").splitlines() if line.strip()),
        "file_id": training_upload.id,
    },
    "validation": {
        "path": str(VALIDATION_FILE),
        "records": sum(1 for line in VALIDATION_FILE.read_text(encoding="utf-8").splitlines() if line.strip()),
        "file_id": validation_upload.id,
    },
}
write_result(result)


for label, uploaded in (("training", training_upload), ("validation", validation_upload)):
    for _ in range(60):
        file_state = client.files.retrieve(uploaded.id)
        result["files"][label]["status"] = file_state.status
        write_result(result)
        print(f"{label} file status: {file_state.status}")
        if file_state.status == "processed":
            break
        if file_state.status == "error":
            raise RuntimeError(f"{label} file import failed: {file_state}")
        time.sleep(10)
    else:
        raise TimeoutError(f"{label} file did not finish processing within 10 minutes.")

job = client.fine_tuning.jobs.create(
    model=BASE_MODEL,
    training_file=training_upload.id,
    validation_file=validation_upload.id,
    suffix="nl2py-eastus2",
    hyperparameters={
        "n_epochs": 1,
        "learning_rate_multiplier": 1.3,
        "batch_size": 1,
    },
    extra_body={"trainingType": "globalStandard"},
)

result["job"] = {
    "job_id": job.id,
    "status": job.status,
    "created_at": datetime.fromtimestamp(job.created_at, timezone.utc).isoformat(),
    "training_type": "globalStandard",
    "base_model": job.model,
    "training_file_id": training_upload.id,
    "validation_file_id": validation_upload.id,
    "fine_tuned_model": None,
    "trained_tokens": None,
    "result_files": [],
    "error": None,
    "events": [],
}
write_result(result)
print(f"Submitted {job.id}")

last_status = None
connection_failures = 0
while True:
    try:
        job = client.fine_tuning.jobs.retrieve(job.id)
        connection_failures = 0
    except Exception as exc:
        connection_failures += 1
        print(f"Monitoring request failed ({connection_failures}/20): {type(exc).__name__}: {exc}")
        if connection_failures >= 20:
            raise
        time.sleep(30)
        continue

    if job.status != last_status:
        print(f"Status: {job.status}")
        last_status = job.status

    result["job"].update(
        {
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
            "result_files": list(job.result_files or []),
            "finished_at": (
                datetime.fromtimestamp(job.finished_at, timezone.utc).isoformat()
                if job.finished_at
                else None
            ),
            "error": job.error.model_dump() if job.error else None,
        }
    )
    try:
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=20).data
        result["job"]["events"] = [
            {
                "created_at": datetime.fromtimestamp(event.created_at, timezone.utc).isoformat(),
                "level": event.level,
                "message": event.message,
            }
            for event in events
        ]
    except Exception as exc:
        print(f"Event retrieval failed: {type(exc).__name__}: {exc}")
    write_result(result)

    if job.status in {"succeeded", "failed", "cancelled"}:
        break
    time.sleep(30)

if job.status != "succeeded":
    result["completed_at"] = utc_now()
    result["status"] = f"fine_tuning_{job.status}"
    if job.error:
        result["errors"].append(
            {
                "stage": "fine_tuning",
                "code": job.error.code,
                "message": job.error.message,
            }
        )
    write_result(result)
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0)

deployment_name = f"nl2py-eus2-{job.id[-10:]}"
deployment_url = (
    f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    f"/providers/Microsoft.CognitiveServices/accounts/{ACCOUNT_NAME}"
    f"/deployments/{deployment_name}?api-version=2024-10-01"
)
deployment_attempts = []
for sku in ("GlobalStandard", "Standard"):
    token = credential.get_token("https://management.azure.com/.default").token
    body = {
        "sku": {"name": sku, "capacity": 10},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": job.fine_tuned_model,
                "version": "1",
            }
        },
    }
    response = requests.put(
        deployment_url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    deployment_attempts.append(
        {
            "sku": sku,
            "status_code": response.status_code,
            "response": response.text[:4000],
        }
    )
    if response.ok:
        break

if not response.ok:
    result["deployment"] = {
        "name": deployment_name,
        "status": "failed",
        "attempts": deployment_attempts,
    }
    result["errors"].append(
        {
            "stage": "deployment",
            "code": str(response.status_code),
            "message": response.text[:4000],
        }
    )
    result["completed_at"] = utc_now()
    result["status"] = "fine_tuning_succeeded_deployment_failed"
    write_result(result)
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0)

for _ in range(60):
    token = credential.get_token("https://management.azure.com/.default").token
    response = requests.get(
        deployment_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    deployment_state = response.json()["properties"]["provisioningState"]
    print(f"Deployment state: {deployment_state}")
    if deployment_state == "Succeeded":
        break
    if deployment_state in {"Failed", "Canceled"}:
        raise RuntimeError(response.text)
    time.sleep(15)
else:
    raise TimeoutError("Fine-tuned deployment did not become ready within 15 minutes.")

result["deployment"] = {
    "name": deployment_name,
    "status": "succeeded",
    "sku": deployment_attempts[-1]["sku"],
    "url": deployment_url,
    "attempts": deployment_attempts,
}
write_result(result)

validation_record = json.loads(
    next(line for line in VALIDATION_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
)
system_prompt = next(item["content"] for item in validation_record["messages"] if item["role"] == "system")
prompt = next(item["content"] for item in validation_record["messages"] if item["role"] == "user")
reference = next(item["content"] for item in validation_record["messages"] if item["role"] == "assistant")


def generate(deployment):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content


base_output = generate(BASE_DEPLOYMENT)
fine_tuned_output = generate(deployment_name)
judge_prompt = f"""Evaluate two Python solutions for the task below against the reference.
Return strict JSON with keys base_correctness, base_conciseness, fine_tuned_correctness,
fine_tuned_conciseness, preferred, and explanation. Scores must be integers 1-10.

TASK:
{prompt}

REFERENCE:
{reference}

BASE:
{base_output}

FINE_TUNED:
{fine_tuned_output}
"""
judge_response = client.chat.completions.create(
    model=TEACHER_DEPLOYMENT,
    messages=[{"role": "user", "content": judge_prompt}],
    max_completion_tokens=1024,
)
judge_text = judge_response.choices[0].message.content
try:
    match = judge_text[judge_text.index("{") : judge_text.rindex("}") + 1]
    judge = json.loads(match)
except Exception:
    judge = {"raw": judge_text}

result["evaluation"] = {
    "prompt": prompt,
    "reference": reference,
    "base_deployment": BASE_DEPLOYMENT,
    "fine_tuned_deployment": deployment_name,
    "base_output": base_output,
    "fine_tuned_output": fine_tuned_output,
    "teacher_deployment": TEACHER_DEPLOYMENT,
    "judge": judge,
}
result["completed_at"] = utc_now()
result["status"] = "succeeded"
write_result(result)
print(json.dumps(result, indent=2, default=str))
