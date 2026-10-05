"""Live validation for the synthetic tool-use SFT cookbook.

This runner intentionally has no preserved-data or prior-job fallback. Every
regional attempt generates fresh rows and trains only on those rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    DataGenerationJob,
    DataGenerationJobInputs,
    DataGenerationJobOutputOptions,
    DataGenerationJobScenario,
    DataGenerationModelOptions,
    FileDataGenerationJobSource,
    ToolUseFineTuningDataGenerationJobOptions,
)
from azure.identity import AzureCliCredential


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUTS = HERE / "outputs"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = OUTPUTS / RUN_ID

STUDENT_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_SAMPLES_PER_BATCH = 60
GENERATION_BATCHES = 3
POLL_SECONDS = 15
FILE_TIMEOUT_SECONDS = 900
DATAGEN_TIMEOUT_SECONDS = 3600
JOB_TIMEOUT_SECONDS = 14_400

def project_config(
    region: str,
    endpoint: str,
    subscription_id: str,
    resource_group: str,
) -> dict[str, str]:
    normalized = endpoint.rstrip("/")
    host = normalized.split("://", 1)[-1].split("/", 1)[0]
    resource = host.split(".services.ai.azure.com", 1)[0]
    project = normalized.rsplit("/", 1)[-1]
    if not resource or not project:
        raise ValueError(f"Invalid Foundry project endpoint: {endpoint}")
    return {
        "region": region,
        "endpoint": normalized,
        "resource": resource,
        "project": project,
        "resource_group": resource_group,
        "subscription": subscription_id,
    }


def configured_projects() -> list[dict[str, str]]:
    subscription_id = os.environ["FOUNDRY_SUBSCRIPTION_ID"]
    resource_group = os.environ["FOUNDRY_RESOURCE_GROUP"]
    configured = [
        project_config(
            "primary",
            os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            subscription_id,
            resource_group,
        )
    ]
    for region, variable in (
        ("eastus2", "FOUNDRY_EASTUS2_PROJECT_ENDPOINT"),
        ("swedencentral", "FOUNDRY_SWEDENCENTRAL_PROJECT_ENDPOINT"),
    ):
        endpoint = os.environ.get(variable)
        if endpoint:
            configured.append(
                project_config(region, endpoint, subscription_id, resource_group)
            )
    return configured


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_schema(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {key: clean_schema(item) for key, item in value.items()}
        if cleaned.get("required") == []:
            cleaned.pop("required")
        return cleaned
    if isinstance(value, list):
        return [clean_schema(item) for item in value]
    return value


def openai_tools_to_openapi(tools: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {}
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported tool type: {tool.get('type')}")
        function = tool["function"]
        name = function["name"]
        paths[f"/{name}"] = {
            "post": {
                "operationId": name,
                "summary": function.get("description", "")[:120],
                "description": function.get("description", ""),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": clean_schema(
                                function.get("parameters")
                                or {"type": "object", "properties": {}}
                            )
                        }
                    },
                },
                "responses": {"200": {"description": "Success"}},
            }
        }
    return {
        "openapi": "3.0.3",
        "info": {"title": "Zava tool catalog", "version": "1.0.0"},
        "paths": paths,
    }


def wait_for_file(client: Any, file_id: str) -> Any:
    deadline = time.monotonic() + FILE_TIMEOUT_SECONDS
    while True:
        uploaded = client.files.retrieve(file_id)
        status = str(uploaded.status).lower()
        if status == "processed":
            return uploaded
        if status in {"error", "expired", "failed", "cancelled"}:
            raise RuntimeError(
                f"File {file_id} ended in {status}: "
                f"{getattr(uploaded, 'status_details', None) or getattr(uploaded, 'error', None)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"File {file_id} was not processed within the timeout.")
        time.sleep(POLL_SECONDS)


def upload_file(client: Any, path: Path, purpose: str) -> Any:
    with path.open("rb") as handle:
        uploaded = client.files.create(file=(path.name, handle), purpose=purpose)
    return wait_for_file(client, uploaded.id)


def deployment_for_model(
    deployments: list[Any], model_name: str, model_version: str
) -> Any:
    matches = [
        deployment
        for deployment in deployments
        if getattr(deployment, "model_name", None) == model_name
        and str(getattr(deployment, "model_version", "")) == model_version
    ]
    if not matches:
        raise RuntimeError(f"No deployment for {model_name}:{model_version}.")
    return matches[0]


def teacher_deployment(deployments: list[Any]) -> Any:
    preferences = [
        ("gpt-4.1", "2025-04-14"),
        ("gpt-5.4", "2026-03-05"),
        ("gpt-5.4-mini", "2026-03-17"),
    ]
    for name, version in preferences:
        matches = [
            deployment
            for deployment in deployments
            if getattr(deployment, "model_name", None) == name
            and str(getattr(deployment, "model_version", "")) == version
        ]
        if matches:
            return matches[0]
    raise RuntimeError("No supported teacher deployment was found.")


def submit_datagen(
    project: AIProjectClient,
    model_client: Any,
    spec_file_id: str,
    teacher: str,
    batch: int,
) -> tuple[dict[str, Any], Path]:
    output_name = f"synthetic-tool-use-{RUN_ID.lower()}-{batch}"
    options = ToolUseFineTuningDataGenerationJobOptions(
        max_samples=MAX_SAMPLES_PER_BATCH,
        model_options=DataGenerationModelOptions(model=teacher),
    )
    job = DataGenerationJob(
        inputs=DataGenerationJobInputs(
            name=output_name,
            scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
            sources=[FileDataGenerationJobSource(id=spec_file_id)],
            options=options,
            output_options=DataGenerationJobOutputOptions(name=output_name),
        )
    )
    poller = project.beta.datasets.begin_create_generation_job(job=job)
    job_id = poller.details.get("job_id")
    if not job_id:
        raise RuntimeError("Data-generation submission did not return a job ID.")
    print(f"data-generation batch {batch}: {job_id}", flush=True)
    result = poller.result(timeout=DATAGEN_TIMEOUT_SECONDS)
    output_path = RUN_DIR / f"generated-{batch}.jsonl"
    with output_path.open("wb") as destination:
        for output in result.outputs or []:
            destination.write(model_client.files.content(file_id=output.id).read())
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Data-generation job {job_id} produced no downloaded data.")
    return (
        {
            "id": job_id,
            "status": str(poller.status()),
            "generated_samples": result.generated_samples,
            "output_file_ids": [output.id for output in result.outputs or []],
            "download": str(output_path),
            "sha256": sha256(output_path),
        },
        output_path,
    )


def normalize_arguments(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value or "{}")
    return value or {}


def normalized_calls(tool_calls: list[Any]) -> list[tuple[str, Any]]:
    calls = []
    for call in tool_calls:
        if isinstance(call, dict):
            function = call.get("function") or {}
            calls.append(
                (function.get("name"), normalize_arguments(function.get("arguments")))
            )
        else:
            function = call.function
            calls.append(
                (function.name, normalize_arguments(function.arguments))
            )
    return calls


def evaluation_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns = []
    for row in rows:
        messages = row["messages"]
        for index, message in enumerate(messages):
            calls = message.get("tool_calls") if message.get("role") == "assistant" else None
            if calls:
                turns.append(
                    {
                        "messages": messages[:index],
                        "expected": normalized_calls(calls),
                    }
                )
    if not turns:
        raise RuntimeError("Held-out data contains no assistant tool-call turns.")
    return turns


def score_calls(
    expected: list[tuple[str, Any]], actual: list[tuple[str, Any]]
) -> dict[str, Any]:
    expected_names = [name for name, _ in expected]
    actual_names = [name for name, _ in actual]
    exact = expected == actual
    names_exact = expected_names == actual_names
    expected_set = set(expected_names)
    actual_set = set(actual_names)
    overlap = len(expected_set & actual_set)
    precision = overlap / len(actual_set) if actual_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    return {
        "exact": exact,
        "names_exact": names_exact,
        "tool_name_precision": precision,
        "tool_name_recall": recall,
        "score": 10 if exact else 8 if names_exact else 1 if not overlap else 2 + 6 * recall,
    }


def evaluate(
    model_client: Any,
    deployment: str,
    tools: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    results = []
    for index, turn in enumerate(turns, start=1):
        for attempt in range(1, 6):
            try:
                response = model_client.chat.completions.create(
                    model=deployment,
                    messages=turn["messages"],
                    tools=tools,
                    temperature=0,
                    max_completion_tokens=2048,
                )
                actual = normalized_calls(response.choices[0].message.tool_calls or [])
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
        scored = score_calls(turn["expected"], actual)
        results.append(
            {
                "turn": index,
                "expected": turn["expected"],
                "actual": actual,
                **scored,
            }
        )
    count = len(results)
    return {
        "deployment": deployment,
        "count": count,
        "exact_match_rate": sum(row["exact"] for row in results) / count,
        "tool_name_match_rate": sum(row["names_exact"] for row in results) / count,
        "tool_name_precision": sum(row["tool_name_precision"] for row in results)
        / count,
        "tool_name_recall": sum(row["tool_name_recall"] for row in results) / count,
        "average_score": sum(row["score"] for row in results) / count,
        "rows": results,
    }


def validate_generated_rows(
    rows: list[dict[str, Any]], operation_ids: set[str]
) -> list[dict[str, Any]]:
    accepted = []
    for row_index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Generated row {row_index} has no usable messages.")
        found_call = False
        for message in messages:
            for call in message.get("tool_calls") or []:
                found_call = True
                function = call.get("function") or {}
                if function.get("name") not in operation_ids:
                    raise ValueError(
                        f"Generated row {row_index} used unknown tool "
                        f"{function.get('name')}."
                    )
                normalize_arguments(function.get("arguments"))
        if not found_call:
            raise ValueError(f"Generated row {row_index} has no assistant tool call.")
        accepted.append(row)
    return accepted


def monitor_job(model_client: Any, job_id: str) -> Any:
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    last_status = None
    while True:
        job = model_client.fine_tuning.jobs.retrieve(job_id)
        status = str(job.status).lower()
        if status != last_status:
            print(f"fine-tuning {job_id}: {status}", flush=True)
            last_status = status
        if status == "succeeded":
            return job
        if status in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Fine-tuning job {job_id} ended in {status}: {job.error}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Fine-tuning job {job_id} exceeded the timeout.")
        time.sleep(60)


def run_az(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["az", *arguments, "--output", "json", "--only-show-errors"],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def deploy_fine_tuned_model(project_config: dict[str, str], model: str) -> str:
    deployment_name = f"synthetic-tools-{RUN_ID[4:13].lower()}"
    run_az(
        [
            "cognitiveservices",
            "account",
            "deployment",
            "create",
            "--subscription",
            project_config["subscription"],
            "--resource-group",
            project_config["resource_group"],
            "--name",
            project_config["resource"],
            "--deployment-name",
            deployment_name,
            "--model-name",
            model,
            "--model-version",
            "1",
            "--model-format",
            "OpenAI",
            "--sku-name",
            "GlobalStandard",
            "--sku-capacity",
            "1",
        ]
    )
    return deployment_name


def project_link(project_config: dict[str, str]) -> str:
    return (
        "https://ai.azure.com/foundryProject/overview"
        f"?wsid=/subscriptions/{project_config['subscription']}"
        f"/resourceGroups/{project_config['resource_group']}"
        f"/providers/Microsoft.CognitiveServices/accounts/{project_config['resource']}"
        f"/projects/{project_config['project']}"
    )


def run_project(
    project_config: dict[str, str],
    tools: list[dict[str, Any]],
    openapi_path: Path,
    operation_ids: set[str],
    credential: AzureCliCredential,
) -> dict[str, Any]:
    project = AIProjectClient(
        endpoint=project_config["endpoint"],
        credential=credential,
        allow_preview=True,
        polling_interval=POLL_SECONDS,
    )
    model_client = project.get_openai_client()
    deployments = list(project.deployments.list())
    teacher = teacher_deployment(deployments)
    student = deployment_for_model(deployments, "gpt-4.1-mini", "2025-04-14")
    summary: dict[str, Any] = {
        **project_config,
        "project_link": project_link(project_config),
        "started_at": utc_now(),
        "teacher_deployment": teacher.name,
        "teacher_model": f"{teacher.model_name}-{teacher.model_version}",
        "student_deployment": student.name,
        "student_model": STUDENT_MODEL,
        "available_deployments": [
            {
                "name": deployment.name,
                "model_name": getattr(deployment, "model_name", None),
                "model_version": getattr(deployment, "model_version", None),
            }
            for deployment in deployments
        ],
    }
    spec_file = upload_file(model_client, openapi_path, "user_data")
    summary["openapi_file"] = {
        "id": spec_file.id,
        "status": str(spec_file.status),
        "sha256": sha256(openapi_path),
    }

    generation_jobs = []
    generated_paths = []
    for batch in range(1, GENERATION_BATCHES + 1):
        generation_job, generated_path = submit_datagen(
            project, model_client, spec_file.id, teacher.name, batch
        )
        generation_jobs.append(generation_job)
        generated_paths.append(generated_path)
    summary["data_generation_jobs"] = generation_jobs

    unique_rows: dict[str, dict[str, Any]] = {}
    for generated_path in generated_paths:
        for row in read_jsonl(generated_path):
            unique_rows.setdefault(json.dumps(row, sort_keys=True), row)
    rows = validate_generated_rows(list(unique_rows.values()), operation_ids)
    if len(rows) < 20:
        raise RuntimeError(
            f"Fresh data generation yielded only {len(rows)} unique valid rows."
        )
    combined_path = RUN_DIR / "generated-exact-combined.jsonl"
    write_jsonl(combined_path, rows)

    random.Random(42).shuffle(rows)
    train_count = max(10, int(len(rows) * 0.8))
    validation_count = max(1, int(len(rows) * 0.1))
    if train_count + validation_count >= len(rows):
        validation_count = 1
        train_count = len(rows) - 2
    train_rows = rows[:train_count]
    validation_rows = rows[train_count : train_count + validation_count]
    test_rows = rows[train_count + validation_count :]
    train_path = RUN_DIR / "train.jsonl"
    validation_path = RUN_DIR / "validation.jsonl"
    test_path = RUN_DIR / "test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    write_jsonl(test_path, test_rows)
    summary["lineage"] = {
        "generated_combined": {
            "path": str(combined_path),
            "sha256": sha256(combined_path),
            "rows": len(rows),
        },
        "train": {
            "path": str(train_path),
            "sha256": sha256(train_path),
            "rows": len(train_rows),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
            "rows": len(validation_rows),
        },
        "test": {
            "path": str(test_path),
            "sha256": sha256(test_path),
            "rows": len(test_rows),
        },
        "fallback_used": False,
    }

    turns = evaluation_turns(test_rows)
    baseline = evaluate(model_client, student.name, tools, turns)
    write_json(RUN_DIR / "baseline-evaluation.json", baseline)
    summary["baseline_evaluation"] = {
        key: value for key, value in baseline.items() if key != "rows"
    }

    train_file = upload_file(model_client, train_path, "fine-tune")
    validation_file = upload_file(model_client, validation_path, "fine-tune")
    summary["training_file"] = {
        "id": train_file.id,
        "status": str(train_file.status),
        "sha256": sha256(train_path),
    }
    summary["validation_file"] = {
        "id": validation_file.id,
        "status": str(validation_file.status),
        "sha256": sha256(validation_path),
    }

    job = model_client.fine_tuning.jobs.create(
        model=STUDENT_MODEL,
        training_file=train_file.id,
        validation_file=validation_file.id,
        hyperparameters={"n_epochs": 3, "learning_rate_multiplier": 1.0},
        suffix="synthetic-tool-use",
        extra_body={"trainingType": "globalStandard"},
    )
    summary["fine_tuning_job"] = {
        "id": job.id,
        "status": str(job.status),
        "submitted_at": utc_now(),
    }
    write_json(RUN_DIR / "run-summary.json", summary)

    completed_job = monitor_job(model_client, job.id)
    summary["fine_tuning_job"].update(
        {
            "status": str(completed_job.status),
            "fine_tuned_model": completed_job.fine_tuned_model,
            "trained_tokens": getattr(completed_job, "trained_tokens", None),
            "finished_at": getattr(completed_job, "finished_at", None),
            "error": getattr(completed_job, "error", None),
        }
    )
    deployment_name = deploy_fine_tuned_model(
        project_config, completed_job.fine_tuned_model
    )
    deadline = time.monotonic() + 1800
    while True:
        try:
            resolved = project.deployments.get(deployment_name)
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(30)
    summary["fine_tuned_deployment"] = {
        "name": resolved.name,
        "model_name": getattr(resolved, "model_name", None),
        "model_version": getattr(resolved, "model_version", None),
    }

    fine_tuned = evaluate(model_client, deployment_name, tools, turns)
    write_json(RUN_DIR / "fine-tuned-evaluation.json", fine_tuned)
    summary["fine_tuned_evaluation"] = {
        key: value for key, value in fine_tuned.items() if key != "rows"
    }
    summary["outcome"] = {
        "average_score_lift": (
            fine_tuned["average_score"] - baseline["average_score"]
        ),
        "average_score_lift_percent": (
            (fine_tuned["average_score"] - baseline["average_score"])
            / baseline["average_score"]
            * 100
            if baseline["average_score"]
            else None
        ),
        "exact_match_rate_delta": (
            fine_tuned["exact_match_rate"] - baseline["exact_match_rate"]
        ),
        "tool_name_match_rate_delta": (
            fine_tuned["tool_name_match_rate"]
            - baseline["tool_name_match_rate"]
        ),
    }
    summary["completed_at"] = utc_now()
    return summary


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    tools = json.loads((DATA / "zava_tools_openai.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (DATA / "zava_tools_openapi.json").read_text(encoding="utf-8")
    )
    openapi = openai_tools_to_openapi(tools)
    operation_ids = {
        operation["post"]["operationId"] for operation in openapi["paths"].values()
    }
    reference_ids = {
        operation["post"]["operationId"]
        for operation in reference["paths"].values()
    }
    if operation_ids != reference_ids:
        raise RuntimeError("Fresh OpenAPI conversion differs from the preserved tool set.")
    openapi_path = RUN_DIR / "zava-tools-openapi.json"
    write_json(openapi_path, openapi)

    credential = AzureCliCredential(process_timeout=60)
    failures = []
    for project_config in configured_projects():
        print(
            f"attempting {project_config['region']}: {project_config['endpoint']}",
            flush=True,
        )
        try:
            summary = run_project(
                project_config, tools, openapi_path, operation_ids, credential
            )
            summary["regional_failures"] = failures
            write_json(RUN_DIR / "run-summary.json", summary)
            print(json.dumps(summary, indent=2, default=str), flush=True)
            return
        except Exception as error:
            failure = {
                "region": project_config["region"],
                "endpoint": project_config["endpoint"],
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            write_json(RUN_DIR / "regional-failures.json", failures)
            print(json.dumps(failure, indent=2), flush=True)
    raise RuntimeError("All requested project regions failed live validation.")


if __name__ == "__main__":
    main()
