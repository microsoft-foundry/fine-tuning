"""Run the teacher-student demo end to end with fresh, traceable data."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import (
    Deployment,
    DeploymentModel,
    DeploymentProperties,
    Sku,
)
from openai import APIConnectionError, APIStatusError, RateLimitError


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SYSTEM_PROMPT = "Clippy is a factual chatbot that is also sarcastic."
GRADER_PROMPT = """You are an independent evaluator. Score the candidate from 1 to 10.
Factual correctness is mandatory: a materially wrong answer must receive overall_score <= 2.
Sarcasm should be clear but not abusive. Return only JSON with numeric factual_score,
sarcasm_score, overall_score and a short reason.

Question: {question}
Reference fact: {reference}
Candidate: {candidate}
"""
GRADER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "teacher_student_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "factual_score": {"type": "number", "minimum": 1, "maximum": 10},
            "sarcasm_score": {"type": "number", "minimum": 1, "maximum": 10},
            "overall_score": {"type": "number", "minimum": 1, "maximum": 10},
            "reason": {"type": "string"},
        },
        "required": [
            "factual_score",
            "sarcasm_score",
            "overall_score",
            "reason",
        ],
        "additionalProperties": False,
    },
}
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}
TERMINAL_FILE_STATES = {"processed", "error", "expired", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before running live validation.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Evaluator did not return JSON: {text[:200]}")
    value = json.loads(text[start : end + 1])
    required_fields = {"factual_score", "sarcasm_score", "overall_score", "reason"}
    if not required_fields <= value.keys():
        raise ValueError(
            f"Evaluator response is missing fields: {required_fields - value.keys()}"
        )
    for name in ("factual_score", "sarcasm_score", "overall_score"):
        score = float(value[name])
        if not 1 <= score <= 10:
            raise ValueError(f"{name} must be between 1 and 10.")
    return value


def retry(operation: Callable[[], Any], label: str, attempts: int = 12) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            if (
                attempt == attempts
                or isinstance(exc, APIStatusError)
                and status not in {408, 409, 429, 500, 502, 503, 504}
            ):
                raise
            headers = getattr(exc, "response", None)
            retry_after = None
            if headers is not None:
                retry_after = headers.headers.get("retry-after")
            delay = float(retry_after) if retry_after else min(60, 2**attempt)
            print(f"{label} attempt {attempt} failed ({status}); retrying in {delay:g}s")
            time.sleep(delay)
    raise AssertionError("retry loop exhausted")


def select_project(
    endpoints: list[str],
    credential: AzureCliCredential,
    required_deployments: set[str],
) -> tuple[str, AIProjectClient, list[dict[str, Any]], list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    for endpoint in endpoints:
        try:
            project = AIProjectClient(endpoint=endpoint, credential=credential)
            deployments = [deployment.as_dict() for deployment in project.deployments.list()]
            names = {deployment["name"] for deployment in deployments}
            missing = sorted(required_deployments - names)
            if missing:
                raise RuntimeError(f"missing deployments: {missing}")
            return endpoint, project, deployments, failures
        except Exception as exc:
            failures.append(
                {"endpoint": endpoint, "error": f"{type(exc).__name__}: {exc}"}
            )
    raise RuntimeError(f"No project satisfied the deployment contract: {failures}")


def wait_for_file(client: Any, file_id: str, timeout_seconds: float) -> Any:
    started = time.monotonic()
    while True:
        uploaded = client.files.retrieve(file_id)
        status = str(uploaded.status).casefold()
        print(f"file {file_id}: {status}")
        if status in TERMINAL_FILE_STATES:
            if status != "processed":
                raise RuntimeError(
                    f"File {file_id} ended as {status}: "
                    f"{getattr(uploaded, 'error', None)}"
                )
            return uploaded
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"File {file_id} did not process before timeout.")
        time.sleep(5)


def upload_file(client: Any, path: Path, timeout_seconds: float) -> Any:
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="fine-tune")
    return wait_for_file(client, uploaded.id, timeout_seconds)


def wait_for_job(client: Any, job_id: str, timeout_seconds: float) -> Any:
    started = time.monotonic()
    seen_events: set[str] = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        events = client.fine_tuning.jobs.list_events(job_id, limit=100)
        for event in reversed(list(getattr(events, "data", events))):
            event_id = str(getattr(event, "id", ""))
            if event_id and event_id not in seen_events:
                seen_events.add(event_id)
                print(f"{getattr(event, 'created_at', '')}: {event.message}")
        status = str(job.status).casefold()
        print(f"job {job_id}: {status}")
        if status in TERMINAL_JOB_STATES:
            if status != "succeeded":
                raise RuntimeError(
                    f"Fine-tuning job ended as {status}: {getattr(job, 'error', None)}"
                )
            return job
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Job {job_id} did not reach terminal state before timeout.")
        time.sleep(30)


def create_fine_tuned_deployment(
    credential: AzureCliCredential,
    endpoint: str,
    fine_tuned_model: str,
    deployment_name: str,
) -> None:
    subscription_id = required("FOUNDRY_SUBSCRIPTION_ID")
    resource_group = required("FOUNDRY_RESOURCE_GROUP")
    account_name = urlparse(endpoint).hostname.removesuffix(".services.ai.azure.com")
    management = CognitiveServicesManagementClient(credential, subscription_id)
    deployment = Deployment(
        properties=DeploymentProperties(
            model=DeploymentModel(
                format="OpenAI",
                name=fine_tuned_model,
                version="1",
            )
        ),
        sku=Sku(
            name=os.environ.get("FOUNDRY_DEPLOYMENT_SKU", "GlobalStandard"),
            capacity=int(os.environ.get("FOUNDRY_DEPLOYMENT_CAPACITY", "10")),
        ),
    )
    management.deployments.begin_create_or_update(
        resource_group_name=resource_group,
        account_name=account_name,
        deployment_name=deployment_name,
        deployment=deployment,
    ).result()


def main() -> None:
    teacher_deployment = required("FOUNDRY_TEACHER_DEPLOYMENT")
    student_deployment = required("FOUNDRY_STUDENT_DEPLOYMENT")
    evaluator_deployment = required("FOUNDRY_EVALUATOR_DEPLOYMENT")
    student_model = os.environ.get(
        "FOUNDRY_STUDENT_MODEL", "gpt-4.1-nano-2025-04-14"
    )
    endpoint_candidates = [
        value.strip()
        for value in required("FOUNDRY_PROJECT_ENDPOINTS").split(",")
        if value.strip()
    ]
    source_count = int(os.environ.get("FOUNDRY_SOURCE_COUNT", "60"))
    cutoff = float(os.environ.get("TEACHER_SCORE_CUTOFF", "6"))
    file_timeout = float(os.environ.get("FOUNDRY_FILE_TIMEOUT_SECONDS", "600"))
    job_timeout = float(os.environ.get("FOUNDRY_JOB_TIMEOUT_SECONDS", "10800"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUTS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    credential = AzureCliCredential()
    endpoint, project, deployments, endpoint_failures = select_project(
        endpoint_candidates,
        credential,
        {teacher_deployment, student_deployment, evaluator_deployment},
    )
    model_client = project.get_openai_client()
    deployment_map = {deployment["name"]: deployment for deployment in deployments}
    print(f"Selected project: {endpoint}")
    print(
        json.dumps(
            {
                name: deployment_map[name]
                for name in (
                    teacher_deployment,
                    student_deployment,
                    evaluator_deployment,
                )
            },
            indent=2,
        )
    )

    questions = read_jsonl(DATA / "qa.jsonl")
    if source_count < 30 or source_count > len(questions):
        raise ValueError(f"FOUNDRY_SOURCE_COUNT must be between 30 and {len(questions)}.")
    selected = list(questions)
    random.Random(42).shuffle(selected)
    selected = selected[:source_count]
    holdout_count = max(10, source_count // 5)
    generation_questions = selected[:-holdout_count]
    held_out = selected[-holdout_count:]
    write_jsonl(run_dir / "selected-source.jsonl", selected)
    write_jsonl(run_dir / "held-out.jsonl", held_out)

    def judge(question: str, reference: str, candidate: str) -> dict[str, Any]:
        response = retry(
            lambda: model_client.responses.create(
                model=evaluator_deployment,
                input=GRADER_PROMPT.format(
                    question=question,
                    reference=reference,
                    candidate=candidate,
                ),
                text={"format": GRADER_RESPONSE_FORMAT},
            ),
            "evaluator",
        )
        return parse_json_object(response.output_text)

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(generation_questions, start=1):
        response = retry(
            lambda row=row: model_client.responses.create(
                model=teacher_deployment,
                input=f"{SYSTEM_PROMPT}\n\nUser question: {row['question']}",
            ),
            "teacher",
        )
        teacher_answer = response.output_text.strip()
        candidates.append(
            {
                "question": row["question"],
                "reference_answer": row["answer"],
                "teacher_answer": teacher_answer,
                "evaluation": judge(row["question"], row["answer"], teacher_answer),
                "teacher_deployment": teacher_deployment,
                "evaluator_deployment": evaluator_deployment,
                "prompt_version": "sarcasm-v1",
                "generated_at": utc_now(),
            }
        )
        write_jsonl(run_dir / "teacher-candidates.jsonl", candidates)
        print(f"generated and judged {index}/{len(generation_questions)}")

    accepted = [
        row
        for row in candidates
        if float(row["evaluation"]["overall_score"]) >= cutoff
    ]
    if len(accepted) < 20:
        raise RuntimeError(
            f"Only {len(accepted)} fresh teacher rows passed cutoff {cutoff:g}."
        )
    random.Random(42).shuffle(accepted)
    validation_count = max(10, round(0.2 * len(accepted)))
    if len(accepted) - validation_count < 10:
        raise RuntimeError("Fresh accepted rows cannot form valid train/validation splits.")
    validation_source = accepted[:validation_count]
    training_source = accepted[validation_count:]

    def training_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["teacher_answer"]},
            ]
        }

    train_path = write_jsonl(
        run_dir / "train.jsonl", [training_row(row) for row in training_source]
    )
    validation_path = write_jsonl(
        run_dir / "validation.jsonl",
        [training_row(row) for row in validation_source],
    )
    lineage = {
        "selected_source_sha256": sha256(run_dir / "selected-source.jsonl"),
        "teacher_candidates_sha256": sha256(run_dir / "teacher-candidates.jsonl"),
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "generated_count": len(candidates),
        "accepted_count": len(accepted),
        "training_count": len(training_source),
        "validation_count": len(validation_source),
        "held_out_count": len(held_out),
        "score_cutoff": cutoff,
    }
    (run_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )

    train_file = upload_file(model_client, train_path, file_timeout)
    validation_file = upload_file(model_client, validation_path, file_timeout)
    job = model_client.fine_tuning.jobs.create(
        model=student_model,
        training_file=train_file.id,
        validation_file=validation_file.id,
        method={
            "type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": 3}},
        },
        suffix="teacher-student-basics",
        extra_body={"trainingType": "globalStandard"},
    )
    print(f"submitted fine-tuning job {job.id}")
    completed_job = wait_for_job(model_client, job.id, job_timeout)
    fine_tuned_model = completed_job.fine_tuned_model
    if not fine_tuned_model:
        raise RuntimeError("Succeeded job did not expose a fine-tuned model ID.")

    fine_tuned_deployment = f"teacher-student-ft-{job.id[-8:]}"
    create_fine_tuned_deployment(
        credential,
        endpoint,
        fine_tuned_model,
        fine_tuned_deployment,
    )
    resolved = project.deployments.get(fine_tuned_deployment)
    if resolved.model_name != fine_tuned_model:
        raise RuntimeError(
            f"Deployment model mismatch: {resolved.model_name} != {fine_tuned_model}"
        )

    def evaluate(deployment: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(held_out, start=1):
            response = retry(
                lambda row=row: model_client.responses.create(
                    model=deployment,
                    input=f"{SYSTEM_PROMPT}\n\nUser question: {row['question']}",
                ),
                f"{deployment} inference",
            )
            answer = response.output_text.strip()
            score = judge(row["question"], row["answer"], answer)
            rows.append(
                {
                    "question": row["question"],
                    "reference_answer": row["answer"],
                    "answer": answer,
                    "evaluation": score,
                }
            )
            print(f"evaluated {deployment} {index}/{len(held_out)}")
        return {
            "deployment": deployment,
            "count": len(rows),
            "average": sum(
                float(row["evaluation"]["overall_score"]) for row in rows
            )
            / len(rows),
            "rows": rows,
        }

    evaluations = {
        "teacher": evaluate(teacher_deployment),
        "student_base": evaluate(student_deployment),
        "student_fine_tuned": evaluate(fine_tuned_deployment),
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluations, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    project_name = endpoint.rstrip("/").split("/")[-1]
    portal_resource_id = os.environ.get("FOUNDRY_PORTAL_RESOURCE_ID", "").strip()
    portal_account_name = os.environ.get(
        "FOUNDRY_PORTAL_ACCOUNT_NAME",
        urlparse(endpoint).hostname.removesuffix(".services.ai.azure.com"),
    ).strip()
    job_link = (
        f"https://ai.azure.com/nextgen/r/{portal_resource_id},"
        f"{portal_account_name},,"
        f"{project_name},{project_name}/build/fine-tune/{job.id}/logs"
        if portal_resource_id
        else None
    )
    summary = {
        "run_id": run_id,
        "started_from_fresh_data": True,
        "completed_at": utc_now(),
        "project_endpoint": endpoint,
        "endpoint_failures_before_selection": endpoint_failures,
        "resource": {
            "subscription_id": required("FOUNDRY_SUBSCRIPTION_ID"),
            "resource_group": required("FOUNDRY_RESOURCE_GROUP"),
            "account": urlparse(endpoint).hostname.removesuffix(
                ".services.ai.azure.com"
            ),
            "project": project_name,
        },
        "models": {
            "teacher": deployment_map[teacher_deployment],
            "student_base": deployment_map[student_deployment],
            "evaluator": deployment_map[evaluator_deployment],
            "fine_tuned_model": fine_tuned_model,
        },
        "deployments": {
            "teacher": teacher_deployment,
            "student_base": student_deployment,
            "evaluator": evaluator_deployment,
            "student_fine_tuned": fine_tuned_deployment,
        },
        "files": {
            "training": {"id": train_file.id, "status": train_file.status},
            "validation": {
                "id": validation_file.id,
                "status": validation_file.status,
            },
        },
        "job": {
            "id": completed_job.id,
            "status": completed_job.status,
            "fine_tuned_model": fine_tuned_model,
            "link": job_link,
        },
        "lineage": lineage,
        "evaluation": {
            name: {"count": value["count"], "average": value["average"]}
            for name, value in evaluations.items()
        },
        "outcome": {
            "fine_tuned_minus_base": (
                evaluations["student_fine_tuned"]["average"]
                - evaluations["student_base"]["average"]
            ),
            "fine_tuned_minus_teacher": (
                evaluations["student_fine_tuned"]["average"]
                - evaluations["teacher"]["average"]
            ),
        },
    }
    summary_path = run_dir / "run-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Live validation summary: {summary_path}")


if __name__ == "__main__":
    main()
