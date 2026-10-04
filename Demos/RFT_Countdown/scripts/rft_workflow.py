import ast
import asyncio
import json
import operator
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from azure.identity import AzureCliCredential
from openai import APIConnectionError, APIStatusError, RateLimitError

from scripts.client_utils import PROJECT_ENDPOINT, get_openai_client


RUN_RECORDS_PATH = Path("run_records.jsonl")
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
TERMINAL_EVAL_STATUSES = {"completed", "failed", "cancelled"}
SUBSCRIPTION_ID = "ba7979f7-d040-49c9-af1a-7414402bf622"
RESOURCE_GROUP = "prakharg-demo-2026"
ACCOUNT_NAME = "eastus2-prakharg-demo-2026"


def record_event(notebook: str, event: str, **details) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notebook": notebook,
        "event": event,
        **details,
    }
    with RUN_RECORDS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def fine_tune_job_url(job_id: str) -> str:
    return f"{PROJECT_ENDPOINT}/openai/v1/fine_tuning/jobs/{job_id}"


def ensure_fine_tuned_deployment(
    notebook: str,
    model_name: str,
    deployment_name: str,
    poll_seconds: int = 15,
) -> str:
    credential = AzureCliCredential(process_timeout=120)
    url = (
        f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
        f"/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.CognitiveServices/accounts/{ACCOUNT_NAME}"
        f"/deployments/{deployment_name}"
    )
    api_version = "2024-10-01"

    def request(method: str, body: dict | None = None):
        token = credential.get_token("https://management.azure.com/.default").token
        return requests.request(
            method,
            url,
            params={"api-version": api_version},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )

    response = request("GET")
    if response.status_code == 404:
        payload = {
            "sku": {"name": "GlobalStandard", "capacity": 10},
            "properties": {
                "model": {
                    "format": "OpenAI",
                    "name": model_name,
                    "version": "1",
                }
            },
        }
        response = request("PUT", payload)
        if response.status_code not in {200, 201, 202}:
            raise RuntimeError(
                f"Deployment creation failed: {response.status_code} {response.text}"
            )
        record_event(
            notebook,
            "deployment_created",
            deployment_name=deployment_name,
            model=model_name,
            resource_id=response.json().get("id"),
            sku="GlobalStandard",
            capacity=10,
        )
        print(f"Deployment submitted: {deployment_name}")
    elif not response.ok:
        raise RuntimeError(
            f"Deployment lookup failed: {response.status_code} {response.text}"
        )

    while True:
        response = request("GET")
        if not response.ok:
            raise RuntimeError(
                f"Deployment polling failed: {response.status_code} {response.text}"
            )
        deployment = response.json()
        state = deployment["properties"]["provisioningState"]
        print(f"Deployment {deployment_name}: {state}")
        if state == "Succeeded":
            record_event(
                notebook,
                "deployment_status",
                deployment_name=deployment_name,
                status=state,
                resource_id=deployment.get("id"),
            )
            return deployment_name
        if state in {"Failed", "Canceled"}:
            raise RuntimeError(
                f"Deployment {deployment_name} ended with {state}: {deployment}"
            )
        time.sleep(poll_seconds)


def create_rft_job(
    notebook: str,
    model: str,
    training_file: str,
    validation_file: str,
    method: dict,
    suffix: str,
    max_attempts: int = 3,
):
    client = get_openai_client()
    if RUN_RECORDS_PATH.exists():
        records = [
            json.loads(line)
            for line in RUN_RECORDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prior_jobs = [
            record
            for record in records
            if record.get("notebook") == notebook
            and record.get("event") == "fine_tune_created"
            and record.get("grader_type") == method["reinforcement"]["grader"]["type"]
        ]
        if prior_jobs:
            prior_job_id = prior_jobs[-1]["job_id"]
            try:
                prior_job = client.fine_tuning.jobs.retrieve(prior_job_id)
                if prior_job.status not in {"failed", "cancelled"}:
                    print(
                        f"Resuming existing fine-tuning job: {prior_job_id} "
                        f"({prior_job.status})"
                    )
                    record_event(
                        notebook,
                        "fine_tune_resumed",
                        job_id=prior_job_id,
                        status=prior_job.status,
                    )
                    return prior_job
            except (RateLimitError, APIConnectionError, APIStatusError) as exc:
                record_event(
                    notebook,
                    "fine_tune_resume_lookup_failed",
                    job_id=prior_job_id,
                    error=str(exc),
                )
                raise

    for attempt in range(1, max_attempts + 1):
        try:
            job = client.fine_tuning.jobs.create(
                model=model,
                training_file=training_file,
                validation_file=validation_file,
                method=method,
                suffix=suffix,
            )
            record_event(
                notebook,
                "fine_tune_created",
                attempt=attempt,
                job_id=job.id,
                model=model,
                training_file=training_file,
                validation_file=validation_file,
                grader_type=method["reinforcement"]["grader"]["type"],
                job_url=fine_tune_job_url(job.id),
            )
            print(f"Fine-tuning job submitted: {job.id}")
            print(f"Job API link: {fine_tune_job_url(job.id)}")
            return job
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = isinstance(exc, (RateLimitError, APIConnectionError)) or (
                status_code is not None and status_code >= 500
            )
            record_event(
                notebook,
                "fine_tune_create_failed",
                attempt=attempt,
                retryable=retryable,
                status_code=status_code,
                error=str(exc),
            )
            if not retryable or attempt == max_attempts:
                raise
            delay = 30 * attempt
            print(f"Transient create failure; retrying in {delay}s: {exc}")
            time.sleep(delay)


def wait_for_fine_tune(notebook: str, job_id: str, poll_seconds: int = 60):
    client = get_openai_client()
    previous_status = None
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = isinstance(exc, (RateLimitError, APIConnectionError)) or (
                status_code is not None and status_code >= 500
            )
            record_event(
                notebook,
                "fine_tune_poll_failed",
                job_id=job_id,
                retryable=retryable,
                status_code=status_code,
                error=str(exc),
            )
            if not retryable:
                raise
            print(f"Transient polling failure; retrying in {poll_seconds}s: {exc}")
            time.sleep(poll_seconds)
            continue
        if job.status != previous_status:
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"{job_id}: {job.status}"
            )
            record_event(
                notebook,
                "fine_tune_status",
                job_id=job_id,
                status=job.status,
                fine_tuned_model=job.fine_tuned_model,
                error=job.error.model_dump() if job.error else None,
                trained_tokens=job.trained_tokens,
            )
            previous_status = job.status
        if job.status in TERMINAL_JOB_STATUSES:
            if job.status != "succeeded":
                raise RuntimeError(
                    f"Fine-tuning job {job_id} ended with {job.status}: {job.error}"
                )
            return job
        time.sleep(poll_seconds)


async def wait_for_eval_run(
    notebook: str,
    eval_client,
    eval_id: str,
    run_id: str,
    poll_seconds: int = 30,
):
    previous_status = None
    while True:
        run = await eval_client.get_eval_run_sdk(eval_id, run_id)
        status = run.get("status")
        if status != previous_status:
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"{run_id}: {status}"
            )
            record_event(
                notebook,
                "eval_run_status",
                eval_id=eval_id,
                run_id=run_id,
                status=status,
                result_counts=run.get("result_counts"),
                error=run.get("error"),
            )
            previous_status = status
        if status in TERMINAL_EVAL_STATUSES:
            if status != "completed":
                raise RuntimeError(
                    f"Evaluation run {run_id} ended with {status}: {run.get('error')}"
                )
            return run
        await asyncio.sleep(poll_seconds)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _evaluate_expression(node: ast.AST, numbers: list[float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, numbers)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        numbers.append(float(node.value))
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression(node.left, numbers)
        right = _evaluate_expression(node.right, numbers)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError("Only numeric literals and +, -, *, / operations are allowed")


def validate_countdown_response(response: str | dict, target: int, nums: list[int]) -> dict:
    payload = json.loads(response) if isinstance(response, str) else response
    expression = payload["expression"]
    reported_result = float(payload["result"])
    used_numbers: list[float] = []
    calculated_result = _evaluate_expression(
        ast.parse(expression, mode="eval"), used_numbers
    )
    expected_numbers = Counter(float(number) for number in nums)
    actual_numbers = Counter(used_numbers)
    valid = (
        actual_numbers == expected_numbers
        and abs(calculated_result - reported_result) < 1e-9
        and abs(calculated_result - target) < 1e-9
    )
    return {
        "valid": valid,
        "target": target,
        "numbers": nums,
        "expression": expression,
        "reported_result": reported_result,
        "calculated_result": calculated_result,
        "used_all_numbers_once": actual_numbers == expected_numbers,
        "exact_target": abs(calculated_result - target) < 1e-9,
    }


def solve_and_validate(
    notebook: str,
    model: str,
    instruction: str,
    response_schema: dict,
    cases: list[dict],
) -> list[dict]:
    client = get_openai_client()
    results = []
    for case in cases:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": instruction},
                {
                    "role": "user",
                    "content": f"Target: {case['target']}\nNumbers: {case['nums']}",
                },
            ],
            response_format=response_schema,
            max_completion_tokens=20000,
        )
        output = response.choices[0].message.content
        validation = validate_countdown_response(
            output, case["target"], case["nums"]
        )
        results.append(validation)
        print(json.dumps(validation, indent=2))
    passed = sum(result["valid"] for result in results)
    record_event(
        notebook,
        "countdown_validation",
        model=model,
        passed=passed,
        total=len(results),
        results=results,
    )
    if passed == 0:
        raise RuntimeError(f"Model {model} did not solve any validation cases exactly")
    return results
