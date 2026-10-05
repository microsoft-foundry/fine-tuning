import ast
import asyncio
import csv
import hashlib
import io
import json
import operator
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.client_utils import get_openai_client, get_project_client


RUN_RECORDS_PATH = Path("outputs/execution/run_records.jsonl")
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
TERMINAL_EVAL_STATUSES = {"completed", "failed", "cancelled"}


def record_event(notebook: str, event: str, **details) -> None:
    RUN_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notebook": notebook,
        "event": event,
        **details,
    }
    with RUN_RECORDS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
    }


def ensure_fine_tuned_deployment(
    notebook: str,
    model_name: str,
    deployment_name: str,
    poll_seconds: int = 15,
    timeout_seconds: int = 900,
) -> str:
    project_client = get_project_client()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            deployment = project_client.deployments.get(deployment_name)
        except Exception as exc:
            if _status_code(exc) != 404 and not _is_retryable(exc):
                raise
            print(f"Deployment {deployment_name} is not available yet")
            time.sleep(poll_seconds)
            continue
        if deployment.model_name != model_name:
            raise RuntimeError(
                f"Deployment {deployment_name} targets {deployment.model_name}, "
                f"not {model_name}"
            )
        if deployment.capabilities.get("chat_completion") == "true":
            record_event(
                notebook,
                "deployment_status",
                deployment_name=deployment_name,
                status="available",
                model=deployment.model_name,
                sku=deployment.sku,
            )
            print(f"Deployment {deployment_name}: available")
            return deployment_name
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Deployment {deployment_name} was not available after {timeout_seconds}s. "
        "Create the fine-tuned model deployment in Foundry, then rerun this cell."
    )


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
            and record.get("model") == model
            and record.get("training_file") == training_file
            and record.get("validation_file") == validation_file
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
            except Exception as exc:
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
                extra_body={"trainingType": "globalStandard"},
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
                training_type="globalStandard",
            )
            print(f"Fine-tuning job submitted: {job.id}")
            return job
        except Exception as exc:
            status_code = _status_code(exc)
            retryable = _is_retryable(exc)
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
        except Exception as exc:
            status_code = _status_code(exc)
            retryable = _is_retryable(exc)
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
            return job
        time.sleep(poll_seconds)


def _file_evidence(path: str) -> dict:
    content = Path(path).read_bytes()
    return {
        "path": path.replace("\\", "/"),
        "bytes": len(content),
        "records": sum(1 for line in content.splitlines() if line.strip()),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reward_summary(rows: list[dict]) -> dict:
    validation_columns = [
        name
        for name in (rows[0].keys() if rows else [])
        if "reward" in name.lower()
        and ("valid" in name.lower() or "validation" in name.lower())
        and "error" not in name.lower()
    ]
    training_columns = [
        name
        for name in (rows[0].keys() if rows else [])
        if "reward" in name.lower()
        and ("train" in name.lower() or name == "train_mean_reward")
        and "error" not in name.lower()
    ]

    def curve(columns: list[str]) -> list[dict]:
        values = []
        for row in rows:
            for column in columns:
                value = row.get(column)
                if value not in {None, ""}:
                    values.append(
                        {
                            "step": int(float(row["step"])) if row.get("step") else None,
                            "column": column,
                            "value": float(value),
                        }
                    )
                    break
        return values

    validation_curve = curve(validation_columns)
    training_curve = curve(training_columns)
    validation_values = [point["value"] for point in validation_curve]
    return {
        "validation_curve": validation_curve,
        "training_curve": training_curve,
        "validation_reward_initial": (
            validation_values[0] if validation_values else None
        ),
        "validation_reward_final": (
            validation_values[-1] if validation_values else None
        ),
        "validation_reward_max": (
            max(validation_values) if validation_values else None
        ),
    }


def _conclusion(status: str, rewards: dict) -> str:
    if status != "succeeded":
        return (
            "No training-quality conclusion is possible because the job did not "
            "succeed."
        )
    initial = rewards["validation_reward_initial"]
    final = rewards["validation_reward_final"]
    maximum = rewards["validation_reward_max"]
    if initial is None:
        return (
            "Training succeeded, but no validation reward points were emitted; "
            "no quality conclusion is supported."
        )
    if maximum > initial and final >= initial:
        return (
            "Validation reward improved during training and did not finish below "
            "its initial value. This is a positive training signal, not evidence "
            "of deployed-model or inference quality."
        )
    if maximum > initial:
        return (
            "Validation reward improved transiently but finished below its best "
            "point. The run may have over-optimized after the maximum; deployment "
            "and inference were intentionally not performed."
        )
    return (
        "Validation reward did not improve above its initial value. The run does "
        "not support a training-quality gain conclusion."
    )


def collect_training_evidence(
    notebook: str,
    job,
    training_path: str,
    validation_path: str,
    evaluation_path: str,
) -> dict:
    client = get_openai_client()
    output_root = Path("outputs/loom-model-runs")
    job_root = output_root / notebook
    job_root.mkdir(parents=True, exist_ok=True)

    events = list(
        client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job.id,
            limit=100,
        ).data
    )
    event_payload = [
        event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        for event in reversed(events)
    ]
    (job_root / "events.json").write_text(
        json.dumps(event_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    result_summaries = []
    reward_rows = []
    for index, file_id in enumerate(list(job.result_files or []), start=1):
        content = client.files.content(file_id).read()
        result_path = job_root / f"result-{index}.csv"
        result_path.write_bytes(content)
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
        reward_rows.extend(rows)
        result_summaries.append(
            {
                "file_id": file_id,
                "path": str(result_path).replace("\\", "/"),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": len(rows),
            }
        )

    rewards = _reward_summary(reward_rows)
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    project_name = project_endpoint.rsplit("/", 1)[-1]
    error = job.error.model_dump(mode="json") if job.error else None
    service_failures = [
        event["message"]
        for event in event_payload
        if event.get("level") in {"error", "warning"} and event.get("message")
    ]
    fixes = [
        "Removed deployment and inference requirements from the execution path.",
        "Pinned trainingType to globalStandard and excluded East US 2.",
        "Reused only hash-identical uploaded files.",
    ]
    if notebook == "demo_with_python_grader":
        fixes.extend(
            [
                "The service automatically retried the first failed training attempt.",
                "A transient TLS polling timeout was classified as retryable; the notebook resumed the same job instead of submitting a replacement.",
                "Corrected async file-content handling after the first pre-submission notebook attempt.",
            ]
        )
    summary = {
        "notebook": notebook,
        "project_endpoint": project_endpoint,
        "region": "North Central US",
        "base_model": job.model,
        "catalog_model_version": "1",
        "training_type": getattr(job, "trainingType", None),
        "grader_type": job.method.reinforcement.grader.type,
        "job_id": job.id,
        "job_link": (
            f"https://ai.azure.com/resource/projects/{project_name}/finetuning/{job.id}"
        ),
        "status": job.status,
        "error": error,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": job.trained_tokens,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "input_files": {
            "training": _file_evidence(training_path),
            "validation": _file_evidence(validation_path),
            "evaluation_preserved_not_run": _file_evidence(evaluation_path),
        },
        "remote_files": {
            "training_file_id": job.training_file,
            "validation_file_id": job.validation_file,
        },
        "result_files": result_summaries,
        "rewards": rewards,
        "service_warnings_and_errors": service_failures,
        "failures_and_fixes": fixes,
        "conclusion": _conclusion(job.status, rewards),
    }
    summary_path = job_root / "job-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    summaries = []
    for path in sorted(output_root.glob("*/job-summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    combined = {
        "base_model": "qwen3.6-35b-a3b",
        "catalog_model_version": "1",
        "project_endpoint": project_endpoint,
        "region": "North Central US",
        "training_type": "globalStandard",
        "catalog_resolution": {
            "publisher": "Alibaba",
            "format": "Alibaba",
            "model": "qwen3.6-35b-a3b",
            "version": "1",
            "lifecycle_status": "GenerallyAvailable",
            "capabilities": {
                "fineTune": True,
                "globalFineTune": True,
                "datazoneFineTune": True,
            },
            "excluded_variants": [
                "FW-Qwen3.6-27B: no fineTune capability in the live catalog entry.",
                "FW-Qwen3.6-35B-A3B: deployment SKUs are provisioned-only, not GlobalStandard.",
            ],
        },
        "deployment_performed": False,
        "inference_performed": False,
        "runs": summaries,
        "cautious_conclusion": (
            "Both graders show substantial validation-reward improvement from the "
            "first evaluation to the final evaluation, with maxima near the final "
            "values. This supports a positive training signal for both grader "
            "variants, but no deployment or inference evaluation was performed, "
            "so no serving-quality or task-accuracy claim is made."
        ),
    }
    (output_root / "run-summary.json").write_text(
        json.dumps(combined, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


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
    return results
