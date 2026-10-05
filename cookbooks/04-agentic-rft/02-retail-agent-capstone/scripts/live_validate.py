from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential

ROOT = Path(__file__).resolve().parents[1]
COOKBOOKS_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(COOKBOOKS_ROOT))

from cookbook_utils import (  # noqa: E402
    expected_tool_call,
    load_json,
    load_jsonl,
    normalize_tool_call,
    resolve_tool_endpoints,
    sha256_file,
    validate_conversation_rows,
)
from shared.deployment import invoke_chat  # noqa: E402
from shared.foundry_operations import (  # noqa: E402
    create_or_reuse_fine_tuning_job,
    upload_or_reuse_file,
)

SFT_BASE_MODEL = "gpt-4.1-mini-2025-04-14"
RFT_BASE_MODEL = "o4-mini-2025-04-16"
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def epoch_iso(value: Any) -> str | None:
    return (
        datetime.fromtimestamp(value, UTC).isoformat()
        if isinstance(value, (int, float))
        else None
    )


def azure_cli() -> str:
    executable = shutil.which("az.cmd") or shutil.which("az")
    if not executable:
        raise RuntimeError("Azure CLI was not found on PATH")
    return executable


def region_config(name: str, slug: str, project_endpoint: str) -> dict[str, str]:
    parsed = urlparse(project_endpoint.rstrip("/"))
    if parsed.scheme != "https" or "/api/projects/" not in parsed.path:
        raise ValueError(f"Invalid Foundry project endpoint: {project_endpoint}")
    account = parsed.netloc.removesuffix(".services.ai.azure.com")
    project = parsed.path.rsplit("/", 1)[-1]
    return {
        "name": name,
        "slug": slug,
        "account": account,
        "project": project,
        "endpoint": project_endpoint.rstrip("/"),
    }


def endpoint(region: dict[str, str]) -> str:
    return region["endpoint"]


def resolve_arm_account_id(account: str) -> str:
    query = (
        "Resources "
        "| where type =~ 'microsoft.cognitiveservices/accounts' "
        f"| where name == '{account}' "
        "| project id"
    )
    completed = subprocess.run(
        [
            azure_cli(),
            "graph",
            "query",
            "-q",
            query,
            "--first",
            "2",
            "--output",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = json.loads(completed.stdout).get("data", [])
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one ARM account named {account!r}, found {len(matches)}"
        )
    return matches[0]["id"]


def json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return value
    return str(value)


def write_report(report: dict[str, Any]) -> None:
    output = ROOT / "outputs" / "live-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def existing_evaluations() -> dict[str, Any]:
    output = ROOT / "outputs" / "live-validation.json"
    if not output.is_file():
        return {}
    loaded = json.loads(output.read_text(encoding="utf-8"))
    evaluations = loaded.get("evaluations", {})
    return evaluations if isinstance(evaluations, dict) else {}


def validate_lineage() -> dict[str, Any]:
    manifest = load_json(ROOT / "data" / "manifest.json")
    entries = {}
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual["bytes"] != entry["bytes"] or actual["sha256"] != entry["sha256"]:
            raise RuntimeError(f"Lineage validation failed for {entry['path']}")
        entries[entry["path"]] = {**actual, "records": entry.get("records")}

    sft_train = load_jsonl(ROOT / "data" / "sft-train.jsonl")
    sft_validation = load_jsonl(ROOT / "data" / "sft-validation.jsonl")
    rft_train = load_jsonl(ROOT / "data" / "rft-train.jsonl")
    rft_validation = load_jsonl(ROOT / "data" / "rft-validation.jsonl")
    validate_conversation_rows(sft_train, label="SFT train")
    validate_conversation_rows(sft_validation, label="SFT validation")
    required = (
        "messages",
        "tools",
        "reference_tool_calls",
        "reference_policy_args",
        "reference_policy_outcome",
    )
    validate_conversation_rows(rft_train, label="RFT train", required_fields=required)
    validate_conversation_rows(
        rft_validation, label="RFT validation", required_fields=required
    )
    return {
        "selection": "preserved-curated-splits",
        "validated_at": utc_now(),
        "files": entries,
        "row_counts": {
            "sft_train": len(sft_train),
            "sft_validation": len(sft_validation),
            "rft_train": len(rft_train),
            "rft_validation": len(rft_validation),
            "held_out": len(load_jsonl(ROOT / "data" / "next-tool-evaluation.jsonl")),
        },
    }


def discover_region(
    region: dict[str, str], credential: AzureCliCredential
) -> dict[str, Any]:
    with AIProjectClient(
        endpoint=endpoint(region),
        credential=credential,
        allow_preview=True,
    ) as project_client:
        deployments = [
            {
                "name": item.name,
                "model_name": getattr(item, "model_name", None),
                "model_version": getattr(item, "model_version", None),
                "sku": json_value(getattr(item, "sku", None)),
            }
            for item in project_client.deployments.list()
        ]
        openai_client = project_client.get_openai_client()
        try:
            jobs = list(openai_client.fine_tuning.jobs.list(limit=200))
        finally:
            openai_client.close()
    successful_models = sorted(
        {job.model for job in jobs if str(job.status).casefold() == "succeeded"}
    )
    return {
        "region": region["name"],
        "endpoint": endpoint(region),
        "deployments": deployments,
        "successful_fine_tuning_models": successful_models,
        "supports_selected_sft": SFT_BASE_MODEL in successful_models,
        "supports_selected_rft": RFT_BASE_MODEL in successful_models,
    }


def submit_path(
    *,
    label: str,
    model: str,
    train_path: Path,
    validation_path: Path,
    method: dict[str, Any],
    discoveries: dict[str, dict[str, Any]],
    credential: AzureCliCredential,
    regions: list[dict[str, str]],
) -> dict[str, Any]:
    failures = []
    for region in regions:
        region_key = region["slug"]
        discovery = discoveries[region_key]
        capability_key = (
            "supports_selected_sft" if label == "sft" else "supports_selected_rft"
        )
        if not discovery[capability_key]:
            failures.append(
                {
                    "region": region["name"],
                    "stage": "capability",
                    "error": f"{model} has no successful runtime history in this project",
                }
            )
            continue
        try:
            with AIProjectClient(
                endpoint=endpoint(region),
                credential=credential,
                allow_preview=True,
            ) as project_client:
                client = project_client.get_openai_client()
                try:
                    train_upload = upload_or_reuse_file(
                        client,
                        demo_slug="retail-agent-capstone",
                        purpose_name=f"{label}-training",
                        path=train_path,
                    )
                    validation_upload = upload_or_reuse_file(
                        client,
                        demo_slug="retail-agent-capstone",
                        purpose_name=f"{label}-validation",
                        path=validation_path,
                    )
                    suffix_hash = hashlib.sha256(
                        (
                            f"{model}:{train_upload.details['sha256']}:"
                            f"{validation_upload.details['sha256']}:{label}"
                        ).encode()
                    ).hexdigest()[:8]
                    submission = create_or_reuse_fine_tuning_job(
                        client,
                        model=model,
                        training_file_id=train_upload.resource_id,
                        validation_file_id=validation_upload.resource_id,
                        suffix=f"retail-capstone-{label}-{suffix_hash}",
                        method=method,
                    )
                finally:
                    client.close()
            return {
                "label": label,
                "region": region,
                "endpoint": endpoint(region),
                "model": model,
                "training_file": json_value(train_upload),
                "validation_file": json_value(validation_upload),
                "job_id": submission.resource_id,
                "submission_status": submission.status,
                "reused": submission.reused,
                "failover_failures": failures,
            }
        except Exception as error:
            failures.append(
                {
                    "region": region["name"],
                    "stage": "submission",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    raise RuntimeError(f"{label.upper()} submission failed in every region: {failures}")


def retrieve_job(
    record: dict[str, Any], credential: AzureCliCredential
) -> Any:
    with AIProjectClient(
        endpoint=record["endpoint"],
        credential=credential,
        allow_preview=True,
    ) as project_client:
        client = project_client.get_openai_client()
        try:
            return client.fine_tuning.jobs.retrieve(record["job_id"])
        finally:
            client.close()


def monitor_jobs(
    report: dict[str, Any],
    credential: AzureCliCredential,
    poll_seconds: int,
) -> None:
    pending = set(report["jobs"])
    while pending:
        for label in list(pending):
            record = report["jobs"][label]
            job = retrieve_job(record, credential)
            status = str(job.status).casefold()
            if record.get("status") != status:
                print(f"[{utc_now()}] {label} {record['job_id']}: {status}", flush=True)
            record["status"] = status
            record["last_observed_at"] = utc_now()
            record["fine_tuned_model"] = getattr(job, "fine_tuned_model", None)
            record["error"] = json_value(getattr(job, "error", None))
            record["result_files"] = getattr(job, "result_files", None)
            record["trained_tokens"] = getattr(job, "trained_tokens", None)
            record["created_at"] = epoch_iso(getattr(job, "created_at", None))
            record["finished_at"] = epoch_iso(getattr(job, "finished_at", None))
            record["api_url"] = (
                f"{record['endpoint']}/openai/v1/fine_tuning/jobs/{record['job_id']}"
            )
            if status in TERMINAL_STATUSES:
                pending.remove(label)
        write_report(report)
        if pending:
            time.sleep(poll_seconds)


def _numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key)]


def collect_training_metrics(
    report: dict[str, Any], credential: AzureCliCredential
) -> dict[str, Any]:
    metrics = {}
    for label, record in report["jobs"].items():
        result_files = record.get("result_files") or []
        if not result_files:
            continue
        with AIProjectClient(
            endpoint=record["endpoint"],
            credential=credential,
            allow_preview=True,
        ) as project_client:
            client = project_client.get_openai_client()
            try:
                content = client.files.content(result_files[0]).read()
            finally:
                client.close()
        output_path = ROOT / "outputs" / f"{label}-results.csv"
        output_path.write_bytes(content)
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
        if label == "sft":
            metrics[label] = {
                "steps": len(rows),
                "train_loss_initial": float(rows[0]["train_loss"]),
                "train_loss_final": float(rows[-1]["train_loss"]),
                "train_loss_min": min(_numeric_values(rows, "train_loss")),
                "train_token_accuracy_initial": float(
                    rows[0]["train_mean_token_accuracy"]
                ),
                "train_token_accuracy_final": float(
                    rows[-1]["train_mean_token_accuracy"]
                ),
                "validation_loss_final": _numeric_values(rows, "valid_loss")[-1],
                "validation_token_accuracy_final": _numeric_values(
                    rows, "valid_mean_token_accuracy"
                )[-1],
                "full_validation_loss_final": _numeric_values(
                    rows, "full_valid_loss"
                )[-1],
                "full_validation_token_accuracy_final": _numeric_values(
                    rows, "full_valid_mean_token_accuracy"
                )[-1],
            }
        else:
            row = rows[-1]
            selected = (
                "step",
                "train_mean_reward",
                "full_valid_mean_reward",
                "completion_tokens_mean",
                "mean_unresponsive_rewards",
                "reasoning_tokens_mean",
                "sampling_duration",
                "training_duration",
                "eval_duration",
                "total_duration",
                "scores/graders/score_model/Policy Outcome Consistency Grader/train_reward_mean",
                "scores/graders/score_model/Policy Outcome Consistency Grader/valid_reward_mean",
                "errors/graders/score_model/Policy Outcome Consistency Grader/errors/train_sample_parse_error_mean",
                "errors/graders/score_model/Policy Outcome Consistency Grader/errors/valid_sample_parse_error_mean",
            )
            metrics[label] = {
                key: (float(row[key]) if row.get(key) else None) for key in selected
            }
    return metrics


def create_deployment(record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] != "succeeded" or not record["fine_tuned_model"]:
        raise RuntimeError(f"{record['label']} did not produce a deployable model")
    region = record["region"]
    deployment_name = (
        f"retail-{record['label']}-{record['job_id'].removeprefix('ftjob-')[-8:]}"
    )
    url = (
        "https://management.azure.com"
        f"{resolve_arm_account_id(region['account'])}/deployments/"
        f"{deployment_name}?api-version=2024-10-01"
    )
    body = {
        "sku": {"name": "GlobalStandard", "capacity": 10},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": record["fine_tuned_model"],
                "version": "1",
            },
            "versionUpgradeOption": "NoAutoUpgrade",
        },
    }
    completed = subprocess.run(
        [
            azure_cli(),
            "rest",
            "--method",
            "put",
            "--url",
            url,
            "--body",
            json.dumps(body),
            "--output",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)
    return {
        "name": deployment_name,
        "model": record["fine_tuned_model"],
        "provisioning_state": response.get("properties", {}).get("provisioningState"),
        "resource_id": response.get("id"),
    }


def wait_for_deployment(
    record: dict[str, Any],
    deployment: dict[str, Any],
    credential: AzureCliCredential,
) -> None:
    for _ in range(40):
        with AIProjectClient(
            endpoint=record["endpoint"],
            credential=credential,
            allow_preview=True,
        ) as project_client:
            try:
                found = project_client.deployments.get(deployment["name"])
                deployment["resolved"] = json_value(found)
                return
            except Exception:
                time.sleep(15)
    raise TimeoutError(f"Deployment {deployment['name']} did not resolve")


def predict(client: Any, deployment_name: str, row: dict[str, Any]) -> dict[str, Any]:
    item = row["item"]
    for attempt in range(1, 6):
        try:
            try:
                response = invoke_chat(
                    client,
                    deployment_name=deployment_name,
                    messages=item["messages"],
                    tools=item["tools"],
                    tool_choice="required",
                    max_tokens=512,
                )
            except Exception as parameter_error:
                if "max_completion_tokens" not in str(parameter_error):
                    raise
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=item["messages"],
                    tools=item["tools"],
                    tool_choice="required",
                    max_completion_tokens=4096,
                    reasoning_effort="low",
                )
            return normalize_tool_call(response.choices[0].message)
        except Exception as error:
            chain = []
            current: BaseException | None = error
            while current is not None:
                chain.append(f"{type(current).__name__}: {current}".casefold())
                current = current.__cause__
            text = " ".join(chain)
            transient = any(
                marker in text
                for marker in (
                    "timeout",
                    "timed out",
                    "connection",
                    "rate limit",
                    "ratelimit",
                    "retryerror",
                    "too_many_requests",
                    "did not contain a tool call",
                )
            )
            if not transient or attempt == 5:
                raise
            time.sleep(min(15 * attempt, 60))
    raise AssertionError("unreachable")


def evaluate_deployment(
    record: dict[str, Any],
    deployment_name: str,
    credential: AzureCliCredential,
    limit: int | None,
    workers: int,
) -> dict[str, Any]:
    rows = load_jsonl(ROOT / "data" / "next-tool-evaluation.jsonl")
    selected = rows if limit is None else rows[:limit]
    with AIProjectClient(
        endpoint=record["endpoint"],
        credential=credential,
        allow_preview=True,
    ) as project_client:
        client = project_client.get_openai_client()
        try:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(predict, client, deployment_name, row): (index, row)
                    for index, row in enumerate(selected)
                }
                for completed, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    index, row = futures[future]
                    actual = future.result()
                    expected = expected_tool_call(row)
                    results.append(
                        {
                            "index": index,
                            "exact": actual == expected,
                            "tool_name_match": actual.get("name") == expected["name"],
                        }
                    )
                    if completed % 10 == 0 or completed == len(selected):
                        print(
                            f"{deployment_name}: evaluated {completed}/{len(selected)}",
                            flush=True,
                        )
        finally:
            client.close()
    exact = sum(result["exact"] for result in results)
    tool_name_match = sum(result["tool_name_match"] for result in results)
    return {
        "exact": exact,
        "tool_name_match": tool_name_match,
        "total": len(results),
        "exact_rate": exact / len(results),
        "tool_name_match_rate": tool_name_match / len(results),
        "non_mutating": True,
        "contract": "Model predicts one next tool call; no returned tool is executed.",
    }


def base_deployment(discovery: dict[str, Any]) -> str:
    candidates = [
        item["name"]
        for item in discovery["deployments"]
        if item["model_name"] == "gpt-4.1-mini"
        and item["model_version"] == "2025-04-14"
    ]
    if not candidates:
        raise RuntimeError("No gpt-4.1-mini base deployment was discovered")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-endpoint", required=True)
    parser.add_argument(
        "--failover-endpoint",
        action="append",
        default=[],
        help="Repeat in desired failover order.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--evaluation-workers", type=int, default=4)
    args = parser.parse_args()

    tools_server_url = os.environ.get("FOUNDRY_TOOLS_SERVER_URL", "").strip()
    if not tools_server_url:
        raise RuntimeError("FOUNDRY_TOOLS_SERVER_URL is required for guarded RFT")

    credential = AzureCliCredential(process_timeout=60)
    configured_endpoints = [args.primary_endpoint, *args.failover_endpoint]
    labels = ["Primary", "Failover 1", "Failover 2"]
    regions = [
        region_config(labels[index], f"region-{index + 1}", configured)
        for index, configured in enumerate(configured_endpoints)
    ]
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "lineage": validate_lineage(),
        "regions": {},
        "jobs": {},
        "deployments": {},
        "evaluations": existing_evaluations(),
    }
    for region in regions:
        print(f"Discovering {region['name']}", flush=True)
        report["regions"][region["slug"]] = discover_region(region, credential)
        write_report(report)

    grader = load_json(ROOT / "configs" / "rft-policy-grader.json")
    rft_method = {
        "type": "reinforcement",
        "reinforcement": {
            "grader": grader,
            "tools": resolve_tool_endpoints(
                ROOT / "configs" / "rft-tools.json", tools_server_url
            ),
            "max_episode_steps": 10,
            "hyperparameters": {
                "eval_interval": 1,
                "eval_samples": 2,
                "compute_multiplier": 1,
                "reasoning_effort": "medium",
                "n_epochs": 1,
                "batch_size": 10,
                "learning_rate_multiplier": 1,
            },
        },
    }
    methods = {
        "sft": {"type": "supervised"},
        "rft": rft_method,
    }
    paths = {
        "sft": (
            SFT_BASE_MODEL,
            ROOT / "data" / "sft-train.jsonl",
            ROOT / "data" / "sft-validation.jsonl",
        ),
        "rft": (
            RFT_BASE_MODEL,
            ROOT / "data" / "rft-train.jsonl",
            ROOT / "data" / "rft-validation.jsonl",
        ),
    }
    for label in ("sft", "rft"):
        model, train_path, validation_path = paths[label]
        print(f"Submitting or reusing {label.upper()}", flush=True)
        report["jobs"][label] = submit_path(
            label=label,
            model=model,
            train_path=train_path,
            validation_path=validation_path,
            method=methods[label],
            discoveries=report["regions"],
            credential=credential,
            regions=regions,
        )
        write_report(report)

    monitor_jobs(report, credential, args.poll_seconds)
    failures = [
        label for label, record in report["jobs"].items() if record["status"] != "succeeded"
    ]
    if failures:
        raise RuntimeError(f"Training did not succeed: {failures}")
    report["training_metrics"] = collect_training_metrics(report, credential)
    write_report(report)

    for label, record in report["jobs"].items():
        deployment = create_deployment(record)
        wait_for_deployment(record, deployment, credential)
        report["deployments"][label] = deployment
        write_report(report)

    primary = report["regions"]["region-1"]
    if "base" not in report["evaluations"]:
        report["evaluations"]["base"] = evaluate_deployment(
            report["jobs"]["sft"],
            base_deployment(primary),
            credential,
            args.evaluation_limit,
            args.evaluation_workers,
        )
        write_report(report)
    for label in ("sft", "rft"):
        if label not in report["evaluations"]:
            report["evaluations"][label] = evaluate_deployment(
                report["jobs"][label],
                report["deployments"][label]["name"],
                credential,
                args.evaluation_limit,
                args.evaluation_workers,
            )
            write_report(report)
    report["completed_at"] = utc_now()
    report["outcome"] = "succeeded"
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
