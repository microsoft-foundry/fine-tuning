"""Run the Zava retail Qwen training-only experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "loom-model-runs"
SUMMARY = OUTPUT / "run-summary.json"
MODEL = "qwen3-32b"
MODEL_VERSION = "1"
PROJECT_ENDPOINT = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
PROJECT_REGION = os.environ.get("AZURE_AI_REGION", "customer-configured")
SUFFIX = os.environ.get(
    "AZURE_FINE_TUNING_SUFFIX",
    "zava-retail-qwen3-32b-assistant-defaults",
)
TERMINAL = {"succeeded", "failed", "cancelled", "canceled"}

DATASETS = {
    "sft_training": (DATA / "sft_train.jsonl", 386),
    "sft_validation": (DATA / "sft_test.jsonl", 103),
    "rft_training": (DATA / "rft_train_fixed.jsonl", 10),
    "rft_validation": (DATA / "rft_test_fixed.jsonl", 10),
    "held_out": (DATA / "eval.jsonl", 280),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path, expected_rows: int) -> dict[str, Any]:
    content = path.read_bytes()
    rows = len(content.splitlines())
    if rows != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} rows, found {rows}")
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": rows,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def write_summary(summary: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def make_qwen_compatible(source: Path, destination: Path) -> dict[str, Any]:
    content_defaults = 0
    tool_call_defaults = 0
    rows = 0
    with source.open(encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line in reader:
            row = json.loads(line)
            rows += 1
            for message in row["messages"]:
                if message.get("role") != "assistant":
                    continue
                if message.get("content") is None:
                    message["content"] = ""
                    content_defaults += 1
                if message.get("tool_calls") is None:
                    message["tool_calls"] = []
                    tool_call_defaults += 1
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "source": str(source.relative_to(ROOT)),
        "path": str(destination.relative_to(ROOT)),
        "rows": rows,
        "content_defaults_added": content_defaults,
        "tool_call_defaults_added": tool_call_defaults,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def find_existing_job(client: Any) -> Any | None:
    matches = [
        job
        for job in client.fine_tuning.jobs.list(limit=100)
        if job.model == MODEL and getattr(job, "suffix", None) == SUFFIX
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {MODEL} jobs use suffix {SUFFIX}: "
            + ", ".join(job.id for job in matches)
        )
    return matches[0] if matches else None


def download_metrics(client: Any, job: Any) -> list[dict[str, Any]]:
    metric_files = []
    for file_id in getattr(job, "result_files", []) or []:
        metadata = client.files.retrieve(file_id)
        filename = getattr(metadata, "filename", None) or f"{file_id}.csv"
        destination = OUTPUT / f"qwen3-32b-{filename}"
        content = client.files.content(file_id)
        if hasattr(content, "write_to_file"):
            content.write_to_file(destination)
        else:
            payload = getattr(content, "content", content)
            destination.write_bytes(payload)
        metric_files.append(
            {
                "file_id": file_id,
                "path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    return metric_files


def summarize_metrics(metric_files: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for record in metric_files:
        path = ROOT / record["path"]
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows.extend(csv.DictReader(handle))
    if not rows:
        return {"rows": 0, "available_metrics": [], "final": {}}

    def number(value: str | None) -> float | None:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None

    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"step", "timestamp"} and number(value) is not None
        }
    )
    final = {
        key: number(rows[-1].get(key))
        for key in metric_names
        if number(rows[-1].get(key)) is not None
    }
    return {
        "rows": len(rows),
        "available_metrics": metric_names,
        "final": final,
    }


def main() -> None:
    if not PROJECT_ENDPOINT:
        raise ValueError(
            "Set AZURE_AI_PROJECT_ENDPOINT to your Microsoft Foundry project endpoint"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "started_at": utc_now(),
        "project_endpoint": PROJECT_ENDPOINT,
        "region": PROJECT_REGION,
        "training_type": "GlobalStandard",
        "resolved_model": MODEL,
        "model_version": MODEL_VERSION,
        "model_format": "Alibaba",
        "selection_reason": (
            "qwen3-32b is used for the training-only SFT path. Confirm model and "
            "GlobalStandard availability in the selected project before running."
        ),
        "training_only": True,
        "deployments_created": 0,
        "inference_requests": 0,
        "datasets": {
            name: file_record(path, rows)
            for name, (path, rows) in DATASETS.items()
        },
        "method_support": {
            "supervised": {
                "supported": True,
                "evidence": (
                    "The selected project must expose qwen3-32b version 1 with "
                    "GlobalStandard fine-tuning support."
                ),
            },
            "reinforcement": {
                "supported": False,
                "evidence": (
                    "Microsoft Foundry RFT documentation dated 2026-05-14 lists "
                    "only o4-mini 2025-04-16 and gated gpt-5 2025-08-07."
                ),
                "submission": "skipped",
            },
        },
        "source_file_policy": {
            "original_files_modified": False,
            "upload_transform": (
                "Ignored upload copies add empty content and tool_calls defaults "
                "only to assistant messages where fields are omitted; "
                "conversation/tool-call values are unchanged."
            ),
            "renderer_expectation": (
                "Qwen accepts standard tool-call dictionaries but its current Loom "
                "renderer requires content and tool_calls keys on every message."
            ),
        },
        "held_out_usage": (
            "Preserved and verified only; inference/evaluation was intentionally skipped."
        ),
    }
    qwen_train_path = OUTPUT / "sft_train_qwen3_32b.jsonl"
    qwen_validation_path = OUTPUT / "sft_test_qwen3_32b.jsonl"
    summary["model_compatibility_transforms"] = {
        "original_files_modified": False,
        "semantic_values_modified": False,
        "training": make_qwen_compatible(
            DATASETS["sft_training"][0], qwen_train_path
        ),
        "validation": make_qwen_compatible(
            DATASETS["sft_validation"][0], qwen_validation_path
        ),
    }
    write_summary(summary)

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    with AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
        allow_preview=True,
    ) as project_client:
        with project_client.get_openai_client() as client:
            job = find_existing_job(client)
            if job is None:
                train_file = client.files.create(
                    file=qwen_train_path, purpose="fine-tune"
                )
                validation_file = client.files.create(
                    file=qwen_validation_path, purpose="fine-tune"
                )
                client.files.wait_for_processing(train_file.id)
                client.files.wait_for_processing(validation_file.id)
                summary["uploads"] = {
                    "sft_training_file_id": train_file.id,
                    "sft_validation_file_id": validation_file.id,
                    "byte_identical_to_source": False,
                    "source_files_remain_byte_identical": True,
                }
                write_summary(summary)

                job = client.fine_tuning.jobs.create(
                    model=MODEL,
                    training_file=train_file.id,
                    validation_file=validation_file.id,
                    method={
                        "type": "supervised",
                        "supervised": {
                            "hyperparameters": {
                                "n_epochs": 1,
                                "batch_size": 1,
                                "learning_rate_multiplier": 1,
                            }
                        },
                    },
                    suffix=SUFFIX,
                    extra_body={"trainingType": "GlobalStandard"},
                )
                reused = False
            else:
                reused = True

            summary["sft_job"] = {
                "id": job.id,
                "status": job.status,
                "reused": reused,
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "method": "supervised",
                "training_type": "GlobalStandard",
                "training_file_id": job.training_file,
                "validation_file_id": job.validation_file,
                "hyperparameters": {
                    "n_epochs": 1,
                    "batch_size": 1,
                    "learning_rate_multiplier": 1,
                },
                "submitted_at": utc_now(),
            }
            write_summary(summary)

            last_status = None
            while job.status not in TERMINAL:
                if job.status != last_status:
                    print(f"{utc_now()} {job.id}: {job.status}", flush=True)
                    last_status = job.status
                time.sleep(60)
                job = client.fine_tuning.jobs.retrieve(job.id)

            summary["sft_job"].update(
                {
                    "status": job.status,
                    "finished_at": utc_now(),
                    "fine_tuned_model": getattr(job, "fine_tuned_model", None),
                    "trained_tokens": getattr(job, "trained_tokens", None),
                    "error": (
                        job.error.to_dict()
                        if getattr(job, "error", None)
                        and hasattr(job.error, "to_dict")
                        else getattr(job, "error", None)
                    ),
                    "result_files": list(getattr(job, "result_files", []) or []),
                }
            )
            summary["metrics_files"] = download_metrics(client, job)
            summary["metrics"] = summarize_metrics(summary["metrics_files"])
            summary["completed_at"] = utc_now()
            summary["renderer_behavior"] = (
                "Original OpenAI tool-call messages passed preprocessing and reached "
                "training."
                if summary["metrics"]["rows"]
                else (
                    "Assistant-scoped defaults passed file preprocessing, but Loom "
                    "normalized empty tool_calls arrays back to null. The qwen3 "
                    "renderer then raised TypeError while iterating "
                    "message['tool_calls']; no metric row was emitted."
                )
            )
            summary["conclusion"] = {
                "status": (
                    "completed" if job.status == "succeeded" else "blocked"
                ),
                "sft": (
                    "Use the captured validation loss/token-accuracy metrics."
                    if summary["metrics"]["rows"]
                    else (
                        "No validation loss or token-accuracy conclusion is "
                        "available because the renderer failed before step 1."
                    )
                ),
                "rft": (
                    "Unsupported for qwen3-32b; no reward conclusion or RFT "
                    "submission was produced."
                ),
                "blocker": (
                    "The service converts valid empty assistant tool_calls arrays "
                    "to null, while loom_cookbook/renderers/qwen3.py assumes an "
                    "iterable. Supplying a non-empty replacement would invent tool "
                    "calls and alter the preserved dataset semantics."
                ),
            }
            write_summary(summary)
            print(json.dumps(summary["sft_job"], indent=2, default=str), flush=True)
            print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
