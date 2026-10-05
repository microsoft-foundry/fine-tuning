from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    DataGenerationJob,
    DataGenerationJobInputs,
    DataGenerationJobOutputOptions,
    DataGenerationJobScenario,
    TracesDataGenerationJobOptions,
    TracesDataGenerationJobSource,
)
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import (
    Deployment,
    DeploymentModel,
    DeploymentProperties,
    Sku,
)
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SYSTEM_PROMPT = (DATA / "zava_system_prompt.md").read_text(encoding="utf-8")
TOOLS = json.loads((DATA / "zava_tools.json").read_text(encoding="utf-8"))
RESPONSE_TOOLS = [{"type": "function", **tool["function"]} for tool in TOOLS]
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)bearer\s+[a-z0-9._-]+"),
    re.compile(r"(?i)(api[_ -]?key|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
]
PROMPT_TEMPLATES = [
    "Order {order_id} arrived damaged. The headphones have a cracked case. Please refund them.",
    "The shoes in order {order_id} are the wrong size. Exchange them for size medium.",
    "Order {order_id} says delivered, but the Bluetooth speaker is missing. I need a replacement.",
    "Please cancel order {order_id}; it is still processing.",
    "The smartwatch in order {order_id} stopped working after one week. Please replace it.",
    "Order {order_id} was delivered five days late. Return the lamp and include any shipping credit.",
    "I want to return only the tablet stand from order {order_id} and keep the other items.",
    "Tracking for order {order_id} has not moved in a week. Find my hiking backpack.",
    "Can I exchange the office chair in order {order_id} for the next model and pay the difference?",
    "Am I eligible for a refund on the espresso machine from order {order_id}? I am a Gold customer.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the ignored .env file.")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def status_value(resource: Any) -> str:
    return str(getattr(resource, "status", "")).split(".")[-1].casefold()


def tool_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    order_id = arguments.get("order_id", "ZA-0000")
    if name == "get_order_details":
        return {
            "order_id": order_id,
            "customer": {"loyalty_tier": "Gold"},
            "items": [{
                "item_id": "ITEM-1",
                "sku": "SKU-DEMO-1",
                "name": "demo item",
                "category": "electronics",
                "price": 129.0,
                "final_sale": False,
            }],
            "payment_method": "card",
        }
    if name == "get_fulfillment_status":
        return {
            "order_id": order_id,
            "status": "delivered",
            "late_delivery": True,
            "days_late": 5,
            "days_since_delivery": 7,
        }
    if name == "check_resolution_policy":
        return {
            "order_id": order_id,
            "item_id": arguments.get("item_id", "ITEM-1"),
            "eligible": True,
            "allowed_actions": ["refund", "exchange", "replacement"],
            "restocking_fee_percent": 0,
        }
    if name == "check_inventory":
        return {
            "sku": arguments.get("sku", "SKU-DEMO-1"),
            "available": True,
            "quantity": 25,
        }
    if name == "calculate_resolution":
        return {
            "order_id": order_id,
            "approved": True,
            "refund_amount": 129.0,
            "shipping_credit": 10.0,
            "restocking_fee": 0.0,
        }
    if name == "submit_resolution":
        return {
            "order_id": order_id,
            "status": "submitted",
            "resolution_id": f"RES-{order_id}",
        }
    return {"ok": True}


def build_prompts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    templates = list(PROMPT_TEMPLATES)
    prompts = []
    for index in range(count):
        template = templates[index % len(templates)]
        order_id = f"PARITY-{seed}-{index:04d}-{rng.randint(1000, 9999)}"
        prompts.append(template.format(order_id=order_id))
    if len(set(prompts)) != count:
        raise RuntimeError("Fresh prompt construction did not produce unique conversations.")
    return prompts


def run_teacher_conversation(
    responses: Any,
    model: str,
    prompt: str,
    max_rounds: int,
) -> dict[str, Any]:
    response = responses.create(model=model, input=prompt, store=True)
    response_ids = [response.id]
    calls = []
    for _ in range(max_rounds):
        function_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]
        if not function_calls:
            return {
                "prompt": prompt,
                "response_ids": response_ids,
                "tool_calls": calls,
                "status": response.status,
            }
        outputs = []
        for call in function_calls:
            arguments = json.loads(call.arguments or "{}")
            calls.append({"name": call.name, "arguments": arguments})
            outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(tool_result(call.name, arguments)),
            })
        response = responses.create(
            model=model,
            previous_response_id=response.id,
            input=outputs,
            store=True,
        )
        response_ids.append(response.id)
    raise RuntimeError(f"Conversation exceeded {max_rounds} tool rounds.")


def generate_fresh_conversations(
    project: AIProjectClient,
    agent_name: str,
    teacher_deployment: str,
    prompts: list[str],
    concurrency: int,
    max_rounds: int,
    output_path: Path,
) -> dict[str, Any]:
    responses = project.get_openai_client(agent_name=agent_name).responses
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                run_teacher_conversation,
                responses,
                teacher_deployment,
                prompt,
                max_rounds,
            ): prompt
            for prompt in prompts
        }
        for completed, future in enumerate(as_completed(futures), 1):
            prompt = futures[future]
            try:
                successes.append(future.result())
            except Exception as exc:
                failures.append({
                    "prompt": prompt,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if completed % 25 == 0 or completed == len(prompts):
                print(
                    f"conversation generation {completed}/{len(prompts)} "
                    f"ok={len(successes)} failed={len(failures)}"
                )
    report = {
        "started_at": utc_now(),
        "requested_conversations": len(prompts),
        "successful_conversations": len(successes),
        "failed_conversations": len(failures),
        "duration_seconds": round(time.monotonic() - started, 2),
        "successes": sorted(successes, key=lambda item: item["prompt"]),
        "failures": failures,
    }
    write_json(output_path, report)
    if failures:
        raise RuntimeError(
            f"Fresh trace generation failed for {len(failures)} conversations; "
            "refusing a partial compatibility run."
        )
    return report


def acquire_traces(
    project: AIProjectClient,
    model_client: Any,
    agent_name: str,
    agent_version: str,
    start: datetime,
    end: datetime,
    max_samples: int,
    raw_path: Path,
) -> dict[str, Any]:
    job = DataGenerationJob(inputs=DataGenerationJobInputs(
        name=f"agent-traces-parity-{end:%Y%m%d-%H%M%S}",
        scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
        sources=[TracesDataGenerationJobSource(
            agent_name=agent_name,
            agent_version=agent_version,
            start_time=start,
            end_time=end,
            description=f"Fresh parity traces for {agent_name}:{agent_version}",
        )],
        options=TracesDataGenerationJobOptions(
            max_samples=max_samples,
            train_split=0.8,
        ),
        output_options=DataGenerationJobOutputOptions(
            name=f"agent-traces-parity-{end:%Y%m%d-%H%M%S}"
        ),
    ))
    poller = project.beta.datasets.begin_create_generation_job(job=job)
    job_id = poller.details.get("job_id")
    if not job_id:
        raise RuntimeError("Trace acquisition did not return a job ID.")
    print("trace acquisition job:", job_id)
    result = poller.result()
    current = project.beta.datasets.get_generation_job(job_id)
    if status_value(current) != "succeeded":
        raise RuntimeError(
            f"Trace acquisition ended as {status_value(current)}: "
            f"{getattr(current, 'error', None)}"
        )
    outputs = list(result.outputs or [])
    if not outputs:
        raise RuntimeError("Trace acquisition succeeded without output files.")
    with raw_path.open("wb") as destination:
        for output in outputs:
            blob = model_client.files.content(file_id=output.id).read()
            if destination.tell() and blob and not blob.startswith(b"\n"):
                destination.write(b"\n")
            destination.write(blob)
            if blob and not blob.endswith(b"\n"):
                destination.write(b"\n")
    return {
        "job_id": job_id,
        "status": status_value(current),
        "output_file_ids": [output.id for output in outputs],
        "generated_samples": result.generated_samples,
        "lookback_start": start.isoformat(),
        "lookback_end": end.isoformat(),
        "max_samples": max_samples,
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
    }


def message_key(message: dict[str, Any]) -> tuple[Any, ...]:
    calls = tuple(
        (
            call.get("id"),
            (call.get("function") or {}).get("name"),
            (call.get("function") or {}).get("arguments"),
        )
        for call in message.get("tool_calls", [])
    )
    return (
        message.get("role"),
        message.get("content"),
        message.get("tool_call_id"),
        calls,
    )


def transform_trace_row(row: dict[str, Any]) -> dict[str, Any] | None:
    row_digest = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()
    seen = set()
    messages = []
    for source in row.get("messages", []):
        message = dict(source)
        key = message_key(message)
        if key not in seen:
            seen.add(key)
            messages.append(message)
    merged = []
    for message in messages:
        if (
            merged
            and message.get("role") == "assistant"
            and message.get("tool_calls")
            and merged[-1].get("role") == "assistant"
            and merged[-1].get("tool_calls")
        ):
            merged[-1]["tool_calls"].extend(message["tool_calls"])
        else:
            merged.append(message)
    pending: list[str] = []
    for message_index, message in enumerate(merged):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if message.get("content") in {None, "", "null"}:
                message.pop("content", None)
            pending = []
            for call_index, call in enumerate(message["tool_calls"]):
                if not call.get("id"):
                    seed = f"{row_digest}:{message_index}:{call_index}".encode()
                    call["id"] = "call_" + hashlib.sha256(seed).hexdigest()[:24]
                pending.append(call["id"])
        elif message.get("role") == "tool":
            if message.get("tool_call_id") in pending:
                pending.remove(message["tool_call_id"])
            elif pending:
                message["tool_call_id"] = pending.pop(0)
            else:
                return None
    if not any(
        message.get("role") == "assistant" and message.get("tool_calls")
        for message in merged
    ):
        return None
    merged = [message for message in merged if message.get("role") != "system"]
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + merged,
        "tools": TOOLS,
        "parallel_tool_calls": True,
    }


def redact_value(value: Any, counter: list[int]) -> Any:
    if isinstance(value, str):
        result = value
        for pattern in SENSITIVE_PATTERNS:
            result, matches = pattern.subn("[REDACTED]", result)
            counter[0] += matches
        return result
    if isinstance(value, list):
        return [redact_value(item, counter) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, counter) for key, item in value.items()}
    return value


def validate_sequence(row: dict[str, Any]) -> None:
    pending: list[str] = []
    for message in row["messages"]:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            pending.extend(call["id"] for call in message["tool_calls"])
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("Tool response references an unknown tool-call id.")
            pending.remove(call_id)


def initial_prompt(row: dict[str, Any]) -> str:
    return next(
        str(message.get("content") or "")
        for message in row["messages"]
        if message.get("role") == "user"
    )


def prepare_canonical_splits(
    raw_rows: list[dict[str, Any]],
    fresh_prompts: list[str],
    run_dir: Path,
    target_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    prompt_set = set(fresh_prompts)
    grouped: dict[str, list[dict[str, Any]]] = {}
    redaction_hits = 0
    transformed_count = 0
    for raw in raw_rows:
        transformed = transform_trace_row(raw)
        if transformed is None:
            continue
        transformed_count += 1
        counter = [0]
        transformed = redact_value(transformed, counter)
        redaction_hits += counter[0]
        validate_sequence(transformed)
        prompt = initial_prompt(transformed)
        if prompt in prompt_set:
            grouped.setdefault(prompt, []).append(transformed)
    if len(grouped) < target_count:
        raise RuntimeError(
            f"Only {len(grouped)} of {target_count} fresh independent conversations "
            "were present in the trace export."
        )
    selected_prompts = sorted(grouped)[:target_count]
    canonical = {
        prompt: max(
            grouped[prompt],
            key=lambda row: (
                len(row["messages"]),
                canonical_json(row),
            ),
        )
        for prompt in selected_prompts
    }
    keys = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in selected_prompts
    ]
    key_to_prompt = dict(zip(keys, selected_prompts))
    random.Random(42).shuffle(keys)
    split_keys = {
        "train": keys[:80],
        "validation": keys[80:90],
        "test": keys[90:100],
    }
    splits = {
        name: [canonical[key_to_prompt[key]] for key in values]
        for name, values in split_keys.items()
    }
    paths = {}
    for name, rows in splits.items():
        path = run_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    key_sets = {name: set(values) for name, values in split_keys.items()}
    overlaps = {
        "train_validation": len(key_sets["train"] & key_sets["validation"]),
        "train_test": len(key_sets["train"] & key_sets["test"]),
        "validation_test": len(key_sets["validation"] & key_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Conversation split overlap detected: {overlaps}")
    lineage = {
        "raw_rows": len(raw_rows),
        "transformed_tool_rows": transformed_count,
        "fresh_conversations_found": len(grouped),
        "selected_canonical_conversations": len(canonical),
        "canonical_selection": "longest repaired snapshot; canonical JSON tie-break",
        "split_conversation_counts": {
            name: len(rows) for name, rows in splits.items()
        },
        "split_row_counts": {
            name: len(rows) for name, rows in splits.items()
        },
        "conversation_overlap": overlaps,
        "redaction_hits": redaction_hits,
        "redaction_hits_after": sum(
            len(pattern.findall(canonical_json(splits)))
            for pattern in SENSITIVE_PATTERNS
        ),
        "split_prompt_sha256": split_keys,
        "split_sha256": {name: sha256(path) for name, path in paths.items()},
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "tools_sha256": hashlib.sha256(
            (DATA / "zava_tools.json").read_bytes()
        ).hexdigest(),
    }
    if lineage["redaction_hits_after"]:
        raise RuntimeError("Sensitive-pattern matches remain after redaction.")
    write_json(run_dir / "lineage.json", lineage)
    return splits, lineage


def calls_from_messages(
    messages: list[dict[str, Any]],
    *,
    first_action_only: bool,
) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        for call in message["tool_calls"]:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments or {}
                )
            except json.JSONDecodeError:
                arguments = {"_raw": raw_arguments}
            result.append({
                "name": function.get("name"),
                "arguments": arguments,
            })
        if first_action_only:
            break
    return result


def tool_call_score(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> int:
    if not expected:
        return 10 if not actual else 5
    if not actual:
        return 1
    expected_names = [call["name"] for call in expected]
    actual_names = [call["name"] for call in actual]
    expected_set = set(expected_names)
    actual_set = set(actual_names)
    overlap = expected_set & actual_set
    if not overlap:
        return 1
    if expected_set != actual_set:
        ratio = len(overlap) / len(expected_set | actual_set)
        return max(2, int(round(2 + ratio * 6)))
    expected_args = {call["name"]: call["arguments"] for call in expected}
    actual_args = {call["name"]: call["arguments"] for call in actual}
    return (
        10
        if all(
            expected_args[name] == actual_args.get(name)
            for name in expected_names
        )
        else 8
    )


def predict_first_action(
    model_client: Any,
    deployment: str,
    prompt: str,
) -> list[dict[str, Any]]:
    response = model_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        temperature=0,
        max_completion_tokens=2048,
    )
    calls = response.choices[0].message.tool_calls or []
    return [{
        "name": call.function.name,
        "arguments": json.loads(call.function.arguments or "{}"),
    } for call in calls]


def predict_full_trajectory(
    model_client: Any,
    deployment: str,
    prompt: str,
) -> list[dict[str, Any]]:
    response = model_client.responses.create(
        model=deployment,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        tools=RESPONSE_TOOLS,
        temperature=0,
        store=True,
    )
    calls = []
    for _ in range(8):
        function_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]
        if not function_calls:
            break
        outputs = []
        for item in function_calls:
            arguments = json.loads(item.arguments or "{}")
            calls.append({"name": item.name, "arguments": arguments})
            outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(tool_result(item.name, arguments)),
            })
        response = model_client.responses.create(
            model=deployment,
            previous_response_id=response.id,
            input=outputs,
            tools=RESPONSE_TOOLS,
            temperature=0,
            store=True,
        )
    return calls


def evaluate_deployment(
    model_client: Any,
    deployment: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    details = []
    for index, row in enumerate(rows):
        prompt = initial_prompt(row)
        expected_first = calls_from_messages(
            row["messages"], first_action_only=True
        )
        expected_full = calls_from_messages(
            row["messages"], first_action_only=False
        )
        actual_first = predict_first_action(model_client, deployment, prompt)
        actual_full = predict_full_trajectory(model_client, deployment, prompt)
        primary_score = tool_call_score(expected_first, actual_first)
        expected_names = [call["name"] for call in expected_full]
        actual_names = [call["name"] for call in actual_full]
        paired_count = min(len(expected_full), len(actual_full))
        exact_arguments = sum(
            expected_full[pair]["name"] == actual_full[pair]["name"]
            and expected_full[pair]["arguments"] == actual_full[pair]["arguments"]
            for pair in range(paired_count)
        )
        details.append({
            "row": index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "primary": {
                "expected": expected_first,
                "actual": actual_first,
                "score": primary_score,
                "pass": primary_score >= 8,
            },
            "strict": {
                "expected": expected_full,
                "actual": actual_full,
                "structural_score": tool_call_score(
                    expected_full, actual_full
                ),
                "exact_name_sequence": expected_names == actual_names,
                "exact_full_trajectory": expected_full == actual_full,
                "aligned_argument_exact_count": exact_arguments,
                "aligned_call_count": paired_count,
                "extra_call_count": max(0, len(actual_full) - len(expected_full)),
            },
        })
        print(
            f"{deployment} evaluation {index + 1}/{len(rows)} "
            f"primary={primary_score}"
        )
    primary_scores = [item["primary"]["score"] for item in details]
    strict = [item["strict"] for item in details]
    aligned_calls = sum(item["aligned_call_count"] for item in strict)
    return {
        "model": deployment,
        "count": len(details),
        "primary_original_compatible": {
            "method": "first assistant action; original tool_call_score",
            "average_score": sum(primary_scores) / len(primary_scores),
            "pass_threshold": 8,
            "pass_rate_percent": (
                100 * sum(score >= 8 for score in primary_scores)
                / len(primary_scores)
            ),
        },
        "secondary_strict": {
            "full_trajectory_average_score": (
                sum(item["structural_score"] for item in strict) / len(strict)
            ),
            "exact_name_sequence_percent": (
                100 * sum(item["exact_name_sequence"] for item in strict)
                / len(strict)
            ),
            "exact_full_trajectory_percent": (
                100 * sum(item["exact_full_trajectory"] for item in strict)
                / len(strict)
            ),
            "aligned_argument_accuracy_percent": (
                100
                * sum(item["aligned_argument_exact_count"] for item in strict)
                / aligned_calls
                if aligned_calls
                else 0
            ),
            "rows_with_extra_calls_percent": (
                100 * sum(item["extra_call_count"] > 0 for item in strict)
                / len(strict)
            ),
            "extra_call_count": sum(item["extra_call_count"] for item in strict),
        },
        "details": details,
    }


def wait_for_file(
    model_client: Any,
    file_id: str,
    timeout_seconds: float,
) -> Any:
    started = time.monotonic()
    while True:
        uploaded = model_client.files.retrieve(file_id)
        current = status_value(uploaded)
        if current == "processed":
            return uploaded
        if current in {"error", "failed", "expired", "cancelled"}:
            raise RuntimeError(
                f"Uploaded file {file_id} ended as {current}: "
                f"{getattr(uploaded, 'status_details', None)}"
            )
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"File {file_id} processing timed out.")
        time.sleep(5)


def upload_exact(
    model_client: Any,
    path: Path,
    timeout_seconds: float,
) -> Any:
    with path.open("rb") as handle:
        uploaded = model_client.files.create(
            file=(path.name, handle),
            purpose="fine-tune",
        )
    return wait_for_file(model_client, uploaded.id, timeout_seconds)


def matching_existing_job(
    model_client: Any,
    expected: dict[str, Any],
) -> Any | None:
    requested_id = os.environ.get("FOUNDRY_REUSE_JOB_ID", "").strip()
    if not requested_id:
        return None
    job = model_client.fine_tuning.jobs.retrieve(requested_id)
    method = getattr(job, "method", None)
    hyperparameters = getattr(job, "hyperparameters", None)
    method_data = (
        method.model_dump(mode="json")
        if hasattr(method, "model_dump")
        else method
    )
    hyperparameter_data = (
        hyperparameters.model_dump(mode="json")
        if hasattr(hyperparameters, "model_dump")
        else hyperparameters
    )
    actual = {
        "status": status_value(job),
        "model": getattr(job, "model", None),
        "training_file": getattr(job, "training_file", None),
        "validation_file": getattr(job, "validation_file", None),
        "method": method_data,
        "hyperparameters": hyperparameter_data,
    }
    write_json(expected["run_dir"] / "reuse_comparison.json", {
        "expected": {
            key: value for key, value in expected.items() if key != "run_dir"
        },
        "actual": json.loads(
            json.dumps(actual, default=lambda value: getattr(value, "__dict__", str(value)))
        ),
    })
    if (
        actual["status"] == "succeeded"
        and actual["model"] == expected["model"]
        and actual["training_file"] == expected["training_file"]
        and actual["validation_file"] == expected["validation_file"]
        and actual["method"]["type"] == expected["method"]
        and actual["hyperparameters"] == {
            "n_epochs": expected["n_epochs"],
            "learning_rate_multiplier": expected["learning_rate_multiplier"],
            "batch_size": expected["batch_size"],
        }
    ):
        return job
    raise RuntimeError(
        f"FOUNDRY_REUSE_JOB_ID={requested_id} is not an exact uploaded-file/model match."
    )


def monitor_job(
    model_client: Any,
    job_id: str,
    timeout_seconds: float,
) -> Any:
    started = time.monotonic()
    seen_events: set[str] = set()
    while True:
        job = model_client.fine_tuning.jobs.retrieve(job_id)
        events = model_client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=100,
        )
        for event in reversed(list(getattr(events, "data", events))):
            event_id = str(getattr(event, "id", ""))
            if event_id and event_id not in seen_events:
                seen_events.add(event_id)
                print(f"{getattr(event, 'created_at', '')}: {event.message}")
        current = status_value(job)
        print(f"fine-tuning job {job_id}: {current}")
        if current in TERMINAL_JOB_STATES:
            if current != "succeeded":
                raise RuntimeError(
                    f"Fine-tuning job ended as {current}: "
                    f"{getattr(job, 'error', None)}"
                )
            return job
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Fine-tuning job {job_id} timed out.")
        time.sleep(30)


def ensure_deployment(
    credential: DefaultAzureCredential,
    endpoint: str,
    fine_tuned_model: str,
    deployment_name: str,
) -> None:
    subscription_id = required("FOUNDRY_SUBSCRIPTION_ID")
    resource_group = required("FOUNDRY_RESOURCE_GROUP")
    account_name = urlparse(endpoint).hostname.removesuffix(
        ".services.ai.azure.com"
    )
    management = CognitiveServicesManagementClient(
        credential, subscription_id
    )
    deployment = Deployment(
        properties=DeploymentProperties(
            model=DeploymentModel(
                format="OpenAI",
                name=fine_tuned_model,
                version="1",
            )
        ),
        sku=Sku(
            name=os.environ.get(
                "FOUNDRY_DEPLOYMENT_SKU", "GlobalStandard"
            ),
            capacity=int(os.environ.get("FOUNDRY_DEPLOYMENT_CAPACITY", "500")),
        ),
    )
    management.deployments.begin_create_or_update(
        resource_group_name=resource_group,
        account_name=account_name,
        deployment_name=deployment_name,
        deployment=deployment,
    ).result()


def download_training_metrics(
    model_client: Any,
    job: Any,
    run_dir: Path,
) -> dict[str, Any]:
    result_files = list(getattr(job, "result_files", None) or [])
    summaries = []
    for index, file_id in enumerate(result_files):
        path = run_dir / f"training_results_{index}.csv"
        path.write_bytes(model_client.files.content(file_id=file_id).read())
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        summaries.append({
            "file_id": file_id,
            "path": str(path),
            "sha256": sha256(path),
            "row_count": len(rows),
            "last_row": rows[-1] if rows else None,
            "minimum_valid_loss": min(
                (
                    float(row["valid_loss"])
                    for row in rows
                    if row.get("valid_loss") not in {None, ""}
                ),
                default=None,
            ),
        })
    return {"result_files": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-conversations", type=int, default=120)
    parser.add_argument("--target-conversations", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--ingestion-wait-seconds", type=int, default=120)
    parser.add_argument("--max-trace-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target_conversations != 100:
        raise ValueError("Original compatibility requires exactly 100 conversations.")
    if args.fresh_conversations < args.target_conversations:
        raise ValueError("Generate at least 100 fresh conversations.")

    load_dotenv(ROOT / ".env")
    endpoint = required("FOUNDRY_PROJECT_ENDPOINT")
    agent_name = required("FOUNDRY_AGENT_NAME")
    agent_version = required("FOUNDRY_AGENT_VERSION")
    teacher_deployment = os.environ.get(
        "FOUNDRY_TEACHER_DEPLOYMENT", "gpt-4.1-mini"
    )
    student_deployment = os.environ.get(
        "FOUNDRY_STUDENT_DEPLOYMENT", "gpt-4.1-nano"
    )
    student_model = os.environ.get(
        "FOUNDRY_STUDENT_MODEL", "gpt-4.1-nano-2025-04-14"
    )
    file_timeout = float(os.environ.get(
        "FOUNDRY_FILE_TIMEOUT_SECONDS", "900"
    ))
    job_timeout = float(os.environ.get(
        "FOUNDRY_JOB_TIMEOUT_SECONDS", "21600"
    ))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUTS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print("run directory:", run_dir)

    credential = DefaultAzureCredential()
    project = AIProjectClient(
        endpoint=endpoint,
        credential=credential,
        allow_preview=True,
    )
    model_client = project.get_openai_client()
    agent = project.agents.get_version(
        agent_name=agent_name,
        agent_version=agent_version,
    )
    connections = list(project.connections.list())
    app_insights = [
        connection.name
        for connection in connections
        if "appinsight" in str(getattr(connection, "type", "")).casefold()
        or "application_insights" in str(
            getattr(connection, "type", "")
        ).casefold()
    ]
    if not app_insights:
        raise RuntimeError("The project has no Application Insights connection.")
    deployment_names = {
        deployment.name for deployment in project.deployments.list()
    }
    for required_deployment in (teacher_deployment, student_deployment):
        if required_deployment not in deployment_names:
            raise RuntimeError(
                f"Required deployment {required_deployment!r} is absent."
            )

    prompts = build_prompts(args.fresh_conversations, args.seed)
    write_json(run_dir / "fresh_prompts.json", {
        "count": len(prompts),
        "sha256": hashlib.sha256(
            "\n".join(prompts).encode("utf-8")
        ).hexdigest(),
        "prompts": prompts,
    })
    acquisition_start = datetime.now(timezone.utc) - timedelta(minutes=1)
    generation = generate_fresh_conversations(
        project,
        agent.name,
        teacher_deployment,
        prompts,
        args.concurrency,
        max_rounds=8,
        output_path=run_dir / "trace_generation.json",
    )
    print(
        f"waiting {args.ingestion_wait_seconds}s for App Insights ingestion"
    )
    time.sleep(args.ingestion_wait_seconds)
    acquisition_end = datetime.now(timezone.utc) + timedelta(seconds=15)
    raw_path = run_dir / "raw_traces.jsonl"
    acquisition = acquire_traces(
        project,
        model_client,
        agent.name,
        agent_version,
        acquisition_start,
        acquisition_end,
        args.max_trace_samples,
        raw_path,
    )
    raw_rows = read_jsonl(raw_path)
    splits, lineage = prepare_canonical_splits(
        raw_rows,
        prompts,
        run_dir,
        args.target_conversations,
    )

    train_path = run_dir / "train.jsonl"
    validation_path = run_dir / "validation.jsonl"
    train_file = upload_exact(model_client, train_path, file_timeout)
    validation_file = upload_exact(
        model_client, validation_path, file_timeout
    )
    exact_uploads = {
        "training": {
            "local_path": str(train_path),
            "local_sha256": sha256(train_path),
            "file_id": train_file.id,
            "bytes": train_path.stat().st_size,
        },
        "validation": {
            "local_path": str(validation_path),
            "local_sha256": sha256(validation_path),
            "file_id": validation_file.id,
            "bytes": validation_path.stat().st_size,
        },
    }
    write_json(run_dir / "uploads.json", exact_uploads)

    expected = {
        "run_dir": run_dir,
        "model": student_model,
        "training_file": train_file.id,
        "validation_file": validation_file.id,
        "method": "supervised",
        "n_epochs": 3,
        "learning_rate_multiplier": 1.0,
        "batch_size": 1,
    }
    job = matching_existing_job(model_client, expected)
    if job is None:
        job = model_client.fine_tuning.jobs.create(
            model=student_model,
            training_file=train_file.id,
            validation_file=validation_file.id,
            method={
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "n_epochs": 3,
                        "learning_rate_multiplier": 1.0,
                        "batch_size": 1,
                    }
                },
            },
            suffix=f"agent-traces-parity-{run_id[-7:-1]}",
            extra_body={"trainingType": "GlobalStandard"},
        )
    completed_job = monitor_job(model_client, job.id, job_timeout)
    metrics = download_training_metrics(
        model_client, completed_job, run_dir
    )

    fine_tuned_model = completed_job.fine_tuned_model
    deployment_name = (
        os.environ.get("FOUNDRY_FINE_TUNED_DEPLOYMENT", "").strip()
        or f"agent-traces-parity-{job.id[-8:]}"
    )
    ensure_deployment(
        credential,
        endpoint,
        fine_tuned_model,
        deployment_name,
    )
    ready = model_client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "Reply with ready."}],
        max_completion_tokens=8,
    )
    if not ready.choices:
        raise RuntimeError("Fine-tuned deployment returned no choices.")

    evaluations = {
        "teacher": evaluate_deployment(
            model_client, teacher_deployment, splits["test"]
        ),
        "student_base": evaluate_deployment(
            model_client, student_deployment, splits["test"]
        ),
        "student_fine_tuned": evaluate_deployment(
            model_client, deployment_name, splits["test"]
        ),
    }
    base_primary = evaluations["student_base"][
        "primary_original_compatible"
    ]
    fine_tuned_primary = evaluations["student_fine_tuned"][
        "primary_original_compatible"
    ]
    gain = (
        100
        * (
            fine_tuned_primary["average_score"]
            - base_primary["average_score"]
        )
        / base_primary["average_score"]
        if base_primary["average_score"]
        else None
    )
    evaluations["outcome"] = {
        "primary_relative_average_score_gain_percent": gain,
        "primary_pass_rate_gain_points": (
            fine_tuned_primary["pass_rate_percent"]
            - base_primary["pass_rate_percent"]
        ),
        "historical_calculation": "(fine_tuned_average - base_average) / base_average * 100",
    }
    write_json(run_dir / "evaluation.json", evaluations)

    project_name = endpoint.rstrip("/").split("/")[-1]
    account_name = urlparse(endpoint).hostname.removesuffix(
        ".services.ai.azure.com"
    )
    resource_id = os.environ.get(
        "FOUNDRY_PROJECT_RESOURCE_ID", ""
    ).strip()
    job_link = (
        f"https://ai.azure.com/nextgen/r/{resource_id.lstrip('/')},"
        f"{account_name},,{project_name},{project_name}/build/fine-tune/"
        f"{job.id}/logs"
        if resource_id
        else None
    )
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        "original_compatibility_contract": {
            "source_notebook_requested_trace_rows": 100,
            "source_readme_historical_conversations": (
                "approximately 100 conversations across 5 sessions"
            ),
            "source_generator_defaults": {
                "default_conversations": 30,
                "README_example_conversations": 40,
            },
            "independent_conversations": 100,
            "canonical_rows": 100,
            "splits": {"train": 80, "validation": 10, "test": 10},
            "teacher_model": "gpt-4.1-mini-2025-04-14",
            "student_model": "gpt-4.1-nano-2025-04-14",
            "epochs": 3,
            "learning_rate_multiplier": 1.0,
            "batch_size": 1,
            "primary_evaluator": (
                "first assistant action/tool-call set and arguments"
            ),
            "score_function": {
                "exact_names_and_arguments": 10,
                "same_name_set_argument_difference": 8,
                "partial_name_overlap": "2-8 by overlap/union ratio",
                "no_overlap_or_missing_output": 1,
                "no_reference_no_output": 10,
                "no_reference_with_output": 5,
            },
            "pass_threshold": 8,
            "evaluation_context": (
                "system prompt plus first user message, temperature 0, "
                "chat-completions tools, first assistant action only"
            ),
            "gain": "(fine_tuned_average - base_average) / base_average * 100",
            "historical_result": {
                "base_average": 7.38,
                "base_pass_rate_percent": 60.0,
                "fine_tuned_average": 8.60,
                "fine_tuned_pass_rate_percent": 100.0,
                "reported_gain_percent": 16.5,
            },
            "source_ambiguity": (
                "The original README says approximately 100 conversations across "
                "five sessions while the notebook requests 100 transformed trace "
                "rows. This run resolves that ambiguity conservatively as 100 "
                "independent conversations and one canonical row per conversation."
            ),
        },
        "project": {
            "endpoint": endpoint,
            "resource_id": resource_id or None,
            "application_insights_connections": app_insights,
        },
        "teacher": {
            "agent_name": agent.name,
            "agent_version": str(agent.version),
            "deployment": teacher_deployment,
        },
        "student": {
            "base_model": student_model,
            "base_deployment": student_deployment,
            "fine_tuned_model": fine_tuned_model,
            "fine_tuned_deployment": deployment_name,
        },
        "generation": {
            key: value
            for key, value in generation.items()
            if key not in {"successes"}
        },
        "acquisition": acquisition,
        "lineage": lineage,
        "uploads": exact_uploads,
        "training": {
            "job_id": job.id,
            "job_link": job_link,
            "status": status_value(completed_job),
            "training_type": "GlobalStandard",
            "model": student_model,
            "method": "supervised",
            "hyperparameters": {
                "n_epochs": 3,
                "learning_rate_multiplier": 1.0,
                "batch_size": 1,
            },
            "training_file_id": train_file.id,
            "validation_file_id": validation_file.id,
            "fine_tuned_model": fine_tuned_model,
            "trained_tokens": getattr(completed_job, "trained_tokens", None),
            "created_at": getattr(completed_job, "created_at", None),
            "finished_at": getattr(completed_job, "finished_at", None),
            "metrics": metrics,
        },
        "evaluation": evaluations,
    }
    write_json(run_dir / "live_validation_summary.json", summary)
    write_json(OUTPUTS / "latest.json", {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": str(run_dir / "live_validation_summary.json"),
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
