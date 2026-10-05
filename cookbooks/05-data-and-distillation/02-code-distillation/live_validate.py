from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import data_designer.config as dd
from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from data_designer.engine.validators.python import (
    RECORD_ID_COLUMN_NAME,
    PythonValidator,
)
from data_designer.interface import DataDesigner
from data_designer.interface.errors import (
    DataDesignerEarlyShutdownError,
    DataDesignerGenerationError,
)
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
OUTPUTS = ROOT / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUTS.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant that writes clean, well-structured "
    "Python code to solve the user's request."
)
STUDENT_MODEL_DEFAULT = "gpt-4.1-mini-2025-04-14"
PARITY_GENERATED_COUNT = 2_000
PARITY_TRAIN_COUNT = 1_500
PARITY_EVALUATION_COUNT = 84
PARITY_COHORT_COUNT = PARITY_TRAIN_COUNT + PARITY_EVALUATION_COUNT
PARITY_QUALITY_THRESHOLD = 3.0
PARITY_SEED = 42
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}
TERMINAL_FILE_STATES = {"processed", "error", "expired", "failed", "cancelled"}
AZ_COMMAND = shutil.which("az.cmd") or shutil.which("az")

CORRECTNESS_PROMPT = (
    "You are evaluating Python code written to solve a programming task.\n\n"
    "## Task\n{instruction}\n\n"
    "## Reference Solution\n{reference}\n\n"
    "## Candidate Code\n{candidate}\n\n"
    "Score ONLY correctness: does the candidate code correctly solve the task?\n"
    "- 9-10: Fully correct\n- 7-8: Mostly correct, minor issues\n"
    "- 5-6: Partially correct\n- 3-4: Major issues\n- 1-2: Fails entirely\n\n"
    "Respond with ONLY a single integer from 1 to 10."
)
CONCISENESS_PROMPT = (
    "You are evaluating Python code written to solve a programming task.\n\n"
    "## Task\n{instruction}\n\n"
    "## Reference Solution\n{reference}\n\n"
    "## Candidate Code\n{candidate}\n\n"
    "Score ONLY conciseness: is the code appropriately minimal and clean?\n"
    "- 9-10: Clean, minimal\n- 7-8: Reasonable length\n"
    "- 5-6: Somewhat verbose\n- 3-4: Very verbose\n- 1-2: Bloated\n\n"
    "Respond with ONLY a single integer from 1 to 10."
)

SANDBOX_INSTRUCTION = (
    "Write a Python script that reads APP_MODE, defaulting to development. "
    "Print 'Debug features enabled' for development, 'Running in production mode' "
    "for production, and 'Unknown mode: <value>' otherwise. Match "
    "case-insensitively."
)
SANDBOX_CASES = [
    ("development", "Debug features enabled"),
    ("DEVELOPMENT", "Debug features enabled"),
    ("production", "Running in production mode"),
    ("Staging", "Unknown mode: staging"),
]


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured.")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_code_fence(source: str) -> str:
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines)
    return text.strip()


def syntax_valid(source: str) -> bool:
    ast.parse(strip_code_fence(source))
    return True


def average_score(value: Any) -> float:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        return 0.0
    nested = [
        entry.get("score")
        for entry in value.values()
        if isinstance(entry, dict) and isinstance(entry.get("score"), (int, float))
    ]
    if nested:
        return sum(nested) / len(nested)
    legacy = value.get("scores", {})
    scores = [score for score in legacy.values() if isinstance(score, (int, float))]
    return sum(scores) / len(scores) if scores else 0.0


def validator_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in {"true", "false"}:
            return text.casefold() == "true"
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return True
    if isinstance(value, dict):
        return bool(value.get("is_valid", True))
    return True


def install_utf8_python_validator() -> None:
    def _write_code_to_file_utf8(
        self: Any,
        row: Any,
        code_column: str,
        path: str,
    ) -> None:
        file_path = (
            Path(path)
            / f"{self._get_module_name(row[RECORD_ID_COLUMN_NAME], code_column)}.py"
        )
        value = row[code_column]
        file_path.write_text(
            "" if value is None else str(value),
            encoding="utf-8",
            errors="replace",
            newline="\n",
        )

    PythonValidator._write_code_to_file = _write_code_to_file_utf8


def build_designer_config(
    teacher: str,
    parallelism: int = 12,
) -> dd.DataDesignerConfigBuilder:
    config = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="teacher",
                model=teacher,
                provider="foundry-project",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=1.0,
                    top_p=1.0,
                    timeout=180,
                    max_parallel_requests=parallelism,
                    extra_body={"max_completion_tokens": 2048},
                ),
            ),
            dd.ModelConfig(
                alias="teacher-code",
                model=teacher,
                provider="foundry-project",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.4,
                    top_p=0.95,
                    timeout=180,
                    max_parallel_requests=parallelism,
                    extra_body={"max_completion_tokens": 4096},
                ),
            ),
        ]
    )
    config.add_column(
        dd.SamplerColumnConfig(
            name="industry_sector",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Healthcare", "Finance", "Technology"]
            ),
        )
    )
    config.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="industry_sector",
                values={
                    "Healthcare": [
                        "Electronic Health Records (EHR) Systems",
                        "Telemedicine Platforms",
                        "AI-Powered Diagnostic Tools",
                    ],
                    "Finance": [
                        "Fraud Detection Software",
                        "Automated Trading Systems",
                        "Personal Finance Apps",
                    ],
                    "Technology": [
                        "Cloud Computing Platforms",
                        "AI and Machine Learning Platforms",
                        "DevOps and CI/CD Tools",
                    ],
                },
            ),
        )
    )
    config.add_column(
        dd.SamplerColumnConfig(
            name="code_complexity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Beginner", "Intermediate", "Advanced"]
            ),
        )
    )
    config.add_column(
        dd.SamplerColumnConfig(
            name="code_concept",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="code_complexity",
                values={
                    "Beginner": [
                        "Variables",
                        "Data Types",
                        "Functions",
                        "Loops",
                        "Classes",
                    ],
                    "Intermediate": [
                        "List Comprehensions",
                        "OOP",
                        "Lambda Functions",
                        "Web Frameworks",
                        "Pandas",
                    ],
                    "Advanced": [
                        "Multithreading",
                        "Context Managers",
                        "Generators",
                    ],
                },
            ),
        )
    )
    config.add_column(
        dd.SamplerColumnConfig(
            name="instruction_phrase",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "Write a function that",
                    "Create a class that",
                    "Implement a script",
                    "Can you create a function",
                    "Develop a module that",
                ]
            ),
        )
    )
    config.add_column(
        dd.LLMTextColumnConfig(
            name="instruction",
            model_alias="teacher",
            system_prompt=(
                "You are an expert at generating clear and specific programming tasks."
            ),
            prompt=(
                'Generate an instruction to create Python code that solves a specific problem.\n'
                'The instruction should begin with: "{{ instruction_phrase }}".\n\n'
                "Guidelines:\n"
                "* Industry Relevance: Pertains to {{ industry_sector }} / {{ topic }}.\n"
                "* Code Complexity: Tailored to {{ code_complexity }} level using "
                "{{ code_concept }}.\n"
                "* Clarity: Clear, unambiguous, with sufficient context.\n"
            ),
        )
    )
    config.add_column(
        dd.LLMCodeColumnConfig(
            name="code_implementation",
            model_alias="teacher-code",
            code_lang=dd.CodeLang.PYTHON,
            system_prompt=(
                "You are an expert Python programmer. You respond ONLY with Python code. "
                "Do NOT include any explanatory text. Wrap your response in a ```python "
                "code block."
            ),
            prompt=(
                "Write Python code for the following instruction:\n"
                "Instruction: {{ instruction }}\n\n"
                "Guidelines:\n"
                "* Clean, complete, self-contained, and correct code.\n"
                "* Import any necessary libraries.\n"
                "* Code complexity: {{ code_complexity }} level using {{ code_concept }}.\n"
                "* Return ONLY Python code inside a ```python block.\n"
            ),
        )
    )
    scores = [
        dd.Score(
            name="Relevance",
            description="Adherence to instructions",
            options={
                4: "Perfectly meets requirements.",
                3: "Meets most requirements.",
                2: "Moderate deviation.",
                1: "Significant deviations.",
                0: "Does not adhere.",
            },
        ),
        dd.Score(
            name="Pythonic",
            description="Pythonic code and best practices",
            options={
                4: "Exemplifies Pythonic principles.",
                3: "Closely follows conventions.",
                2: "Generally follows conventions.",
                1: "Loosely follows conventions.",
                0: "Not Pythonic.",
            },
        ),
        dd.Score(
            name="Readability",
            description="Readability and maintainability",
            options={
                4: "Excellently formatted, follows PEP 8.",
                3: "Well-formatted and readable.",
                2: "Somewhat readable.",
                1: "Minimal formatting.",
                0: "Unreadable.",
            },
        ),
        dd.Score(
            name="Efficiency",
            description="Efficiency and performance",
            options={
                4: "Highly efficient.",
                3: "Efficient with minor optimization areas.",
                2: "Moderately efficient.",
                1: "Poor efficiency.",
                0: "Highly inefficient.",
            },
        ),
    ]
    config.add_column(
        dd.LLMJudgeColumnConfig(
            name="code_judge_result",
            model_alias="teacher-code",
            prompt=(
                "You are an expert Python programmer and code reviewer.\n\n"
                "Score the Generated Python Code based on the Natural Language Prompt.\n\n"
                "Natural Language Prompt:\n{{ instruction }}\n\n"
                "Generated Python Code:\n{{ code_implementation }}"
            ),
            scores=scores,
        )
    )
    config.add_column(
        dd.ValidationColumnConfig(
            name="code_validity_result",
            validator_type=dd.ValidatorType.CODE,
            target_columns=["code_implementation"],
            validator_params=dd.CodeValidatorParams(code_lang=dd.CodeLang.PYTHON),
            batch_size=100,
        )
    )
    return config


def generate_parity_data(
    credential: AzureCliCredential,
    model_client: Any,
    teacher: str,
    generated_count: int,
    train_count: int,
    evaluation_count: int,
    quality_threshold: float,
    generation_parallelism: int,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    if (
        generated_count != PARITY_GENERATED_COUNT
        or train_count != PARITY_TRAIN_COUNT
        or evaluation_count != PARITY_EVALUATION_COUNT
        or quality_threshold != PARITY_QUALITY_THRESHOLD
    ):
        raise RuntimeError(
            "Primary parity mode requires exactly 2,000 generated, 1,500 training, "
            "84 held-out evaluation rows, and quality threshold 3.0."
        )
    install_utf8_python_validator()
    raw_rows: list[dict[str, Any]] = []
    remaining = generated_count
    batch_index = 0
    while remaining:
        batch_count = min(1_000, remaining)
        result = None
        for attempt in range(1, 4):
            token = credential.get_token("https://ai.azure.com/.default").token
            provider = dd.ModelProvider(
                name="foundry-project",
                endpoint=str(model_client.base_url).rstrip("/"),
                provider_type="openai",
                api_key=token,
            )
            designer = DataDesigner(
                artifact_path=OUTPUTS / "data-designer",
                model_providers=[provider],
            )
            try:
                result = designer.create(
                    build_designer_config(teacher, generation_parallelism),
                    num_records=batch_count,
                    dataset_name=(
                        f"code-distillation-parity-{batch_index:02d}-"
                        f"attempt-{attempt:02d}"
                    ),
                )
                break
            except (DataDesignerEarlyShutdownError, DataDesignerGenerationError):
                if attempt == 3:
                    raise
                time.sleep(30)
        if result is None:
            raise RuntimeError(f"Data Designer batch {batch_index} did not run.")
        batch_rows = result.load_dataset().to_dict(orient="records")
        if len(batch_rows) != batch_count:
            raise RuntimeError(
                f"Data Designer batch {batch_index} returned {len(batch_rows)} rows; "
                f"{batch_count} were requested."
            )
        raw_rows.extend(batch_rows)
        remaining -= batch_count
        batch_index += 1
    raw_path = OUTPUTS / "generated-raw.jsonl"
    write_jsonl(raw_path, raw_rows)
    if len(raw_rows) != generated_count:
        raise RuntimeError(
            f"Data Designer returned {len(raw_rows)} rows; parity requires {generated_count}."
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    duplicate_count = 0
    for index, row in enumerate(raw_rows):
        try:
            instruction = str(row["instruction"]).strip()
            code = str(row["code_implementation"]).strip()
            score = average_score(row["code_judge_result"])
            if not validator_passed(row.get("code_validity_result")):
                raise ValueError("Data Designer syntax validator rejected the code.")
            syntax_valid(code)
            if score < quality_threshold:
                raise ValueError(
                    f"judge average {score:.2f} below {quality_threshold:.1f}"
                )
            dedup_key = instruction.casefold()[:160]
            if dedup_key in seen_instructions:
                duplicate_count += 1
                continue
            seen_instructions.add(dedup_key)
            accepted.append(
                {
                    "source_index": index,
                    "instruction": instruction,
                    "reference": code,
                    "average_quality": score,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": code},
                    ],
                }
            )
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            rejected.append({"source_index": index, "reason": str(exc)})

    if len(accepted) < train_count + evaluation_count:
        raise RuntimeError(
            f"Only {len(accepted)} unique rows passed the original quality contract; "
            f"{train_count + evaluation_count} are required for exact parity."
        )

    random.Random(PARITY_SEED).shuffle(accepted)
    parity_rows = accepted[: train_count + evaluation_count]
    surplus_rows = accepted[train_count + evaluation_count :]
    train_source = parity_rows[:train_count]
    evaluation_source = parity_rows[train_count:]
    train_rows = [{"messages": row["messages"]} for row in train_source]
    evaluation_rows = [
        {
            "messages": row["messages"],
            "source_index": row["source_index"],
            "average_quality": row["average_quality"],
        }
        for row in evaluation_source
    ]

    train_path = OUTPUTS / "train.jsonl"
    validation_path = OUTPUTS / "validation-and-evaluation.jsonl"
    accepted_path = OUTPUTS / "accepted-filtered.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, [{"messages": row["messages"]} for row in evaluation_source])
    write_jsonl(accepted_path, accepted)

    lineage = {
        "parity_contract": {
            "generated_requested": generated_count,
            "training_rows": train_count,
            "validation_rows": evaluation_count,
            "independent_evaluation_rows": evaluation_count,
            "validation_and_evaluation_are_same_holdout": True,
            "quality_threshold": quality_threshold,
            "seed": PARITY_SEED,
        },
        "generated_count": len(raw_rows),
        "quality_and_syntax_accepted_count": len(accepted),
        "deduplicated_count": duplicate_count,
        "rejected_count": len(rejected),
        "rejected": rejected,
        "parity_cohort_count": len(parity_rows),
        "surplus_accepted_count": len(surplus_rows),
        "training_count": len(train_rows),
        "validation_count": len(evaluation_rows),
        "independent_evaluation_count": len(evaluation_rows),
        "raw_sha256": sha256(raw_path),
        "accepted_sha256": sha256(accepted_path),
        "training_sha256": sha256(train_path),
        "validation_evaluation_sha256": sha256(validation_path),
        "rule": (
            "All train and held-out rows come only from this run's 2,000 generated "
            "rows after the original score, syntax, and deduplication gates. The "
            "deterministically shuffled parity cohort is capped at the documented "
            "winning scale of 1,500 train plus 84 held-out rows."
        ),
        "generation_parallelism": generation_parallelism,
    }
    write_json(OUTPUTS / "lineage.json", lineage)
    return train_path, validation_path, lineage, evaluation_rows


def load_completed_generation(
    directory: Path,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    train_path = directory / "train.jsonl"
    validation_path = directory / "validation-and-evaluation.jsonl"
    lineage_path = directory / "lineage.json"
    for path in (train_path, validation_path, lineage_path):
        if not path.is_file():
            raise RuntimeError(f"Resume generation artifact is missing: {path}")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if (
        lineage.get("generated_count") != PARITY_GENERATED_COUNT
        or lineage.get("training_count") != PARITY_TRAIN_COUNT
        or lineage.get("validation_count") != PARITY_EVALUATION_COUNT
        or sha256(train_path) != lineage.get("training_sha256")
        or sha256(validation_path)
        != lineage.get("validation_evaluation_sha256")
    ):
        raise RuntimeError(
            "Resume generation artifacts do not match the exact parity contract."
        )
    evaluation_rows = []
    for index, row in enumerate(read_jsonl(validation_path)):
        evaluation_rows.append(
            {
                "messages": row["messages"],
                "source_index": index,
                "average_quality": None,
            }
        )
    return train_path, validation_path, lineage, evaluation_rows


def wait_for_file(client: Any, file_id: str, timeout: int, poll_interval: int) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        uploaded = client.files.retrieve(file_id)
        status = str(uploaded.status).casefold()
        if status in TERMINAL_FILE_STATES:
            if status != "processed":
                raise RuntimeError(f"File {file_id} ended as {status}: {uploaded}")
            return uploaded
        time.sleep(poll_interval)
    raise TimeoutError(f"File {file_id} did not process within {timeout} seconds.")


def upload(client: Any, path: Path, timeout: int, poll_interval: int) -> Any:
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="fine-tune")
    return wait_for_file(client, uploaded.id, timeout, poll_interval)


def wait_for_job(client: Any, job_id: str, timeout: int, poll_interval: int) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = str(job.status).casefold()
        print(datetime.now(timezone.utc).isoformat(), job_id, status, flush=True)
        if status in TERMINAL_JOB_STATES:
            return job
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not reach terminal state within {timeout} seconds.")


def matching_completed_job(
    project_endpoint: str,
    teacher: str,
    student_model: str,
    train_hash: str,
    validation_hash: str,
) -> dict[str, Any] | None:
    expected = {
        "project_endpoint": project_endpoint,
        "teacher_deployment": teacher,
        "student_model": student_model,
        "training_sha256": train_hash,
        "validation_sha256": validation_hash,
        "generated_count": PARITY_GENERATED_COUNT,
        "training_count": PARITY_TRAIN_COUNT,
        "validation_count": PARITY_EVALUATION_COUNT,
        "n_epochs": 1,
        "learning_rate_multiplier": 1.3,
        "batch_size": 1,
    }
    for candidate in sorted((ROOT / "outputs").glob("*/reuse-contract.json"), reverse=True):
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(record.get(key) == value for key, value in expected.items()):
            terminal = candidate.parent / "terminal-job.json"
            if terminal.is_file():
                job = json.loads(terminal.read_text(encoding="utf-8"))
                if str(job.get("status", "")).casefold() == "succeeded":
                    return {"record": record, "job": job, "directory": str(candidate.parent)}
    return None


def parse_account_name(project_endpoint: str) -> str:
    match = re.match(r"https://([^.]+)\.services\.ai\.azure\.com/", project_endpoint)
    if not match:
        raise RuntimeError("Could not derive the Foundry account name from the project endpoint.")
    return match.group(1)


def resolve_arm_account(account: str) -> dict[str, str]:
    if not AZ_COMMAND:
        raise RuntimeError("Azure CLI executable was not found on PATH.")
    query = (
        "Resources | where type =~ 'microsoft.cognitiveservices/accounts' "
        f"| where name == '{account}' "
        "| project subscriptionId, resourceGroup, name, location, id"
    )
    raw = subprocess.check_output(
        [AZ_COMMAND, "graph", "query", "-q", query, "-o", "json"],
        text=True,
        encoding="utf-8",
    )
    data = json.loads(raw)["data"]
    if len(data) != 1:
        raise RuntimeError(f"Expected one ARM account for {account}, found {len(data)}.")
    return data[0]


def deploy_model(
    project: AIProjectClient,
    project_endpoint: str,
    model: str,
    job_id: str,
) -> tuple[str, str]:
    for deployment in project.deployments.list():
        if str(deployment.get("modelName")) == model:
            return str(deployment["name"]), "existing"
    arm = resolve_arm_account(parse_account_name(project_endpoint))
    deployment_name = f"nl2py-parity-{job_id.removeprefix('ftjob-')[-12:]}"
    subprocess.run(
        [
            AZ_COMMAND,
            "cognitiveservices",
            "account",
            "deployment",
            "create",
            "--subscription",
            arm["subscriptionId"],
            "--resource-group",
            arm["resourceGroup"],
            "--name",
            arm["name"],
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
            "10",
            "-o",
            "none",
        ],
        check=True,
        text=True,
    )
    return deployment_name, "created"


def generate_candidate(client: Any, deployment: str, instruction: str) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        temperature=0,
        max_completion_tokens=16_384,
    )
    return response.choices[0].message.content or ""


def parse_integer_score(text: str) -> int:
    match = re.search(r"(?<!\d)(10|[1-9])(?!\d)", text.strip())
    if not match:
        raise ValueError(f"Evaluator did not return an integer score: {text!r}")
    return int(match.group(1))


def judge_dimension(
    client: Any,
    evaluator: str,
    prompt_template: str,
    instruction: str,
    reference: str,
    candidate: str,
) -> int:
    response = client.chat.completions.create(
        model=evaluator,
        messages=[
            {
                "role": "user",
                "content": prompt_template.format(
                    instruction=instruction,
                    reference=reference,
                    candidate=candidate,
                ),
            }
        ],
        temperature=0,
        max_completion_tokens=32,
    )
    return parse_integer_score(response.choices[0].message.content or "")


def evaluate_one(
    client: Any,
    evaluator: str,
    deployment: str,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    messages = row["messages"]
    instruction = messages[1]["content"]
    reference = messages[2]["content"]
    candidate = generate_candidate(client, deployment, instruction)
    correctness = judge_dimension(
        client,
        evaluator,
        CORRECTNESS_PROMPT,
        instruction,
        reference,
        candidate,
    )
    conciseness = judge_dimension(
        client,
        evaluator,
        CONCISENESS_PROMPT,
        instruction,
        reference,
        candidate,
    )
    combined = (correctness + conciseness) / 2
    return {
        "index": index,
        "source_index": row["source_index"],
        "instruction": instruction,
        "reference": reference,
        "candidate": candidate,
        "correctness": correctness,
        "conciseness": conciseness,
        "combined": combined,
        "pass_at_8_combined": combined >= 8,
        "legacy_pass_at_8_correctness": correctness >= 8,
        "candidate_characters": len(candidate),
        "reference_characters": len(reference),
    }


def summarize_evaluation(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    if count != PARITY_EVALUATION_COUNT:
        raise RuntimeError(
            f"Primary evaluation scored {count} rows; parity requires "
            f"{PARITY_EVALUATION_COUNT}."
        )
    return {
        "count": count,
        "correctness": sum(item["correctness"] for item in items) / count,
        "conciseness": sum(item["conciseness"] for item in items) / count,
        "combined": sum(item["combined"] for item in items) / count,
        "pass_at_8_combined_percent": (
            sum(item["pass_at_8_combined"] for item in items) / count * 100
        ),
        "legacy_pass_at_8_correctness_percent": (
            sum(item["legacy_pass_at_8_correctness"] for item in items) / count * 100
        ),
        "mean_candidate_characters": (
            sum(item["candidate_characters"] for item in items) / count
        ),
    }


def evaluate_deployment(
    client: Any,
    evaluator: str,
    deployment: str,
    rows: list[dict[str, Any]],
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                client,
                evaluator,
                deployment,
                index,
                row,
            ): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            item = future.result()
            items.append(item)
            print(
                f"evaluation {deployment}: {len(items)}/{len(rows)}",
                flush=True,
            )
    items.sort(key=lambda item: item["index"])
    return items, summarize_evaluation(items)


def hosted_sandbox(
    client: Any,
    sandbox_model: str,
    candidate: str,
) -> dict[str, Any]:
    payload = {
        "source": strip_code_fence(candidate),
        "cases": [
            {"APP_MODE": mode, "expected": expected}
            for mode, expected in SANDBOX_CASES
        ],
    }
    prompt = (
        "Use the code interpreter tool to test the supplied untrusted Python source "
        "inside the hosted tool container, never in the client process. Write it to "
        "candidate.py, execute it once per case with only APP_MODE and "
        "PYTHONIOENCODING=utf-8 in the child environment, a 5-second timeout, and "
        "capture stdout, stderr, and return code. Return only JSON with a cases array "
        "containing APP_MODE, expected, actual, returncode, stderr, and passed, plus "
        "all_passed. Payload:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    response = client.responses.create(
        model=sandbox_model,
        input=prompt,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        tool_choice="required",
        max_output_tokens=4_096,
    )
    text = response.output_text or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Hosted sandbox did not return JSON: {text!r}")
    result = json.loads(text[start : end + 1])
    result["execution_boundary"] = "Foundry hosted code-interpreter container"
    return result


def download_training_metrics(client: Any, completed: Any) -> dict[str, Any]:
    if not completed.result_files:
        return {"available": False}
    content = client.files.content(completed.result_files[0])
    csv_text = content.text
    path = OUTPUTS / "training-results.csv"
    path.write_text(csv_text, encoding="utf-8")
    rows = list(csv.DictReader(StringIO(csv_text)))
    numeric_rows = [row for row in rows if row.get("train_loss")]
    summary: dict[str, Any] = {
        "available": True,
        "result_file_id": completed.result_files[0],
        "sha256": sha256(path),
        "steps": len(numeric_rows),
    }
    if numeric_rows:
        first, final = numeric_rows[0], numeric_rows[-1]
        for key in (
            "train_loss",
            "train_mean_token_accuracy",
            "valid_loss",
            "valid_mean_token_accuracy",
            "full_valid_loss",
            "full_valid_mean_token_accuracy",
        ):
            values = [row.get(key, "").strip() for row in numeric_rows]
            present = [float(value) for value in values if value]
            if present:
                summary[f"initial_{key}"] = present[0]
                summary[f"final_{key}"] = present[-1]
                summary[f"minimum_{key}"] = min(present)
        summary["first_row"] = first
        summary["final_row"] = final
    write_json(OUTPUTS / "training-metrics.json", summary)
    return summary


def main() -> None:
    project_endpoint = require_env("FOUNDRY_PROJECT_ENDPOINT")
    teacher = require_env("FOUNDRY_TEACHER_DEPLOYMENT")
    student_base = require_env("FOUNDRY_STUDENT_DEPLOYMENT")
    evaluator = require_env("FOUNDRY_EVALUATOR_DEPLOYMENT")
    student_model = os.environ.get(
        "FOUNDRY_STUDENT_MODEL", STUDENT_MODEL_DEFAULT
    ).strip()
    sandbox_model = os.environ.get(
        "FOUNDRY_SANDBOX_MODEL", evaluator
    ).strip()
    generated_count = env_int("FOUNDRY_NUM_RECORDS", PARITY_GENERATED_COUNT)
    train_count = env_int("FOUNDRY_TRAIN_RECORDS", PARITY_TRAIN_COUNT)
    evaluation_count = env_int(
        "FOUNDRY_EVALUATION_RECORDS", PARITY_EVALUATION_COUNT
    )
    quality_threshold = env_float(
        "FOUNDRY_QUALITY_THRESHOLD", PARITY_QUALITY_THRESHOLD
    )
    file_timeout = env_int("FOUNDRY_FILE_TIMEOUT_SECONDS", 1_800)
    job_timeout = env_int("FOUNDRY_JOB_TIMEOUT_SECONDS", 86_400)
    poll_interval = env_int("FOUNDRY_POLL_INTERVAL_SECONDS", 30)
    eval_workers = env_int("FOUNDRY_EVALUATION_WORKERS", 4)
    generation_parallelism = env_int("FOUNDRY_GENERATION_PARALLELISM", 12)

    credential = AzureCliCredential()
    project = AIProjectClient(endpoint=project_endpoint, credential=credential)
    deployments = [dict(item) for item in project.deployments.list()]
    names = {str(item["name"]) for item in deployments}
    for role, deployment in {
        "teacher": teacher,
        "student base": student_base,
        "evaluator": evaluator,
        "sandbox": sandbox_model,
    }.items():
        if deployment not in names:
            raise RuntimeError(
                f"Configured {role} deployment {deployment!r} is unavailable."
            )
    client = project.get_openai_client()

    resume_output = os.environ.get("FOUNDRY_RESUME_OUTPUT", "").strip()
    if resume_output:
        train_path, validation_path, lineage, evaluation_rows = (
            load_completed_generation(Path(resume_output))
        )
    else:
        train_path, validation_path, lineage, evaluation_rows = (
            generate_parity_data(
                credential,
                client,
                teacher,
                generated_count,
                train_count,
                evaluation_count,
                quality_threshold,
                generation_parallelism,
            )
        )
    reuse_contract = {
        "project_endpoint": project_endpoint,
        "teacher_deployment": teacher,
        "student_model": student_model,
        "training_sha256": lineage["training_sha256"],
        "validation_sha256": lineage["validation_evaluation_sha256"],
        "generated_count": generated_count,
        "training_count": train_count,
        "validation_count": evaluation_count,
        "n_epochs": 1,
        "learning_rate_multiplier": 1.3,
        "batch_size": 1,
    }
    write_json(OUTPUTS / "reuse-contract.json", reuse_contract)
    reusable = matching_completed_job(
        project_endpoint,
        teacher,
        student_model,
        lineage["training_sha256"],
        lineage["validation_evaluation_sha256"],
    )
    if reusable:
        completed_dict = reusable["job"]
        completed = client.fine_tuning.jobs.retrieve(completed_dict["id"])
        job_action = f"reused exact job from {reusable['directory']}"
    else:
        train_file = upload(client, train_path, file_timeout, poll_interval)
        validation_file = upload(
            client, validation_path, file_timeout, poll_interval
        )
        if sha256(train_path) != lineage["training_sha256"]:
            raise RuntimeError("Training file changed after lineage was recorded.")
        if sha256(validation_path) != lineage["validation_evaluation_sha256"]:
            raise RuntimeError("Validation file changed after lineage was recorded.")
        job = client.fine_tuning.jobs.create(
            model=student_model,
            training_file=train_file.id,
            validation_file=validation_file.id,
            method={
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "n_epochs": 1,
                        "learning_rate_multiplier": 1.3,
                        "batch_size": 1,
                    }
                },
            },
            suffix="nl2py-parity",
            extra_body={"trainingType": "globalStandard"},
        )
        write_json(OUTPUTS / "submission.json", job.model_dump())
        completed = wait_for_job(client, job.id, job_timeout, poll_interval)
        job_action = "submitted new exact-contract job"
    write_json(OUTPUTS / "terminal-job.json", completed.model_dump())
    if str(completed.status).casefold() != "succeeded" or not completed.fine_tuned_model:
        raise RuntimeError(
            f"Fine-tuning job ended as {completed.status}: {completed.error}"
        )

    training_metrics = download_training_metrics(client, completed)
    fine_tuned_deployment, deployment_action = deploy_model(
        project,
        project_endpoint,
        completed.fine_tuned_model,
        completed.id,
    )

    evaluation_details: dict[str, Any] = {}
    evaluation_summary: dict[str, Any] = {}
    for role, deployment in (
        ("teacher", teacher),
        ("student_base", student_base),
        ("student_fine_tuned", fine_tuned_deployment),
    ):
        items, summary = evaluate_deployment(
            client,
            evaluator,
            deployment,
            evaluation_rows,
            eval_workers,
        )
        evaluation_details[role] = items
        evaluation_summary[role] = summary
        write_json(OUTPUTS / f"evaluation-{role}.json", items)

    base = evaluation_summary["student_base"]
    fine_tuned = evaluation_summary["student_fine_tuned"]
    gain = {
        "combined_absolute_points": fine_tuned["combined"] - base["combined"],
        "combined_relative_percent": (
            (fine_tuned["combined"] - base["combined"]) / base["combined"] * 100
        ),
        "correctness_absolute_points": (
            fine_tuned["correctness"] - base["correctness"]
        ),
        "conciseness_absolute_points": (
            fine_tuned["conciseness"] - base["conciseness"]
        ),
        "mean_code_length_change_percent": (
            (
                fine_tuned["mean_candidate_characters"]
                - base["mean_candidate_characters"]
            )
            / base["mean_candidate_characters"]
            * 100
        ),
        "formula": (
            "(fine_tuned_combined - base_combined) / base_combined * 100, "
            "computed from unrounded 84-row means"
        ),
    }

    sandbox: dict[str, Any] = {}
    for role, deployment in (
        ("teacher", teacher),
        ("student_base", student_base),
        ("student_fine_tuned", fine_tuned_deployment),
    ):
        candidate = generate_candidate(client, deployment, SANDBOX_INSTRUCTION)
        sandbox[role] = {
            "deployment": deployment,
            "candidate": candidate,
            "result": hosted_sandbox(client, sandbox_model, candidate),
        }
    write_json(OUTPUTS / "sandbox-evaluation.json", sandbox)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_endpoint": project_endpoint,
        "models": {
            "teacher_deployment": teacher,
            "evaluator_deployment": evaluator,
            "student_base_deployment": student_base,
            "student_model": student_model,
            "fine_tuned_model": completed.fine_tuned_model,
            "fine_tuned_deployment": fine_tuned_deployment,
            "sandbox_model": sandbox_model,
        },
        "lineage": lineage,
        "job": {
            "action": job_action,
            "id": completed.id,
            "status": completed.status,
            "created_at": completed.created_at,
            "finished_at": completed.finished_at,
            "fine_tuned_model": completed.fine_tuned_model,
            "result_files": completed.result_files,
        },
        "deployment_action": deployment_action,
        "training_metrics": training_metrics,
        "primary_evaluation": evaluation_summary,
        "gain": gain,
        "sandbox": {
            role: {
                "deployment": result["deployment"],
                "all_passed": result["result"].get("all_passed"),
                "execution_boundary": result["result"].get("execution_boundary"),
                "cases": result["result"].get("cases"),
            }
            for role, result in sandbox.items()
        },
        "discrepancies_preserved": [
            "The original checked-in run is 11 train / 1 validation.",
            "The original README describes about 85 train / 4 validation from 100 requested.",
            "The historical winner documents 2,000 requested, about 1,500 training rows, and 84 held-out rows.",
            "The original split code used 5 percent, which does not mathematically yield both exactly 1,500 train and 84 held-out rows.",
            "The original prose defines Pass@8 on combined score, while its implementation counted correctness only; this report records both.",
            "The +6.4 percent historical claim cannot be reproduced from rounded 8.6 and 9.2 scores (which imply about 7.0 percent); this run computes relative gain from unrounded means.",
        ],
        "live_fixes": [
            "The first 2,000-row attempt used the original max_parallel_requests=2 and exceeded the one-hour Entra token lifetime before generation completed.",
            "Generation transport parallelism was raised to complete the unchanged recipe within one token lifetime; prompts, temperatures, gates, scale, split, and training method were unchanged.",
            "The optional Data Designer schema processor was removed because data-designer 0.9.3 rejected it during shutdown; the same accepted rows are converted directly to the identical chat messages schema before hashing and upload.",
            "Data Designer 0.9.3 dropped one of two internal row groups when asked for 2,000 rows in one create call; parity generation now uses two explicit 1,000-row calls and concatenates all 2,000 live rows before filtering.",
            "The original notebook's UTF-8 PythonValidator patch was restored after a generated Unicode identifier triggered the known Windows cp1252 failure.",
            "A concurrency-24 attempt triggered a transient East US 2 provider-error burst; transport concurrency was reduced to 12 and every 1,000-row batch/retry now receives a fresh Entra token.",
            "The completed 2,000-row generation was resumed by exact verified file hashes after upload hit the project file quota; it was not regenerated or replaced.",
        ],
        "output_directory": str(OUTPUTS),
    }
    write_json(OUTPUTS / "parity-report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
