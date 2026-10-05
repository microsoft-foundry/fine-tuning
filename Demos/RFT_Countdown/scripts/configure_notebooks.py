import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INSTRUCTION = '''instruction = (
    "You are an expert in arithmetic problem solving. Given a target number and a list of numbers, "
    "combine every number exactly once using addition (+), subtraction (-), multiplication (*), "
    "or division (/) to reach the target. Use parentheses to control order of operations. "
    "Return only JSON with string fields expression and result. "
    "If an exact solution is impossible, return the closest valid result."
)'''

RESPONSE_SCHEMA = '''response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_expression",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["expression", "result"],
            "properties": {
                "expression": {"type": "string"},
                "result": {"type": "string"}
            },
            "additionalProperties": False
        }
    }
}'''

DATA_SAMPLE = '''from datasets import load_dataset
import random

dataset = load_dataset("predibase/countdown", split="train")
random.seed(42)
print("Countdown dataset samples")
for index in random.sample(range(len(dataset)), 5):
    item = dataset[index]
    print(f"Target: {item['target']} Numbers: {item['nums']}")'''

CONSTANTS = '''import os

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
BASE_MODEL = os.getenv("FOUNDRY_BASE_MODEL", "qwen3.6-35b-a3b")
GRADER_MODEL = "o3-mini"
print("Project endpoint:", PROJECT_ENDPOINT)
print("RFT base model:", BASE_MODEL)
print("Training type: globalStandard")
print("RFT grader model:", GRADER_MODEL)'''

MODEL_GRADER = '''custom_grader = {
    "name": "countdown_model_grader",
    "type": "score_model",
    "model": GRADER_MODEL,
    "input": [
        {
            "role": "developer",
            "content": (
                "Grade a Countdown answer. Verify that the JSON output has expression and result, "
                "that the expression uses each input number exactly once, uses only +, -, *, / and "
                "parentheses, that the reported result equals the expression, and that it reaches "
                "the target. Score 5 for an exact valid solution, 4 when off by 1, 3 when off by "
                "2-5, 2 when farther away but otherwise valid, 1 for a result mismatch, and 0 for "
                "invalid syntax or number usage. Return only a score and brief reasoning."
            )
        },
        {
            "role": "user",
            "content": (
                "target={{item.target}} numbers={{item.nums}} "
                "output={{sample.output_text}}"
            )
        }
    ],
    "pass_threshold": 5,
    "range": [0.0, 5.0]
}'''

PYTHON_GRADER = '''from scripts.eval_utils import python_grader_source

custom_grader = {
    "type": "python",
    "source": python_grader_source,
}'''

EVAL_DATA = '''from pathlib import Path

eval_ready_path = "data/countdown_eval_100.jsonl"
if not Path(eval_ready_path).is_file():
    raise FileNotFoundError(eval_ready_path)
print("Preserved evaluation file (not executed):", eval_ready_path)'''

UPLOAD_EVAL = '''print(
    "Evaluation upload skipped: this run is training-only and preserves the "
    "original evaluation file unchanged."
)'''

TRAIN_DATA = '''import hashlib
from pathlib import Path

train_rft_path = "data/countdown_train_100.jsonl"
valid_rft_path = "data/countdown_valid_50.jsonl"

expected_files = {
    train_rft_path: {
        "records": 100,
        "bytes": 54602,
        "sha256": "2554d6a9e9aaebcd67ddbd12ad8df361fa28f769a3a2c5d271db09bf54ba23ad",
    },
    valid_rft_path: {
        "records": 50,
        "bytes": 27304,
        "sha256": "ea0326d3c53c3f5da9dc2c5836fdafd017a9168c55b7dc38b6132ed63b7cf0aa",
    },
    eval_ready_path: {
        "records": 100,
        "bytes": 5535,
        "sha256": "1f7fe8ef30a0f70fcc5db112cf385916a2a200fad702fcd0711d81818ef86413",
    },
}
for path, expected in expected_files.items():
    content = Path(path).read_bytes()
    actual = {
        "records": sum(1 for line in content.splitlines() if line.strip()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if actual != expected:
        raise RuntimeError(f"Preserved input changed: {path}: {actual} != {expected}")
    print(path, actual)'''

UPLOAD_TRAIN = '''from scripts.io_utils import upload_file

train_file_id = await upload_file(
    "countdown_train_100-qwen36.jsonl", train_rft_path, purpose="fine-tune"
)
valid_file_id = await upload_file(
    "countdown_valid_50-qwen36.jsonl", valid_rft_path, purpose="fine-tune"
)
print("Training file ID:", train_file_id)
print("Validation file ID:", valid_file_id)'''

SUBMIT_JOB = '''from scripts.rft_workflow import create_rft_job, wait_for_fine_tune

method = {
    "type": "reinforcement",
    "reinforcement": {
        "hyperparameters": {
            "eval_interval": 5,
            "eval_samples": 2,
            "reasoning_effort": "medium",
            "n_epochs": 1,
            "batch_size": 4,
            "learning_rate_multiplier": 2,
        },
        "grader": custom_grader,
        "response_format": response_schema,
    },
}

finetune_job = create_rft_job(
    notebook=NOTEBOOK_NAME,
    model=BASE_MODEL,
    training_file=train_file_id,
    validation_file=valid_file_id,
    method=method,
    suffix=JOB_SUFFIX,
)
finetune_job = wait_for_fine_tune(NOTEBOOK_NAME, finetune_job.id)
from scripts.rft_workflow import collect_training_evidence

job_summary = collect_training_evidence(
    NOTEBOOK_NAME,
    finetune_job,
    train_rft_path,
    valid_rft_path,
    eval_ready_path,
)
print("Terminal status:", finetune_job.status)
print("Validation reward initial:", job_summary["rewards"]["validation_reward_initial"])
print("Validation reward final:", job_summary["rewards"]["validation_reward_final"])
print("Validation reward max:", job_summary["rewards"]["validation_reward_max"])
print(job_summary["conclusion"])'''

VALIDATION_CASES = '''validation_cases = [
    {"target": 24, "nums": [1, 2, 3, 4]},
    {"target": 19, "nums": [2, 3, 4, 5]},
    {"target": 50, "nums": [6, 7, 8, 9]},
]'''

VALIDATE = '''from scripts.rft_workflow import solve_and_validate

fine_tuned_validation_results = solve_and_validate(
    NOTEBOOK_NAME,
    fine_tuned_deployment,
    instruction,
    response_schema,
    validation_cases,
)
print(
    "Exact countdown solutions:",
    sum(result["valid"] for result in fine_tuned_validation_results),
    "/",
    len(fine_tuned_validation_results),
)'''


def model_eval_cells(name: str) -> dict[int, str]:
    return {
        17: '''print("Baseline inference skipped by design.")''',
        19: '''print("No deployment-based baseline metrics were collected.")''',
        20: '''print("Model grader semantics preserved; grader model:", GRADER_MODEL)''',
        31: '''print("Fine-tuned deployment and inference skipped by design.")''',
        32: '''print("Use the validation reward curve in job_summary for conclusions.")''',
        33: '''print("Training-only Countdown run complete.")''',
    }


def python_eval_cells(name: str) -> dict[int, str]:
    return {
        16: '''print("Baseline inference skipped by design.")''',
        18: '''print("No deployment-based baseline metrics were collected.")''',
        19: '''print("Python grader semantics preserved; grader type:", custom_grader["type"])''',
        30: '''print("Fine-tuned deployment and inference skipped by design.")''',
        31: '''print("Use the validation reward curve in job_summary for conclusions.")''',
        32: '''print("Training-only Countdown run complete.")''',
    }


def configure(path: Path, python_grader: bool) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    replacements = {
        2: DATA_SAMPLE,
        5: INSTRUCTION,
        8: RESPONSE_SCHEMA,
        10: PYTHON_GRADER if python_grader else CONSTANTS,
        13 if python_grader else 14: EVAL_DATA,
        15 if python_grader else 16: UPLOAD_EVAL,
        23 if python_grader else 24: TRAIN_DATA,
        25 if python_grader else 26: UPLOAD_TRAIN,
        27 if python_grader else 28: SUBMIT_JOB,
    }
    if python_grader:
        replacements[10] = CONSTANTS + "\n\n" + PYTHON_GRADER
        replacements.update(python_eval_cells(path.stem))
        name_cell = 10
    else:
        replacements[11] = MODEL_GRADER
        replacements.update(model_eval_cells(path.stem))
        name_cell = 10

    suffix = (
        "qwen36-countdown-python-grader"
        if python_grader
        else "qwen36-countdown-model-grader"
    )
    replacements[name_cell] += (
        f'\nNOTEBOOK_NAME = "{path.stem}"'
        f'\nJOB_SUFFIX = "{suffix}"'
    )

    for index, source in replacements.items():
        cells[index]["cell_type"] = "code"
        cells[index]["source"] = [line + "\n" for line in source.splitlines()]
        cells[index]["outputs"] = []
        cells[index]["execution_count"] = None

    retained_code = set(replacements)
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and index not in retained_code:
            cell["source"] = []
            cell["outputs"] = []
            cell["execution_count"] = None

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace("`o4-mini`", "`qwen3.6-35b-a3b`")
        source = source.replace("o4-mini", "qwen3.6-35b-a3b")
        source = source.replace(
            "Evaluating Base Models",
            "Preserving Evaluation Inputs (Execution Skipped)",
        )
        source = source.replace(
            "Fine-Tuned Model Evaluation",
            "Training-Only Reward Analysis",
        )
        source = source.replace(
            "Deployment & Inference",
            "Terminal Training Evidence",
        )
        if "approaches o3" in source or "inference cost" in source:
            source = (
                "## Training-only conclusion\n\n"
                "No deployment or inference is performed. Conclusions are limited "
                "to the validation reward curve and its initial, final, and maximum "
                "values from the terminal training result."
            )
        cell["source"] = [line + "\n" for line in source.splitlines()]

    notebook.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "RFT Countdown Python 3.12",
        "language": "python",
        "name": "python3",
    }
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


requested = set(sys.argv[1:])
if not requested or "demo.ipynb" in requested:
    configure(ROOT / "demo.ipynb", python_grader=False)
if not requested or "demo_with_python_grader.ipynb" in requested:
    configure(ROOT / "demo_with_python_grader.ipynb", python_grader=True)
