import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENDPOINT = (
    "https://eastus2-prakharg-demo-2026.services.ai.azure.com/api/projects/"
    "eastus2-prakharg-demo-2026"
)

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

CONSTANTS = f'''import os
os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "{PROJECT_ENDPOINT}"

BASE_MODEL = "o4-mini-2025-04-16"
BASELINE_DEPLOYMENT = "gpt-5.4-mini"
GRADER_MODEL = "o3-mini"
print("Project endpoint:", os.environ["AZURE_AI_PROJECT_ENDPOINT"])
print("RFT base model:", BASE_MODEL)
print("Baseline deployment:", BASELINE_DEPLOYMENT)
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

EVAL_DATA = '''from datasets import load_dataset
from scripts.dataset_utils import save_dataset_in_eval_format

test_dataset = load_dataset("predibase/countdown", split="test")
eval_ready_path = "data/countdown_eval_100.jsonl"
save_dataset_in_eval_format(test_dataset, eval_ready_path, max_records=100)'''

UPLOAD_EVAL = '''from scripts.io_utils import upload_file

eval_file_id = await upload_file(
    file_name="countdown_evals_100.jsonl",
    file_path=eval_ready_path,
    purpose="evals",
)
print("Eval file ID:", eval_file_id)'''

TRAIN_DATA = '''from datasets import load_dataset
from scripts.dataset_utils import save_dataset_as_jsonl, convert_to_rft_dataset

train_raw_path = "data/countdown_train_raw.jsonl"
valid_raw_path = "data/countdown_valid_raw.jsonl"
train_rft_path = "data/countdown_train_100.jsonl"
valid_rft_path = "data/countdown_valid_50.jsonl"

dataset = load_dataset("predibase/countdown", split="train")
train_split = dataset.select(range(500))
valid_split = dataset.select(range(500, 1000))
save_dataset_as_jsonl(train_split, train_raw_path)
save_dataset_as_jsonl(valid_split, valid_raw_path)
convert_to_rft_dataset(train_raw_path, train_rft_path, instruction, max_records=100)
convert_to_rft_dataset(valid_raw_path, valid_rft_path, instruction, max_records=50)'''

UPLOAD_TRAIN = '''from scripts.io_utils import upload_file

train_file_id = await upload_file(
    "countdown_train_100.jsonl", train_rft_path, purpose="fine-tune"
)
valid_file_id = await upload_file(
    "countdown_valid_50.jsonl", valid_rft_path, purpose="fine-tune"
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
fine_tuned_model = finetune_job.fine_tuned_model
print("Fine-tuned model:", fine_tuned_model)

from scripts.rft_workflow import ensure_fine_tuned_deployment
fine_tuned_deployment = ensure_fine_tuned_deployment(
    NOTEBOOK_NAME,
    fine_tuned_model,
    DEPLOYMENT_NAME,
)'''

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
        17: VALIDATION_CASES + '''
from scripts.rft_workflow import solve_and_validate

baseline_validation_results = solve_and_validate(
    NOTEBOOK_NAME,
    BASELINE_DEPLOYMENT,
    instruction,
    response_schema,
    validation_cases,
)
print("Baseline validation complete")''',
        19: '''baseline_passed = sum(
    result["valid"] for result in baseline_validation_results
)
print("Baseline exact solutions:", baseline_passed, "/", len(validation_cases))''',
        20: '''print("Baseline grader model:", GRADER_MODEL)''',
        31: VALIDATE,
        32: '''fine_tuned_passed = sum(
    result["valid"] for result in fine_tuned_validation_results
)
print("Fine-tuned exact solutions:", fine_tuned_passed, "/", len(validation_cases))''',
        33: '''if fine_tuned_passed == 0:
    raise RuntimeError("Fine-tuned model produced no exact Countdown solutions")
print("Countdown outcome validated")''',
    }


def python_eval_cells(name: str) -> dict[int, str]:
    return {
        16: VALIDATION_CASES + '''
from scripts.rft_workflow import solve_and_validate

baseline_validation_results = solve_and_validate(
    NOTEBOOK_NAME,
    BASELINE_DEPLOYMENT,
    instruction,
    response_schema,
    validation_cases,
)
print("Baseline validation complete")''',
        18: '''baseline_passed = sum(
    result["valid"] for result in baseline_validation_results
)
print("Baseline exact solutions:", baseline_passed, "/", len(validation_cases))''',
        19: '''print("RFT grader type:", custom_grader["type"])''',
        30: VALIDATE,
        31: '''fine_tuned_passed = sum(
    result["valid"] for result in fine_tuned_validation_results
)
print("Fine-tuned exact solutions:", fine_tuned_passed, "/", len(validation_cases))''',
        32: '''if fine_tuned_passed == 0:
    raise RuntimeError("Fine-tuned model produced no exact Countdown solutions")
print("Countdown outcome validated")''',
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

    suffix = "countdown-python-grader" if python_grader else "countdown-model-grader"
    deployment_name = (
        "countdown-python-grader-ft"
        if python_grader
        else "countdown-model-grader-ft"
    )
    replacements[name_cell] += (
        f'\nNOTEBOOK_NAME = "{path.stem}"'
        f'\nJOB_SUFFIX = "{suffix}"'
        f'\nDEPLOYMENT_NAME = "{deployment_name}"'
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
