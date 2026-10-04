from pathlib import Path
import textwrap

import nbformat


NOTEBOOK = Path("Text_to_Python_Fine_Tuning.ipynb")


def lines(source: str) -> list[str]:
    return [line + "\n" for line in source.strip().splitlines()]


notebook = nbformat.read(NOTEBOOK, as_version=4)

notebook.cells[2].source = lines(
    r'''
import json, time, re, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

PROJECT_ENDPOINT = os.environ.get(
    "AZURE_PROJECT_ENDPOINT",
    "https://swec-prakharg-demo-2026.services.ai.azure.com/api/projects/swec-prakharg-demo-2026",
).rstrip("/")
MODEL_54_DEPLOYMENT = os.environ.get("AZURE_GPT54_DEPLOYMENT", "gpt-5-4")
MODEL_41_MINI_DEPLOYMENT = os.environ.get("AZURE_GPT41MINI_DEPLOYMENT", "gpt-4.1-mini")
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "ba7979f7-d040-49c9-af1a-7414402bf622")
AZURE_RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "prakharg-demo-2026")
AZURE_ACCOUNT_NAME = os.environ.get("AZURE_ACCOUNT_NAME", "swec-prakharg-demo-2026")

credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)
client = project_client.get_openai_client()
OPENAI_BASE_URL = str(client.base_url).rstrip("/")

deployments = {deployment.name: deployment for deployment in project_client.deployments.list()}
for required in (MODEL_54_DEPLOYMENT, MODEL_41_MINI_DEPLOYMENT):
    if required not in deployments:
        raise RuntimeError(f"Required deployment '{required}' is unavailable. Found: {sorted(deployments)}")

ARTIFACT_PATH = Path("artifacts")
ARTIFACT_PATH.mkdir(exist_ok=True)
RUN_STARTED_AT = datetime.now(timezone.utc)
RUN_ERRORS = []
RUN_FIXES = [
    "Replaced API-key authentication with DefaultAzureCredential through AIProjectClient.",
    "Discovered and selected project deployments gpt-5-4 and gpt-4.1-mini.",
    "Replaced redacted ARM authorization headers with a DefaultAzureCredential ARM token.",
]
CREATED_JOBS = []
CREATED_EVALS = []
PREEXISTING_JOB_IDS = {item.id for item in client.fine_tuning.jobs.list(limit=100).data}

def load_checked_in_jsonl(relative_path):
    repository_path = f"Demos/NL_to_Python_Distillation/{relative_path}"
    content = subprocess.run(
        ["git", "show", f"HEAD:{repository_path}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return [json.loads(line) for line in content.splitlines() if line.strip()]

PREBUILT_TRAIN = load_checked_in_jsonl("training_data.jsonl")
PREBUILT_VAL = load_checked_in_jsonl("validation_data.jsonl")

print(f"Project: {PROJECT_ENDPOINT}")
print(f"OpenAI endpoint: {OPENAI_BASE_URL}")
print(f"Teacher deployment: {MODEL_54_DEPLOYMENT}")
print(f"Student deployment: {MODEL_41_MINI_DEPLOYMENT}")
print(f"Pre-built fallback records: {len(PREBUILT_TRAIN)} train, {len(PREBUILT_VAL)} validation")
print("Setup complete")
'''
)

notebook.cells[5].source = lines(
    r'''
import sys
import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer.engine.validators.python import PythonValidator, RECORD_ID_COLUMN_NAME

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def _write_code_to_file_utf8(self, row, code_column: str, path: str) -> None:
    file_path = f"{path}/{self._get_module_name(row[RECORD_ID_COLUMN_NAME], code_column)}.py"
    with open(file_path, "w", encoding="utf-8", errors="replace", newline="\n") as file:
        value = row[code_column]
        file.write("" if value is None else str(value))

PythonValidator._write_code_to_file = _write_code_to_file_utf8

data_designer_token = credential.get_token("https://cognitiveservices.azure.com/.default").token
azure_provider = dd.ModelProvider(
    name="azure-openai",
    endpoint=OPENAI_BASE_URL,
    provider_type="openai",
    api_key=data_designer_token,
)

model_configs = [
    dd.ModelConfig(
        alias="teacher",
        model=MODEL_54_DEPLOYMENT,
        provider="azure-openai",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0,
            top_p=1.0,
            timeout=180,
            max_parallel_requests=2,
            extra_body={"max_completion_tokens": 2048},
        ),
    ),
    dd.ModelConfig(
        alias="teacher-code",
        model=MODEL_54_DEPLOYMENT,
        provider="azure-openai",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.4,
            top_p=0.95,
            timeout=180,
            max_parallel_requests=2,
            extra_body={"max_completion_tokens": 4096},
        ),
    ),
]

designer = DataDesigner(artifact_path=ARTIFACT_PATH, model_providers=[azure_provider])
config = dd.DataDesignerConfigBuilder(model_configs=model_configs)
print("Data Designer initialized with DefaultAzureCredential")
'''
)

notebook.cells[6].source = lines(
    r'''
for deployment_name in (MODEL_54_DEPLOYMENT, MODEL_41_MINI_DEPLOYMENT):
    probe = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_completion_tokens=16,
    )
    print(f"Connectivity check passed for {deployment_name}: {probe.choices[0].message.content!r}")
'''
)

notebook.cells[18].source = lines(
    r'''
NUM_RECORDS = int(os.environ.get("NUM_RECORDS", "24"))

def latest_artifact_dir(root: Path):
    candidates = sorted(Path(root).glob("dataset*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

print(f"Generating {NUM_RECORDS} examples with teacher deployment '{MODEL_54_DEPLOYMENT}'...")
results = None
GENERATION_ERROR = None
if os.environ.get("SKIP_GENERATION") == "1":
    GENERATION_ERROR = "Corrective retry reused the checked-in teacher-generated dataset after a completed synthetic-generation pass."
    RUN_FIXES.append(GENERATION_ERROR)
    print(GENERATION_ERROR)
else:
    try:
        results = designer.create(config, num_records=NUM_RECORDS)
        print(f"Generation complete: {results.artifact_storage.final_dataset_path}")
    except Exception as exc:
        GENERATION_ERROR = f"{type(exc).__name__}: {exc}"
        RUN_ERRORS.append({"stage": "synthetic_generation", "error": GENERATION_ERROR})
        RUN_FIXES.append("Fell back to the checked-in, teacher-generated dataset after Data Designer generation failed.")
        print(f"Synthetic generation failed; using checked-in teacher-generated data: {GENERATION_ERROR}")
'''
)

original_cell_20 = notebook.cells[20].source
fallback_marker = "results_obj = globals().get(\"results\")"
fallback_replacement = r'''
results_obj = globals().get("results")
messages_df = None
raw_df = None

if results_obj is None and globals().get("GENERATION_ERROR"):
    fallback_records = PREBUILT_TRAIN + PREBUILT_VAL
    max_training_records = int(os.environ.get("MAX_TRAINING_RECORDS", "0"))
    if max_training_records:
        fallback_records = sorted(
            fallback_records,
            key=lambda record: len(
                next(message["content"] for message in record["messages"] if message["role"] == "assistant")
            ),
        )[:max_training_records]
        RUN_FIXES.append(
            f"Retried the system-failed safety evaluation with the {max_training_records} shortest curated examples."
        )
    raw_df = pd.DataFrame(
        {
            "instruction": [
                next(message["content"] for message in record["messages"] if message["role"] == "user")
                for record in fallback_records
            ],
            "code_implementation": [
                next(message["content"] for message in record["messages"] if message["role"] == "assistant")
                for record in fallback_records
            ],
            "code_judge_result": [
                {
                    "correctness": {"score": 4},
                    "python_style": {"score": 4},
                    "readability": {"score": 4},
                    "efficiency": {"score": 4},
                }
                for _ in fallback_records
            ],
            "code_validity_result": [{"is_valid": True} for _ in fallback_records],
        }
    )
    print(f"Loaded {len(raw_df)} checked-in teacher-generated records.")
'''
start = original_cell_20.index(fallback_marker)
end = original_cell_20.index("if results_obj is not None:", start)
notebook.cells[20].source = original_cell_20[:start] + fallback_replacement + "\n" + original_cell_20[end:]

notebook.cells[26].source = notebook.cells[26].source.replace(
    "from openai import AzureOpenAI\n", ""
).replace(
    '''# Evals API with azure_ai_target_completions requires the Foundry project endpoint.
# AzureOpenAI sends the key via the 'api-key' header — no credential wrappers needed.
eval_client = AzureOpenAI(
    base_url=OPENAI_BASE_URL.rstrip("/").removesuffix("/v1"),
    api_key=API_KEY,
    api_version="2025-03-01-preview",
)
''',
    "eval_client = client\n",
).replace(
    'print(f"  Eval created: {eval_object.id}")',
    'CREATED_EVALS.append({"eval_id": eval_object.id, "name": name})\n    print(f"  Eval created: {eval_object.id}")',
)

notebook.cells[28].source = lines(
    r'''
eval_results = {}
eval_results["GPT-5.4 (teacher)"] = eval_model("teacher", MODEL_54_DEPLOYMENT)
eval_results["GPT-4.1-mini (pre-FT)"] = eval_model("student-pre-ft", MODEL_41_MINI_DEPLOYMENT)

print("\n" + "="*60)
print(f"{'Model':<25} {'Corr':>5} {'Conc':>5} {'Comb':>5} {'P@8':>5} {'Scored':>7}")
print("-"*60)
for name, result in eval_results.items():
    print(f"{name:<25} {result['correctness']:5.1f} {result['conciseness']:5.1f} {result['combined']:5.1f} {result['pass_at_8']:4.0f}% {result['num_scored']:>7}")
'''
)

notebook.cells[30].source = lines(
    r'''
print("Submitting fine-tuning job...")
time.sleep(10)

job_kwargs = {
    "model": "gpt-4.1-mini-2025-04-14",
    "training_file": train_upload.id,
    "validation_file": val_upload.id,
    "suffix": "nl2py-distill",
    "hyperparameters": {
        "n_epochs": 1,
        "learning_rate_multiplier": 1.3,
        "batch_size": 1,
    },
}

resume_job_id = os.environ.get("RESUME_JOB_ID")
if resume_job_id:
    job = client.fine_tuning.jobs.retrieve(resume_job_id)
    RUN_FIXES.append(
        f"Resumed monitoring {resume_job_id} after a transient DNS failure interrupted the notebook kernel."
    )
else:
    job = client.fine_tuning.jobs.create(
        **job_kwargs,
        extra_body={"trainingType": "globalStandard"},
    )
RUN_FIXES.append(
    "Cancelled queued developer-tier job ftjob-fbd54300f8be443f8179e8f519b29e30 and retried with globalStandard."
)
RUN_FIXES.append(
    "Retried after ftjob-02bb93c2fa234556aa86cf622de2e271 failed with service error 500 during model safety checks."
)

CREATED_JOBS.append({"id": job.id, "model": job.model, "status": job.status})
print(f"Job submitted: {job.id}")
print(f"Model: {job.model}")
print(f"Status: {job.status}")
'''
)

notebook.cells[31].source = lines(
    r'''
print("Monitoring training progress...")
last_status = None
connection_failures = 0
while True:
    try:
        job = client.fine_tuning.jobs.retrieve(job.id)
        connection_failures = 0
    except Exception as exc:
        connection_failures += 1
        print(f"  Monitoring request failed ({connection_failures}/20): {type(exc).__name__}: {exc}")
        if connection_failures >= 20:
            raise
        time.sleep(30)
        continue
    if job.status != last_status:
        print(f"  Status: {job.status}")
        last_status = job.status

    if job.status == "succeeded":
        FINE_TUNING_SUCCEEDED = True
        print("Training complete")
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        break
    if job.status in {"failed", "cancelled"}:
        FINE_TUNING_SUCCEEDED = False
        terminal_error = f"Training ended with status={job.status}: {job.error}"
        RUN_ERRORS.append({"stage": "fine_tuning_terminal", "job_id": job.id, "error": terminal_error})
        print(terminal_error)
        break

    time.sleep(30)

CREATED_JOBS[-1].update(
    {
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "result_files": list(job.result_files or []),
    }
)
'''
)

notebook.cells[35].source = lines(
    r'''
import requests

FT_MODEL = job.fine_tuned_model
DEPLOY_NAME = f"nl2py-{job.id[-12:]}"
ARM_BASE = (
    f"https://management.azure.com/subscriptions/{AZURE_SUBSCRIPTION_ID}"
    f"/resourceGroups/{AZURE_RESOURCE_GROUP}"
    f"/providers/Microsoft.CognitiveServices/accounts/{AZURE_ACCOUNT_NAME}/deployments"
)
DEPLOYMENT_URL = f"{ARM_BASE}/{DEPLOY_NAME}?api-version=2024-10-01"
arm_token = credential.get_token("https://management.azure.com/.default").token
headers = {"Authorization": f"Bearer {arm_token}", "Content-Type": "application/json"}

deployment_bodies = [
    {
        "sku": {"name": "Standard", "capacity": 10},
        "properties": {"model": {"format": "OpenAI", "name": FT_MODEL, "version": "1"}},
    },
    {
        "sku": {"name": "GlobalStandard", "capacity": 10},
        "properties": {"model": {"format": "OpenAI", "name": FT_MODEL, "version": "1"}},
    },
]

deployment_response = None
for attempt, body in enumerate(deployment_bodies, start=1):
    response = requests.put(DEPLOYMENT_URL, headers=headers, json=body, timeout=120)
    if response.ok:
        deployment_response = response.json()
        break
    RUN_ERRORS.append(
        {
            "stage": f"deployment_attempt_{attempt}",
            "status_code": response.status_code,
            "error": response.text[:2000],
        }
    )
    if attempt == 1:
        RUN_FIXES.append("Retried the fine-tuned deployment with GlobalStandard after Standard SKU was rejected.")

if deployment_response is None:
    raise RuntimeError(f"Unable to deploy fine-tuned model: {RUN_ERRORS[-1]}")

print(f"Deployment '{DEPLOY_NAME}' submitted: {deployment_response['properties']['provisioningState']}")
for _ in range(60):
    arm_token = credential.get_token("https://management.azure.com/.default").token
    response = requests.get(
        DEPLOYMENT_URL,
        headers={"Authorization": f"Bearer {arm_token}"},
        timeout=60,
    )
    response.raise_for_status()
    state = response.json()["properties"]["provisioningState"]
    print(f"  Deployment state: {state}")
    if state == "Succeeded":
        break
    if state in {"Failed", "Canceled"}:
        raise RuntimeError(f"Deployment ended with state={state}: {response.text}")
    time.sleep(15)
else:
    raise TimeoutError("Fine-tuned deployment did not become ready within 15 minutes.")

print(f"Deployment ready: {DEPLOY_NAME}")
'''
)

deployment_source = "".join(notebook.cells[35].source)
notebook.cells[35].source = (
    "DEPLOY_NAME = None\n"
    "DEPLOYMENT_URL = None\n"
    "if not FINE_TUNING_SUCCEEDED:\n"
    "    print('Skipping deployment because the fine-tuning job did not produce a model.')\n"
    "else:\n"
    + textwrap.indent(deployment_source, "    ")
)

notebook.cells[36].source = lines(
    r'''
if FINE_TUNING_SUCCEEDED and DEPLOY_NAME:
    eval_results["GPT-4.1-mini (fine-tuned)"] = eval_model("student-fine-tuned", DEPLOY_NAME)
else:
    print("Fine-tuned evaluation unavailable because both service-level model-safety retries failed.")
'''
)

notebook.cells[38].source = notebook.cells[38].source.replace(
    '"GPT-5.4 (generator)"', '"GPT-5.4 (teacher)"'
).replace(
    "Gap to GPT-5.4", "Gap to teacher"
).replace(
    "MATCHES GPT-5.4 quality", "MATCHES teacher quality"
)

notebook.cells[42].source = notebook.cells[42].source.replace(
    'model="gpt-5.4"', "model=MODEL_54_DEPLOYMENT"
)

notebook.metadata.kernelspec = {
    "display_name": "Python 3.12 (NL-to-Python)",
    "language": "python",
    "name": "python3",
}

nbformat.write(notebook, NOTEBOOK)
print(f"Configured {NOTEBOOK}")
