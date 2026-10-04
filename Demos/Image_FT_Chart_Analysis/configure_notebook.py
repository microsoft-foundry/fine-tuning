import nbformat


NOTEBOOK = "fine-tune-aoai-gpt4-1-for-chart-analysis.ipynb"


nb = nbformat.read(NOTEBOOK, as_version=4)

nb.cells[2].source = """import os
import json
import time
from io import BytesIO, StringIO

import pandas as pd
from PIL import Image
import base64
import requests
from IPython.display import display, Markdown
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import matplotlib.pyplot as plt

from datasets import load_dataset
from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient

%config InlineBackend.figure_format = 'retina'"""

nb.cells[3].source = """# Central variables
project_name = "chart-qa"
version = os.getenv("CHART_FT_RUN_VERSION", "run-20261002")
ft_model_deployment = f"{project_name}-{version}"
base_model_deployment = "gpt-4.1"
base_model_name = "gpt-4.1"
base_model_version = "2025-04-14"

project_endpoint = "https://ncus-prakharg-demo-2026.services.ai.azure.com/api/projects/ncus-prakharg-demo-2026"
subscription_id = "ba7979f7-d040-49c9-af1a-7414402bf622"
resource_name = "ncus-prakharg-demo-2026"
rg_name = "prakharg-demo-2026"
resource_location = "northcentralus"

# Dataset parameters
train_samples = int(os.getenv("CHART_FT_TRAIN_SAMPLES", "500"))
val_samples = int(os.getenv("CHART_FT_VAL_SAMPLES", "100"))
test_samples = int(os.getenv("CHART_FT_TEST_SAMPLES", "100"))
evaluation_samples = int(os.getenv("CHART_FT_EVAL_SAMPLES", "100"))
bold_start, bold_end = '\\033[1m', '\\033[0m'

SYSTEM_PROMPT = \"\"\"You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanations unless absolutely necessary.\"\"\"

credential = AzureCliCredential(
    subscription=subscription_id,
    process_timeout=60,
)
project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
client = project_client.get_openai_client()

arm_token = credential.get_token("https://management.azure.com/.default").token
arm_headers = {"Authorization": f"Bearer {arm_token}", "Content-Type": "application/json"}
arm_deployments_url = (
    f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{rg_name}"
    f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments"
)

def ensure_base_deployment():
    request_url = f"{arm_deployments_url}/{base_model_deployment}"
    deploy_data = {
        "sku": {"name": "GlobalStandard", "capacity": 100},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": base_model_name,
                "version": base_model_version,
            },
            "versionUpgradeOption": "OnceNewDefaultVersionAvailable",
        },
    }
    response = requests.put(
        request_url,
        params={"api-version": "2024-10-01"},
        headers=arm_headers,
        json=deploy_data,
        timeout=180,
    )
    response.raise_for_status()
    details = response.json()
    print(
        f"Baseline deployment {base_model_deployment}: "
        f"{details['properties']['provisioningState']} ({details['sku']['capacity']}K TPM)"
    )
    return details

base_deployment_details = ensure_base_deployment()"""

nb.cells[5].source = nb.cells[5].source.replace(
    "@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))",
    "@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=5, max=60))",
)

nb.cells[11].source = """# Upload training and validation files, then wait for completed imports.
train_file = client.files.create(
    file=open(f"{project_name}-{version}-train.jsonl", "rb"),
    purpose="fine-tune",
)
val_file = client.files.create(
    file=open(f"{project_name}-{version}-val.jsonl", "rb"),
    purpose="fine-tune",
)

for label, uploaded_file in [("training", train_file), ("validation", val_file)]:
    while uploaded_file.status not in {"processed", "error"}:
        print(f"{label} file {uploaded_file.id}: {uploaded_file.status}", flush=True)
        time.sleep(15)
        uploaded_file = client.files.retrieve(uploaded_file.id)
    print(f"{label} file {uploaded_file.id}: {uploaded_file.status}", flush=True)
    if uploaded_file.status != "processed":
        raise RuntimeError(f"{label} file import failed: {uploaded_file.status_details}")
    if label == "training":
        train_file = uploaded_file
    else:
        val_file = uploaded_file"""

nb.cells[13].source = """# Create and monitor the fine-tuning job. Retry only service-side failures.
file_train = train_file.id
file_val = val_file.id
suffix = f"{project_name}-{version}"
terminal_states = {"succeeded", "failed", "cancelled"}
fine_tuning_attempts = [
    job_id
    for job_id in os.getenv("CHART_FT_PRIOR_JOBS", "").split(",")
    if job_id
]

for attempt in range(1, 3):
    ft_job = client.fine_tuning.jobs.create(
        suffix=suffix if attempt == 1 else f"{suffix}-retry-{attempt}",
        training_file=file_train,
        validation_file=file_val,
        model=base_model_name,
        seed=42,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": int(os.getenv("CHART_FT_EPOCHS", "1")),
                    "batch_size": None,
                    "learning_rate_multiplier": None,
                }
            },
        },
        extra_body={
            "trainingType": os.getenv(
                "CHART_FT_TRAINING_TYPE",
                "developerTier",
            )
        },
    )
    fine_tuning_attempts.append(ft_job.id)
    print(f"Fine-tuning attempt {attempt}: {ft_job.id} ({ft_job.status})", flush=True)

    last_status = None
    while ft_job.status not in terminal_states:
        time.sleep(60)
        ft_job = client.fine_tuning.jobs.retrieve(ft_job.id)
        if ft_job.status != last_status:
            print(f"{ft_job.id}: {ft_job.status}", flush=True)
            last_status = ft_job.status

    print(f"{ft_job.id}: terminal status {ft_job.status}", flush=True)
    if ft_job.status == "succeeded":
        break

    error = ft_job.error.to_dict() if ft_job.error else {}
    print(f"{ft_job.id} error: {error}", flush=True)
    error_code = str(error.get("code", ""))
    if attempt == 2 or not error_code.startswith("5"):
        raise RuntimeError(f"Fine-tuning failed: {error}")

if ft_job.status != "succeeded":
    raise RuntimeError(f"Fine-tuning did not succeed: {ft_job.status}")"""

nb.cells[17].source = """# Retrieve fine-tuning metrics from result file
result_file_id = ft_job.to_dict()["result_files"][0]
results_content = client.files.content(result_file_id).content.decode()
results_df = pd.read_csv(StringIO(results_content))
display(results_df.tail())
plot_learning_curves(results_df, smoothing_window=20)"""

nb.cells[20].source = """# Record the resulting model and project resource.
fine_tuned_model = ft_job.to_dict()["fine_tuned_model"]
print(f"Azure AI resource: {resource_name} ({resource_location})")
print(f"Fine-tuned model: {fine_tuned_model}")"""

nb.cells[21].source = """# Deploy the succeeded fine-tuned model with Azure CLI credentials.
request_url = f"{arm_deployments_url}/{ft_model_deployment}"
deploy_data = {
    "sku": {"name": "GlobalStandard", "capacity": 100},
    "properties": {
        "model": {
            "format": "OpenAI",
            "name": fine_tuned_model,
            "version": "1",
        }
    },
}

print(f"Creating deployment {ft_model_deployment}...", flush=True)
response = requests.put(
    request_url,
    params={"api-version": "2024-10-01"},
    headers=arm_headers,
    json=deploy_data,
    timeout=180,
)
print(response.status_code, response.reason, flush=True)
if not response.ok:
    print(response.text, flush=True)
response.raise_for_status()
ft_deployment_details = response.json()
print(
    f"Fine-tuned deployment: {ft_deployment_details['properties']['provisioningState']} "
    f"({ft_deployment_details['sku']['capacity']}K TPM)",
    flush=True,
)"""

nb.cells[29].source = """# Process a deterministic evaluation subset with the baseline model.
ds_eval = ds_test.head(evaluation_samples).copy()
print(f"Evaluating {ds_eval.shape[0]} held-out examples.")
ds_eval["base-pred"] = [
    query_image(row.image, row.question, base_model_deployment)
    for row in tqdm(ds_eval.itertuples(), total=ds_eval.shape[0], desc="Baseline inference")
]"""

nb.cells[30].source = """# Process the same held-out subset with the fine-tuned model.
ds_eval["ft-pred"] = [
    query_image(row.image, row.question, ft_model_deployment)
    for row in tqdm(ds_eval.itertuples(), total=ds_eval.shape[0], desc="Fine-tuned inference")
]"""

nb.cells[31].source = nb.cells[31].source.replace(
    "@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))",
    "@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=5, max=60))",
)

nb.cells[33].source = """# Validate prediction accuracy of the baseline model.
ds_eval["base-eval"] = [
    evaluate(question, answer, prediction, base_model_deployment)
    for question, answer, prediction in tqdm(
        zip(ds_eval["question"], ds_eval["answer"], ds_eval["base-pred"]),
        total=ds_eval.shape[0],
        desc="Baseline judging",
    )
]"""

nb.cells[34].source = """# Validate fine-tuned predictions with the same base-model judge.
ds_eval["ft-eval"] = [
    evaluate(question, answer, prediction, base_model_deployment)
    for question, answer, prediction in tqdm(
        zip(ds_eval["question"], ds_eval["answer"], ds_eval["ft-pred"]),
        total=ds_eval.shape[0],
        desc="Fine-tuned judging",
    )
]"""

nb.cells[35].source = """# Store evaluation results while retaining images.
ds_eval.to_hdf(f"eval-{project_name}-{version}.h5", key="df", mode="w", index=False)"""

nb.cells[36].source = """base_correct_count = ds_eval["base-eval"].value_counts().get("CORRECT", 0)
base_eval_observations = ds_eval.shape[0]
ft_correct_count = ds_eval["ft-eval"].value_counts().get("CORRECT", 0)
ft_eval_observations = ds_eval.shape[0]

chart_data = {
    "title": "GPT-4.1 ChartQA accuracy - baseline vs fine-tuned model",
    "baseline": "GPT-4.1",
    "fine-tuned": "GPT-4.1 fine-tuned",
    "baseline accuracy": base_correct_count / base_eval_observations,
    "fine-tuned accuracy": ft_correct_count / ft_eval_observations,
}
chart_data["absolute improvement"] = (
    chart_data["fine-tuned accuracy"] - chart_data["baseline accuracy"]
)
chart_data["relative improvement"] = (
    chart_data["absolute improvement"] / chart_data["baseline accuracy"]
    if chart_data["baseline accuracy"]
    else None
)

models = [chart_data["baseline"], chart_data["fine-tuned"]]
accuracies = [chart_data["baseline accuracy"], chart_data["fine-tuned accuracy"]]
plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=["blue", "green"])
plt.title(chart_data["title"])
plt.ylabel("Accuracy")
plt.xlabel("Model")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.4f}", ha="center", fontsize=10)
plt.tight_layout()
plt.show()

run_summary = {
    "project_endpoint": project_endpoint,
    "subscription_id": subscription_id,
    "resource_group": rg_name,
    "resource_name": resource_name,
    "resource_location": resource_location,
    "base_model": base_model_name,
    "base_model_version": base_model_version,
    "base_deployment": base_model_deployment,
    "fine_tuning_attempts": fine_tuning_attempts,
    "fine_tuning_job_id": ft_job.id,
    "fine_tuned_model": fine_tuned_model,
    "fine_tuned_deployment": ft_model_deployment,
    "training_file_id": file_train,
    "validation_file_id": file_val,
    "result_file_id": result_file_id,
    "train_samples": train_samples,
    "validation_samples": val_samples,
    "test_samples": test_samples,
    "evaluation_samples": evaluation_samples,
    "metrics": chart_data,
}
with open(f"run-summary-{version}.json", "w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2)

print(json.dumps(chart_data, indent=2))"""

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

nbformat.write(nb, NOTEBOOK)
