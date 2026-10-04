import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name(
    "fine-tune-aoai-gpt4-1-action-detection.ipynb"
)


def lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[2]["source"] = lines(
    """import json
import os
import time
from io import StringIO
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import requests
from openai import AzureOpenAI
from sklearn.metrics import classification_report
from tqdm import tqdm
import kagglehub
from azure.identity import AzureCliCredential, get_bearer_token_provider

%config InlineBackend.figure_format = 'retina'

from VideoFTTools import DatasetHelper, VideoExtractor, VideoAnalyzer, Evaluator
from VideoFTTools import upload_frame_to_blob_as_jpeg, date_sorted_df
"""
)

cells[3]["source"] = lines(
    """# Azure CLI authentication; no API keys or secrets are used.
project_endpoint = "https://koreacentral-prakharg-demo-2026.services.ai.azure.com/api/projects/koreacentral-prakharg-demo-2026"
api_version = "2025-05-01"
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://ai.azure.com/.default",
)
client = AzureOpenAI(
    azure_endpoint=project_endpoint,
    azure_ad_token_provider=token_provider,
    api_version=api_version,
)
account_endpoint = "https://koreacentral-prakharg-demo-2026.cognitiveservices.azure.com/"
inference_token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)
inference_client = AzureOpenAI(
    azure_endpoint=account_endpoint,
    azure_ad_token_provider=inference_token_provider,
    api_version="2024-10-21",
)

# Central variables
project_name, version = "action-recognition-global-vision", "20261002"
ft_deployment = f"{project_name}-{version}-ft"
base_deployment = f"{project_name}-{version}-base"
base_model = "gpt-4.1-2025-04-14"

# A compact, stratified run keeps the demo economical while exercising the full workflow.
top_n_classes = 3
train_samples_per_class = 12
val_samples_per_class = 4
test_samples_per_class = 5
no_frames = 3
random_seed = 0

# Keep downloaded and generated data in this demo directory.
os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), ".kagglehub")

container_name = "frames"
connection_string = None

# Target Azure AI Services project/account.
subscription_id = "ba7979f7-d040-49c9-af1a-7414402bf622"
resource_name = "koreacentral-prakharg-demo-2026"
rg_name = "prakharg-demo-2026"
project_resource_id = (
    f"/subscriptions/{subscription_id}/resourceGroups/{rg_name}"
    f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
    f"/projects/{resource_name}"
)

bold_start, bold_end = '\\033[1m', '\\033[0m'
run_started_at = pd.Timestamp.now("UTC").isoformat()
job_status_history = []
"""
)

cells[8]["source"] = lines(
    """top_classes = orig_train_df['label'].value_counts().index[:top_n_classes]

def stratified_sample(df, samples_per_class):
    scoped = df[df['label'].isin(top_classes)]
    return (
        scoped.groupby('label', group_keys=False)
        .sample(n=samples_per_class, random_state=random_seed)
        .reset_index(drop=True)
    )

train_df = stratified_sample(orig_train_df, train_samples_per_class)
val_df = stratified_sample(orig_val_df, val_samples_per_class)
test_df = stratified_sample(orig_test_df, test_samples_per_class)

dataset_helper.plot_label_counts(train_df, 'Reduced training dataset')
dataset_helper.plot_label_counts(val_df, 'Reduced validation dataset')
dataset_helper.plot_label_counts(test_df, 'Reduced test dataset')
print({
    "classes": list(top_classes),
    "train": len(train_df),
    "validation": len(val_df),
    "test": len(test_df),
    "frames_per_video": no_frames,
})
"""
)

cells[12]["source"] = lines(
    """# Pick a random video and preview privacy-preserving motion/edge frames.
video_path = os.path.join(
    dataset_path,
    train_df.sample(n=1, random_state=random_seed)['clip_path'].iloc[0],
)
video_extractor = VideoExtractor(video_path, privacy_preserving=True)
frames = video_extractor.extract_n_video_frames(n=no_frames)
video_extractor.display_frames(frames, height=180)
"""
)

cells[17]["source"] = lines(
    """top_classes_str = ', '.join(top_classes)

system_message = f\"\"\"You are an expert in identifying human activities in video clips.
You are provided with a time-ordered series of privacy-preserving edge frames from a video.
Use the shapes, objects, poses, and motion progression across the ordered frames to identify
one activity from this list:
{top_classes_str}

Provide the result as a valid JSON object in the exact format below:
{{
    "activity": "The identified activity from the provided list, spelled exactly as given."
}}
\"\"\"
"""
)

for cell_index in (19, 39):
    source = "".join(cells[cell_index]["source"])
    source = source.replace(
        "VideoExtractor(uri=full_path)",
        "VideoExtractor(uri=full_path, privacy_preserving=True)",
    )
    source = source.replace(
        "VideoExtractor(uri)",
        "VideoExtractor(uri, privacy_preserving=True)",
    )
    cells[cell_index]["source"] = lines(source)

cells[20]["source"] = lines(
    """# Generate training file.
train_jsonl = f"{project_name}-{version}-train.jsonl"
print(f"Creating dataset: {train_jsonl}")
train_ds = generate_ft_dataset(
    train_df,
    frames_per_video=no_frames,
    image_source='base64',
)
with open(train_jsonl, "w", encoding="utf-8") as jsonl_file:
    for message in train_ds:
        jsonl_file.write(json.dumps(message) + "\\n")
print(f"Training examples: {len(train_ds)}")
print(f"File size: {os.path.getsize(train_jsonl) / (1024 * 1024):.2f} MB")
"""
)

cells[21]["source"] = lines(
    """# Generate validation file.
val_jsonl = f"{project_name}-{version}-val.jsonl"
print(f"Creating dataset: {val_jsonl}")
val_ds = generate_ft_dataset(
    val_df,
    frames_per_video=no_frames,
    image_source='base64',
)
with open(val_jsonl, "w", encoding="utf-8") as jsonl_file:
    for message in val_ds:
        jsonl_file.write(json.dumps(message) + "\\n")
print(f"Validation examples: {len(val_ds)}")
print(f"File size: {os.path.getsize(val_jsonl) / (1024 * 1024):.2f} MB")
"""
)

cells[23]["source"] = lines(
    """def wait_for_file(file_id, timeout_seconds=900):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        uploaded = client.files.retrieve(file_id)
        print(f"File {file_id}: {uploaded.status}")
        if uploaded.status == "processed":
            return uploaded
        if uploaded.status == "error":
            raise RuntimeError(f"File processing failed: {uploaded}")
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for file {file_id}")

with open(train_jsonl, "rb") as handle:
    train_file = client.files.create(file=handle, purpose="fine-tune")
with open(val_jsonl, "rb") as handle:
    val_file = client.files.create(file=handle, purpose="fine-tune")

train_file = wait_for_file(train_file.id)
val_file = wait_for_file(val_file.id)
print({"training_file": train_file.id, "validation_file": val_file.id})
"""
)

cells[25]["source"] = lines(
    """# Create or reuse the successful global fine-tuning job for this run.
file_train = train_file.id
file_val = val_file.id
suffix = f"{project_name}-{version}"[-40:]

matching_jobs = [
    job for job in client.fine_tuning.jobs.list(limit=100).data
    if job.suffix == suffix and job.model == base_model
]
succeeded_jobs = [job for job in matching_jobs if job.status == "succeeded"]
if succeeded_jobs:
    ft_job = max(succeeded_jobs, key=lambda job: job.created_at)
    print(f"Reusing successful fine-tuning job: {ft_job.id}")
else:
    ft_job = client.fine_tuning.jobs.create(
        suffix=suffix,
        training_file=file_train,
        validation_file=file_val,
        model=base_model,
        seed=random_seed,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 1,
                }
            },
        },
        extra_body={"trainingType": "globalStandard"},
    )

terminal_states = {"succeeded", "failed", "cancelled"}
last_status = None
while True:
    ft_job = client.fine_tuning.jobs.retrieve(ft_job.id)
    status = ft_job.status
    if status != last_status:
        snapshot = {
            "time": pd.Timestamp.now("UTC").isoformat(),
            "job_id": ft_job.id,
            "status": status,
        }
        job_status_history.append(snapshot)
        print(snapshot)
        last_status = status
    if status in terminal_states:
        break
    time.sleep(30)

if ft_job.status != "succeeded":
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=ft_job.id,
        limit=100,
    )
    raise RuntimeError(
        json.dumps(
            {
                "job": ft_job.to_dict(),
                "events": [event.to_dict() for event in events.data],
            },
            default=str,
        )
    )
print(f"Fine-tuning succeeded: {ft_job.id}")
"""
)

cells[27]["source"] = lines(
    """# The submitted job was monitored to a terminal state in the previous cell.
ft_job = client.fine_tuning.jobs.retrieve(ft_job.id)
"""
)

cells[32]["source"] = lines(
    """fine_tuned_model = ft_job.fine_tuned_model
print({
    "base_model": base_model,
    "fine_tuned_model": fine_tuned_model,
    "training_type": ft_job.to_dict().get("trainingType"),
})
"""
)

cells[33]["source"] = lines(
    """management_token = credential.get_token(
    "https://management.azure.com/.default"
).token
deploy_headers = {
    "Authorization": f"Bearer {management_token}",
    "Content-Type": "application/json",
}
deploy_api_version = "2024-10-01"
account_deployments_url = (
    f"https://management.azure.com/subscriptions/{subscription_id}"
    f"/resourceGroups/{rg_name}"
    f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
    "/deployments"
)

def put_deployment(name, model_name, model_version, capacity=10):
    request_url = f"{account_deployments_url}/{name}"
    payload = {
        "sku": {"name": "GlobalStandard", "capacity": capacity},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": model_name,
                "version": model_version,
            }
        },
    }
    response = requests.put(
        request_url,
        params={"api-version": deploy_api_version},
        headers=deploy_headers,
        json=payload,
        timeout=120,
    )
    if response.status_code not in {200, 201, 202}:
        raise RuntimeError(
            f"Deployment {name} failed: {response.status_code} {response.text}"
        )
    deadline = time.time() + 1800
    while time.time() < deadline:
        state_response = requests.get(
            request_url,
            params={"api-version": deploy_api_version},
            headers=deploy_headers,
            timeout=60,
        )
        state_response.raise_for_status()
        state = state_response.json()["properties"]["provisioningState"]
        print(f"Deployment {name}: {state}")
        if state == "Succeeded":
            return state_response.json()
        if state in {"Failed", "Canceled"}:
            raise RuntimeError(json.dumps(state_response.json(), indent=2))
        time.sleep(20)
    raise TimeoutError(f"Timed out waiting for deployment {name}")

base_deployment_result = put_deployment(
    base_deployment,
    "gpt-4.1",
    "2025-04-14",
)
ft_deployment_result = put_deployment(
    ft_deployment,
    fine_tuned_model,
    "1",
)
print({
    "base_deployment": base_deployment,
    "ft_deployment": ft_deployment,
})
"""
)

cells[35]["source"] = lines(
    """video_analyzer = VideoAnalyzer(inference_client, ft_deployment)
display(test_df.head())
"""
)

cells[36]["source"] = lines(
    """# Pick a deterministic test video.
sample = test_df.sort_values(["label", "clip_path"]).iloc[0]
ground_truth_label = sample['label']
clip_path = sample['clip_path']
video_path = os.path.join(dataset_path, clip_path)
video_extractor = VideoExtractor(video_path, privacy_preserving=True)
frames = video_extractor.extract_n_video_frames(n=no_frames)
video_extractor.display_frames(frames, height=180)
"""
)

cells[40]["source"] = lines(
    """# Base and fine-tuned model evaluation.
base_video_analyzer = VideoAnalyzer(inference_client, base_deployment)
ft_video_analyzer = VideoAnalyzer(inference_client, ft_deployment)

test_df = classify_videos_in_df(
    test_df=test_df,
    video_analyzer=base_video_analyzer,
    local_path=dataset_path,
    frames_per_video=no_frames,
    column_name='base_predicted_label',
)
test_df = classify_videos_in_df(
    test_df=test_df,
    video_analyzer=ft_video_analyzer,
    local_path=dataset_path,
    frames_per_video=no_frames,
    column_name='ft_predicted_label',
)

base_accuracy = float(
    (test_df["label"] == test_df["base_predicted_label"]).mean()
)
ft_accuracy = float(
    (test_df["label"] == test_df["ft_predicted_label"]).mean()
)
print({
    "base_accuracy": base_accuracy,
    "fine_tuned_accuracy": ft_accuracy,
    "accuracy_delta": ft_accuracy - base_accuracy,
})
"""
)

cells[41]["source"] = lines(
    """# Save evaluation results.
eval_results_path = (
    f"{project_name}-{version}-eval-results-{test_df.shape[0]}-entries.csv"
)
test_df.to_csv(eval_results_path, index=False)
print(eval_results_path)
"""
)

cells[42]["source"] = lines(
    """# Reload the evaluation artifact produced by this run.
test_df = pd.read_csv(eval_results_path)
display(test_df.head())
"""
)

cells[47]["source"] = lines(
    """    run_completed_at = pd.Timestamp.now("UTC").isoformat()
execution_summary = {
    "project_endpoint": project_endpoint,
    "project_resource_id": project_resource_id,
    "base_model": base_model,
    "fine_tuning_job_id": ft_job.id,
    "fine_tuning_status": ft_job.status,
    "training_type": ft_job.to_dict().get("trainingType"),
    "fine_tuned_model": fine_tuned_model,
    "base_deployment": base_deployment,
    "fine_tuned_deployment": ft_deployment,
    "classes": list(top_classes),
    "train_examples": len(train_df),
    "validation_examples": len(val_df),
    "test_examples": len(test_df),
    "frames_per_video": no_frames,
    "base_accuracy": base_accuracy,
    "fine_tuned_accuracy": ft_accuracy,
    "accuracy_delta": ft_accuracy - base_accuracy,
    "job_status_history": job_status_history,
    "started_at": run_started_at,
    "completed_at": run_completed_at,
}
with open("video-ft-execution-summary.json", "w", encoding="utf-8") as handle:
    json.dump(execution_summary, handle, indent=2)
print(json.dumps(execution_summary, indent=2))
"""
)

notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3.12",
    "language": "python",
    "name": "python3",
}
NOTEBOOK.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
