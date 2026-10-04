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
from importlib.metadata import version as package_version
from io import StringIO
from typing import Dict, List

import kagglehub
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from tqdm import tqdm

%config InlineBackend.figure_format = 'retina'

from VideoFTTools import DatasetHelper, VideoExtractor, VideoAnalyzer, Evaluator
from VideoFTTools import upload_frame_to_blob_as_jpeg, date_sorted_df
"""
)

cells[3]["source"] = lines(
    """# Foundry SDK 2.x project authentication; no API keys or account endpoints.
load_dotenv()
project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=credential,
)
client = project_client.get_openai_client()

# Central variables
project_name, version = "action-recognition-global-vision", "20261002"
ft_deployment = f"{project_name}-{version}-ft"
base_deployment = f"{project_name}-{version}-base"
base_model = "gpt-4.1-2025-04-14"

# A compact, stratified run keeps the demo economical while exercising the workflow.
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
bold_start, bold_end = '\\033[1m', '\\033[0m'
run_started_at = pd.Timestamp.now("UTC").isoformat()
job_status_history = []
deployment_status_history = []

print({
    "project_endpoint": project_endpoint,
    "azure_ai_projects": package_version("azure-ai-projects"),
    "authentication": "DefaultAzureCredential",
})
"""
)

cells[31]["source"] = lines(
    """## Validate project deployments

The stable Foundry SDK 2.x deployment surface supports project-scoped listing
and retrieval. This notebook validates and monitors the Korea Central base and
fine-tuned deployments through `project_client.deployments`, with no manual
management-plane routes or parent account endpoint.
"""
)

cells[33]["source"] = lines(
    """# Monitor project-scoped deployments through Foundry SDK 2.x.
def wait_for_deployment(name, expected_model_name, timeout_seconds=900):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            deployment = project_client.deployments.get(name)
        except ResourceNotFoundError:
            print({"deployment": name, "status": "not_found"})
            time.sleep(20)
            continue

        details = deployment.as_dict()
        deployed_model = details.get("modelName")
        status = (
            "available"
            if deployed_model == expected_model_name
            else "model_mismatch"
        )
        snapshot = {
            "time": pd.Timestamp.now("UTC").isoformat(),
            "deployment": name,
            "status": status,
            "model": deployed_model,
            "sku": details.get("sku", {}).get("name"),
        }
        deployment_status_history.append(snapshot)
        print(snapshot)
        if status == "available":
            return details
        raise RuntimeError(
            f"Deployment {name} targets {deployed_model}, "
            f"expected {expected_model_name}"
        )
    raise TimeoutError(f"Timed out waiting for project deployment {name}")


base_deployment_result = wait_for_deployment(
    base_deployment,
    "gpt-4.1",
)
ft_deployment_result = wait_for_deployment(
    ft_deployment,
    fine_tuned_model,
)
print({
    "base_deployment": base_deployment,
    "ft_deployment": ft_deployment,
})
"""
)

cells[35]["source"] = lines(
    """video_analyzer = VideoAnalyzer(client, ft_deployment)
display(test_df.head())
"""
)

cells[40]["source"] = lines(
    """# Base and fine-tuned model evaluation.
base_video_analyzer = VideoAnalyzer(client, base_deployment)
ft_video_analyzer = VideoAnalyzer(client, ft_deployment)

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

cells[47]["source"] = lines(
    """run_completed_at = pd.Timestamp.now("UTC").isoformat()
execution_summary = {
    "project_endpoint": project_endpoint,
    "foundry_sdk_version": package_version("azure-ai-projects"),
    "authentication": "DefaultAzureCredential",
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
    "deployment_status_history": deployment_status_history,
    "base_deployment_details": base_deployment_result,
    "fine_tuned_deployment_details": ft_deployment_result,
    "privacy_preserving_frames": True,
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
