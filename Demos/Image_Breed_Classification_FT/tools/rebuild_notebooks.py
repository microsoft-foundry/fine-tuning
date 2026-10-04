from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


vlm = nbf.v4.new_notebook()
vlm.metadata.kernelspec = {
    "display_name": "Image Breed Demo (Python 3.12)",
    "language": "python",
    "name": "python3",
}
vlm.cells = [
    markdown(
        """
# Vision fine-tuning for dog breed classification

This executable demo uses `DefaultAzureCredential` with the specified Azure AI
Foundry project, the supported `gpt-4o-2024-08-06` vision fine-tuning model,
and a bounded four-breed Stanford Dogs subset.
"""
    ),
    code(
        """
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score

from demo_common import (
    ACCOUNT_NAME, BASE_DEPLOYMENT, BASE_MODEL, FT_DEPLOYMENT, PROJECT_ENDPOINT,
    SUCCESSFUL_JOB_ID,
    build_demo_dataset, classification_prompt, classify, deploy_fine_tuned_model,
    download_dataset, get_openai_client, image_data_uri, normalize_prediction,
    project_job_link, wait_for_file, wait_for_job, write_fine_tuning_jsonl,
)

ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
client = get_openai_client()
print("Project:", PROJECT_ENDPOINT)
print("Base model:", BASE_MODEL)
print("Base deployment:", BASE_DEPLOYMENT)
"""
    ),
    code(
        """
images_root = download_dataset(ROOT / "data")
dataset_df = build_demo_dataset(images_root)
dataset_df.to_csv(OUTPUTS / "dataset_manifest.csv", index=False)
print(dataset_df.groupby(["split", "breed"]).size().unstack(fill_value=0))
"""
    ),
    code(
        """
labels = sorted(dataset_df["breed"].unique())
prompt = classification_prompt(labels)
test_df = dataset_df[dataset_df["split"] == "test"].copy()

base_predictions = []
for row in test_df.itertuples(index=False):
    raw = classify(client, BASE_DEPLOYMENT, prompt, image_data_uri(Path(row.image_path)))
    base_predictions.append(normalize_prediction(raw, labels))
    print(row.id, row.breed, "=>", base_predictions[-1])

test_df["base_prediction"] = base_predictions
base_accuracy = accuracy_score(test_df["breed"], test_df["base_prediction"])
print(f"Base accuracy: {base_accuracy:.3f}")
"""
    ),
    code(
        """
train_file = OUTPUTS / "train_classification.jsonl"
validation_file = OUTPUTS / "validation_classification.jsonl"
write_fine_tuning_jsonl(
    dataset_df[dataset_df["split"] == "train"], train_file, prompt
)
write_fine_tuning_jsonl(
    dataset_df[dataset_df["split"] == "validation"], validation_file, prompt
)
print("Training examples:", sum(1 for _ in train_file.open(encoding="utf-8")))
print("Validation examples:", sum(1 for _ in validation_file.open(encoding="utf-8")))
"""
    ),
    code(
        """
training_upload = client.files.create(file=train_file.open("rb"), purpose="fine-tune")
validation_upload = client.files.create(
    file=validation_file.open("rb"), purpose="fine-tune"
)
print("Training file:", training_upload.id)
print("Validation file:", validation_upload.id)
wait_for_file(client, training_upload.id)
wait_for_file(client, validation_upload.id)

job = client.fine_tuning.jobs.retrieve(SUCCESSFUL_JOB_ID)
print("Using completed job:", job.id)
print("Job link:", project_job_link(job.id))
print(job.model_dump_json(indent=2))
if job.status != "succeeded":
    raise RuntimeError(f"Fine-tuning job {job.id} ended as {job.status}: {job.error}")
"""
    ),
    code(
        """
deployment = deploy_fine_tuned_model(job.fine_tuned_model)
print("Fine-tuned deployment:", deployment["name"])
print("Provisioning state:", deployment["properties"]["provisioningState"])
"""
    ),
    code(
        """
ft_predictions = []
for row in test_df.itertuples(index=False):
    raw = classify(client, FT_DEPLOYMENT, prompt, image_data_uri(Path(row.image_path)))
    ft_predictions.append(normalize_prediction(raw, labels))
    print(row.id, row.breed, "=>", ft_predictions[-1])

test_df["ft_prediction"] = ft_predictions
ft_accuracy = accuracy_score(test_df["breed"], test_df["ft_prediction"])
accuracy_delta = ft_accuracy - base_accuracy
print(f"Fine-tuned accuracy: {ft_accuracy:.3f}")
print(f"Accuracy delta: {accuracy_delta:+.3f}")

test_df.to_csv(OUTPUTS / "accuracy_predictions.csv", index=False)
run_summary = {
    "project_endpoint": PROJECT_ENDPOINT,
    "resource": ACCOUNT_NAME,
    "base_model": BASE_MODEL,
    "base_deployment": BASE_DEPLOYMENT,
    "fine_tuned_model": job.fine_tuned_model,
    "fine_tuned_deployment": FT_DEPLOYMENT,
    "training_file_id": training_upload.id,
    "validation_file_id": validation_upload.id,
    "job_id": job.id,
    "job_link": project_job_link(job.id),
    "job_status": job.status,
    "base_accuracy": base_accuracy,
    "fine_tuned_accuracy": ft_accuracy,
    "accuracy_delta": accuracy_delta,
    "test_examples": len(test_df),
}
(OUTPUTS / "run_summary.json").write_text(
    json.dumps(run_summary, indent=2), encoding="utf-8"
)
run_summary
"""
    ),
]

latency = nbf.v4.new_notebook()
latency.metadata.kernelspec = vlm.metadata.kernelspec
latency.cells = [
    markdown(
        """
# Base versus fine-tuned model latency

Measures sequential client-observed latency on the same image set and writes
per-request and aggregate comparison artifacts.
"""
    ),
    code(
        """
from pathlib import Path
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

from demo_common import (
    BASE_DEPLOYMENT, FT_DEPLOYMENT, classification_prompt, classify,
    get_openai_client, image_data_uri,
)

ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"
LATENCY_OUTPUTS = ROOT / "latency_outputs"
LATENCY_OUTPUTS.mkdir(exist_ok=True)

run_summary = json.loads((OUTPUTS / "run_summary.json").read_text(encoding="utf-8"))
manifest = pd.read_csv(OUTPUTS / "dataset_manifest.csv")
test_df = manifest[manifest["split"] == "test"].head(8).copy()
labels = sorted(manifest["breed"].unique())
prompt = classification_prompt(labels)
client = get_openai_client()
print("Comparing:", BASE_DEPLOYMENT, "vs", FT_DEPLOYMENT)
"""
    ),
    code(
        """
records = []
for deployment, model_kind in (
    (BASE_DEPLOYMENT, "base"),
    (FT_DEPLOYMENT, "fine_tuned"),
):
    for row in test_df.itertuples(index=False):
        image_uri = image_data_uri(Path(row.image_path))
        started = time.perf_counter()
        prediction = classify(client, deployment, prompt, image_uri)
        latency_ms = (time.perf_counter() - started) * 1000
        records.append(
            {
                "model_kind": model_kind,
                "deployment": deployment,
                "image_id": row.id,
                "latency_ms": latency_ms,
                "prediction": prediction,
            }
        )
        print(model_kind, row.id, f"{latency_ms:.1f} ms")

latencies = pd.DataFrame(records)
latencies.to_csv(LATENCY_OUTPUTS / "per_request_latencies.csv", index=False)
"""
    ),
    code(
        """
summary = (
    latencies.groupby(["model_kind", "deployment"])["latency_ms"]
    .agg(
        count="count",
        mean_ms="mean",
        median_ms="median",
        min_ms="min",
        max_ms="max",
        std_ms="std",
    )
    .reset_index()
)
for percentile in (0.90, 0.95, 0.99):
    values = (
        latencies.groupby(["model_kind", "deployment"])["latency_ms"]
        .quantile(percentile)
        .rename(f"p{int(percentile * 100)}_ms")
        .reset_index()
    )
    summary = summary.merge(values, on=["model_kind", "deployment"])

base_mean = float(summary.loc[summary["model_kind"] == "base", "mean_ms"].iloc[0])
ft_mean = float(
    summary.loc[summary["model_kind"] == "fine_tuned", "mean_ms"].iloc[0]
)
summary["throughput_rps"] = 1000 / summary["mean_ms"]
summary.to_csv(LATENCY_OUTPUTS / "latency_summary.csv", index=False)

latency_comparison = {
    "base_mean_ms": base_mean,
    "fine_tuned_mean_ms": ft_mean,
    "fine_tuned_minus_base_ms": ft_mean - base_mean,
    "fine_tuned_change_percent": ((ft_mean / base_mean) - 1) * 100,
    "requests_per_model": len(test_df),
}
(LATENCY_OUTPUTS / "latency_comparison.json").write_text(
    json.dumps(latency_comparison, indent=2), encoding="utf-8"
)
display(summary)
latency_comparison
"""
    ),
    code(
        """
ax = latencies.boxplot(column="latency_ms", by="model_kind", grid=False)
ax.set_title("Base vs fine-tuned latency")
ax.set_xlabel("Model")
ax.set_ylabel("Latency (ms)")
plt.suptitle("")
plt.tight_layout()
plt.savefig(LATENCY_OUTPUTS / "latency_boxplot.png", dpi=150)
plt.show()
"""
    ),
]

nbf.write(vlm, ROOT / "images_classification_vlm.ipynb")
nbf.write(latency, ROOT / "latency_base_ft_models.ipynb")
