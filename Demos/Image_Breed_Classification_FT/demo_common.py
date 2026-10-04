from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai import OpenAI
from PIL import Image

PROJECT_ENDPOINT = (
    "https://eastus2-prakharg-demo-2026.services.ai.azure.com/api/projects/"
    "eastus2-prakharg-demo-2026"
)
SUBSCRIPTION_ID = "ba7979f7-d040-49c9-af1a-7414402bf622"
RESOURCE_GROUP = "prakharg-demo-2026"
ACCOUNT_NAME = "eastus2-prakharg-demo-2026"
BASE_MODEL = "gpt-4o-2024-08-06"
BASE_DEPLOYMENT = "image-breed-gpt-4o-base"
FT_DEPLOYMENT = "image-breed-gpt-4o-ft"
SUCCESSFUL_JOB_ID = "ftjob-25d544adc35d47e1bbcf92b3977037d3"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "jessicali9530/stanford-dogs-dataset"
)
BREED_FOLDERS = (
    "n02085620-Chihuahua",
    "n02092339-Weimaraner",
    "n02109525-Saint_Bernard",
    "n02111277-Newfoundland",
)
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
TERMINAL_FILE_STATUSES = {"processed", "error", "expired"}


def configure_azure_cli_path() -> None:
    scripts = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if scripts not in path_parts:
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")


def get_openai_client() -> OpenAI:
    configure_azure_cli_path()
    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    return project.get_openai_client()


def download_dataset(root: Path) -> Path:
    images_root = root / "stanford_dogs_dataset" / "images" / "Images"
    if images_root.exists():
        return images_root

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "stanford-dogs-dataset.zip"
    if not archive.exists():
        urllib.request.urlretrieve(DATASET_URL, archive)

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root / "stanford_dogs_dataset")
    archive.unlink(missing_ok=True)

    if not images_root.exists():
        raise FileNotFoundError(f"Dataset images were not extracted to {images_root}")
    return images_root


def clean_breed(folder: str) -> str:
    return folder.split("-", 1)[1].replace("_", " ").title()


def image_data_uri(path: Path, size: int = 384) -> str:
    from io import BytesIO

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        output = BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def build_demo_dataset(images_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for folder in BREED_FOLDERS:
        paths = sorted((images_root / folder).glob("*.jpg"))[:20]
        if len(paths) < 20:
            raise ValueError(f"Expected at least 20 images for {folder}, found {len(paths)}")
        breed = clean_breed(folder)
        for index, path in enumerate(paths):
            split = "train" if index < 12 else "validation" if index < 16 else "test"
            rows.append(
                {
                    "id": f"{folder}-{index:02d}",
                    "breed": breed,
                    "image_path": str(path.resolve()),
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def classification_prompt(labels: list[str]) -> str:
    return (
        "Classify the dog in the image as exactly one of these breeds: "
        + ", ".join(labels)
        + ". Return only the breed name."
    )


def classify(
    client: OpenAI,
    deployment: str,
    prompt: str,
    image_uri: str,
) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_uri, "detail": "low"},
                    }
                ],
            },
        ],
        max_tokens=20,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def normalize_prediction(value: str, labels: list[str]) -> str:
    cleaned = value.strip().strip("`\"' .")
    lowered = cleaned.casefold()
    for label in labels:
        if label.casefold() == lowered or label.casefold() in lowered:
            return label
    return cleaned


def write_fine_tuning_jsonl(
    frame: pd.DataFrame,
    output_path: Path,
    prompt: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in frame.itertuples(index=False):
            example = {
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_uri(Path(row.image_path)),
                                    "detail": "low",
                                },
                            }
                        ],
                    },
                    {"role": "assistant", "content": row.breed},
                ]
            }
            stream.write(json.dumps(example) + "\n")


def wait_for_job(
    client: OpenAI,
    job_id: str,
    poll_seconds: int = 60,
) -> Any:
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {job.id}: {job.status}")
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        time.sleep(poll_seconds)


def wait_for_file(
    client: OpenAI,
    file_id: str,
    poll_seconds: int = 10,
) -> Any:
    while True:
        file = client.files.retrieve(file_id)
        status = str(file.status)
        print(f"File {file.id}: {status}")
        if status in TERMINAL_FILE_STATUSES:
            if status != "processed":
                raise RuntimeError(f"File {file.id} ended as {status}: {file.status_details}")
            return file
        time.sleep(poll_seconds)


def deploy_fine_tuned_model(model_name: str) -> dict[str, Any]:
    configure_azure_cli_path()
    command = [
        "az",
        "cognitiveservices",
        "account",
        "deployment",
        "create",
        "--subscription",
        SUBSCRIPTION_ID,
        "--resource-group",
        RESOURCE_GROUP,
        "--name",
        ACCOUNT_NAME,
        "--deployment-name",
        FT_DEPLOYMENT,
        "--model-format",
        "OpenAI",
        "--model-name",
        model_name,
        "--model-version",
        "1",
        "--sku-name",
        "GlobalStandard",
        "--sku-capacity",
        "10",
        "--output",
        "json",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    show_command = [
        "az",
        "cognitiveservices",
        "account",
        "deployment",
        "show",
        "--subscription",
        SUBSCRIPTION_ID,
        "--resource-group",
        RESOURCE_GROUP,
        "--name",
        ACCOUNT_NAME,
        "--deployment-name",
        FT_DEPLOYMENT,
        "--output",
        "json",
    ]
    for _ in range(60):
        result = subprocess.run(
            show_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        deployment = json.loads(result.stdout)
        state = deployment["properties"]["provisioningState"]
        print(f"Deployment {FT_DEPLOYMENT}: {state}")
        if state == "Succeeded":
            return deployment
        if state in {"Failed", "Canceled"}:
            raise RuntimeError(f"Deployment {FT_DEPLOYMENT} ended as {state}")
        time.sleep(30)
    raise TimeoutError(f"Deployment {FT_DEPLOYMENT} did not become ready")


def project_job_link(job_id: str) -> str:
    return (
        "https://ai.azure.com/fine-tuning/"
        + job_id
        + "?wsid=/subscriptions/"
        + SUBSCRIPTION_ID
        + "/resourceGroups/"
        + RESOURCE_GROUP
        + "/providers/Microsoft.CognitiveServices/accounts/"
        + ACCOUNT_NAME
        + "/projects/"
        + ACCOUNT_NAME
    )
