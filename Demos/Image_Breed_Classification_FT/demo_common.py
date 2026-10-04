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
from PIL import Image

PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
ACCOUNT_NAME = os.environ["AZURE_AI_ACCOUNT_NAME"]
BASE_MODEL = "gpt-4o-2024-08-06"
BASE_DEPLOYMENT = "image-breed-gpt-4o-base"
FT_DEPLOYMENT_PREFIX = "image-breed-gpt-4o-ft"
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


def get_project_client() -> AIProjectClient:
    configure_azure_cli_path()
    return AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )


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
    client: Any,
    deployment: str,
    prompt: str,
    image_uri: str,
) -> str:
    for attempt in range(1, 13):
        try:
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
        except Exception as error:
            body = getattr(error, "body", None) or {}
            code = (body.get("error") or body).get("code") if isinstance(body, dict) else None
            if code != "BadRequestForDependentService" or attempt == 12:
                raise
            print(
                f"Deployment {deployment} is still propagating; "
                f"retrying in 30 seconds ({attempt}/12)"
            )
            time.sleep(30)
    raise RuntimeError(f"Classification retries exhausted for {deployment}")


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
    client: Any,
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
    client: Any,
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


def deploy_fine_tuned_model(
    project_client: AIProjectClient,
    model_name: str,
    deployment_name: str,
) -> Any:
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
        deployment_name,
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
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(
            f"Deployment {deployment_name} creation failed: {details}"
        ) from error
    for _ in range(60):
        try:
            deployment = project_client.deployments.get(deployment_name)
        except Exception as error:
            print(f"Deployment {deployment_name}: not visible yet ({error})")
        else:
            print(
                f"Deployment {deployment_name}: "
                f"{deployment.model_name} version {deployment.model_version}"
            )
            if deployment.model_name == model_name:
                return deployment
        time.sleep(30)
    raise TimeoutError(
        f"Deployment {deployment_name} did not expose model {model_name}"
    )
