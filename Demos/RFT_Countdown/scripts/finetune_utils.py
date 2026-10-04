import pandas as pd

from scripts.client_utils import get_openai_client


def create_finetune_sft(
    training_file_id: str,
    model: str,
    suffix: str | None = None,
    hyperparameters: dict | None = None,
) -> str:
    payload = {
        "training_file": training_file_id,
        "model": model,
        "suffix": suffix,
    }
    if hyperparameters is not None:
        payload["hyperparameters"] = hyperparameters
    with get_openai_client() as client:
        job = client.fine_tuning.jobs.create(**payload)
    print(f"Supervised fine-tuning job created: {job.id}")
    return job.id


def list_finetunes():
    with get_openai_client() as client:
        jobs = list(client.fine_tuning.jobs.list())
    print(f"Found {len(jobs)} fine-tuning jobs.")
    return jobs


def get_finetune_status(fine_tune_id: str) -> str:
    with get_openai_client() as client:
        job = client.fine_tuning.jobs.retrieve(fine_tune_id)
    print(f"Status for fine-tuning job {fine_tune_id}: {job.status}")
    return job.status


def print_finetune_details(fine_tune_id: str):
    with get_openai_client() as client:
        job = client.fine_tuning.jobs.retrieve(fine_tune_id)

    details = {
        "ID": job.id,
        "Status": job.status,
        "Model": job.model,
        "Fine-Tuned Model": job.fine_tuned_model or "Not available",
        "Created At": pd.to_datetime(job.created_at, unit="s"),
        "Finished At": (
            pd.to_datetime(job.finished_at, unit="s")
            if job.finished_at
            else "Not finished"
        ),
        "Training File": job.training_file,
        "Validation File": job.validation_file or "Not provided",
        "Trained Tokens": job.trained_tokens or "Not available",
        "Estimated Finish": (
            pd.to_datetime(job.estimated_finish, unit="s")
            if job.estimated_finish
            else "Not available"
        ),
        "Error": job.error.message if job.error and job.error.message else "None",
    }
    frame = pd.DataFrame(details.items(), columns=["Attribute", "Value"])
    print(frame.to_string(index=False))
    return job
