from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# API keys and endpoint
OAI_API_TYPE = os.getenv("OAI_API_TYPE", "azure").lower()
AZURE_API_KEY = os.getenv("AZURE_API_KEY", None)
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT", "") + "/openai/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "") + "/v1"

# Initialize the client based on the API type
base_url = AZURE_API_ENDPOINT if OAI_API_TYPE != "openai" else OPENAI_API_BASE
api_key = AZURE_API_KEY if OAI_API_TYPE != "openai" else OPENAI_API_KEY

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

def create_finetune_sft(training_file_id: str, model: str, suffix: str = None, hyperparameters: dict = None):
    """
    Kick off a supervised fine-tuning (SFT) job using the deprecated `hyperparameters` field.

    Args:
        training_file_id (str): The file ID of the training dataset.
        model (str): The base model to fine-tune.
        suffix (str, optional): A suffix for the fine-tuned model name.
        hyperparameters (dict, optional): A dictionary of hyperparameters with keys:
            - batch_size (int or "auto"): The batch size for training.
            - learning_rate_multiplier (float or "auto"): The learning rate multiplier.
            - n_epochs (int or "auto"): The number of epochs for training.

    Returns:
        str: The ID of the fine-tuning job, or None if creation failed.
    """
    try:
        # Prepare the payload for the fine-tuning job
        payload = {
            "training_file": training_file_id,
            "model": model,
            "suffix": suffix,
        }

        # Add hyperparameters only if provided
        if hyperparameters is not None:
            payload["hyperparameters"] = hyperparameters

        # Create the fine-tuning job
        finetune_job = client.fine_tuning.jobs.create(**payload)
        fine_tune_id = finetune_job.id
        print(f"Supervised fine-tuning job created successfully with ID: {fine_tune_id}")
        return fine_tune_id
    except Exception as e:
        print(f"Failed to create supervised fine-tuning job: {e}")
        return None


def list_finetunes():
    """
    List all fine-tuning jobs.

    Returns:
        list: A list of fine-tuning jobs with their details.
    """
    try:
        response = client.fine_tuning.jobs.list()
        fine_tunes = response.get("data", [])
        print(f"Found {len(fine_tunes)} fine-tuning jobs.")
        return fine_tunes
    except Exception as e:
        print(f"Failed to fetch fine-tuning jobs: {e}")
        return []


def get_finetune_status(fine_tune_id: str):
    """
    Get the status and summary of a specific fine-tuning job.

    Args:
        fine_tune_id (str): The ID of the fine-tuning job.

    Returns:
        dict: A dictionary containing the status and details of the fine-tuning job.
    """
    try:
        finetune_job = client.fine_tuning.jobs.retrieve(fine_tune_id)
        print(f"Status for Fine-tuning Job {fine_tune_id}: {finetune_job.status}")
        print(f"Details: {finetune_job}")
        return finetune_job.status
    except Exception as e:
        print(f"Failed to fetch fine-tuning job details: {e}")
        return {}


def print_finetune_details(fine_tune_id: str):
    """
    Retrieve and print the details of a specific fine-tuning job in a well-formatted table.

    Args:
        fine_tune_id (str): The ID of the fine-tuning job.

    Returns:
        dict: A dictionary containing the details of the fine-tuning job.
    """
    try:
        # Retrieve the fine-tuning job details
        finetune_job = client.fine_tuning.jobs.retrieve(fine_tune_id)

        # Extract relevant details
        details = {
            "ID": finetune_job.id,
            "Status": finetune_job.status,
            "Model": finetune_job.model,
            "Fine-Tuned Model": finetune_job.fine_tuned_model or "Not available",
            "Created At": pd.to_datetime(finetune_job.created_at, unit="s"),
            "Finished At": pd.to_datetime(finetune_job.finished_at, unit="s") if finetune_job.finished_at else "Not finished",
            "Training File": finetune_job.training_file,
            "Validation File": finetune_job.validation_file or "Not provided",
            "Batch Size": finetune_job.hyperparameters.batch_size,
            "Learning Rate Multiplier": finetune_job.hyperparameters.learning_rate_multiplier,
            "Epochs": finetune_job.hyperparameters.n_epochs,
            "Trained Tokens": finetune_job.trained_tokens or "Not available",
            "Estimated Finish": pd.to_datetime(finetune_job.estimated_finish, unit="s") if finetune_job.estimated_finish else "Not available",
            "Error": finetune_job.error.message if finetune_job.error and finetune_job.error.message else "None",
        }

        # Add user-provided suffix if it exists (OpenAI-specific)
        if hasattr(finetune_job, "user_provided_suffix"):
            details["User-Provided Suffix"] = finetune_job.user_provided_suffix or "None"

        # Convert details to a DataFrame for better formatting
        df = pd.DataFrame(details.items(), columns=["Attribute", "Value"])

        # Print the details as a table
        print("\nFine-Tuning Job Details:")
        print(df.to_string(index=False))

        return finetune_job
    except Exception as e:
        print(f"Failed to fetch fine-tuning job details: {e}")
        return {}