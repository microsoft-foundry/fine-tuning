# Import required libraries
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from the .env file
load_dotenv(override=True)

# API keys and endpoint
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT", "") + "/openai/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_BASE", "") + "/v1"

async def upload_file(file_name: str, file_path: str, purpose: str = "fine-tune") -> str:
    """
    Upload a file to either Azure or OpenAI based on the configuration in the .env file.
    If a file with the same name already exists, return its ID instead of uploading again.

    Args:
        file_name (str): The name of the file to upload.
        file_path (str): The path to the file to upload.
        purpose (str): The purpose of the file upload (e.g., "fine-tune", "evals"). Defaults to "fine-tune".

    Returns:
        str: The file ID of the uploaded or existing file, or an empty string if the operation fails.
    """
    use_openai = os.getenv("OAI_API_TYPE", "azure").lower() == "openai"

    base_url = AZURE_API_ENDPOINT if not use_openai else OPENAI_API_ENDPOINT
    api_key = AZURE_API_KEY if not use_openai else OPENAI_API_KEY

    print(f"Using {'OpenAI' if use_openai else 'Azure'} API for file upload...")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Check if the file already exists
    list_response = await client.files.list()

    files = list_response.data
    for file in files:
        if file.filename == file_name:
            print(f"File '{file_name}' already exists on OpenAI. Returning existing file ID.")
            return file.id

    # File does not exist, proceed with upload
    try:
        with open(file_path, 'rb') as f:
            response = await client.files.create(
                file=f,
                purpose= purpose, # type: ignore
            )

        print(f"File uploaded successfully to {'OpenAI' if use_openai else 'Azure'}.")
        return response.id
    except Exception as e:
        print(f"Failed to upload file to {'OpenAI' if use_openai else 'Azure'}: {e}")
        return ''