from scripts.client_utils import get_async_openai_client

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
    print("Using Foundry project Entra ID authentication for file upload...")
    client = get_async_openai_client()

    try:
        list_response = await client.files.list()
        for file in list_response.data:
            if file.filename == file_name:
                print(f"File '{file_name}' already exists. Returning existing file ID.")
                return file.id

        with open(file_path, "rb") as f:
            response = await client.files.create(
                file=(file_name, f, "application/jsonl"),
                purpose=purpose,
            )

        print("File uploaded successfully.")
        return response.id
    except Exception as e:
        raise RuntimeError(f"Failed to upload file '{file_path}': {e}") from e
    finally:
        await client.close()