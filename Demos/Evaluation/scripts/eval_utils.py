import json
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import json
from tabulate import tabulate

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from the .env file
load_dotenv()

# API keys and endpoint
OAI_API_TYPE = os.getenv("OAI_API_TYPE", "azure").lower()
AZURE_API_KEY = os.getenv("AZURE_API_KEY", None)
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT", "") + "/openai/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "") + "/v1"

# -------------------------------------------------------------------------------
# --                           Eval Client Class                               --
# -------------------------------------------------------------------------------
class AsyncEvalClient:

    def __init__(self):
        """ 
        Initialize the AsyncEvalClient with the appropriate OpenAI client based on the API type.
        """
        params = {"aoai-evals": "preview"} if OAI_API_TYPE != "openai" else None
        base_url = AZURE_API_ENDPOINT if OAI_API_TYPE != "openai" else OPENAI_API_BASE
        api_key = AZURE_API_KEY if OAI_API_TYPE != "openai" else OPENAI_API_KEY

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_query=params
        )

    # ---------------------------- Evaluation File Upload ----------------------------
    async def upload_file(self, file_name: str, file_path: str, purpose: str = "evals") -> str:
        """
        Upload a file to either Azure or OpenAI based on the configuration in the .env file.
        If a file with the same name already exists, return its ID instead of uploading again.

        Args:
            file_name (str): The name of the file to upload.
            file_path (str): The path to the file to upload.
            purpose (str): The purpose of the file upload (e.g., "fine-tune", "evals"). Defaults to "evals".

        Returns:
            str: The file ID of the uploaded or existing file, or an empty string if the operation fails.
        """

        # Check if the file already exists
        list_response = await self.client.files.list()

        files = list_response.data
        for file in files:
            if file.filename == file_name:
                print(f"File '{file_name}' already exists. Returning existing file ID.")
                return file.id

        # File does not exist, proceed with upload
        try:
            with open(file_path, 'rb') as f:
                response = await self.client.files.create(
                    file=f,
                    purpose= purpose, # type: ignore
                )

            print(f"File uploaded successfully to {'OpenAI' if OAI_API_TYPE == 'openai' else 'Azure'}.")
            return response.id
        except Exception as e:
            print(f"Failed to upload file to {'OpenAI' if OAI_API_TYPE == 'openai' else 'Azure'}: {e}")
            return ''

    # ---------------------------- Evaluation Methods ----------------------------
    # Create an evaluation using the SDK
    async def create_eval_sdk(self, name, data_source_config, testing_criteria):
        """
        Create a new evaluation using the SDK.
        Parameters:
            name (str): The name of the evaluation.
            data_source_config (dict): Configuration for the data source.
            testing_criteria (list): List of testing criteria for the evaluation.
        Returns:
            str: The ID of the created evaluation, or None if creation failed.
        """
        try:
            response = await self.client.evals.create(
                name=name,
                data_source_config=data_source_config,
                testing_criteria=testing_criteria
            )

            eval_id = response.to_dict()["id"]
            print(f"Evaluation created successfully with ID: {eval_id}")
            return eval_id

        except Exception as e:
            print(f"Failed to create evaluation. Error: {e}")
            return None


    # List all evaluations using the SDK
    async def get_eval_list_sdk(self):
        """
        List all evaluations using the SDK.
        Returns:
            list: A list of evaluations, each represented as a dictionary.
        """
        response = await self.client.evals.list()
        print("Fetched evaluations successfully.")
        return response.data


    # Get details of a specific evaluation using the SDK
    async def get_eval_sdk(self, eval_id):
        """
        Get details of a specific evaluation using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation to retrieve.
        Returns:
            dict: A dictionary containing evaluation details, including the name and None if retrieval failed.
        """
        try:
            response = await self.client.evals.retrieve(eval_id=eval_id)
            return response.to_dict()
        except Exception as e:
            print(f"Failed to fetch evaluation details for ID: {eval_id}. Error: {e}")
            return {"name": f"Unknown Evaluation ({eval_id})"}


    # Delete an evaluation using the SDK
    async def delete_eval_sdk(self, eval_id):
        """
        Delete an evaluation using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation to delete.
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            await self.client.evals.delete(eval_id=eval_id)
            print(f"Evaluation with ID {eval_id} deleted successfully.")
            return True
        except Exception as e:
            print(f"Failed to delete evaluation with ID: {eval_id}. Error: {e}")
            return False


    # -------------------------- Evaluation Run Methods --------------------------
    # Create a new evaluation run using the SDK
    async def create_eval_run_sdk(self, eval_id, name, data_source, metadata=None) -> dict:
        """
        Create a new evaluation run using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation to run.
            name (str): The name of the evaluation run.
            data_source (dict): Data source configuration for the evaluation run.
            metadata (dict, optional): Additional metadata for the evaluation run.
        Returns:
            dict: The response from the SDK containing the evaluation run details, or an empty dictionary if creation failed.
        """
        try:
            response = await self.client.evals.runs.create(
                eval_id=eval_id,
                name=name,
                metadata=metadata,
                data_source=data_source
            )
            eval_run_id = response.to_dict().get("id", "Unknown ID")
            print(f"Created evaluation run for {name}: {eval_run_id}")
            return response.to_dict()
        except Exception as e:
            print(f"Failed to create evaluation run. Error: {e}")
            return {}


    # Get a list of evaluation runs for a specific evaluation using the SDK
    async def get_eval_run_list_sdk(self, eval_id) -> list:
        """
        Get a list of evaluation runs for a specific evaluation using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation to retrieve runs for.
        """
        response = await self.client.evals.runs.list(eval_id=eval_id)
        return response.data


    # Get details of a specific evaluation run using the SDK
    async def get_eval_run_sdk(self, eval_id, run_id) -> dict:
        """
        Get details of a specific evaluation run using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the evaluation run to retrieve.
        Returns:
            dict: A dictionary containing evaluation run details, or an empty dictionary if retrieval failed.
        """
        try:
            response = await self.client.evals.runs.retrieve(eval_id=eval_id, run_id=run_id)
            return response.to_dict()
        except Exception as e:
            print(f"Failed to fetch evaluation run details for ID: {run_id}. Error: {e}")
            return {}


    # Get the output items of a specific evaluation run using the SDK
    async def get_eval_run_output_items_sdk(self, eval_id, run_id) -> list:
        """
        Get the output items of a specific evaluation run using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the evaluation run to retrieve output items for.
        Returns:
            list: A list of output items for the evaluation run, or an empty list if retrieval failed.
        """
        try:
            response = await self.client.evals.runs.output_items.list(eval_id=eval_id, run_id=run_id)
            return response.data
        except Exception as e:
            print(f"Failed to fetch output items for evaluation run ID: {run_id}. Error: {e}")
            return []


    # Get the single output item of a specific evaluation run using the SDK
    async def get_eval_run_output_item_sdk(self, eval_id, run_id, item_id) -> dict:
        """
        Get a specific output item of a specific evaluation run using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the evaluation run to retrieve the output item for.
            item_id (str): The ID of the output item to retrieve.
        Returns:
            dict: A dictionary containing the output item details, or an empty dictionary if retrieval failed.
        """
        try:
            response = await self.client.evals.runs.output_items.retrieve(
                eval_id=eval_id,
                run_id=run_id,
                output_item_id=item_id
            )
            return response.to_dict()
        except Exception as e:
            print(f"Failed to fetch output item for evaluation run ID: {run_id}, item ID: {item_id}. Error: {e}")
            return {}


    # Cancel a specific evaluation run using the SDK
    async def cancel_eval_run_sdk(self, eval_id, run_id) -> bool:
        """
        Cancel a specific evaluation run using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the evaluation run to cancel.
        Returns:
            bool: True if cancellation was successful, False otherwise.
        """
        try:
            await self.client.evals.runs.cancel(eval_id=eval_id, run_id=run_id)
            print(f"Evaluation run with ID {run_id} cancelled successfully.")
            return True
        except Exception as e:
            print(f"Failed to cancel evaluation run with ID: {run_id}. Error: {e}")
            return False


    # Delete a specific evaluation run using the SDK
    async def delete_eval_run_sdk(self, eval_id, run_id) -> bool:
        """
        Delete a specific evaluation run using the SDK.
        Parameters:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the evaluation run to delete.
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:    
            await self.client.evals.runs.delete(eval_id=eval_id, run_id=run_id)
            print(f"Evaluation run with ID {run_id} deleted successfully.")
            return True
        except Exception as e:
            print(f"Failed to delete evaluation run with ID: {run_id}. Error: {e}")
            return False