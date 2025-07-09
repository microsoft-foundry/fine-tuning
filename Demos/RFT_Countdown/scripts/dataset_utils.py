# Import required libraries
import requests
import time
import json
import asyncio

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv(override=True)

# API keys and endpoint
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_BASE")

import json

def save_dataset_as_jsonl(dataset, file_path, max_records=None):
    """
    Save a dataset to a JSONL file with all values converted to strings.
    Only includes samples where 'nums' has exactly 4 elements.

    Args:
        dataset: The dataset to save.
        file_path: The path to the output JSONL file.
        max_records: The maximum number of records to save (optional).
    """
    fetch_multiplier = 10
    if max_records:
        dataset = dataset.shuffle(seed=123).select(range(max_records * fetch_multiplier))
    
    count = 0
    with open(file_path, "w", encoding="utf-8") as f:
        for example in dataset:
            nums = example.get("nums", [])
            # Accept both list and string representations of nums
            if isinstance(nums, str):
                try:
                    nums_eval = json.loads(nums)
                except Exception:
                    nums_eval = []
            else:
                nums_eval = nums
            if isinstance(nums_eval, list) and len(nums_eval) == 4:
                stringified_example = {key: str(value) for key, value in example.items()}
                f.write(json.dumps(stringified_example) + "\n")
                count += 1
                if max_records and count >= max_records:
                    break
    print(f"✅ Saved {count} records to {file_path} in evaluation format (only samples with 4 nums)")

def save_dataset_in_eval_format(dataset, file_path, max_records=None):
    """
    Save a dataset to a JSONL file in evaluation format by enclosing each record in an 'item' field
    and converting all values to strings. Only includes samples where 'nums' has exactly 4 elements.

    Args:
        dataset: The dataset to save. Can be a list of dictionaries or a dataset object.
        file_path: The path to the output JSONL file.
        max_records: The maximum number of records to save (optional).
    """
    fetch_multiplier = 5
    if max_records:
        dataset = dataset.shuffle(seed=123).select(range(max_records * fetch_multiplier))

    count = 0
    with open(file_path, "w", encoding="utf-8") as f:
        for example in dataset:
            nums = example.get("nums", [])
            # Accept both list and string representations of nums
            if isinstance(nums, str):
                try:
                    nums_eval = json.loads(nums)
                except Exception:
                    nums_eval = []
            else:
                nums_eval = nums
            if isinstance(nums_eval, list) and len(nums_eval) == 4:
                stringified_example = {key: str(value) for key, value in example.items()}
                formatted_example = {"item": stringified_example}
                f.write(json.dumps(formatted_example) + "\n")
                count += 1
                if max_records and count >= max_records:
                    break
    print(f"✅ Saved {count} records to {file_path} in evaluation format (only samples with 4 nums)")

    
def convert_to_rft_dataset(input_path, output_path, system_prompt, max_records=10):
    """
    Converts the dataset to RFT format for fine-tuning and saves it.

    Args:
        input_path: Path to the input dataset file.
        output_path: Path to save the converted dataset.
        system_prompt: The instruction or system prompt to include in the RFT format.
        max_records: Maximum number of records to process.
    """
    count = 0
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if count >= max_records:
                break
            data = json.loads(line)
            target = str(data.get("target", ""))
            nums = str(data.get("nums", ""))
            
            rft_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{system_prompt}Input:: \n\n Target: \"{target}\" Numbers: \"{nums}\""
                    }
                ],
                "target": f"{target}",
                "nums": f"{nums}"
            }

            outfile.write(json.dumps(rft_record) + '\n')
            count += 1

    print(f"✅ Converted {count} records to RFT format and saved to {output_path}")

