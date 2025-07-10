import requests
import time
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for percentile calculations
import matplotlib.patches as mpatches

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# API keys and endpoint
OAI_API_TYPE = os.getenv("OAI_API_TYPE", "azure").lower()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT") + "/openai"
API_VERSION = os.getenv("API_VERSION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


def get_api_config():
    """
    Get the appropriate API configuration based on the OAI_API_TYPE environment variable.

    Returns:
        dict: A dictionary containing the base URL and headers for the API.
    """
    if OAI_API_TYPE == "openai":
        return {
            "base_url": OPENAI_API_BASE + "/v1",
            "headers": {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        }
    else:
        return {
            "base_url": AZURE_API_ENDPOINT,
            "headers": {"api-key": AZURE_API_KEY},
        }


async def create_eval( name: str, grader_model: str, pass_threshold: float):
    """
    Create an evaluation with the given parameters.

    Args:
        pass_threshold (float): The pass threshold for the evaluation.
        grader_model (str): The grader model to use.
        name (str): The name of the evaluation.

    Returns:
        str: The ID of the created evaluation, or None if creation failed.
    """
    api_config = get_api_config()

    GRADER_PROMPT = """
        "You are a mathematical evaluator. Given a reference target number, a list of input numbers, and a model's output (an arithmetic expression and its reported result), your task is to evaluate the correctness and closeness of the model's answer.\n\nInput values are passed in as **strings**, including the number list and target. You must:\n1. Convert the `target` string to a number.\n2. Convert the `numbers` string (e.g., '[100, 75, 50, 25, 6, 3]') into an actual list of numbers.\n3. Parse and validate the `output_expression` — ensure it is a valid arithmetic expression.\n4. Evaluate the expression yourself and confirm it matches the model's reported `output_result`.\n5. Check that **all input numbers are used exactly once** — no missing, extra, or repeated numbers.\n6. Compare the evaluated result with the target and assign a score.\n\nScoring Rules:\n- 5: Expression is valid, numbers used exactly once, result matches target.\n- 4: Off by ±1, but everything else is correct.\n- 3: Off by ±2 to ±5, with correct number usage and syntax.\n- 2: Off by more than 5, but expression is valid and uses all numbers.\n- 1: Minor issues — e.g., result mismatch or slightly incorrect number usage.\n- 0: Major issues — invalid expression, incorrect number usage, or wrong result.\n\nOutput Format:\nScore: <0 - 5>\nReasoning: <brief justification>\n\nOnly respond with the score and reasoning."        
        """


    response = await asyncio.to_thread(
        requests.post,
        f'{api_config["base_url"]}/evals',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
        json={
            'name': name,
            'data_source_config': {
                "type": "custom",
                "include_sample_schema": True,
                "item_schema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "nums": {"type": "string"}
                    }
                }
            },
            'testing_criteria': [
                {
                    "name": "custom grader",
                    "type": "score_model",
                    "model": grader_model,
                    "input": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": {
                                "type": "input_text",
                                "text": GRADER_PROMPT
                            }
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": {
                                "type": "input_text",
                                "text": "**target** : {{item.target}}\n\n**numbers**: {{item.nums}}\n\n**output**: {{sample.output_text}}"
                            }
                        }
                    ],
                    "pass_threshold": pass_threshold,
                    "range": [0.0, 5.0]
                }

            ]
        }
    )

    if response.status_code in (200, 201):
        eval_id = response.json().get('id')
        print(f"Evaluation created successfully with ID: {eval_id}")
        return eval_id
    else:
        print(f"Failed to create evaluation. Status Code: {response.status_code}")
        print(response.text)
        return None


async def create_eval_run(eval_id, file_id, model_deployment=None, eval_run_name=None, use_sample=True, system_prompt=""):
    """
    Create an evaluation run for a specific model deployment or a custom evaluation run.

    Args:
        eval_id (str): The evaluation ID.
        file_id (str): The file ID to use for the evaluation.
        model_deployment (str, optional): The model deployment name. Required if `use_sample` is True.
        eval_run_name (str, optional): The evaluation run name. Required if `use_sample` is False.
        use_sample (bool, optional): Whether to use the sample-based evaluation. Defaults to True.
        system_prompt (str): The system prompt to use for the evaluation run.

    Returns:
        None
    """
    api_config = get_api_config()

    # system_prompt is now mandatory, so no default assignment

    # Construct the JSON payload based on the `use_sample` flag
    if use_sample:
        if not model_deployment:
            raise ValueError("`model_deployment` is required when `use_sample` is True.")
        payload = {
            "name": f"sc_{model_deployment}",
            "data_source": {
                "type": "completions",
                "source": {"type": "file_id", "id": file_id},
                "input_messages": {
                    "type": "template",
                    "template": [
                        {"type": "message", "role": "developer", "content": {"type": "input_text", "text": system_prompt}},
                        {"type": "message", "role": "user", "content": {"type": "input_text", "text": "target: {{item.target}}\nnumbers: {{item.nums}}"}}
                    ]
                },
                "model": model_deployment,
                "sampling_params": {
                    "max_completions_tokens": 20000
                }
            },
        }
    else:
        if not eval_run_name:
            raise ValueError("`eval_run_name` is required when `use_sample` is False.")
        payload = {
            "name": eval_run_name,
            "data_source": {
                "type": "jsonl",
                "source": {"type": "file_id", "id": file_id}
            },
        }

    # Make the API request
    response = await asyncio.to_thread(
        requests.post,
        f'{api_config["base_url"]}/evals/{eval_id}/runs',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
        json=payload)

    # print response code and error message if any
    if response.status_code not in (200, 201):
        print(f"Failed to create evaluation run. Status Code: {response.status_code}")
        print(response.text)
        return None

    # Log the response status
    if use_sample:
        print(f"Create Eval Run Status for {model_deployment}: {response.status_code}")
    else:
        print(f"Create Eval Run Status for {eval_run_name}: {response.status_code}")


# Main function to handle multiple deployments
async def run_evaluations(eval_id, file_id, deployments):
    for deployment in deployments:
        await create_eval_run(eval_id, file_id, model_deployment=deployment)
    print(f"Evaluation ID: {eval_id}")


async def get_eval_runs_list(eval_id: str) -> list:
    """
    Fetch the list of evaluation runs for a given evaluation ID.

    Args:
        eval_id (str): The evaluation ID.

    Returns:
        list: A list of evaluation runs with their details.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}/runs',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    print(f"Get Evaluation Runs: {eval_id}")
    content = response.json().get('data', None)
    list_runs = []

    if content:
        for run in content:
            r = {
                'id': run.get('id', None),
                'name': run.get('name', None),
                'status': run.get('status', None),
                'model': run.get('model', None),
            }
            result = run.get('result_counts', None)
            if result:
                passed = result.get('passed', 0)
                errored = result.get('errored', 0)
                failed = result.get('failed', 0)
                total = result.get('total', 0)
                pass_percentage = round((passed * 100) / (passed + failed), 2) if (passed + failed) > 0 else 0
                error_percentage = round((errored * 100) / total, 2) if total > 0 else 0
                # abs_pass_percentage = round((passed * 100) / total, 2) if total > 0 else 0
                r['pass_percentage'] = pass_percentage
                r['error_percentage'] = error_percentage
                # r['abs_pass_percentage'] = abs_pass_percentage
            list_runs.append(r)

    return list_runs


async def get_eval_details(eval_id: str) -> dict:
    """
    Fetch the details of a specific evaluation.

    Args:
        eval_id (str): The evaluation ID.

    Returns:
        dict: A dictionary containing evaluation details, including the name.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch evaluation details for ID: {eval_id}. Status Code: {response.status_code}")
        return {"name": f"Unknown Evaluation ({eval_id})"}


async def display_evaluation_summary(eval_ids: list):
    """
    Fetch and display a summary of evaluation runs for a list of evaluation IDs, including a horizontal bar chart
    showing both pass percentage and absolute pass percentage.

    Args:
        eval_ids (list): A list of evaluation IDs.
    """
    all_eval_runs = []
    eval_id_to_name = {}
    eval_id_to_color = {}

    # Assign unique colors for each evaluation ID
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    for i, eval_id in enumerate(eval_ids):
        eval_id_to_color[eval_id] = colors[i % len(colors)]

    # Fetch evaluation runs and details for each evaluation ID
    for eval_id in eval_ids:
        eval_runs = await get_eval_runs_list(eval_id)

        # Fetch evaluation details using the helper method
        eval_details = await get_eval_details(eval_id)
        eval_name = eval_details.get('name', f'Unknown Evaluation ({eval_id})')
        eval_id_to_name[eval_id] = eval_name

        # Add evaluation ID to each run for color coding
        for run in eval_runs:
            run['eval_id'] = eval_id
            all_eval_runs.append(run)

    # Combine all evaluation runs into a single DataFrame
    if all_eval_runs:
        df = pd.DataFrame(all_eval_runs)
        df = df[['id', 'name', 'model', 'status', 'pass_percentage', 'error_percentage', 'eval_id']]  # Select relevant columns
        df['eval_name'] = df['eval_id'].map(eval_id_to_name)  # Map eval_id to eval_name
        df['model'] = df['model'].str[:15]  # Truncate model names to 15 characters
        df = df.sort_values(by=['pass_percentage'], ascending=[False])  # Sort by pass_percentage descending

        print("\n" + "=" * 50)
        print("Combined Evaluation Summary")
        print("=" * 50)
        print(df.to_string(index=False, header=["Run ID", "Run Name", "Model", "Status", "Pass Percentage (%)", "Error Percentage (%)", "Evaluation ID", "Evaluation Name"]))
        print("=" * 50)

        # Dynamically adjust the figure height based on the number of rows
        num_rows = len(df)
        fig_height = max(3, num_rows * 0.75)  # Set a minimum height of 3 and scale with 0.5 per row

        # Create a horizontal bar chart with rows sorted by pass percentage across all eval_ids
        plt.figure(figsize=(12, fig_height))

        df['display_label'] = df['model'].where(
            (df['model'].str.strip() != '') & (df['model'] != 'None') & (df['model'].notna()),
            df['name']
        )

        # Plot pass percentage
        plt.barh(
            df['display_label'], 
            df['pass_percentage'], 
            color=[eval_id_to_color[eval_id] for eval_id in df['eval_id']], 
            edgecolor='black', 
            label='Pass Percentage (%)'
        )

        # Add legend for eval colors
        legend_handles = [
            mpatches.Patch(color=color, label=f"{eval_id_to_name[eval_id]} ({eval_id})")
            for eval_id, color in eval_id_to_color.items()
        ]
        plt.legend(handles=legend_handles, title="Evaluation", loc='lower right')

        plt.xlabel('Percentage (%)')
        plt.ylabel('Model')
        plt.title("Pass Percentage by Model Across Evaluations")
        plt.xlim(0, 100)  # Set x-axis scale explicitly to 0-100
        plt.gca().invert_yaxis()  # Invert y-axis to show the highest percentage at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("\n" + "=" * 50)
        print("No evaluation runs found for the provided Evaluation IDs.")
        print("=" * 50)


async def list_evaluations() -> list:
    """
    Fetch the list of all evaluations.

    Returns:
        list: A list of evaluations with their details.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        content = response.json().get('data', [])
        print("Fetched evaluations successfully.")
        return content
    else:
        print(f"Failed to fetch evaluations. Status Code: {response.status_code}")
        return []


async def delete_evaluation(eval_id: str) -> bool:
    """
    Delete an evaluation by its ID.

    Args:
        eval_id (str): The evaluation ID to delete.

    Returns:
        bool: True if the evaluation was deleted successfully, False otherwise.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.delete,
        f'{api_config["base_url"]}/evals/{eval_id}',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 204:
        print(f"Evaluation with ID {eval_id} deleted successfully.")
        return True
    else:
        print(f"Failed to delete evaluation with ID: {eval_id}. Status Code: {response.status_code}")
        return False


async def delete_all_evaluations():
    """
    Delete all evaluations.

    Returns:
        None
    """
    # Fetch all evaluations
    evaluations = await list_evaluations()

    if not evaluations:
        print("No evaluations found to delete.")
        return

    # Iterate through each evaluation and delete it
    for evaluation in evaluations:
        eval_id = evaluation.get('id')
        if eval_id:
            success = await delete_evaluation(eval_id)
            if success:
                print(f"Deleted evaluation with ID: {eval_id}")
            else:
                print(f"Failed to delete evaluation with ID: {eval_id}")


async def get_eval_run_output_items(eval_id: str, run_id: str) -> list:
    """
    Fetch the output items for a specific evaluation run and extract the result scores.

    Args:
        eval_id (str): The evaluation ID.
        run_id (str): The run ID.

    Returns:
        list: A list of scores for the output items.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}/runs/{run_id}/output_items',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        output_items = response.json().get('data', [])
        scores = []
        for item in output_items:
            results = item.get('results', [])
            for result in results:
                score = result.get('score')
                if score is not None:
                    scores.append(score)
        return scores
    else:
        print(f"Failed to fetch output items for run {run_id}. Status Code: {response.status_code}")
        return []
    
import json

def convert_to_eval_format_separate_fields(input_path, output_path):
    converted_records = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            if not messages or not isinstance(messages[0], dict):
                continue  # skip malformed

            prompt = messages[0].get("content", "").strip()
            final_answer_val = str(data.get("final_answer", ""))  # Convert to string
            solution_val = str(data.get("solution", ""))  # Convert to string

            record = {
                "item": {
                    "input": prompt,
                    "final_answer": final_answer_val,
                    "solution": solution_val
                }
            }
            converted_records.append(record)

    # Write to output file in .jsonl format
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in converted_records:
            outfile.write(json.dumps(item) + '\n')

    print(f"✅ Converted {len(converted_records)} records to {output_path}")

async def display_evaluation_details(eval_id: str, eval_run_id: str, status_filter: str = None, max_records: int = 10):
    """
    Display detailed information for a specific evaluation run, including sample data, scores, and status.

    Args:
        eval_id (str): The evaluation ID.
        eval_run_id (str): The evaluation run ID.
        status_filter (str, optional): Filter for "pass" or "fail" samples. Defaults to None (no filter).
        max_records (int, optional): Maximum number of records to display. Defaults to 10.
    """
    api_config = get_api_config()

    # Fetch output items for the evaluation run
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}/runs/{eval_run_id}/output_items',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code != 200:
        print(f"Failed to fetch output items for run {eval_run_id}. Status Code: {response.status_code}")
        return

    output_items = response.json().get('data', [])
    if not output_items:
        print(f"No output items found for evaluation run {eval_run_id}.")
        return

    # Filter output items based on status (pass/fail) if specified
    if status_filter:
        output_items = [
            item for item in output_items
            if item.get('status', '').lower() == status_filter.lower()
        ]

    # Limit the number of records to display
    output_items = output_items[:max_records]

    # Collect details for each sample
    records = []
    for i, item in enumerate(output_items, start=1):
        datasource_item = item.get('datasource_item', {})
        target = datasource_item.get('target', 'N/A')
        nums = datasource_item.get('nums', 'N/A')

        output = item.get('sample', {}).get('output', [])
        if output and isinstance(output, list):
            assistant_output = next((o.get('content', 'N/A') for o in output if o.get('role') == 'assistant'), 'N/A')
            try:
                assistant_data = json.loads(assistant_output)
                expression = assistant_data.get('expression', 'N/A')
                result = assistant_data.get('result', 'N/A')
                # Try to round result if it's a number
                try:
                    result_num = float(result)
                    result = f"{result_num:.2f}"
                except (ValueError, TypeError):
                    pass
            except (json.JSONDecodeError, TypeError):
                expression = "N/A"
                result = "N/A"
        else:
            expression = "N/A"
            result = "N/A"

        results = item.get('results', [{}])
        score = results[0].get('score', 'N/A') if results else 'N/A'
        status = item.get('status', 'N/A')

        records.append({
            "Sample": i,
            "Target": target,
            "Numbers": nums,
            "Result": result,
            "Expression": expression,
            "Score": score,
            "Status": status
        })

    # Display as a pretty pandas DataFrame
    print("\n" + "=" * 50)
    print(f"Evaluation Details for Run ID: {eval_run_id}")
    print("=" * 50)
    df = pd.DataFrame(records)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No records to display.")
    print("=" * 50)
    print(f"Displayed {len(records)} records (filtered by status: {status_filter}).")
    print("=" * 50)

#######

import pandas as pd
import json
import asyncio
import requests
from tabulate import tabulate

def parse_score(score):
    try:
        return float(score)
    except (TypeError, ValueError):
        return None

def get_sample_key(item):
    ds = item.get("datasource_item", {})
    target = ds.get("target")
    nums = ds.get("nums")
    return f"{target}::{','.join(map(str, nums))}" if target is not None and nums else None

async def fetch_eval_outputs(eval_id, eval_run_id):
    api_config = get_api_config()
    params = {'limit': 100}
    if OAI_API_TYPE == "azure":
        params['api-version'] = API_VERSION
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}/runs/{eval_run_id}/output_items',
        params=params,
        headers=api_config["headers"],
    )
    if response.status_code != 200:
        print(f"❌ Failed to fetch run: {eval_run_id}. Code: {response.status_code}")
        return []
    return response.json().get('data', [])

def extract_key_fields(item):
    datasource = item.get("datasource_item", {})
    output = item.get("sample", {}).get("output", [])
    result, expr = "N/A", "N/A"
    if output and isinstance(output, list):
        assistant_msg = next((o.get('content') for o in output if o.get("role") == "assistant"), None)
        if assistant_msg:
            try:
                parsed = json.loads(assistant_msg)
                result = parsed.get("result", "N/A")
                expr = parsed.get("expression", "N/A")
            except Exception:
                pass
    # Set score to 0.0 if result is 'N/A' or not a number
    try:
        result_num = float(result)
    except Exception:
        result_num = None
    results = item.get("results", [])
    score = None
    if results and isinstance(results, list) and len(results) > 0:
        score = parse_score(results[0].get("score"))
    else:
        score = 0.0
    if result_num is None:
        score = 0.0
    return {
        "target": datasource.get("target"),
        "nums": datasource.get("nums"),
        "result": result_num if result_num is not None else "N/A",
        "expression": expr,
        "score": score,
        "status": item.get("status", "N/A").lower()
    }

def classify_comparison(run1, run2):
    s1, s2 = run1["score"], run2["score"]
    st1, st2 = run1["status"], run2["status"]

    # Improvement cases
    if st1 == "fail" and st2 == "pass":
        return "improved"
    if s1 is None and s2 is not None:
        return "improved"
    if s1 is not None and s2 is not None and s2 > s1:
        return "improved"

    # Degradation cases
    if st1 == "pass" and st2 == "fail":
        return "degraded"
    if s1 is not None and s2 is not None and s2 < s1:
        return "degraded"

    # Same
    if s1 == s2 and st1 == st2:
        return "same"

    return "other"

def safe_round(val):
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return val

async def compare_eval_runs_generic(
    eval_id_1,
    run_id_1,
    eval_id_2,
    run_id_2,
    label_1="Run 1",
    label_2="Run 2",
    filter_by="all",  # Options: "improved", "degraded", "same", "all"
    max_records=20
):
    run1_items = await fetch_eval_outputs(eval_id_1, run_id_1)
    run2_items = await fetch_eval_outputs(eval_id_2, run_id_2)

    # Map both runs by composite key
    run1_map = {get_sample_key(item): extract_key_fields(item) for item in run1_items}
    run2_map = {get_sample_key(item): extract_key_fields(item) for item in run2_items}

    # Intersect keys to compare only common records
    common_keys = set(run1_map.keys()) & set(run2_map.keys())
    comparisons = []

    for i, key in enumerate(sorted(common_keys)):
        r1 = run1_map[key]
        r2 = run2_map[key]
        comp_type = classify_comparison(r1, r2)

        if filter_by != "all" and comp_type != filter_by:
            continue

        target, nums = key.split("::")
        comparisons.append({
            "Sample": i + 1,
            "Target": int(target),
            "Numbers": str(r1["nums"]),
            f"{label_1} Result": safe_round(r1["result"]),
            f"{label_2} Result": safe_round(r2["result"]),
            f"{label_1} Expr": r1["expression"],
            f"{label_2} Expr": r2["expression"],
            f"{label_1} Score": safe_round(r1["score"]),
            f"{label_2} Score": safe_round(r2["score"]),
            f"{label_1} Status": r1["status"],
            f"{label_2} Status": r2["status"],
            "Change": comp_type
        })

    if not comparisons:
        print(f"✅ No results matching filter: '{filter_by}'.")
        return

    df = pd.DataFrame(comparisons[:max_records])
    total = len(comparisons)
    type_counts = df['Change'].value_counts().to_dict()

    summary_stats = {}
    for comp_type in ['improved', 'degraded', 'same', 'other']:
        count = type_counts.get(comp_type, 0)
        percent = (count / total) * 100 if total > 0 else 0
        summary_stats[comp_type] = {
            "count": count,
            "percent": percent
        }

    # Prepare filter summary for header
    if filter_by in summary_stats:
        filter_count = summary_stats[filter_by]["count"]
        filter_percent = summary_stats[filter_by]["percent"]
        filter_summary = f"{filter_by.capitalize()} = {filter_count} ({filter_percent:.1f}%)"
    elif filter_by == "all":
        filter_summary = f"Total = {total}"
    else:
        filter_summary = f"{filter_by.capitalize()} = 0 (0.0%)"

    # Print the counts and percentages
    print(f"\n📊 Evaluation Comparison: {label_1} vs {label_2} | Filter: '{filter_by}' | {filter_summary}\n")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

    # Optionally, print the full summary for all types below the table
    # print("\nSummary by Change Type:")
    # for comp_type, stats in summary_stats.items():
    #     print(f"  {comp_type.capitalize():<9}: {stats['count']:>3} ({stats['percent']:5.1f}%)")



# def download eval run output as csv
def download_eval_run_output(eval_id: str, run_id: str, output_path: str):
    """
    Download the output items of an evaluation run and save them to a CSV file.

    Args:
        eval_id (str): The evaluation ID.
        run_id (str): The evaluation run ID.
        output_path (str): The path where the CSV file will be saved.

    Returns:
        None
    """
    api_config = get_api_config()
    response = requests.get(
        f'{api_config["base_url"]}/evals/{eval_id}/runs/{run_id}/output_items',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        output_items = response.json().get('data', [])
        df = pd.DataFrame(output_items)
        df.to_csv(output_path, index=False)
        print(f"Output items downloaded successfully to {output_path}")
    else:
        print(f"Failed to download output items. Status Code: {response.status_code}")