import json
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

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


# -------------------------------------------------------------------------------
# --                           Helper Functions                                --
# -------------------------------------------------------------------------------
client = AsyncEvalClient()

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

    GRADER_PROMPT = """
        "You are a mathematical evaluator. Given a reference target number, a list of input numbers, and a model's output (an arithmetic expression and its reported result), your task is to evaluate the correctness and closeness of the model's answer.\n\nInput values are passed in as **strings**, including the number list and target. You must:\n1. Convert the `target` string to a number.\n2. Convert the `numbers` string (e.g., '[100, 75, 50, 25, 6, 3]') into an actual list of numbers.\n3. Parse and validate the `output_expression` — ensure it is a valid arithmetic expression.\n4. Evaluate the expression yourself and confirm it matches the model's reported `output_result`.\n5. Check that **all input numbers are used exactly once** — no missing, extra, or repeated numbers.\n6. Compare the evaluated result with the target and assign a score.\n\nScoring Rules:\n- 5: Expression is valid, numbers used exactly once, result matches target.\n- 4: Off by ±1, but everything else is correct.\n- 3: Off by ±2 to ±5, with correct number usage and syntax.\n- 2: Off by more than 5, but expression is valid and uses all numbers.\n- 1: Minor issues — e.g., result mismatch or slightly incorrect number usage.\n- 0: Major issues — invalid expression, incorrect number usage, or wrong result.\n\nOutput Format:\nScore: <0 - 5>\nReasoning: <brief justification>\n\nOnly respond with the score and reasoning."        
        """

    # Create the evaluation using the SDK
    # Note: The data_source_config and testing_criteria are defined inline
    # Adjust the item_schema as needed for your specific evaluation requirements
    return await client.create_eval_sdk(
            name=name,
            data_source_config={
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
            testing_criteria=[
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
        )

python_grader_source = """
import json, re, ast


def safe_eval(e):
    return _eval(ast.parse(e, mode='eval').body)


def _eval(n):
    if isinstance(n, ast.Constant):
        return n.value

    if isinstance(n, ast.BinOp) and type(n.op) in {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }:
        return {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
        }[type(n.op)](_eval(n.left), _eval(n.right))

    if isinstance(n, ast.UnaryOp) and type(n.op) in {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }:
        return {
            ast.UAdd: lambda a: +a,
            ast.USub: lambda a: -a,
        }[type(n.op)](_eval(n.operand))

    raise ValueError('bad expr')


def grade(sample, item) -> float:
    try:
        expr = sample['output_json']['expression']
        expr_val = safe_eval(expr)

        # Check numbers used
        if sorted(map(int, re.findall(r'-?\d+', expr))) != sorted(
            map(int, json.loads(item['nums']))
        ):
            return 0

        sr = int(float(sample['output_json']['result']))
        it = int(float(item['target']))

        if expr_val != sr:
            return 1
        if sr == it:
            return 5
        if abs(sr - it) <= 1:
            return 4
        if abs(sr - it) <= 5:
            return 3
        return 2

    except:
        return 0
"""

async def create_eval_python_grader(name: str, pass_threshold: float):
    """
    Create an evaluation with the given parameters.

    Args:
        pass_threshold (float): The pass threshold for the evaluation.
        name (str): The name of the evaluation.

    Returns:
        str: The ID of the created evaluation, or None if creation failed.
    """
    return await client.create_eval_sdk(
            name=name,
            data_source_config={
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
            testing_criteria=[
                {
                    "name": "custom grader",
                    "type": "python",
                    "source": python_grader_source,
                    "pass_threshold": pass_threshold
                }
            ]
        )

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

    # Make the SDK request
    return await client.create_eval_run_sdk(
        eval_id=eval_id,
        name=payload["name"],
        data_source=payload["data_source"]
        # metadata can be added here if needed
    )


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

    run_list = await client.get_eval_run_list_sdk(eval_id)

    print(f"Get Evaluation Runs: {eval_id}")
    list_runs = []

    if run_list:
        for run in run_list:
            r = {
                'id': run.id,
                'name': run.name,
                'status': run.status,
                'model': run.model,
            }
            result = run.result_counts
            if result:
                passed = result.passed
                errored = result.errored
                failed = result.failed
                total = result.total
                pass_percentage = round((passed * 100) / (passed + failed), 2) if (passed + failed) > 0 else 0
                error_percentage = round((errored * 100) / total, 2) if total > 0 else 0
                # abs_pass_percentage = round((passed * 100) / total, 2) if total > 0 else 0
                r['pass_percentage'] = pass_percentage
                r['error_percentage'] = error_percentage
                # r['abs_pass_percentage'] = abs_pass_percentage
            list_runs.append(r)

    return list_runs


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
        eval_details = await client.get_eval_sdk(eval_id)
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


async def delete_all_evaluations():
    """
    Delete all evaluations.

    Returns:
        None
    """
    # Fetch all evaluations
    evaluations = await client.get_eval_list_sdk()

    if not evaluations:
        print("No evaluations found to delete.")
        return

    # Iterate through each evaluation and delete it
    for evaluation in evaluations:
        eval_id = evaluation.id
        if eval_id:
            await client.delete_eval_sdk(eval_id)


async def get_eval_run_output_items(eval_id: str, run_id: str) -> list:
    """
    Fetch the output items for a specific evaluation run and extract the result scores.

    Args:
        eval_id (str): The evaluation ID.
        run_id (str): The run ID.

    Returns:
        list: A list of scores for the output items.
    """
    output_items = await client.get_eval_run_output_items_sdk(eval_id, run_id)

    scores = []
    for item in output_items:
        results = item.results
        for result in results:
            score = result.score
            if score is not None:
                scores.append(score)
    return scores
    
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

    # Fetch output items for the evaluation run
    output_items = await client.get_eval_run_output_items_sdk(eval_id, eval_run_id) 

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
from tabulate import tabulate

def parse_score(score):
    try:
        return float(score)
    except (TypeError, ValueError):
        return None

def get_sample_key(item):
    ds = item.datasource_item
    target = ds.get("target")
    nums = ds.get("nums")
    return f"{target}::{','.join(map(str, nums))}" if target is not None and nums else None

def extract_key_fields(item):
    datasource = getattr(item, "datasource_item", {})
    output = getattr(item.sample, "output", []) if item.sample else []
    result, expr = "N/A", "N/A"

    # Extract result and expression from assistant's output
    if output and isinstance(output, list):
        assistant_msg = next((o.content for o in output if getattr(o, "role", None) == "assistant"), None)
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

    # Extract score from results
    results = getattr(item, "results", [])
    score = 0.0
    if results and isinstance(results, list) and len(results) > 0:
        try:
            score = float(results[0].get("score", 0.0))
        except Exception:
            score = 0.0

    # Override score if result is invalid
    if result_num is None:
        score = 0.0
    return {
        "target": datasource.get("target", "N/A"),
        "nums": datasource.get("nums", "N/A"),
        "result": result_num if result_num is not None else "N/A",
        "expression": expr,
        "score": score,
        "status": getattr(item, "status", "N/A").lower()
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
    run1_items = await client.get_eval_run_output_items_sdk(eval_id_1, run_id_1)
    run2_items = await client.get_eval_run_output_items_sdk(eval_id_2, run_id_2)

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
    output_items = client.get_eval_run_output_items_sdk(eval_id, run_id)

    if not output_items:
        print(f"No output items found for evaluation run {run_id}.")
        return

    df = pd.DataFrame(output_items)
    df.to_csv(output_path, index=False)
    print(f"Output items downloaded successfully to {output_path}")