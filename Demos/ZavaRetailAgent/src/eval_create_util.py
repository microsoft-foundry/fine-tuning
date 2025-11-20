"""
Azure OpenAI Evaluation utilities for creating evaluations and runs.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


def create_azure_evaluation(
    client, pass_threshold: float, python_grader_path: str
):
    """
    Create an Azure OpenAI evaluation with custom Python grader.

    Args:
        client: Azure OpenAI client
        pass_threshold: Score threshold for passing the evaluation
        python_grader_path: Path to the Python grader file (required)

    Returns:
        Evaluation object with ID
    """
    print(f"🎯 Creating evaluation with Python grader...")

    # Read the Python grader
    with open(python_grader_path) as f:
        python_grader = f.read()

    # Create the evaluation
    evaluation = client.evals.create(
        name=f"Tool Calling Evaluation - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        data_source_config={
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array"},
                    "tools": {"type": "array"},
                    "expected_output": {"type": "object"},
                },
                "required": ["messages", "tools"],
            },
            "include_sample_schema": True,
        },
        testing_criteria=[
            {
                "type": "python",
                "name": "Tool Use Evaluator",
                "source": python_grader,
                "pass_threshold": pass_threshold,
            }
        ],
    )

    print(f"✅ Evaluation created successfully!")
    print(f"   🆔 Evaluation ID: {evaluation.id}")
    print(f"   📝 Name: {evaluation.name}")
    print(f"   🧮 Grader: Tool Use Evaluator (Python)")
    print(f"   🎯 Pass Threshold: {pass_threshold}")

    return evaluation


def create_evaluation_runs(
    client,
    evaluation_id: str,
    models_to_evaluate: List[str],
    eval_file_id: str,
    tools_file_path: Path,
):
    """
    Create evaluation runs for each model using the same evaluation dataset with sampling parameters.

    Args:
        client: Azure OpenAI client
        evaluation_id: ID of the created evaluation
        models_to_evaluate: List of model names
        eval_file_id: File ID of the evaluation data
        tools_file_path: Path to the tools JSON file

    Returns:
        List of evaluation run objects
    """
    print(f"🚀 Creating evaluation runs for {len(models_to_evaluate)} models...")

    # Load tools from the provided JSON file path
    if not tools_file_path.exists():
        raise FileNotFoundError(f"Tools file not found: {tools_file_path}")

    with open(tools_file_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    print(f"🔧 Loaded {len(tools)} tools from: {tools_file_path.name}")

    eval_runs = []

    print(f"📁 Using evaluation file ID: {eval_file_id}")

    for model in models_to_evaluate:
        print(f"📊 Creating evaluation run for: {model}")

        # Create data source with sampling parameters
        data_source = {
            "type": "completions",
            "model": model,
            "source": {"type": "file_id", "id": eval_file_id},
            "input_messages": {
                "item_reference": "item.messages",
                "type": "item_reference",
            },
            "sampling_params": {
                "tools": tools
            },
        }

        eval_run = client.evals.runs.create(
            name=model,
            eval_id=evaluation_id,
            data_source=data_source,
        )

        eval_runs.append(eval_run)
        print(f"   ✅ Run ID: {eval_run.id}")

    print(f"\n📋 Evaluation runs created:")
    for run in eval_runs:
        print(f"   🏃 {run.name}: {run.id}")

    return eval_runs
