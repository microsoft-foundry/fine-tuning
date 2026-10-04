"""Inspect the first output item from a Foundry evaluation run."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from foundry_clients import create_openai_client, create_project_client

load_dotenv(ROOT / ".env")

eval_id = os.environ["FOUNDRY_EVAL_ID"]

with create_project_client() as project_client, create_openai_client(project_client) as client:
    first_run = next(iter(client.evals.runs.list(eval_id=eval_id)))
    first_item = next(
        iter(client.evals.runs.output_items.list(eval_id=eval_id, run_id=first_run.id))
    )
    print(first_item.model_dump_json(indent=2))
