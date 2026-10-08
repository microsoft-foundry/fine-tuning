"""Run recipe smokes."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4


COOKBOOK_ROOT = Path(__file__).resolve().parents[1]
RECIPE_COMMANDS = [
    # tulu3_sft recipe
    """python -m interactive_training.recipes.tulu3_sft.train_azure model_name=Qwen/Qwen3.8-27B
    learning_rate=5e-4 lora_rank=32 batch_size=128 max_length=1200 num_epochs=1 max_steps=30
    eval_every=10 save_every=10 max_wall_clock_seconds=7200""",

    # tool_ner_rl recipe
    """python -m interactive_training.recipes.tool_ner_rl.train_azure model_name=Qwen/Qwen3.6-35B-A3B
    max_steps=2 max_test_examples=16 tab_eval_documents=1 eval_every=1 save_every=1
    max_wall_clock_seconds=7200""",
]


def read_session_ids(log_path: Path) -> list[str]:
    metadata = json.loads((log_path / "run_meta.json").read_text(encoding="utf-8"))
    azure = metadata["azure"]
    session_ids = [
        azure.get("session_id"),
        azure.get("reference_session_id"),
        *azure.get("session_ids", {}).values(),
    ]
    ids = [session_id for session_id in session_ids if session_id is not None]
    if not ids or any(
        not isinstance(session_id, str) or not session_id.startswith("session_")
        for session_id in ids
    ):
        raise ValueError(f"Missing or invalid session IDs in {log_path / 'run_meta.json'}")
    return list(dict.fromkeys(ids))


def verify_sessions(log_path: Path, project_endpoint: str, timeout_minutes: int) -> None:
    from azure.ai.finetuningsessions import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FoundryFeaturesOptInKeys, SessionStatus
    from azure.core.credentials import AzureKeyCredential
    from azure.identity import DefaultAzureCredential

    session_ids = read_session_ids(log_path)
    api_key = os.environ.get("AZURE_AI_API_KEY")
    credential = AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()
    options = {} if api_key else {"credential_scopes": ["https://ai.azure.com/.default"]}
    deadline = time.monotonic() + timeout_minutes * 60
    pending = set(session_ids)
    client = None
    try:
        client = FineTuningSessionClient(endpoint=project_endpoint, credential=credential, **options)
        while pending:
            for session_id in list(pending):
                session = client.sessions.get(
                    session_id=session_id,
                    foundry_features=FoundryFeaturesOptInKeys.FINETUNING_SESSIONS_V1_PREVIEW,
                    api_version="v1",
                )
                if session.status == SessionStatus.SUCCEEDED:
                    pending.remove(session_id)
                elif session.status != SessionStatus.RUNNING and session.status != SessionStatus.QUEUED:
                    raise RuntimeError(
                        f"Session from {log_path} has status {session.status!s}, not succeeded"
                    )
            if pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Sessions from {log_path} did not reach succeeded within {timeout_minutes} minutes"
                    )
                time.sleep(min(15, remaining))
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            if not api_key:
                credential.close()
    print(f"Confirmed {len(session_ids)} session(s) succeeded for {log_path.name}", flush=True)


def run_recipes(project_endpoint: str, gap_minutes: int, session_timeout_minutes: int = 10) -> None:
    endpoint = urlsplit(project_endpoint)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or not endpoint.path.startswith("/api/projects/")
        or len(endpoint.path.split("/")) != 4
        or not endpoint.path.split("/")[-1]
    ):
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT must be an HTTPS Foundry project URL")
    if gap_minutes < 0:
        raise ValueError("gap-minutes must be nonnegative")
    if session_timeout_minutes < 1:
        raise ValueError("session-timeout-minutes must be positive")

    run_root = COOKBOOK_ROOT / "logs" / f"recipe-smoke-{uuid4().hex}"
    run_root.mkdir(parents=True)
    for index, command in enumerate(RECIPE_COMMANDS):
        if index:
            time.sleep(gap_minutes * 60)
        parts = shlex.split(command)
        name = parts[2].split(".")[-2].replace("_", "-")
        log_path = run_root / name
        args = [
            sys.executable,
            *parts[1:],
            f"project_endpoint={project_endpoint}",
            f"log_path=./logs/{run_root.name}/{name}",
            "behavior_if_log_dir_exists=raise",
        ]
        display_args = [*args]
        display_args[len(parts)] = "project_endpoint=<redacted>"
        print(f"Running {name}: {shlex.join(display_args)}", flush=True)
        driver_log = run_root / f"{name}.driver.log"
        with driver_log.open("x", encoding="utf-8") as output:
            os.chmod(driver_log, 0o600)
            try:
                result = subprocess.run(
                    args, cwd=COOKBOOK_ROOT, check=False,
                    stdout=output, stderr=subprocess.STDOUT, timeout=9000,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"{name} driver timed out; inspect the private logs at {run_root} "
                    "and the remote session before retrying"
                ) from None
        if result.returncode:
            raise RuntimeError(
                f"{name} failed with exit code {result.returncode}; inspect {driver_log} "
                "and any owned remote session before retrying"
            )
        verify_sessions(log_path, project_endpoint, session_timeout_minutes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-minutes", type=int, default=10)
    parser.add_argument("--session-timeout-minutes", type=int, default=10)
    args = parser.parse_args()
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    if not endpoint:
        parser.error("set AZURE_AI_PROJECT_ENDPOINT to the Foundry project URL")
    if args.gap_minutes < 0:
        parser.error("--gap-minutes must be nonnegative")
    run_recipes(endpoint, args.gap_minutes, args.session_timeout_minutes)


if __name__ == "__main__":
    main()
