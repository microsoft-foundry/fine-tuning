import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from azure.ai.finetuningsessions import FineTuningSession

from interactive_training.utils.file_utils import read_jsonl
from interactive_training.utils.trace import scope, update_scope_context

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)


def public_checkpoint_path(path: str) -> str:
    """Use the SDK-supported session/checkpoint form instead of a service URI."""
    if "://" not in path:
        return path
    _, _, location = path.partition("://")
    if len(location.split("/")) == 2 and all(location.split("/")):
        return location
    session, separator, checkpoint = location.partition("/weights/")
    if not separator or not session or not checkpoint or "/" in checkpoint:
        raise ValueError("Unrecognized training checkpoint path returned by the service")
    return f"{session}/{checkpoint}"


@scope
def load_checkpoints_file(log_dir: str) -> list[dict[str, Any]]:
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(checkpoint_path):
        # Common case on fresh runs (timestamped log dirs); not interesting at INFO.
        logger.debug(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    update_scope_context({"checkpoint_path": checkpoint_path})
    return read_jsonl(checkpoint_path)


@scope
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
    """
    Get the last checkpoint from the checkpoints.jsonl file in the specified log directory.

    Args:
        log_dir: The directory to check.
        required_key: The key to check for in the checkpoint.
            We might save partial checkpoints (e.g. sampler) in the same file,
            so we need to filter to the rows that have a fully-resumable checkpoint.

    Returns:
        The last checkpoint, or None if no checkpoint is found.
    """
    checkpoints = load_checkpoints_file(log_dir)
    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    if checkpoints_with_key:
        logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{required_key}' in {log_dir}"
        )
        logger.info(f"Using last checkpoint: {checkpoints_with_key[-1]}")
        return checkpoints_with_key[-1]
    else:
        logger.debug(f"No checkpoints found with key {required_key} in {log_dir}")
        return None


@dataclass(frozen=True)
class CheckpointSelection:
    """Which checkpoint should bootstrap a new training session.

    Attributes:
        checkpoint_path: The ``state_path`` to initialize from, or None for a
            clean start from the base model.
        is_resume: True if the path came from the local ledger (crash recovery),
            False if it came from an explicit ``load_checkpoint_path`` (continual FT).
        ignored_load_checkpoint_path: Set when an explicit ``load_checkpoint_path``
            was supplied but preempted by auto-resume (so callers can warn).
    """

    checkpoint_path: str | None
    is_resume: bool
    ignored_load_checkpoint_path: str | None


def select_resume_checkpoint(
    resume_info: dict[str, Any] | None,
    load_checkpoint_path: str | None,
) -> CheckpointSelection:
    """Decide which checkpoint bootstraps a session.

    Auto-resume wins when the ``log_path`` already has a resumable ledger, so the
    restored weights stay consistent with the dataset cursor (which the training
    loop reads from that same ledger). An explicit ``load_checkpoint_path`` only
    applies to a fresh ``log_path``; otherwise it is ignored and reported via
    ``ignored_load_checkpoint_path``. This mirrors tinker-cookbook's resume-wins
    ordering and avoids mixing CLI-checkpoint weights with a stale local cursor.
    """
    if resume_info and "state_path" in resume_info:
        return CheckpointSelection(
            checkpoint_path=resume_info["state_path"],
            is_resume=True,
            ignored_load_checkpoint_path=load_checkpoint_path or None,
        )
    if load_checkpoint_path:
        return CheckpointSelection(
            checkpoint_path=load_checkpoint_path,
            is_resume=False,
            ignored_load_checkpoint_path=None,
        )
    return CheckpointSelection(
        checkpoint_path=None,
        is_resume=False,
        ignored_load_checkpoint_path=None,
    )


@scope
async def save_checkpoint_async(
    training_client: FineTuningSession,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
    ttl_seconds: int | None = None,
    *,
    step_number: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save model checkpoint.

    `kind` controls *what* is saved, and the choice has real GPU-memory consequences:

    - "state":   training-side checkpoint (weights + optimizer) for resume only.
                 Cheap. Safe to call during training.
    - "sampler": exports a sampler-format snapshot AND launches a vLLM inference
                 engine to serve it on the worker GPUs. Allocates a large slice of
                 GPU memory (`gpu_memory_utilization`, default 0.8 per device in
                 the training backend). Will OOM if the FSDP-sharded trainer is still resident.
    - "both":    state + sampler. RL recipes use this because they need to sample
                 between optim steps. SFT recipes should NOT use this for periodic
                 saves \u2014 use "state" instead, and export inference-ready weights as
                 a separate post-training step.

    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name, ttl_seconds=ttl_seconds, step_number=step_number, metrics=metrics)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(
            name, ttl_seconds=ttl_seconds
        )

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    if "state_path" in paths:
        paths["state_path"] = public_checkpoint_path(paths["state_path"])
    update_scope_context(paths)
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    if step_number is not None:
        full_dict["step"] = step_number
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


def record_checkpoint_evaluation(
    *,
    log_path: str,
    step: int,
    checkpoint_path: str,
    metrics: dict[str, Any],
) -> None:
    """Record an evaluation only after its durable checkpoint is available."""
    record = {**metrics, "step": step, "checkpoint_path": checkpoint_path}
    os.makedirs(log_path, exist_ok=True)
    path = os.path.join(log_path, "checkpoint_evaluations.jsonl")
    line = (json.dumps(record) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise RuntimeError(
                f"Short write to {path}: wrote {written}/{len(line)} bytes"
            )
    finally:
        os.close(descriptor)


@scope
def save_checkpoint(
    training_client: FineTuningSession,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
    ttl_seconds: int | None = None,
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    return asyncio.run(
        save_checkpoint_async(
            training_client,
            name=name,
            log_path=log_path,
            kind=kind,
            loop_state=loop_state,
            ttl_seconds=ttl_seconds,
        )
    )
