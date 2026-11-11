from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Callable
from typing import Any

EXPECTED_HYPERPARAMETERS = {
    "batch_size": 128,
    "eval_interval": 25,
    "eval_samples": 2,
    "learning_rate_multiplier": 2.0,
    "n_epochs": 2,
    "reasoning_effort": "medium",
}


def assess_completed_job_reuse(
    job: Any,
    *,
    base_model: str,
    retrieve_file: Callable[[str], Any],
    retrieve_content: Callable[[str], bytes],
    training_sha256: str,
    validation_sha256: str,
    local_regrade_matches_service: bool,
) -> dict[str, Any]:
    reinforcement = getattr(getattr(job, "method", None), "reinforcement", None)
    hyperparameters = getattr(reinforcement, "hyperparameters", None)
    actual_hyperparameters = {
        name: getattr(hyperparameters, name, None)
        for name in EXPECTED_HYPERPARAMETERS
    }
    training_file = retrieve_file(job.training_file)
    validation_file = retrieve_file(job.validation_file)
    training_content = retrieve_content(job.training_file)
    validation_content = retrieve_content(job.validation_file)

    checks = {
        "terminal_success": job.status == "succeeded",
        "base_model": job.model == base_model,
        "method": getattr(getattr(job, "method", None), "type", None)
        == "reinforcement",
        "hyperparameters": actual_hyperparameters == EXPECTED_HYPERPARAMETERS,
        "training_bytes": training_file.bytes == len(training_content),
        "validation_bytes": validation_file.bytes == len(validation_content),
        "training_sha256": hashlib.sha256(training_content).hexdigest()
        == training_sha256,
        "validation_sha256": hashlib.sha256(validation_content).hexdigest()
        == validation_sha256,
        "local_regrade_matches_service": local_regrade_matches_service,
    }
    return {
        "reusable": all(checks.values()),
        "checks": checks,
        "actual_hyperparameters": actual_hyperparameters,
    }


def summarize_result_csv(content: str) -> dict[str, list[float]]:
    rows = list(csv.DictReader(io.StringIO(content)))

    def values(column: str) -> list[float]:
        return [
            float(row[column])
            for row in rows
            if row.get(column) not in {None, ""}
        ]

    return {
        "training_reward": values("train_mean_reward"),
        "validation_reward": values("full_valid_mean_reward"),
        "policy_gradient_loss_sum": values("pg_loss_sum"),
        "total_loss_sum": values("total_loss_sum"),
        "entropy_loss_sum": values("entropy_loss_sum"),
    }
