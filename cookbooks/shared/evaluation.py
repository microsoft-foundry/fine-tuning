"""Foundry evaluation creation and bounded polling helpers."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from .logging import get_logger, log_event, safe_reference
from .retry import retry_call

TERMINAL_EVALUATION_STATUSES = {"completed", "failed", "canceled", "cancelled"}


def create_evaluation(
    client: Any,
    *,
    name: str,
    data_source_config: Any,
    testing_criteria: Iterable[Any],
) -> Any:
    return retry_call(
        lambda: client.evals.create(
            name=name,
            data_source_config=data_source_config,
            testing_criteria=list(testing_criteria),
        ),
        operation_name="create evaluation",
    )


def create_evaluation_run(
    client: Any,
    *,
    evaluation_id: str,
    name: str,
    data_source: Any,
    metadata: dict[str, str] | None = None,
) -> Any:
    return retry_call(
        lambda: client.evals.runs.create(
            eval_id=evaluation_id,
            name=name,
            data_source=data_source,
            metadata=metadata,
        ),
        operation_name="create evaluation run",
    )


def monitor_evaluation_run(
    client: Any,
    *,
    evaluation_id: str,
    run_id: str,
    poll_interval_seconds: float = 5,
    timeout_seconds: float = 3600,
    require_success: bool = True,
) -> Any:
    started = time.monotonic()
    last_status: str | None = None
    logger = get_logger()
    while True:
        run = retry_call(
            lambda: client.evals.runs.retrieve(
                eval_id=evaluation_id,
                run_id=run_id,
            ),
            operation_name="retrieve evaluation run",
        )
        status = str(getattr(run, "status", "")).casefold()
        if status != last_status:
            log_event(
                logger,
                operation="evaluation-run",
                state=status or "unknown",
                message="Evaluation run state changed",
                fields={"reference": safe_reference(run_id)},
            )
            last_status = status
        if status in TERMINAL_EVALUATION_STATUSES:
            if require_success and status != "completed":
                raise RuntimeError(
                    f"Evaluation run ended in terminal state {status}: "
                    f"{getattr(run, 'error', None)}"
                )
            return run
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Evaluation run {safe_reference(run_id)} did not finish within "
                f"{timeout_seconds:g} seconds"
            )
        time.sleep(poll_interval_seconds)
