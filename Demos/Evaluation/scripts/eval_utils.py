"""Foundry cloud evaluation helpers for the evaluation demos."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AzureAIModelTargetParam,
    ModelSamplingConfigParam,
    TargetCompletionEvalRunDataSource,
)
from azure.identity import DefaultAzureCredential


TERMINAL_EVALUATION_STATUSES = {"completed", "failed", "canceled"}


def build_model_target_data_source(
    *,
    items: Sequence[dict[str, Any]],
    model: str,
    input_messages: dict[str, Any],
    max_completion_tokens: int,
    top_p: float = 1.0,
) -> TargetCompletionEvalRunDataSource:
    """Build a stable Foundry model-target evaluation data source."""
    return TargetCompletionEvalRunDataSource(
        type="azure_ai_target_completions",
        source={
            "type": "file_content",
            "content": list(items),
        },
        input_messages=input_messages,
        target=AzureAIModelTargetParam(
            type="azure_ai_model",
            model=model,
            sampling_params=ModelSamplingConfigParam(
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
            ),
        ),
    )


class FoundryEvaluationClient:
    """Own the Foundry project client and its OpenAI-compatible eval operation."""

    def __init__(self, project_endpoint: str) -> None:
        self.credential = DefaultAzureCredential(process_timeout=90)
        self.project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=self.credential,
        )
        self._protocol_client = self.project_client.get_openai_client()

    def create_evaluation(
        self,
        *,
        name: str,
        data_source_config: dict[str, Any],
        testing_criteria: Iterable[dict[str, Any]],
    ):
        evaluation = self._protocol_client.evals.create(
            name=name,
            data_source_config=data_source_config,
            testing_criteria=list(testing_criteria),
        )
        print(f"Evaluation created: {evaluation.id}")
        return evaluation

    def create_run(
        self,
        *,
        evaluation_id: str,
        name: str,
        data_source: TargetCompletionEvalRunDataSource,
        metadata: dict[str, str] | None = None,
    ):
        run = self._protocol_client.evals.runs.create(
            eval_id=evaluation_id,
            name=name,
            data_source=data_source,
            metadata=metadata,
        )
        print(f"Evaluation run created: {run.id}")
        return run

    def wait_for_run(
        self,
        *,
        evaluation_id: str,
        run_id: str,
        polling_interval: float = 5,
    ):
        """Poll the Foundry cloud evaluation until it reaches a terminal state."""
        last_status = None
        while True:
            run = self._protocol_client.evals.runs.retrieve(
                eval_id=evaluation_id,
                run_id=run_id,
            )
            if run.status != last_status:
                print(f"Evaluation run {run_id}: {run.status}")
                last_status = run.status
            if run.status in TERMINAL_EVALUATION_STATUSES:
                if run.status != "completed":
                    raise RuntimeError(
                        f"Evaluation run {run_id} ended with status {run.status}: "
                        f"{getattr(run, 'error', None)}"
                    )
                return run
            time.sleep(polling_interval)

    def list_output_items(self, *, evaluation_id: str, run_id: str) -> list[dict[str, Any]]:
        items = self._protocol_client.evals.runs.output_items.list(
            eval_id=evaluation_id,
            run_id=run_id,
        )
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in items
        ]

    def close(self) -> None:
        self._protocol_client.close()
        self.project_client.close()
        self.credential.close()

    def __enter__(self) -> "FoundryEvaluationClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
