"""Deployment discovery, explicit creation hooks, and model invocation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .retry import retry_call


def ensure_deployment(
    project_client: Any,
    *,
    deployment_name: str,
    expected_model_name: str | None = None,
    create: Callable[[], Any] | None = None,
    poll_interval_seconds: float = 15,
    timeout_seconds: float = 600,
) -> Any:
    """Return an existing deployment or run an explicit management creation action.

    `azure-ai-projects` 2.7 exposes deployment discovery but not deployment
    creation. Callers therefore supply a visible management-SDK/CLI action rather
    than the cookbook silently invoking an external tool.
    """
    try:
        deployment = project_client.deployments.get(deployment_name)
    except Exception:
        if create is None:
            raise RuntimeError(
                f"Deployment {deployment_name!r} was not found. Supply a create "
                "callback that uses your approved management path."
            )
        create()
        deployment = None

    started = time.monotonic()
    while True:
        if deployment is None:
            try:
                deployment = retry_call(
                    lambda: project_client.deployments.get(deployment_name),
                    operation_name="retrieve deployment",
                )
            except Exception:
                deployment = None
        if deployment is not None:
            model_name = getattr(deployment, "model_name", None)
            if isinstance(deployment, dict):
                model_name = deployment.get("model_name") or deployment.get("modelName")
            if expected_model_name is None or model_name == expected_model_name:
                return deployment
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Deployment {deployment_name!r} did not become visible with the "
                f"expected model within {timeout_seconds:g} seconds"
            )
        time.sleep(poll_interval_seconds)


def invoke_chat(
    openai_client: Any,
    *,
    deployment_name: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float = 0,
    **kwargs: Any,
) -> Any:
    """Invoke a chat-compatible deployment without obscuring the SDK response."""
    payload: dict[str, Any] = {
        "model": deployment_name,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return retry_call(
        lambda: openai_client.chat.completions.create(**payload),
        operation_name="invoke deployment",
    )


def invoke_response(
    openai_client: Any,
    *,
    deployment_name: str,
    input: Any,
    **kwargs: Any,
) -> Any:
    """Invoke a Responses-compatible deployment and return the native response."""
    return retry_call(
        lambda: openai_client.responses.create(
            model=deployment_name,
            input=input,
            **kwargs,
        ),
        operation_name="invoke deployment",
    )
