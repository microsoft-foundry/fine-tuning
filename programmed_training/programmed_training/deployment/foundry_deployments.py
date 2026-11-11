"""Create and delete Fireworks deployments from Foundry training checkpoints."""

from __future__ import annotations

import json
import re
import time
from enum import StrEnum
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


class DeploymentTarget(StrEnum):
    FIREWORKS = "Fireworks"


_ARM_API_VERSION = "2026-03-01"
_DEPLOYMENT_TIMEOUT_SECONDS = 60 * 60
_POLL_INTERVAL_SECONDS = 30
_FAILED_STATES = {"failed", "canceled"}
_PROJECT_ID = re.compile(
    r"(?P<account_id>/subscriptions/[A-Za-z0-9-]+/resourceGroups/[A-Za-z0-9_.()-]+/"
    r"providers/Microsoft\.CognitiveServices/accounts/"
    r"(?P<account>[a-z0-9][a-z0-9-]*))/projects/(?P<project>[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_DEPLOYMENT_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _project_parts(project_id: str) -> tuple[str, str]:
    match = _PROJECT_ID.fullmatch(project_id)
    if match is None:
        raise ValueError("Expected a full Microsoft.CognitiveServices Foundry project ARM ID")
    return match.group("account_id"), (
        f"https://{match.group('account')}.services.ai.azure.com/api/projects/{match.group('project')}"
    )


def _deployment_url(account_id: str, deployment_name: str) -> str:
    if not _DEPLOYMENT_NAME.fullmatch(deployment_name):
        raise ValueError("Expected a valid Foundry deployment name")
    return (
        f"https://management.azure.com{account_id}/deployments/"
        f"{quote(deployment_name, safe='')}?api-version={_ARM_API_VERSION}"
    )


def _arm_request(
    credential: DefaultAzureCredential, method: str, url: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    token = credential.get_token("https://management.azure.com/.default").token
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=120) as response:
            content = response.read()
            return json.loads(content) if content else {}
    except HTTPError as exc:
        if method == "GET" and exc.code == 404:
            return {}
        raise RuntimeError(f"ARM {method} request failed with HTTP {exc.code}") from exc


def _wait(credential: DefaultAzureCredential, url: str, deployment: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + _DEPLOYMENT_TIMEOUT_SECONDS
    while True:
        state = deployment["properties"]["provisioningState"]
        if state == "Succeeded":
            return deployment
        if state.casefold() in _FAILED_STATES:
            raise RuntimeError(f"Deployment ended in state {state}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Deployment did not succeed within the timeout; inspect remote state")
        time.sleep(min(_POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))
        deployment = _arm_request(credential, "GET", url)
        if not deployment:
            raise RuntimeError("Deployment disappeared while provisioning")


def create_deployment(
    project_id: str,
    session_id: str,
    target: DeploymentTarget,
    sku_capacity: int,
) -> dict[str, Any]:
    """Create or reuse a Fireworks deployment; return its ARM deployment object."""
    account_id, endpoint = _project_parts(project_id)
    if not re.fullmatch(r"(?:session|model)_[A-Za-z0-9_-]+", session_id):
        raise ValueError("Expected a session_ or model_ ID")
    if target != DeploymentTarget.FIREWORKS:
        raise ValueError(f"Unsupported deployment target: {target!r}")
    if isinstance(sku_capacity, bool) or not isinstance(sku_capacity, int) or sku_capacity <= 0:
        raise ValueError("sku_capacity must be a positive integer")

    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=endpoint, credential=credential) as client:
            with client.get_openai_client() as openai_client:
                checkpoints = list(openai_client.fine_tuning.jobs.checkpoints.list(session_id))
        if not checkpoints:
            raise ValueError(f"No checkpoints found for {session_id}")
        finals = [
            checkpoint for checkpoint in checkpoints
            if "final" in checkpoint.fine_tuned_model_checkpoint.casefold()
        ]
        checkpoint = max(
            enumerate(finals or checkpoints),
            key=lambda pair: (
                pair[1].created_at if pair[1].created_at is not None else -1,
                pair[0],
            ),
        )[1]
        model = checkpoint.fine_tuned_model_checkpoint
        deployment_name = model.split(":", 1)[0]
        url = _deployment_url(account_id, deployment_name)
        body = {
            "sku": {"name": "GlobalProvisionedManaged", "capacity": sku_capacity},
            "properties": {
                "model": {
                    "format": "FireworksCustom",
                    "name": model,
                    "version": "1",
                    "sourceAccount": account_id,
                },
                "versionUpgradeOption": "NoAutoUpgrade",
                "raiPolicyName": "Microsoft.DefaultV2",
            },
        }
        deployment = _arm_request(credential, "GET", url)
        if deployment:
            if deployment["properties"]["model"]["name"] != model:
                raise RuntimeError(f"Deployment {deployment_name} already targets another checkpoint")
            if deployment["sku"]["capacity"] != sku_capacity:
                raise RuntimeError(f"Deployment {deployment_name} already has another SKU capacity")
        else:
            deployment = _arm_request(credential, "PUT", url, body)
        return _wait(credential, url, deployment)


def delete_deployment(project_id: str, deployment_name: str) -> None:
    """Delete a deployment from the Foundry account and wait for its removal."""
    account_id, _ = _project_parts(project_id)
    url = _deployment_url(account_id, deployment_name)
    with DefaultAzureCredential() as credential:
        if not _arm_request(credential, "GET", url):
            raise ValueError(f"Deployment {deployment_name} does not exist")
        _arm_request(credential, "DELETE", url)
        deadline = time.monotonic() + _DEPLOYMENT_TIMEOUT_SECONDS
        while _arm_request(credential, "GET", url):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Deployment {deployment_name} was not deleted within the timeout")
            time.sleep(min(_POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))
