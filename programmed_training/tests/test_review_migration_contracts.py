"""Bounded docs/metadata regressions; no resolver, Git, or network calls."""

from pathlib import Path
import re
from types import SimpleNamespace
import tomllib
from unittest.mock import Mock, call

from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def _name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def test_documented_full_suite_install_includes_deployment_extra():
    docs = _read("docs/contributing.md")
    assert "pip install -e '.[test,deployment]'" in docs
    project = tomllib.loads(_read("pyproject.toml"))["project"]
    assert {"test", "deployment"} <= project["optional-dependencies"].keys()
    deployment = {
        _name(Requirement(value).name)
        for value in project["optional-dependencies"]["deployment"]
    }
    assert {"azure-ai-projects", "openai"} <= deployment


@pytest.fixture
def deployment():
    # Full-suite setup deliberately includes the deployment extra. Missing
    # azure-ai-projects must not be hidden by a skip or an invented stub module.
    from programmed_training.deployment import foundry_deployments

    return foundry_deployments


def test_wait_polls_until_succeeded_instead_of_returning_creating(monkeypatch, deployment):
    creating = {"properties": {"provisioningState": "Creating"}}
    succeeded = {"name": "offline-model", "properties": {"provisioningState": "Succeeded"}}
    request = Mock(side_effect=[creating, succeeded])
    sleep = Mock()
    monkeypatch.setattr(deployment, "_arm_request", request)
    monkeypatch.setattr(deployment, "time", SimpleNamespace(monotonic=lambda: 0, sleep=sleep))
    credential = object()
    url = "https://management.azure.com/offline-only"

    assert deployment._wait(credential, url, creating) is succeeded
    assert request.call_args_list == [call(credential, "GET", url)] * 2
    assert sleep.call_args_list == [call(30)] * 2
    assert deployment._DEPLOYMENT_TIMEOUT_SECONDS == 3600
    docs = _read("docs/deployment-and-inference.md")
    assert "blocks, polling provisioning state for up to one hour" in docs
    assert "only after provisioning succeeds" in docs


@pytest.mark.parametrize("state", ["Failed", "Canceled"])
def test_wait_raises_for_failed_or_canceled_state_without_cleanup(monkeypatch, deployment, state):
    request, sleep = Mock(), Mock()
    monkeypatch.setattr(deployment, "_arm_request", request)
    monkeypatch.setattr(deployment, "time", SimpleNamespace(monotonic=lambda: 0, sleep=sleep))

    with pytest.raises(RuntimeError, match=state):
        deployment._wait(object(), "offline-only", {"properties": {"provisioningState": state}})

    request.assert_not_called()
    sleep.assert_not_called()


def test_wait_timeout_does_not_cancel_or_delete_remote_deployment(monkeypatch, deployment):
    request, sleep = Mock(), Mock()
    monkeypatch.setattr(deployment, "_arm_request", request)
    monkeypatch.setattr(
        deployment, "time", SimpleNamespace(monotonic=Mock(side_effect=[0, 3600]), sleep=sleep)
    )

    with pytest.raises(TimeoutError, match="inspect remote state"):
        deployment._wait(object(), "offline-only", {"properties": {"provisioningState": "Creating"}})

    request.assert_not_called()
    sleep.assert_not_called()