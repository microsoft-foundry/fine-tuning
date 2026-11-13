from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceNotFoundError

from shared import deployment as deployment_module
from shared.deployment import ensure_deployment
from shared.retry import RetryError


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.delays = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        assert delay >= 0
        self.delays.append(delay)
        self.now += delay


@pytest.fixture
def deployment_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(deployment_module, "time", clock)
    return clock


def test_deployment_creation_requires_not_found() -> None:
    created = []

    class ForbiddenError(RuntimeError):
        status_code = 403

    def get(name):
        raise ForbiddenError("Access denied")

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    with pytest.raises(ForbiddenError):
        ensure_deployment(
            project, deployment_name="customer-model", create=lambda: created.append(True)
        )
    assert not created


def test_not_found_can_use_explicit_creation_callback() -> None:
    created = []
    deployment = SimpleNamespace(model_name="fine-tuned-model")

    def get(name):
        if not created:
            raise ResourceNotFoundError("Not found")
        return deployment

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    result = ensure_deployment(
        project,
        deployment_name="customer-model",
        expected_model_name="fine-tuned-model",
        create=lambda: created.append(True),
    )
    assert result is deployment
    assert created == [True]


@pytest.mark.parametrize("shape", ["object", "model_name", "modelName"])
@pytest.mark.parametrize("intermediate", ["mismatch", "missing-model", "not-found"])
def test_deployment_polling_refreshes_mismatched_model(deployment_clock, shape, intermediate) -> None:
    created = []
    calls = []
    expected = "fine-tuned-model"
    if shape == "object":
        wrong = SimpleNamespace(model_name="old-model")
        matching = SimpleNamespace(model_name=expected)
    else:
        wrong = {shape: "old-model"}
        matching = {shape: expected}
    unfinished = {
        "mismatch": wrong,
        "missing-model": SimpleNamespace(),
        "not-found": ResourceNotFoundError("Not found during an external update"),
    }[intermediate]
    responses = iter([wrong, unfinished, matching])

    def get(name):
        calls.append((name, deployment_clock.now))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    result = ensure_deployment(
        project, deployment_name="customer-model", expected_model_name=expected,
        create=lambda: created.append(True), poll_interval_seconds=2, timeout_seconds=7,
    )

    assert result is matching
    assert calls == [("customer-model", 0), ("customer-model", 2), ("customer-model", 4)]
    assert deployment_clock.delays == [2, 2]
    assert created == []


def test_deployment_polling_refreshes_after_explicit_creation(deployment_clock) -> None:
    created = []
    calls = []
    matching = SimpleNamespace(model_name="fine-tuned-model")
    responses = iter([
        ResourceNotFoundError("Not found"),
        SimpleNamespace(model_name="old-model"),
        ResourceNotFoundError("Not yet visible"),
        matching,
    ])

    def get(name):
        calls.append((name, deployment_clock.now))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    result = ensure_deployment(
        project, deployment_name="customer-model", expected_model_name="fine-tuned-model",
        create=lambda: created.append(True), poll_interval_seconds=2, timeout_seconds=7,
    )

    assert result is matching
    assert calls == [
        ("customer-model", 0), ("customer-model", 0),
        ("customer-model", 2), ("customer-model", 4),
    ]
    assert deployment_clock.delays == [2, 2]
    assert created == [True]


@pytest.mark.parametrize("unfinished", ["mismatch", "missing-model", "not-found"])
def test_deployment_polling_uses_one_deadline_on_mismatch(deployment_clock, unfinished) -> None:
    calls = []
    created = []

    def get(name):
        calls.append((name, deployment_clock.now))
        if len(calls) > 1:
            if unfinished == "not-found":
                raise ResourceNotFoundError("Not yet visible")
            if unfinished == "missing-model":
                return SimpleNamespace()
        return SimpleNamespace(model_name="old-model")

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    with pytest.raises(TimeoutError, match="within 5 seconds"):
        ensure_deployment(
            project, deployment_name="customer-model", expected_model_name="fine-tuned-model",
            create=lambda: created.append(True), poll_interval_seconds=2, timeout_seconds=5,
        )

    assert calls == [("customer-model", 0), ("customer-model", 2), ("customer-model", 4)]
    assert deployment_clock.delays == [2, 2, 1]
    assert deployment_clock.now == 5
    assert created == []


def test_deployment_polling_bounds_transient_retry_by_remaining_deadline(
    deployment_clock, monkeypatch,
) -> None:
    calls = []
    created = []
    monkeypatch.setattr("shared.retry.random.random", lambda: 0.5)

    class UnavailableError(RuntimeError):
        status_code = 503

    def get(name):
        calls.append((name, deployment_clock.now))
        if len(calls) > 1:
            raise UnavailableError("Temporary service failure")
        return SimpleNamespace(model_name="old-model")

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    with pytest.raises(RetryError, match="cannot retry within"):
        ensure_deployment(
            project, deployment_name="customer-model", expected_model_name="fine-tuned-model",
            create=lambda: created.append(True), poll_interval_seconds=4, timeout_seconds=5,
        )

    assert calls == [("customer-model", 0), ("customer-model", 4)]
    assert deployment_clock.delays == [4, 1]
    assert deployment_clock.now == 5
    assert created == []
