from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from azure.core.exceptions import ServiceRequestError, ServiceResponseError

from shared import foundry_operations as operations
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
def polling_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(operations, "time", clock)
    monkeypatch.setattr("shared.retry.random.random", lambda: 0.5)
    return clock


def _forbidden(*args, **kwargs):
    raise AssertionError("Polling must not create, delete, or cancel resources")


def _client(kind, retrieve):
    resource = SimpleNamespace(
        retrieve=retrieve, create=_forbidden, delete=_forbidden, cancel=_forbidden,
    )
    child = (
        SimpleNamespace(files=resource)
        if kind == "file"
        else SimpleNamespace(fine_tuning=SimpleNamespace(jobs=resource))
    )
    options = []

    def with_options(*, max_retries):
        options.append({"max_retries": max_retries})
        assert max_retries == 0
        return child

    # Retrieval is available only on the configured child, never the original.
    return SimpleNamespace(with_options=with_options), options


def _poll(kind, client, **kwargs):
    if kind == "file":
        return operations.wait_for_file(client, "file-example", **kwargs)
    return operations.monitor_fine_tuning_job(client, "job-example", **kwargs)


def _success(kind):
    return SimpleNamespace(status="processed" if kind == "file" else "succeeded")


def test_native_sdk_supports_the_polling_options_and_request_timeout_keywords() -> None:
    from openai import OpenAI
    from openai.resources.files import Files
    from openai.resources.fine_tuning.jobs import Jobs

    for method, keyword in (
        (OpenAI.with_options, "max_retries"),
        (Files.retrieve, "timeout"),
        (Jobs.retrieve, "timeout"),
    ):
        assert inspect.signature(method).parameters[keyword].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("parameter", ["timeout_seconds", "poll_interval_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_polling_rejects_nonpositive_or_nonfinite_parameters_before_client_use(
    polling_clock, kind, parameter, value,
) -> None:
    client = SimpleNamespace(with_options=_forbidden)
    with pytest.raises(ValueError, match=parameter):
        _poll(kind, client, **{parameter: value})
    assert polling_clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_polling_passes_remaining_request_timeout_and_returns_native_success(
    polling_clock, kind,
) -> None:
    calls = []
    success = _success(kind)

    def retrieve(resource_id, *, timeout):
        calls.append((polling_clock.now, timeout))
        polling_clock.now += 0.25
        return SimpleNamespace(status="queued") if len(calls) == 1 else success

    client, options = _client(kind, retrieve)
    assert _poll(
        kind, client, poll_interval_seconds=1, timeout_seconds=5,
    ) is success
    assert calls == [(0, 5), (1.25, 3.75)]
    assert polling_clock.delays == [1]
    assert options == [{"max_retries": 0}]


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("duration", [5, 6])
@pytest.mark.parametrize("state", ["success", "queued", "failed"])
def test_polling_rejects_every_response_returned_at_or_after_the_deadline(
    polling_clock, kind, duration, state,
) -> None:
    calls = []
    result = _success(kind) if state == "success" else SimpleNamespace(status=state)

    def retrieve(resource_id, *, timeout):
        calls.append(timeout)
        polling_clock.now += duration
        return result

    client, _ = _client(kind, retrieve)
    kwargs = {"require_success": False} if kind == "job" else {}
    with pytest.raises(TimeoutError, match="within 5 seconds"):
        _poll(kind, client, timeout_seconds=5, **kwargs)
    assert calls == [5]
    assert polling_clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("error_type", [ServiceResponseError, ValueError])
def test_polling_checks_the_deadline_after_a_request_error(
    polling_clock, kind, error_type,
) -> None:
    failure = error_type("Transport failed")
    calls = []

    def retrieve(resource_id, *, timeout):
        calls.append(timeout)
        polling_clock.now = 5
        raise failure

    client, _ = _client(kind, retrieve)
    with pytest.raises(TimeoutError, match="within 5 seconds") as error:
        _poll(kind, client, timeout_seconds=5)
    assert error.value.__cause__ is failure
    assert calls == [5]
    assert polling_clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_polling_caps_sleep_and_never_retrieves_after_expiry(polling_clock, kind) -> None:
    calls = []

    def retrieve(resource_id, *, timeout):
        calls.append((polling_clock.now, timeout))
        return SimpleNamespace(status="running")

    client, _ = _client(kind, retrieve)
    with pytest.raises(TimeoutError, match="within 5 seconds"):
        _poll(kind, client, poll_interval_seconds=60, timeout_seconds=5)
    assert calls == [(0, 5)]
    assert polling_clock.delays == [5]
    assert polling_clock.now == 5


@pytest.mark.parametrize("kind", ["file", "job"])
def test_polling_shares_one_budget_across_requests_polls_and_retries(polling_clock, kind) -> None:
    calls = []
    transient = ServiceRequestError("Transport failed")

    def retrieve(resource_id, *, timeout):
        calls.append((polling_clock.now, timeout))
        if len(calls) == 1:
            polling_clock.now += 3
            return SimpleNamespace(status="running")
        polling_clock.now += 0.25
        raise transient

    client, _ = _client(kind, retrieve)
    with pytest.raises(TimeoutError, match="within 5 seconds") as error:
        _poll(kind, client, poll_interval_seconds=1, timeout_seconds=5)
    assert calls == [(0, 5), (4, 1)]
    assert polling_clock.delays == [1, 0.75]
    assert polling_clock.now == 5
    assert isinstance(error.value.__cause__, RetryError)
    assert error.value.__cause__.__cause__ is transient


@pytest.mark.parametrize("kind", ["file", "job"])
def test_polling_can_recover_transient_reads_within_the_shared_deadline(polling_clock, kind) -> None:
    calls = []
    success = _success(kind)

    def retrieve(resource_id, *, timeout):
        calls.append((polling_clock.now, timeout))
        polling_clock.now += 0.5
        if len(calls) == 1:
            raise ServiceResponseError("Transport failed")
        return SimpleNamespace(status="running") if len(calls) == 2 else success

    client, options = _client(kind, retrieve)
    assert _poll(kind, client, poll_interval_seconds=1, timeout_seconds=5) is success
    assert calls == [(0, 5), (1.5, 3.5), (3, 2)]
    assert polling_clock.delays == [1, 1]
    assert options == [{"max_retries": 0}]


@pytest.mark.parametrize("kind", ["file", "job"])
def test_client_setup_cannot_start_a_retrieve_after_the_deadline(polling_clock, kind) -> None:
    client, options = _client(kind, _forbidden)
    configure = client.with_options

    def with_options(*, max_retries):
        polling_clock.now = 5
        return configure(max_retries=max_retries)

    client.with_options = with_options
    with pytest.raises(TimeoutError, match="within 5 seconds"):
        _poll(kind, client, timeout_seconds=5)
    assert options == [{"max_retries": 0}]
    assert polling_clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_terminal_success_is_not_accepted_after_logging_uses_remaining_time(
    polling_clock, monkeypatch, kind,
) -> None:
    def log_event(*args, **kwargs):
        polling_clock.now = 5

    monkeypatch.setattr(operations, "log_event", log_event)
    client, _ = _client(kind, lambda resource_id, *, timeout: _success(kind))
    with pytest.raises(TimeoutError, match="within 5 seconds"):
        _poll(kind, client, timeout_seconds=5)
    assert polling_clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_timely_terminal_failures_remain_visible_without_cancellation(polling_clock, kind) -> None:
    failure = SimpleNamespace(status="failed", error="Training failed", status_details="Invalid data")
    client, _ = _client(kind, lambda resource_id, *, timeout: failure)
    with pytest.raises(RuntimeError, match="terminal state failed"):
        _poll(kind, client, timeout_seconds=5)
    if kind == "job":
        assert _poll(kind, client, timeout_seconds=5, require_success=False) is failure
    assert polling_clock.delays == []