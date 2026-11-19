"""Offline deadline and durability regressions for the real Stage 05 helpers."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


STAGE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("stage05_runtime_deadline_tests", STAGE / "stage05_runtime.py")
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.delays = []
        self.deadline = 5.0

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        assert 0 < delay <= self.deadline - self.now
        self.delays.append(delay)
        self.now += delay


class APIConnectionError(Exception):
    """Match the OpenAI exception name without importing or contacting the SDK."""


class APITimeoutError(APIConnectionError):
    """Preserve the SDK's timeout/connection exception inheritance."""


def forbidden(*_args, **_kwargs):
    raise AssertionError("The unconfigured client or a mutation must not be used by polling")


def client_for(kind, retrieve):
    resource = SimpleNamespace(retrieve=retrieve, create=forbidden, delete=forbidden, cancel=forbidden)
    configured = (
        SimpleNamespace(files=resource)
        if kind == "file"
        else SimpleNamespace(fine_tuning=SimpleNamespace(jobs=resource))
    )
    options = []

    def with_options(**kwargs):
        assert kwargs == {"max_retries": 0}
        options.append(kwargs)
        return configured

    return SimpleNamespace(with_options=with_options), options


def poll(kind, client, **kwargs):
    operation = runtime.wait_for_file_processed if kind == "file" else runtime.monitor_fine_tuning_job
    return operation(client, "unit-resource", **kwargs)


def success(kind):
    return SimpleNamespace(status="processed" if kind == "file" else "succeeded")


@pytest.fixture
def clock(monkeypatch):
    selected = Clock()
    monkeypatch.setattr(runtime, "time", selected)
    return selected


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("parameter", ["poll_interval_seconds", "timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_invalid_budget_fails_before_client_configuration(clock, kind, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        poll(kind, SimpleNamespace(with_options=forbidden), **{parameter: value})
    assert clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_each_request_receives_the_remaining_budget_with_sdk_retries_disabled(clock, kind):
    calls = []
    terminal = success(kind)

    def retrieve(resource_id, *, timeout):
        assert resource_id == "unit-resource"
        calls.append((clock.now, timeout))
        clock.now += 0.25
        return SimpleNamespace(status="running") if len(calls) == 1 else terminal

    client, options = client_for(kind, retrieve)
    assert poll(kind, client, poll_interval_seconds=1, timeout_seconds=5) is terminal
    assert calls == [(0, 5), (1.25, 3.75)]
    assert clock.delays == [1]
    assert options == [{"max_retries": 0}]


@pytest.mark.parametrize("kind", ["file", "job"])
def test_request_timeout_never_exceeds_budget_after_deadline_roundoff(clock, kind):
    clock.now = 3.05
    clock.deadline = clock.now + 5
    assert clock.deadline - clock.now > 5
    calls = []
    terminal = success(kind)

    def retrieve(_resource_id, *, timeout):
        calls.append(timeout)
        return terminal

    client, _ = client_for(kind, retrieve)
    assert poll(kind, client, timeout_seconds=5) is terminal
    assert calls == [5]


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("duration", [5, 6])
@pytest.mark.parametrize("state", ["success", "failed", "running"])
def test_all_responses_including_terminal_success_are_rejected_at_expiry(clock, kind, duration, state):
    calls = []

    def retrieve(_resource_id, *, timeout):
        calls.append(timeout)
        clock.now += duration
        return success(kind) if state == "success" else SimpleNamespace(status=state)

    client, _ = client_for(kind, retrieve)
    with pytest.raises(TimeoutError, match="timeout"):
        poll(kind, client, timeout_seconds=5)
    assert calls == [5]
    assert clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError, ValueError])
def test_deadline_is_checked_after_transient_and_nontransient_request_errors(clock, kind, error_type):
    failure = error_type("Synthetic transport failure")

    def retrieve(_resource_id, *, timeout):
        assert timeout == 5
        clock.now = 5
        raise failure

    client, _ = client_for(kind, retrieve)
    with pytest.raises(TimeoutError) as caught:
        poll(kind, client, timeout_seconds=5)
    assert caught.value.__cause__ is failure
    assert clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_capped_poll_sleep_never_starts_another_retrieve_at_expiry(clock, kind):
    calls = []

    def retrieve(_resource_id, *, timeout):
        calls.append((clock.now, timeout))
        return SimpleNamespace(status="running")

    client, _ = client_for(kind, retrieve)
    with pytest.raises(TimeoutError):
        poll(kind, client, poll_interval_seconds=60, timeout_seconds=5)
    assert calls == [(0, 5)]
    assert clock.delays == [5]


@pytest.mark.parametrize("kind", ["file", "job"])
def test_requests_poll_sleeps_and_retry_backoff_share_one_deadline(clock, kind):
    calls = []
    failure = APIConnectionError("Synthetic transient failure")

    def retrieve(_resource_id, *, timeout):
        calls.append((clock.now, timeout))
        if len(calls) == 1:
            clock.now += 3
            return SimpleNamespace(status="running")
        clock.now += 0.25
        raise failure

    client, _ = client_for(kind, retrieve)
    with pytest.raises(TimeoutError) as caught:
        poll(kind, client, poll_interval_seconds=1, timeout_seconds=5)
    assert caught.value.__cause__ is failure
    assert calls == [(0, 5), (4, 1)]
    assert clock.delays == [1, 0.75]


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError])
def test_a_transient_read_can_recover_without_resetting_the_deadline(clock, kind, error_type):
    calls = []
    terminal = success(kind)

    def retrieve(_resource_id, *, timeout):
        calls.append((clock.now, timeout))
        clock.now += 0.5
        if len(calls) == 1:
            raise error_type("Synthetic transient failure")
        return SimpleNamespace(status="running") if len(calls) == 2 else terminal

    client, _ = client_for(kind, retrieve)
    assert poll(kind, client, poll_interval_seconds=1, timeout_seconds=5) is terminal
    assert calls == [(0, 5), (1.5, 3.5), (3, 2)]
    assert clock.delays == [1, 1]


@pytest.mark.parametrize("kind", ["file", "job"])
def test_retry_attempt_limit_preserves_the_last_error_without_resetting_the_deadline(clock, kind):
    clock.deadline = 30
    calls = []
    failure = APIConnectionError("Synthetic repeated read-only failure")

    def retrieve(_resource_id, *, timeout):
        calls.append((clock.now, timeout))
        raise failure

    client, options = client_for(kind, retrieve)
    with pytest.raises(APIConnectionError) as caught:
        poll(kind, client, timeout_seconds=30)
    assert caught.value is failure
    assert calls == [(0, 30), (1, 29), (3, 27), (7, 23), (15, 15)]
    assert clock.delays == [1, 2, 4, 8]
    assert options == [{"max_retries": 0}]


@pytest.mark.parametrize("kind", ["file", "job"])
@pytest.mark.parametrize("status,message,transient", [
    (400, "Bad request", False),
    (401, "Unauthorized", False),
    (403, "Forbidden", False),
    (404, "Unrelated missing resource", False),
    (404, "Project not found", True),
    (404, "No such file object", True),
    (408, "Request timeout", True),
    (409, "Conflict", True),
    (429, "Throttled", True),
    (500, "Server error", True),
    (503, "Service unavailable", True),
])
def test_http_read_errors_retry_only_the_existing_transient_contract(clock, kind, status, message, transient):
    failure = RuntimeError(message)
    failure.status_code = status
    calls = []
    terminal = success(kind)

    def retrieve(_resource_id, *, timeout):
        calls.append((clock.now, timeout))
        if len(calls) == 1:
            raise failure
        return terminal

    client, _ = client_for(kind, retrieve)
    if transient:
        assert poll(kind, client, timeout_seconds=5) is terminal
        assert calls == [(0, 5), (1, 4)] and clock.delays == [1]
    else:
        with pytest.raises(RuntimeError) as caught:
            poll(kind, client, timeout_seconds=5)
        assert caught.value is failure
        assert calls == [(0, 5)] and clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_poll_oversleep_is_detected_before_another_get(clock, monkeypatch, kind):
    calls = []

    def retrieve(_resource_id, *, timeout):
        calls.append(timeout)
        return SimpleNamespace(status="running")

    def oversleep(delay):
        assert delay == 1
        clock.now = 6

    monkeypatch.setattr(clock, "sleep", oversleep)
    client, _ = client_for(kind, retrieve)
    with pytest.raises(TimeoutError):
        poll(kind, client, poll_interval_seconds=1, timeout_seconds=5)
    assert calls == [5]


@pytest.mark.parametrize("kind", ["file", "job"])
def test_client_configuration_cannot_start_a_request_after_expiry(clock, kind):
    def with_options(**kwargs):
        assert kwargs == {"max_retries": 0}
        clock.now = 5
        configured, _ = client_for(kind, forbidden)
        return configured.with_options(**kwargs)

    with pytest.raises(TimeoutError):
        poll(kind, SimpleNamespace(with_options=with_options), timeout_seconds=5)
    assert clock.delays == []


@pytest.mark.parametrize("kind", ["file", "job"])
def test_status_inspection_cannot_accept_a_terminal_response_after_expiry(clock, kind):
    class SlowStatus:
        @property
        def status(self):
            clock.now = 5
            return "processed" if kind == "file" else "succeeded"

    client, _ = client_for(kind, lambda _resource_id, *, timeout: SlowStatus())
    with pytest.raises(TimeoutError):
        poll(kind, client, timeout_seconds=5)


@pytest.mark.parametrize("kind", ["file", "job"])
def test_logging_time_is_not_added_back_to_the_retry_sleep_budget(clock, monkeypatch, kind):
    def retrieve(_resource_id, *, timeout):
        raise APIConnectionError("Synthetic transient failure")

    def slow_print(*_args, **_kwargs):
        clock.now = 5

    monkeypatch.setattr("builtins.print", slow_print)
    client, _ = client_for(kind, retrieve)
    with pytest.raises(TimeoutError):
        poll(kind, client, timeout_seconds=5)
    assert clock.delays == []


def test_stage05_writer_uses_shared_fsync_before_atomic_replace(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(STAGE.parent))
    from shared import experiment_manifest

    events = []
    fsync = os.fsync
    replace = os.replace
    outputs = tmp_path / "outputs"
    destination = outputs / "run" / "run-state.json"

    def record_fsync(descriptor):
        events.append("fsync")
        return fsync(descriptor)

    def record_replace(source, target):
        assert events == ["fsync"], "File fsync must precede publication of the journal"
        assert Path(source).parent == destination.parent
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"job_id": "unit-job"}
        events.append("replace")
        return replace(source, target)

    monkeypatch.setattr(experiment_manifest.os, "fsync", record_fsync)
    monkeypatch.setattr(experiment_manifest.os, "replace", record_replace)
    runtime.write_json_atomic(destination, {"job_id": "unit-job"}, outputs_root=outputs)
    assert events == (["fsync", "replace"] if os.name == "nt" else ["fsync", "replace", "fsync"])
    assert json.loads(destination.read_text(encoding="utf-8")) == {"job_id": "unit-job"}
    assert not list(destination.parent.glob("*.tmp"))


def test_nonfinite_journal_payload_preserves_the_previous_durable_state(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(STAGE.parent))
    outputs = tmp_path / "outputs"
    destination = outputs / "run" / "run-state.json"
    runtime.write_json_atomic(destination, {"pending_operation": "job_create"}, outputs_root=outputs)
    original = destination.read_bytes()
    with pytest.raises(ValueError):
        runtime.write_json_atomic(destination, {"value": float("nan")}, outputs_root=outputs)
    assert destination.read_bytes() == original
    assert not list(destination.parent.glob("*.tmp"))