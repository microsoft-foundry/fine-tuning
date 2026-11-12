"""Recovery through real SDK clients, adapters, loops, and checkpoint journals."""

import asyncio
import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call
from urllib.parse import urlsplit

import chz
import pytest
from azure.ai.finetuningsessions import TrainingEngineError
from azure.ai.finetuningsessions._exceptions import _classify_poll_failure
from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

from programmed_training import auto_resume, checkpoint_utils
from programmed_training.rl.train import AsyncConfig, StreamMinibatchConfig
from tests.sdk_recovery_fixtures import ModelState, Service
from tests.test_dynamic_sampling import (
    ControlledDataset, MixedRewardGroupBuilder, _make_config, _make_mock_session,
)


@pytest.fixture
def fast_backoff(monkeypatch):
    monkeypatch.setattr(auto_resume.random, "uniform", lambda *_: 0)


@pytest.fixture
def backoff_delays(monkeypatch):
    delays = []
    sleep = asyncio.sleep

    async def immediate_sleep(seconds):
        if seconds > 0:
            delays.append(seconds)
        await sleep(0)

    monkeypatch.setattr(auto_resume.random, "uniform", lambda *_: 1)
    monkeypatch.setattr(
        auto_resume, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": immediate_sleep})
    )
    return delays


@pytest.mark.xfail(
    condition=version("azure-ai-finetuningsessions") == "1.0.0b1",
    reason="SDK 1.0.0b1 retries should_retry engine_dead responses before recovery can handle them",
    strict=True,
)
def test_retryable_engine_dead_is_classified_as_terminal_for_recovery():
    error = _classify_poll_failure(
        {
            "status": "failed",
            "error_code": "engine_dead",
            "error": "synthetic engine loss",
            "should_retry": True,
        },
        session_id="session_0",
    )
    assert isinstance(error, TrainingEngineError)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async-stream"])
@pytest.mark.parametrize("failure_stage", ["poll", "submit"])
@pytest.mark.parametrize("close_engine_dead", [
    False,
    pytest.param(
        True,
        marks=pytest.mark.xfail(
            condition=version("azure-ai-finetuningsessions") == "1.0.0b1",
            reason="SDK 1.0.0b1 does not close an engine-dead session idempotently",
            strict=True,
        ),
    ),
])
async def test_real_sdk_and_training_loops_recover_checkpoint(
    tmp_path, monkeypatch, fast_backoff, mode, failure_stage, close_engine_dead
):
    from programmed_training.rl import train_azure

    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([0.0, 1.0]), MixedRewardGroupBuilder([1.0, 0.0])]
        for _ in range(8)
    ])
    cfg = _make_config(
        dataset, tmp_path, save_every=2, max_steps=6,
        async_config=AsyncConfig(max_steps_off_policy=1, groups_per_batch=2) if "async" in mode else None,
        stream_minibatch_config=StreamMinibatchConfig(groups_per_batch=2, num_minibatches=2) if "stream" in mode else None,
    )
    monkeypatch.setattr(train_azure, "get_tokenizer", lambda _: _make_mock_session().get_tokenizer())
    service = Service(
        failure_stage=failure_stage,
        failed_close_response=(409, {"detail": {
            "reason": "engine_dead", "error_code": "engine_dead",
            "failure_reason": "lease expired", "message": "The training engine stopped",
        }}) if close_engine_dead else None,
    )
    client = service.client()
    session_kwargs = dict(base_model="test-model", lora_config=LoRAConfig(rank=16), type="training")
    session = await client.create_session(**session_kwargs)
    await asyncio.wait_for(auto_resume.run(
        lambda c, s: train_azure.main(cfg, c, s),
        config=auto_resume.AutoResumeConfig(),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=session_kwargs,
    ), timeout=15)

    assert len(service.creations) == 2
    replacement, payload, restored = service.creations[1]
    assert payload["from_checkpoint"]["checkpoint_id"] == "2"
    assert restored == ModelState(weights=2, optimizer=114)
    assert service.failed_states == [(session, ModelState(weights=4, optimizer=128))]
    assert service.states[replacement] == ModelState(weights=6, optimizer=142)
    paths = auto_resume.attempt_log_paths(str(tmp_path))
    assert len(paths) == 2
    assert checkpoint_utils.load_checkpoints_file(paths[0])[-1]["name"] != "final"
    initial_meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert initial_meta["azure"]["session_id"] == session
    assert initial_meta["auto_resume"]["status"] == "failed"
    replacement_meta = json.loads((Path(paths[1]) / "run_meta.json").read_text())
    assert replacement_meta["azure"]["session_id"] == replacement
    assert replacement_meta["azure"]["from_checkpoint"]["checkpoint_id"] == "2"
    assert replacement_meta["auto_resume"]["status"] == "completed"
    assert replacement_meta["auto_resume"]["run_id"] == initial_meta["auto_resume"]["run_id"]
    assert replacement_meta["auto_resume"]["previous_log_path"] == paths[0]
    assert json.loads((Path(paths[1]) / "config.json").read_text())["log_path"] == paths[1]
    status = json.loads((tmp_path / "recovery_status.json").read_text())
    assert status["event"] == "completed"
    assert status["elapsed_seconds"] >= replacement_meta["auto_resume"]["elapsed_seconds"]
    rows = checkpoint_utils.load_checkpoints_file(paths[1])
    assert rows[-1]["name"] == "final"
    assert rows[-1]["batch"] == 6
    assert replacement.removeprefix("session_") in rows[-1]["state_path"]
    assert set(service.closed) == {session, replacement}
    assert not [task for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()
                and task.get_name().startswith(("trajectory_group_worker", "training_loop", "dataloader_loop"))]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload", [
    (409, {"detail": {"reason": "model_busy", "message": "Model is busy"}}),
    (409, ["malformed error body"]),
    (403, {"detail": "Forbidden"}),
    (500, {"error_code": "engine_oom", "message": "Try a smaller batch"}),
])
async def test_close_conflict_or_nonresumable_error_blocks_replacement(
    tmp_path, fast_backoff, status, payload
):
    service = Service(failures=0, failed_close_response=(status, payload))
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    service.failed_states.append((session, ModelState()))
    original = TrainingEngineError("lost", error_code="engine_dead", session_id=session)

    async def workload(c, s):
        raise original

    with pytest.raises(TrainingEngineError) as caught:
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
            client=client, session_id=session, client_factory=service.client,
            session_kwargs=kwargs,
        )
    assert caught.value is original
    assert len(service.creations) == 1
    assert service.closed == [session]
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert any(row["event"] == "cleanup_failed" for row in events)


@pytest.mark.asyncio
async def test_close_engine_dead_for_another_session_blocks_replacement(tmp_path, fast_backoff):
    service = Service(failures=0, failed_close_response=(409, {"detail": {
        "reason": "engine_dead", "error_code": "engine_dead", "message": "The training engine stopped",
    }}))
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    reference = await client.create_session(**kwargs)
    service.failed_states.append((reference, ModelState()))
    original = TrainingEngineError("lost", error_code="engine_dead", session_id=session)

    async def workload(c, s):
        raise original

    with pytest.raises(TrainingEngineError) as caught:
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
            client=client, session_id=session, client_factory=service.client,
            session_kwargs=kwargs, initial_session_closer=lambda: client.close_session(reference),
        )
    assert caught.value is original
    assert len(service.creations) == 2
    assert service.closed == [reference]
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert any(row["event"] == "cleanup_failed" for row in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 1, 3, 5])
async def test_repeated_losses_without_new_checkpoint_progress_exhaust_budget(
    tmp_path, monkeypatch, backoff_delays, max_retries
):
    from programmed_training.rl import train_azure

    dataset = ControlledDataset([[MixedRewardGroupBuilder([0.0, 1.0])] for _ in range(8)])
    cfg = _make_config(dataset, tmp_path, save_every=2, max_steps=6)
    monkeypatch.setattr(train_azure, "get_tokenizer", lambda _: _make_mock_session().get_tokenizer())
    service = Service(failures=10)
    client = service.client()
    kwargs = dict(base_model="test-model", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    with pytest.raises(TrainingEngineError):
        await auto_resume.run(
            lambda c, s: train_azure.main(cfg, c, s),
            config=auto_resume.AutoResumeConfig(auto_resume_max_retries=max_retries),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
        )
    assert len(service.creations) == max_retries + 1
    assert len(service.failed_states) == max_retries + 1
    assert len(set(service.closed)) == max_retries + 1
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "exhausted"
    assert events[-1]["error_code"] == "engine_dead"
    assert events[-1]["max_retries"] == max_retries
    assert events[-1]["retries_since_progress"] == max_retries
    assert sum(event["event"] == "attempt" for event in events) == max_retries + 1
    assert sum(event["event"] == "backoff" for event in events) == max_retries
    assert not any(event["event"] == "retry_budget_reset" for event in events)
    assert backoff_delays == [10 * 2 ** min(index, 2) for index in range(max_retries)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async-stream"])
async def test_durable_progress_allows_more_than_three_lifetime_recoveries(
    tmp_path, monkeypatch, backoff_delays, mode
):
    from programmed_training.rl import train_azure

    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([0.0, 1.0]), MixedRewardGroupBuilder([1.0, 0.0])]
        for _ in range(16)
    ])
    cfg = _make_config(dataset, tmp_path, save_every=2, max_steps=12)
    if "async" in mode:
        cfg = chz.replace(cfg, async_config=AsyncConfig(
            max_steps_off_policy=1, groups_per_batch=2,
        ))
    if "stream" in mode:
        cfg = chz.replace(cfg, stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=2, num_minibatches=2,
        ))
    monkeypatch.setattr(train_azure, "get_tokenizer", lambda _: _make_mock_session().get_tokenizer())
    service = Service(failures=4, failure_weights=[4, 6, 8, 10])
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    await auto_resume.run(
        lambda c, s: train_azure.main(cfg, c, s),
        config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
        client=client, session_id=session, client_factory=service.client, session_kwargs=kwargs,
    )
    assert len(service.creations) == 5
    assert len(set(service.closed)) == 5
    assert service.states[service.creations[-1][0]] == ModelState(12, 184)
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    resets = [event for event in events if event["event"] == "retry_budget_reset"]
    assert [event["checkpoint_progress"]["batch"] for event in resets] == [4, 6, 8, 10]
    assert all(event["previous_retries"] == 1 for event in resets)
    assert [event["replacement"] for event in events if event["event"] == "backoff"] == [1, 2, 3, 4]
    assert backoff_delays == [10] * 4
    assert events[-1]["event"] == "completed"
    assert events[-1]["replacement"] == 4
    assert events[-1]["retries_since_progress"] == 0
    rows = checkpoint_utils.load_checkpoints_file(auto_resume.latest_attempt_log_path(str(tmp_path)))
    assert rows[-1]["batch"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("creation_failures", [1, 10])
@pytest.mark.parametrize("max_retries", [1, 3, 5])
async def test_failed_replacement_initialization_consumes_budget(
    tmp_path, monkeypatch, fast_backoff, creation_failures, max_retries
):
    from programmed_training.rl import train_azure

    dataset = ControlledDataset([[MixedRewardGroupBuilder([0.0, 1.0])] for _ in range(8)])
    cfg = _make_config(dataset, tmp_path, save_every=2, max_steps=6)
    monkeypatch.setattr(train_azure, "get_tokenizer", lambda _: _make_mock_session().get_tokenizer())
    service = Service(creation_failures=creation_failures)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    operation = auto_resume.run(
        lambda c, s: train_azure.main(cfg, c, s),
        config=auto_resume.AutoResumeConfig(auto_resume_max_retries=max_retries),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
    )
    if creation_failures >= max_retries:
        with pytest.raises(TrainingEngineError):
            await operation
    else:
        await operation
        assert service.states[service.creations[-1][0]] == ModelState(6, 142)
    expected_sessions = min(creation_failures + 2, max_retries + 1)
    assert len(service.creations) == expected_sessions
    assert len(set(service.closed)) == expected_sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,codes,error_session,error_code", [
    (False, ("engine_dead",), "session_0", "engine_dead"),
    (True, (), "session_0", "engine_dead"),
    (True, ("other",), "session_0", "engine_dead"),
    (True, ("engine_dead",), "session_reference", "engine_dead"),
    (True, ("engine_dead",), "session_0", None),
])
async def test_nonmatching_failures_do_not_replace(
    tmp_path, enabled, codes, error_session, error_code, fast_backoff
):
    service = Service()
    client = service.client()
    session = await client.create_session(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    error = TrainingEngineError("test", error_code=error_code, session_id=error_session)
    async def fail(c, s):
        raise error
    try:
        with pytest.raises(TrainingEngineError) as caught:
            await auto_resume.run(
                fail, config=auto_resume.AutoResumeConfig(auto_resume=enabled, resumable_error_codes=codes),
                log_path=str(tmp_path), client=client, session_id=session,
                client_factory=service.client, session_kwargs={},
            )
        assert caught.value is error
        assert len(service.creations) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("max_retries", [0, 3])
@pytest.mark.parametrize("training_type,user_metadata,basic_tier", [
    ("DeveloperTier", None, True),
    ("GlobalStandard", None, False),
    ("DatazoneStandard", None, False),
    (None, None, False),
    (None, {"DeveloperTier": True}, True),
    (None, {"DeveloperTier": False}, False),
    ("GlobalStandard", {"DeveloperTier": True}, False),
    ("DeveloperTier", {"DeveloperTier": False}, True),
])
async def test_disabled_basic_tier_warns_without_enabling_recovery(
    tmp_path, caplog, enabled, max_retries, training_type, user_metadata, basic_tier
):
    service = Service(failures=0)
    client = service.client()
    kwargs = dict(
        base_model="test", type="training", lora_config=LoRAConfig(rank=16),
        training_type=training_type, user_metadata=user_metadata,
    )
    session = await client.create_session(**kwargs)
    workload = AsyncMock(return_value="completed")
    config = (
        auto_resume.AutoResumeConfig(auto_resume_max_retries=max_retries) if enabled
        else auto_resume.AutoResumeConfig(auto_resume=False, auto_resume_max_retries=max_retries)
    )
    try:
        with caplog.at_level("WARNING", logger=auto_resume.__name__):
            result = await auto_resume.run(
                workload, config=config, log_path=str(tmp_path), client=client,
                session_id=session, client_factory=service.client, session_kwargs=kwargs,
            )
        assert result == "completed"
        workload.assert_awaited_once_with(client, session)
        warnings = [
            record for record in caplog.records
            if record.name == auto_resume.__name__ and "disables automatic recovery" in record.message
        ]
        assert len(warnings) == int(basic_tier and (not enabled or max_retries == 0))
        if warnings:
            assert "DeveloperTier" in warnings[0].message
            assert "auto_resume=True" in warnings[0].message
        assert (tmp_path / "recovery.jsonl").exists() == enabled
        assert len(service.creations) == 1
    finally:
        if not enabled:
            await client.close_session(session)
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,failed_result,journal_error", [
    ("state", None, False),
    ("both", None, False),
    ("sampler", None, False),
    ("state", "state", False),
    ("both", "sampler", False),
    ("state", None, True),
])
async def test_retry_reset_requires_complete_training_checkpoint_commit(
    tmp_path, monkeypatch, kind, failed_result, journal_error
):
    budget = auto_resume.Budget(replacements=2, retries_since_progress=2)
    state = auto_resume.Run(None, budget, str(tmp_path), checkpoint_progress=(0, 2))
    attempt = auto_resume.Attempt(str(tmp_path), "session_0", frozenset({"engine_dead"}), None)
    training = SimpleNamespace()
    for result_kind, method in [
        ("state", "save_state_async"), ("sampler", "save_weights_for_sampler_async"),
    ]:
        result = AsyncMock(
            return_value=SimpleNamespace(path=f"loom://model_0/weights/{result_kind}"),
            side_effect=ValueError("save failed") if failed_result == result_kind else None,
        )
        setattr(training, method, AsyncMock(return_value=SimpleNamespace(result_async=result)))
    if journal_error:
        monkeypatch.setattr(
            checkpoint_utils, "open", Mock(side_effect=OSError("journal failed")), raising=False
        )
    run_token, attempt_token = auto_resume._run.set(state), auto_resume._attempt.set(attempt)
    try:
        operation = checkpoint_utils.save_checkpoint_async(
            training, name="new", log_path=str(tmp_path), loop_state={"batch": 4}, kind=kind,
        )
        if failed_result or journal_error:
            with pytest.raises(OSError if journal_error else ValueError):
                await operation
        else:
            await operation
        reset = kind != "sampler" and not failed_result and not journal_error
        assert budget.retries_since_progress == (0 if reset else 2)
        assert budget.replacements == 2
        assert state.checkpoint_progress == ((0, 4) if reset else (0, 2))
        assert (tmp_path / "recovery.jsonl").exists() == reset
        if reset:
            row = checkpoint_utils.load_checkpoints_file(str(tmp_path))[-1]
            event = json.loads((tmp_path / "recovery_status.json").read_text())
            assert event["event"] == "retry_budget_reset"
            assert event["checkpoint"] == row["state_path"]
            assert event["previous_retries"] == 2
            metadata = json.loads((tmp_path / "run_meta.json").read_text())
            assert metadata["auto_resume"]["retries_since_progress"] == 0
    finally:
        auto_resume._attempt.reset(attempt_token)
        auto_resume._run.reset(run_token)


@pytest.mark.parametrize("previous,loop_state,reset", [
    (None, {"batch": 0}, True),
    ((0, 4), {"batch": 4}, False),
    ((0, 4), {"batch": 3}, False),
    ((0, 4), {"epoch": 1, "batch": 0}, True),
    ((1, 0), {"epoch": 0, "batch": 9}, False),
])
def test_retry_reset_uses_checkpoint_cursor_not_checkpoint_identity(
    tmp_path, previous, loop_state, reset
):
    budget = auto_resume.Budget(replacements=2, retries_since_progress=2)
    state = auto_resume.Run(None, budget, str(tmp_path), checkpoint_progress=previous)
    attempt = auto_resume.Attempt(str(tmp_path), "session_0", frozenset({"engine_dead"}), None)
    run_token, attempt_token = auto_resume._run.set(state), auto_resume._attempt.set(attempt)
    try:
        auto_resume.checkpoint_committed(
            str(tmp_path), {**loop_state, "state_path": "loom://model_0/weights/new"},
        )
        assert budget.retries_since_progress == (0 if reset else 2)
        assert budget.replacements == 2
    finally:
        auto_resume._attempt.reset(attempt_token)
        auto_resume._run.reset(run_token)


@pytest.mark.asyncio
async def test_checkpoint_fallback_does_not_reset_the_durable_high_water_mark(
    tmp_path, fast_backoff
):
    from programmed_training.rl.train_azure import AzureSDKTrainingClient

    service = Service(failures=0)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)

    async def workload(c, s):
        progress = 1 if s == session else 3
        await checkpoint_utils.save_checkpoint_async(
            AzureSDKTrainingClient(c, s, object()), name="saved",
            log_path=auto_resume.current_log_path(str(tmp_path)), loop_state={"batch": progress},
        )
        if s == "session_1":
            service.unavailable.add((s, "saved"))
        raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)

    with pytest.raises(TrainingEngineError):
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(auto_resume_max_retries=1),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
        )
    assert len(service.creations) == 3
    assert service.creations[-1][1]["from_checkpoint"]["source_session_id"] == session
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "retry_budget_reset" for event in events) == 1
    assert events[-1]["event"] == "exhausted"
    assert events[-1]["replacement"] == 2
    assert events[-1]["retries_since_progress"] == 1
    assert len(set(service.closed)) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("owning_session", [False, True])
async def test_supervised_attempt_owns_prepared_client_cleanup(
    tmp_path, monkeypatch, fail, owning_session
):
    service = Service(failures=0)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    training = SimpleNamespace(close_poller=AsyncMock())
    main = AsyncMock(side_effect=ValueError("training failed") if fail else None)
    session_closer = AsyncMock(wraps=client.close_session)
    monkeypatch.setattr("programmed_training.supervised.train.main", main)
    operation = auto_resume.run_supervised(
        SimpleNamespace(log_path=str(tmp_path), save_every=1, max_wall_clock_seconds=None),
        training, recovery_config=auto_resume.AutoResumeConfig(),
        client=client, session_id=session, client_kwargs={}, session_kwargs=kwargs,
        initial_session_closer=(lambda: session_closer(session)) if owning_session else None,
    )
    if fail:
        with pytest.raises(ValueError, match="training failed"):
            await operation
    else:
        await operation
    training.close_poller.assert_awaited_once()
    assert session_closer.await_count == int(owning_session)
    assert service.closed == [session]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["directory", "checkpoint", "progress", "metadata"])
@pytest.mark.parametrize("owning_session", [False, True])
async def test_supervised_setup_failure_closes_prepared_resources(
    tmp_path, monkeypatch, failure_stage, owning_session
):
    service = Service(failures=0)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    training = SimpleNamespace(close_poller=AsyncMock())
    main = AsyncMock()
    session_closer = AsyncMock(wraps=client.close_session)
    monkeypatch.setattr("programmed_training.supervised.train.main", main)
    close = AsyncMock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    error_type = OSError
    if failure_stage == "directory":
        monkeypatch.setattr(auto_resume.os, "makedirs", Mock(side_effect=OSError("directory unavailable")))
    elif failure_stage == "checkpoint":
        monkeypatch.setattr(
            checkpoint_utils, "get_last_checkpoint", Mock(side_effect=OSError("ledger unavailable"))
        )
    elif failure_stage == "progress":
        error_type = ValueError
        monkeypatch.setattr(
            checkpoint_utils, "get_last_checkpoint",
            Mock(return_value={"state_path": "session_source/checkpoint", "batch": "invalid"}),
        )
    else:
        error_type = json.JSONDecodeError
        (tmp_path / "run_meta.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(error_type):
        await auto_resume.run_supervised(
            SimpleNamespace(log_path=str(tmp_path), save_every=1, max_wall_clock_seconds=None),
            training, recovery_config=auto_resume.AutoResumeConfig(),
            client=client, session_id=session, client_kwargs={}, session_kwargs=kwargs,
            initial_session_closer=(lambda: session_closer(session)) if owning_session else None,
        )
    main.assert_not_awaited()
    training.close_poller.assert_awaited_once()
    assert session_closer.await_count == int(owning_session)
    assert service.closed == [session]
    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_cleanup_failure_preserves_original_error(tmp_path, monkeypatch, caplog):
    client = SimpleNamespace(
        close_session=AsyncMock(side_effect=RuntimeError("close failed")),
        close=AsyncMock(),
    )
    error = OSError("ledger unavailable")
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", Mock(side_effect=error))
    workload = AsyncMock()

    with pytest.raises(OSError) as caught:
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
            client=client, session_id="session_primary", client_factory=Mock(),
            session_kwargs={},
        )
    assert caught.value is error
    workload.assert_not_awaited()
    client.close_session.assert_awaited_once_with("session_primary")
    client.close.assert_awaited_once()
    assert "Recovery setup cleanup failed" in caplog.text


@pytest.mark.asyncio
async def test_reference_metadata_and_initial_session_owner_stay_per_attempt(tmp_path, fast_backoff):
    service = Service(failures=0)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    initial_close = AsyncMock(wraps=client.close_session)
    (tmp_path / "run_meta.json").write_text(json.dumps({"azure": {"session_id": session}}))

    async def workload(c, s):
        auto_resume.record_reference_session(str(tmp_path), f"reference_{s}")
        if s == session:
            raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)

    await auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
        client=client, session_id=session, client_factory=service.client,
        session_kwargs=kwargs, initial_session_closer=lambda: initial_close(session),
    )
    initial_close.assert_awaited_once_with(session)
    assert service.closed == [row[0] for row in service.creations]
    paths = auto_resume.attempt_log_paths(str(tmp_path))
    assert len(paths) == 2
    for path, creation in zip(paths, service.creations, strict=True):
        session_id = creation[0]
        metadata = json.loads((Path(path) / "run_meta.json").read_text())
        assert metadata["azure"] == {
            "session_id": session_id, "reference_session_id": f"reference_{session_id}",
            "from_checkpoint": None,
        }


@pytest.mark.asyncio
async def test_cold_restart_preserves_original_warm_start(tmp_path, fast_backoff):
    service = Service()
    client = service.client()
    source = await client.create_session(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    service.checkpoints[(source, "warm")] = ModelState(10, 170)
    original = FromCheckpoint(source_session_id=source, checkpoint_id="warm")
    kwargs = dict(base_model="test", type="training", from_checkpoint=original, lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    async def workload(c, s):
        if s == session:
            raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)
        assert checkpoint_utils.get_last_checkpoint(auto_resume.current_log_path(str(tmp_path))) is None
        assert service.states[s] == ModelState(10, 170)
    await auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
    )
    assert service.creations[-1][1]["from_checkpoint"]["checkpoint_id"] == "warm"
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert any(event["event"] == "restart_original_input" for event in events)


@pytest.mark.asyncio
async def test_selection_falls_back_without_splitting_checkpoint_and_progress(tmp_path):
    service = Service()
    service.checkpoints[("session_0", "older")] = ModelState(2, 114)
    service.checkpoints[("session_1", "newer")] = ModelState(4, 128)
    service.unavailable.add(("session_1", "newer"))
    rows = [
        {"batch": 2, "state_path": "loom://model_0/weights/older"},
        {"batch": 4, "state_path": "loom://model_1/weights/newer"},
        {"sampler_path": "sampler"},
    ]
    (tmp_path / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    async with service.client() as client:
        selected = await auto_resume._select_checkpoint(client, str(tmp_path))
        assert selected == rows[0]
        service.unavailable.add(("session_0", "older"))
        with pytest.raises(RuntimeError, match="No committed"):
            await auto_resume._select_checkpoint(client, str(tmp_path))
        service.list_error = 401
        from azure.core.exceptions import ClientAuthenticationError
        with pytest.raises(ClientAuthenticationError):
            await auto_resume._select_checkpoint(client, str(tmp_path))


@pytest.mark.asyncio
async def test_nested_rollout_failure_interrupts_pending_sibling_and_drains(tmp_path, fast_backoff):
    service = Service()
    client = service.client()
    session = await client.create_session(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    stopped = asyncio.Event()
    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    async def engine_loss(s):
        await asyncio.sleep(0)
        raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)
    async def workload(c, s):
        if s == session:
            await auto_resume.gather(blocked(), engine_loss(s), return_exceptions=True)
        else:
            assert stopped.is_set()
    await asyncio.wait_for(auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16)),
    ), timeout=5)
    assert len(service.creations) == 2


@pytest.mark.parametrize("codes", ["engine_dead", ("",), (" engine_dead",), (None,), None])
def test_invalid_error_allowlist_is_rejected(codes):
    with pytest.raises((ValueError, TypeError)):
        auto_resume.AutoResumeConfig(resumable_error_codes=codes)


@pytest.mark.asyncio
async def test_real_supervised_loop_restores_optimizer_and_cursor(tmp_path, monkeypatch, fast_backoff):
    from azure.ai.finetuningsessions.models import Datum, ModelInput, ModelInputChunk, TensorData
    from programmed_training.rl.train_azure import AzureSDKTrainingClient
    from programmed_training.supervised import train
    from programmed_training.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

    class Dataset(SupervisedDataset):
        def __len__(self):
            return 8

        def get_batch(self, index):
            return [Datum(
                model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2])]),
                loss_fn_inputs={
                    "target_tokens": TensorData(data=[2, 3]),
                    "weights": TensorData(data=[1.0, 1.0]),
                },
            )]

        def set_epoch(self, seed=0):
            pass

    @chz.chz
    class Builder(SupervisedDatasetBuilder):
        def build(self):
            return Dataset(), None

    cfg = train.Config(
        log_path=str(tmp_path), model_name="test", dataset_builder=Builder(),
        max_steps=6, save_every=2, eval_every=0, sanitize_logs=True, pipeline_depth=2,
    )
    tokenizer = _make_mock_session().get_tokenizer()
    monkeypatch.setattr(train, "get_tokenizer", lambda _: tokenizer)
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    async def workload(c, s):
        await train.main(cfg, AzureSDKTrainingClient(c, s, tokenizer))
    await auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
    )
    assert len(service.creations) == 2
    assert service.creations[1][2] == ModelState(2, 114)
    assert service.states[service.creations[1][0]] == ModelState(6, 142)
    final = checkpoint_utils.get_last_checkpoint(auto_resume.latest_attempt_log_path(str(tmp_path)))
    assert (final["epoch"], final["batch"]) == (0, 6)


class NerService(Service):
    def __init__(self, tokenizer):
        super().__init__(failures=0)
        self.tokenizer = tokenizer
        self.first_turns = 0
        self.finalizations = {}
        self.injected = False

    def reply(self, request):
        response = super().reply(request)
        if request.method != "POST" or not urlsplit(request.url).path.endswith("/sample"):
            return response
        session = urlsplit(request.url).path.split("/sessions/")[1].split("/")[0]
        body = json.loads(request.content)
        prompt = self.tokenizer.decode([
            token for chunk in body["prompt"]["chunks"] for token in chunk["tokens"]
        ])
        last_user = prompt.rsplit("Source document:", 1)[-1]
        second_turn = "<tool_response>" in last_user
        request_id = response.json()["request_id"]
        if second_turn:
            self.finalizations[session] = self.finalizations.get(session, 0) + 1
            if session == "session_0" and self.states[session].weights >= 3 and not self.injected:
                self.injected = True
                self.failed_states.append((session, ModelState(**vars(self.states[session]))))
                self.requests[request_id] = ("fail", {})
                return response
            spans = re.findall(r'"span_id"\s*:\s*"([^"]+)"', last_user)
            entities = [{"label": "GIVENNAME", "span_ids": spans[-1:]}] if spans else []
            function, parameter, value = "finalize", "entities", entities
        else:
            self.first_turns += 1
            proposals = [{"text": "Jane", "label": "GIVENNAME"}] if self.first_turns % 2 else []
            function, parameter, value = "ground_entities", "proposals", proposals
        text = (
            f"<tool_call>\n<function={function}>\n<parameter={parameter}>\n"
            f"{json.dumps(value)}\n</parameter>\n</function>\n</tool_call><|im_end|>"
        )
        tokens = self.tokenizer.encode(text)
        self.requests[request_id] = ("sample", {
            "sequences": [{"tokens": tokens, "logprobs": [-0.5] * len(tokens), "stop_reason": "stop"}]
        })
        return response


@pytest.mark.asyncio
async def test_tool_ner_launcher_recovers_real_two_turn_async_flow(
    tmp_path, monkeypatch, fast_backoff
):
    from programmed_training.recipes.tool_ner_rl import ner_env, train_azure as recipe
    from programmed_training.recipes.tool_ner_rl.ner_task import GoldEntity, NerTask
    from programmed_training.rl import train_azure
    from tests.test_qwen35_schema_tool_parsing import _CharacterTokenizer

    tokenizer = _CharacterTokenizer()
    service = NerService(tokenizer)
    monkeypatch.setattr("azure.ai.finetuningsessions.aio.FineTuningSessionClient", lambda **_: service.client())
    monkeypatch.setattr(auto_resume, "FineTuningSessionClient", lambda **_: service.client())
    monkeypatch.setenv("AZURE_AI_API_KEY", "offline")
    monkeypatch.setattr(train_azure, "get_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(ner_env.tokenizer_utils, "get_tokenizer", lambda _: tokenizer)
    task = NerTask(
        uid="offline-recovery", source_text="Jane called.",
        labels=("GIVENNAME",), gold_entities=frozenset({GoldEntity(0, 4, "GIVENNAME")}),
    )
    builder = ner_env.NerEnvGroupBuilder(
        task=task, model_name="test", renderer_name="qwen3_5_disable_thinking",
        group_size=4, max_trajectory_tokens=100_000,
    )
    dataset = ControlledDataset([[builder, builder] for _ in range(12)])
    async def build(_self):
        return dataset, None
    monkeypatch.setattr(ner_env.NerDatasetBuilder, "__call__", build)
    config = recipe.CLIConfig(
        project_endpoint="https://unit.invalid", model_name="test",
        renderer_name="qwen3_5_disable_thinking",
        auto_resume=True, max_steps_off_policy=1,
        log_path=str(tmp_path / "run"), group_size=4, pods_keep=2,
        groups_per_batch=2, max_steps=6, save_every=2, eval_every=0,
        max_wall_clock_seconds=120, max_oversample_rounds=20,
    )
    await asyncio.wait_for(recipe.cli_main(config), timeout=180)
    assert service.injected, "The test must reach the real second-turn failure site"
    assert len(service.creations) == 2
    replacement = service.creations[1][0]
    assert service.creations[1][1]["from_checkpoint"]["checkpoint_id"] == "2"
    assert service.finalizations[replacement] > 0
    assert service.states[replacement].weights == 6
    replacement_path = auto_resume.latest_attempt_log_path(str(tmp_path / "run"))
    rows = checkpoint_utils.load_checkpoints_file(replacement_path)
    assert rows[-1]["batch"] == 6
    assert (tmp_path / "run/train_filter_manifest.json").exists()
    assert json.loads((tmp_path / "run/run_meta.json").read_text())["azure"]["session_id"] != replacement
    assert json.loads((Path(replacement_path) / "run_meta.json").read_text())["azure"]["session_id"] == replacement


@pytest.mark.asyncio
async def test_dpo_recovery_keeps_original_frozen_reference(tmp_path, monkeypatch, fast_backoff):
    from types import SimpleNamespace
    from programmed_training.recipes.preference.dpo import train_azure as recipe
    from tests.test_dpo_recipe import _datum

    class Dataset:
        def __len__(self):
            return 8
        def get_batch(self, index):
            return [_datum([0.0, 1.0]), _datum([0.0, 1.0])]
        def set_epoch(self, epoch):
            pass

    tmp_path = tmp_path / "run"
    builder = SimpleNamespace(build=lambda: (Dataset(), None))
    service = Service()
    monkeypatch.setattr(recipe, "get_dataset_builder", lambda *_: builder)
    monkeypatch.setattr(recipe, "get_tokenizer", lambda _: _make_mock_session().get_tokenizer())
    monkeypatch.setattr("azure.ai.finetuningsessions.aio.FineTuningSessionClient", lambda **_: service.client())
    monkeypatch.setenv("AZURE_AI_API_KEY", "offline")
    await recipe.cli_main(recipe.CLIConfig(
        project_endpoint="https://unit.invalid", auto_resume=True,
        log_path=str(tmp_path), max_steps=6, save_every=2, eval_every=0,
    ))
    assert len(service.creations) == 3
    replacement, request, restored = service.creations[1]
    assert request["from_checkpoint"]["checkpoint_id"] == "step_000002"
    assert restored == ModelState(2, 114)
    reference, request, restored = service.creations[2]
    assert request["from_checkpoint"]["checkpoint_id"] == "dpo_reference"
    assert restored == ModelState(0, 100)
    assert service.states[reference] == ModelState(0, 100)
    assert service.states[replacement] == ModelState(6, 142)
    final = checkpoint_utils.get_last_checkpoint(auto_resume.latest_attempt_log_path(str(tmp_path)))
    assert final["completed_updates"] == 6
    assert final["reference_state_path"] == "loom://model_0/weights/dpo_reference"


@pytest.mark.asyncio
async def test_cleanup_timeout_prevents_overlapping_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_resume, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    service = Service()
    client = service.client()
    close = AsyncMock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    release = asyncio.Event()
    started = asyncio.Event()
    stubborn_task = None
    async def stubborn():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
    error = TrainingEngineError("lost", error_code="engine_dead", session_id=session)
    async def workload(c, s):
        nonlocal stubborn_task
        stubborn_task = auto_resume.create_task(stubborn())
        await started.wait()
        raise error
    try:
        with pytest.raises(TrainingEngineError) as caught:
            await auto_resume.run(
                workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
                log_path=str(tmp_path), client=client, session_id=session,
                client_factory=service.client, session_kwargs=kwargs,
            )
        assert caught.value is error
        assert len(service.creations) == 1
        assert json.loads((tmp_path / "recovery_status.json").read_text())["event"] == "failed"
        close.assert_awaited_once()
    finally:
        release.set()
        if stubborn_task is not None:
            await stubborn_task
        await client.close_session(session)
        await client.close()


@pytest.mark.asyncio
async def test_cancellation_is_terminal_and_drains_workers(tmp_path):
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    started = asyncio.Event()
    stopped = asyncio.Event()
    async def workload(c, s):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    task = asyncio.create_task(auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
    ))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert service.closed == [session]
    assert len(service.creations) == 1
    assert json.loads((tmp_path / "recovery_status.json").read_text())["event"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("save_progress", [False, True])
async def test_staged_recipe_shares_budget_without_replaying_completed_stages(
    tmp_path, fast_backoff, save_progress
):
    from programmed_training.rl.train_azure import AzureSDKTrainingClient

    services = []
    completed = []
    @auto_resume.shared_run
    async def staged():
        for stage in range(4):
            log_path = tmp_path / str(stage)
            log_path.mkdir()
            service = Service()
            services.append(service)
            client = service.client()
            kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
            session = await client.create_session(**kwargs)
            async def workload(c, s):
                if s == session:
                    raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)
                if save_progress:
                    await checkpoint_utils.save_checkpoint_async(
                        AzureSDKTrainingClient(c, s, object()), name="progress",
                        log_path=auto_resume.current_log_path(str(log_path)),
                        loop_state={"batch": 1},
                    )
            await auto_resume.run(
                workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
                log_path=str(log_path), client=client, session_id=session,
                client_factory=service.client, session_kwargs=kwargs,
            )
            completed.append(stage)
    if save_progress:
        await staged()
        assert completed == [0, 1, 2, 3]
        assert [len(service.creations) for service in services] == [2] * 4
    else:
        with pytest.raises(TrainingEngineError):
            await staged()
        assert completed == [0, 1, 2]
        assert [len(service.creations) for service in services] == [2, 2, 2, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_fatal", [False, True])
async def test_unrelated_fatal_rollout_is_not_hidden_by_engine_loss(tmp_path, earlier_fatal):
    attempt = auto_resume.Attempt(str(tmp_path), "session_primary", frozenset({"engine_dead"}), None)
    token = auto_resume._attempt.set(attempt)
    fatal_seen = asyncio.Event()
    fatal = ValueError("invalid tool response")
    engine = TrainingEngineError("lost", error_code="engine_dead", session_id=attempt.session_id)

    async def fail():
        fatal_seen.set()
        raise fatal

    async def lose_engine():
        if earlier_fatal:
            await fatal_seen.wait()
            await asyncio.sleep(0)
        raise engine

    try:
        with pytest.raises(ExceptionGroup) as caught:
            await auto_resume.gather(
                fail(), lose_engine(), return_exceptions=True,
                retryable_error=lambda error: isinstance(error, TimeoutError),
            )
        assert set(caught.value.exceptions) == {fatal, engine}
    finally:
        auto_resume._attempt.reset(token)


@pytest.mark.asyncio
async def test_attached_session_without_checkpoint_cannot_cold_restart(tmp_path, fast_backoff):
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)

    async def fail(c, s):
        raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)

    with pytest.raises(RuntimeError, match="attached session"):
        await auto_resume.run(
            fail, config=auto_resume.AutoResumeConfig(auto_resume=True),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
            restart_from_original_input=False,
        )
    assert len(service.creations) == 1


@pytest.mark.asyncio
async def test_failed_reference_allocation_is_closed_without_restarting_primary(tmp_path):
    service = Service(creation_failures=1)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)

    async def workload(c, s):
        await auto_resume.create_owned_session(c, **kwargs)

    with pytest.raises(TrainingEngineError) as caught:
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
        )
    assert caught.value.session_id == "session_1"
    assert len(service.creations) == 2
    assert service.closed == ["session_1", session]


@pytest.mark.asyncio
@pytest.mark.parametrize("expire", [False, True])
async def test_recovery_retains_original_deadline(tmp_path, monkeypatch, fast_backoff, expire):
    clock = [auto_resume.time.time()]
    original_deadline = clock[0] + 10
    monkeypatch.setattr(auto_resume.time, "time", lambda: clock[0])
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    failure = TrainingEngineError("lost", error_code="engine_dead", session_id=session)

    async def workload(c, s):
        assert auto_resume.current_deadline() == original_deadline
        if s == session:
            clock[0] += 11 if expire else 1
            raise failure

    operation = auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
        max_wall_clock_seconds=10,
    )
    if expire:
        with pytest.raises(TrainingEngineError) as caught:
            await operation
        assert caught.value is failure
    else:
        await operation
    assert len(service.creations) == (1 if expire else 2)


@pytest.mark.asyncio
async def test_active_workload_finalizes_after_original_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from programmed_training.rl.train_azure import AzureSDKTrainingClient

    clock = [auto_resume.time.time()]
    monkeypatch.setattr(
        auto_resume, "time",
        SimpleNamespace(time=lambda: clock[0], monotonic=auto_resume.time.monotonic),
    )
    service = Service(failures=0)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    finalized = []

    async def workload(c, s):
        deadline = auto_resume.current_deadline()
        assert deadline is not None
        clock[0] = deadline + 0.02
        await checkpoint_utils.save_checkpoint_async(
            AzureSDKTrainingClient(c, s, object()), name="final",
            log_path=auto_resume.current_log_path(str(tmp_path)), loop_state={"batch": 1},
        )
        finalized.append("evaluation")
        return "graceful completion"

    result = await asyncio.wait_for(auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
        client=client, session_id=session, client_factory=service.client,
        session_kwargs=kwargs, max_wall_clock_seconds=0.02,
    ), timeout=5)
    assert result == "graceful completion"
    assert finalized == ["evaluation"]
    assert checkpoint_utils.get_last_checkpoint(str(tmp_path))["name"] == "final"
    assert service.closed == [session]
    assert len(service.creations) == 1
    assert json.loads((tmp_path / "recovery_status.json").read_text())["event"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("return_exceptions", [False, True])
@pytest.mark.parametrize("exit_kind", ["caller_cancelled", "child_cancelled", "fatal"])
async def test_gather_drains_children_before_returning_to_active_attempt(
    tmp_path, return_exceptions, exit_kind
):
    attempt = auto_resume.Attempt(str(tmp_path), "session_primary", frozenset({"engine_dead"}), None)
    token = auto_resume._attempt.set(attempt)
    started = asyncio.Event()
    stopped = asyncio.Event()
    release = asyncio.Event()

    async def blocked():
        started.set()
        try:
            await release.wait()
        finally:
            stopped.set()

    async def exit_child():
        await started.wait()
        if exit_kind == "child_cancelled":
            raise asyncio.CancelledError()
        raise ValueError("fatal sibling")

    children = [asyncio.create_task(blocked())]
    if exit_kind != "caller_cancelled":
        children.append(asyncio.create_task(exit_child()))
    caller = asyncio.create_task(auto_resume.gather(
        *children, return_exceptions=return_exceptions,
    ))
    try:
        await started.wait()
        if exit_kind == "fatal" and return_exceptions:
            release.set()
            results = await caller
            assert isinstance(results[1], ValueError)
        else:
            if exit_kind == "caller_cancelled":
                caller.cancel()
            with pytest.raises(ValueError if exit_kind == "fatal" else asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=2)
        assert stopped.is_set()
        assert all(child.done() for child in children)
        assert attempt.accepting_work
        assert not attempt.failed_gathers
    finally:
        for task in [caller, *children]:
            if not task.done():
                task.cancel()
        await asyncio.gather(caller, *children, return_exceptions=True)
        auto_resume._attempt.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume_enabled", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
async def test_dpo_pre_supervisor_metadata_failure_closes_session(
    tmp_path, monkeypatch, auto_resume_enabled, cancelled, close_fails
):
    from programmed_training.recipes.preference.dpo import train_azure as recipe

    service = Service(failures=0)
    client = service.client()
    close_session = AsyncMock(wraps=client.close_session)
    if close_fails:
        close_session.side_effect = RuntimeError("session close failed")
    monkeypatch.setattr(client, "close_session", close_session)
    close_client = AsyncMock(wraps=client.close)
    monkeypatch.setattr(client, "close", close_client)
    monkeypatch.setenv("AZURE_AI_API_KEY", "offline")
    monkeypatch.setattr(
        "azure.ai.finetuningsessions.aio.FineTuningSessionClient", lambda **_: client,
    )
    monkeypatch.setattr(
        recipe, "get_dataset_builder", lambda *_: SimpleNamespace(build=lambda: ([object()], None)),
    )
    monkeypatch.setattr(recipe, "get_tokenizer", lambda _: object())
    primary = asyncio.CancelledError("metadata cancelled") if cancelled else OSError("metadata failed")
    original_write = recipe.write_run_metadata

    def write_metadata(path, metadata):
        if "azure" in metadata:
            raise primary
        original_write(path, metadata)

    monkeypatch.setattr(recipe, "write_run_metadata", write_metadata)
    supervisor = AsyncMock()
    monkeypatch.setattr(auto_resume, "run", supervisor)
    with pytest.raises(type(primary)) as caught:
        await recipe.cli_main(recipe.CLIConfig(
            project_endpoint="https://unit.invalid", log_path=str(tmp_path / "run"),
            auto_resume=auto_resume_enabled,
        ))
    assert caught.value is primary
    supervisor.assert_not_awaited()
    close_session.assert_awaited_once_with("session_0")
    close_client.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume_enabled", [False, True])
@pytest.mark.parametrize("failure_stage", ["metadata", "tokenizer", "adapter", "idle", "training", "success"])
@pytest.mark.parametrize("close_fails", [False, True])
async def test_visual_sft_session_ownership(
    tmp_path, monkeypatch, auto_resume_enabled, failure_stage, close_fails
):
    from programmed_training.recipes.visual_spatial import train_sft_azure as recipe

    service = Service(failures=0)
    client = service.client()
    close_session = AsyncMock(wraps=client.close_session)
    if close_fails:
        close_session.side_effect = RuntimeError("session close failed")
    monkeypatch.setattr(client, "close_session", close_session)
    close_client = AsyncMock(wraps=client.close)
    monkeypatch.setattr(client, "close", close_client)
    monkeypatch.setenv("AZURE_AI_API_KEY", "offline")
    monkeypatch.setattr(
        "azure.ai.finetuningsessions.aio.FineTuningSessionClient", lambda **_: client,
    )
    monkeypatch.setattr(recipe.model_info, "require_vision_language_model", lambda *_, **__: None)
    monkeypatch.setattr(recipe, "VisualSpatialSftBuilder", lambda **_: object())
    monkeypatch.setattr(
        recipe, "VisualSpatialAccuracyEvaluator",
        lambda **_: SimpleNamespace(preflight=lambda **_: SimpleNamespace(examples=1, max_total_tokens=10)),
    )
    monkeypatch.setattr(recipe.train, "prepare_datasets", lambda _: ([object()], None))
    primary = (
        asyncio.CancelledError("idle cancelled") if failure_stage == "idle"
        else OSError(f"{failure_stage} failed")
    )
    original_write = recipe._write_run_meta

    def write_metadata(path, metadata):
        if failure_stage == "metadata" and metadata["azure"]["session_id"] is not None:
            raise primary
        original_write(path, metadata)

    monkeypatch.setattr(recipe, "_write_run_meta", write_metadata)
    monkeypatch.setattr(
        recipe, "get_tokenizer", Mock(side_effect=primary) if failure_stage == "tokenizer" else lambda _: object(),
    )
    if failure_stage == "adapter":
        monkeypatch.setattr(recipe, "AzureSDKTrainingClient", Mock(side_effect=primary))
    if failure_stage == "idle":
        async def cancelled_sleep(_):
            raise primary
        monkeypatch.setattr(
            recipe, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": cancelled_sleep}),
        )
    monkeypatch.setattr(
        recipe.train, "main", AsyncMock(side_effect=primary if failure_stage == "training" else None),
    )
    supervisor = AsyncMock(wraps=auto_resume.run_supervised)
    monkeypatch.setattr(auto_resume, "run_supervised", supervisor)
    cfg = recipe.CLIConfig(
        project_endpoint="https://unit.invalid", model_context_length=1024,
        renderer_name="test", log_path=str(tmp_path / "run"),
        auto_resume=auto_resume_enabled, initial_idle_seconds=1 if failure_stage == "idle" else 0,
    )
    if failure_stage == "success" and not close_fails:
        await recipe.cli_main(cfg)
    else:
        with pytest.raises(RuntimeError if failure_stage == "success" else type(primary)) as caught:
            await recipe.cli_main(cfg)
        if failure_stage != "success":
            assert caught.value is primary
    if failure_stage in {"training", "success"}:
        supervisor.assert_awaited_once()
    else:
        supervisor.assert_not_awaited()
    close_session.assert_awaited_once_with("session_0")
    assert close_client.await_count == (2 if auto_resume_enabled and supervisor.await_count else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_during_cancellation", [False, True])
async def test_top_level_worker_failure_is_not_hidden_by_recovery(
    tmp_path, fast_backoff, fail_during_cancellation
):
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    engine = TrainingEngineError("lost", error_code="engine_dead", session_id=session)
    fatal = ValueError("invalid worker state")
    started = asyncio.Event()

    async def worker():
        started.set()
        if fail_during_cancellation:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise fatal
        raise fatal

    async def lose_engine():
        if fail_during_cancellation:
            await started.wait()
        raise engine

    async def workload(c, s):
        tasks = [
            auto_resume.create_task(lose_engine()),
            auto_resume.create_task(worker()),
        ]
        if not fail_during_cancellation:
            await asyncio.sleep(0)
        await auto_resume.gather(*tasks)

    with pytest.raises(ExceptionGroup) as caught:
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
        )
    assert set(caught.value.exceptions) == {engine, fatal}
    assert len(service.creations) == 1
    assert service.closed == [session]
    assert json.loads((tmp_path / "recovery_status.json").read_text())["event"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("return_exceptions", [False, True])
@pytest.mark.parametrize("retryable_sibling", [False, True])
async def test_nested_rollout_cancellation_preserves_failure_policy(
    tmp_path, fast_backoff, return_exceptions, retryable_sibling
):
    service = Service(failures=0)
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    engine = TrainingEngineError("lost", error_code="engine_dead", session_id=session)
    sibling_error = TimeoutError("sample timed out") if retryable_sibling else ValueError("invalid tool cleanup")
    started = asyncio.Event()

    async def sibling():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise sibling_error

    async def lose_engine():
        await started.wait()
        raise engine

    async def rollout_group():
        await auto_resume.gather(
            lose_engine(), sibling(), return_exceptions=return_exceptions,
            retryable_error=lambda error: isinstance(error, TimeoutError),
        )

    async def workload(c, s):
        if s == session:
            await auto_resume.gather(rollout_group())

    operation = auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(), log_path=str(tmp_path),
        client=client, session_id=session, client_factory=service.client,
        session_kwargs=kwargs,
    )
    recovered = return_exceptions and retryable_sibling
    if recovered:
        await operation
    else:
        with pytest.raises(ExceptionGroup) as caught:
            await operation
        assert set(caught.value.exceptions) == {engine, sibling_error}
    assert len(service.creations) == (2 if recovered else 1)
    assert set(service.closed) == {row[0] for row in service.creations}
    assert json.loads((tmp_path / "recovery_status.json").read_text())["event"] == (
        "completed" if recovered else "failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("include_fatal", [False, True])
async def test_exception_group_requires_every_error_to_be_recoverable(
    tmp_path, fast_backoff, include_fatal
):
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)

    async def workload(c, s):
        if s == session:
            errors = [TrainingEngineError("lost", error_code="engine_dead", session_id=s)]
            if include_fatal:
                errors.append(ValueError("invalid tool state"))
            raise ExceptionGroup("TaskGroup evaluation failed", errors)

    operation = auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
    )
    if include_fatal:
        with pytest.raises(ExceptionGroup):
            await operation
    else:
        await operation
    assert len(service.creations) == (1 if include_fatal else 2)


@pytest.mark.asyncio
async def test_cleanup_failure_still_closes_other_resources_and_primary(tmp_path):
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    failed_close = AsyncMock(side_effect=RuntimeError("secondary close failed"))
    other_close = AsyncMock()
    engine = TrainingEngineError("lost", error_code="engine_dead", session_id=session)

    async def workload(c, s):
        auto_resume.defer_close(failed_close)
        auto_resume.defer_close(other_close)
        raise engine

    with pytest.raises(TrainingEngineError) as caught:
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
        )
    assert caught.value is engine
    failed_close.assert_awaited_once()
    other_close.assert_awaited_once()
    assert service.closed == [session]
    assert len(service.creations) == 1
    events = [json.loads(line) for line in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert any(row["event"] == "cleanup_failed" for row in events)
    assert events[-1]["event"] == "failed"


@pytest.mark.asyncio
async def test_remote_loggers_are_separate_grouped_and_closed_on_failure(
    tmp_path, monkeypatch, fast_backoff
):
    from programmed_training.utils import ml_log

    wandb = SimpleNamespace(init=Mock(return_value=Mock()), finish=Mock())
    monkeypatch.setattr(ml_log, "_wandb_available", True)
    monkeypatch.setattr(ml_log, "wandb", wandb)
    monkeypatch.setenv("WANDB_API_KEY", "offline")
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)

    async def workload(c, s):
        ml_log.setup_logging(
            auto_resume.current_log_path(str(tmp_path)), wandb_project="offline",
            do_configure_logging_module=False,
        )
        if s == session:
            raise TrainingEngineError("lost", error_code="engine_dead", session_id=s)

    await auto_resume.run(
        workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
        log_path=str(tmp_path), client=client, session_id=session,
        client_factory=service.client, session_kwargs=kwargs,
    )
    assert wandb.finish.call_args_list == [call(exit_code=1), call(exit_code=0)]
    assert wandb.init.call_count == 2
    initial, replacement = [invocation.kwargs for invocation in wandb.init.call_args_list]
    assert initial["group"] == replacement["group"]
    assert initial["dir"] != replacement["dir"]


def test_trace_reinitialization_stops_previous_writer(tmp_path):
    from programmed_training.utils import trace

    trace.trace_init(flush_interval_sec=0.01, output_file=str(tmp_path / "initial.jsonl"))
    initial = trace._trace_collector
    try:
        trace.trace_init(flush_interval_sec=0.01, output_file=str(tmp_path / "replacement.jsonl"))
        assert not initial.flusher_thread.is_alive()
        assert trace._trace_collector.flusher_thread.is_alive()
    finally:
        trace.trace_shutdown()


@pytest.mark.asyncio
async def test_journal_failure_does_not_skip_session_cleanup(tmp_path, monkeypatch):
    service = Service()
    client = service.client()
    kwargs = dict(base_model="test", type="training", lora_config=LoRAConfig(rank=16))
    session = await client.create_session(**kwargs)
    monkeypatch.setattr(auto_resume, "_record", Mock(side_effect=OSError("journal unavailable")))

    async def workload(c, s):
        pytest.fail("A journal failure must stop before submitting training")

    with pytest.raises(OSError, match="journal unavailable"):
        await auto_resume.run(
            workload, config=auto_resume.AutoResumeConfig(auto_resume=True),
            log_path=str(tmp_path), client=client, session_id=session,
            client_factory=service.client, session_kwargs=kwargs,
        )
    assert service.closed == [session]


def test_tool_ner_evaluation_uses_the_active_attempt_directory(tmp_path, monkeypatch):
    from programmed_training.recipes.tool_ner_rl import eval_azure, ner_env, tab_data
    from programmed_training.recipes.tool_ner_rl.train_azure import CLIConfig, TabTrainingEvaluator

    monkeypatch.setattr(
        tab_data, "load_tab_dev_tasks", lambda **_: SimpleNamespace(tasks=())
    )
    monkeypatch.setattr(eval_azure, "_tab_manifest", lambda *_: {"mode": "offline"})
    monkeypatch.setattr(ner_env, "build_eval_dataset", lambda *_, **__: [])
    replacement = tmp_path / "replacement"
    attempt = auto_resume.Attempt(str(replacement), "session_new", frozenset(), None)
    token = auto_resume._attempt.set(attempt)
    try:
        evaluator = TabTrainingEvaluator(
            CLIConfig(project_endpoint="https://unit.invalid"), tmp_path / "initial"
        )
    finally:
        auto_resume._attempt.reset(token)
    assert evaluator.output_root == replacement / "tab_dev"
    assert (replacement / "tab_dev/manifest.json").is_file()
    assert not (tmp_path / "initial/tab_dev").exists()
