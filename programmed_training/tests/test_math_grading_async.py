"""Math-only liveness and actual recipe module-launch regressions (offline)."""

import asyncio
import contextvars
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import aiohttp
import pytest

from programmed_training.recipes.math_rl import math_env, math_grading
from programmed_training.recipes.math_rl.grading_executor import (
    GradingExecutor,
    run_grading,
)
from tests.math_grading_worker_helpers import grade_after_marker


class _Renderer:
    def __init__(self, text):
        self.text = text

    def parse_response(self, action):
        return {"role": "assistant", "content": self.text}, True

    def get_stop_sequences(self):
        return []

    def build_generation_prompt(self, messages, prefill=None):
        from azure.ai.finetuningsessions.models import ModelInput

        return ModelInput(chunks=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("forced, prefill", [(False, False), (True, False), (True, True)])
async def test_event_loop_heartbeat_runs_during_real_terminal_grading(
    tmp_path, monkeypatch, forced, prefill
):
    marker = tmp_path / "grading.txt"
    monkeypatch.setattr(
        math_env, "grade_answer", partial(grade_after_marker, marker_path=str(marker))
    )
    renderer = _Renderer("not boxed" if forced else r"\boxed{0.5}")
    env = math_env.MathEnv(
        "one half", "1/2", renderer, forced_commit=forced,
        forced_commit_prefill_boxed=prefill, forced_commit_penalty=0.1,
    )
    if forced:
        opened = await env.step([])
        assert not opened.episode_done and not marker.exists()
        renderer.text = "0.5}" if prefill else r"\boxed{0.5}"

    grading = asyncio.create_task(env.step([]))
    heartbeats_during_grading = 0
    async with asyncio.timeout(45):
        while not grading.done():
            if marker.exists():
                heartbeats_during_grading += 1
            await asyncio.sleep(0.01)
    result = await grading

    assert marker.exists(), "The actual isolated grader must have begun work"
    assert heartbeats_during_grading > 0, "MathEnv.step blocked the event loop during grading"
    assert result.episode_done
    assert result.metrics["correct"] == 1.0
    assert result.metrics["forced_commit_used"] == float(forced)
    assert result.reward == (0.9 if forced else 1.0)


@pytest.mark.asyncio
async def test_step_does_not_swallow_synchronous_check_answer_errors(monkeypatch):
    env = math_env.MathEnv("one half", "1/2", _Renderer(r"\boxed{0.5}"))

    def fail(text):
        raise ValueError("terminal grading failed")

    monkeypatch.setattr(env, "check_answer", fail)
    with pytest.raises(ValueError, match="terminal grading failed"):
        await env.step([])


def test_direct_check_answer_remains_synchronous_and_supported():
    env = math_env.MathEnv("one half", "1/2", _Renderer(r"\boxed{0.5}"))

    assert env.check_answer(r"\boxed{0.5}") is True
    assert env.check_answer(r"\boxed{2}") is False
    assert env.check_answer("not boxed") is False


def test_real_math_train_azure_module_launch_does_not_reenter_in_workers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "entry-executions.txt"
    # On Windows do not let the test's own deadline target only a venv
    # redirector. This still launches the real recipe with Python's -m.
    executable, worker_environment = math_grading._worker_interpreter()
    environment = worker_environment if worker_environment is not None else os.environ.copy()
    environment.update(
        MATH_GRADING_LAUNCH_MARKER=str(marker),
        MATH_GRADING_SOURCE_ROOT=str(root),
        PYTHONPATH=os.pathsep.join(
            [str(root / "tests/math_grading_launch"), str(root),
             str(root.parent / "azure-ai-finetuningsessions")]
        ),
        PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled",
    )
    completed = subprocess.run(
        [executable, "-B", "-m", "programmed_training.recipes.math_rl.train_azure"],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        encoding="utf-8", timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = marker.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "A grading worker executed the real training entry module"
    report = next(
        line.partition("=")[2] for line in completed.stdout.splitlines()
        if line.startswith("MATH_MODULE_LAUNCH_OK=")
    )
    assert json.loads(report)["training_modules"] == []


async def _wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_shared_grading_admission_leaves_http_and_dns_available(tmp_path, monkeypatch):
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    original_executor = loop._default_executor
    http_executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(http_executor)
    release = threading.Event()
    started = []
    context = contextvars.ContextVar("grading_test_context", default="missing")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("NETRC", str(tmp_path / "absent-netrc"))

    async def handle_get(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    def grade(text):
        started.append((context.get(), threading.current_thread().name))
        assert release.wait(10), "Test grading was not released"
        return True

    tasks = []
    try:
        async with GradingExecutor(max_workers=2) as executor:
            submitted = []
            submit = executor._executor.submit

            def record_submission(*args, **kwargs):
                submitted.append(True)
                return submit(*args, **kwargs)

            monkeypatch.setattr(executor._executor, "submit", record_submission)
            for label in ("train", "evaluation", "queued"):
                env = math_env.MathEnv("one half", "1/2", _Renderer(r"\boxed{0.5}"))
                monkeypatch.setattr(env, "check_answer", grade)
                token = context.set(label)
                tasks.append(asyncio.create_task(env.step([])))
                context.reset(token)
            try:
                await _wait_until(lambda: len(started) == 2)
                async with asyncio.timeout(5):
                    assert await asyncio.to_thread(lambda: "HTTP setup") == "HTTP setup"
                    assert await loop.getaddrinfo("localhost", 80)
                    async with await asyncio.start_server(handle_get, "127.0.0.1", 0) as server:
                        port = server.sockets[0].getsockname()[1]
                        async with (
                            aiohttp.ClientSession(trust_env=True) as session,
                            session.get(f"http://localhost:{port}/") as response,
                        ):
                            assert await response.text() == "ok"
                assert len(submitted) == 2, "Excess grades must wait before submission"
                assert {label for label, _ in started} == {"train", "evaluation"}
                assert all(name.startswith("math-grading") for _, name in started)
            finally:
                release.set()
                results = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(result.metrics["correct"] == 1.0 for result in results)
            assert len(submitted) == 3
    finally:
        release.set()
        loop.set_default_executor(original_executor)
        http_executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_running", [False, True])
async def test_cancellation_does_not_release_admission_before_worker_completion(cancel_running):
    started = threading.Event()
    release = threading.Event()
    later_started = []

    def first_grade():
        started.set()
        assert release.wait(10), "Test grading was not released"
        return True

    def later_grade(label, *, correct):
        later_started.append(label)
        return correct

    async with GradingExecutor(max_workers=1) as executor:
        first = asyncio.create_task(run_grading(first_grade))
        await _wait_until(started.is_set)
        waiting = asyncio.create_task(run_grading(later_grade, "waiting", correct=True))
        await asyncio.sleep(0)
        cancelled = first if cancel_running else waiting
        cancelled.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            replacement = asyncio.create_task(
                run_grading(later_grade, "replacement", correct=True)
            )
            await asyncio.sleep(0)
            assert not later_started
            assert len(executor._jobs) == 1
        finally:
            release.set()
        assert await replacement is True
        if cancel_running:
            assert await waiting is True
            assert later_started == ["waiting", "replacement"]
        else:
            assert await first is True
            assert later_started == ["replacement"]


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_and_joins_grading_threads():
    started = threading.Event()
    release = threading.Event()
    executor = GradingExecutor(max_workers=1)

    def grade():
        started.set()
        assert release.wait(10), "Test grading was not released"
        return True

    async def owner():
        async with executor:
            await run_grading(grade)

    task = asyncio.create_task(owner())
    try:
        await _wait_until(started.is_set)
        task.cancel()
        await _wait_until(lambda: executor._closed)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "Cancellation must not abandon the grading thread"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not executor._jobs
    assert all(not thread.is_alive() for thread in executor._executor._threads)


@pytest.mark.asyncio
async def test_submission_failure_does_not_leak_a_grading_slot(monkeypatch):
    async with GradingExecutor(max_workers=1) as executor:
        submit = executor._executor.submit

        def fail_submission(*args, **kwargs):
            raise RuntimeError("submission failed")

        monkeypatch.setattr(executor._executor, "submit", fail_submission)
        with pytest.raises(RuntimeError, match="submission failed"):
            await run_grading(lambda: True)
        monkeypatch.setattr(executor._executor, "submit", submit)
        async with asyncio.timeout(5):
            assert await run_grading(lambda: True) is True


@pytest.mark.asyncio
async def test_run_scope_restores_unscoped_executor_behavior():
    def thread_name():
        return threading.current_thread().name

    async with GradingExecutor(max_workers=1):
        assert (await run_grading(thread_name)).startswith("math-grading")
    assert not (await run_grading(thread_name)).startswith("math-grading")


@pytest.mark.asyncio
async def test_cancelled_run_waits_for_real_subprocess_timeout_and_cleanup(tmp_path, monkeypatch):
    marker = tmp_path / "grading.txt"
    reaped = []
    stop_worker = math_grading._stop_worker

    def record_cleanup(process):
        stop_worker(process)
        reaped.append((process.pid, process.poll(), Path(process.args[-1])))

    monkeypatch.setattr(math_grading, "_stop_worker", record_cleanup)
    monkeypatch.setattr(
        math_env, "grade_answer", partial(grade_after_marker, marker_path=str(marker))
    )
    executor = GradingExecutor(max_workers=1)

    async def owner():
        async with executor:
            env = math_env.MathEnv("one half", "1/2", _Renderer(r"\boxed{0.5}"), timeout=0.2)
            await env.step([])

    task = asyncio.create_task(owner())
    await _wait_until(marker.exists)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(reaped) == 1
    assert reaped[0][0] == int(marker.read_text(encoding="utf-8"))
    assert reaped[0][1] is not None
    assert not reaped[0][2].exists()
    assert all(not thread.is_alive() for thread in executor._executor._threads)


@pytest.mark.asyncio
@pytest.mark.parametrize("training_fails", [False, True])
async def test_real_cli_owns_one_grading_scope_and_closes_it_on_error(
    tmp_path, monkeypatch, training_fails
):
    from azure.ai.finetuningsessions import aio

    from programmed_training.recipes.math_rl import train_azure

    executors = []
    run = GradingExecutor.run

    async def record_run(self, *args, **kwargs):
        executors.append(self)
        return await run(self, *args, **kwargs)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def create_session(self, **kwargs):
            return "offline-session"

        async def close(self):
            assert executors[0]._closed
            assert all(not thread.is_alive() for thread in executors[0]._executor._threads)

    async def offline_training(*args, **kwargs):
        for _ in ("baseline", "training", "final evaluation"):
            env = math_env.MathEnv("one half", "1/2", _Renderer(r"\boxed{0.5}"))
            monkeypatch.setattr(env, "check_answer", lambda text: True)
            assert (await asyncio.create_task(env.step([]))).reward == 1.0
        if training_fails:
            raise ValueError("training failed")

    monkeypatch.setenv("AZURE_AI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(aio, "FineTuningSessionClient", Client)
    monkeypatch.setattr(train_azure.auto_resume, "run_rl", offline_training)
    monkeypatch.setattr(GradingExecutor, "run", record_run)
    config = train_azure.CLIConfig(
        project_endpoint="https://offline.invalid", env="gsm8k",
        log_path=str(tmp_path / "run"), grading_concurrency=2,
    )
    if training_fails:
        with pytest.raises(ValueError, match="training failed"):
            await train_azure.cli_main(config)
    else:
        await train_azure.cli_main(config)
    assert len(executors) == 3
    assert all(executor is executors[0] for executor in executors)
    assert executors[0]._executor._max_workers == 2
    metadata = json.loads((tmp_path / "run" / "run_meta.json").read_text(encoding="utf-8"))
    assert metadata["training"]["grading_concurrency"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, -1])
async def test_cli_rejects_invalid_grading_admission_before_creating_resources(tmp_path, concurrency):
    from programmed_training.recipes.math_rl import train_azure

    config = train_azure.CLIConfig(
        project_endpoint="https://offline.invalid", log_path=str(tmp_path / "run"),
        grading_concurrency=concurrency,
    )
    with pytest.raises(ValueError, match="grading_concurrency must be positive"):
        await train_azure.cli_main(config)
    assert not (tmp_path / "run").exists()