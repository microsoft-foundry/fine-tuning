"""Cookbook-owned recovery of complete training attempts, enabled by default."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import json
import logging
import os
import random
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import chz
from azure.ai.finetuningsessions import TrainingEngineError
from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import FromCheckpoint
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)
T = TypeVar("T")
CLEANUP_TIMEOUT_SECONDS = 60.0


@chz.chz
class AutoResumeConfig:
    auto_resume: bool = True
    auto_resume_max_retries: int = 3
    resumable_error_codes: tuple[str, ...] = ("engine_dead",)

    @chz.validate
    def _validate_recovery(self) -> None:
        if type(self.auto_resume_max_retries) is not int or self.auto_resume_max_retries < 0:
            raise ValueError("auto_resume_max_retries must be a nonnegative integer")
        if not isinstance(self.resumable_error_codes, (tuple, list, set, frozenset)) or any(
            not isinstance(code, str) or not code or code != code.strip()
            for code in self.resumable_error_codes
        ):
            raise ValueError("resumable_error_codes must contain nonempty exact error codes")


@dataclass
class Budget:
    replacements: int = 0
    retries_since_progress: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Attempt:
    log_path: str
    session_id: str
    codes: frozenset[str]
    resume_info: dict[str, Any] | None
    tasks: set[asyncio.Future[Any]] = field(default_factory=set)
    closers: list[Callable[[], Awaitable[Any]]] = field(default_factory=list)
    accepting_work: bool = True
    failed_gathers: list[
        tuple[list[asyncio.Future[Any]], Callable[[BaseException], bool] | None]
    ] = field(default_factory=list)
    run_id: str = ""
    index: int = 0
    status: str = "running"
    session_closer: Callable[[], Awaitable[Any]] | None = None


@dataclass
class Run:
    deadline: float | None
    budget: Budget
    log_path: str
    log_paths: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    end_batch: int | None = None
    max_retries: int = 3
    checkpoint_progress: tuple[int, int] | None = None


_attempt: contextvars.ContextVar[Attempt | None] = contextvars.ContextVar(
    "cookbook_recovery_attempt", default=None
)
_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar(
    "cookbook_recovery_run", default=None
)
_budget: contextvars.ContextVar[Budget | None] = contextvars.ContextVar(
    "cookbook_recovery_budget", default=None
)


def current_attempt() -> Attempt | None:
    return _attempt.get()


def current_log_path(default: str) -> str:
    attempt = current_attempt()
    return attempt.log_path if attempt is not None else default


def attempt_config(config: T) -> T:
    attempt = current_attempt()
    if attempt is None:
        return config
    changes: dict[str, Any] = {"log_path": attempt.log_path}
    name = getattr(config, "wandb_name", None)
    if attempt.index and name:
        changes["wandb_name"] = f"{name}-recovery-{attempt.index}"
    return chz.replace(config, **changes)


def attempt_log_paths(log_path: str) -> list[str]:
    """Read the explicit attempt lineage, without merging per-session artifacts."""
    from programmed_training.utils.file_utils import read_jsonl

    paths = [log_path]
    journal = os.path.join(log_path, "recovery.jsonl")
    if os.path.exists(journal):
        for event in read_jsonl(journal):
            if event["event"] == "attempt":
                path = event["log_path"]
                if not isinstance(path, str):
                    raise ValueError("Recovery attempt log_path must be a string")
                if all(os.path.abspath(path) != os.path.abspath(old) for old in paths):
                    paths.append(path)
    return paths


def latest_attempt_log_path(log_path: str) -> str:
    return attempt_log_paths(log_path)[-1]


def current_deadline(default: float | None = None) -> float | None:
    run = _run.get()
    return run.deadline if run is not None else default


def fixed_end_batch(end_batch: int) -> int:
    run = _run.get()
    if run is None:
        return end_batch
    if run.end_batch is None:
        run.end_batch = end_batch
    return run.end_batch


def _session_key(session_id: str) -> str:
    return session_id.removeprefix("session_").removeprefix("model_")


def is_resumable(error: BaseException) -> bool:
    attempt = current_attempt()
    if isinstance(error, BaseExceptionGroup):
        return all(is_resumable(child) for child in error.exceptions)
    return (
        attempt is not None
        and isinstance(error, TrainingEngineError)
        and error.error_code in attempt.codes
        and isinstance(error.session_id, str)
        and _session_key(error.session_id) == _session_key(attempt.session_id)
    )


def session_errors(function):
    """Attribute SDK submission errors at the adapter's explicit session boundary."""
    @functools.wraps(function)
    async def wrapped(self, *args, **kwargs):
        attempt = current_attempt()
        if attempt is not None and not attempt.accepting_work:
            raise RuntimeError("The interrupted training attempt is shutting down")
        try:
            return await function(self, *args, **kwargs)
        except TrainingEngineError as exc:
            if current_attempt() is not None and exc.session_id is None:
                exc.session_id = self._session_id
            raise
    return wrapped


def own_future(future: asyncio.Future[T]) -> asyncio.Future[T]:
    attempt = current_attempt()
    if attempt is not None:
        if not attempt.accepting_work:
            future.cancel()
            raise RuntimeError("The interrupted training attempt is shutting down")
        attempt.tasks.add(future)
        def discard_success(done: asyncio.Future[Any]) -> None:
            if done.cancelled() or done.exception() is None:
                attempt.tasks.discard(done)
        future.add_done_callback(discard_success)
    return future


def create_task(coro: Coroutine[Any, Any, T], *, name: str | None = None) -> asyncio.Task[T]:
    task = asyncio.create_task(coro, name=name)
    own_future(task)
    return task


async def gather(
    *operations: Awaitable[Any],
    return_exceptions: bool = False,
    retryable_error: Callable[[BaseException], bool] | None = None,
) -> list[Any]:
    if current_attempt() is None:
        return await asyncio.gather(*operations, return_exceptions=return_exceptions)
    tasks = [own_future(asyncio.ensure_future(operation)) for operation in operations]
    try:
        if return_exceptions:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if any(task.cancelled() for task in done):
                    raise asyncio.CancelledError()
                errors = [
                    error for task in tasks
                    if task.done() and not task.cancelled() and (error := task.exception()) is not None
                ]
                if retryable_error is not None and any(is_resumable(error) for error in errors):
                    fatal = [
                        error for error in errors
                        if not is_resumable(error) and not retryable_error(error)
                    ]
                    if fatal:
                        raise BaseExceptionGroup("Concurrent rollout failures", errors)
                for task in done:
                    if not task.cancelled() and (error := task.exception()) is not None:
                        if is_resumable(error):
                            raise error
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    except BaseException as error:
        if is_resumable(error):
            attempt = current_attempt()
            assert attempt is not None
            attempt.failed_gathers.append((tasks, retryable_error if return_exceptions else None))
        else:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _concurrent_failure(attempt: Attempt, failure: BaseException) -> BaseException:
    if not is_resumable(failure):
        return failure
    errors = [failure]
    for tasks, retryable_error in attempt.failed_gathers:
        for task in tasks:
            if task.done() and not task.cancelled():
                error = task.exception()
                if (
                    error is not None and all(error is not old for old in errors)
                    and not (retryable_error is not None and retryable_error(error))
                ):
                    errors.append(error)
    if any(not is_resumable(error) for error in errors):
        return BaseExceptionGroup("Concurrent training failures", errors)
    return failure


def defer_close(closer: Callable[[], Awaitable[Any]]) -> None:
    attempt = current_attempt()
    if attempt is None:
        raise RuntimeError("Deferred cleanup requires an active recovery attempt")
    attempt.closers.append(closer)


async def create_owned_session(client: FineTuningSessionClient, **kwargs: Any) -> str:
    """Own secondary sessions, including a known allocation that fails initialization."""
    try:
        session_id = await client.create_session(**kwargs)
    except TrainingEngineError as exc:
        if current_attempt() is not None and exc.session_id is not None:
            failed_session = exc.session_id
            defer_close(lambda: client.close_session(failed_session))
        raise
    if current_attempt() is not None:
        defer_close(lambda: client.close_session(session_id))
    return session_id


def shared_run(function):
    """Share recovery counters across a recipe's sequential stages."""
    @functools.wraps(function)
    async def wrapped(*args, **kwargs):
        token = _budget.set(_budget.get() or Budget())
        try:
            return await function(*args, **kwargs)
        finally:
            _budget.reset(token)
    return wrapped


def parse_checkpoint(path: str) -> FromCheckpoint:
    stripped = path.split("://", 1)[-1]
    parts = stripped.split("/")
    if len(parts) == 3 and parts[1] == "weights":
        session_id, _, checkpoint_id = parts
    elif len(parts) == 2:
        session_id, checkpoint_id = parts
    else:
        raise ValueError(f"Invalid training checkpoint path: {path!r}")
    if not session_id or not checkpoint_id:
        raise ValueError(f"Invalid training checkpoint path: {path!r}")
    return FromCheckpoint(
        source_session_id=f"session_{_session_key(session_id)}",
        checkpoint_id=checkpoint_id,
    )


async def _select_checkpoint(
    client: FineTuningSessionClient, log_path: str, *, log_paths: list[str] | None = None
) -> dict[str, Any] | None:
    from programmed_training import checkpoint_utils

    rows = [
        row for path in (log_paths or [log_path])
        for row in checkpoint_utils.load_checkpoints_file(path)
        if "state_path" in row
    ]
    by_session: dict[str, set[str]] = {}
    for row in reversed(rows):
        checkpoint = parse_checkpoint(row["state_path"])
        source = checkpoint.source_session_id
        if source not in by_session:
            try:
                listing = await client.checkpoints.list(
                    source, api_version="v1", foundry_features="FineTuningSessions=V1Preview"
                )
            except ResourceNotFoundError:
                logger.warning("Checkpoint source session %s is unavailable", source)
                by_session[source] = set()
            else:
                by_session[source] = {
                    item.checkpoint_id for item in listing.checkpoints
                    if item.checkpoint_type == "training"
                }
        if checkpoint.checkpoint_id in by_session[source]:
            return row
        logger.warning("Skipping unavailable training checkpoint %s", row["state_path"])
    if rows:
        raise RuntimeError("No committed training checkpoint is still available")
    return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _error_fields(error: BaseException) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_code": getattr(error, "error_code", None),
        "error": str(error),
    }
    if isinstance(error, BaseExceptionGroup):
        children = [_error_fields(child) for child in error.exceptions]
        codes = {child["error_code"] for child in children}
        fields["error_code"] = next(iter(codes)) if len(codes) == 1 else None
        fields["errors"] = children
    return fields


def _record(state: Run, *, event: str, **fields: Any) -> None:
    record = {
        "time": time.time(), "run_id": state.budget.run_id, "event": event,
        "replacement": state.budget.replacements,
        "retries_since_progress": state.budget.retries_since_progress,
        "max_retries": state.max_retries,
        "log_path": state.log_paths[-1] if state.log_paths else state.log_path,
        "elapsed_seconds": round(time.monotonic() - state.started_at, 3),
        **fields,
    }
    logger.info("Cookbook auto-resume: %s", record)
    with open(os.path.join(state.log_path, "recovery.jsonl"), "a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")
    _write_json(Path(state.log_path) / "recovery_status.json", record)


def _attempt_metadata(attempt: Attempt, **fields: Any) -> None:
    path = Path(attempt.log_path) / "run_meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metadata.setdefault("azure", {})["session_id"] = attempt.session_id
    metadata.setdefault("auto_resume", {}).update(
        run_id=attempt.run_id, attempt=attempt.index, status=attempt.status, **fields
    )
    _write_json(path, metadata)


def record_reference_session(log_path: str, session_id: str) -> None:
    path = Path(current_log_path(log_path)) / "run_meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["azure"]["reference_session_id"] = session_id
    _write_json(path, metadata)


def _checkpoint_progress(checkpoint: dict[str, Any]) -> tuple[int, int]:
    epoch, batch = checkpoint.get("epoch", 0), checkpoint.get("batch")
    if any(type(value) is not int or value < 0 for value in (epoch, batch)):
        raise ValueError("Recovery checkpoint epoch and batch must be nonnegative integers")
    return epoch, batch


def checkpoint_committed(log_path: str, checkpoint: dict[str, Any]) -> None:
    state, attempt = _run.get(), current_attempt()
    if (
        state is None or attempt is None or not attempt.accepting_work
        or os.path.abspath(log_path) != attempt.log_path or "state_path" not in checkpoint
    ):
        return
    progress = _checkpoint_progress(checkpoint)
    if state.checkpoint_progress is not None and progress <= state.checkpoint_progress:
        return
    state.checkpoint_progress = progress
    previous_retries = state.budget.retries_since_progress
    if previous_retries:
        state.budget.retries_since_progress = 0
        _attempt_metadata(attempt, retries_since_progress=0)
        _record(
            state, event="retry_budget_reset", session_id=attempt.session_id,
            previous_retries=previous_retries, checkpoint=checkpoint["state_path"],
            checkpoint_progress={"epoch": progress[0], "batch": progress[1]},
        )


async def _bounded(operation: Awaitable[T], seconds: float) -> T:
    task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait([task], timeout=max(0.0, seconds))
        if not done:
            raise TimeoutError("Cookbook recovery operation exceeded its time limit")
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            def observe_late_failure(done: asyncio.Future[Any]) -> None:
                if not done.cancelled() and (error := done.exception()) is not None:
                    logger.error(
                        "Recovery operation failed after cancellation",
                        exc_info=(type(error), error, error.__traceback__),
                    )
            task.add_done_callback(observe_late_failure)


async def _cleanup(
    attempt: Attempt,
    client: FineTuningSessionClient,
    *,
    timeout_seconds: float | None = None,
) -> None:
    attempt.accepting_work = False
    for task in attempt.tasks:
        if not task.done():
            task.cancel()
    timeout = CLEANUP_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    end = time.monotonic() + timeout
    resource_end = end - min(1.0, timeout / 10)
    try:
        if attempt.tasks:
            done, pending = await asyncio.wait(
                attempt.tasks, timeout=max(0.0, resource_end - time.monotonic())
            )
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                raise TimeoutError("Failed-attempt work did not stop; refusing an overlapping restart")
        errors: list[BaseException] = []
        for closer in attempt.closers:
            try:
                await _bounded(closer(), resource_end - time.monotonic())
            except BaseException as exc:
                logger.exception("Recovery resource cleanup failed")
                errors.append(exc)
        try:
            close = (
                attempt.session_closer() if attempt.session_closer is not None
                else client.close_session(attempt.session_id)
            )
            await _bounded(close, resource_end - time.monotonic())
        except TrainingEngineError as exc:
            if not is_resumable(exc):
                errors.append(exc)
            else:
                logger.warning("Interrupted session could not be closed: %s", exc)
        except BaseException as exc:
            errors.append(exc)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("Recovery resource cleanup failed", errors)
    finally:
        await _bounded(client.close(), end - time.monotonic())


async def run(
    workload: Callable[[FineTuningSessionClient, str], Awaitable[T]],
    *,
    config: AutoResumeConfig,
    log_path: str,
    client: FineTuningSessionClient,
    session_id: str,
    client_factory: Callable[[], FineTuningSessionClient],
    session_kwargs: dict[str, Any],
    max_wall_clock_seconds: float | None = None,
    restart_from_original_input: bool = True,
    initial_closers: tuple[Callable[[], Awaitable[Any]], ...] = (),
    initial_session_closer: Callable[[], Awaitable[Any]] | None = None,
    cleanup_timeout_seconds: float | None = None,
) -> T:
    if not config.auto_resume or config.auto_resume_max_retries == 0:
        training_type = session_kwargs.get("training_type")
        metadata = session_kwargs.get("user_metadata")
        basic_tier = training_type == "DeveloperTier" or (
            training_type is None and isinstance(metadata, dict)
            and metadata.get("DeveloperTier") is True
        )
        if basic_tier:
            setting = "auto_resume=False" if not config.auto_resume else "auto_resume_max_retries=0"
            logger.warning(
                "%s disables automatic recovery for basic-tier "
                "(DeveloperTier) training. Engine eviction or failure will stop "
                "this run; set auto_resume=True and auto_resume_max_retries>0 "
                "to enable best-effort recovery.", setting,
            )
    if not config.auto_resume:
        return await workload(client, session_id)

    from programmed_training import checkpoint_utils

    codes = frozenset(config.resumable_error_codes)
    budget = _budget.get() or Budget()
    log_path = os.path.abspath(os.path.expanduser(log_path))
    initial_attempt = Attempt(
        log_path, session_id, codes, None, closers=list(initial_closers),
        run_id=budget.run_id, session_closer=initial_session_closer,
    )
    try:
        os.makedirs(log_path, exist_ok=True)
        selected = checkpoint_utils.get_last_checkpoint(log_path)
        state = Run(
            time.time() + max_wall_clock_seconds
            if max_wall_clock_seconds is not None and max_wall_clock_seconds > 0 else None,
            budget=budget, log_path=log_path,
            max_retries=config.auto_resume_max_retries,
            checkpoint_progress=_checkpoint_progress(selected) if selected is not None else None,
        )
        metadata_path = Path(log_path) / "run_meta.json"
        original_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
    except BaseException:
        logger.exception("Cookbook recovery setup failed")
        try:
            await _cleanup(initial_attempt, client, timeout_seconds=cleanup_timeout_seconds)
        except BaseException:
            logger.exception("Recovery setup cleanup failed; preserving setup failure")
        raise
    original_checkpoint = session_kwargs.get("from_checkpoint")
    previous_session: str | None = None
    previous_log_path: str | None = None
    terminal_event = "failed"
    run_token = _run.set(state)
    try:
        while True:
            index = len(state.log_paths)
            attempt_path = (
                os.path.join(log_path, "recovery_attempts", f"{budget.run_id}-attempt-{index}")
                if index else log_path
            )
            attempt = initial_attempt if not index else Attempt(
                attempt_path, session_id, codes, selected, run_id=budget.run_id, index=index
            )
            attempt.resume_info = selected
            token = _attempt.set(attempt)
            failure: BaseException | None = None
            attempt_started = time.monotonic()
            try:
                if index:
                    os.makedirs(attempt_path, exist_ok=False)
                    metadata = copy.deepcopy(original_metadata)
                    metadata["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
                    if metadata.get("run_name"):
                        metadata["run_name"] += f"-recovery-{index}"
                    _write_json(Path(attempt_path) / "run_meta.json", metadata)
                state.log_paths.append(attempt_path)
                source = parse_checkpoint(selected["state_path"]) if selected else original_checkpoint
                metadata = json.loads((Path(attempt_path) / "run_meta.json").read_text(
                    encoding="utf-8"
                )) if (Path(attempt_path) / "run_meta.json").exists() else {}
                metadata.setdefault("azure", {})["from_checkpoint"] = (
                    {"source_session_id": source.source_session_id,
                     "checkpoint_id": source.checkpoint_id} if source else None
                )
                _write_json(Path(attempt_path) / "run_meta.json", metadata)
                _attempt_metadata(
                    attempt, initial_log_path=log_path, previous_log_path=previous_log_path,
                    previous_session_id=previous_session,
                    recovery_journal=os.path.join(log_path, "recovery.jsonl"),
                    replacement=budget.replacements,
                    retries_since_progress=budget.retries_since_progress,
                    max_retries=state.max_retries,
                )
                _record(state, event="attempt", session_id=session_id,
                        attempt=index, resumable_error_codes=sorted(codes),
                        checkpoint=selected["state_path"] if selected else None)
                remaining = None if state.deadline is None else state.deadline - time.time()
                if remaining is not None and remaining <= 0:
                    terminal_event = "deadline_exceeded"
                    raise TimeoutError("Original cookbook run deadline expired")
                task = create_task(workload(client, session_id), name="cookbook_training_attempt")
                await asyncio.wait([task])
                result = task.result()
                attempt.status = "completed"
            except BaseException as exc:
                failure = exc
                attempt.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                logger.error("Cookbook training attempt failed", exc_info=True)
                _record(state, event="attempt_failed", session_id=session_id, **_error_fields(exc))
            finally:
                try:
                    try:
                        _record(state, event="cleanup_started", session_id=session_id)
                    finally:
                        await _cleanup(attempt, client, timeout_seconds=cleanup_timeout_seconds)
                    if failure is not None:
                        failure = _concurrent_failure(attempt, failure)
                    _record(state, event="cleanup_completed", session_id=session_id)
                except BaseException as cleanup_error:
                    attempt.status = "failed"
                    _record(state, event="cleanup_failed", session_id=session_id,
                            **_error_fields(cleanup_error))
                    if failure is None:
                        raise
                    logger.exception("Recovery cleanup failed; preserving the training failure")
                    raise failure
                finally:
                    try:
                        eligible = failure is not None and is_resumable(failure)
                        _attempt_metadata(
                            attempt,
                            elapsed_seconds=round(time.monotonic() - attempt_started, 3),
                            retries_since_progress=budget.retries_since_progress,
                            **(_error_fields(failure) if failure is not None else {}),
                        )
                    finally:
                        _attempt.reset(token)
            if failure is None:
                _record(state, event="completed", session_id=session_id,
                        attempt_log_paths=state.log_paths)
                return result
            if not eligible:
                raise failure
            if budget.retries_since_progress >= state.max_retries:
                terminal_event = "exhausted"
                raise failure
            previous_session, previous_log_path = session_id, attempt_path
            while True:
                budget.replacements += 1
                budget.retries_since_progress += 1
                delay = (10 * 2 ** min(budget.retries_since_progress - 1, 2)) * random.uniform(0.8, 1.2)
                if state.deadline is not None and time.time() + delay >= state.deadline:
                    terminal_event = "deadline_exceeded"
                    raise failure
                _record(state, event="backoff", session_id=session_id, seconds=delay,
                        **_error_fields(failure))
                await asyncio.sleep(delay)
                client = client_factory()
                creating = False
                try:
                    remaining = 600.0 if state.deadline is None else state.deadline - time.time()
                    selected = await _bounded(
                        _select_checkpoint(client, log_path, log_paths=state.log_paths), remaining
                    )
                    if selected is None and not restart_from_original_input:
                        raise RuntimeError(
                            "Cannot restart an attached session without a committed training checkpoint"
                        )
                    kwargs = dict(session_kwargs)
                    kwargs["from_checkpoint"] = (
                        parse_checkpoint(selected["state_path"]) if selected else original_checkpoint
                    )
                    _record(state, event="resume" if selected else "restart_original_input",
                            prior_session_id=session_id, replacement=budget.replacements,
                            **_error_fields(failure),
                            checkpoint=selected["state_path"] if selected else None)
                    remaining = 600.0 if state.deadline is None else state.deadline - time.time()
                    creating = True
                    session_id = await _bounded(client.create_session(**kwargs), remaining)
                    break
                except BaseException as exc:
                    _record(state, event="replacement_failed",
                            session_id=getattr(exc, "session_id", None), **_error_fields(exc))
                    retry = (
                        creating and isinstance(exc, TrainingEngineError)
                        and exc.error_code in codes and exc.session_id is not None
                        and budget.retries_since_progress < state.max_retries
                    )
                    try:
                        if creating and isinstance(exc, TrainingEngineError) and exc.session_id is not None:
                            await _bounded(
                                client.close_session(exc.session_id),
                                CLEANUP_TIMEOUT_SECONDS if cleanup_timeout_seconds is None else cleanup_timeout_seconds,
                            )
                    except Exception:
                        logger.exception("Replacement initialization cleanup failed")
                        retry = False
                    finally:
                        try:
                            await _bounded(
                                client.close(),
                                CLEANUP_TIMEOUT_SECONDS if cleanup_timeout_seconds is None else cleanup_timeout_seconds,
                            )
                        except Exception:
                            logger.exception("Replacement client cleanup failed")
                            retry = False
                    if not retry:
                        if (
                            creating and isinstance(exc, TrainingEngineError)
                            and exc.error_code in codes
                            and budget.retries_since_progress >= state.max_retries
                        ):
                            terminal_event = "exhausted"
                        raise
                    failure = exc
    except BaseException as exc:
        _record(
            state, event="cancelled" if isinstance(exc, asyncio.CancelledError) else terminal_event,
            session_id=session_id, attempt_log_paths=state.log_paths, **_error_fields(exc),
        )
        raise
    finally:
        _run.reset(run_token)


async def run_rl(
    cfg,
    client: FineTuningSessionClient,
    session_id: str,
    *,
    recovery_config: AutoResumeConfig,
    client_kwargs: dict[str, Any],
    session_kwargs: dict[str, Any],
    restart_from_original_input: bool = True,
    client_factory: Callable[[], FineTuningSessionClient] | None = None,
    **training_kwargs: Any,
) -> None:
    from programmed_training.rl.train_azure import main

    async def workload(attempt_client, attempt_session):
        await main(cfg, attempt_client, attempt_session, **training_kwargs)

    if recovery_config.auto_resume and cfg.save_every <= 0 and cfg.eval_every <= 0:
        logger.warning("No periodic training recovery points; unsaved work will be lost on restart")
    await run(
        workload, config=recovery_config, log_path=cfg.log_path,
        client=client, session_id=session_id,
        client_factory=client_factory or (lambda: FineTuningSessionClient(**client_kwargs)),
        session_kwargs=session_kwargs,
        max_wall_clock_seconds=cfg.max_wall_clock_seconds,
        restart_from_original_input=restart_from_original_input,
    )


async def run_supervised(
    cfg,
    training_client,
    *,
    recovery_config: AutoResumeConfig,
    client: FineTuningSessionClient,
    session_id: str,
    client_kwargs: dict[str, Any],
    session_kwargs: dict[str, Any],
    prepared_datasets=None,
    restart_from_original_input: bool = True,
    initial_session_closer: Callable[[], Awaitable[Any]] | None = None,
) -> None:
    from programmed_training.rl.train_azure import AzureSDKTrainingClient
    from programmed_training.supervised.train import main

    async def workload(attempt_client, attempt_session):
        attempt_training = (
            training_client if attempt_client is client else AzureSDKTrainingClient(
                attempt_client, attempt_session, training_client.get_tokenizer()
            )
        )
        if attempt_client is not client and current_attempt() is not None:
            defer_close(attempt_training.close_poller)
        if prepared_datasets is None:
            await main(cfg, attempt_training)
        else:
            await main(cfg, attempt_training, prepared_datasets=prepared_datasets)

    if recovery_config.auto_resume and cfg.save_every <= 0:
        logger.warning("No periodic SFT recovery points; unsaved work will be lost on restart")
    await run(
        workload, config=recovery_config, log_path=cfg.log_path,
        client=client, session_id=session_id,
        client_factory=lambda: FineTuningSessionClient(**client_kwargs),
        session_kwargs=session_kwargs,
        max_wall_clock_seconds=cfg.max_wall_clock_seconds,
        restart_from_original_input=restart_from_original_input,
        initial_closers=(training_client.close_poller,),
        initial_session_closer=initial_session_closer,
    )
