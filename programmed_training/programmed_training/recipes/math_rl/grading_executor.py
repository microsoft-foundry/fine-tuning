"""Run-scoped admission for threads supervising isolated grading processes."""

import asyncio
import contextvars
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, ParamSpec, Self, TypeVar

DEFAULT_GRADING_CONCURRENCY = min(8, max(1, (os.cpu_count() or 1) // 2))

P = ParamSpec("P")
T = TypeVar("T")
_current_executor: contextvars.ContextVar["GradingExecutor | None"] = contextvars.ContextVar(
    "math_grading_executor", default=None
)


class GradingExecutor:
    """Share one bounded executor across a run's training and evaluation tasks."""

    def __init__(self, max_workers: int = DEFAULT_GRADING_CONCURRENCY):
        if max_workers < 1:
            raise ValueError("grading concurrency must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="math-grading"
        )
        self._admission = asyncio.Semaphore(max_workers)
        self._jobs: set[Future[Any]] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        self._token = _current_executor.set(self)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        _current_executor.reset(self._token)
        await self.aclose()

    async def run(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        if self._closed:
            raise RuntimeError("grading executor is closed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("grading executor belongs to a different event loop")
        await self._admission.acquire()
        try:
            if self._closed:
                raise RuntimeError("grading executor is closed")
            context = contextvars.copy_context()
            job = self._executor.submit(context.run, partial(func, *args, **kwargs))
            self._jobs.add(job)
            self._idle.clear()
            job.add_done_callback(
                lambda finished: self._loop.call_soon_threadsafe(self._job_done, finished)
            )
        except BaseException:
            self._admission.release()
            raise
        return await asyncio.wrap_future(job, loop=self._loop)

    def _job_done(self, job: Future[Any]) -> None:
        # A cancelled asyncio waiter does not mean its thread/child has exited.
        self._jobs.remove(job)
        self._admission.release()
        if not self._jobs:
            self._idle.set()

    async def aclose(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        cancelled = False
        while self._jobs:
            try:
                await self._idle.wait()
            except asyncio.CancelledError:
                # Finish subprocess cleanup before the owning loop can close.
                cancelled = True
        self._executor.shutdown(wait=True)
        if cancelled:
            raise asyncio.CancelledError


async def run_grading(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    executor = _current_executor.get()
    if executor is not None:
        return await executor.run(func, *args, **kwargs)
    # Preserve unscoped callers' existing resource/lifecycle behavior.
    return await asyncio.to_thread(func, *args, **kwargs)
