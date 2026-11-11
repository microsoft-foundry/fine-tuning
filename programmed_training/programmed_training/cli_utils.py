import errno
import logging
import os
import shutil
import sys
from typing import Awaitable, Callable, Literal, TypeVar

logger = logging.getLogger(__name__)

LogdirBehavior = Literal["delete", "resume", "ask", "raise"]


_T = TypeVar("_T")


# Loggers that emit one or more INFO-level lines per HTTP call against the
# Foundry endpoint. Left at INFO they bury the recipe's own progress logs in
# request/response header dumps. Recipes that call ``configure_logging`` (or
# the inner train loops in ``rl/train.py`` / ``rl/train_azure.py``) clamp
# them down to WARNING.
_NOISY_LOGGERS = (
    "httpx",
    "pylatexenc",
    "azure.core.pipeline",
    "azure.identity",
)


def configure_logging(level: int = logging.INFO, *, verbose_http: bool = False) -> None:
    """Configure root logging for a recipe entrypoint.

    Sets ``logging.basicConfig`` with the cookbook's standard format and, by
    default, clamps the chatty Azure SDK / HTTP loggers
    (``azure.core.pipeline``, ``azure.identity``, ``httpx``, ``pylatexenc``) to
    WARNING so they don't drown out the recipe's own progress output.

    Pass ``verbose_http=True`` (typically wired to the recipe's ``verbose_http``
    CLI flag) to skip the WARNING clamp; those loggers then inherit ``level``
    and emit their full INFO-level request/response traffic.

    Idempotent: calling it more than once is safe (``basicConfig`` is a no-op
    when handlers are already attached, and the per-logger ``setLevel`` calls
    just re-assert the desired level).
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    if not verbose_http:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


async def run_recipe(coro_factory: Callable[[], Awaitable[_T]]) -> _T:
    """Run a recipe entrypoint coroutine with loud failure semantics.

    The default Python behaviour for a recipe that has its stdout/stderr closed
    out from under it (e.g. piped through PowerShell ``Select-Object -First N``,
    or a parent process that exits early) is to raise ``BrokenPipeError`` from
    the next ``print``/``logger`` call, which then propagates up and is
    swallowed by the interpreter's default SIGPIPE handling on shutdown — so
    the process exits 0 with no traceback even though the training loop
    terminated mid-run. The user sees a "successful" run that actually only
    completed iteration 0 (bug-bash items #24 and #25).

    This helper:

      - awaits the recipe coroutine,
      - logs any exception with full traceback (so even if stderr is closed,
        we have one last chance via the file logger),
      - exits with a non-zero code on ``BrokenPipeError`` / EPIPE so callers
        and CI can tell the run did not finish cleanly,
      - and re-raises everything else so existing error reporting is unchanged.
    """
    try:
        return await coro_factory()
    except BrokenPipeError:
        # stdout/stderr was closed out from under us. Logging here will likely
        # also fail, but try once via the file logger; then exit non-zero so
        # the run is not mistaken for a clean completion.
        try:
            logger.exception(
                "Recipe terminated by BrokenPipeError: stdout/stderr was closed "
                "while the training loop was still running. The run is "
                "incomplete. If you piped this command through 'Select-Object "
                "-First N', 'head', or similar, remove the pipe (write to a "
                "file with '> logs.txt' instead)."
            )
        except Exception:
            pass
        sys.exit(1)
    except OSError as exc:
        if exc.errno == errno.EPIPE:
            try:
                logger.exception(
                    "Recipe terminated by EPIPE: stdout/stderr was closed while "
                    "the training loop was still running. The run is incomplete."
                )
            except Exception:
                pass
            sys.exit(1)
        raise


def check_log_dir(log_dir: str, behavior_if_exists: LogdirBehavior):
    """
    Call this at the beginning of CLI entrypoint to training scripts. This handles
    cases that occur if we're trying to log to a directory that already exists.
    The user might want to resume, overwrite, or delete it.

    Args:
        log_dir: The directory to check.
        behavior_if_exists: What to do if the log directory already exists.

        "ask": Ask user if they want to delete the log directory.
        "resume": Continue to the training loop, which means we'll try to resume from the last checkpoint.
        "delete": Delete the log directory and start logging there.
        "raise": Raise an error if the log directory already exists.

    Returns:
        None
    """
    if os.path.exists(log_dir):
        if behavior_if_exists == "delete":
            logger.info(
                f"Log directory {log_dir} already exists. Will delete it and start logging there."
            )
            shutil.rmtree(log_dir)
        elif behavior_if_exists == "ask":
            while True:
                user_input = input(
                    f"Log directory {log_dir} already exists. What do you want to do? [delete, resume, exit]: "
                )
                if user_input == "delete":
                    shutil.rmtree(log_dir)
                    return
                elif user_input == "resume":
                    return
                elif user_input == "exit":
                    exit(0)
                else:
                    logger.warning(
                        f"Invalid input: {user_input}. Please enter 'delete', 'resume', or 'exit'."
                    )
        elif behavior_if_exists == "resume":
            return
        elif behavior_if_exists == "raise":
            raise ValueError(f"Log directory {log_dir} already exists. Will not delete it.")
        else:
            raise AssertionError(f"Invalid behavior_if_exists: {behavior_if_exists}")
    else:
        logger.info(
            f"Log directory {log_dir} does not exist. Will create it and start logging there."
        )
