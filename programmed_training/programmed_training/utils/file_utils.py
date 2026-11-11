import json
import os
import sys
from pathlib import Path


def set_private_file_permissions(descriptor: int, path: str | os.PathLike[str]) -> None:
    """Apply mode 0600, preferring the open descriptor over a path lookup.

    Windows Python <3.13 has no fchmod. Its path-based chmod only controls
    writability, not access ACLs; keep run directories access-controlled on
    Windows. POSIX descriptor permissions and permission errors are preserved.
    """
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, 0o600)
    else:
        os.chmod(path, 0o600)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def default_logs_root() -> Path:
    """Return the default root directory for run logs.

    Resolution order:
      1. ``$INTERACTIVE_POST_TRAINING_LOGS_ROOT`` environment variable, if set (any platform).
      2. ``C:\\interactive-post-training-runs`` on native Windows (short, fixed root that avoids
         OneDrive-redirected ``%USERPROFILE%`` paths and the 260-char path cap).
      3. ``~/interactive-post-training-runs`` everywhere else (Linux, macOS, WSL).

    Recipes write to ``<root>/<recipe>/<run_name>/`` and the dashboard scans
    ``<root>`` for runs. Override per-run with ``log_path=...`` on the CLI,
    or globally with ``INTERACTIVE_POST_TRAINING_LOGS_ROOT=/some/path``.
    """
    env = os.environ.get("INTERACTIVE_POST_TRAINING_LOGS_ROOT")
    if env:
        return Path(env).expanduser()
    if sys.platform.startswith("win"):
        return Path(r"C:\interactive-post-training-runs")
    return Path.home() / "interactive-post-training-runs"
