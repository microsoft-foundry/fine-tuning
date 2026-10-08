#!/usr/bin/env python3
"""Tiny dashboard server: serves dashboard.html and run files from logs roots.

Usage:
    python dashboard_server.py                       # default roots on :8000
    python dashboard_server.py --root /tmp/x         # custom logs root
    python dashboard_server.py --root A --root B     # multiple roots
    python dashboard_server.py --port 9000

Default logs roots scanned (any that exist):
    1. $INTERACTIVE_POST_TRAINING_LOGS_ROOT (if set), else ~/interactive-post-training-runs (Linux/macOS) / C:\\interactive-post-training-runs (Windows)
    2. /tmp/interactive-post-training-examples (legacy path used by some recipes)

When runs come from multiple roots, IDs are prefixed by the root's directory
name (e.g. ``interactive-post-training-runs/math_rl/<run>`` vs ``interactive-post-training-examples/<run>``) to keep
them unique.

The dashboard auto-detects the recipe type (RL vs SFT) from each run's
``metrics.jsonl`` and shows the appropriate charts and KPIs. Open
http://localhost:8000/ and pick a run from the dropdown.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
DASHBOARD_HTML = HERE / "dashboard.html"
LEGACY_TMP_ROOT = Path("/tmp/interactive-post-training-examples")

# Files we'll let the dashboard fetch from a run directory. Keep this strict.
ALLOWED_RUN_FILES = {"metrics.jsonl", "config.json", "logs.log", "run_meta.json"}
ALLOWED_RUN_PREFIXES = ("train_iteration_", "eval_test_iteration_", "eval_")


def _root_prefixes(roots: list[Path]) -> dict[Path, str]:
    """Use identical, collision-safe root IDs for discovery and request routing."""
    counts: dict[str, int] = {}
    for root in roots:
        counts[root.name] = counts.get(root.name, 0) + 1
    prefixes = {}
    for root in roots:
        if len(roots) == 1:
            prefixes[root] = ""
        elif counts[root.name] > 1:
            digest = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:6]
            prefixes[root] = f"{root.name}-{digest}/"
        else:
            prefixes[root] = f"{root.name}/"
    return prefixes


def discover_runs(roots: list[Path]) -> list[dict]:
    """Find directories under any of ``roots`` that look like run dirs.

    A run dir is any directory containing metrics.jsonl OR config.json.
    When multiple roots are scanned, run IDs are prefixed by the root's
    directory name to keep them unique (and human-readable in the dropdown).
    Returns a list of {id, label, mtime} sorted by mtime desc.
    """
    runs: list[dict] = []
    prefixes = _root_prefixes(roots)

    seen_ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        prefix = prefixes[root]
        for dirpath, _dirnames, filenames in os.walk(root):
            if "metrics.jsonl" in filenames or "config.json" in filenames:
                p = Path(dirpath)
                rel = p.relative_to(root).as_posix() or p.name
                run_id = f"{prefix}{rel}"
                if run_id in seen_ids:
                    continue
                seen_ids.add(run_id)
                try:
                    mtime = max(
                        (p / f).stat().st_mtime
                        for f in filenames
                        if f in ALLOWED_RUN_FILES
                    )
                except ValueError:
                    mtime = p.stat().st_mtime
                runs.append({"id": run_id, "label": run_id, "mtime": mtime})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def list_iterations(run_dir: Path) -> list[str]:
    if not run_dir.is_dir():
        return []
    out = [
        f.name
        for f in run_dir.iterdir()
        if f.is_file()
        and f.suffix == ".html"
        and f.name.startswith(ALLOWED_RUN_PREFIXES)
    ]
    out.sort()
    return out


def safe_join(root: Path, *parts: str) -> Path | None:
    """Join parts under root, refusing path traversal. Returns None if unsafe."""
    candidate = (root.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class Handler(BaseHTTPRequestHandler):
    # set on the class by main()
    runs_roots: list[Path] = [Path("/tmp/interactive-post-training-examples")]
    dashboard_html: Path = DASHBOARD_HTML

    def log_message(self, fmt: str, *args) -> None:  # quieter
        return

    # -- helpers ----------------------------------------------------------
    def _resolve_run_dir(self, run_id: str) -> Path | None:
        """Map a run_id back to an absolute run directory under one of the roots.

        In multi-root mode, IDs are prefixed by the root's directory name.
        Falls back to single-root behaviour (treat the ID as a relative path)
        when only one root is configured.
        """
        if not run_id or ".." in run_id.split("/") or run_id.startswith("/"):
            return None
        roots = self.runs_roots
        if len(roots) == 1:
            return safe_join(roots[0], run_id)
        head, _, rest = run_id.partition("/")
        for root, prefix in _root_prefixes(roots).items():
            if prefix.removesuffix("/") == head:
                return safe_join(root, rest) if rest else root
        return None

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(404, f"Not found: {path.name}")
            return
        if content_type is None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            data = path.read_bytes()
        except OSError as e:
            self.send_error(500, str(e))
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _resolve_run_file(self, run_id: str, filename: str) -> Path | None:
        if (
            filename not in ALLOWED_RUN_FILES
            and not (
                filename.endswith(".html")
                and filename.startswith(ALLOWED_RUN_PREFIXES)
            )
        ):
            return None
        if "/" in filename or "\\" in filename:
            return None
        run_dir = self._resolve_run_dir(run_id)
        if run_dir is None:
            return None
        return safe_join(run_dir, filename)

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/dashboard.html", "/dashboard_sft.html", "/index.html"):
            self._send_file(self.dashboard_html, "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            runs = discover_runs(self.runs_roots)
            self._send_json({
                "roots": [str(r) for r in self.runs_roots],
                # kept for backwards-compatibility with older dashboards that
                # may read `root` rather than `roots`.
                "root": str(self.runs_roots[0]) if self.runs_roots else "",
                "runs": runs,
            })
            return

        # /api/run/<run_id...>/iterations
        # /api/run/<run_id...>/file/<filename>
        if path.startswith("/api/run/"):
            rest = path[len("/api/run/") :]
            if rest.endswith("/iterations"):
                run_id = rest[: -len("/iterations")]
                run_dir = self._resolve_run_dir(run_id)
                if run_dir is None or not run_dir.is_dir():
                    self.send_error(404, "Unknown run")
                    return
                self._send_json({"iterations": list_iterations(run_dir)})
                return
            if "/file/" in rest:
                run_id, _, filename = rest.partition("/file/")
                target = self._resolve_run_file(run_id, filename)
                if target is None:
                    self.send_error(400, "Invalid path")
                    return
                self._send_file(target)
                return

        self.send_error(404, "Not found")


def main() -> None:
    # Late import so `--help` works even if interactive_training isn't on sys.path.
    # default_logs_root() already honors $INTERACTIVE_POST_TRAINING_LOGS_ROOT, so we don't need to
    # re-check it in the argparse default (doing so would also bypass ~ expansion).
    try:
        from interactive_training.utils.file_utils import default_logs_root
        default_root = str(default_logs_root())
    except ImportError:
        if os.name == "nt":
            default_root = r"C:\interactive-post-training-runs"
        else:
            default_root = str(Path.home() / "interactive-post-training-runs")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "Directory to scan for run folders. Pass multiple times to scan "
            f"several roots. Default: {default_root} (plus {LEGACY_TMP_ROOT} "
            "if it exists)."
        ),
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host.lower() == "localhost"
    if not loopback:
        print(
            "WARNING: this dashboard has no authentication. Non-loopback binding "
            "can expose run configuration, logs, and training data. Prefer a "
            "private tunnel or an authenticated proxy.",
            file=sys.stderr,
        )

    if args.root:
        roots = [Path(r).resolve() for r in args.root]
    else:
        roots = [Path(default_root).resolve()]
        # Auto-include the legacy /tmp/interactive-post-training-examples location if it has
        # anything in it, so older runs are still visible without flags.
        if LEGACY_TMP_ROOT.exists() and LEGACY_TMP_ROOT.resolve() not in roots:
            roots.append(LEGACY_TMP_ROOT.resolve())

    # De-dupe while preserving order.
    seen: set[Path] = set()
    Handler.runs_roots = [r for r in roots if not (r in seen or seen.add(r))]
    Handler.dashboard_html = DASHBOARD_HTML
    print(f"Serving dashboard at http://{args.host}:{args.port}/")
    print("Logs roots:")
    for r in Handler.runs_roots:
        tag = "" if r.exists() else "  (does not exist yet)"
        print(f"  - {r}{tag}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
