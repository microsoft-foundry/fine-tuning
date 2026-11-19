"""Exercise the actual logs helper and dashboard without opening a server."""

import builtins
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import dashboard_server as dashboard
from programmed_training.utils import file_utils


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ENV = "INTERACTIVE_POST_TRAINING_LOGS_ROOT"
LEGACY_ENV = "LOOM_LOGS_ROOT"


@pytest.fixture(autouse=True)
def isolated_logs_environment(monkeypatch, tmp_path):
    assert Path(dashboard.__file__).resolve() == ROOT / "dashboard_server.py"
    assert Path(file_utils.__file__).resolve() == (
        ROOT / "programmed_training" / "utils" / "file_utils.py"
    )
    monkeypatch.delenv(PUBLIC_ENV, raising=False)
    monkeypatch.delenv(LEGACY_ENV, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("empty_env", [False, True])
def test_default_write_root_preserves_preview_platform_defaults(monkeypatch, platform, empty_env):
    monkeypatch.setattr(file_utils, "sys", SimpleNamespace(platform=platform))
    if empty_env:
        monkeypatch.setenv(PUBLIC_ENV, "")
        monkeypatch.setenv(LEGACY_ENV, "")
    expected = Path(r"C:\interactive-post-training-runs") if platform == "win32" else Path.home() / "interactive-post-training-runs"

    assert file_utils.default_logs_root() == expected


@pytest.mark.parametrize(
    ("public", "legacy", "selected"),
    [
        ("~/public", None, "public"),
        ("~/public", "~/legacy", "public"),
        ("~/public", "", "public"),
        (None, "~/legacy", "legacy"),
        ("", "~/legacy", "legacy"),
    ],
)
def test_public_env_wins_and_loom_only_logging_still_works(monkeypatch, public, legacy, selected):
    if public is not None:
        monkeypatch.setenv(PUBLIC_ENV, public)
    if legacy is not None:
        monkeypatch.setenv(LEGACY_ENV, legacy)
    expected = Path.home() / selected

    assert file_utils.default_logs_root() == expected
    assert not expected.exists(), "Selecting a root must not create or duplicate logs"


def test_dashboard_temporary_root_constants_keep_both_actual_spellings():
    assert dashboard.LEGACY_TMP_ROOT == Path("/tmp/loom-examples")
    assert dashboard.RENAMED_TMP_ROOT == Path("/tmp/interactive-post-training-examples")


@pytest.fixture
def history_roots(monkeypatch, tmp_path):
    # Relocate real history directory names into the fixture, never the host's
    # /tmp or C:\ roots. Use the actual helper and discovery/routing code.
    monkeypatch.setattr(file_utils, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(dashboard, "sys", SimpleNamespace(platform="linux"))
    roots = [
        Path.home() / "loom-runs",
        Path.home() / "interactive-post-training-runs",
        tmp_path / "tmp" / "loom-examples",
        tmp_path / "tmp" / "interactive-post-training-examples",
    ]
    monkeypatch.setattr(dashboard, "LEGACY_TMP_ROOT", roots[2])
    monkeypatch.setattr(dashboard, "RENAMED_TMP_ROOT", roots[3])
    for root in roots:
        run = root / "math_rl" / "same-run"
        run.mkdir(parents=True)
        (run / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    return [root.resolve() for root in roots]


def test_automatic_scan_discovers_and_routes_legacy_and_renamed_history(history_roots):
    roots = dashboard.default_scan_roots()
    assert roots == [history_roots[1], history_roots[0], *history_roots[2:]]
    runs = dashboard.discover_runs(roots)
    assert len(runs) == 4
    assert len({run["id"] for run in runs}) == 4
    handler = object.__new__(dashboard.Handler)
    handler.runs_roots = roots
    assert {handler._resolve_run_dir(run["id"]) for run in runs} == {
        root / "math_rl" / "same-run" for root in roots
    }
    for run in runs:
        metrics = handler._resolve_run_file(run["id"], "metrics.jsonl")
        assert metrics is not None and metrics.is_file()


@pytest.mark.parametrize("public_selected", [False, True])
def test_configured_write_root_is_first_without_hiding_history(monkeypatch, history_roots, public_selected):
    monkeypatch.setenv(LEGACY_ENV, "~/configured-legacy")
    selected = Path.home() / "configured-legacy"
    if public_selected:
        monkeypatch.setenv(PUBLIC_ENV, "~/configured-public")
        selected = Path.home() / "configured-public"

    roots = dashboard.default_scan_roots()

    assert roots == [selected.resolve(), *history_roots]
    assert roots[0] == file_utils.default_logs_root().resolve()
    assert not selected.exists(), "A future write root should still be listed"
    if public_selected:
        assert (Path.home() / "configured-legacy").resolve() not in roots


def test_history_scan_deduplicates_resolved_aliases(monkeypatch, history_roots):
    legacy_tmp = history_roots[2]
    monkeypatch.setenv(PUBLIC_ENV, str(legacy_tmp / ".." / legacy_tmp.name))

    roots = dashboard.default_scan_roots()

    assert roots[0] == legacy_tmp
    assert len(roots) == len(set(roots)) == 4
    assert set(roots) == set(history_roots)


@pytest.mark.parametrize("public_selected", [False, True])
def test_uninstalled_dashboard_fallback_obeys_same_environment_precedence(
    monkeypatch, tmp_path, public_selected
):
    monkeypatch.setattr(dashboard, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(dashboard, "LEGACY_TMP_ROOT", tmp_path / "absent-legacy")
    monkeypatch.setattr(dashboard, "RENAMED_TMP_ROOT", tmp_path / "absent-renamed")
    monkeypatch.setenv(LEGACY_ENV, "~/legacy")
    if public_selected:
        monkeypatch.setenv(PUBLIC_ENV, "~/public")
    original_import = builtins.__import__

    def without_cookbook(name, *args, **kwargs):
        if name == "programmed_training.utils.file_utils":
            raise ImportError("cookbook is not installed")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", without_cookbook)
        roots = dashboard.default_scan_roots()

    expected = Path.home() / ("public" if public_selected else "legacy")
    assert roots == [expected.resolve()]


def test_explicit_cli_roots_replace_automatic_history_scan(monkeypatch, tmp_path):
    auto = tmp_path / "automatic"
    explicit = (Path.home() / "manual").resolve()
    monkeypatch.setattr(dashboard, "default_scan_roots", lambda: [auto])
    monkeypatch.setattr(dashboard.Handler, "runs_roots", [])
    monkeypatch.setattr(
        sys, "argv",
        ["dashboard_server.py", "--root", "~/manual", "--root", str(explicit), "--port", "8765"],
    )
    server = Mock()
    factory = Mock(return_value=server)
    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", factory)

    dashboard.main()

    assert dashboard.Handler.runs_roots == [explicit]
    factory.assert_called_once_with(("127.0.0.1", 8765), dashboard.Handler)
    server.serve_forever.assert_called_once_with()
    server.server_close.assert_called_once_with()