"""Dashboard discovery and routing agree and never widen the file allowlist."""

import pytest

from dashboard_server import Handler, discover_runs


def _handler(roots):
    handler = object.__new__(Handler)
    handler.runs_roots = roots
    return handler


@pytest.mark.parametrize("duplicate_names", [False, True])
def test_multi_root_discovered_runs_resolve(tmp_path, duplicate_names):
    roots = [
        tmp_path / "one" / "logs",
        tmp_path / "two" / ("logs" if duplicate_names else "other"),
    ]
    for root in roots:
        (root / "run").mkdir(parents=True)
        (root / "run" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    runs = discover_runs(roots)
    assert len({run["id"] for run in runs}) == 2
    assert {_handler(roots)._resolve_run_dir(run["id"]) for run in runs} == {
        root / "run" for root in roots
    }


@pytest.mark.parametrize("filename", ["eval_secret.txt", "train_iteration_key.env", "eval_data.json"])
def test_iteration_prefix_does_not_allow_non_html_files(tmp_path, filename):
    assert _handler([tmp_path])._resolve_run_file("run", filename) is None


@pytest.mark.parametrize("filename", ["metrics.jsonl", "run_meta.json", "eval_test_iteration_0.html"])
def test_allowed_run_files_still_resolve(tmp_path, filename):
    assert _handler([tmp_path])._resolve_run_file("run", filename) == tmp_path / "run" / filename


@pytest.mark.parametrize("run_id", ["../outside", "/outside", "run/../../outside"])
def test_run_path_traversal_is_rejected(tmp_path, run_id):
    assert _handler([tmp_path])._resolve_run_dir(run_id) is None