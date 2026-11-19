import json
import sys
from datetime import datetime, timezone

import pytest

from programmed_training.recipes.rl_numerics_check.report import build_report, chart
from programmed_training.recipes.rl_numerics_check import report
from programmed_training.utils.ml_log import JsonLogger


def write_run(path, *, depth=False):
    (path / "config.json").write_text(json.dumps({
        "model_name": "model <test>", "max_steps": 26,
    }))
    (path / "run_meta.json").write_text(json.dumps({
        "azure": {"session_id": "session_test"},
    }))
    rows = []
    for step in range(3):
        row = {
            "step": step, "optim/kl_sample_train_v1": 0.001,
            "time/total": 60,
        }
        if depth:
            row.update({
                "optim/kl_sample_train_by_depth/0_8k/tokens": 10,
                "optim/kl_sample_train_by_depth/0_8k/v1": 0.002,
            })
        rows.append(json.dumps(row))
    (path / "metrics.jsonl").write_text("\n".join(rows))


@pytest.mark.parametrize("depth", [False, True])
def test_report_uses_recorded_steps_and_does_not_invent_depth_data(tmp_path, depth):
    write_run(tmp_path, depth=depth)
    result = build_report([tmp_path])
    assert "Completed updates: 3" in result
    assert "third-update milestone: reached" in result
    assert "estimated remaining step time: 0.38 h" in result
    assert "model &lt;test&gt;" in result
    assert "No recorded data; not reconstructed." in result if not depth else (
        result.count("<svg ") == 2
    )
    assert "not recorded" in result
    assert "Missing depth buckets are not zero" in result
    assert "Primary metric: sampler/trainer KL v1 (optim/kl_sample_train_v1)" in result
    assert result.index("<th>optim/kl_sample_train_v1</th>") < result.index("<th>env/all/reward/total</th>")
    assert result.index("Primary: sampler/trainer KL v1 by optimizer-step index") < result.index("Step-0 KL v1")
    assert "This finite-sample estimator can be negative" in result


def test_chart_preserves_nonpositive_estimates_and_rejects_nonfinite_values():
    assert "linear (includes zero or negative estimates)" in chart("KL", [("run", [(0, -0.1), (1, 0)])])
    with pytest.raises(ValueError, match="Non-finite"):
        chart("KL", [("run", [(0, float("nan"))])])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_report_preserves_actual_invalid_logger_rows_in_step_table(tmp_path, value):
    write_run(tmp_path)
    (tmp_path / "metrics.jsonl").unlink()
    logger = JsonLogger(tmp_path)
    logger.log_metrics({"optim/kl_sample_train_v1": 0.02, "time/total": 60}, step=0)
    logger.log_metrics({
        "optim/kl_sample_train_v1": value, "time/total": value,
        "optim/kl_sample_train_by_depth/0_8k/tokens": value,
        "optim/kl_sample_train_by_depth/0_8k/v1": value,
    }, step=1)
    result = build_report([tmp_path])
    assert "Completed updates: 2" in result
    assert "invalid measurement" in result and "__nonfinite__" in result
    assert "excluded from numeric charts" in result
    assert "estimated remaining step time" not in result


def test_report_rejects_duplicate_step_records(tmp_path):
    write_run(tmp_path)
    path = tmp_path / "metrics.jsonl"
    first = path.read_text().splitlines()[0]
    path.write_text(first + "\n" + first)
    with pytest.raises(ValueError, match="unique and increasing"):
        build_report([tmp_path])


def test_report_handles_no_step_limit_without_inventing_eta(tmp_path):
    write_run(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_name": "test", "max_steps": None}))
    result = build_report([tmp_path])
    assert "estimated remaining step time" not in result


def test_report_distinguishes_launcher_elapsed_time_from_final_loop_timing(tmp_path):
    write_run(tmp_path)
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "azure": {"session_id": "test"},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_preparation_seconds": 60,
    }))
    result = build_report([tmp_path])
    assert "Elapsed since launcher start" in result
    assert "dataset preparation: 1.0 min" in result
    assert "not proof the process is still alive" in result
    (tmp_path / "timing.json").write_text(json.dumps({"e2e_wall_clock_seconds": 120}))
    result = build_report([tmp_path])
    assert "Recorded training-loop wall time: 2.0 min" in result
    assert "Elapsed since launcher start" not in result


@pytest.mark.parametrize("initial_metrics", [False, True])
def test_report_cli_follows_explicit_recovery_lineage_without_merging_updates(
    tmp_path, monkeypatch, initial_metrics,
):
    child = tmp_path / "recovery_attempts" / "attempt-1"
    child.mkdir(parents=True)
    if initial_metrics:
        write_run(tmp_path)
    write_run(child)
    (child / "run_meta.json").write_text(json.dumps({"azure": {"session_id": "session_replacement"}}))
    (tmp_path / "recovery.jsonl").write_text(
        json.dumps({"event": "attempt", "log_path": str(tmp_path)}) + "\n"
        + json.dumps({"event": "attempt", "log_path": str(child)}) + "\n"
    )
    output = tmp_path / "evidence.html"
    monkeypatch.setattr(sys, "argv", [
        "report", str(tmp_path), "--include-recovery-attempts", "--output", str(output),
    ])
    report.main()
    result = output.read_text()
    assert "session_replacement" in result
    assert result.count("Completed updates: 3") == (2 if initial_metrics else 1)
    assert "Completed updates: 6" not in result
    assert ("Recorded attempts without completed metrics" in result) is not initial_metrics


def test_report_cli_rejects_recovery_lineage_without_any_recorded_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "report", str(tmp_path), "--include-recovery-attempts", "--output", str(tmp_path / "report.html"),
    ])
    with pytest.raises(ValueError, match="No completed step metrics"):
        report.main()
    assert not (tmp_path / "report.html").exists()
