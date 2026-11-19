"""Extract only local fragments; never execute a whole notebook or cloud client."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared import sft_submission

ROOT = Path(__file__).parents[2]
RECIPES = [
    ("00-foundations/02-first-sft-bug-detection", "gpt-oss-120b", "bug-detection-oss", False),
    ("01-core-sft/01-news-summarization", "Muse-Glimmer-30B", "cnn-dailymail-muse", True),
    ("01-core-sft/02-scientific-summarization", "qwen3.8-27b", "pubmed-qwen38", False),
]


def source_text(cell):
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    # Preserve already terminated source fragments. Some retained cells store
    # unterminated logical lines instead of standard nbformat fragments.
    return "".join(source) if any(line.endswith(("\n", "\r")) for line in source) else "\n".join(source)


def notebook(relative):
    return json.loads((ROOT / relative / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))


def code_cell(document, marker):
    return next(
        source_text(cell) for cell in document["cells"]
        if cell["cell_type"] == "code" and marker in source_text(cell)
    )


def recipe_values(document):
    tree = ast.parse(code_cell(document, "EXPECTED_MODEL_VERSION ="))
    names = {"MODEL", "REGION", "EXPECTED_MODEL_VERSION", "TRAINING_TYPE", "HYPERPARAMETERS", "EXPECTED"}
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }


@pytest.mark.parametrize("relative,model,suffix,extra_body", RECIPES)
def test_owned_notebook_sources_parse_with_unique_ids_and_no_execution(relative, model, suffix, extra_body) -> None:
    document = notebook(relative)
    identifiers = [cell.get("id") or cell.get("metadata", {}).get("id") for cell in document["cells"]]
    assert all(identifiers) and len(set(identifiers)) == len(identifiers)
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(source_text(cell))
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
    values = recipe_values(document)
    assert values["MODEL"] == model
    assert values["EXPECTED_MODEL_VERSION"] == "1"
    assert "actual base-model version remains explicitly unverified" in source_text(document["cells"][0])


@pytest.mark.parametrize("relative,model,suffix,extra_body", RECIPES)
def test_submission_cell_delegates_exact_recipe_to_journal_only(tmp_path, monkeypatch, relative, model, suffix, extra_body) -> None:
    document = notebook(relative)
    values = recipe_values(document)
    calls = []
    client = object()  # Direct files/jobs calls on this object would fail.
    outputs = tmp_path / "outputs"
    state_path = outputs / "submission-state.json"

    def submit(actual_client, **kwargs):
        assert actual_client is client
        calls.append(kwargs)
        outputs.mkdir()
        state_path.write_text(json.dumps({
            "job_id": "ftjob-notebook", "training_file_id": "file-training",
            "validation_file_id": "file-validation",
            "version_evidence": sft_submission.model_version_evidence("1"),
        }), encoding="utf-8")
        return SimpleNamespace(id="ftjob-notebook", status="queued")

    monkeypatch.setattr(sft_submission, "submit_or_resume_sft_job", submit)
    namespace = {
        **values, "openai_client": client, "json": json,
        "STATE_PATH": state_path, "OUTPUT_DIR": outputs,
        "PROJECT_ENDPOINT": "https://unit-test.invalid/api/projects/notebook",
        "TRAINING_PATH": tmp_path / "training.jsonl", "VALIDATION_PATH": tmp_path / "validation.jsonl",
        "dataset_evidence": {kind: {"sha256": values["EXPECTED"][kind]["sha256"]}
                             for kind in ("training", "validation")},
    }
    source = code_cell(document, "fine_tune_job = submit_or_resume_sft_job(")
    exec(compile(source, "sft-submission-fragment", "exec"), namespace)
    assert len(calls) == 1
    submitted = calls[0]
    assert submitted["state_path"] == state_path
    assert submitted["outputs_root"] == outputs
    assert submitted["model"] == model
    assert submitted["expected_model_version"] == "1"
    assert submitted["hyperparameters"] == values["HYPERPARAMETERS"]
    assert submitted["training_type"] == values["TRAINING_TYPE"]
    assert submitted["suffix"] == suffix
    assert submitted.get("method_in_extra_body", False) is extra_body
    assert submitted["expected_training_sha256"] == values["EXPECTED"]["training"]["sha256"]
    assert submitted["expected_validation_sha256"] == values["EXPECTED"]["validation"]["sha256"]
    assert namespace["version_evidence"]["actual_model_version"] is None


@pytest.mark.parametrize("relative,model,suffix,extra_body", RECIPES)
def test_summary_fragment_never_reports_expected_version_as_actual(relative, model, suffix, extra_body) -> None:
    document = notebook(relative)
    source = code_cell(document, "summary = {")
    assignment = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "summary"
    )
    namespace = {
        "completed_job": SimpleNamespace(
            id="ftjob-notebook", status="succeeded", fine_tuned_model="artifact",
            trained_tokens=100, model_version="9", metadata={"model_version": "1"},
        ),
        "MODEL": model, "version_evidence": sft_submission.model_version_evidence("1"),
        "TRAINING_TYPE": "GlobalStandard", "HYPERPARAMETERS": {},
        "dataset_evidence": {}, "result_paths": [], "metrics": {}, "conclusion": "Training signals only",
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "sft-summary-fragment", "exec"), namespace)
    summary = namespace["summary"]
    assert summary["expected_model_version"] == "1"
    assert summary["actual_model_version"] is None
    assert summary["model_version_verification"] == "unverified"
    assert "model_version" not in summary