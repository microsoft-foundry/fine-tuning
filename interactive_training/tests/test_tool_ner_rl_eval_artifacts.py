import json
import os
import stat

from pathlib import Path

import pytest

from interactive_training.recipes.tool_ner_rl.eval_artifacts import (
    NerEvalArtifactObserver,
    _overlap_metrics,
    _tab_document_summary,
    _tab_reference_rows,
)

from interactive_training.recipes.tool_ner_rl.ner_task import GoldEntity, NerTask

from interactive_training.rl.metric_util import ValidationRecord

def _assert_posix_permissions(path):
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _record(uid, source, predictions, gold):
    payload = {
        "uid": uid,
        "source_text": source,
        "predictions": predictions,
        "gold": gold,
        "turns": 2,
        "generated_tokens": 300,
        "finalized": True,
        "parse_error": False,
        "hit_token_cap": False,
        "tool_counts": {"ground_entities": 1, "finalize": 1},
    }
    return ValidationRecord(
        step=0,
        input=uid,
        response=json.dumps(payload),
        ground_truth=None,
        tags=("openpii", "en"),
    )

def test_observer_writes_exact_metrics_and_deterministic_error_sample(tmp_path):
    source = "Jane Smith"
    erroneous = _record(
        "error-doc",
        source,
        predictions=[
            {"start": 0, "end": 4, "label": "SURNAME", "text": "Jane"},
            {"start": 5, "end": 9, "label": "SURNAME", "text": "Smit"},
        ],
        gold=[
            {"start": 0, "end": 4, "label": "GIVENNAME", "text": "Jane"},
            {"start": 5, "end": 10, "label": "SURNAME", "text": "Smith"},
        ],
    )
    perfect = _record(
        "perfect-doc",
        "42",
        predictions=[{"start": 0, "end": 2, "label": "AGE", "text": "42"}],
        gold=[{"start": 0, "end": 2, "label": "AGE", "text": "42"}],
    )
    observer = NerEvalArtifactObserver(tmp_path, seed=42, error_sample_size=1)

    observer.on_validation_complete([erroneous, perfect])

    summary = json.loads((tmp_path / "error_summary.json").read_text())
    sample = json.loads((tmp_path / "error_review_sample.json").read_text())
    protocol_errors = json.loads((tmp_path / "protocol_errors.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "per_document.jsonl").read_text().splitlines()]

    assert summary["documents"] == 2
    assert summary["totals"] == {"tp": 1, "fp": 2, "fn": 2}
    assert summary["micro"]["f1"] == pytest.approx(1 / 3)
    assert summary["labeled_overlap"]["f1"] == pytest.approx(2 / 3)
    assert summary["binary_overlap"]["f1"] == pytest.approx(1.0)
    assert summary["binary_pii_exact_span"] == {
        "label": "PII",
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(2 / 3),
        "unit": "entity_span",
        "wrong_label_on_exact_span": 1,
        "label_accuracy_on_exact_spans": pytest.approx(0.5),
    }
    assert summary["micro_f1_document_bootstrap_95ci"]["samples"] == 2_000
    assert summary["micro_f1_document_bootstrap_95ci"]["seed"] == 42
    assert 0.0 <= summary["micro_f1_document_bootstrap_95ci"]["p2_5"]
    assert summary["micro_f1_document_bootstrap_95ci"]["p97_5"] <= 1.0
    assert summary["error_kinds"] == {"boundary_mismatch": 2, "label_mismatch": 2}
    assert summary["per_label"]["AGE"]["f1"] == 1.0
    assert summary["protocol"]["finalized_documents"] == 2
    assert summary["protocol"]["parse_error_documents"] == 0
    assert summary["protocol"]["token_cap_documents"] == 0
    assert summary["density_slices"]["1-4"]["f1"] == pytest.approx(1 / 3)
    assert sample[0]["uid"] == "error-doc"
    assert protocol_errors == []
    assert [row["uid"] for row in rows] == ["error-doc", "perfect-doc"]
    assert observer.summary_metrics["base/eval/error_documents"] == 1.0
    assert observer.summary_metrics["base/eval/finalized_rate"] == 1.0
    assert observer.summary_metrics["base/eval/binary_pii_f1"] == pytest.approx(20 / 21)
    assert observer.summary_metrics["base/eval/binary_pii_exact_span_f1"] == pytest.approx(
        2 / 3
    )
    assert observer.summary_metrics["base/eval/labeled_overlap_f1"] == pytest.approx(2 / 3)
    assert observer.summary_metrics["base/eval/binary_overlap_f1"] == 1.0

    for filename in (
        "error_summary.json",
        "error_review_sample.json",
        "per_document.jsonl",
        "protocol_errors.json",
    ):
        _assert_posix_permissions(tmp_path / filename)


def test_private_writer_replaces_old_file_atomically_with_posix_permissions(tmp_path):
    from interactive_training.recipes.tool_ner_rl.private_files import private_text_writer

    output = tmp_path / "artifact.json"
    output.write_text("old", encoding="utf-8")
    if os.name == "posix":
        output.chmod(0o644)
    with private_text_writer(output) as stream:
        temporary_path, = tmp_path.glob(".artifact.json.*.tmp")
        _assert_posix_permissions(temporary_path)
        assert temporary_path.stat().st_size == 0
        stream.write("new")
        assert output.read_text() == "old"
    assert output.read_text() == "new"
    _assert_posix_permissions(output)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("writer_name", ["json", "jsonl", "manifest"])
def test_private_writers_preserve_existing_output_on_failure(tmp_path, writer_name):
    from interactive_training.recipes.tool_ner_rl.eval_artifacts import _write_private_json, _write_private_jsonl
    from interactive_training.recipes.tool_ner_rl.offline_filter import write_manifest

    writer = {"json": _write_private_json, "jsonl": _write_private_jsonl, "manifest": write_manifest}[writer_name]
    output = tmp_path / "artifact.json"
    output.write_text("old", encoding="utf-8")
    value = [{"invalid": object()}] if writer_name == "jsonl" else {"invalid": object()}
    with pytest.raises(TypeError):
        writer(output, value)
    assert output.read_text() == "old"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink creation does not require Windows privileges")
def test_private_writer_replaces_symlink_without_writing_target(tmp_path):
    from interactive_training.recipes.tool_ner_rl.eval_artifacts import _write_private_json

    target = tmp_path / "target"
    target.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "artifact.json"
    output.symlink_to(target)
    _write_private_json(output, {"synthetic": "example"})
    assert target.read_text() == "unchanged"
    assert not output.is_symlink()
    _assert_posix_permissions(output)

def test_binary_pii_is_perfect_when_exact_spans_have_swapped_labels(tmp_path):
    observer = NerEvalArtifactObserver(tmp_path, seed=42)
    observer.on_validation_complete(
        [
            _record(
                "swapped",
                "Jane Smith",
                predictions=[
                    {"start": 0, "end": 4, "label": "SURNAME", "text": "Jane"},
                    {"start": 5, "end": 10, "label": "GIVENNAME", "text": "Smith"},
                ],
                gold=[
                    {"start": 0, "end": 4, "label": "GIVENNAME", "text": "Jane"},
                    {"start": 5, "end": 10, "label": "SURNAME", "text": "Smith"},
                ],
            )
        ]
    )

    summary = json.loads((tmp_path / "error_summary.json").read_text())
    assert summary["micro"]["f1"] == 0.0
    assert summary["binary_pii"]["f1"] == 1.0
    assert summary["binary_pii_exact_span"]["f1"] == 1.0
    assert summary["binary_pii_exact_span"]["wrong_label_on_exact_span"] == 2
    assert summary["binary_pii_exact_span"]["label_accuracy_on_exact_spans"] == 0.0
    assert summary["labeled_overlap"]["f1"] == 0.0
    assert summary["binary_overlap"]["f1"] == 1.0

def test_binary_pii_ignores_name_segmentation_and_inter_entity_whitespace(tmp_path):
    observer = NerEvalArtifactObserver(tmp_path, seed=42)
    observer.on_validation_complete(
        [
            _record(
                "merged-name",
                "John Smith",
                predictions=[
                    {"start": 0, "end": 10, "label": "GIVENNAME", "text": "John Smith"},
                ],
                gold=[
                    {"start": 0, "end": 4, "label": "GIVENNAME", "text": "John"},
                    {"start": 5, "end": 10, "label": "SURNAME", "text": "Smith"},
                ],
            )
        ]
    )

    summary = json.loads((tmp_path / "error_summary.json").read_text())
    assert summary["micro"]["f1"] == 0.0
    assert summary["binary_pii_exact_span"]["f1"] == 0.0
    assert summary["binary_pii"]["f1"] == 1.0
    assert summary["binary_pii"]["unit"] == "non_whitespace_source_character"
    assert summary["labeled_overlap"]["f1"] == pytest.approx(2 / 3)
    assert summary["binary_overlap"]["f1"] == pytest.approx(2 / 3)

def test_overlap_uses_maximum_cardinality_one_to_one_matching():
    predictions = {
        GoldEntity(0, 10, "GIVENNAME"),
        GoldEntity(0, 4, "GIVENNAME"),
    }
    gold = {
        GoldEntity(0, 4, "GIVENNAME"),
        GoldEntity(5, 10, "GIVENNAME"),
    }

    metrics = _overlap_metrics(predictions, gold, require_same_label=True)

    assert metrics == {
        "true_positives": 2,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }

def test_tab_document_summary_deduplicates_overlap_and_scores_native_recall():
    shared = {
        "dataset_name": "tab",
        "document_uid": "case-1",
        "annotation_layer": "annotator1",
        "source_start": 0,
        "source_text": "Alice and Bob",
    }
    rows = [
        {
            **shared,
            "predictions": [
                {"start": 0, "end": 5, "label": "PERSON", "text": "Alice"}
            ],
            "gold": [
                {"start": 0, "end": 5, "label": "PERSON", "text": "Alice"},
                {"start": 10, "end": 13, "label": "PERSON", "text": "Bob"},
            ],
            "gold_attributes": [
                {
                    "start": 0,
                    "end": 5,
                    "label": "PERSON",
                    "identifier_type": "DIRECT",
                    "entity_id": "person-1",
                },
                {
                    "start": 10,
                    "end": 13,
                    "label": "PERSON",
                    "identifier_type": "QUASI",
                    "entity_id": "person-1",
                },
            ],
        },
        {
            **shared,
            "predictions": [
                {"start": 0, "end": 5, "label": "PERSON", "text": "Alice"}
            ],
            "gold": [],
            "gold_attributes": [],
        },
    ]

    summary = _tab_document_summary(rows)

    assert summary is not None
    assert summary["exact"]["true_positives"] == 1
    assert summary["exact"]["false_positives"] == 0
    assert summary["exact"]["false_negatives"] == 1
    assert summary["identifier_recall"]["direct"]["recall"] == 1.0
    assert summary["identifier_recall"]["quasi"]["recall"] == 0.0
    assert summary["coreference_complete_masking"]["recall"] == 0.0

def test_tab_all_references_receive_identical_predictions_and_missing_windows_fail():
    tasks = tuple(NerTask(
        uid=layer, source_text="Jane", labels=("PERSON",),
        gold_entities=frozenset({GoldEntity(0, 4, "PERSON")}) if layer == "one" else frozenset(),
        dataset_name="tab", document_uid="doc", annotation_layer=layer,
    ) for layer in ("one", "two"))
    sampled = [{"document_uid": "doc", "source_start": 0, "source_text": "Jane",
                "predictions": [{"start": 0, "end": 4, "label": "PERSON"}]}]
    rows = _tab_reference_rows(sampled, tasks)
    assert rows[0]["predictions"] == rows[1]["predictions"]
    assert len(rows[0]["gold"]) == 1
    assert not rows[1]["gold"]
    with pytest.raises(ValueError, match="exactly one"):
        _tab_reference_rows([], tasks)
    with pytest.raises(ValueError, match="exactly one"):
        _tab_reference_rows(sampled + sampled, tasks)
