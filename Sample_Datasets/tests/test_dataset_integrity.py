"""Offline regression tests; do not execute any dataset-supplied Python code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SAMPLES = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sample_dataset_integrity", SAMPLES / "dataset_integrity.py")
assert SPEC is not None and SPEC.loader is not None
integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integrity)


def chat(code: str) -> dict:
    return {"messages": [{"role": "user", "content": "Unit-test-only Python target"},
                         {"role": "assistant", "content": code}]}


@pytest.mark.parametrize("target", [
    "answer = 42\n",
    "```python\nanswer = 42\n```",
    "  ```py\r\nanswer = 42\r\n```\n",
    "```\nanswer = 42\n```",
    "raise AssertionError('parsing this target must not execute it')",
])
def test_ast_audit_accepts_only_parsable_targets_without_execution(target):
    assert integrity.code_diagnostics(chat(target)) == []


@pytest.mark.parametrize("target", [
    "def broken(:\n    pass",
    "```python\ndef broken(:\n    pass\n```",
    "Here is the code:\n```python\nanswer = 42\n```",
    "```python\nanswer = 42\n```\nThe answer is above.",
    "```python\nanswer = 42\n```\n```python\nother = 1\n```",
    "", "   ",
])
def test_ast_audit_does_not_guess_prose_or_repair_broken_answers(target):
    assert integrity.code_diagnostics(chat(target))


def test_ast_audit_rejects_ambiguous_target_shape():
    row = chat("answer = 42")
    row["messages"].append({"role": "assistant", "content": "other = 1"})
    assert integrity.code_diagnostics(row)[0]["kind"] == "chat_shape"


@pytest.fixture(scope="module")
def code_audit():
    return integrity.audit_code_sources()


def test_real_code_sources_match_both_pinned_copies_and_have_explicit_decisions(code_audit):
    report, records = code_audit
    assert report["syntax_target"] == "3.11"
    assert report["semantic_or_runtime_validation"] is False
    for split, expected in integrity.CODE_SOURCES.items():
        source = report["sources"][split]
        assert source["rows"] == expected["rows"] == len(records[split])
        assert source["sha256"] == expected["sha256"]
        assert source["accepted"] + source["rejected"] == source["rows"]
        assert [record["source_line"] for record in source["records"]] == list(range(1, source["rows"] + 1))
        assert source["rejected"] > 0, "Original generated sources must not be claimed all AST-valid"
        for entry, decision in zip(records[split], source["records"]):
            assert entry["sha256"] == decision["raw_line_sha256"]
            assert bool(decision["diagnostics"]) == (decision["decision"] == "reject")


def test_review_ast_failure_counts_are_recomputed_not_assumed(code_audit):
    report, _ = code_audit
    status = integrity.loads((SAMPLES / "validation-status.json").read_text(encoding="utf-8"))
    for split, reported in status["nl_to_python"]["review_reported_ast_invalid_rows"].items():
        observed = report["sources"][split]["rejected"]
        assert observed == reported, (
            f"Review reported {reported} {split} failures; observed {observed} with {report['parser']}. "
            "Investigate parser/wrapper/source differences; do not claim the review count verified."
        )


def test_explicit_curation_preserves_source_and_ledgers_every_exclusion(tmp_path, code_audit):
    original_report, records = code_audit
    output = tmp_path / "curated-v1"
    manifest = integrity.curate_code(output)
    assert manifest["sources"] == original_report["sources"]
    assert integrity.loads((output / "curation-manifest.json").read_text(encoding="utf-8")) == manifest
    for split, entries in records.items():
        decisions = manifest["sources"][split]["records"]
        expected = b"".join(entry["raw"] for entry, decision in zip(entries, decisions)
                            if decision["decision"] == "accept")
        name = f"chat_{split}.curated.jsonl"
        actual = (output / name).read_bytes()
        assert actual == expected
        assert integrity.digest(actual) == manifest["artifacts"][name]["sha256"]
        _, curated = integrity.read_records(output / name)
        assert len(curated) == manifest["sources"][split]["accepted"]
        assert all(not integrity.code_diagnostics(entry["row"]) for entry in curated)
        original = SAMPLES / integrity.CODE_SOURCES[split]["sample"]
        assert integrity.digest(integrity.canonical_source_bytes(original.read_bytes())) == integrity.CODE_SOURCES[split]["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        integrity.curate_code(output)


@pytest.mark.parametrize("sample_eol,maintained_eol", [
    (b"\n", b"\n"), (b"\n", b"\r\n"), (b"\r\n", b"\n"), (b"\r\n", b"\r\n"),
])
def test_code_provenance_matches_git_and_windows_bytes_without_rewriting_sources(
    tmp_path, sample_eol, maintained_eol,
):
    for expected in integrity.CODE_SOURCES.values():
        canonical = integrity.canonical_source_bytes((SAMPLES / expected["sample"]).read_bytes())
        for relative, eol in (("Sample_Datasets/" + expected["sample"], sample_eol),
                              (expected["maintained"], maintained_eol)):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical.replace(b"\n", eol))
    report, _ = integrity.audit_code_sources(tmp_path)
    for split, expected in integrity.CODE_SOURCES.items():
        assert report["sources"][split]["sha256"] == expected["sha256"]
        assert report["sources"][split]["archive_raw_sha256"] == expected["archive_raw_sha256"]
    # Newline canonicalization must not accept changed supervision.
    path = tmp_path / "Sample_Datasets" / integrity.CODE_SOURCES["train"]["sample"]
    path.write_bytes(path.read_bytes().replace(b"assistant", b"ASSISTANT", 1))
    with pytest.raises(ValueError, match="pinned provenance"):
        integrity.audit_code_sources(tmp_path)


def test_medmcq_missing_label_is_not_an_empty_or_guessed_reference():
    source = SAMPLES / "Reinforcement_Fine_Tuning" / "MedMCQ" / "medmcq_validation.jsonl"
    with source.open(encoding="utf-8") as stream:
        first = integrity.loads(next(stream))
    assert "reference_answer" not in first
    assert any("reference_answer" in problem for problem in integrity.medmcq_diagnostics(first))
    grader = integrity.loads((source.parent / "stringcheck_grader.json").read_text(encoding="utf-8"))
    assert "{{item.reference_answer}}" in (grader["input"], grader["reference"])


def test_review_medmcq_missing_label_extent_is_recomputed_from_validation_bytes():
    status = integrity.loads((SAMPLES / "validation-status.json").read_text(encoding="utf-8"))
    source = SAMPLES / "Reinforcement_Fine_Tuning" / "MedMCQ" / "medmcq_validation.jsonl"
    _, entries = integrity.read_records(source)
    missing = [entry for entry in entries if "reference_answer" not in entry["row"]]
    assert len(missing) == status["medmcq"]["review_reported_validation_rows_missing_labels"]
    assert len(missing) == len(entries)
    assert all(integrity.medmcq_diagnostics(entry["row"]) for entry in entries)


def original_label(**changes):
    # Minimal synthetic unit-test fixture, never a medical dataset answer.
    return {"id": "unit-fixture-1", "question": "Unit fixture question?",
            "opa": "First", "opb": "Second", "opc": "Third", "opd": "Fourth",
            "cop": 3, "choice_type": "single", **changes}


def label_inputs(tmp_path, labels):
    labels_path = tmp_path / "original-labels.jsonl"
    labels_raw = b"".join((json.dumps(label) + "\n").encode("utf-8") for label in labels)
    labels_path.write_bytes(labels_raw)
    target = tmp_path / "unlabelled.jsonl"
    target.write_text(json.dumps({"messages": [{"role": "user", "content":
                       integrity.original_medmcqa_prompt(original_label())}]}) + "\n", encoding="utf-8")
    return target, labels_path, integrity.digest(labels_raw)


def restore(target, labels, digest, output):
    return integrity.prepare_medmcq(target, labels, digest, "https://example.org/original-data",
                                    "unit-test-immutable-archive-v1", output)


def test_medmcq_restoration_is_exact_unique_and_source_traceable(tmp_path):
    target, labels, digest = label_inputs(tmp_path, [original_label()])
    original_target = target.read_bytes()
    manifest = restore(target, labels, digest, tmp_path / "prepared")
    _, prepared = integrity.read_records(tmp_path / "prepared" / manifest["artifact"]["name"])
    assert prepared[0]["row"]["reference_answer"] == "C"
    assert not integrity.medmcq_diagnostics(prepared[0]["row"])
    assert manifest["records"][0]["source_id"] == "unit-fixture-1"
    assert manifest["records"][0]["source_line"] == 1
    assert manifest["artifact"]["sha256"] == integrity.digest((tmp_path / "prepared" / manifest["artifact"]["name"]).read_bytes())
    assert target.read_bytes() == original_target


@pytest.mark.parametrize("labels", [
    [original_label(), original_label(id="duplicate-even-with-the-same-answer")],
    [original_label(), original_label(id="conflicting-label", cop=1)],
    [original_label(opa="Different option, not an exact match")],
    [original_label(question="Different question, not a positional match")],
    [original_label(cop=0)],
    [original_label(cop=True)],
    [original_label(choice_type="multi")],
])
def test_medmcq_refuses_ambiguous_unmatched_or_nonoriginal_labels_without_outputs(tmp_path, labels):
    target, labels_path, digest = label_inputs(tmp_path, labels)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        restore(target, labels_path, digest, output)
    assert not output.exists()


def test_medmcq_wrong_digest_or_existing_conflicting_answer_is_blocked(tmp_path):
    target, labels, digest = label_inputs(tmp_path, [original_label()])
    with pytest.raises(ValueError, match="SHA-256"):
        restore(target, labels, "0" * 64, tmp_path / "wrong-digest")
    row = integrity.loads(target.read_text(encoding="utf-8"))
    row["reference_answer"] = "A"
    target.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        restore(target, labels, digest, tmp_path / "wrong-answer")
    assert not (tmp_path / "wrong-digest").exists()
    assert not (tmp_path / "wrong-answer").exists()


@pytest.mark.parametrize("content", [None, "omitted"])
def test_sft_tool_target_content_union_accepts_null_or_absent_only_with_calls(content):
    assistant = {"role": "assistant", "tool_calls": [{"id": "fixture-call", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"}}]}
    if content is None:
        assistant["content"] = None
    row = {"messages": [{"role": "user", "content": "Unit fixture"}, assistant]}
    integrity.validate_sft_target_row(row)
    with pytest.raises(ValueError, match="incomplete tool replies"):
        integrity.validate_sft_row(row)
    row["messages"].append({"role": "tool", "tool_call_id": "fixture-call", "content": "{}"})
    integrity.validate_sft_row(row)
    with pytest.raises(ValueError, match="omit/null"):
        integrity.validate_sft_row({"messages": [{"role": "user", "content": "Unit fixture"},
                                                  {"role": "assistant", "content": None}]})


def tool_trace(call_count=1):
    return {"messages": [
        {"role": "user", "content": "Unit fixture"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"fixture-{number}", "type": "function",
             "function": {"name": "lookup", "arguments": "{}"}}
            for number in range(call_count)
        ]},
    ]}


@pytest.mark.parametrize("call_count", [1, 2, 3])
def test_dialogue_rejects_unanswered_final_assistant_calls_unconditionally(call_count):
    row = tool_trace(call_count)
    with pytest.raises(ValueError, match="incomplete tool replies"):
        integrity.validate_sft_row(row)
    integrity.validate_sft_target_row(row)


@pytest.mark.parametrize("validator", ["validate_sft_row", "validate_sft_target_row"])
def test_neither_schema_accepts_a_partially_answered_batch(validator):
    row = tool_trace(2)
    row["messages"].append({"role": "tool", "tool_call_id": "fixture-0", "content": "{}"})
    with pytest.raises(ValueError, match="incomplete tool replies"):
        getattr(integrity, validator)(row)


@pytest.mark.parametrize("reply_order", [(0, 1), (1, 0)])
def test_complete_multiple_call_replies_are_valid_dialogues_and_targets(reply_order):
    row = tool_trace(2)
    row["messages"].extend(
        {"role": "tool", "tool_call_id": f"fixture-{number}", "content": "{}"}
        for number in reply_order
    )
    integrity.validate_sft_row(row)
    integrity.validate_sft_target_row(row)
    row["messages"].append({"role": "assistant", "content": "Unit answer"})
    integrity.validate_sft_row(row)


@pytest.mark.parametrize("validator", ["validate_sft_row", "validate_sft_target_row"])
def test_final_target_exception_never_hides_interrupted_or_unmatched_replies(validator):
    row = tool_trace()
    row["messages"].append({"role": "assistant", "content": "Not a tool reply"})
    with pytest.raises(ValueError, match="missing tool replies"):
        getattr(integrity, validator)(row)
    row["messages"][-1] = {"role": "tool", "tool_call_id": "unknown", "content": "{}"}
    with pytest.raises(ValueError, match="unmatched tool response"):
        getattr(integrity, validator)(row)


def test_sft_multimodal_union_accepts_bundled_input_and_text_target_parts():
    row = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Unit fixture image question"},
        {"type": "image_url", "image_url": {"url": "https://example.org/unit-fixture.png"}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Unit fixture text target"}]}]}
    integrity.validate_sft_row(row)
    row["messages"][1]["content"] = row["messages"][0]["content"]
    with pytest.raises(ValueError, match="assistant images"):
        integrity.validate_sft_row(row)


@pytest.mark.parametrize("relative", [
    "Supervised_Fine_Tuning/Multimodal-chartqa/training.jsonl",
    "Supervised_Fine_Tuning/Multimodal-chartqa/validation.jsonl",
    "Supervised_Fine_Tuning/Tool-Calling/stock_toolcall_training.jsonl",
    "Supervised_Fine_Tuning/Tool-Calling/stock_toolcall_validation.jsonl",
])
def test_bundled_multimodal_and_tool_messages_satisfy_documented_local_union(relative):
    count = 0
    with (SAMPLES / relative).open(encoding="utf-8") as stream:
        for line in stream:
            row = integrity.loads(line)
            if relative.startswith("Supervised_Fine_Tuning/Tool-Calling/"):
                # These documented answers predict a call, not its unknown result.
                integrity.validate_sft_target_row(row)
            else:
                integrity.validate_sft_row(row)
            count += 1
    assert count > 0


def test_original_gsm8k_mit_attribution_is_retained_separately_from_modifications():
    notice = (SAMPLES / "Supervised_Fine_Tuning" / "Text-GSM8K" / "LICENSE.md").read_text(encoding="utf-8")
    assert "Copyright (c) 2021 OpenAI" in notice
    assert "Microsoft Corporation. Modifications" in notice
    assert "Permission is hereby granted, free of charge" in notice
    assert "The above copyright notice and this permission notice shall be included" in notice


def test_status_metadata_distinguishes_verified_preparation_from_shipped_artifacts():
    status = integrity.loads((SAMPLES / "validation-status.json").read_text(encoding="utf-8"))
    assert status["medmcq"]["prepared_artifacts"] is None
    assert status["nl_to_python"]["prepared_artifacts"] is None
    assert status["medmcq"]["recomputed_missing_label_count"] == 6150
    assert status["nl_to_python"]["recomputed_ast_invalid_rows"] == {"train": 95, "validation": 5}
    assert status["nl_to_python"]["verified_curated_rows"] == {"train": 1481, "validation": 78}