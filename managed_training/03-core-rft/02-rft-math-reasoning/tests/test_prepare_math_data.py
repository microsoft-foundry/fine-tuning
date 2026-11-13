"""Independent row-1 mathematics and immutable-source/runtime-data regressions."""

from __future__ import annotations

import hashlib
import json
import shutil
from math import gcd, isqrt
from pathlib import Path

import pytest

from prepare_math_data import PREFIX, SOURCE_ROW1_PROBLEM, prepare_math_data
from rft_common import canonical_jsonl_bytes, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data/preserved/openr1-math-training-rft-100.jsonl"
VALIDATION = ROOT / "data/preserved/openr1-math-validation-rft-10.jsonl"


def minimum_integral_disguisable_triangles():
    # Independent integer constraints, not a reference fed to the text grader.
    # Every smaller-perimeter candidate is included in this finite search.
    return [
        (a, b, c)
        for a in range(1, 19)
        for b in range(a + 1, 19)
        for c in range(b + 1, 19)
        if a + b + c <= 19 and a + b > c and b * b == a * c
    ]


def test_full_source_row1_is_two_part_and_nine_is_incomplete():
    row = load_jsonl(TRAINING)[0]
    prompt = row["messages"][0]["content"]
    assert prompt == PREFIX + SOURCE_ROW1_PROBLEM
    assert "(a) Find the side lengths" in prompt
    assert "(b) Let K be an arbitrary integral disguisable triangle" in prompt
    assert "Prove that at least two side lengths" in prompt
    assert row["answer"] == "9"  # Preserve the defective historical label, not endorse it.
    derived = minimum_integral_disguisable_triangles()
    assert derived == [(4, 6, 9)]
    assert sum(derived[0]) == 19
    assert row["answer"] != str(derived[0])


def test_primitive_square_side_classification_independently_of_grader_or_labels():
    # The general divisibility/Bezout proof is in data/README.md. Here every
    # integral GP triangle with sides <=100 exercises its reduced n/m ratio.
    primitive = []
    for a in range(1, 101):
        for b in range(a + 1, 101):
            for c in range(b + 1, min(a + b, 101)):
                if b * b != a * c:
                    continue
                divisor = gcd(a, b)
                m, n = a // divisor, b // divisor
                assert gcd(m, n) == 1 and m < n
                assert a % (m * m) == 0
                k = a // (m * m)
                assert (a, b, c) == (k * m * m, k * m * n, k * n * n)
                assert gcd(gcd(a, b), c) == k
                if k == 1:
                    primitive.append((a, b, c))
                    assert isqrt(a) ** 2 == a and isqrt(c) ** 2 == c
    assert (4, 6, 9) in primitive
    assert primitive


def test_runtime_preparation_changes_only_declared_row_and_retains_source(tmp_path, capsys):
    source_bytes = {path: path.read_bytes() for path in (TRAINING, VALIDATION)}
    manifest_bytes = (ROOT / "data/hashes.json").read_bytes()
    prepared = prepare_math_data(ROOT, tmp_path / "prepared")
    training_path, validation_path = prepared["training_path"], prepared["validation_path"]
    original_lines = canonical_jsonl_bytes(source_bytes[TRAINING]).splitlines(keepends=True)
    runtime_lines = training_path.read_bytes().splitlines(keepends=True)
    assert len(runtime_lines) == len(original_lines) == 100
    assert runtime_lines[1:] == original_lines[1:]
    assert runtime_lines[0] != original_lines[0]
    row = load_jsonl(training_path)[0]
    assert row["answer"] == "(4, 6, 9)"
    # Mathematics is computed independently; neither side is sent to the grader.
    assert tuple(json.loads(row["answer"].replace("(", "[").replace(")", "]"))) == (
        minimum_integral_disguisable_triangles()[0]
    )
    assert "Cookbook runtime adaptation (answer-only subtask" in row["messages"][0]["content"]
    assert "proof subpart (b) is not part" in row["messages"][0]["content"]
    assert validation_path.read_bytes() == canonical_jsonl_bytes(source_bytes[VALIDATION])
    for path, original in source_bytes.items():
        assert path.read_bytes() == original
    assert (ROOT / "data/hashes.json").read_bytes() == manifest_bytes
    report = json.loads(prepared["report_path"].read_text(encoding="utf-8"))
    assert report == prepared["report"]
    assert report["curation"][0]["proof_subpart"] == "excluded_from_graded_task"
    assert report["split_overlap"] == 0
    for kind, path in (("training", training_path), ("validation", validation_path)):
        content = path.read_bytes()
        assert b"\r\n" not in content
        assert report["runtime_artifacts"][kind] == {
            "path": path.name, "bytes": len(content), "rows": 100 if kind == "training" else 10,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    assert "proof subpart (b) is not graded" in capsys.readouterr().out


def test_preparation_is_deterministic_and_does_not_rewrite_existing_outputs(tmp_path):
    first = prepare_math_data(ROOT, tmp_path)
    paths = [first[key] for key in ("training_path", "validation_path", "report_path")]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    second = prepare_math_data(ROOT, tmp_path)
    assert first == second
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


def test_mismatched_prepared_input_is_not_overwritten(tmp_path):
    prepared = prepare_math_data(ROOT, tmp_path)
    path = prepared["training_path"]
    changed = path.read_bytes() + b"\n"
    path.write_bytes(changed)
    with pytest.raises(RuntimeError, match="Retain it and reconcile"):
        prepare_math_data(ROOT, tmp_path)
    assert path.read_bytes() == changed


def test_preparation_cannot_target_preserved_data():
    with pytest.raises(ValueError, match="must not write under data/preserved"):
        prepare_math_data(ROOT, ROOT / "data/preserved/forbidden")


def test_lf_and_crlf_sources_prepare_identical_content_without_rewriting_provenance(tmp_path):
    copy_root = tmp_path / "source-copy"
    (copy_root / "data").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/hashes.json", copy_root / "data/hashes.json")
    (copy_root / "data/preserved").mkdir()
    copies = {}
    for path in (TRAINING, VALIDATION):
        copy_path = copy_root / "data/preserved" / path.name
        copy_path.write_bytes(canonical_jsonl_bytes(path.read_bytes()).replace(b"\n", b"\r\n"))
        copies[copy_path] = copy_path.read_bytes()
    canonical = prepare_math_data(ROOT, tmp_path / "lf-output")
    crlf = prepare_math_data(copy_root, tmp_path / "crlf-output")
    for key in ("training_path", "validation_path", "report_path"):
        assert canonical[key].read_bytes() == crlf[key].read_bytes()
    assert all(path.read_bytes() == content for path, content in copies.items())