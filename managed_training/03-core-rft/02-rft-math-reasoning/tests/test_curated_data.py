from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction
from math import dist, sqrt
from pathlib import Path

import pytest

from rft_common import canonical_jsonl_bytes, load_jsonl, validate_hash_manifest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data/preserved/openr1-math-training-rft-100.jsonl"
VALIDATION = ROOT / "data/preserved/openr1-math-validation-rft-10.jsonl"
PREFIX = (
    "You are a mathematical reasoning expert. Solve problems with detailed step-by-step "
    "thinking and provide final answers in \\boxed{} format. Show all intermediate "
    "calculations and explain your reasoning clearly.\n\n"
)


@pytest.fixture
def grader():
    path = ROOT / "graders/boxed_math_grader.py"
    spec = importlib.util.spec_from_file_location("curated_math_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_110_references_still_pass_the_unmodified_boxed_grader(grader):
    training = load_jsonl(TRAINING)
    validation = load_jsonl(VALIDATION)
    assert (len(training), len(validation)) == (100, 10)
    for row in training + validation:
        assert set(row) == {"messages", "answer"}
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
        assert grader.grade({"output_text": r"\boxed{" + row["answer"] + "}"}, row) == 1.0
    assert {row["messages"][0]["content"] for row in training}.isdisjoint(
        row["messages"][0]["content"] for row in validation
    )
    validate_hash_manifest(ROOT / "data/hashes.json", ROOT)
    validate_hash_manifest(ROOT / "graders/hashes.json", ROOT / "graders")


def test_only_four_reviewed_rows_differ_from_the_exact_batch1_artifact():
    manifest = json.loads((ROOT / "data/hashes.json").read_text(encoding="utf-8"))
    lines = canonical_jsonl_bytes(TRAINING.read_bytes()).splitlines(keepends=True)
    for number in (34, 55, 96):
        row = json.loads(lines[number - 1])
        current = json.dumps(row["messages"][0]["content"]).encode("utf-8")
        original = json.dumps(PREFIX + manifest["curation"]["training"][str(number)]["original_problem"]).encode("utf-8")
        assert lines[number - 1].count(current) == 1
        lines[number - 1] = lines[number - 1].replace(current, original)
    assert lines[13].count(b'"answer": "(20, 30)"') == 1
    lines[13] = lines[13].replace(b'"answer": "(20, 30)"', b'"answer": "30"')
    # Reconstruct the exact pre-review canonical file: every other byte is preserved.
    assert hashlib.sha256(b"".join(lines)).hexdigest() == (
        "bc7c73c5f8a10a10d26338cba7d7b78d570092a91b9956bcf54e42d61f41cc78"
    )
    training = load_jsonl(TRAINING)
    for number, answer in {
        68: "-2, -1, 4, 5",
        74: "8, 9, 10, 11, 12, 13, 14, 15",
        76: "729",
        79: "9, 23, 37",
        100: "a=5, b=3, c=7",
    }.items():
        assert training[number - 1]["answer"] == answer
    assert hashlib.sha256(canonical_jsonl_bytes(VALIDATION.read_bytes())).hexdigest() == (
        "beabdabc9ec3bf5509f03cb9a7e93cef6f068b9c0fbc970a105910b3de97b4dd"
    )
    assert hashlib.sha256(canonical_jsonl_bytes(
        (ROOT / "graders/boxed_math_grader.py").read_bytes(),
    )).hexdigest() == "212fcd43ec56cfa43cae1e902853f7395c5d11ca2afe6d418219a67c7ae8781e"


def test_row14_requires_both_lamp_counts_in_prompt_order(grader):
    row = load_jsonl(TRAINING)[13]
    assert row["answer"] == "(20, 30)"
    solutions = [
        (small, large)
        for small in range(51)
        for large in range(51)
        if small + large == 50 and 40 * small + 75 * large == 3050
    ]
    assert solutions == [(20, 30)]
    assert grader.grade({"output_text": r"\boxed{(20,30)}"}, row) == 1.0
    assert grader.grade({"output_text": r"\boxed{30}"}, row) == 0.0
    assert grader.grade({"output_text": r"\boxed{(30,20)}"}, row) == 0.0


def test_missing_figures_are_explicit_local_adaptations_with_original_text_retained():
    manifest = json.loads((ROOT / "data/hashes.json").read_text(encoding="utf-8"))
    assert manifest["adapted_rows"] == {"training": [34, 55, 96], "validation": []}
    training = load_jsonl(TRAINING)
    labeled = {
        number for number, row in enumerate(training, start=1)
        if "Cookbook adaptation (self-contained; not a reconstruction" in row["messages"][0]["content"]
    }
    assert labeled == {34, 55, 96}
    for number in labeled:
        entry = manifest["curation"]["training"][str(number)]
        assert entry["change"] == "locally_adapted_self_contained_prompt"
        assert entry["answer"] == training[number - 1]["answer"]
        assert number in manifest["corrected_rows"]["training"]
        assert entry["verification"]
        assert "Figure" in entry["original_problem"] or "As shown" in entry["original_problem"]
    assert 14 in manifest["corrected_rows"]["training"]
    assert "not an assertion that upstream contains" in manifest["source_attribution"]


def test_row34_equilateral_geometry_is_self_contained_and_has_length_one():
    prompt = load_jsonl(TRAINING)[33]["messages"][0]["content"]
    assert "C=(0,0), A=(1,0), and B=(0,1)" in prompt
    assert "opposite side of line AB from C" in prompt
    assert "same side of line CD as B" in prompt
    c, a, b = (0, 0), (1, 0), (0, 1)
    t = (1 + sqrt(3)) / 2
    d, e = (t, t), (-0.5, 1 + sqrt(3) / 2)
    assert dist(a, b) == pytest.approx(sqrt(2))
    assert dist(a, d) == pytest.approx(dist(a, b))
    assert dist(b, d) == pytest.approx(dist(a, b))
    assert dist(c, e) == pytest.approx(dist(c, d))
    assert dist(d, e) == pytest.approx(dist(c, d))
    assert (d[0] + d[1] - 1) * (c[0] + c[1] - 1) < 0
    cross_b = d[0] * b[1] - d[1] * b[0]
    cross_e = d[0] * e[1] - d[1] * e[0]
    assert cross_b * cross_e > 0
    assert dist(b, e) == pytest.approx(1)


def test_row55_defines_the_uniform_law_and_the_new_program():
    row = load_jsonl(TRAINING)[54]
    prompt = row["messages"][0]["content"]
    assert "uniformly distributed on [1,9]" in prompt
    assert "10*X+15" in prompt
    threshold = Fraction(55 - 15, 10)
    assert threshold == 4
    assert (9 - threshold) / (9 - 1) == Fraction(5, 8)
    assert "(D) 5/8" in prompt and row["answer"] == "D"


def test_row96_defines_the_ray_and_right_angle_and_has_cf_four():
    row = load_jsonl(TRAINING)[95]
    prompt = row["messages"][0]["content"]
    assert "ray DC beyond C" in prompt and "right-angled at F" in prompt
    assert "hypotenuse DE" in prompt
    d, c, a, b = (0, 0), (10, 0), (3, 4), (7, 4)
    e = tuple(2 * b[index] - d[index] for index in (0, 1))
    f = (e[0], 0)
    assert (dist(a, d), dist(b, c), dist(a, b), dist(d, c)) == (5, 5, 4, 10)
    assert e == (14, 8) and f[0] > c[0]
    assert sum((d[index] - f[index]) * (e[index] - f[index]) for index in (0, 1)) == 0
    assert dist(c, f) == 4
    assert "(D) 4.0" in prompt and row["answer"] == "D"