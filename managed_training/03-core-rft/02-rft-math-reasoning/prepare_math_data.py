"""Explicit offline runtime curation; never rewrite preserved source artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rft_common import canonical_jsonl_bytes, load_jsonl, validate_hash_manifest


POLICY = "openr1-row1-answer-only-v1"
PREFIX = (
    "You are a mathematical reasoning expert. Solve problems with detailed step-by-step "
    "thinking and provide final answers in \\boxed{} format. Show all intermediate "
    "calculations and explain your reasoning clearly.\n\n"
)
SOURCE_ROW1_PROBLEM = (
    "Call a scalene triangle K [i]disguisable[/i] if there exists a triangle K′ similar "
    "to K with two shorter sides precisely as long as the two longer sides of K, "
    "respectively. Call a disguisable triangle [i]integral[/i] if the lengths of all "
    "its sides are integers.\r\n"
    "(a) Find the side lengths of the integral disguisable triangle with the smallest "
    "possible perimeter.\r\n"
    "(b) Let K be an arbitrary integral disguisable triangle for which no smaller integral\r\n"
    "disguisable triangle similar to it exists. Prove that at least two side lengths of K are\r\n"
    "perfect squares."
)
RUNTIME_ROW1_PROBLEM = (
    "Cookbook runtime adaptation (answer-only subtask of preserved training row 1). "
    "Call a scalene triangle K disguisable if there exists a triangle K′ similar to K "
    "whose two shorter sides equal the two longer sides of K, respectively. An integral "
    "disguisable triangle has integer side lengths. Find all three side lengths of the "
    "integral disguisable triangle with the smallest possible perimeter. Put the three "
    "side lengths in a single boxed tuple in increasing order. The source's separate "
    "proof subpart (b) is not part of this answer-only graded task."
)


def _retain_or_create(path: Path, content: bytes) -> None:
    """Do not silently replace a different prepared input used by a pending job."""
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(
                "Prepared math data differs from the requested curation. Retain it and "
                "reconcile submission state before choosing a separate output directory."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def prepare_math_data(demo_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Prepare a labeled 100/10 runtime split from hash-verified preserved inputs.

    Only source training row 1 is adapted: part (a)'s full tuple replaces the
    incomplete label 9; the proof task is explicitly excluded from the prompt.
    The proof/minimality derivation is documented in data/README.md and tested
    independently of the stored answer and text grader.
    """
    root = Path(demo_root).resolve()
    destination = Path(output_dir).resolve()
    preserved = (root / "data" / "preserved").resolve()
    if destination == preserved or preserved in destination.parents:
        raise ValueError("Runtime preparation must not write under data/preserved")
    validate_hash_manifest(root / "data" / "hashes.json", root)
    sources = {
        "training": preserved / "openr1-math-training-rft-100.jsonl",
        "validation": preserved / "openr1-math-validation-rft-10.jsonl",
    }
    rows = {kind: load_jsonl(path) for kind, path in sources.items()}
    if (len(rows["training"]), len(rows["validation"])) != (100, 10):
        raise ValueError("Expected the preserved 100/10 split")
    for row in rows["training"] + rows["validation"]:
        if (
            set(row) != {"messages", "answer"}
            or not isinstance(row["answer"], str) or not row["answer"]
            or len(row["messages"]) != 1
            or row["messages"][0].get("role") != "user"
            or not isinstance(row["messages"][0].get("content"), str)
        ):
            raise ValueError("Expected one user prompt and a nonempty reference answer")
    if rows["training"][0] != {
        "messages": [{"content": PREFIX + SOURCE_ROW1_PROBLEM, "role": "user"}],
        "answer": "9",
    }:
        raise ValueError("Preserved row 1 changed; independently review before adapting it")
    runtime_row = {
        "messages": [{"content": PREFIX + RUNTIME_ROW1_PROBLEM, "role": "user"}],
        "answer": "(4, 6, 9)",
    }
    training_prompts = {
        row["messages"][0]["content"] for row in [runtime_row, *rows["training"][1:]]
    }
    validation_prompts = {row["messages"][0]["content"] for row in rows["validation"]}
    if not training_prompts.isdisjoint(validation_prompts):
        raise ValueError("Training/validation leakage detected")
    canonical_sources = {
        kind: canonical_jsonl_bytes(path.read_bytes()) for kind, path in sources.items()
    }
    training_lines = canonical_sources["training"].splitlines(keepends=True)
    if len(training_lines) != 100:
        raise ValueError("Expected exactly 100 physical source training lines")
    # Retain every other source line exactly, including earlier answer/prompt curation.
    training_lines[0] = json.dumps(
        runtime_row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8") + b"\n"
    prepared = {
        "training": b"".join(training_lines),
        "validation": canonical_sources["validation"],
    }
    paths = {
        "training": destination / "openr1-math-training-answer-only-100.jsonl",
        "validation": destination / "openr1-math-validation-10.jsonl",
    }

    def evidence(content: bytes, count: int) -> dict[str, Any]:
        return {"bytes": len(content), "rows": count,
                "sha256": hashlib.sha256(content).hexdigest()}

    report = {
        "schema_version": 1,
        "policy": POLICY,
        "normalization": "CRLF-to-LF",
        "source_artifacts": {
            kind: {"path": path.relative_to(root).as_posix(),
                   **evidence(canonical_sources[kind], len(rows[kind]))}
            for kind, path in sources.items()
        },
        "runtime_artifacts": {
            kind: {"path": paths[kind].name, **evidence(content, len(rows[kind]))}
            for kind, content in prepared.items()
        },
        "curation": [{
            "split": "training", "source_row": 1, "change": "answer_only_subtask",
            "source_answer": "9", "runtime_answer": runtime_row["answer"],
            "proof_subpart": "excluded_from_graded_task",
            "derivation": "data/README.md#runtime-row-1-curation-and-independent-proof",
        }],
        "split_overlap": 0,
    }
    for kind, content in prepared.items():
        _retain_or_create(paths[kind], content)
    report_path = destination / "preparation-report.json"
    _retain_or_create(report_path, (json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8"))
    print(
        "Runtime curation: preserved training row 1 is now an answer-only subtask "
        "with all three sides; proof subpart (b) is not graded. Preserved source "
        "bytes/hashes and the 100/10 split are unchanged."
    )
    return {"training_path": paths["training"], "validation_path": paths["validation"],
            "report_path": report_path, "report": report}