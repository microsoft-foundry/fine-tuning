from __future__ import annotations

import json

import pytest

from shared.dataset_validation import (
    DatasetValidationError,
    hash_file,
    validate_jsonl,
    validate_split_isolation,
)


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(prompt: str, answer: str = "ok") -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def test_validates_and_hashes_jsonl(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    _write(path, [_row("one"), _row("two"), _row("two")])
    report = validate_jsonl(path)
    assert report.valid
    assert report.rows == 3
    assert report.duplicate_rows == 1
    assert report.sha256 == hash_file(path)


def test_detects_schema_and_secret_errors(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    _write(
        path,
        [
            {"messages": [{"role": "assistant", "content": "first"}]},
            _row("api_key=abcdefghijklmnop"),
        ],
    )
    report = validate_jsonl(path)
    assert not report.valid
    with pytest.raises(DatasetValidationError):
        report.require_valid()


def test_rejects_split_overlap(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    duplicate = _row("shared")
    _write(train, [duplicate, _row("train-only")])
    _write(validation, [duplicate, _row("validation-only")])
    with pytest.raises(DatasetValidationError, match="contamination"):
        validate_split_isolation({"train": train, "validation": validation})
