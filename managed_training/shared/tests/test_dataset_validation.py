from __future__ import annotations

import json

import pytest

from shared.dataset_validation import (
    DatasetValidationError,
    hash_file,
    validate_jsonl,
    validate_split_isolation,
)
from shared.validate_dataset import build_manifest, canonical_upload_bytes


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


def test_builds_versioned_adaptation_manifest(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write(train, [_row("train-one"), _row("train-two")])
    _write(validation, [_row("validation-one")])

    manifest = build_manifest(train, validation)

    assert manifest["schema_version"] == "1.0"
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["validation"]["rows"] == 1
    assert manifest["isolation"]["overlaps"] == []


def test_manifest_upload_evidence_normalizes_crlf(tmp_path) -> None:
    lf_train = tmp_path / "lf-train.jsonl"
    crlf_train = tmp_path / "crlf-train.jsonl"
    validation = tmp_path / "validation.jsonl"
    lf_content = "".join(
        json.dumps(row) + "\n"
        for row in (_row("train-one"), _row("train-two"))
    ).encode("utf-8")
    lf_train.write_bytes(lf_content)
    crlf_train.write_bytes(lf_content.replace(b"\n", b"\r\n"))
    _write(validation, [_row("validation-one")])

    lf_manifest = build_manifest(lf_train, validation)
    prepared = tmp_path / "prepared"
    crlf_manifest = build_manifest(
        crlf_train,
        validation,
        canonical_output_dir=prepared,
    )

    assert (
        lf_manifest["splits"]["train"]["upload_sha256"]
        == crlf_manifest["splits"]["train"]["upload_sha256"]
    )
    assert (
        lf_manifest["splits"]["train"]["upload_bytes"]
        == crlf_manifest["splits"]["train"]["upload_bytes"]
    )
    assert (
        crlf_manifest["splits"]["train"]["source_sha256"]
        != crlf_manifest["splits"]["train"]["upload_sha256"]
    )
    assert (prepared / "train.jsonl").read_bytes() == canonical_upload_bytes(crlf_train)


def test_canonical_output_cannot_overwrite_source(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write(train, [_row("train")])
    _write(validation, [_row("validation")])

    with pytest.raises(ValueError, match="must not overwrite"):
        build_manifest(train, validation, canonical_output_dir=tmp_path)
