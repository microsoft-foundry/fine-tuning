"""Command-line validation for separately versioned managed datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .dataset_validation import validate_jsonl, validate_split_isolation


def canonical_upload_bytes(path: str | Path) -> bytes:
    """Return the LF-normalized bytes used by managed JSONL uploads."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def _canonical_evidence(path: Path) -> dict[str, object]:
    content = canonical_upload_bytes(path)
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_manifest(
    train: str | Path,
    validation: str | Path,
    *,
    canonical_output_dir: str | Path | None = None,
) -> dict[str, object]:
    selected = {
        "train": Path(train),
        "validation": Path(validation),
    }
    reports = {
        name: validate_jsonl(path).require_valid()
        for name, path in selected.items()
    }
    isolation = validate_split_isolation(selected)
    canonical_evidence = {
        name: _canonical_evidence(path)
        for name, path in selected.items()
    }
    output_dir = Path(canonical_output_dir) if canonical_output_dir else None
    canonical_paths: dict[str, str] = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, path in selected.items():
            output_path = output_dir / f"{name}.jsonl"
            if output_path.resolve() == path.resolve():
                raise ValueError(
                    f"Canonical output must not overwrite its source file: {path}"
                )
            output_path.write_bytes(canonical_upload_bytes(path))
            canonical_paths[name] = str(output_path)

    return {
        "schema_version": "1.0",
        "splits": {
            name: {
                "path": report.path,
                "rows": report.rows,
                "unique_rows": report.unique_rows,
                "duplicate_rows": report.duplicate_rows,
                "source_bytes": report.bytes,
                "source_sha256": report.sha256,
                "upload_bytes": canonical_evidence[name]["bytes"],
                "upload_sha256": canonical_evidence[name]["sha256"],
                "upload_path": canonical_paths.get(name),
                "roles": report.roles,
                "warnings": [
                    issue.message
                    for issue in report.issues
                    if issue.severity == "warning"
                ],
            }
            for name, report in reports.items()
        },
        "isolation": isolation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate managed fine-tuning train/validation JSONL offline."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument(
        "--canonical-output-dir",
        type=Path,
        help="Write validated LF-normalized train.jsonl and validation.jsonl copies.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        args.train,
        args.validation,
        canonical_output_dir=args.canonical_output_dir,
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
