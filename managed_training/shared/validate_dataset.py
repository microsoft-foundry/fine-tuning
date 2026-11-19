"""Command-line validation for separately versioned managed datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
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


def _preflight_paths(
    sources: dict[str, Path],
    *,
    canonical_output_dir: Path | None = None,
    manifest_output: Path | None = None,
) -> dict[str, Path]:
    output_paths = (
        {name: canonical_output_dir / f"{name}.jsonl" for name in sources}
        if canonical_output_dir is not None
        else {}
    )
    paths = {f"{name} source": path for name, path in sources.items()}
    paths.update({f"{name} canonical output": path for name, path in output_paths.items()})
    if manifest_output is not None:
        paths["manifest output"] = manifest_output
    resolved = {name: path.resolve() for name, path in paths.items()}
    for (left_name, left), (right_name, right) in combinations(resolved.items(), 2):
        if (
            left == right
            or left in right.parents
            or right in left.parents
            or (left.exists() and right.exists() and left.samefile(right))
        ):
            raise ValueError(
                f"Dataset paths must not overwrite or alias one another "
                f"({left_name}, {right_name}): {paths[left_name]} / {paths[right_name]}"
            )
    return output_paths


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
    output_dir = Path(canonical_output_dir) if canonical_output_dir is not None else None
    output_paths = _preflight_paths(selected, canonical_output_dir=output_dir)
    reports = {
        name: validate_jsonl(path).require_valid()
        for name, path in selected.items()
    }
    isolation = validate_split_isolation(selected)
    canonical_evidence = {
        name: _canonical_evidence(path)
        for name, path in selected.items()
    }
    canonical_paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, path in selected.items():
            output_path = output_paths[name]
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
        description="Validate managed training train/validation JSONL offline."
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

    # Include the manifest in the same preflight, before build_manifest can write
    # either canonical copy or create its output directory.
    _preflight_paths(
        {"train": args.train, "validation": args.validation},
        canonical_output_dir=args.canonical_output_dir,
        manifest_output=args.output,
    )
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
