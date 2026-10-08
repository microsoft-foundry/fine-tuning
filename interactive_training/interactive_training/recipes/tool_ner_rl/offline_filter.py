from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interactive_training.recipes.tool_ner_rl.private_files import private_text_writer


@dataclass(frozen=True)
class OfflineFilterConfig:
    f1_threshold: float = 0.90
    min_clean_replicates: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.f1_threshold <= 1.0:
            raise ValueError("f1_threshold must be between 0 and 1")
        if self.min_clean_replicates < 2:
            raise ValueError("min_clean_replicates must be at least 2")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            uid = str(record.get("uid", ""))
            if not uid:
                raise ValueError(f"Missing UID in {path}:{line_number}")
            if uid in records:
                raise ValueError(f"Duplicate UID {uid!r} in {path}")
            records[uid] = record
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _gold_fingerprint(record: dict[str, Any]) -> str:
    payload = json.dumps(record.get("gold", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_offline_filter_manifest(
    artifact_paths: list[Path],
    config: OfflineFilterConfig = OfflineFilterConfig(),
) -> dict[str, Any]:
    if len(artifact_paths) < config.min_clean_replicates:
        raise ValueError(
            f"Need at least {config.min_clean_replicates} base-eval artifacts; "
            f"got {len(artifact_paths)}"
        )

    replicate_records = [_read_records(path) for path in artifact_paths]
    expected_uids = set(replicate_records[0])
    for path, records in zip(artifact_paths[1:], replicate_records[1:], strict=True):
        if set(records) != expected_uids:
            missing = sorted(expected_uids - set(records))[:5]
            extra = sorted(set(records) - expected_uids)[:5]
            raise ValueError(
                f"Replicate UID mismatch for {path}: missing={missing}, extra={extra}"
            )

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for uid in sorted(expected_uids):
        rows = [records[uid] for records in replicate_records]
        gold_fingerprints = {_gold_fingerprint(row) for row in rows}
        if len(gold_fingerprints) != 1:
            raise ValueError(f"Gold annotations differ across replicates for UID {uid}")

        replicate_scores = [float(row["metrics"]["f1"]) for row in rows]
        clean = [
            bool(row.get("finalized"))
            and not bool(row.get("parse_error"))
            and not bool(row.get("hit_token_cap"))
            and not bool(row.get("tool_errors"))
            for row in rows
        ]
        consistently_mastered = (
            sum(clean) >= config.min_clean_replicates
            and all(clean)
            and all(score > config.f1_threshold for score in replicate_scores)
        )
        entry = {
            "uid": uid,
            "gold_entities": len(rows[0].get("gold", [])),
            "replicate_f1": replicate_scores,
            "mean_f1": sum(replicate_scores) / len(replicate_scores),
            "min_f1": min(replicate_scores),
            "max_f1": max(replicate_scores),
            "clean_replicates": sum(clean),
            "reason": "consistently_mastered" if consistently_mastered else "eligible",
        }
        (excluded if consistently_mastered else eligible).append(entry)

    return {
        "schema_version": 1,
        "selection_rule": (
            "exclude only when every replicate is clean and document F1 is strictly "
            "above f1_threshold in every replicate"
        ),
        "config": {
            "f1_threshold": config.f1_threshold,
            "min_clean_replicates": config.min_clean_replicates,
        },
        "sources": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "records": len(records),
            }
            for path, records in zip(artifact_paths, replicate_records, strict=True)
        ],
        "summary": {
            "total_documents": len(expected_uids),
            "eligible_documents": len(eligible),
            "excluded_documents": len(excluded),
            "eligible_gold_entities": sum(item["gold_entities"] for item in eligible),
            "excluded_gold_entities": sum(item["gold_entities"] for item in excluded),
        },
        "eligible": eligible,
        "excluded": excluded,
    }


def eligible_uids(manifest: dict[str, Any]) -> set[str]:
    return {str(item["uid"]) for item in manifest.get("eligible", [])}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with private_text_writer(path) as file:
        json.dump(manifest, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative NER offline difficulty-filter manifest."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--f1-threshold", type=float, default=0.90)
    parser.add_argument("--min-clean-replicates", type=int, default=2)
    args = parser.parse_args()
    manifest = build_offline_filter_manifest(
        args.artifacts,
        OfflineFilterConfig(
            f1_threshold=args.f1_threshold,
            min_clean_replicates=args.min_clean_replicates,
        ),
    )
    write_manifest(args.output, manifest)
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()