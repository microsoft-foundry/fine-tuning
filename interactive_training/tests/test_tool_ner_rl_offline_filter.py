import json
import os
import stat

import pytest

from interactive_training.recipes.tool_ner_rl.offline_filter import (
    OfflineFilterConfig,
    build_offline_filter_manifest,
    eligible_uids,
    write_manifest,
)


def _record(uid, f1, *, clean=True, gold_count=2):
    return {
        "uid": uid,
        "gold": [
            {"start": index, "end": index + 1, "label": "AGE", "text": str(index)}
            for index in range(gold_count)
        ],
        "metrics": {"f1": f1},
        "finalized": clean,
        "parse_error": not clean,
        "hit_token_cap": False,
        "tool_errors": [],
    }


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_filter_excludes_only_consistently_mastered_clean_documents(tmp_path):
    first = _write_jsonl(
        tmp_path / "first.jsonl",
        [
            _record("mastered", 1.0),
            _record("variable", 1.0),
            _record("hard", 0.2),
            _record("protocol", 1.0, clean=False),
        ],
    )
    second = _write_jsonl(
        tmp_path / "second.jsonl",
        [
            _record("mastered", 0.95),
            _record("variable", 0.7),
            _record("hard", 0.0),
            _record("protocol", 1.0),
        ],
    )

    manifest = build_offline_filter_manifest(
        [first, second],
        OfflineFilterConfig(f1_threshold=0.9),
    )

    assert manifest["summary"] == {
        "total_documents": 4,
        "eligible_documents": 3,
        "excluded_documents": 1,
        "eligible_gold_entities": 6,
        "excluded_gold_entities": 2,
    }
    assert eligible_uids(manifest) == {"variable", "hard", "protocol"}
    assert [item["uid"] for item in manifest["excluded"]] == ["mastered"]


def test_filter_requires_matching_replicate_uids_and_gold(tmp_path):
    first = _write_jsonl(tmp_path / "first.jsonl", [_record("one", 1.0)])
    different_uid = _write_jsonl(tmp_path / "uid.jsonl", [_record("two", 1.0)])
    different_gold = _write_jsonl(
        tmp_path / "gold.jsonl",
        [_record("one", 1.0, gold_count=1)],
    )

    with pytest.raises(ValueError, match="UID mismatch"):
        build_offline_filter_manifest([first, different_uid])
    with pytest.raises(ValueError, match="Gold annotations differ"):
        build_offline_filter_manifest([first, different_gold])


def test_filter_requires_two_replicates_and_valid_threshold(tmp_path):
    first = _write_jsonl(tmp_path / "first.jsonl", [_record("one", 1.0)])
    with pytest.raises(ValueError, match="at least 2"):
        build_offline_filter_manifest([first])
    with pytest.raises(ValueError, match="between 0 and 1"):
        OfflineFilterConfig(f1_threshold=1.1)


def test_write_manifest_round_trips_with_posix_permissions(tmp_path):
    output = tmp_path / "manifest.json"
    write_manifest(output, {"eligible": [{"uid": "one"}]})

    assert eligible_uids(json.loads(output.read_text())) == {"one"}
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600