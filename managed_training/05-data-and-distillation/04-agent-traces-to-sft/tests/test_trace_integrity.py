"""Offline trace regression tests: no model, tool execution, or notebook execution."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("agent_trace_integrity", MODULE / "trace_integrity.py")
assert SPEC is not None and SPEC.loader is not None
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def fixture_row(user="Unit conversation A", call_id="call_trace_0_1_0", answer=True):
    messages = [
        {"role": "system", "content": "Common unit-test policy"},
        {"role": "user", "content": user},
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
         "function": {"name": "fixture_lookup", "arguments": '{"sku":"unit-sku","order_id":"unit-order"}'}}]},
    ]
    if answer:
        messages.extend([{"role": "tool", "tool_call_id": call_id, "content": '{"ok":true,"count":1}'},
                         {"role": "assistant", "content": "Unit-test-only result"}])
    return {"messages": messages}


def test_generated_call_ids_and_json_spacing_do_not_hide_identical_conversations():
    left = fixture_row()
    right = fixture_row(call_id="another-generated-call-id")
    right["messages"][2]["content"] = None
    right["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{ "order_id": "unit-order", "sku": "unit-sku" }'
    right["messages"][3]["content"] = '{"count":1,"ok":true}'
    assert left != right
    assert trace.normalized_messages(left) == trace.normalized_messages(right)
    with pytest.raises(ValueError, match="leakage"):
        trace.require_disjoint({"train": [left], "val": [right]})


def test_message_prefixes_are_detected_in_either_direction_after_normalization():
    full = fixture_row()
    prefix = fixture_row(call_id="short-prefix-generated-id", answer=False)
    for splits in ({"train": [full], "val": [prefix]}, {"train": [prefix], "test": [full]}):
        report = trace.leakage_report(splits)
        assert len(report["message_prefix_overlap"]) == 1
        with pytest.raises(ValueError):
            trace.require_disjoint(splits)


def test_shared_first_user_request_groups_divergent_and_extended_snapshots():
    left = fixture_row()
    right = fixture_row(call_id="branch-id")
    right["messages"][-1]["content"] = "Different continuation of the same unit conversation"
    right["messages"].append({"role": "user", "content": "Unit follow-up"})
    assert trace.normalized_messages(left) != trace.normalized_messages(right)
    assert trace.conversation_groups([left, right]) == [[0, 1]]
    assert trace.leakage_report({"train": [left], "val": [right]})["conversation_overlap"]


def test_common_system_prompt_does_not_group_unrelated_conversations():
    first, second = fixture_row(), fixture_row(user="Independent unit conversation B")
    assert trace.conversation_groups([first, second]) == [[0], [1]]
    trace.require_disjoint({"train": [first], "val": [second]})


def test_business_identifiers_and_message_text_are_not_erased_as_generated_ids():
    left = fixture_row()
    right = copy.deepcopy(left)
    right["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"sku":"other-unit-sku","order_id":"unit-order"}'
    assert trace.normalized_messages(left) != trace.normalized_messages(right)


def test_whole_group_split_preserves_all_unit_rows_and_is_deterministic():
    rows = [fixture_row(), fixture_row(call_id="short-id", answer=False)]
    rows.extend(fixture_row(user=f"Independent unit conversation {number}") for number in range(10))
    selected, groups = trace.group_split(rows)
    assert (selected, groups) == trace.group_split(rows)
    assert sorted(index for indices in selected.values() for index in indices) == list(range(len(rows)))
    assert all(indices for indices in selected.values())
    assert any(0 in indices and 1 in indices for indices in selected.values())
    trace.require_disjoint({name: [rows[index] for index in indices] for name, indices in selected.items()})


def test_too_few_conversations_fails_instead_of_falling_back_to_row_split():
    rows = [fixture_row(), fixture_row(call_id="duplicate-id"), fixture_row(answer=False)]
    with pytest.raises(ValueError, match="three independent"):
        trace.group_split(rows)


@pytest.fixture(scope="module")
def preparation():
    return trace.build_prepared()


def test_real_preparation_has_disjoint_conversations_and_no_source_row_loss(preparation):
    payloads, manifest = preparation
    raw, originals = trace.load_sources()
    output = trace.read_jsonl(payloads["traces-clean.jsonl"])
    assert len(output) == len(originals["traces-clean.jsonl"]) == 100
    assert manifest["algorithm"] == trace.ALGORITHM
    assert manifest["status"] == "prepared_offline_source_contract_validated_not_cloud_tested"
    assert sorted(record["source_line"] for record in manifest["records"]) == list(range(1, 101))
    assert sum(manifest["artifacts"][name]["rows"] for name in trace.SPLITS) == 100
    assert all(manifest["artifacts"][name]["rows"] > 0 for name in trace.SPLITS)
    trace.require_disjoint({name: trace.read_jsonl(payloads[name]) for name in trace.SPLITS})
    for name, content in raw.items():
        assert trace.sha256(content) == trace.SOURCE_HASHES[name]
        source = (trace.SOURCE / name).read_bytes()
        assert (source.replace(b"\r\n", b"\n") if name in trace.TEXT_SOURCE_FIXTURES else source) == content
    assert any(manifest["legacy_split_leakage"].values()), "Historical row split must not be reported isolation-safe"


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"])
def test_fixture_source_guards_match_git_and_windows_text_bytes(tmp_path, eol, preparation):
    source, _ = trace.load_sources()
    for name, content in source.items():
        selected = content.replace(b"\n", eol) if name in trace.TEXT_SOURCE_FIXTURES else content
        (tmp_path / name).write_bytes(selected)
    payloads, manifest = trace.build_prepared(tmp_path)
    assert payloads == preparation[0]
    assert manifest == preparation[1]
    path = tmp_path / "zava_tools.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        trace.build_prepared(tmp_path)


def test_policy_amendment_changes_only_reviewed_descriptions_and_preserves_supervision(preparation):
    payloads, _ = preparation
    raw, originals = trace.load_sources()
    old_prompt = raw["zava_system_prompt.md"].decode("utf-8").replace("\r\n", "\n")
    old_tools = trace.loads(raw["zava_tools.json"].decode("utf-8"))
    new_prompt, new_tools = trace.revised_fixtures(old_prompt, old_tools)
    assert "exchange or replacement" in new_prompt
    assert "Check stock ONLY when processing an exchange" not in new_prompt
    inventory = next(tool for tool in new_tools if tool["function"]["name"] == "check_inventory")
    assert "exchange or replacement" in inventory["function"]["description"]
    assert "refund alone" in inventory["function"]["description"]
    for before, after in zip(old_tools, new_tools):
        assert before["function"]["name"] == after["function"]["name"]
        left, right = copy.deepcopy(before), copy.deepcopy(after)
        if left["function"]["name"] == "check_inventory":
            left["function"].pop("description")
            right["function"].pop("description")
        if left["function"]["name"] == "calculate_resolution":
            for tool in (left, right):
                tool["function"]["parameters"]["properties"]["items"]["items"]["properties"]["action"].pop("description")
        assert left == right
    for before, after in zip(originals["traces-clean.jsonl"], trace.read_jsonl(payloads["traces-clean.jsonl"])):
        unchanged = copy.deepcopy(after)
        unchanged["messages"][0] = before["messages"][0]
        unchanged["tools"] = before["tools"]
        assert unchanged == before
        trace.validate_trace(after, new_prompt, new_tools)


def test_real_replacement_inventory_supervision_is_reported_without_guessing_snapshot_actions(preparation):
    _, manifest = preparation
    usage = manifest["legacy_inventory_usage"]
    assert sum(report["replacement_rows_with_inventory"] for report in usage.values()) > 0
    for report in usage.values():
        assert report["rows_with_inventory"] == len(report["records"])
        for record in report["records"]:
            assert record["replacement_with_inventory"] == ("replacement" in record["recorded_calculated_actions"])
            assert record["action_not_yet_recorded"] == (not record["recorded_calculated_actions"])


def test_manifest_maps_each_original_line_to_unchanged_action_target_and_output_hash(preparation):
    payloads, manifest = preparation
    source_lines = (trace.SOURCE / "traces-clean.jsonl").read_bytes().splitlines(keepends=True)
    split_lines = {name: content.splitlines(keepends=True) for name, content in payloads.items() if name in trace.SPLITS}
    group_splits = {}
    for record in manifest["records"]:
        assert trace.sha256(source_lines[record["source_line"] - 1]) == record["source_raw_line_sha256"]
        actual = split_lines[record["split"]][record["output_line"] - 1]
        assert trace.sha256(actual) == record["output_raw_line_sha256"]
        group_splits.setdefault(record["group_id"], set()).add(record["split"])
    assert all(len(names) == 1 for names in group_splits.values())
    for name, details in manifest["artifacts"].items():
        assert trace.sha256(payloads[name]) == details["sha256"]


def test_published_offline_status_matches_recomputed_preparation(preparation):
    _, manifest = preparation
    status = trace.loads((MODULE / "validation-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "offline_preparation_and_tests_passed_no_cloud_execution"
    assert status["recomputed_counts"]["source_traces"] == len(manifest["records"])
    assert status["recomputed_counts"]["conversation_groups"] == len(manifest["groups"])
    assert status["recomputed_counts"]["prepared_splits"] == {
        name: manifest["artifacts"][file]["rows"]
        for name, file in zip(("train", "validation", "test"), trace.SPLITS)
    }
    assert status["verified_prepared_hashes"] == {
        name: manifest["artifacts"][name]["sha256"] for name in trace.SPLITS
    }


def test_prepared_contract_is_independently_rebuilt_and_rejects_drift(tmp_path):
    directory = tmp_path / "prepared"
    manifest = trace.prepare(directory)
    datasets, lineage = trace.validate_prepared(directory)
    assert lineage["train.jsonl"] == manifest["artifacts"]["train.jsonl"]
    assert len(datasets["traces-clean.jsonl"]) == 100
    with pytest.raises(ValueError, match="exists"):
        trace.prepare(directory)
    path = directory / "val.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="independently rebuilt"):
        trace.validate_prepared(directory)


def test_missing_prepared_data_fails_closed_before_notebook_can_upload(tmp_path):
    with pytest.raises(ValueError, match="legacy row splits are not accepted"):
        trace.validate_prepared(tmp_path / "absent")


def notebook_source(cell):
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    # Editor sources may contain line fragments or already newline-terminated lines.
    return "".join(source) if any(line.endswith(("\n", "\r")) for line in source) else "\n".join(source)


def test_notebook_syntax_and_source_guards_without_executing_any_cell():
    notebook = json.loads((MODULE / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))
    cells = [notebook_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in cells:
        ast.parse(source)
    joined = "\n".join(cells)
    assert 'DATA = ROOT / "outputs" / "prepared"' in joined
    assert "datasets, lineage = validate_prepared(DATA)" in joined
    assert "current_datasets, current_lineage = validate_prepared(DATA)" in joined
    assert joined.index("current_datasets, current_lineage = validate_prepared(DATA)") < joined.index("with create_project_context(config)")
    assert "Random(42).shuffle" not in joined
    assert "validation = upload_once(client, state, VALIDATION_FILE)" in joined
    assert "upload_once(client, state, TEST_FILE)" not in joined
    assert "trace.prepare(" not in joined and "build_prepared(" not in joined