"""Offline trace preparation: preserve supervision, correct policy, split whole conversations.

The original data directory is immutable input. Preparation creates a new explicit
directory. The notebook accepts only independently rebuilt, validated output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


MODULE = Path(__file__).resolve().parent
SOURCE = MODULE / "data"
ALGORITHM = "normalized-conversation-groups-v1"
SEED = 42
TEXT_SOURCE_FIXTURES = frozenset({"zava_system_prompt.md", "zava_tools.json"})
SOURCE_HASHES = {
    "zava_system_prompt.md": "4fc14242b344839f6673d44c429b9bafb2a19536028d61b93aa83f413a024a10",
    "zava_tools.json": "2493bb2265e1d59e3e9023204a87f10fda916507dbc6b6cb453f54bb671f347c",
    "traces-clean.jsonl": "81083aadd6464293b6acfc9356216c6c685a1ba99ae3f6fe34ba9d7b5b864d71",
    "train.jsonl": "1371b81c70d7f8c20f8eff604932670a3eda8ca9437759264dcfce4c06aeeae3",
    "val.jsonl": "865ab59e51c81a29ecc89a0bde347a22d0929630f7cd3db1ffb12f1d406034fa",
    "test.jsonl": "7ea4284e67980cc2db8cfb2a4b5d403da4383bec9821be3fea27316cbd677040",
}
SOURCE_ROWS = {"traces-clean.jsonl": 100, "train.jsonl": 80, "val.jsonl": 10, "test.jsonl": 10}
SPLITS = ("train.jsonl", "val.jsonl", "test.jsonl")
POLICY_REPLACEMENTS = {
    "Check stock ONLY when processing an exchange":
        "Check stock when processing an exchange or replacement; not for a refund alone",
    "If an exchange is requested, call `check_inventory` for the desired SKU.":
        "If an exchange or replacement is requested, call `check_inventory` for the desired SKU before calculating that action.",
}
OLD_INVENTORY_DESCRIPTION = "Check stock for a SKU. Call ONLY when processing an exchange."
NEW_INVENTORY_DESCRIPTION = (
    "Check stock for the desired SKU when processing an exchange or replacement, "
    "including a lost-package replacement. Not needed for a refund alone."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def read_jsonl(raw: bytes) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line {number}")
        row = loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"Non-object JSONL line {number}")
        rows.append(row)
    if not rows:
        raise ValueError("Empty JSONL source")
    return rows


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(canonical(row) + "\n" for row in rows)).encode("utf-8")


def revised_fixtures(prompt: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Reviewed policy edit only; tools' names, parameters and trace actions stay intact."""
    revised_prompt = prompt
    for before, after in POLICY_REPLACEMENTS.items():
        if revised_prompt.count(before) != 1:
            raise ValueError("Original system policy differs from the reviewed source")
        revised_prompt = revised_prompt.replace(before, after)
    revised_tools = copy.deepcopy(tools)
    inventory = [tool["function"] for tool in revised_tools if tool.get("function", {}).get("name") == "check_inventory"]
    calculators = [tool["function"] for tool in revised_tools if tool.get("function", {}).get("name") == "calculate_resolution"]
    if len(inventory) != 1 or inventory[0].get("description") != OLD_INVENTORY_DESCRIPTION or len(calculators) != 1:
        raise ValueError("Original inventory/calculation tool contract differs from reviewed source")
    inventory[0]["description"] = NEW_INVENTORY_DESCRIPTION
    action = calculators[0]["parameters"]["properties"]["items"]["items"]["properties"]["action"]
    if action.get("description") != "e.g. 'refund', 'exchange', 'shipping_credit', 'deny'":
        raise ValueError("Original action description differs from the reviewed source")
    action["description"] = "e.g. 'refund', 'exchange', 'replacement', 'shipping_credit', 'deny'"
    return revised_prompt, revised_tools


def validate_trace(row: dict[str, Any], prompt: str, tools: list[dict[str, Any]]) -> int:
    messages = row.get("messages")
    if (not isinstance(messages, list) or not messages
            or messages[0] != {"role": "system", "content": prompt}
            or row.get("tools") != tools or row.get("parallel_tool_calls") is not True):
        raise ValueError("Trace must match the exact system/tools fixtures and parallel_tool_calls=true")
    catalog = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    pending: set[str] = set()
    seen: set[str] = set()
    tool_turns = 0
    has_user = False
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Trace messages must be objects")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported trace message role")
        has_user = has_user or role == "user"
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list) or ("tool_calls" in message and (role != "assistant" or not calls)):
            raise ValueError("tool_calls must be a nonempty assistant array")
        if calls:
            if pending or (message.get("content") is not None and not isinstance(message["content"], str)):
                raise ValueError("Pending replies or invalid assistant tool-call content")
            tool_turns += 1
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("Malformed function call")
                call_id = call.get("id")
                function = call.get("function")
                if (not isinstance(call_id, str) or not call_id or call_id in seen
                        or not isinstance(function, dict) or function.get("name") not in catalog
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError("Missing/duplicate call ID, unknown tool or malformed arguments")
                arguments = loads(function["arguments"])
                parameters = catalog[function["name"]]
                if (not isinstance(arguments, dict) or not set(parameters.get("required", [])).issubset(arguments)
                        or not set(arguments).issubset(parameters.get("properties", {}))):
                    raise ValueError("Arguments do not match the declared tool parameter names")
                pending.add(call_id)
                seen.add(call_id)
        else:
            if not isinstance(message.get("content"), str):
                raise ValueError("Non-call trace message content must be text")
            if role == "tool":
                if message.get("tool_call_id") not in pending:
                    raise ValueError("Unmatched tool reply")
                pending.remove(message["tool_call_id"])
            elif pending:
                raise ValueError("Missing replies before next conversation turn")
        if pending and position == len(messages) - 1 and not calls:
            raise ValueError("Incomplete tool reply set")
    if not has_user or not tool_turns:
        raise ValueError("Trace requires a user conversation and assistant tool-call supervision")
    return tool_turns


def normalized_messages(row: dict[str, Any]) -> tuple[str, ...]:
    """Ignore shared system policy; alpha-rename all call IDs by first call occurrence.

    Preserve ordinary content and semantic identifiers (orders/SKUs). Canonicalize
    JSON argument/object ordering, not free text or embedded business identifiers.
    Null and omitted assistant tool-call content represent the same empty content.
    """
    identifiers: dict[str, str] = {}
    result = []
    for original in row["messages"]:
        if original["role"] == "system":
            continue
        message = copy.deepcopy(original)
        calls = message.get("tool_calls", [])
        if calls:
            if message.get("content") is None:
                message.pop("content", None)
            for call in calls:
                old_id = call["id"]
                if old_id in identifiers:
                    raise ValueError("Duplicate function call ID during normalization")
                identifiers[old_id] = f"call_{len(identifiers)}"
                call["id"] = identifiers[old_id]
                call["function"]["arguments"] = canonical(loads(call["function"]["arguments"]))
        if message["role"] == "tool":
            if message.get("tool_call_id") not in identifiers:
                raise ValueError("Orphan tool response during normalization")
            message["tool_call_id"] = identifiers[message["tool_call_id"]]
            try:
                message["content"] = canonical(loads(message["content"]))
            except (ValueError, TypeError):
                pass  # A non-JSON tool result remains exact text, never guessed.
        result.append(canonical(message))
    return tuple(result)


def user_conversation(row: dict[str, Any]) -> tuple[str, ...]:
    users = tuple(canonical(message.get("content")) for message in row["messages"] if message["role"] == "user")
    if not users:
        raise ValueError("Cannot identify a trace without user messages")
    return users


def is_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return bool(left) and len(left) <= len(right) and left == right[:len(left)]


def leakage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = {"conversation_overlap": [], "message_prefix_overlap": []}
    cached = {name: [(user_conversation(row), normalized_messages(row)) for row in rows]
              for name, rows in splits.items()}
    names = sorted(cached)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            for left_line, (left_users, left_messages) in enumerate(cached[left], 1):
                for right_line, (right_users, right_messages) in enumerate(cached[right], 1):
                    location = {"left": left, "left_line": left_line, "right": right, "right_line": right_line}
                    # Same initial request is conservatively one conversation, including
                    # divergent assistant completions and partial user snapshots.
                    if left_users[0] == right_users[0]:
                        report["conversation_overlap"].append({**location, "same_user_sequence": left_users == right_users})
                    if is_prefix(left_messages, right_messages) or is_prefix(right_messages, left_messages):
                        report["message_prefix_overlap"].append(location)
    return report


def require_disjoint(splits: dict[str, list[dict[str, Any]]]) -> None:
    report = leakage_report(splits)
    if any(report.values()):
        counts = {kind: len(pairs) for kind, pairs in report.items()}
        raise ValueError(f"Conversation/message-prefix leakage across splits: {counts}")


def inventory_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Source evidence only; incomplete snapshots are not assigned guessed actions."""
    records = []
    for line, row in enumerate(rows, 1):
        inventory_calls = []
        actions: set[str] = set()
        for message in row["messages"]:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = loads(function["arguments"])
                if function["name"] == "check_inventory":
                    inventory_calls.append({"id": call["id"], "sku": arguments.get("sku")})
                elif function["name"] == "calculate_resolution":
                    for item in arguments.get("items", []):
                        if not isinstance(item, dict) or not isinstance(item.get("action"), str):
                            raise ValueError("Malformed recorded calculate_resolution item action")
                        actions.add(item["action"])
        if inventory_calls:
            records.append({"source_line": line, "inventory_calls": inventory_calls,
                            "recorded_calculated_actions": sorted(actions),
                            "replacement_with_inventory": "replacement" in actions,
                            "exchange_with_inventory": "exchange" in actions,
                            "action_not_yet_recorded": not actions})
    return {"rows_with_inventory": len(records),
            "replacement_rows_with_inventory": sum(record["replacement_with_inventory"] for record in records),
            "exchange_rows_with_inventory": sum(record["exchange_with_inventory"] for record in records),
            "action_not_yet_recorded_rows": sum(record["action_not_yet_recorded"] for record in records),
            "records": records}


def conversation_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    users = [user_conversation(row) for row in rows]
    messages = [normalized_messages(row) for row in rows]
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for right in range(len(rows)):
        for left in range(right):
            if (users[left][0] == users[right][0] or is_prefix(messages[left], messages[right])
                    or is_prefix(messages[right], messages[left])):
                parents[root(right)] = root(left)
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def group_split(rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    groups = conversation_groups(rows)
    if len(groups) < 3:
        raise ValueError("At least three independent conversation groups are needed for nonempty train/val/test")
    described = []
    for indices in groups:
        identities = sorted({sha256(canonical(normalized_messages(rows[index])).encode("utf-8")) for index in indices})
        group_id = sha256(canonical(identities).encode("utf-8"))
        described.append({"id": group_id, "source_indices": indices,
                          "order": sha256(f"{SEED}:{group_id}".encode("ascii"))})
    remaining = sorted(described, key=lambda group: group["order"])

    def heldout(reserve: int) -> list[dict[str, Any]]:
        selected = []
        count = 0
        target = len(rows) / 10
        while len(remaining) > reserve:
            best = min(remaining, key=lambda group: (abs(count + len(group["source_indices"]) - target), group["order"]))
            new_count = count + len(best["source_indices"])
            if selected and abs(new_count - target) >= abs(count - target):
                break
            remaining.remove(best)
            selected.append(best)
            count = new_count
        return selected

    validation = heldout(reserve=2)
    test = heldout(reserve=1)
    assignments = {"train.jsonl": remaining, "val.jsonl": validation, "test.jsonl": test}
    split_indices = {name: sorted(index for group in assigned for index in group["source_indices"])
                     for name, assigned in assignments.items()}
    memberships = [{"id": group["id"], "split": name,
                    "source_lines": [index + 1 for index in group["source_indices"]]}
                   for name, assigned in assignments.items() for group in assigned]
    require_disjoint({name: [rows[index] for index in indices] for name, indices in split_indices.items()})
    return split_indices, sorted(memberships, key=lambda group: group["id"])


def load_sources(directory: Path = SOURCE) -> tuple[dict[str, bytes], dict[str, list[dict[str, Any]]]]:
    raw = {name: (directory / name).read_bytes() for name in SOURCE_HASHES}
    for name, expected in SOURCE_HASHES.items():
        if name in TEXT_SOURCE_FIXTURES:
            # These text Git blobs are LF; a Windows checkout may use CRLF.
            # Historical JSONL remains exact raw-byte provenance, not normalized.
            raw[name] = raw[name].replace(b"\r\n", b"\n")
        if sha256(raw[name]) != expected:
            raise ValueError(f"{name}: source SHA-256 mismatch; do not rewrite the preserved inputs")
    datasets = {name: read_jsonl(raw[name]) for name in SOURCE_ROWS}
    for name, count in SOURCE_ROWS.items():
        if len(datasets[name]) != count:
            raise ValueError(f"{name}: source count mismatch")
    return raw, datasets


def build_prepared(directory: Path = SOURCE) -> tuple[dict[str, bytes], dict[str, Any]]:
    raw, datasets = load_sources(directory)
    original_prompt = raw["zava_system_prompt.md"].decode("utf-8").replace("\r\n", "\n")
    original_tools = loads(raw["zava_tools.json"].decode("utf-8"))
    revised_prompt, revised_tools = revised_fixtures(original_prompt, original_tools)
    source_rows = datasets["traces-clean.jsonl"]
    rows = []
    for source in source_rows:
        validate_trace(source, original_prompt, original_tools)
        row = copy.deepcopy(source)
        row["messages"][0]["content"] = revised_prompt
        row["tools"] = copy.deepcopy(revised_tools)
        validate_trace(row, revised_prompt, revised_tools)
        rows.append(row)
    selected, groups = group_split(rows)
    splits = {name: [rows[index] for index in indices] for name, indices in selected.items()}
    payloads = {"traces-clean.jsonl": jsonl_bytes(rows),
                "zava_system_prompt.md": revised_prompt.encode("utf-8"),
                "zava_tools.json": (json.dumps(revised_tools, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")}
    payloads.update({name: jsonl_bytes(split) for name, split in splits.items()})
    assignments = {index: (name, output_line) for name, indices in selected.items()
                   for output_line, index in enumerate(indices, 1)}
    source_lines = raw["traces-clean.jsonl"].splitlines(keepends=True)
    membership = {line: group["id"] for group in groups for line in group["source_lines"]}
    manifest = {
        "schema_version": 1, "algorithm": ALGORITHM, "seed": SEED,
        "status": "prepared_offline_source_contract_validated_not_cloud_tested",
        "identity_rule": "alpha-renamed call IDs; canonical JSON arguments; exact non-system messages; shared first user request groups snapshots conservatively",
        "split_rule": "seeded SHA-256 group ordering; greedy whole-group 80/10/10 row targets; nonempty partitions; no row loss",
        "policy_change": "inventory applies to exchanges and replacements, including lost-package replacements; refund-only does not require stock",
        "policy_edits": {"system_prompt_replacements": POLICY_REPLACEMENTS,
                 "inventory_description": {"before": OLD_INVENTORY_DESCRIPTION, "after": NEW_INVENTORY_DESCRIPTION},
                 "action_description_addition": "replacement"},
        "source_assets": {name: {"sha256": sha256(content),
                    "hash_rule": "CRLF-to-LF" if name in TEXT_SOURCE_FIXTURES else "raw-bytes",
                    **({"rows": SOURCE_ROWS[name]} if name in SOURCE_ROWS else {})}
                          for name, content in raw.items()},
        "legacy_split_leakage": leakage_report({name: datasets[name] for name in SPLITS}),
        "legacy_inventory_usage": {name: inventory_usage(datasets[name]) for name in SPLITS},
        "groups": groups,
        "artifacts": {name: {"sha256": sha256(content), **({"rows": len(read_jsonl(content))} if name.endswith(".jsonl") else {})}
                      for name, content in payloads.items()},
        "records": [{"source_line": index + 1, "source_raw_line_sha256": sha256(source_lines[index]),
                     "normalized_conversation_sha256": sha256(canonical(normalized_messages(row)).encode("utf-8")),
                     "group_id": membership[index + 1], "split": assignments[index][0], "output_line": assignments[index][1],
                     "output_raw_line_sha256": sha256(jsonl_bytes([row]))}
                    for index, row in enumerate(rows)],
    }
    payloads["preparation-manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return payloads, manifest


def prepare(output: Path, source: Path = SOURCE) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory exists; choose a new explicit versioned directory")
    payloads, manifest = build_prepared(source)
    output.mkdir(parents=True, exist_ok=False)
    # The manifest is deliberately last; partial directories are not accepted.
    for name, content in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    return manifest


def validate_prepared(directory: Path, source: Path = SOURCE) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not (directory / "preparation-manifest.json").is_file():
        raise ValueError("Prepared trace data are missing. Run the documented offline preparation; legacy row splits are not accepted.")
    expected, manifest = build_prepared(source)
    for name, content in expected.items():
        if (directory / name).read_bytes() != content:
            raise ValueError(f"{name}: differs from independently rebuilt source-derived preparation")
    datasets = {name: read_jsonl(expected[name]) for name in ("traces-clean.jsonl",) + SPLITS}
    require_disjoint({name: datasets[name] for name in SPLITS})
    lineage = {name: dict(manifest["artifacts"][name]) for name in datasets}
    return datasets, lineage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare", "check"))
    parser.add_argument("--output-dir", type=Path, default=MODULE / "outputs" / "prepared")
    arguments = parser.parse_args(argv)
    if arguments.command == "audit":
        _, datasets = load_sources()
        report = leakage_report({name: datasets[name] for name in SPLITS})
        print(json.dumps({"source_rows": {name: len(rows) for name, rows in datasets.items()},
                          "legacy_split_leakage": report,
                          "legacy_inventory_usage": {name: inventory_usage(datasets[name]) for name in SPLITS}}, indent=2))
        return int(any(report.values()))
    if arguments.command == "prepare":
        manifest = prepare(arguments.output_dir)
        print(json.dumps({"artifacts": manifest["artifacts"], "conversation_groups": len(manifest["groups"])}, indent=2))
    else:
        _, lineage = validate_prepared(arguments.output_dir)
        print(json.dumps(lineage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())