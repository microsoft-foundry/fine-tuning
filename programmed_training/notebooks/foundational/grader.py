import json
from collections import Counter
from typing import Any


def _parse_args(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if value is not None else {}


def _normalize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not function and call.get("name"):
            function = {"name": call.get("name"), "arguments": call.get("arguments", {})}
        name = function.get("name")
        if name:
            normalized.append(
                {
                    "function": {
                        "name": str(name),
                        "arguments": _parse_args(function.get("arguments", {})),
                    }
                }
            )
    return normalized


def grade_tool_calls(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> float:
    actual = _normalize_calls(actual)
    expected = _normalize_calls(expected)

    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    actual_names = [call["function"]["name"] for call in actual]
    expected_names = [call["function"]["name"] for call in expected]

    matched_count = sum((Counter(actual_names) & Counter(expected_names)).values())
    if matched_count == 0:
        return 0.0

    name_score = matched_count / max(len(expected), len(actual))
    if name_score < 1.0:
        return round(0.5 * name_score, 2)

    remaining = list(range(len(expected)))
    arg_scores = []
    for actual_index, actual_name in enumerate(actual_names):
        for expected_index in remaining:
            if actual_name == expected_names[expected_index]:
                actual_args = actual[actual_index]["function"]["arguments"]
                expected_args = expected[expected_index]["function"]["arguments"]
                arg_scores.append(_compare_args(actual_args, expected_args))
                remaining.remove(expected_index)
                break

    avg_arg_score = sum(arg_scores) / len(arg_scores)
    return round(0.5 + 0.5 * avg_arg_score, 2)


def _compare_args(actual: Any, expected: Any) -> float:
    matched, total = _count_leaves(actual, expected)
    return matched / total if total else 1.0


def _count_leaves(actual: Any, expected: Any) -> tuple[int, int]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0, _leaf_count(expected)
        matched = total = 0
        for key in expected:
            if key in actual:
                m, t = _count_leaves(actual[key], expected[key])
            else:
                m, t = 0, _leaf_count(expected[key])
            matched += m
            total += t
        return matched, total

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0, max(len(expected), 1)
        matched = total = 0
        for index, expected_item in enumerate(expected):
            if index < len(actual):
                m, t = _count_leaves(actual[index], expected_item)
            else:
                m, t = 0, 1
            matched += m
            total += t
        return matched, total

    return (1, 1) if actual == expected else (0, 1)


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(item) for item in value.values()) if value else 1
    if isinstance(value, list):
        return sum(_leaf_count(item) for item in value) if value else 1
    return 1
