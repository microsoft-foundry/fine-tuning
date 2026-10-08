"""xLAM function-calling dataset loading and tool-spec conversion.

Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
Each row has:
- query (str): user query
- tools (str, JSON): array of tools, each with {name, description, parameters}
  where parameters is {param_name: {type, description, required}}
- answers (str, JSON): array of {name, arguments} ground-truth tool calls

The dataset uses Python-style type names (int, float, list, ...) while JSON Schema
(used by tool-aware renderers) uses different names (integer, number, array, ...).
We convert at load time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from datasets import Dataset, load_dataset

from interactive_training.renderers import ToolSpec

logger = logging.getLogger(__name__)


# Map xLAM/Python type names to JSON Schema type names.
# Reference: https://json-schema.org/understanding-json-schema/reference/type
_XLAM_TO_JSONSCHEMA_TYPE: dict[str, str] = {
    "int": "integer",
    "integer": "integer",
    "long": "integer",
    "float": "number",
    "number": "number",
    "double": "number",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "dict": "object",
    "object": "object",
    "any": "string",  # JSON Schema has no "any"; default to string
}


def _normalize_type(xlam_type: str) -> str:
    """Convert an xLAM type string to JSON Schema. Strips subscripts like 'list[int]'."""
    base = xlam_type.strip().lower()
    if "[" in base:
        base = base.split("[", 1)[0]
    return _XLAM_TO_JSONSCHEMA_TYPE.get(base, "string")


@dataclass(frozen=True)
class XLAMAnswer:
    """A single ground-truth tool call from the xLAM dataset."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class XLAMTask:
    """A single xLAM function-calling task: query + available tools + gold calls."""

    query: str
    tool_specs: tuple[ToolSpec, ...]  # tuple for immutability (frozen dataclass)
    gold_answers: tuple[XLAMAnswer, ...]


def _xlam_tool_to_spec(tool: dict[str, Any]) -> ToolSpec:
    """Convert one xLAM tool definition to a JSON-Schema ToolSpec."""
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    params = tool.get("parameters", {}) or {}
    if isinstance(params, dict):
        for param_name, param_def in params.items():
            if not isinstance(param_def, dict):
                continue
            param_type = _normalize_type(str(param_def.get("type", "string")))
            properties[str(param_name)] = {
                "type": param_type,
                "description": str(param_def.get("description", "")),
            }
            # xLAM marks `required` explicitly per param. Default True if missing
            # (BFCL-style conservative behavior).
            if param_def.get("required", True):
                required.append(str(param_name))

    return {
        "name": str(tool.get("name", "")),
        "description": str(tool.get("description", "")),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _parse_json_field(value: Any, field_name: str) -> Any | None:
    """xLAM stores `tools` and `answers` as JSON-encoded strings. Parse defensively."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse %s field: %s", field_name, e)
        return None


def load_xlam_tasks(
    seed: int = 0,
    test_size: int = 1000,
) -> tuple[list[XLAMTask], list[XLAMTask]]:
    """Load Salesforce/xlam-function-calling-60k and split into train/test.

    The dataset ships as a single `train` split with no held-out test set, so we
    apply a deterministic seeded shuffle and take the last `test_size` rows as the
    held-out evaluation set.

    Note: this evaluation set is NOT directly comparable to the BFCL leaderboard
    (which uses a separate dataset and a more elaborate scoring harness). It is
    intended as an in-training metric for tracking progress on this same data
    distribution.

    Returns:
        (train_tasks, test_tasks)
    """
    logger.info("Loading Salesforce/xlam-function-calling-60k...")
    ds = cast(Dataset, load_dataset("Salesforce/xlam-function-calling-60k", split="train"))
    ds = ds.shuffle(seed=seed)

    n_total = len(ds)
    test_cutoff = max(0, n_total - max(0, test_size))

    train_tasks: list[XLAMTask] = []
    test_tasks: list[XLAMTask] = []
    skipped = 0

    for i, item in enumerate(ds):
        row = cast(dict[str, Any], item)
        query = row.get("query")
        tools_raw = _parse_json_field(row.get("tools"), "tools")
        answers_raw = _parse_json_field(row.get("answers"), "answers")

        if not isinstance(query, str) or not query.strip():
            skipped += 1
            continue
        if not isinstance(tools_raw, list) or not tools_raw:
            skipped += 1
            continue
        if not isinstance(answers_raw, list) or not answers_raw:
            skipped += 1
            continue

        tool_specs: list[ToolSpec] = []
        for tool in tools_raw:
            if isinstance(tool, dict) and "name" in tool:
                tool_specs.append(_xlam_tool_to_spec(tool))
        if not tool_specs:
            skipped += 1
            continue

        gold_answers: list[XLAMAnswer] = []
        for ans in answers_raw:
            if not isinstance(ans, dict):
                continue
            name = ans.get("name")
            args = ans.get("arguments", {})
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            gold_answers.append(XLAMAnswer(name=name, arguments=args))
        if not gold_answers:
            skipped += 1
            continue

        task = XLAMTask(
            query=query,
            tool_specs=tuple(tool_specs),
            gold_answers=tuple(gold_answers),
        )
        if i < test_cutoff:
            train_tasks.append(task)
        else:
            test_tasks.append(task)

    logger.info(
        "Loaded %d train + %d test tasks (skipped %d malformed rows of %d total)",
        len(train_tasks),
        len(test_tasks),
        skipped,
        n_total,
    )
    return train_tasks, test_tasks
