"""BFCL-style AST matching and reward function for xLAM function calling.

Per-call match follows the Berkeley Function Calling Leaderboard (BFCL) AST
evaluation rules (https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#evaluation-metrics):

- Function name: exact match, with `.` <-> `_` substitution allowed
- Arguments: dict key set must match exactly (no missing required, no extras)
- Per-value matching:
  - bool: strict (no string "true")
  - int/float: int accepted where float expected (Python auto-conversion);
               float NOT accepted where int expected
  - str: case-insensitive after whitespace + `,./-_*^` punctuation stripping
  - list/tuple: order-sensitive, recursive value match
  - dict: key-set match, recursive value match (key order ignored)

Across multiple calls (parallel function calling), we use bipartite greedy
matching and report two scores:
- `correct` (used as the RL reward): matched_calls / max(|pred|, |gold|).
  This gives a smooth gradient for partial successes, which is well-suited
  to RL training (BFCL's all-or-nothing scoring is too sparse).
- `bfcl_strict` (logged as a metric only): 1.0 iff every gold call matched
  exactly AND |pred| == |gold|. Use this metric to compare to BFCL leaderboard
  numbers (though note the leaderboard uses a different test set).

Final reward formula:
    reward = format_coef * (format_score - 1) + correct
where format_score = 1 iff message has at least one parseable tool call AND no
unparseable tool calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from interactive_training.recipes.tool_rl.data import XLAMAnswer
from interactive_training.renderers import ToolCall

logger = logging.getLogger(__name__)


_BFCL_STRIP_PUNCT = ",./-_*^"


def _normalize_string(s: str) -> str:
    """BFCL string normalization: lowercase, strip whitespace, strip select punctuation."""
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    for ch in _BFCL_STRIP_PUNCT:
        s = s.replace(ch, "")
    return s


def _function_name_match(pred: str, gold: str) -> bool:
    """Function name match with BFCL's `.` <-> `_` tolerance."""
    if pred == gold:
        return True
    return pred.replace(".", "_") == gold.replace(".", "_")


def _value_match(pred: Any, gold: Any) -> bool:
    """Single-value match per BFCL semantics. See module docstring for rules."""
    # bool MUST be checked before int (bool is a subclass of int in Python).
    if isinstance(gold, bool) or isinstance(pred, bool):
        return type(pred) is bool and type(gold) is bool and pred == gold

    if gold is None:
        return pred is None
    if pred is None:
        return False

    # Numeric: gold==float allows int pred; gold==int requires int pred.
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        if isinstance(gold, float):
            return float(pred) == float(gold)
        # gold is int -> reject float pred
        return isinstance(pred, int) and pred == gold

    if isinstance(gold, str) and isinstance(pred, str):
        return _normalize_string(pred) == _normalize_string(gold)

    if isinstance(gold, list) and isinstance(pred, list):
        if len(pred) != len(gold):
            return False
        return all(_value_match(p, g) for p, g in zip(pred, gold))

    if isinstance(gold, dict) and isinstance(pred, dict):
        if set(pred.keys()) != set(gold.keys()):
            return False
        return all(_value_match(pred[k], gold[k]) for k in gold)

    # Type mismatch (e.g. gold int vs pred str) — BFCL strict.
    return False


def _arguments_match(pred_args: dict[str, Any], gold_args: dict[str, Any]) -> bool:
    """Arguments match: exact key-set equality + per-value match.

    BFCL forbids extra hallucinated parameters and missing required parameters.
    Since the xLAM dataset only stores the answer's *actual* arguments (which are
    presumed to satisfy the function's `required` set), exact-key-set equality is
    the right strict check here.
    """
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    return all(_value_match(pred_args[k], gold_args[k]) for k in gold_args)


def _single_call_match(pred: ToolCall, gold: XLAMAnswer) -> bool:
    """BFCL-style strict per-call match: function name + arguments."""
    if not _function_name_match(pred.function.name, gold.name):
        return False
    try:
        pred_args = json.loads(pred.function.arguments)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(pred_args, dict):
        return False
    return _arguments_match(pred_args, gold.arguments)


def compute_call_match_score(
    pred_calls: list[ToolCall],
    gold_calls: list[XLAMAnswer],
) -> tuple[float, bool]:
    """Bipartite greedy matching of predicted vs gold tool calls.

    Returns:
        (partial_score, exact_all_or_nothing)
        - partial_score in [0, 1]: matched / max(|pred|, |gold|)
        - exact_all_or_nothing: True iff every gold matched AND |pred| == |gold|
    """
    if not gold_calls:
        # Defensive — load_xlam_tasks filters these out, but be safe.
        empty = not pred_calls
        return (1.0 if empty else 0.0), empty

    matched = 0
    used_pred: set[int] = set()
    for gold in gold_calls:
        for i, pred in enumerate(pred_calls):
            if i in used_pred:
                continue
            if _single_call_match(pred, gold):
                used_pred.add(i)
                matched += 1
                break

    denom = max(len(pred_calls), len(gold_calls))
    partial = matched / denom if denom > 0 else 0.0
    exact = matched == len(gold_calls) and len(pred_calls) == len(gold_calls)
    return partial, exact
