import json, math
from typing import Any, Dict, List, Optional, Tuple

PASS = 1.0  # normalized 0..1

def grade(sample, item) -> float:
    actual_tool_calls = sample.get("output_tools")
    expected_tool_calls = (item.get("expected_output") or {}).get("tool_calls")
    return grade_tool_calls(actual_tool_calls, expected_tool_calls)

def compare_function_calls(actual_calls, expected_calls) -> float:
    return grade_tool_calls(actual_calls, expected_calls)

# ---------- helpers (small + focused) ----------

def _safe_json(v: Any) -> Any:
    if isinstance(v, (dict, list, int, float, bool)) or v is None:
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v.strip()
    return v

def _num_like(s: str):
    s = s.strip()
    if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
        try: return int(s)
        except: pass
    try:
        f = float(s)
        if math.isfinite(f): return f
    except: pass
    return s

def _norm(v: Any) -> Any:
    v = _safe_json(v)
    if isinstance(v, dict):
        return {k: _norm(v[k]) for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, str):
        return _num_like(v)
    return v

def _name_args(call: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    fn = call.get("function", call.get("tool", {}))
    return fn.get("name"), _norm(fn.get("arguments", {}))

def _sim(a: Dict[str, Any], e: Dict[str, Any]) -> float:
    an, aa = _name_args(a)
    en, ea = _name_args(e)
    if not an or not en or an != en:
        return 0.0
    return 1.0 if aa == ea else 0.5

# ---------- core grading (order-invariant, partial credit) ----------

def grade_tool_calls(
    actual_tool_calls: Optional[List[Dict[str, Any]]],
    expected_tool_calls: Optional[List[Dict[str, Any]]],
    *, partial_weight: float = 0.5
) -> float:
    if not actual_tool_calls and not expected_tool_calls:
        return PASS
    if (not actual_tool_calls) != (not expected_tool_calls):
        return 0.0

    actual = actual_tool_calls or []
    expected = expected_tool_calls or []

    unused = list(range(len(actual)))
    exact = name_only = 0

    # pass 1: exact matches
    for e in expected:
        hit = next((i for i in unused if _sim(actual[i], e) == 1.0), None)
        if hit is not None:
            exact += 1
            unused.remove(hit)

    # pass 2: name-only matches
    for e in expected:
        hit = next((i for i in unused if _sim(actual[i], e) == 0.5), None)
        if hit is not None:
            name_only += 1
            unused.remove(hit)

    soft_tp = exact + partial_weight * name_only
    precision = soft_tp / len(actual) if actual else 1.0
    recall    = soft_tp / len(expected) if expected else 1.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return round(f1, 6)
