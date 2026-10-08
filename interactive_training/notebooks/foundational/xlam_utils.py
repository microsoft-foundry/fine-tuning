import json
from typing import Any


_XLAM_TO_JSONSCHEMA_TYPE = {
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
    "any": "string",
}


def parse_jsonish(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default if value is None else value


def _normalize_type(type_name: str) -> str:
    # xLAM uses annotations such as "int, optional", not just plain types.
    base = str(type_name).split(",", 1)[0].strip().lower()
    if "[" in base:
        base = base.split("[", 1)[0]
    return _XLAM_TO_JSONSCHEMA_TYPE.get(base, "string")


def xlam_tools_to_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = parse_jsonish(row.get("tools"), [])
    specs = []
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        properties = {}
        required = []
        params = tool.get("parameters") or {}
        if isinstance(params, dict):
            for name, param in params.items():
                if not isinstance(param, dict):
                    continue
                type_name = str(param.get("type", "string"))
                optional = any(
                    part.strip().lower() == "optional"
                    for part in type_name.split(",")[1:]
                )
                properties[str(name)] = {
                    "type": _normalize_type(type_name),
                    "description": str(param.get("description", "")),
                }
                if param.get("required", not optional):
                    required.append(str(name))
        specs.append(
            {
                "name": str(tool["name"]),
                "description": str(tool.get("description", "")),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )
    return specs


def xlam_expected_tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    answers = parse_jsonish(row.get("answers"), [])
    calls = []
    for answer in answers if isinstance(answers, list) else []:
        if not isinstance(answer, dict) or not answer.get("name"):
            continue
        calls.append(
            {
                "function": {
                    "name": str(answer["name"]),
                    "arguments": answer.get("arguments") or {},
                }
            }
        )
    return calls


def renderer_tool_calls_to_grader(tool_calls: list[Any]) -> list[dict[str, Any]]:
    calls = []
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, dict):
            function = call.get("function")
        if function is None:
            continue
        name = getattr(function, "name", None)
        args = getattr(function, "arguments", None)
        if isinstance(function, dict):
            name = function.get("name")
            args = function.get("arguments", args)
        if name:
            calls.append({"function": {"name": str(name), "arguments": args or {}}})
    return calls
