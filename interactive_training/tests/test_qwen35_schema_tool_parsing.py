"""Schema-aware Qwen3.5/3.6 XML tool argument parsing tests."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from interactive_training.renderers.base import ToolCall, ToolSpec, UnparsedToolCall
from interactive_training.renderers.qwen3_5 import (
    Qwen3_5Renderer,
    _coerce_to_schema_type,
    _parse_qwen3_5_tool_call_xml,
)
from interactive_training.rl.message_env import EnvFromMessageEnv, MessageStepResult
from interactive_training.tool_use.tools import tool


_TOOLS: list[ToolSpec] = [
    {
        "name": "lookup_record",
        "description": "Look up a record.",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "revision": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array"},
                "metadata": {"type": "object"},
                "optional": {"type": ["string", "null"]},
            },
            "required": ["record_id", "revision"],
            "additionalProperties": False,
        },
    }
]


class _Payload(BaseModel):
    count: int


class _Mode(str, Enum):
    READ = "read"


@tool
def _generated_schema_tool(payload: _Payload, arbitrary: Any, mode: _Mode):
    raise NotImplementedError


class _CharacterTokenizer:
    """Tiny tokenizer with a single-token Qwen end marker."""

    _END = "<|im_end|>"
    _END_ID = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens: list[int] = []
        while text:
            if text.startswith(self._END):
                tokens.append(self._END_ID)
                text = text[len(self._END) :]
            else:
                tokens.append(ord(text[0]) + 2)
                text = text[1:]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self._END if token == self._END_ID else chr(token - 2) for token in tokens
        )


@pytest.mark.parametrize(
    ("raw", "schema", "expected"),
    [
        ("1234", {"type": "string"}, "1234"),
        ("true", {"type": "string"}, "true"),
        ("null", {"type": "string"}, "null"),
        ("1234", {"type": "integer"}, 1234),
        ("1.5", {"type": "number"}, 1.5),
        ("true", {"type": "boolean"}, True),
        ("True", {"type": "boolean"}, True),
        ("[1, 2]", {"type": "array"}, [1, 2]),
        ('{"a": 1}', {"type": "object"}, {"a": 1}),
        ("null", {"type": ["string", "null"]}, None),
        ("None", {"type": ["string", "null"]}, None),
        ("1234", {"type": ["string", "null"]}, "1234"),
        ("1234", None, "1234"),
        ("2", {"enum": [1, 2, 3]}, 2),
        ("2", {"enum": ["2", 3]}, "2"),
        ("1234", {"type": "int"}, 1234),
        ("x", {"anyOf": [{"type": "string"}, {"type": "null"}]}, "x"),
    ],
)
def test_coerce_to_declared_schema(raw, schema, expected):
    assert _coerce_to_schema_type(raw, schema) == expected


@pytest.mark.parametrize(
    ("schema", "raw"),
    [
        ({"type": "integer"}, "not-an-int"),
        ({"type": "array"}, "not-an-array"),
        ({"type": "object"}, "[]"),
        ({"$ref": "#/$defs/id"}, "1234"),
    ],
)
def test_invalid_or_unsupported_schema_conversion_fails(schema, raw):
    with pytest.raises(ValueError):
        _coerce_to_schema_type(raw, schema)


def test_xml_parser_uses_function_parameter_schemas():
    raw_inner = """<function=lookup_record>
<parameter=record_id>
1234
</parameter>
<parameter=revision>
1234
</parameter>
<parameter=enabled>
true
</parameter>
</function>"""
    parsed = _parse_qwen3_5_tool_call_xml(
        raw_inner, f"<tool_call>{raw_inner}</tool_call>", _TOOLS
    )

    assert isinstance(parsed, ToolCall)
    assert json.loads(parsed.function.arguments) == {
        "record_id": "1234",
        "revision": 1234,
        "enabled": True,
    }


@pytest.mark.parametrize(
    ("raw_arbitrary", "expected_arbitrary"),
    [
        ("123", 123),
        ("true", True),
        ('{"nested": 1}', {"nested": 1}),
    ],
)
def test_xml_parser_supports_tool_generated_schemas(
    raw_arbitrary: str, expected_arbitrary: object
):
    tool_spec = _generated_schema_tool.to_spec()
    properties = tool_spec["parameters"]["properties"]
    assert "$ref" in properties["payload"]
    assert not ({"type", "enum", "anyOf", "oneOf", "allOf"} & properties["arbitrary"].keys())
    assert "$ref" in properties["mode"]

    raw_inner = f"""<function={tool_spec["name"]}>
<parameter=payload>
{{"count": 2}}
</parameter>
<parameter=arbitrary>
{raw_arbitrary}
</parameter>
<parameter=mode>
read
</parameter>
</function>"""
    parsed = _parse_qwen3_5_tool_call_xml(
        raw_inner, f"<tool_call>{raw_inner}</tool_call>", [tool_spec]
    )

    assert isinstance(parsed, ToolCall)
    assert json.loads(parsed.function.arguments) == {
        "payload": {"count": 2},
        "arbitrary": expected_arbitrary,
        "mode": "read",
    }


def test_xml_parser_enforces_enum_from_tool_generated_ref():
    tool_spec = _generated_schema_tool.to_spec()
    raw_inner = f"""<function={tool_spec["name"]}>
<parameter=mode>
write
</parameter>
</function>"""

    parsed = _parse_qwen3_5_tool_call_xml(
        raw_inner, f"<tool_call>{raw_inner}</tool_call>", [tool_spec]
    )

    assert isinstance(parsed, UnparsedToolCall)


def test_xml_parser_preserves_unknown_parameter_as_string():
    raw_inner = """<function=lookup_record>
<parameter=undeclared>
1234
</parameter>
</function>"""
    parsed = _parse_qwen3_5_tool_call_xml(
        raw_inner, f"<tool_call>{raw_inner}</tool_call>", _TOOLS
    )

    assert isinstance(parsed, ToolCall)
    assert json.loads(parsed.function.arguments) == {"undeclared": "1234"}


def test_xml_parser_returns_unparsed_call_for_schema_mismatch():
    raw_inner = """<function=lookup_record>
<parameter=revision>
not-an-int
</parameter>
</function>"""
    parsed = _parse_qwen3_5_tool_call_xml(
        raw_inner, f"<tool_call>{raw_inner}</tool_call>", _TOOLS
    )

    assert isinstance(parsed, UnparsedToolCall)
    assert "does not match schema" in parsed.error


def test_public_response_parser_recovers_types_from_tools():
    tokenizer = _CharacterTokenizer()
    renderer = Qwen3_5Renderer(tokenizer)
    response_text = """<think>

</think>

<tool_call>
<function=lookup_record>
<parameter=record_id>
1234
</parameter>
<parameter=revision>
1234
</parameter>
</function>
</tool_call><|im_end|>"""

    message, success = renderer.parse_response_with_tools(tokenizer.encode(response_text), _TOOLS)

    assert success
    assert "unparsed_tool_calls" not in message
    arguments = json.loads(message["tool_calls"][0].function.arguments)
    assert arguments == {"record_id": "1234", "revision": 1234}


@pytest.mark.asyncio
async def test_message_env_forwards_tool_specs_to_renderer():
    renderer = MagicMock()
    renderer.get_stop_sequences.return_value = []
    renderer.parse_response_with_tools.return_value = (
        {"role": "assistant", "content": "done"},
        True,
    )
    message_env = MagicMock()
    message_env.step = AsyncMock(
        return_value=MessageStepResult(
            reward=1.0,
            episode_done=True,
            next_messages=[],
        )
    )
    env = EnvFromMessageEnv(renderer=renderer, message_env=message_env, tool_specs=_TOOLS)

    await env.step([10, 11])

    renderer.parse_response_with_tools.assert_called_once_with([10, 11], _TOOLS)