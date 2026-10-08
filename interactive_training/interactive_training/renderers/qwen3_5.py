"""
Qwen3.5 family renderer (text-only).

Qwen3.5 models share the same basic chat format as Qwen3 (im_start/im_end,
thinking, etc.) but with two important differences:

1. The HF chat template always adds <think>...</think> blocks to assistant
   messages that follow the last user message (empty if no reasoning content),
   and always adds <think>\\n to the generation prompt.
2. Tool calling uses an XML "parameter" format:
       <tool_call>
       <function=name>
       <parameter=p>value</parameter>
       </function>
       </tool_call>

   instead of Qwen3's JSON {"name": ..., "arguments": ...} format.

Reference: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/tokenizer_config.json

Includes:
- Qwen3_5Renderer: Qwen3.5 with thinking enabled (default)
- Qwen3_5DisableThinkingRenderer: Qwen3.5 with thinking disabled
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from azure.ai.finetuningsessions.models import ModelInputChunk

from interactive_training.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Role,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    parse_content_blocks,
    parse_response_for_stop_token,
    remove_thinking,
)
from interactive_training.renderers.qwen3 import Qwen3Renderer

logger = logging.getLogger(__name__)

_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*$",
    re.DOTALL,
)
_PARAM_BLOCK_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)

_TYPE_ALIASES = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "long": "integer",
    "short": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
    "arr": "array",
    "sequence": "array",
}


def _resolve_local_ref(ref: str, root_schema: dict[str, Any]) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local JSON Schema references are supported: {ref!r}")

    resolved: object = root_schema
    for encoded_part in ref[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(resolved, dict) or part not in resolved:
            raise ValueError(f"JSON Schema reference does not resolve: {ref!r}")
        resolved = resolved[part]
    return resolved


def _extract_schema_types(
    schema: object,
    root_schema: dict[str, Any] | None = None,
    resolved_refs: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the possible JSON types declared by a parameter schema."""
    if not isinstance(schema, dict):
        return []
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if root_schema is None:
            raise ValueError("Cannot resolve $ref without the enclosing tool schema")
        if ref in resolved_refs:
            raise ValueError(f"Circular JSON Schema reference: {ref!r}")
        return _extract_schema_types(
            _resolve_local_ref(ref, root_schema), root_schema, resolved_refs | {ref}
        )

    types: set[str] = set()
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        types.add(raw_type)
    elif isinstance(raw_type, list):
        types.update(value for value in raw_type if isinstance(value, str))

    enum = schema.get("enum")
    if isinstance(enum, list):
        for value in enum:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for keyword in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(keyword)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_schema_types(choice, root_schema, resolved_refs))

    return list(types)


def _coerce_to_schema_type(
    value: str, schema: object, root_schema: dict[str, Any] | None = None
) -> object:
    """Convert untyped XML text according to its declared JSON Schema."""
    if not isinstance(schema, dict):
        return value

    effective_schema = schema
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if root_schema is None:
            raise ValueError("Cannot resolve $ref without the enclosing tool schema")
        resolved_schema = _resolve_local_ref(ref, root_schema)
        if not isinstance(resolved_schema, dict):
            raise ValueError(f"JSON Schema reference is not a schema object: {ref!r}")
        effective_schema = resolved_schema

    schema_types = {
        _TYPE_ALIASES.get(schema_type.lower(), schema_type.lower())
        for schema_type in _extract_schema_types(effective_schema, root_schema)
    }
    if not schema_types:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    allowed_values = effective_schema.get("enum")

    def allowed(candidate: object) -> bool:
        return not isinstance(allowed_values, list) or candidate in allowed_values

    for candidate in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if candidate not in schema_types:
            continue
        if candidate == "null":
            if value.strip().lower() in {"null", "none"} and allowed(None):
                return None
        elif candidate == "integer":
            try:
                parsed_integer = int(value)
            except ValueError:
                pass
            else:
                if allowed(parsed_integer):
                    return parsed_integer
        elif candidate == "number":
            try:
                number = float(value)
            except ValueError:
                pass
            else:
                parsed_number = int(number) if number.is_integer() else number
                if allowed(parsed_number):
                    return parsed_number
        elif candidate == "boolean":
            normalized = value.strip().lower()
            if normalized in {"true", "1"} and allowed(True):
                return True
            if normalized in {"false", "0"} and allowed(False):
                return False
        elif candidate in {"object", "array"}:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                expected_type = dict if candidate == "object" else list
                if isinstance(parsed, expected_type) and allowed(parsed):
                    return parsed
        else:
            if allowed(value):
                return value

    raise ValueError(f"value {value!r} does not match schema types {sorted(schema_types)}")


def _find_tool_parameters(tools: list[ToolSpec], function_name: str) -> dict[str, Any]:
    for tool in tools:
        if tool.get("name") != function_name:
            continue
        parameters = tool.get("parameters")
        if isinstance(parameters, dict):
            return parameters
        return {}
    logger.warning("Parsed tool %r is not present in the declared tools", function_name)
    return {}


def _format_tool_call_argument_python_style(param_value: object) -> str:
    """Serialize one tool-call argument the way the Qwen3.5 template does.

    Qwen3.5 uses ``tojson`` for mappings and non-string sequences but falls back
    to Jinja's ``| string`` for scalars, so booleans and null render Python-style
    as ``True``/``None``. Qwen3.6 and Qwen3.8 changed this to ``tojson`` for
    everything that is not a string (``true``/``null``) -- see
    :meth:`Qwen3_8Renderer._format_tool_call_argument`.
    """
    if isinstance(param_value, (dict, list)):
        return json.dumps(param_value)
    return str(param_value)


def _format_tool_call_xml(
    tool_call: ToolCall,
    format_argument: Callable[[object], str] = _format_tool_call_argument_python_style,
) -> str:
    """Format a single tool call in Qwen3.5's XML parameter format."""
    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
    lines = [f"<tool_call>\n<function={tool_call.function.name}>"]
    for param_name, param_value in args.items():
        value_str = format_argument(param_value)
        lines.append(f"<parameter={param_name}>\n{value_str}\n</parameter>")
    lines.append("</function>\n</tool_call>")
    return "\n".join(lines)


def _parse_qwen3_5_tool_call_xml(
    raw_inner: str,
    raw_full: str,
    tools: list[ToolSpec] | None = None,
) -> ToolCall | UnparsedToolCall:
    """Parse the body of a <tool_call>...</tool_call> block in XML form."""
    match = _FUNCTION_BLOCK_RE.match(raw_inner)
    if not match:
        return UnparsedToolCall(raw_text=raw_full, error="Malformed Qwen3.5 tool call XML")

    function_name = match.group("name").strip()
    body = match.group("body")
    if not function_name:
        return UnparsedToolCall(raw_text=raw_full, error="Missing function name")

    arguments: dict[str, object] = {}
    parameters = _find_tool_parameters(tools, function_name) if tools is not None else {}
    raw_properties = parameters.get("properties")
    properties = raw_properties if isinstance(raw_properties, dict) else {}
    pos = 0
    for param in _PARAM_BLOCK_RE.finditer(body):
        if body[pos : param.start()].strip():
            return UnparsedToolCall(
                raw_text=raw_full,
                error="Unexpected non-parameter content inside <function> block",
            )

        param_name = param.group("name").strip()
        param_value_text = param.group("value").strip("\n")

        if not param_name:
            return UnparsedToolCall(raw_text=raw_full, error="Empty parameter name")

        if tools is None:
            try:
                param_value: object = json.loads(param_value_text)
            except json.JSONDecodeError:
                param_value = param_value_text
        else:
            param_schema = properties.get(param_name)
            if param_name not in properties and properties:
                logger.warning(
                    "Parsed parameter %r is not declared for tool %r; preserving it as a string",
                    param_name,
                    function_name,
                )
            try:
                param_value = _coerce_to_schema_type(
                    param_value_text, param_schema, parameters
                )
            except ValueError as exc:
                return UnparsedToolCall(raw_text=raw_full, error=str(exc))

        arguments[param_name] = param_value
        pos = param.end()

    if body[pos:].strip():
        return UnparsedToolCall(
            raw_text=raw_full,
            error="Unexpected trailing content inside <function> block",
        )

    return ToolCall(
        function=ToolCall.FunctionBody(
            name=function_name,
            arguments=json.dumps(arguments),
        )
    )


class Qwen3_5Renderer(Qwen3Renderer):
    """
    Renderer for Qwen3.5 models with thinking enabled.

    Differences vs Qwen3Renderer:
    - Generation prompt ends with ``<|im_start|>assistant\\n<think>\\n``.
    - Assistant messages that follow the last user message and have no thinking
      content get an empty ``<think>\\n\\n</think>\\n\\n`` injected (matching the
      HF template's behaviour).
    - Thinking content is rendered with newline padding:
      ``<think>\\n{thinking}\\n</think>\\n\\n`` instead of ``<think>{thinking}</think>``.
    - Tool calls are rendered/parsed in XML ``<function=...><parameter=...>`` form.
    """

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Empty <think> block injected after the assistant header when needed.

        Returns the empty think string ``"<think>\\n\\n</think>\\n\\n"`` for
        assistant messages that come after the last user message and contain no
        thinking content; ``""`` otherwise. Matches the HF Qwen3.5 template.
        """
        if message["role"] != "assistant":
            return ""
        if ctx.last_user_index < 0 or ctx.idx <= ctx.last_user_index:
            return ""

        content = message.get("content", "")
        has_think = False
        if isinstance(content, list):
            has_think = any(p["type"] == "thinking" for p in content)
        elif isinstance(content, str):
            has_think = "<think>" in content

        return "" if has_think else "<think>\n\n</think>\n\n"

    def _format_tool_call_argument(self, param_value: object) -> str:
        """Serialize one tool-call argument. Qwen3.5 renders scalars Python-style."""
        return _format_tool_call_argument_python_style(param_value)

    def _render_tool_calls_text(self, message: Message) -> str:
        """Render tool_calls using Qwen3.5's XML parameter format."""
        content = message.get("content", "")
        if isinstance(content, str):
            has_visible_content = bool(content.strip())
        else:
            has_visible_content = any(
                part["type"] == "text" and bool(part["text"].strip()) for part in content
            )
        separator = "\n\n" if has_visible_content else ""
        return separator + "\n".join(
            _format_tool_call_xml(tc, self._format_tool_call_argument)
            for tc in message["tool_calls"]
        )

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        # Inject empty <think>\n\n</think>\n\n into the assistant header when the
        # template requires it (no training weight, matches the prefill at
        # sampling time).
        header_str += self._assistant_header_suffix(message, ctx)

        content = message["content"]

        if isinstance(content, list):
            parts = content
            if (
                self.strip_thinking_from_history
                and message["role"] == "assistant"
                and not ctx.is_last
            ):
                parts = remove_thinking(parts)
            text_part_indexes = [
                index for index, part in enumerate(parts) if part["type"] == "text"
            ]
            rendered_parts: list[str] = []
            for index, p in enumerate(parts):
                if p["type"] == "thinking":
                    rendered_parts.append(f"<think>\n{p['thinking']}\n</think>\n\n")
                elif p["type"] == "text":
                    text = p["text"]
                    if "tool_calls" in message and text_part_indexes:
                        if index == text_part_indexes[0]:
                            text = text.lstrip()
                        if index == text_part_indexes[-1]:
                            text = text.rstrip()
                    rendered_parts.append(text)
            output_content = "".join(rendered_parts)
        else:
            output_content = content.strip() if "tool_calls" in message else content

        if message["role"] == "tool":
            output_content = self._wrap_qwen_tool_response(output_content)

        if "tool_calls" in message:
            # HF trims text content before tool calls but preserves thinking separators.
            # Tinker-cookbook carries the equivalent fix (6f4d702), so preserve on sync.
            output_content += self._render_tool_calls_text(message)

        output_content += "<|im_end|>"

        header = ModelInputChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[ModelInputChunk] = [
            ModelInputChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    # ------------------------------------------------------------------
    # Generation suffix
    # ------------------------------------------------------------------

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Append ``<think>\\n`` to the assistant header for thinking-mode sampling."""
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n<think>\n"
        return self.tokenizer.encode(header_str, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the prefilled ``<think>\\n`` before parsing sampled tokens.

        The generation suffix prefills ``<think>\\n``, so sampled tokens start
        after that prefix. If the response contains ``</think>`` but does not
        start with ``<think>\\n``, we prepend it so the parser sees a complete
        think block.
        """
        think_prefix_tokens = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        think_suffix_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        if not think_suffix_tokens:
            return response
        suffix_token = think_suffix_tokens[0]

        starts_with_think = (
            len(response) >= len(think_prefix_tokens)
            and response[: len(think_prefix_tokens)] == think_prefix_tokens
        )

        if not starts_with_think and suffix_token in response:
            return think_prefix_tokens + response
        return response

    def _postprocess_parsed_message(
        self, message: Message, tools: list[ToolSpec] | None = None
    ) -> None:
        """Apply Qwen3.5-specific post-processing to a parsed message in-place.

        1. Strips whitespace from thinking content (matches HF template ``|trim``).
        2. Removes the two separator newlines between ``</think>`` and text.
        3. Converts Qwen3.5 XML tool calls from the parent's unparsed_tool_calls.
        """
        content = message.get("content")
        if isinstance(content, list):
            first_text_after_thinking: TextPart | None = None
            seen_thinking = False
            for p in content:
                if p["type"] == "thinking":
                    p["thinking"] = p["thinking"].strip()
                    seen_thinking = True
                elif seen_thinking and p["type"] == "text":
                    first_text_after_thinking = p
                    break

            if first_text_after_thinking is not None and first_text_after_thinking[
                "text"
            ].startswith("\n\n"):
                first_text_after_thinking["text"] = first_text_after_thinking["text"][2:]

        # The Qwen3 parent parser assumes JSON inside <tool_call>; reinterpret
        # any blocks that look like XML <function=...> calls here.
        converted_xml_calls: list[ToolCall] = []
        remaining_unparsed: list[UnparsedToolCall] = []
        for unparsed in message.get("unparsed_tool_calls", []):
            raw = unparsed.raw_text
            if "<function=" not in raw:
                remaining_unparsed.append(unparsed)
                continue
            # The parent stores the full <tool_call>...</tool_call> region; strip
            # the outer tags before handing the inner XML body to the parser.
            inner = raw
            if inner.startswith("<tool_call>"):
                inner = inner[len("<tool_call>") :]
            if inner.endswith("</tool_call>"):
                inner = inner[: -len("</tool_call>")]
            parsed = _parse_qwen3_5_tool_call_xml(inner, raw, tools)
            if isinstance(parsed, ToolCall):
                converted_xml_calls.append(parsed)
            else:
                remaining_unparsed.append(parsed)

        if converted_xml_calls:
            message["tool_calls"] = list(message.get("tool_calls", [])) + converted_xml_calls
        if remaining_unparsed:
            message["unparsed_tool_calls"] = remaining_unparsed
        else:
            message.pop("unparsed_tool_calls", None)

    def _parse_response(
        self, response: list[int], tools: list[ToolSpec] | None
    ) -> tuple[Message, bool]:
        normalized = self._normalize_response_tokens(response)
        assistant_message, parse_success = parse_response_for_stop_token(
            normalized, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]
        result = parse_content_blocks(content)
        if result is not None:
            parts, tool_results = result
            assistant_message["content"] = parts
            tool_calls = [t for t in tool_results if isinstance(t, ToolCall)]
            unparsed = [t for t in tool_results if isinstance(t, UnparsedToolCall)]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if unparsed:
                assistant_message["unparsed_tool_calls"] = unparsed
        else:
            assistant_message["content"] = content

        self._postprocess_parsed_message(assistant_message, tools)
        return assistant_message, True

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return self._parse_response(response, None)

    def parse_response_with_tools(
        self, response: list[int], tools: list[ToolSpec]
    ) -> tuple[Message, bool]:
        return self._parse_response(response, tools)

    # ------------------------------------------------------------------
    # OpenAI mapping
    # ------------------------------------------------------------------

    def to_openai_message(self, message: Message) -> dict:
        """Same as Qwen3, but expose tool-call arguments as a dict.

        Qwen3.5's HF template iterates over arguments with ``|items``, which
        requires a mapping rather than a JSON string.
        """
        result = super().to_openai_message(message)
        if "tool_calls" in result:
            for tc in result["tool_calls"]:
                args_str = tc["function"].get("arguments", "")
                if isinstance(args_str, str) and args_str:
                    try:
                        tc["function"]["arguments"] = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Leave as-is on malformed JSON.
                        pass
        return result

    # ------------------------------------------------------------------
    # Tool spec system prompt
    # ------------------------------------------------------------------

    def _serialize_tool_declaration(self, tool: ToolSpec) -> str:
        """Serialize one tool for the ``<tools>`` block.

        The templates apply ``| tojson``, which transformers overrides with
        ``ensure_ascii=False``. Python's ``json.dumps`` defaults to ``True``, so
        a tool whose description or enum values contain non-ASCII text escapes
        to ``\\uXXXX`` here but stays literal in the template. Qwen3.5/3.6 are
        already in production with the escaped form, so this default preserves
        it byte-for-byte; :class:`~interactive_training.renderers.qwen3_8.Qwen3_8Renderer`
        overrides it to match its own template.
        """
        return json.dumps(tool)

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create the Qwen3.5 system message describing tools in XML form."""
        tools_text = ""
        if tools:
            tool_lines = "\n".join(self._serialize_tool_declaration(tool) for tool in tools)
            tools_text = (
                "# Tools\n\n"
                "You have access to the following functions:\n\n"
                "<tools>\n"
                f"{tool_lines}\n"
                "</tools>\n\n"
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n"
                "<function=example_function_name>\n"
                "<parameter=example_parameter_1>\n"
                "value_1\n"
                "</parameter>\n"
                "<parameter=example_parameter_2>\n"
                "This is the value for the second parameter\n"
                "that can span\n"
                "multiple lines\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n\n"
                "<IMPORTANT>\n"
                "Reminder:\n"
                "- Function calls MUST follow the specified format: "
                "an inner <function=...></function> block must be nested within "
                "<tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language "
                "BEFORE the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with "
                "your current knowledge and do not tell the user about function calls\n"
                "</IMPORTANT>"
            )

        if tools_text:
            content = tools_text + "\n\n" + system_prompt if system_prompt else tools_text
        else:
            content = system_prompt

        return [Message(role="system", content=content)]


class Qwen3_5DisableThinkingRenderer(Qwen3_5Renderer):
    """
    Renderer for Qwen3.5 models with thinking disabled.

    Matches the Qwen3.5 HF template with ``enable_thinking=False``. The only
    difference from :class:`Qwen3_5Renderer` is the generation suffix:
    ``<think>\\n\\n</think>\\n\\n`` instead of ``<think>\\n``, signalling to the
    model that it should respond directly without reasoning.
    """

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n<think>\n\n</think>\n\n"
        return self.tokenizer.encode(header_str, add_special_tokens=False)

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Disable-thinking mode prefills an empty think block, so no fixup needed."""
        return response
