"""Text renderer for Muse Glimmer's ATEM chat format."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import cast

import torch
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.image_processing_utils import ImageProcessor
from interactive_training.renderers.base import (
    ImageProcessorProtocol,
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    ensure_list,
    ensure_text,
    get_image_source,
)
from interactive_training.tokenizer_utils import Tokenizer

_EOT = "<|eot|>"
_EOM = "<|eom|>"
_MESSAGE = "<|message|>"
_START = "<|start|>"

_ATEM_INVOKE_RE = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_ATEM_PARAMETER_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)">(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)


def _render_content(message: Message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    unsupported_types = {
        part["type"] for part in content if part["type"] not in {"text", "thinking"}
    }
    if unsupported_types:
        raise ValueError(
            "Muse Glimmer assistant messages do not support content types: "
            + ", ".join(sorted(unsupported_types))
        )
    return "".join(part["text"] for part in content if part["type"] == "text")


def _thinking_content(message: Message) -> str:
    return "".join(
        part["thinking"]
        for part in ensure_list(message["content"])
        if part["type"] == "thinking"
    )


def _render_atem_call(tool_call: ToolCall) -> str:
    arguments = json.loads(tool_call.function.arguments or "{}")
    lines = [
        "<atem:function_calls>",
        f'<atem:invoke name="{tool_call.function.name}">',
    ]
    for name, value in arguments.items():
        if isinstance(value, (dict, list)):
            rendered_value = json.dumps(value, separators=(",", ":"))
        elif value is None:
            rendered_value = "null"
        elif isinstance(value, bool):
            rendered_value = str(value).lower()
        else:
            rendered_value = str(value)
        lines.append(f'<atem:parameter name="{name}">{rendered_value}</atem:parameter>')
    lines.extend(["</atem:invoke>", "</atem:function_calls>"])
    return "\n".join(lines)


def _parse_atem_call(raw: str) -> ToolCall | UnparsedToolCall:
    invoke = _ATEM_INVOKE_RE.search(raw)
    if invoke is None:
        return UnparsedToolCall(raw_text=raw, error="Malformed Muse Glimmer ATEM call")
    arguments: dict[str, object] = {}
    for parameter in _ATEM_PARAMETER_RE.finditer(invoke.group("body")):
        value = parameter.group("value")
        try:
            parsed_value: object = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        arguments[parameter.group("name")] = parsed_value
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=invoke.group("name"),
            arguments=json.dumps(arguments),
        )
    )


class MuseGlimmerRenderer(Renderer):
    """Render text conversations using Muse Glimmer's official ATEM template."""

    reasoning_strength = "high"

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor | None = None,
        reasoning_strength: str = "high",
        current_date: str | None = None,
    ):
        super().__init__(tokenizer)
        self.image_processor = image_processor
        if reasoning_strength not in {"low", "medium", "high", "xhigh"}:
            raise ValueError(
                f"Unsupported Muse Glimmer reasoning strength: {reasoning_strength}"
            )
        self.reasoning_strength = reasoning_strength
        self.current_date = current_date or datetime.now().strftime("%Y-%m-%d")

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def has_extension_property(self) -> bool:
        return True

    def _system_suffix(self) -> str:
        return (
            f"\n\nReasoning strength: {self.reasoning_strength}."
            '\n\n# Valid recipients: "self", "user".'
            f"{_EOT}"
        )

    def _default_system_content(self) -> str:
        return (
            "You are a helpful AI assistant."
            "\nKnowledge cutoff: 2026-01-04."
            f"\nCurrent date: {self.current_date}."
        )

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        if any(message["role"] == "system" for message in messages):
            return messages
        return [
            Message(role="system", content=self._default_system_content()),
            *messages,
        ]

    def get_stop_sequences(self) -> list[str]:
        return [_EOT]

    def _render_content(self, message: Message) -> list[ModelInputChunk]:
        content = message["content"]
        if isinstance(content, str):
            return [
                ModelInputChunk(
                    tokens=self.tokenizer.encode(content, add_special_tokens=False)
                )
            ]

        chunks: list[ModelInputChunk] = []
        for part in content:
            if part["type"] == "text":
                chunks.append(
                    ModelInputChunk(
                        tokens=self.tokenizer.encode(
                            part["text"], add_special_tokens=False
                        )
                    )
                )
            elif part["type"] in {"image", "image_url"}:
                if self.image_processor is None:
                    raise ValueError(
                        "Muse Glimmer image inputs require an image processor"
                    )
                chunks.extend(
                    [
                        ModelInputChunk(
                            tokens=self.tokenizer.encode(
                                "<|image_start|>", add_special_tokens=False
                            )
                        ),
                        self.image_to_chunk(
                            get_image_source(part),
                            cast(ImageProcessorProtocol, self.image_processor),
                        ),
                        ModelInputChunk(
                            tokens=self.tokenizer.encode(
                                "<|image_end|>", add_special_tokens=False
                            )
                        ),
                    ]
                )
        return chunks

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]

        if role == "system":
            header_text = f"{_START}system{_MESSAGE}"
            system_content = ensure_text(message["content"])
            if "# Valid recipients:" in system_content:
                output_text = system_content + _EOT
            else:
                output_text = system_content + self._system_suffix()
        elif role == "user":
            header_text = f"{_START}user{_MESSAGE}"
            output_chunks = self._render_content(message)
            output_chunks.append(
                ModelInputChunk(
                    tokens=self.tokenizer.encode(_EOT, add_special_tokens=False)
                )
            )
        elif role == "tool":
            tool_name = message.get("name") or message.get("tool_call_id") or ""
            header_text = f"{_START}tool {tool_name}{_MESSAGE}"
            output_chunks = [
                ModelInputChunk(
                    tokens=self.tokenizer.encode(
                        f'<tool_output name="{tool_name}">\n',
                        add_special_tokens=False,
                    )
                ),
                *self._render_content(message),
                ModelInputChunk(
                    tokens=self.tokenizer.encode(
                        f"\n</tool_output>{_EOT}", add_special_tokens=False
                    )
                ),
            ]
        elif role == "assistant":
            header_text = f"{_START}assistant"
            thinking = _thinking_content(message)
            tool_calls = message.get("tool_calls") or []
            segments: list[str] = []
            if thinking:
                segments.append(f" to=self{_MESSAGE}{thinking}{_EOM}{_START}assistant")
            if tool_calls:
                for index, tool_call in enumerate(tool_calls):
                    terminator = _EOT if index == len(tool_calls) - 1 else _EOM
                    segments.append(
                        f" to={tool_call.function.name}{_MESSAGE}"
                        f"{_render_atem_call(tool_call)}{terminator}"
                    )
                    if index != len(tool_calls) - 1:
                        segments.append(f"{_START}assistant")
            else:
                assistant_content = _render_content(message)
                segments.append(f" to=user{_MESSAGE}{assistant_content}{_EOT}")
            output_text = "".join(segments)
        else:
            raise ValueError(f"Unsupported Muse Glimmer message role: {role}")

        if role not in {"user", "tool"}:
            output_chunks = [
                ModelInputChunk(
                    tokens=self.tokenizer.encode(output_text, add_special_tokens=False)
                )
            ]
        return RenderedMessage(
            header=ModelInputChunk(
                tokens=self.tokenizer.encode(header_text, add_special_tokens=False)
            ),
            output=output_chunks,
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> ModelInput:
        messages = self._ensure_system_message(messages)
        return super().build_generation_prompt(messages, role=role, prefill=prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[ModelInput, torch.Tensor]:
        messages = self._ensure_system_message(messages)
        return super().build_supervised_example(messages, train_on_what=train_on_what)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        if role != "assistant":
            raise ValueError("Muse Glimmer only supports assistant generation")
        return self.tokenizer.encode(f"{_START}assistant", add_special_tokens=False)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        text = self.tokenizer.decode(response)
        success = text.endswith(_EOT)
        if success:
            text = text[: -len(_EOT)]

        thinking = ""
        if text.startswith(f" to=self{_MESSAGE}"):
            reasoning_end = text.find(f"{_EOM}{_START}assistant")
            if reasoning_end < 0:
                return Message(role="assistant", content=text), False
            thinking = text[len(f" to=self{_MESSAGE}") : reasoning_end]
            text = text[reasoning_end + len(f"{_EOM}{_START}assistant") :]

        recipient = re.match(
            r" to=(?P<recipient>[^<]+)<\|message\|>(?P<body>.*)", text, re.DOTALL
        )
        if recipient is None:
            return Message(role="assistant", content=text), False

        recipient_name = recipient.group("recipient")
        body = recipient.group("body")
        content: str | list[TextPart | ThinkingPart]
        if thinking:
            content = [ThinkingPart(type="thinking", thinking=thinking)]
            if recipient_name == "user":
                content.append(TextPart(type="text", text=body))
        else:
            content = body if recipient_name == "user" else ""

        message = Message(role="assistant", content=content)
        if recipient_name != "user":
            parsed_call = _parse_atem_call(body)
            if isinstance(parsed_call, ToolCall):
                message["tool_calls"] = [parsed_call]
            else:
                message["unparsed_tool_calls"] = [parsed_call]
                success = False
        return message, success

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        namespaces = list(
            dict.fromkeys(tool["name"].split(".", 1)[0] for tool in tools)
        )
        recipients = [
            '"self"',
            *(f'"{namespace}.*"' for namespace in namespaces),
            '"user"',
        ]
        content = system_prompt or "You are a helpful AI assistant."
        content += f"\n\nReasoning strength: {self.reasoning_strength}."
        if tools:
            schemas = "\n".join(
                json.dumps(tool, separators=(",", ":")) for tool in tools
            )
            content += (
                "\n\nIn this environment you have access to a set of tools you can use "
                "to answer the user's question.\n\n"
                "Invoke a function with this format:\n"
                "<atem:function_calls>\n"
                '<atem:invoke name="$FUNCTION_NAME">\n'
                '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n'
                "</atem:invoke>\n"
                "</atem:function_calls>\n\n"
                "Here are the functions available in JSONSchema format:\n"
                f"{schemas}"
            )
        content += f"\n\n# Valid recipients: {', '.join(recipients)}."
        return [Message(role="system", content=content)]

    def to_openai_message(self, message: Message) -> dict:
        result = super().to_openai_message(message)
        thinking = _thinking_content(message)
        if thinking:
            result["reasoning_content"] = thinking
            result["content"] = _render_content(message)
        for tool_call in result.get("tool_calls", []):
            arguments = tool_call["function"].get("arguments")
            if isinstance(arguments, str):
                tool_call["function"]["arguments"] = json.loads(arguments)
        return result
