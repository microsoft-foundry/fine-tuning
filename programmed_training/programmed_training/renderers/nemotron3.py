"""Nemotron-3 Ultra text rendering, adapted to the private preview cookbook.

Source: thinking-machines-lab/tinker-cookbook, renderers/nemotron3.py
at ad3b5c541cd383b05d1c301e8ccc2d9cc0c5f774.
Copyright 2025 Thinking Machines Lab.
Modified for the cookbook's Azure SDK inputs, Ultra-only text support and
conservative sequence-extension behavior. Licensed under Apache-2.0; a copy
is included at programmed_training/LICENSE-NEMOTRON3-APACHE-2.0.txt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

import torch
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from programmed_training.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Role,
    ToolSpec,
    TrainOnWhat,
)
from programmed_training.renderers.qwen3_5 import Qwen3_5Renderer, _format_tool_call_xml
from programmed_training.tokenizer_utils import Tokenizer


def _xml_fields(fields: Mapping[str, object], excluded: set[str]) -> list[str]:
    return [
        f"<{key}>"
        + (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else str(value)
        )
        + f"</{key}>"
        for key, value in fields.items()
        if key not in excluded
    ]


class Nemotron3UltraRenderer(Qwen3_5Renderer):
    """Ultra's native text format, with full reasoning and truncated history."""

    disables_thinking = False
    _thinking_text_separator = ""

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = True,
        reasoning_effort: Literal["full", "medium"] = "full",
    ):
        if reasoning_effort not in {"full", "medium"}:
            raise ValueError(
                f"Unsupported Nemotron Ultra reasoning effort: {reasoning_effort!r}"
            )
        super().__init__(tokenizer, strip_thinking_from_history=strip_thinking_from_history)
        self.reasoning_effort = reasoning_effort

    @property
    def has_extension_property(self) -> bool:
        # Empty thinking and assistant whitespace can change when sampled text
        # is re-rendered, even when history thinking is preserved.
        return False

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        if messages and messages[0]["role"] == "system":
            return list(messages)
        system = Message(role="system", content="")
        if any("trainable" in message for message in messages):
            system["trainable"] = False
        return [system, *messages]

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> ModelInput:
        return super().build_generation_prompt(
            self._normalize_messages(messages), role=role, prefill=prefill
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[ModelInput, torch.Tensor]:
        return super().build_supervised_example(
            self._normalize_messages(messages), train_on_what=train_on_what
        )

    def _message_content(self, message: Message) -> tuple[str, bool]:
        content = message["content"] or ""
        if isinstance(content, list):
            if any(
                part["type"] not in {"text", "thinking"}
                or (part["type"] == "thinking" and message["role"] != "assistant")
                for part in content
            ):
                raise ValueError(
                    "Nemotron Ultra supports text and assistant thinking content only"
                )
            thinking = "".join(
                part["thinking"] for part in content if part["type"] == "thinking"
            )
            text = "".join(part["text"] for part in content if part["type"] == "text")
            content = f"<think>\n{thinking}</think>{text}" if thinking.strip() else text
        if (
            message["role"] == "assistant"
            and "<think>" not in content
            and "</think>" not in content
        ):
            return "<think></think>" + content, True
        return content, False

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        content, injected_thinking = self._message_content(message)
        role = message["role"]
        previous_tool = ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
        separator = "\n" if ctx.idx else ""
        if previous_tool:
            separator = "" if role == "tool" else "<|im_end|>\n"
        rendered_role = "user" if role == "tool" else role
        header = (
            ""
            if role == "tool" and previous_tool
            else f"{separator}<|im_start|>{rendered_role}\n"
        )

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            historical = self.strip_thinking_from_history and ctx.idx < ctx.last_user_index
            if historical and tool_calls:
                if "</think>" in content:
                    content = content.split("</think>")[-1]
                elif "<think>" in content:
                    content = content.split("<think>")[0]
                content = "<think></think>" + content
                injected_thinking = True
            elif historical and "<think>" in content and "</think>" in content:
                content = "<think></think>" + content.split("</think>")[-1]
                injected_thinking = True
            if not (historical and tool_calls):
                content = content.strip()
            if injected_thinking:
                header += "<think></think>"
                content = content[len("<think></think>") :]
            if tool_calls:
                content += "\n" + "".join(
                    _format_tool_call_xml(call, self._format_tool_call_argument) + "\n"
                    for call in tool_calls
                )
        elif (
            role == "user"
            and ctx.idx == ctx.last_user_index
            and self.reasoning_effort == "medium"
        ):
            content += "\n\n{reasoning effort: efficient}"
        elif role == "tool":
            content = f"<tool_response>\n{content}\n</tool_response>\n"

        # Consecutive tool results share a user block. Its closing token belongs
        # to the next non-tool header, or to the last tool output at end of input.
        if role != "tool" or ctx.is_last:
            content += "<|im_end|>"
        return RenderedMessage(
            header=ModelInputChunk(
                tokens=self.tokenizer.encode(header, add_special_tokens=False)
            ),
            output=[
                ModelInputChunk(
                    tokens=self.tokenizer.encode(content, add_special_tokens=False)
                )
            ],
        )

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        separator = "\n" if ctx.idx else ""
        suffix = ""
        if role == "assistant":
            suffix = "<think></think>" if self.disables_thinking else "<think>\n"
        return self.tokenizer.encode(
            f"{separator}<|im_start|>{role}\n{suffix}", add_special_tokens=False
        )

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        return (
            response
            if self.disables_thinking
            else super()._normalize_response_tokens(response)
        )

    def _serialize_tool_declaration(self, tool: ToolSpec) -> str:
        fields = ["<function>", f"<name>{tool['name']}</name>"]
        if "description" in tool:
            fields.append(f"<description>{tool['description'].strip()}</description>")
        fields.append("<parameters>")
        parameters = tool.get("parameters") or {}
        for name, schema in parameters.get("properties", {}).items():
            fields.extend(["<parameter>", f"<name>{name}</name>"])
            if "type" in schema:
                fields.append(f"<type>{schema['type']}</type>")
            if "description" in schema:
                fields.append(f"<description>{schema['description'].strip()}</description>")
            if "enum" in schema:
                fields.append(f"<enum>{json.dumps(schema['enum'], ensure_ascii=False)}</enum>")
            fields.extend(_xml_fields(schema, {"name", "type", "description", "enum"}))
            fields.append("</parameter>")
        fields.extend(_xml_fields(parameters, {"type", "properties", "required"}))
        if "required" in parameters:
            required = json.dumps(parameters["required"], ensure_ascii=False)
            fields.append(f"<required>{required}</required>")
        fields.append("</parameters>")
        fields.extend(_xml_fields(tool, {"type", "name", "description", "parameters"}))
        fields.append("</function>")
        return "\n".join(fields)

    def _format_tool_call_argument(self, param_value: object) -> str:
        if isinstance(param_value, (dict, list)):
            return json.dumps(param_value, ensure_ascii=False)
        return str(param_value)

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        tools_message = super().create_conversation_prefix_with_tools(tools)[0]
        tools_text = tools_message["content"]
        assert isinstance(tools_text, str)
        content = system_prompt
        if tools_text:
            content += ("\n\n" if system_prompt else "") + tools_text
        return [Message(role="system", content=content)]


class Nemotron3UltraMediumThinkingRenderer(Nemotron3UltraRenderer):
    """Use Ultra's medium-effort instruction on the last user message."""

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, reasoning_effort="medium")


class Nemotron3UltraDisableThinkingRenderer(Nemotron3UltraRenderer):
    """Prefill a closed empty thinking block; supplied SFT reasoning is unchanged."""

    disables_thinking = True


class Nemotron3UltraPreserveThinkingRenderer(Nemotron3UltraRenderer):
    """Match truncate_history_thinking=False without claiming prefix identity."""

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, strip_thinking_from_history=False)
