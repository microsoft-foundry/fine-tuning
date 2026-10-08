"""Renderer for zai-org/GLM-5.2 and GLM-5.2-FP8 (``GlmMoeDsaForCausalLM``).

GLM-5.2 chat format (token-exact with the model's ``chat_template.jinja``)::

    [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{question}<|assistant|><think>{reasoning}</think>{answer}

Key properties, all verified against ``apply_chat_template`` in the conformance
test:

* BOS is the literal ``[gMASK]<sop>`` pair.
* When thinking is enabled (the default) a single ``<|system|>Reasoning Effort:
  Max`` line is injected once, *before* any user-supplied system message.  This
  fixed prefix is folded into :pyattr:`_bos_tokens` (which receives zero training
  weight), exactly where the template emits it.
* Roles use single special tokens: ``<|user|>``, ``<|assistant|>``,
  ``<|system|>``, ``<|observation|>``.  There is **no** end-of-message token; an
  assistant turn is delimited by the next role token (or an EOS token at
  generation time).
* Historical assistant turns (anything at or before the last user message) have
  their thinking collapsed to ``<think></think>``; only the turn after the last
  user message keeps ``<think>{reasoning}</think>``.
* The generation prompt prefills ``<think>`` (thinking on) or ``<think></think>``
  (thinking off), so a *sampled* response begins with the reasoning text directly
  and the open ``<think>`` tag must be re-inserted when parsing.
* Sampling stops on any of the model's EOS ids: ``<|endoftext|>``, ``<|user|>``,
  ``<|observation|>`` (from ``generation_config.eos_token_id``).
"""

from __future__ import annotations

import json
from typing import Iterator

from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.renderers.base import (
    Message,
    MessageDelta,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ensure_list,
    ensure_text,
    parse_think_blocks,
)
from interactive_training.tokenizer_utils import Tokenizer

#: EOS ids GLM-5.2 sampling stops on (``generation_config.eos_token_id``).
_STOP_TOKEN_STRS = ("<|endoftext|>", "<|user|>", "<|observation|>")


class GLM5Renderer(Renderer):
    """GLM-5.2 thinking renderer (reasoning effort ``max`` by default)."""

    #: Whether the ``<think>`` reasoning channel is enabled.
    enable_thinking: bool = True
    #: ``"max"`` (default) or ``"high"`` — controls the injected reasoning-effort line.
    reasoning_effort: str = "max"

    # -- prompt prefix ---------------------------------------------------- #
    def _reasoning_effort_word(self) -> str:
        return "High" if self.reasoning_effort == "high" else "Max"

    @property
    def _bos_tokens(self) -> list[int]:
        prefix = "[gMASK]<sop>"
        if self.enable_thinking and self.reasoning_effort is not None:
            prefix += f"<|system|>Reasoning Effort: {self._reasoning_effort_word()}"
        return self.tokenizer.encode(prefix, add_special_tokens=False)

    # -- stop tokens ------------------------------------------------------ #
    def _stop_token_ids(self) -> list[int]:
        ids: list[int] = []
        for tok_str in _STOP_TOKEN_STRS:
            encoded = self.tokenizer.encode(tok_str, add_special_tokens=False)
            assert len(encoded) == 1, f"Expected single token for {tok_str!r}, got {encoded}"
            ids.append(encoded[0])
        return ids

    def get_stop_sequences(self) -> list[int]:
        return self._stop_token_ids()

    # -- per-message rendering ------------------------------------------- #
    def _render_assistant_output(self, message: Message, ctx: RenderContext) -> str:
        parts = ensure_list(message["content"])
        thinking = "".join(p["thinking"] for p in parts if p["type"] == "thinking")
        text = "".join(p["text"] for p in parts if p["type"] == "text")

        # Keep reasoning only for the turn after the last user message; collapse
        # everything earlier to an empty think block (matches the jinja template's
        # ``loop.index0 > ns.last_user_index`` guard).
        keep_thinking = ctx.idx > ctx.last_user_index
        if keep_thinking and thinking:
            out = f"<think>{thinking}</think>"
        else:
            out = "<think></think>"
        if text.strip():
            out += text.strip()

        for tool_call in message.get("tool_calls") or []:
            out += f"<tool_call>{tool_call.function.name}"
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            for key, value in args.items():
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                out += f"<arg_key>{key}</arg_key><arg_value>{rendered}</arg_value>"
            out += "</tool_call>"
        return out

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]

        if role == "user":
            header_str, output_str = "<|user|>", ensure_text(message["content"])
        elif role == "system":
            header_str, output_str = "<|system|>", ensure_text(message["content"])
        elif role == "assistant":
            header_str = "<|assistant|>"
            output_str = self._render_assistant_output(message, ctx)
        elif role in ("tool", "observation"):
            prev_role = ctx.prev_message["role"] if ctx.prev_message else None
            header_str = "<|observation|>" if prev_role not in ("tool", "observation") else ""
            output_str = f"<tool_response>{ensure_text(message['content'])}</tool_response>"
        else:
            header_str, output_str = "<|system|>", ensure_text(message["content"])

        header = (
            ModelInputChunk(tokens=self.tokenizer.encode(header_str, add_special_tokens=False))
            if header_str
            else None
        )
        output = (
            [ModelInputChunk(tokens=self.tokenizer.encode(output_str, add_special_tokens=False))]
            if output_str
            else []
        )
        return RenderedMessage(header=header, output=output)

    # -- generation prompt ------------------------------------------------ #
    def _default_prefill(self) -> str:
        return "<think>" if self.enable_thinking else "<think></think>"

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> ModelInput:
        if prefill is None:
            prefill = self._default_prefill()
        return super().build_generation_prompt(messages, role=role, prefill=prefill)

    # -- response parsing ------------------------------------------------- #
    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        stop_ids = set(self._stop_token_ids())
        stop_idx = next((i for i, tok in enumerate(response) if tok in stop_ids), None)
        if stop_idx is None:
            # No stop token: malformed/truncated sample -> format error.
            return Message(role="assistant", content=self.tokenizer.decode(response)), False

        text = self.tokenizer.decode(response[:stop_idx])
        # The prompt prefilled an open ``<think>`` that the sampler does not echo;
        # re-insert it so parse_think_blocks recovers the reasoning content. Guard
        # against the rare case where the sampler *did* echo it (no double tag).
        if self.enable_thinking and not text.lstrip().startswith("<think>"):
            text = "<think>" + text

        parts = parse_think_blocks(text)
        content = parts if parts is not None else text
        return Message(role="assistant", content=content), True

    def parse_response_streaming(self, response: list[int]) -> Iterator[MessageDelta]:
        message, _ = self.parse_response(response)
        yield message

    def to_openai_message(self, message: Message) -> dict:
        result = super().to_openai_message(message)
        content = message["content"]
        if isinstance(content, str):
            return result
        thinking = [p["thinking"] for p in content if p["type"] == "thinking"]
        text = [p["text"] for p in content if p["type"] == "text"]
        result["content"] = "".join(text)
        if thinking:
            result["reasoning_content"] = "".join(thinking)
        return result


class GLM5HighReasoningRenderer(GLM5Renderer):
    """GLM-5.2 with ``Reasoning Effort: High`` instead of ``Max``."""

    reasoning_effort = "high"


class GLM5DisableThinkingRenderer(GLM5Renderer):
    """GLM-5.2 with the ``<think>`` channel disabled (``<think></think>`` prefill)."""

    enable_thinking = False

    @property
    def _bos_tokens(self) -> list[int]:
        # No reasoning-effort line when thinking is disabled.
        return self.tokenizer.encode("[gMASK]<sop>", add_special_tokens=False)
