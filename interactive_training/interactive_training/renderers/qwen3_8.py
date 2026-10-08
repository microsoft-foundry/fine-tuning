"""Renderers for Qwen3.8 models.

Qwen3.8 shares Qwen3.5/3.6's architecture (``model_type: qwen3_5``), tokenizer,
special tokens and XML tool-calling format, but its chat template differs in
four ways that are load-bearing for train/serve alignment:

1. **Reasoning effort.** When thinking is enabled the template resolves a
   ``reasoning_effort`` kwarg (``xhigh`` (default), ``medium``, ``low``).
   ``xhigh`` and ``low`` prepend an instruction sentence to the system message,
   creating one if the conversation has none; ``medium`` adds nothing.
2. **Thinking is preserved in history by default.** ``preserve_thinking``
   defaults to true, so *every* assistant message keeps a ``<think>`` block --
   empty when that turn did not reason, and regardless of its position. Qwen3.5
   and 3.6 stripped history thinking and only framed assistants after the last
   user message.
3. **An empty system message renders nothing**, where Qwen3.5/3.6 emitted an
   empty ``<|im_start|>system`` block.
4. **Tool-call arguments are serialized with ``tojson``** unless they are
   strings, so booleans and null render as ``true``/``null`` rather than
   Qwen3.5's Python-style ``True``/``None``.

For scale: Qwen3.6's template differs from Qwen3.5's by 4 lines, which is why
both share the ``qwen3_5`` renderer. Qwen3.8 differs by 34.

Reference: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja
Mirrors thinking-machines-lab/tinker-cookbook ``renderers/qwen3_8.py``.
"""

from __future__ import annotations

import json

import torch

from interactive_training.renderers.base import (
    Message,
    ModelInput,
    RenderContext,
    Role,
    TextPart,
    Tokenizer,
    ToolSpec,
    TrainOnWhat,
)
from interactive_training.renderers.qwen3_5 import Qwen3_5Renderer

# Instruction sentences the HF template injects into the system message per
# reasoning effort. ``medium`` deliberately maps to no instruction: the template
# accepts it but has no branch emitting text for it. Matching that omission is
# what keeps token-exact parity, so it is not normalised away here.
REASONING_EFFORT_INSTRUCTIONS: dict[str, str] = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
        "directly to the conclusion without unnecessary elaboration."
    ),
}

DEFAULT_REASONING_EFFORT = "xhigh"


class Qwen3_8Renderer(Qwen3_5Renderer):
    """Renderer for Qwen3.8 models with thinking enabled.

    Subclasses :class:`Qwen3_5Renderer` for the shared im_start/im_end/thinking
    and XML tool-call infrastructure, overriding only what the Qwen3.8 template
    changes.
    """

    #: Set by :class:`Qwen3_8DisableThinkingRenderer`; suppresses the reasoning
    #: instruction, which the template computes only when thinking is enabled.
    disables_thinking = False

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = False,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ):
        """
        Args:
            tokenizer: The tokenizer to render with.
            strip_thinking_from_history: Defaults to ``False`` to match the HF
                template's ``preserve_thinking=true``. This is the opposite of
                the Qwen3/Qwen3.5 default. Set ``True`` for Qwen3.5/3.6-style
                positional stripping (HF ``preserve_thinking=false``).
            reasoning_effort: One of ``xhigh`` (default), ``medium`` or ``low``.
                Ignored when thinking is disabled.
        """
        if reasoning_effort not in REASONING_EFFORT_INSTRUCTIONS:
            raise ValueError(
                f"Unexpected reasoning effort {reasoning_effort}. Supported types are "
                "xhigh (default), medium, and low."
            )
        super().__init__(
            tokenizer, strip_thinking_from_history=strip_thinking_from_history
        )
        self.reasoning_effort = reasoning_effort

    @property
    def _reasoning_instructions(self) -> str:
        """The instruction sentence for this effort, empty when thinking is off."""
        if self.disables_thinking:
            return ""
        return REASONING_EFFORT_INSTRUCTIONS[self.reasoning_effort]

    @property
    def has_extension_property(self) -> bool:
        """Qwen3.8 cannot claim the extension property despite preserving thinking.

        A turn that did not reason is sampled after the prompt's open
        ``<think>\\n``, so its tokens continue with a lone ``\\n``; history
        instead writes the closed empty block, whose ``\\n\\n`` the tokenizer
        merges into a single token. The sampled sequence is therefore not a
        token-level prefix of the next prompt.

        Note that :meth:`Renderer.build_supervised_examples` (plural) raises
        ``NotImplementedError`` when this is ``False`` -- there is no
        per-message-splitting fallback in the base class, only a ``TODO``.
        Callers that need multiple examples from one conversation must split it
        themselves; the singular :meth:`build_supervised_example` is unaffected
        and is what every cookbook recipe actually uses.

        This overrides :class:`Qwen3Renderer`'s derivation from
        ``strip_thinking_from_history``, which would otherwise report ``True``
        here purely because thinking is preserved.
        """
        return False

    # ------------------------------------------------------------------
    # System-message normalization
    # ------------------------------------------------------------------

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        """Apply the template's system-message handling to the conversation.

        Prepends the reasoning instruction to the system message (the template
        places it before both tool declarations and user-provided content), adds
        a system message carrying just the instruction when the conversation has
        none, and drops an empty system message the template renders nothing for.

        This has to happen during *rendering* rather than only in
        :meth:`create_conversation_prefix_with_tools`, because RL rollouts and
        supervised examples hand the renderer a bare message list. Injecting only
        in the prefix helper would train on a prompt the serving stack never
        produces.

        Idempotent, so it is safe if a caller has already normalized. Rebuilt
        messages keep the original's other fields (e.g. ``trainable`` for
        CUSTOMIZED training), and a synthetic message carries ``trainable=False``
        so the base renderer's all-or-nothing ``trainable`` assertion holds.
        """
        instructions = self._reasoning_instructions

        if messages and messages[0]["role"] == "system":
            first, rest = messages[0], list(messages[1:])
            content = first["content"]

            if isinstance(content, list):
                texts = [p["text"] for p in content if p["type"] == "text"]
                if len(texts) == len(content):
                    # render_content concatenates text parts and applies |trim,
                    # so a text-only list renders exactly like the joined string.
                    content = "".join(texts)
                else:
                    # Non-text parts (the HF template rejects these in system
                    # messages): prepend the instruction and pass through.
                    if not instructions:
                        return list(messages)
                    parts = list(content)
                    if parts[0]["type"] == "text" and parts[0]["text"].startswith(
                        instructions
                    ):
                        return list(messages)
                    parts = [TextPart(type="text", text=f"{instructions}\n\n")] + parts
                    return [Message({**first, "content": parts})] + rest

            content = content.strip()
            if not content and not instructions:
                return rest
            if instructions and not (
                content == instructions or content.startswith(f"{instructions}\n\n")
            ):
                content = f"{instructions}\n\n{content}" if content else instructions
            return [Message({**first, "content": content})] + rest

        if instructions:
            synthetic = Message(role="system", content=instructions)
            # CUSTOMIZED training requires a trainable field on every message,
            # while every other mode requires its absence -- mirror the input.
            if any("trainable" in message for message in messages):
                synthetic["trainable"] = False
            return [synthetic] + list(messages)

        return list(messages)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
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

    # ------------------------------------------------------------------
    # Message rendering
    # ------------------------------------------------------------------

    def _format_tool_call_argument(self, param_value: object) -> str:
        """Qwen3.8 (like 3.6) serializes every non-string value with ``tojson``.

        Booleans and null therefore render as JSON ``true``/``null``, not the
        Python-style ``True``/``None`` the older Qwen3.5 template produced.

        ``ensure_ascii=False`` is required for token-exactness, not cosmetic:
        transformers replaces Jinja's built-in ``tojson`` with its own filter
        that defaults to ``ensure_ascii=False``
        (``transformers/utils/chat_template_utils.py``). Python's default is
        ``True``, so a structured argument such as ``{"tags": ["\u4e2d\u6587"]}``
        would serialize to ``["\\u4e2d\\u6587"]`` here but ``["\u4e2d\u6587"]``
        in the template -- a different token sequence for the same conversation,
        i.e. exactly the train/serve skew this renderer exists to prevent.
        """
        if isinstance(param_value, str):
            return param_value
        return json.dumps(param_value, ensure_ascii=False)

    def _serialize_tool_declaration(self, tool: ToolSpec) -> str:
        """Match the template's ``tojson``, which does not escape non-ASCII.

        Same reasoning as :meth:`_format_tool_call_argument`: Qwen3.8 is not yet
        in production, so unlike Qwen3.5/3.6 it can adopt the token-exact form.
        """
        return json.dumps(tool, ensure_ascii=False)

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Insert an empty think block for assistant messages that did not reason.

        With thinking preserved in history (the default) the HF template frames
        *every* assistant message, so unlike Qwen3.5 this is independent of the
        message's position. With ``strip_thinking_from_history=True`` (HF
        ``preserve_thinking=false``) the template falls back to Qwen3.5's
        positional rule.

        Only structured thinking parts count as reasoning: Qwen3.8's template no
        longer extracts inline ``<think>`` tags from string content -- it reads
        ``reasoning_content`` only -- so an inline block is ordinary content and
        the empty framing block is still written before it.
        """
        if message["role"] != "assistant":
            return ""
        if self.strip_thinking_from_history and ctx.idx <= ctx.last_user_index:
            return ""
        content = message.get("content", "")
        has_reasoning = isinstance(content, list) and any(
            part["type"] == "thinking" for part in content
        )
        return "" if has_reasoning else "<think>\n\n</think>\n\n"


class Qwen3_8DisableThinkingRenderer(Qwen3_8Renderer):
    """Renderer for Qwen3.8 models with thinking disabled.

    Matches the Qwen3.8 HF template with ``enable_thinking=False``: the
    generation suffix closes the think block and no reasoning-effort instruction
    is injected. History framing is unchanged -- ``preserve_thinking`` is
    independent of ``enable_thinking``, so historical assistant messages still
    carry their (possibly empty) think blocks.
    """

    disables_thinking = True

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n<think>\n\n</think>\n\n"
        return self.tokenizer.encode(header_str, add_special_tokens=False)

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Disable-thinking mode prefills an empty think block, so no fixup needed."""
        return response


__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "REASONING_EFFORT_INSTRUCTIONS",
    "Qwen3_8DisableThinkingRenderer",
    "Qwen3_8Renderer",
]
