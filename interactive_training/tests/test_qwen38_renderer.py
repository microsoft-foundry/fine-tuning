"""Qwen3.8 renderer behaviours the HF token-parity suite cannot reach.

``test_qwen35_hf_tool_parity.py`` pins Qwen3.8's rendering against the real HF
template. The properties here are either invisible to that comparison (the
extension property is a statement about *pairs* of renderings) or concern inputs
the template rejects outright (``reasoning_effort`` validation), so they need
direct coverage.
"""

from __future__ import annotations

import pytest
from interactive_training.model_info import get_recommended_renderer_names
from interactive_training.renderers import get_renderer
from interactive_training.renderers.base import Message, TrainOnWhat
from interactive_training.renderers.qwen3_8 import (
    REASONING_EFFORT_INSTRUCTIONS,
    Qwen3_8DisableThinkingRenderer,
    Qwen3_8Renderer,
)
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def _text(renderer, messages) -> str:
    model_input = renderer.build_generation_prompt(messages)
    tokens = [token for chunk in model_input.chunks for token in chunk.tokens]
    return renderer.tokenizer.decode(tokens)


def test_thinking_is_preserved_in_history_by_default(tokenizer):
    """HF ``preserve_thinking`` defaults to true, unlike Qwen3.5/3.6."""
    renderer = Qwen3_8Renderer(tokenizer)
    assert renderer.strip_thinking_from_history is False


def test_extension_property_is_not_claimed(tokenizer):
    """Preserving thinking is not sufficient for the extension property.

    ``Qwen3Renderer`` derives ``has_extension_property`` as
    ``not strip_thinking_from_history``, which would report True here. A turn
    that did not reason is sampled after an open ``<think>\\n`` but recorded in
    history as a closed empty block, so the sampled tokens are not a prefix of
    the next prompt.
    """
    assert Qwen3_8Renderer(tokenizer).has_extension_property is False
    assert Qwen3_8DisableThinkingRenderer(tokenizer).has_extension_property is False


def test_supervised_example_still_builds_without_the_extension_property(tokenizer):
    """The override must be load-bearing, and must not break the singular path.

    Every Qwen renderer in Interactive Training reports ``has_extension_property=False``, but for
    Qwen3.5 that falls out of ``not strip_thinking_from_history``. Qwen3.8 keeps
    thinking, so the same derivation would report *True* -- which is why the
    override exists. This pins both halves: the naive value differs from the
    reported one, and ``build_supervised_example`` still works.
    """
    renderer = Qwen3_8Renderer(tokenizer)
    naive = not renderer.strip_thinking_from_history
    assert naive is True
    assert renderer.has_extension_property is False

    messages: list[Message] = [
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "One."},
    ]
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    tokens = [token for chunk in model_input.chunks for token in chunk.tokens]
    assert len(weights) == len(tokens)
    assert weights.sum().item() > 0


@pytest.mark.parametrize("effort", sorted(REASONING_EFFORT_INSTRUCTIONS))
def test_every_supported_reasoning_effort_is_accepted(tokenizer, effort):
    renderer = Qwen3_8Renderer(tokenizer, reasoning_effort=effort)
    text = _text(renderer, [{"role": "user", "content": "Hi."}])
    instruction = REASONING_EFFORT_INSTRUCTIONS[effort]
    if instruction:
        assert instruction in text
    else:
        # 'medium' is accepted by the template but emits no instruction, and
        # with no system message of its own the conversation gets no system
        # block at all.
        assert "<|im_start|>system" not in text


def test_unsupported_reasoning_effort_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="Unexpected reasoning effort"):
        Qwen3_8Renderer(tokenizer, reasoning_effort="high")


def test_disable_thinking_emits_no_reasoning_instruction(tokenizer):
    """The template computes the instruction only when thinking is enabled."""
    text = _text(
        Qwen3_8DisableThinkingRenderer(tokenizer),
        [{"role": "user", "content": "Hi."}],
    )
    assert REASONING_EFFORT_INSTRUCTIONS["xhigh"] not in text


def test_reasoning_instruction_is_injected_once_for_the_tool_prefix_flow(tokenizer):
    """``create_conversation_prefix_with_tools(...) + messages`` must not double it.

    The prefix helper deliberately does not inject; rendering does. If both did,
    this flow -- the one every tool-using recipe uses -- would emit it twice.
    """
    renderer = Qwen3_8Renderer(tokenizer)
    messages = renderer.create_conversation_prefix_with_tools([], "Be terse.") + [
        {"role": "user", "content": "Hi."}
    ]
    text = _text(renderer, messages)
    assert text.count(REASONING_EFFORT_INSTRUCTIONS["xhigh"]) == 1
    assert "Be terse." in text


def test_normalization_is_idempotent(tokenizer):
    renderer = Qwen3_8Renderer(tokenizer)
    messages: list[Message] = [{"role": "user", "content": "Hi."}]
    once = renderer._normalize_messages(messages)
    twice = renderer._normalize_messages(once)
    assert once == twice


def test_empty_system_message_renders_nothing(tokenizer):
    """Qwen3.5/3.6 emitted an empty system block here; Qwen3.8 drops it."""
    renderer = Qwen3_8Renderer(tokenizer, reasoning_effort="medium")
    normalized = renderer._normalize_messages(
        [
            {"role": "system", "content": "   "},
            {"role": "user", "content": "Hi."},
        ]
    )
    assert [message["role"] for message in normalized] == ["user"]


def test_synthetic_system_message_carries_trainable_when_conversation_does(tokenizer):
    """CUSTOMIZED training asserts every message has ``trainable``.

    A synthetic system message without the field would trip that assertion, so
    it has to mirror the conversation it is inserted into.
    """
    renderer = Qwen3_8Renderer(tokenizer)
    messages: list[Message] = [
        {"role": "user", "content": "Hi.", "trainable": False},
        {"role": "assistant", "content": "Hello.", "trainable": True},
    ]
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.CUSTOMIZED
    )
    tokens = [token for chunk in model_input.chunks for token in chunk.tokens]
    assert len(weights) == len(tokens)
    assert weights.sum().item() > 0


def test_synthetic_system_message_omits_trainable_otherwise(tokenizer):
    """The inverse assertion: non-CUSTOMIZED modes reject a ``trainable`` field."""
    renderer = Qwen3_8Renderer(tokenizer)
    normalized = renderer._normalize_messages([{"role": "user", "content": "Hi."}])
    assert "trainable" not in normalized[0]


def test_model_info_routes_qwen3_8_to_its_own_renderers():
    assert get_recommended_renderer_names(MODEL) == [
        "qwen3_8_low_reasoning",
        "qwen3_8",
        "qwen3_8_medium_reasoning",
        "qwen3_8_disable_thinking",
    ]


def test_auto_selection_uses_low_reasoning_without_disabling_thinking(tokenizer):
    renderer = get_renderer(get_recommended_renderer_names(MODEL)[0], tokenizer)
    assert renderer.reasoning_effort == "low"
    assert renderer.disables_thinking is False
    assert REASONING_EFFORT_INSTRUCTIONS["low"] in _text(
        renderer, [{"role": "user", "content": "Hi."}]
    )


@pytest.mark.parametrize(
    "name,expected_effort",
    [
        ("qwen3_8", "xhigh"),
        ("qwen3_8_medium_reasoning", "medium"),
        ("qwen3_8_low_reasoning", "low"),
    ],
)
def test_named_renderer_variants_select_their_effort(tokenizer, name, expected_effort):
    assert get_renderer(name, tokenizer).reasoning_effort == expected_effort


def test_disable_thinking_variant_resolves_to_the_qwen3_8_class(tokenizer):
    """It must subclass Qwen3_8Renderer, not Qwen3_5DisableThinkingRenderer.

    Inheriting from the 3.5 variant would silently restore Qwen3.5's positional
    ``<think>`` framing for history, which Qwen3.8 does not use.
    """
    renderer = get_renderer("qwen3_8_disable_thinking", tokenizer)
    assert isinstance(renderer, Qwen3_8Renderer)
    assert renderer.strip_thinking_from_history is False
