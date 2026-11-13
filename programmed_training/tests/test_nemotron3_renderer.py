"""Nemotron Ultra renderer contracts that do not require model downloads."""

from __future__ import annotations

import copy
import json

import pytest

from programmed_training import model_info
from programmed_training.renderers import get_renderer
from programmed_training.renderers.base import Message, ToolCall, ToolSpec, TrainOnWhat
from programmed_training.renderers.nemotron3 import (
    Nemotron3UltraDisableThinkingRenderer,
    Nemotron3UltraMediumThinkingRenderer,
    Nemotron3UltraPreserveThinkingRenderer,
    Nemotron3UltraRenderer,
)

MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_256K = MODEL + "-256K"
VARIANTS = [
    ("nemotron3_ultra", Nemotron3UltraRenderer, {}),
    (
        "nemotron3_ultra_medium_thinking",
        Nemotron3UltraMediumThinkingRenderer,
        {"medium_effort": True},
    ),
    (
        "nemotron3_ultra_disable_thinking",
        Nemotron3UltraDisableThinkingRenderer,
        {"enable_thinking": False},
    ),
    (
        "nemotron3_ultra_preserve_thinking",
        Nemotron3UltraPreserveThinkingRenderer,
        {"truncate_history_thinking": False},
    ),
]
TOOLS: list[ToolSpec] = [
    {
        "name": "lookup",
        "description": " Look up a record. ",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": " Identifier ",
                    "default": "1234",
                },
                "enabled": {"type": "boolean"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "enum": [["café"]],
                },
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
    }
]


class CharacterTokenizer:
    """Reversible text tokenizer with single-token model delimiters."""

    special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]

    def encode(self, text, add_special_tokens=False):
        tokens = []
        while text:
            for index, token in enumerate(self.special):
                if text.startswith(token):
                    tokens.append(index)
                    text = text[len(token) :]
                    break
            else:
                tokens.append(ord(text[0]) + len(self.special))
                text = text[1:]
        return tokens

    def decode(self, tokens):
        return "".join(
            self.special[token]
            if token < len(self.special)
            else chr(token - len(self.special))
            for token in tokens
        )


def _flat(model_input):
    return [token for chunk in model_input.chunks for token in chunk.tokens]


def _text(renderer, messages):
    return renderer.tokenizer.decode(_flat(renderer.build_generation_prompt(messages)))


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.mark.parametrize("name,cls,kwargs", VARIANTS)
def test_factory_and_generation_modes(tokenizer, name, cls, kwargs):
    renderer = get_renderer(name, tokenizer)
    assert isinstance(renderer, cls)
    prompt = _text(renderer, [{"role": "user", "content": "Hi."}])
    assert prompt.startswith("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi.")
    assert ("{reasoning effort: efficient}" in prompt) == bool(kwargs.get("medium_effort"))
    expected = "<think></think>" if kwargs.get("enable_thinking") is False else "<think>\n"
    assert prompt.endswith("<|im_start|>assistant\n" + expected)
    assert renderer.strip_thinking_from_history == kwargs.get(
        "truncate_history_thinking", True
    )
    assert renderer.disables_thinking == (kwargs.get("enable_thinking") is False)
    assert renderer.has_extension_property is False


@pytest.mark.parametrize(
    "model",
    [MODEL, MODEL.split("/")[1].lower(), MODEL_256K, MODEL_256K.split("/")[1].lower()],
)
def test_model_routing(model):
    assert model_info.get_recommended_renderer_names(model) == [row[0] for row in VARIANTS]
    assert model_info.get_recommended_renderer_name(model) == "nemotron3_ultra"
    assert (
        model_info.get_recommended_renderer_name(model, prefer_non_thinking=True)
        == "nemotron3_ultra_disable_thinking"
    )
    assert model_info.get_model_attributes(model).is_vl is False


def test_unsupported_effort_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="Unsupported Nemotron Ultra reasoning effort"):
        Nemotron3UltraRenderer(tokenizer, reasoning_effort="low")


@pytest.mark.parametrize("name,cls,kwargs", VARIANTS)
def test_history_thinking_and_input_immutability(tokenizer, name, cls, kwargs):
    renderer = get_renderer(name, tokenizer)
    messages: list[Message] = [
        {"role": "user", "content": "First."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Earlier reasoning."},
                {"type": "text", "text": "One."},
            ],
        },
        {"role": "user", "content": "Next."},
    ]
    original = copy.deepcopy(messages)
    text = _text(renderer, messages)
    assert ("Earlier reasoning." in text) == (
        kwargs.get("truncate_history_thinking") is False
    )
    assert "One." in text
    assert messages == original
    if kwargs.get("medium_effort"):
        assert text.count("{reasoning effort: efficient}") == 1
        assert "First.\n\n{reasoning effort:" not in text


@pytest.mark.parametrize("name,cls,kwargs", VARIANTS)
def test_customized_training_and_synthetic_system_mask(tokenizer, name, cls, kwargs):
    renderer = get_renderer(name, tokenizer)
    messages: list[Message] = [
        {"role": "user", "content": "Hi.", "trainable": False},
        {"role": "assistant", "content": "Hello.", "trainable": True},
    ]
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.CUSTOMIZED
    )
    assert len(weights) == len(_flat(model_input))
    trained = [token for token, weight in zip(_flat(model_input), weights) if weight]
    assert tokenizer.decode(trained) == "Hello.<|im_end|>"
    assert messages[0]["role"] == "user"


@pytest.mark.parametrize("name,cls,kwargs", VARIANTS)
def test_response_parsing_and_ultra_whitespace(tokenizer, name, cls, kwargs):
    renderer = get_renderer(name, tokenizer)
    response = "\n\nAnswer.<|im_end|>"
    if not renderer.disables_thinking:
        response = "Reason.</think>" + response
    message, success = renderer.parse_response(tokenizer.encode(response))
    assert success
    if renderer.disables_thinking:
        assert message["content"] == "\n\nAnswer."
    else:
        assert message["content"] == [
            {"type": "thinking", "thinking": "Reason."},
            {"type": "text", "text": "\n\nAnswer."},
        ]
    assert renderer.parse_response(tokenizer.encode("incomplete"))[1] is False
    assert renderer.get_stop_sequences() == tokenizer.encode("<|im_end|>")


def test_schema_aware_xml_tool_parsing(tokenizer):
    renderer = Nemotron3UltraRenderer(tokenizer)
    call = ToolCall(
        function=ToolCall.FunctionBody(
            name="lookup", arguments='{"record_id":"1234","enabled":true}'
        )
    )
    message, success = renderer.parse_response_with_tools(
        tokenizer.encode(
            'Lookup.</think>\n<tool_call>\n<function=lookup>\n'
            '<parameter=record_id>\n1234\n</parameter>\n'
            '<parameter=enabled>\nTrue\n</parameter>\n'
            '</function>\n</tool_call>\n<|im_end|>'
        ),
        TOOLS,
    )
    assert success
    assert message["tool_calls"][0].function.name == call.function.name
    assert json.loads(message["tool_calls"][0].function.arguments) == json.loads(
        call.function.arguments
    )
    assert "unparsed_tool_calls" not in message


def test_preserving_history_does_not_claim_prefix_identity(tokenizer):
    renderer = Nemotron3UltraPreserveThinkingRenderer(tokenizer)
    messages: list[Message] = [{"role": "user", "content": "Hi."}]
    prompt = _flat(renderer.build_generation_prompt(messages))
    response = tokenizer.encode("</think>Hello.<|im_end|>")
    message, success = renderer.parse_response(response)
    assert success
    next_prompt = _flat(
        renderer.build_generation_prompt(
            [*messages, message, {"role": "user", "content": "Next."}]
        )
    )
    sampled = prompt + response
    assert next_prompt[: len(sampled)] != sampled
    assert renderer.has_extension_property is False


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image", "image": "image.png"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ],
)
def test_multimodal_content_is_rejected(tokenizer, part):
    with pytest.raises(ValueError, match="supports text and assistant thinking"):
        Nemotron3UltraRenderer(tokenizer).build_generation_prompt(
            [{"role": "user", "content": [part]}]
        )


def test_tool_results_share_one_user_block(tokenizer):
    renderer = Nemotron3UltraRenderer(tokenizer)
    prompt = _text(
        renderer,
        [
            {"role": "user", "content": "Look up both."},
            {"role": "tool", "content": "One."},
            {"role": "tool", "content": "Two."},
        ],
    )
    assert (
        "<|im_start|>user\n"
        "<tool_response>\nOne.\n</tool_response>\n"
        "<tool_response>\nTwo.\n</tool_response>\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ) in prompt


def test_tool_declarations_use_xml_and_follow_system_prompt(tokenizer):
    renderer = Nemotron3UltraRenderer(tokenizer)
    messages = renderer.create_conversation_prefix_with_tools(TOOLS, "Be helpful.")
    prompt = _text(renderer, messages + [{"role": "user", "content": "Hi."}])
    assert prompt.startswith("<|im_start|>system\nBe helpful.\n\n# Tools\n")
    assert "<tools>\n<function>\n<name>lookup</name>" in prompt
    assert "<description>Look up a record.</description>" in prompt
    assert "<default>1234</default>" in prompt
    assert '<items>{"type": "string"}</items>' in prompt
    assert "<additionalProperties>False</additionalProperties>" in prompt


@pytest.mark.parametrize("name,cls,kwargs", VARIANTS)
def test_sft_retains_current_answer_reasoning(tokenizer, name, cls, kwargs):
    renderer = get_renderer(name, tokenizer)
    model_input, _ = renderer.build_supervised_example(
        [
            {"role": "user", "content": "Hi."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Current reasoning."},
                    {"type": "text", "text": "Answer."},
                ],
            },
        ]
    )
    assert "<think>\nCurrent reasoning.</think>Answer." in tokenizer.decode(
        _flat(model_input)
    )
