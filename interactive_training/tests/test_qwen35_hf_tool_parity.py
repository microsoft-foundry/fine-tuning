"""Token-exact Qwen3.5/3.6/3.8 tool-call parity with Hugging Face templates."""

from __future__ import annotations

import json

import pytest
from transformers import AutoTokenizer

from interactive_training.renderers import get_renderer
from interactive_training.renderers.base import Message, ToolCall, ToolSpec, TrainOnWhat


# Each row binds a tool-capable Interactive Training renderer to a pinned HF template oracle.
# Compatible renderers added here automatically exercise the full content-shape matrix.
_HF_RENDERER_CASES = [
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "qwen3_5", {}),
    (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "qwen3_5_disable_thinking",
        {"enable_thinking": False},
    ),
    (
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "qwen3_5",
        {},
    ),
    (
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "qwen3_5_disable_thinking",
        {"enable_thinking": False},
    ),
    # Qwen3.8 diverges from the 3.5/3.6 template by 34 lines, so it gets its own
    # renderers. Each effort level is pinned separately because 'medium' is
    # accepted by the template but emits no instruction at all -- a shape a
    # single default-only case would never exercise.
    (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "qwen3_8",
        {},
    ),
    (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "qwen3_8",
        {"reasoning_effort": "xhigh"},
    ),
    (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "qwen3_8_medium_reasoning",
        {"reasoning_effort": "medium"},
    ),
    (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "qwen3_8_low_reasoning",
        {"reasoning_effort": "low"},
    ),
    (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "qwen3_8_disable_thinking",
        {"enable_thinking": False},
    ),
]

_TOOL_CALL_CONTENT_SHAPES = [
    pytest.param("", id="empty-string"),
    pytest.param(" \t\n", id="whitespace-string"),
    pytest.param("Calling the tool.", id="visible-string"),
    pytest.param(" \tCalling the tool.\n", id="padded-visible-string"),
    pytest.param([], id="empty-parts"),
    pytest.param([{"type": "text", "text": ""}], id="empty-text-part"),
    pytest.param(
        [{"type": "text", "text": " \t\n"}], id="whitespace-text-part"
    ),
    pytest.param(
        [{"type": "text", "text": "Calling the tool."}], id="visible-text-part"
    ),
    pytest.param(
        [{"type": "text", "text": " \tCalling the tool.\n"}],
        id="padded-visible-text-part",
    ),
    pytest.param(
        [{"type": "thinking", "thinking": "Need a lookup."}],
        id="thinking-only",
    ),
    pytest.param(
        [
            {"type": "thinking", "thinking": "Need a lookup."},
            {"type": "text", "text": ""},
        ],
        id="thinking-empty-text",
    ),
    pytest.param(
        [
            {"type": "thinking", "thinking": "Need a lookup."},
            {"type": "text", "text": " \t\n"},
        ],
        id="thinking-whitespace-text",
    ),
    pytest.param(
        [
            {"type": "thinking", "thinking": "Need a lookup."},
            {"type": "text", "text": " \tCalling the tool.\n"},
        ],
        id="thinking-visible-text",
    ),
    pytest.param(
        [
            {"type": "text", "text": " \t"},
            {"type": "thinking", "thinking": "Need a lookup."},
            {"type": "text", "text": "\n"},
        ],
        id="whitespace-around-thinking",
    ),
]

_TOOLS: list[ToolSpec] = [
    {
        "name": "lookup_record",
        "description": "Look up a record by identifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "revision": {"type": "integer"},
            },
            "required": ["record_id", "revision"],
            "additionalProperties": False,
        },
    }
]


def _flatten(model_input) -> list[int]:
    return [token for chunk in model_input.chunks for token in chunk.tokens]


def _token_ids(rendered) -> list[int]:
    if isinstance(rendered, list):
        return rendered
    return list(rendered["input_ids"])


@pytest.fixture(scope="module", params=_HF_RENDERER_CASES)
def model_case(request):
    model_name, revision, renderer_name, template_kwargs = request.param
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    renderer = get_renderer(renderer_name, tokenizer)
    return model_name, renderer, tokenizer, template_kwargs


def test_tool_declaration_and_generation_suffix_match_hf(model_case):
    _, renderer, tokenizer, template_kwargs = model_case
    user_messages: list[Message] = [{"role": "user", "content": "Find record 1234."}]

    interactive_training_messages = renderer.create_conversation_prefix_with_tools(_TOOLS) + user_messages
    interactive_training_tokens = _flatten(renderer.build_generation_prompt(interactive_training_messages))

    hf_tokens = _token_ids(
        tokenizer.apply_chat_template(
            user_messages,
            tools=_TOOLS,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )

    assert interactive_training_tokens == hf_tokens


@pytest.mark.parametrize("assistant_content", _TOOL_CALL_CONTENT_SHAPES)
def test_historical_tool_call_with_ambiguous_scalars_matches_hf(
    model_case, assistant_content
):
    _, renderer, tokenizer, template_kwargs = model_case
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="lookup_record",
            arguments='{"record_id":"1234","revision":1234,"tags":["1","2"]}',
        )
    )
    messages: list[Message] = [
        {"role": "user", "content": "Find record 1234."},
        {"role": "assistant", "content": assistant_content, "tool_calls": [tool_call]},
        {"role": "tool", "content": "found", "tool_call_id": tool_call.id},
        {"role": "user", "content": "Summarize it."},
    ]

    interactive_training_tokens = _flatten(renderer.build_generation_prompt(messages))
    hf_messages = [renderer.to_openai_message(message) for message in messages]
    hf_tokens = _token_ids(
        tokenizer.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )

    assert interactive_training_tokens == hf_tokens


@pytest.mark.parametrize("assistant_content", _TOOL_CALL_CONTENT_SHAPES)
def test_tool_call_content_shape_supervised_weights_remain_aligned(
    model_case, assistant_content
):
    _, renderer, tokenizer, template_kwargs = model_case
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="lookup_record",
            arguments='{"record_id":"1234","revision":1234}',
        )
    )
    messages: list[Message] = [
        {"role": "user", "content": "Find record 1234."},
        {"role": "assistant", "content": assistant_content, "tool_calls": [tool_call]},
    ]

    interactive_training_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    interactive_training_tokens = _flatten(interactive_training_input)
    hf_messages = [renderer.to_openai_message(message) for message in messages]
    hf_tokens = _token_ids(
        tokenizer.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
    )

    # HF owns the final turn-separator newline; Interactive Training owns separators in the next
    # message header. With no next message, only that terminal newline differs.
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    assert hf_tokens[-len(newline_tokens) :] == newline_tokens
    assert interactive_training_tokens == hf_tokens[: -len(newline_tokens)]
    assert len(weights) == len(interactive_training_tokens)
    assert weights.sum().item() > 0

# Templates disagree on how a non-string tool argument is serialized:
#   Qwen3.5  -> `tojson` for mappings/sequences, Jinja `| string` otherwise, so
#               booleans and null render Python-style as `True` / `None`.
#   Qwen3.6  -> `| string` only for strings, `tojson` otherwise: `true` / `null`.
#   Qwen3.8  -> same as 3.6.
# The other cases in this file only pass strings, ints and lists, where `str()`
# and `json.dumps()` agree -- so none of them can see this difference.
_MODELS_SERIALIZING_SCALARS_AS_JSON = {"Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B"}

# Qwen3.6 is routed to the `qwen3_5` renderer, which implements Qwen3.5's
# Python-style scalars. That is a real, pre-existing divergence from its own HF
# template. It is pinned rather than silently fixed because changing it alters
# the token stream for a model already in production; asserting the exact
# current behaviour means a future fix has to come here and say so.
_MODELS_WITH_KNOWN_SCALAR_DIVERGENCE = {"Qwen/Qwen3.6-35B-A3B"}


def test_boolean_and_null_tool_arguments_match_hf(model_case):
    model_name, renderer, tokenizer, template_kwargs = model_case
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="lookup_record",
            arguments='{"record_id":"1234","flag":true,"missing":null,"revision":7}',
        )
    )
    messages: list[Message] = [
        {"role": "user", "content": "Find record 1234."},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "found", "tool_call_id": tool_call.id},
        {"role": "user", "content": "Summarize it."},
    ]

    interactive_training_tokens = _flatten(renderer.build_generation_prompt(messages))
    hf_messages = [renderer.to_openai_message(message) for message in messages]
    hf_tokens = _token_ids(
        tokenizer.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )

    interactive_training_text = tokenizer.decode(interactive_training_tokens)
    if model_name in _MODELS_SERIALIZING_SCALARS_AS_JSON:
        expected_scalars = ("<parameter=flag>\ntrue\n", "<parameter=missing>\nnull\n")
    else:
        expected_scalars = ("<parameter=flag>\nTrue\n", "<parameter=missing>\nNone\n")

    if model_name in _MODELS_WITH_KNOWN_SCALAR_DIVERGENCE:
        # Characterization, not an endorsement: pin both that Interactive Training still emits
        # Python-style scalars and that HF does not, so this cannot regress or be
        # fixed unnoticed.
        assert "<parameter=flag>\nTrue\n" in interactive_training_text
        assert "<parameter=flag>\ntrue\n" in tokenizer.decode(hf_tokens)
        assert interactive_training_tokens != hf_tokens
        return

    for expected in expected_scalars:
        assert expected in interactive_training_text
    assert interactive_training_tokens == hf_tokens


# Non-ASCII is a second, independent way the `tojson` filter can diverge.
# transformers replaces Jinja's built-in `tojson` with its own implementation
# that defaults to ensure_ascii=False (transformers/utils/chat_template_utils.py),
# whereas Python's json.dumps defaults to True. Only Qwen3.8 is fixed, because
# Qwen3.5/3.6 already serve production traffic with the escaped form.
_MODELS_MATCHING_HF_NON_ASCII = {"Qwen/Qwen3.8-27B"}

_UNICODE_TOOLS: list[ToolSpec] = [
    {
        "name": "lookup_record",
        "description": "Rechercher un enregistrement dans le café.",
        "parameters": {
            "type": "object",
            "properties": {"tags": {"type": "array"}, "note": {"type": "string"}},
            "required": ["tags", "note"],
            "additionalProperties": False,
        },
    }
]


def test_non_ascii_tool_call_arguments_match_hf(model_case):
    """A structured tool argument containing non-ASCII must survive `tojson`."""
    model_name, renderer, tokenizer, template_kwargs = model_case
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="lookup_record",
            arguments=json.dumps({"tags": ["中文"], "note": "café"}, ensure_ascii=False),
        )
    )
    messages: list[Message] = [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "found", "tool_call_id": tool_call.id},
        {"role": "user", "content": "Summarize."},
    ]

    interactive_training_tokens = _flatten(renderer.build_generation_prompt(messages))
    hf_messages = [renderer.to_openai_message(message) for message in messages]
    hf_tokens = _token_ids(
        tokenizer.apply_chat_template(
            hf_messages, tokenize=True, add_generation_prompt=True, **template_kwargs
        )
    )
    interactive_training_text = tokenizer.decode(interactive_training_tokens)
    hf_text = tokenizer.decode(hf_tokens)

    # The plain string argument is never routed through json.dumps, so it stays
    # literal everywhere. Asserting it keeps the escaped-list assertion below
    # honest: it shows the divergence is specific to structured values.
    assert "<parameter=note>\ncafé\n" in interactive_training_text
    assert "<parameter=note>\ncafé\n" in hf_text
    assert '<parameter=tags>\n["中文"]\n' in hf_text

    if model_name in _MODELS_MATCHING_HF_NON_ASCII:
        assert '<parameter=tags>\n["中文"]\n' in interactive_training_text
        assert interactive_training_tokens == hf_tokens
    else:
        # Characterization of the pre-existing Qwen3.5/3.6 behaviour.
        assert '<parameter=tags>\n["\\u4e2d\\u6587"]\n' in interactive_training_text
        assert interactive_training_tokens != hf_tokens


def test_non_ascii_tool_declaration_matches_hf(model_case):
    """A tool whose description contains non-ASCII must survive `tojson`."""
    model_name, renderer, tokenizer, template_kwargs = model_case
    prefix = renderer.create_conversation_prefix_with_tools(_UNICODE_TOOLS)
    interactive_training_text = "".join(str(message.get("content", "")) for message in prefix)
    hf_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tools=_UNICODE_TOOLS,
        tokenize=False,
        **template_kwargs,
    )

    assert "café" in hf_text
    if model_name in _MODELS_MATCHING_HF_NON_ASCII:
        assert "café" in interactive_training_text
        assert "caf\\u00e9" not in interactive_training_text
    else:
        assert "caf\\u00e9" in interactive_training_text
        assert "café" not in interactive_training_text
