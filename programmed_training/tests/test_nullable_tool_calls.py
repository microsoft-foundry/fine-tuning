"""Regression tests for nullable tool-call fields from Hugging Face datasets."""

from __future__ import annotations

from typing import cast

import pytest

from programmed_training.renderers.base import Message, RenderContext, ToolCall
from programmed_training.renderers.gpt_oss import GptOssRenderer
from programmed_training.renderers.muse_glimmer import MuseGlimmerRenderer
from programmed_training.renderers.qwen3 import Qwen3Renderer, Qwen3VLRenderer
from programmed_training.renderers.qwen3_5 import Qwen3_5Renderer
from programmed_training.supervised.data import conversation_to_datum


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


@pytest.mark.parametrize(
    "renderer",
    [
        Qwen3Renderer(_CharacterTokenizer()),
        Qwen3VLRenderer(_CharacterTokenizer(), object()),
        Qwen3_5Renderer(_CharacterTokenizer()),
    ],
)
def test_null_tool_calls_render_like_omitted_tool_calls(renderer):
    omitted: Message = {"role": "assistant", "content": "Plain response."}
    normalized = cast(
        Message,
        {"role": "assistant", "content": "Plain response.", "tool_calls": None},
    )
    context = RenderContext(idx=0, is_last=True, last_user_index=-1)

    assert renderer.render_message(normalized, context) == renderer.render_message(
        omitted, context
    )


@pytest.mark.parametrize(
    "renderer",
    [
        GptOssRenderer(_CharacterTokenizer()),
        MuseGlimmerRenderer(_CharacterTokenizer(), current_date="2026-10-05"),
        Qwen3Renderer(_CharacterTokenizer()),
        Qwen3VLRenderer(_CharacterTokenizer(), object()),
        Qwen3_5Renderer(_CharacterTokenizer()),
    ],
)
def test_null_tool_call_content_renders_like_empty_content(renderer):
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="lookup", arguments="{}"),
    )
    empty: Message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call],
    }
    normalized = cast(
        Message,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        },
    )
    context = RenderContext(idx=0, is_last=True, last_user_index=-1)

    assert renderer.render_message(normalized, context) == renderer.render_message(
        empty, context
    )


def test_arrow_tool_call_dictionaries_are_normalized_before_rendering():
    renderer = Qwen3_5Renderer(_CharacterTokenizer())
    conversation = cast(
        list[Message],
        [
            {"role": "user", "content": "Look it up.", "tool_calls": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{}",
                        },
                    }
                ],
            },
        ],
    )

    datum = conversation_to_datum(
        conversation,
        renderer,
        max_length=None,
    )

    assert datum.model_input.chunks
