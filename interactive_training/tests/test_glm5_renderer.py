"""Token-exact conformance tests for the GLM-5.2 renderer.

These verify that :class:`GLM5Renderer` reproduces the exact token IDs of
``zai-org/GLM-5.2-FP8``'s ``apply_chat_template`` for the generation-prompt path,
plus a ``parse_response`` round-trip that recovers the prompt-prefilled
``<think>`` reasoning channel.  Skipped unless the GLM-5.2 tokenizer is on disk.
"""

from __future__ import annotations

import glob
import os

import pytest

from interactive_training.renderers import get_renderer
from interactive_training.renderers.glm5 import (
    GLM5DisableThinkingRenderer,
    GLM5HighReasoningRenderer,
    GLM5Renderer,
)

_MODEL_ID = "zai-org/GLM-5.2-FP8"


def _tokenizer_on_disk() -> bool:
    roots = [os.environ.get("HF_HOME", ""), "/mnt/tianyouzhang_extended/hf_cache"]
    for root in roots:
        if root and glob.glob(
            os.path.join(root, "hub", "models--zai-org--GLM-5.2-FP8", "snapshots", "*", "tokenizer_config.json")
        ):
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _tokenizer_on_disk(), reason="GLM-5.2 tokenizer not available on disk"
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
    except ValueError as exc:
        # Transformers <5 cannot resolve GLM-5.2's ``TokenizersBackend`` tokenizer
        # class. The renderer's production path handles this by loading
        # ``tokenizer.json`` directly (tokenizer_utils._load_glm5_tokenizer_from_json),
        # but this conformance test compares against ``apply_chat_template`` and so
        # needs the packaged chat template, which the fallback tokenizer lacks. Skip
        # cleanly on such environments instead of erroring.
        if "TokenizersBackend" in str(exc):
            pytest.skip(f"Installed transformers cannot load the GLM-5.2 tokenizer: {exc}")
        raise


def _flat(model_input) -> list[int]:
    out: list[int] = []
    for chunk in model_input.chunks:
        out.extend(list(chunk.tokens))
    return out


def _hf(tok, messages, **kwargs) -> list[int]:
    out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs
    )
    # transformers >=5 returns a BatchEncoding (dict-like) when tokenize=True.
    return list(out["input_ids"]) if not isinstance(out, list) else out


def test_user_thinking_on(tok):
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    assert _flat(GLM5Renderer(tok).build_generation_prompt(msgs)) == _hf(tok, msgs)


def test_user_thinking_off(tok):
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    mine = _flat(GLM5DisableThinkingRenderer(tok).build_generation_prompt(msgs))
    assert mine == _hf(tok, msgs, enable_thinking=False)


def test_system_plus_user(tok):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert _flat(GLM5Renderer(tok).build_generation_prompt(msgs)) == _hf(tok, msgs)


def test_reasoning_effort_high(tok):
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    mine = _flat(GLM5HighReasoningRenderer(tok).build_generation_prompt(msgs))
    assert mine == _hf(tok, msgs, reasoning_effort="high")


def test_multiturn_history_thinking_stripped(tok):
    # The renderer takes parsed assistant content; HF takes the raw <think> string.
    ren_msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "add them"},
                {"type": "text", "text": "The answer is 4."},
            ],
        },
        {"role": "user", "content": "And 3+3?"},
    ]
    hf_msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>add them</think>The answer is 4."},
        {"role": "user", "content": "And 3+3?"},
    ]
    assert _flat(GLM5Renderer(tok).build_generation_prompt(ren_msgs)) == _hf(tok, hf_msgs)


def test_factory_routing(tok):
    assert isinstance(get_renderer("glm5", tok), GLM5Renderer)
    assert isinstance(get_renderer("glm5_disable_thinking", tok), GLM5DisableThinkingRenderer)
    assert isinstance(get_renderer("glm5_high_reasoning", tok), GLM5HighReasoningRenderer)


def test_parse_response_recovers_thinking(tok):
    def enc(s):
        return tok.encode(s, add_special_tokens=False)

    renderer = GLM5Renderer(tok)
    # Sampler output after the prompt's <think> prefill: reasoning + </think> + answer + EOS.
    response = enc("Let me add. 2+2=4.") + enc("</think>") + enc("The answer is 4.") + enc("<|endoftext|>")
    msg, ok = renderer.parse_response(response)
    assert ok
    parts = msg["content"]
    thinking = "".join(p["thinking"] for p in parts if p["type"] == "thinking")
    text = "".join(p["text"] for p in parts if p["type"] == "text")
    assert "add" in thinking
    assert "answer is 4" in text


def test_parse_response_stops_at_user_token(tok):
    def enc(s):
        return tok.encode(s, add_special_tokens=False)

    renderer = GLM5Renderer(tok)
    response = enc("reasoning") + enc("</think>") + enc("Answer.") + enc("<|user|>")
    msg, ok = renderer.parse_response(response)
    assert ok
    text = "".join(p["text"] for p in msg["content"] if p["type"] == "text")
    assert text == "Answer."


def test_parse_response_no_stop_is_format_error(tok):
    renderer = GLM5Renderer(tok)
    response = tok.encode("incomplete output", add_special_tokens=False)
    _msg, ok = renderer.parse_response(response)
    assert ok is False


def test_stop_sequences_are_eos_ids(tok):
    renderer = GLM5Renderer(tok)
    expected = [
        tok.encode(t, add_special_tokens=False)[0]
        for t in ("<|endoftext|>", "<|user|>", "<|observation|>")
    ]
    assert renderer.get_stop_sequences() == expected
