import asyncio
import base64
import io
import json

import pytest

from azure.ai.finetuningsessions.models import ModelInputChunk

from interactive_training import model_info
from interactive_training.image_types import ImageChunk
from interactive_training.renderers import get_renderer
from interactive_training.renderers.base import (
    Message,
    RenderContext,
    ThinkingPart,
    ToolCall,
    TrainOnWhat,
    _load_image_reference,
)
from interactive_training.renderers.muse_glimmer import MuseGlimmerRenderer
from interactive_training.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from interactive_training.tool_use.mcp import mcp_call_result_to_tool_result
from interactive_training.tool_use.tools import handle_tool_call, tool
from interactive_training.tool_use.types import ToolResult


class ReversibleTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


def _decode_prompt(renderer, messages):
    prompt = renderer.build_generation_prompt(messages)
    return renderer.tokenizer.decode(
        [token for chunk in prompt.chunks for token in chunk.tokens]
    )


def test_model_info_routes_muse_glimmer_to_renderer():
    attributes = model_info.get_model_attributes("meta-models/Muse-Glimmer-30B")

    assert attributes.is_vl
    assert attributes.max_context_length == 32_768
    model_info.require_vision_language_model(
        "meta-models/Muse-Glimmer-30B", workload="dataset='visual_spatial'"
    )
    assert model_info.get_recommended_renderer_names(
        "meta-models/Muse-Glimmer-30B"
    ) == ["muse_glimmer"]


def test_model_info_rejects_text_only_model_for_vision_workload():
    with pytest.raises(ValueError, match="requires a vision-language model"):
        model_info.require_vision_language_model(
            "Qwen/Qwen3-32B", workload="dataset='visual_spatial'"
        )


@pytest.mark.parametrize(
    ("name", "reasoning_strength"),
    [
        ("muse_glimmer_low_reasoning", "low"),
        ("muse_glimmer_medium_reasoning", "medium"),
        ("muse_glimmer", "high"),
        ("muse_glimmer_xhigh_reasoning", "xhigh"),
    ],
)
def test_factory_constructs_muse_glimmer_reasoning_strength(name, reasoning_strength):
    renderer = get_renderer(name, ReversibleTokenizer())

    assert isinstance(renderer, MuseGlimmerRenderer)
    assert renderer.reasoning_strength == reasoning_strength


def test_generation_prompt_matches_atem_text_format():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer(), current_date="2026-08-13")

    prompt = _decode_prompt(renderer, [Message(role="user", content="Hello")])

    assert prompt == (
        "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant."
        "\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-08-13."
        "\n\nReasoning strength: high."
        '\n\n# Valid recipients: "self", "user".<|eot|>'
        "<|start|>user<|message|>Hello<|eot|><|start|>assistant"
    )


def test_image_prompt_uses_native_muse_sentinels(monkeypatch):
    image_processor = object()
    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=image_processor,
        current_date="2026-08-13",
    )
    image_chunk = ImageChunk(
        data=b"\xff\xd8\xffjpeg", format="jpeg", expected_tokens=12
    )
    monkeypatch.setattr(
        "interactive_training.renderers.base.image_to_chunk",
        lambda image, processor: image_chunk,
    )

    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "image", "image": "https://example.test/grid.png"},
                    {"type": "text", "text": "How many triangles?"},
                ],
            )
        ]
    )

    assert prompt.chunks[-6:] == [
        ModelInputChunk(tokens=list(b"<|image_start|>")),
        image_chunk,
        ModelInputChunk(tokens=list(b"<|image_end|>")),
        ModelInputChunk(tokens=list(b"How many triangles?")),
        ModelInputChunk(tokens=list(b"<|eot|>")),
        ModelInputChunk(tokens=list(b"<|start|>assistant")),
    ]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/grid.png",
        "data:image/png;base64,iVBORw0KGgo=",
    ],
)
def test_image_prompt_accepts_foundry_image_url_parts(monkeypatch, url):
    image_chunk = ImageChunk(
        data=b"\xff\xd8\xffjpeg", format="jpeg", expected_tokens=12
    )
    received_images = []
    monkeypatch.setattr(
        "interactive_training.renderers.base.image_to_chunk",
        lambda image, processor: received_images.append(image) or image_chunk,
    )
    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=object(),
        current_date="2026-08-13",
    )

    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": "high"},
                    }
                ],
            )
        ]
    )

    assert image_chunk in prompt.chunks
    assert received_images == [url]


@pytest.mark.parametrize("source_kind", ["public_url", "data_uri"])
def test_image_prompt_converts_foundry_image_url_to_real_chunk(
    monkeypatch, source_kind
):
    Image = pytest.importorskip("PIL.Image")
    image_buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color="red").save(image_buffer, format="PNG")
    image_data = image_buffer.getvalue()

    if source_kind == "public_url":
        source = "https://example.test/grid.png"
        monkeypatch.setattr(
            "interactive_training.renderers.base._load_image_reference",
            lambda reference: image_data,
        )
    else:
        source = "data:image/png;base64," + base64.b64encode(image_data).decode()

    class ImageProcessor:
        merge_size = 1

        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            assert (height, width) == (2, 3)
            return 6

    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=ImageProcessor(),
        current_date="2026-08-13",
    )
    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": source, "detail": "high"},
                    }
                ],
            )
        ]
    )

    image_chunks = [chunk for chunk in prompt.chunks if isinstance(chunk, ImageChunk)]
    assert len(image_chunks) == 1
    assert image_chunks[0].format == "jpeg"
    assert image_chunks[0].data.startswith(b"\xff\xd8")
    assert image_chunks[0].expected_tokens == 6


def test_image_prompt_preserves_multiple_foundry_images_in_order(monkeypatch):
    before_chunk = ImageChunk(
        data=b"\xff\xd8\xffbefore", format="jpeg", expected_tokens=10
    )
    after_chunk = ImageChunk(
        data=b"\xff\xd8\xffafter", format="jpeg", expected_tokens=11
    )
    chunks_by_url = {
        "https://example.test/before.png": before_chunk,
        "https://example.test/after.png": after_chunk,
    }
    received_images = []
    monkeypatch.setattr(
        "interactive_training.renderers.base.image_to_chunk",
        lambda image, processor: received_images.append(image) or chunks_by_url[image],
    )
    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=object(),
        current_date="2026-08-13",
    )

    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Compare these two screenshots."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/before.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/after.png"},
                    },
                ],
            )
        ]
    )

    assert received_images == [
        "https://example.test/before.png",
        "https://example.test/after.png",
    ]
    assert prompt.chunks[-9:-2] == [
        ModelInputChunk(tokens=list(b"Compare these two screenshots.")),
        ModelInputChunk(tokens=list(b"<|image_start|>")),
        before_chunk,
        ModelInputChunk(tokens=list(b"<|image_end|>")),
        ModelInputChunk(tokens=list(b"<|image_start|>")),
        after_chunk,
        ModelInputChunk(tokens=list(b"<|image_end|>")),
    ]


@pytest.mark.parametrize(
    ("content_kind", "has_text", "has_image"),
    [
        ("text_only", True, False),
        ("image_only", False, True),
        ("text_and_image", True, True),
    ],
)
def test_mcp_tool_output_flows_from_tool_call_to_generation_prompt(
    monkeypatch, content_kind, has_text, has_image
):
    Image = pytest.importorskip("PIL.Image")
    image_buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color="red").save(image_buffer, format="PNG")
    image_data = image_buffer.getvalue()
    image_url = "https://images.example.test/screenshot.png"
    requested_urls = []
    monkeypatch.setattr(
        "interactive_training.renderers.base._download_image_reference",
        lambda url: requested_urls.append(url) or image_data,
    )
    text_part = {"type": "text", "text": "Current screen:\n"}

    @tool
    async def fetch_image(url: str) -> ToolResult:
        """Fetch an image from a public HTTPS URL."""
        image_bytes = await asyncio.to_thread(_load_image_reference, url)
        mcp_image = {
            "type": "image",
            "data": base64.b64encode(image_bytes).decode(),
            "mimeType": "image/png",
        }
        content_by_kind = {
            "text_only": [{"type": "text", "text": "Current screen:\n"}],
            "image_only": [mcp_image],
            "text_and_image": [text_part, mcp_image],
        }
        return mcp_call_result_to_tool_result(
            {"content": content_by_kind[content_kind]}
        )

    tool_call = ToolCall(
        id="call_123",
        function=ToolCall.FunctionBody(
            name="fetch_image",
            arguments=json.dumps({"url": image_url}),
        ),
    )
    tool_result = asyncio.run(handle_tool_call({"fetch_image": fetch_image}, tool_call))

    assert requested_urls == [image_url]
    assert tool_result.messages[0].get("name") == "fetch_image"
    assert tool_result.messages[0].get("tool_call_id") == "call_123"

    class ImageProcessor:
        merge_size = 1

        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            assert (height, width) == (2, 3)
            return 6

    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=ImageProcessor(),
        current_date="2026-08-13",
    )

    prompt = renderer.build_generation_prompt(
        [
            Message(role="user", content="What is on screen?"),
            Message(role="assistant", content="", tool_calls=[tool_call]),
            *tool_result.messages,
        ]
    )

    assert (
        ModelInputChunk(tokens=list(b"<|start|>tool fetch_image<|message|>"))
        in prompt.chunks
    )
    assert (
        ModelInputChunk(tokens=list(b"Current screen:\n")) in prompt.chunks
    ) is has_text

    image_chunks = [chunk for chunk in prompt.chunks if isinstance(chunk, ImageChunk)]
    assert bool(image_chunks) is has_image
    if image_chunks:
        assert len(image_chunks) == 1
        assert image_chunks[0].format == "jpeg"
        assert image_chunks[0].data.startswith(b"\xff\xd8")
        assert image_chunks[0].expected_tokens == 6
        image_index = prompt.chunks.index(image_chunks[0])
        assert prompt.chunks[image_index - 1] == ModelInputChunk(
            tokens=list(b"<|image_start|>")
        )
        assert prompt.chunks[image_index + 1] == ModelInputChunk(
            tokens=list(b"<|image_end|>")
        )


def test_tool_image_output_rejects_private_url():
    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=object(),
        current_date="2026-08-13",
    )

    with pytest.raises(ValueError, match="public unicast"):
        renderer.build_generation_prompt(
            [
                Message(
                    role="tool",
                    name="screenshot",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://127.0.0.1/screen.png"},
                        }
                    ],
                )
            ]
        )


def test_supervised_example_matches_systemless_generation_prefix():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer(), current_date="2026-08-13")
    prompt_messages = [Message(role="user", content="Hello")]
    generation_prompt = renderer.build_generation_prompt(prompt_messages)
    model_input, weights = renderer.build_supervised_example(
        [*prompt_messages, Message(role="assistant", content="Hi")]
    )
    generation_tokens = [
        token for chunk in generation_prompt.chunks for token in chunk.tokens
    ]
    supervised_tokens = [
        token for chunk in model_input.chunks for token in chunk.tokens
    ]

    assert supervised_tokens[: len(generation_tokens)] == generation_tokens
    assert weights[: len(generation_tokens)].tolist() == [0.0] * len(generation_tokens)
    assert renderer.tokenizer.decode(supervised_tokens[len(generation_tokens) :]) == (
        " to=user<|message|>Hi<|eot|>"
    )


def test_supervised_example_preserves_explicit_system_message():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer(), current_date="2026-08-13")
    messages = [
        Message(role="system", content="Be concise."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    rendered = renderer.tokenizer.decode(
        [token for chunk in model_input.chunks for token in chunk.tokens]
    )

    assert rendered.count("<|start|>system<|message|>") == 1
    assert "Be concise." in rendered


def test_plain_text_response_round_trips():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    sampled = " to=user<|message|>Answer.<|eot|>"

    message, success = renderer.parse_response(renderer.tokenizer.encode(sampled))
    prompt = renderer.build_generation_prompt(
        [Message(role="user", content="Question?"), message]
    )
    rerendered = renderer.tokenizer.decode(
        [token for chunk in prompt.chunks for token in chunk.tokens]
    )

    assert success
    assert sampled in rerendered


def test_reasoning_response_round_trips():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    sampled = (
        " to=self<|message|>Think carefully.<|eom|>"
        "<|start|>assistant to=user<|message|>Answer.<|eot|>"
    )

    message, success = renderer.parse_response(renderer.tokenizer.encode(sampled))

    assert success
    assert message["content"] == [
        ThinkingPart(type="thinking", thinking="Think carefully."),
        {"type": "text", "text": "Answer."},
    ]

    prompt = renderer.build_generation_prompt(
        [Message(role="user", content="Question?"), message]
    )
    rerendered = renderer.tokenizer.decode(
        [token for chunk in prompt.chunks for token in chunk.tokens]
    )
    assert sampled in rerendered


@pytest.mark.parametrize("part_type", ["image", "image_url"])
def test_assistant_message_rejects_unsupported_image_content(part_type):
    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    image_part = (
        {"type": "image", "image": "https://example.test/image.png"}
        if part_type == "image"
        else {
            "type": "image_url",
            "image_url": {"url": "https://example.test/image.png"},
        }
    )

    with pytest.raises(
        ValueError,
        match=f"assistant messages do not support content types: {part_type}",
    ):
        renderer.build_generation_prompt(
            [Message(role="assistant", content=[image_part])]
        )


def test_final_reasoning_answer_renders_next_observation_after_episode_ends():
    class FinalAnswerEnv(MessageEnv):
        async def initial_observation(self):
            return [Message(role="user", content="Question?")]

        async def step(self, message):
            return MessageStepResult(
                reward=1.0,
                episode_done=True,
                next_messages=[Message(role="user", content="Question?"), message],
            )

    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    sampled = (
        " to=self<|message|>Think carefully.<|eom|>"
        "<|start|>assistant to=user<|message|>Answer.<|eot|>"
    )
    env = EnvFromMessageEnv(renderer, FinalAnswerEnv())

    result = asyncio.run(env.step(renderer.tokenizer.encode(sampled)))
    rerendered = renderer.tokenizer.decode(
        [token for chunk in result.next_observation.chunks for token in chunk.tokens]
    )

    assert result.episode_done
    assert result.reward == 1.0
    assert sampled in rerendered


def test_reasoning_message_converts_to_openai_fields():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    message = Message(
        role="assistant",
        content=[
            ThinkingPart(type="thinking", thinking="Think carefully."),
            {"type": "text", "text": "Answer."},
        ],
    )

    result = renderer.to_openai_message(message)

    assert result["reasoning_content"] == "Think carefully."
    assert result["content"] == "Answer."


@pytest.mark.parametrize("thinking_length", [0, 1, 19, 257])
@pytest.mark.parametrize("result_kind", ["text", "image", "mixed"])
@pytest.mark.parametrize("call_count", [1, 3])
@pytest.mark.parametrize("train_on", [TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.ALL_ASSISTANT_MESSAGES])
def test_thinking_with_ragged_tool_results_and_supervision(thinking_length, result_kind, call_count, train_on):
    from PIL import Image

    class ImageProcessor:
        merge_size = 1

        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            return height * width

    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(), image_processor=ImageProcessor(), current_date="2026-09-23"
    )
    thinking_before = "Inspect views. " * thinking_length
    thinking_after = "Compare observations. " * (thinking_length + 3) if thinking_length else ""
    calls = [ToolCall(id=f"view-{index}", function=ToolCall.FunctionBody(
        name="capture", arguments=json.dumps({"view": index})
    )) for index in range(call_count)]
    before_content = [ThinkingPart(type="thinking", thinking=thinking_before)] if thinking_before else ""
    before = Message(role="assistant", content=before_content, tool_calls=calls)
    messages = [Message(role="user", content="Inspect the views."), before]
    expected_dimensions = []
    for index, call in enumerate(calls):
        content = []
        if result_kind != "image":
            content.append({"type": "text", "text": "tool-observation " * (index * 23 + 1)})
        if result_kind != "text":
            for image_index in range(index + 1):
                dimensions = (3 + index * 7, 2 + image_index * 5)
                buffer = io.BytesIO()
                Image.new("RGB", dimensions).save(buffer, format="PNG")
                content.append({"type": "image", "data": base64.b64encode(buffer.getvalue()).decode(), "mimeType": "image/png"})
                expected_dimensions.append(dimensions)
        messages.extend(mcp_call_result_to_tool_result(
            {"content": content}, call_id=call.id, name="capture"
        ).messages)
    final_content = ([ThinkingPart(type="thinking", thinking=thinking_after)] if thinking_after else []) + [
        {"type": "text", "text": "Views compared."}
    ]
    final = Message(role="assistant", content=final_content)
    messages.append(final)

    prompt = renderer.build_generation_prompt(messages)
    rendered_text = renderer.tokenizer.decode([
        token for chunk in prompt.chunks if isinstance(chunk, ModelInputChunk) for token in chunk.tokens
    ])
    assert rendered_text.count(" to=self<|message|>") == (2 if thinking_length else 0)
    assert rendered_text.count("assistant to=capture<|message|>") == call_count
    if thinking_length:
        assert rendered_text.index(thinking_before) < rendered_text.index("assistant to=capture<|message|>")
        assert rendered_text.rindex("</tool_output>") < rendered_text.index(thinking_after)
        assert rendered_text.index(thinking_after) < rendered_text.index("Views compared.")
    images = [chunk for chunk in prompt.chunks if isinstance(chunk, ImageChunk)]
    assert [chunk.expected_tokens for chunk in images] == [width * height for width, height in expected_dimensions]
    for chunk, dimensions in zip(images, expected_dimensions, strict=True):
        with Image.open(io.BytesIO(chunk.data)) as decoded:
            assert decoded.size == dimensions

    for message, thinking in [(before, thinking_before), (final, thinking_after)]:
        converted = renderer.to_openai_message(message)
        assert converted.get("reasoning_content", "") == thinking
        if thinking:
            assert thinking not in (converted.get("content") or "")
    assert len(renderer.to_openai_message(before)["tool_calls"]) == call_count

    model_input, weights = renderer.build_supervised_example(messages, train_on_what=train_on)
    offset = 0
    trained_tokens = []
    for chunk in model_input.chunks:
        length = len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        chunk_weights = weights[offset : offset + length]
        assert len(chunk_weights) == length
        if isinstance(chunk, ImageChunk):
            assert not chunk_weights.any()
        else:
            trained_tokens.extend(token for token, weight in zip(chunk.tokens, chunk_weights, strict=True) if weight)
        offset += length
    assert offset == len(weights)
    trained_text = renderer.tokenizer.decode(trained_tokens)
    assert "tool-observation" not in trained_text
    assert "<tool_output" not in trained_text
    assert "Views compared." in trained_text
    if thinking_after:
        assert thinking_after in trained_text
        assert (thinking_before in trained_text) == (train_on == TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    assert (" to=capture<|message|>" in trained_text) == (train_on == TrainOnWhat.ALL_ASSISTANT_MESSAGES)


def test_atem_tool_call_round_trips():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    call = ToolCall(
        function=ToolCall.FunctionBody(
            name="weather.lookup",
            arguments='{"city": "Seattle", "days": 2}',
        )
    )
    message = Message(role="assistant", content="", tool_calls=[call])
    rendered = renderer.render_message(
        message,
        RenderContext(idx=1, is_last=True),
    )
    sampled = renderer.tokenizer.decode(rendered.output[0].tokens)

    parsed, success = renderer.parse_response(renderer.tokenizer.encode(sampled))

    assert success
    assert parsed["tool_calls"][0].function.name == "weather.lookup"
    assert json.loads(parsed["tool_calls"][0].function.arguments) == {
        "city": "Seattle",
        "days": 2,
    }


def test_tool_prefix_advertises_atem_schema_and_namespace():
    renderer = MuseGlimmerRenderer(ReversibleTokenizer())
    prefix = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "weather.lookup",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
        system_prompt="Be concise.",
    )

    prompt = _decode_prompt(renderer, prefix)

    assert "<atem:function_calls>" in prompt
    assert '"name":"weather.lookup"' in prompt
    assert '# Valid recipients: "self", "weather.*", "user".' in prompt
    assert prompt.count("# Valid recipients:") == 1
