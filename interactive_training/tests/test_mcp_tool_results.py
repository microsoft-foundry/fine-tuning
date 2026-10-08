import asyncio
import base64
import io
import json

import pytest
from mcp.types import ResourceLink

from interactive_training.renderers.base import ToolCall
from interactive_training.tool_use.agent_tool_message_env import AgentToolMessageEnv
from interactive_training.tool_use.mcp import (
    mcp_call_result_to_tool_result,
    mcp_content_to_parts,
    tools_from_mcp_session,
)
from interactive_training.tool_use.tools import handle_tool_call

Image = pytest.importorskip("PIL.Image")


def _encoded_image(image_format="PNG", size=(3, 2)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode()


_PNG = _encoded_image()


@pytest.mark.asyncio
@pytest.mark.parametrize("call_count", [3, 6])
@pytest.mark.parametrize("call_limit", [None, 2])
@pytest.mark.parametrize("thinking_length", [0, 1, 37, 257])
async def test_parallel_mcp_ragged_results_preserve_call_order(call_count, call_limit, thinking_length):
    executed_count = min(call_count, call_limit or call_count)
    all_started = asyncio.Event()
    completed = [asyncio.Event() for _ in range(executed_count)]
    started = []
    completion_order = []
    contents = []
    for index in range(call_count):
        content = [{"type": "text", "text": "detail " * (1 + index * 17)}]
        for image_index in range(index % 3):
            size = (3 + index * 19, 2 + image_index * 31 + index)
            content.append({"type": "image", "data": _encoded_image(size=size), "mimeType": "image/png"})
        contents.append(content)

    class Session:
        async def list_tools(self, *, cursor=None):
            return {"tools": [{
                "name": "capture", "description": "Return image and text results.",
                "inputSchema": {"type": "object", "properties": {"index": {"type": "integer"}}},
            }]}

        async def call_tool(self, name, arguments):
            index = arguments["index"]
            started.append(index)
            if len(started) == executed_count:
                all_started.set()
            await all_started.wait()
            if index + 1 < executed_count:
                await completed[index + 1].wait()
            completion_order.append(index)
            completed[index].set()
            return {"content": contents[index]}

    async def reward_fn(history):
        return 0.0, {}

    env = AgentToolMessageEnv(
        tools=await tools_from_mcp_session(Session()),
        initial_messages=[{"role": "user", "content": "Capture views"}],
        max_turns=3, max_tool_calls=call_limit, reward_fn=reward_fn,
    )
    await env.initial_observation()
    calls = [ToolCall(id=f"call-{index}", function=ToolCall.FunctionBody(
        name="capture", arguments=json.dumps({"index": index})
    )) for index in range(call_count)]
    thinking = "Inspect each view. " * thinking_length
    assistant_content = ([{"type": "thinking", "thinking": thinking}] if thinking else []) + [
        {"type": "text", "text": "Collecting views."}
    ]
    assistant_message = {"role": "assistant", "content": assistant_content, "tool_calls": calls}
    result = await asyncio.wait_for(
        env.step(assistant_message), timeout=5
    )
    assert result.next_messages[1] == assistant_message
    assert sorted(started) == list(range(executed_count))
    assert completion_order == list(reversed(range(executed_count)))
    messages = [message for message in result.next_messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in messages] == [f"call-{index}" for index in range(executed_count)]
    for index, message in enumerate(messages):
        expected_parts = []
        for part in contents[index]:
            if part["type"] == "text":
                expected_parts.append(part)
            else:
                expected_parts.append({"type": "image", "image": f"data:image/png;base64,{part['data']}"})
        assert message["content"] == expected_parts
    assert result.episode_done == (call_limit is not None)
    assert result.metrics.get("max_tool_calls", 0) == int(call_limit is not None)
    if not result.episode_done:
        final_content = ([{"type": "thinking", "thinking": "Compare results. " * (thinking_length + 2)}]
                         if thinking else []) + [{"type": "text", "text": "Views collected."}]
        final_message = {"role": "assistant", "content": final_content}
        final_result = await env.step(final_message)
        assert final_result.episode_done
        assert final_result.next_messages[-1] == final_message
        assert final_result.next_messages[1] == assistant_message
        assert [message for message in final_result.next_messages if message["role"] == "tool"] == messages


def test_mcp_call_result_preserves_text_image_order():
    result = mcp_call_result_to_tool_result(
        {
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "data": _PNG, "mimeType": "image/png"},
                {"type": "text", "text": "after"},
            ]
        },
        call_id="call-1",
        name="generate_image",
    )

    assert result.messages == [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{_PNG}",
                },
                {"type": "text", "text": "after"},
            ],
            "tool_call_id": "call-1",
            "name": "generate_image",
        }
    ]


@pytest.mark.asyncio
async def test_discovered_mcp_tool_normalizes_results_automatically():
    class Session:
        def __init__(self):
            self.list_cursors = []
            self.calls = []

        async def list_tools(self, *, cursor=None):
            self.list_cursors.append(cursor)
            if cursor is None:
                return {
                    "tools": [
                        {
                            "name": "capture_dashboard",
                            "description": "Capture a service dashboard.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"service": {"type": "string"}},
                                "required": ["service"],
                            },
                        }
                    ],
                    "nextCursor": "page-2",
                }
            return {"tools": []}

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {
                "content": [
                    {"type": "text", "text": "Current dashboard:"},
                    {"type": "image", "data": _PNG, "mimeType": "image/png"},
                ]
            }

    session = Session()
    tools = await tools_from_mcp_session(session)

    assert session.list_cursors == [None, "page-2"]
    assert len(tools) == 1
    assert tools[0].to_spec() == {
        "name": "capture_dashboard",
        "description": "Capture a service dashboard.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    }

    result = await handle_tool_call(
        {tools[0].name: tools[0]},
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(
                name="capture_dashboard",
                arguments=json.dumps({"service": "checkout"}),
            ),
        ),
    )

    assert session.calls == [("capture_dashboard", {"service": "checkout"})]
    assert result.messages == [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Current dashboard:"},
                {"type": "image", "image": f"data:image/png;base64,{_PNG}"},
            ],
            "tool_call_id": "call-1",
            "name": "capture_dashboard",
        }
    ]


@pytest.mark.asyncio
async def test_discovered_mcp_tool_returns_standard_error_result():
    class Session:
        async def list_tools(self, *, cursor=None):
            return {
                "tools": [
                    {
                        "name": "broken_tool",
                        "description": "Return malformed content.",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }

        async def call_tool(self, name, arguments):
            return {"content": [{"type": "image", "data": "invalid"}]}

    tool = (await tools_from_mcp_session(Session()))[0]

    result = await handle_tool_call(
        {tool.name: tool},
        ToolCall(
            id="call-2",
            function=ToolCall.FunctionBody(name="broken_tool", arguments="{}"),
        ),
    )

    assert result.messages[0]["role"] == "tool"
    assert result.messages[0]["tool_call_id"] == "call-2"
    assert result.messages[0]["name"] == "broken_tool"
    assert "MCP tool execution failed" in result.messages[0]["content"]
    assert result.metadata == {"error": "execution_failed"}


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (
            {
                "type": "resource",
                "resource": {
                    "uri": "memory://description",
                    "mimeType": "text/plain",
                    "text": "description",
                },
            },
            {"type": "text", "text": "description"},
        ),
        (
            {
                "type": "resource",
                "resource": {
                    "uri": "memory://image",
                    "mimeType": "image/png",
                    "blob": _PNG,
                },
            },
            {"type": "image", "image": f"data:image/png;base64,{_PNG}"},
        ),
        (
            {
                "type": "resource_link",
                "uri": "https://images.example.test/output.png",
                "name": "output.png",
                "mimeType": "image/png",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://images.example.test/output.png"},
            },
        ),
    ],
)
def test_mcp_resources_are_normalized(monkeypatch, block, expected):
    monkeypatch.setattr(
        "interactive_training.renderers.base._resolve_public_image_url",
        lambda url: "203.0.113.1",
    )
    assert mcp_content_to_parts([block]) == [expected]


def test_mcp_resource_link_model_normalizes_uri(monkeypatch):
    monkeypatch.setattr(
        "interactive_training.renderers.base._resolve_public_image_url",
        lambda url: "203.0.113.1",
    )
    resource_link = ResourceLink(
        type="resource_link",
        uri="https://images.example.test/output.png",
        name="output.png",
        mimeType="image/png",
    )

    assert mcp_content_to_parts([resource_link]) == [
        {
            "type": "image_url",
            "image_url": {"url": "https://images.example.test/output.png"},
        }
    ]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "data": "not-base64", "mimeType": "image/png"},
        {"type": "image", "data": _PNG, "mimeType": "image/jpeg"},
        {"type": "image", "data": _PNG, "mimeType": "image/gif"},
        {
            "type": "resource_link",
            "uri": "http://images.example.test/output.png",
            "mimeType": "image/png",
        },
        {
            "type": "resource_link",
            "uri": "https://127.0.0.1/output.png",
            "mimeType": "image/png",
        },
        {
            "type": "resource_link",
            "uri": "https://example.test/audio.mp3",
            "mimeType": "audio/mpeg",
        },
        {"type": "audio", "data": "data", "mimeType": "audio/wav"},
    ],
)
def test_mcp_unsupported_or_invalid_content_is_rejected(block):
    with pytest.raises(ValueError):
        mcp_content_to_parts([block])
