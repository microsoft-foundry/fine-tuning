from interactive_training.renderers import ToolCall
from interactive_training.renderers.glm5 import GLM5Renderer


def _renderer() -> GLM5Renderer:
    return GLM5Renderer(tokenizer=object())


def test_to_openai_message_preserves_tool_calls_with_reasoning() -> None:
    tool_call = ToolCall(
        id="call_weather",
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"city":"Seattle"}',
        ),
    )

    result = _renderer().to_openai_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should check the forecast."},
                {"type": "text", "text": "Checking now."},
            ],
            "tool_calls": [tool_call],
        }
    )

    assert result == {
        "role": "assistant",
        "content": "Checking now.",
        "reasoning_content": "I should check the forecast.",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_weather",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Seattle"}',
                },
            }
        ],
    }


def test_to_openai_message_preserves_tool_response_metadata() -> None:
    result = _renderer().to_openai_message(
        {
            "role": "tool",
            "content": "Rain",
            "tool_call_id": "call_weather",
            "name": "get_weather",
        }
    )

    assert result == {
        "role": "tool",
        "content": "Rain",
        "tool_call_id": "call_weather",
        "name": "get_weather",
    }
