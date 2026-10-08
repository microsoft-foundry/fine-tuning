from interactive_training.tool_use import simple_tool_result


def test_simple_tool_result_preserves_ordered_multimodal_content():
    content = [
        {"type": "text", "text": "before"},
        {"type": "image", "image": "data:image/png;base64,AAAA"},
        {"type": "text", "text": "after"},
    ]

    result = simple_tool_result(
        content,
        call_id="call-1",
        name="screenshot",
    )

    assert result.messages == [
        {
            "role": "tool",
            "content": content,
            "tool_call_id": "call-1",
            "name": "screenshot",
        }
    ]
