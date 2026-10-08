"""Local image-returning MCP server for the alien-color recall recipe."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool
from PIL import Image, ImageDraw

from interactive_training.recipes.mcp_image_tool.data import ALIEN_WORD_TO_COLOR
from interactive_training.tool_use import ToolInput, tools_from_mcp_session

SERVER = Server("interactive-post-training-mcp-alien-colors")
TOOL_DESCRIPTION = (
    "Look up an alien word by returning only the color it represents "
    "as an image. Submit one chosen word. Prefer known matches and avoid repeated calls."
)

COLOR_HEX = {
    "red": "#e53935",
    "orange": "#fb8c00",
    "yellow": "#fdd835",
    "green": "#43a047",
    "blue": "#1e88e5",
    "purple": "#8e24aa",
    "pink": "#ec407a",
    "black": "#111111",
    "white": "#ffffff",
    "brown": "#6d4c41",
}


def color_swatch_png(alien_word: str) -> bytes:
    color = ALIEN_WORD_TO_COLOR[alien_word]
    image = Image.new("RGB", (640, 360), COLOR_HEX[color])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 639, 359), outline="#777777", width=4)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@SERVER.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="reveal_alien_color",
            description=TOOL_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "alien_word": {
                        "type": "string",
                        "enum": list(ALIEN_WORD_TO_COLOR),
                    }
                },
                "required": ["alien_word"],
                "additionalProperties": False,
            },
        )
    ]


@SERVER.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any],
) -> list[TextContent | ImageContent]:
    if name != "reveal_alien_color":
        raise ValueError(f"Unknown tool: {name}")
    alien_word = arguments.get("alien_word")
    if not isinstance(alien_word, str) or alien_word not in ALIEN_WORD_TO_COLOR:
        raise ValueError("alien_word must be one of the advertised alien words")
    image_data = base64.b64encode(color_swatch_png(alien_word)).decode("ascii")
    return [
        TextContent(type="text", text=f"Oracle result for {alien_word}:"),
        ImageContent(type="image", data=image_data, mimeType="image/png"),
    ]


async def run_server() -> None:
    async with stdio_server() as (read, write):
        await SERVER.run(
            read,
            write,
            InitializationOptions(
                server_name="interactive-post-training-mcp-alien-colors",
                server_version="1.0.0",
                capabilities=SERVER.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


async def run_smoke() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "interactive_training.recipes.mcp_image_tool.server", "--server"],
    )
    async with (
        stdio_client(server) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await tools_from_mcp_session(session)
        color_tool = next(tool for tool in tools if tool.name == "reveal_alien_color")
        tool_result = await color_tool.run(
            ToolInput(arguments={"alien_word": "varkesh"}, call_id="call_123")
        )

    content = tool_result.messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "text",
        "text": "Oracle result for varkesh:",
    }
    image_part = content[1]
    assert image_part["type"] == "image"
    image_uri = image_part["image"]
    assert isinstance(image_uri, str)
    encoded = image_uri.partition(",")[2]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (640, 360)
        assert image.getpixel((320, 180)) == (229, 57, 53)

    print("MCP image normalized for Interactive Training: alien_word=varkesh, color=red")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a local protocol smoke test"
    )
    args = parser.parse_args()
    if args.server:
        asyncio.run(run_server())
    elif args.smoke:
        asyncio.run(run_smoke())
    else:
        parser.error("one of --server or --smoke is required")


if __name__ == "__main__":
    main()
