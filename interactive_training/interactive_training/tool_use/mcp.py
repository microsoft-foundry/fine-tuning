"""Adapters from MCP tool results to Interactive Training's provider-neutral message schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from interactive_training.renderers.base import (
    ContentPart,
    Message,
    ToolSpec,
    _get_encoded_image_format,
    _load_image_reference,
    _open_verified_image,
    _validate_public_image_url,
)
from interactive_training.tool_use.types import ToolInput, ToolResult

_SUPPORTED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
_IMAGE_FORMAT_BY_MEDIA_TYPE = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class MCPClientSession(Protocol):
    """Structural subset of an MCP client session used by Interactive Training."""

    async def list_tools(self, *, cursor: str | None = None) -> object: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object: ...


class MCPTool:
    """A discovered MCP tool exposed through Interactive Training's standard Tool protocol."""

    def __init__(
        self,
        session: MCPClientSession,
        *,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
    ) -> None:
        self._session = session
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters_schema

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        try:
            result = await self._session.call_tool(self.name, input.arguments)
            return mcp_call_result_to_tool_result(
                result,
                call_id=input.call_id or "",
                name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            from interactive_training.tool_use.tools import error_tool_result

            return error_tool_result(
                f"MCP tool execution failed: {exc}",
                call_id=input.call_id or "",
                name=self.name,
                error_type="execution_failed",
            )


def _as_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"{label} must be an MCP mapping or model")


def _field(value: Mapping[str, Any], camel_name: str, snake_name: str) -> Any:
    return value.get(camel_name, value.get(snake_name))


def _image_data_uri(data: object, mime_type: object) -> str:
    if not isinstance(data, str):
        raise TypeError("MCP image data must be a base64 string")
    if not isinstance(mime_type, str):
        raise TypeError("MCP image content must include mimeType")
    normalized_mime_type = mime_type.lower()
    if normalized_mime_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
        raise ValueError("MCP images must be JPEG, PNG, or WebP")
    data_uri = f"data:{normalized_mime_type};base64,{data}"
    image_bytes = _load_image_reference(data_uri)
    if (
        _get_encoded_image_format(image_bytes)
        != _IMAGE_FORMAT_BY_MEDIA_TYPE[normalized_mime_type]
    ):
        raise ValueError("MCP image data does not match its declared MIME type")
    with _open_verified_image(image_bytes):
        pass
    return data_uri


def _resource_to_part(block: Mapping[str, Any]) -> ContentPart:
    resource = _as_mapping(block.get("resource"), label="MCP embedded resource")
    text = resource.get("text")
    if isinstance(text, str):
        return {"type": "text", "text": text}

    mime_type = _field(resource, "mimeType", "mime_type")
    return {
        "type": "image",
        "image": _image_data_uri(resource.get("blob"), mime_type),
    }


def _resource_link_to_part(block: Mapping[str, Any]) -> ContentPart:
    uri = block.get("uri")
    mime_type = _field(block, "mimeType", "mime_type")
    if not isinstance(uri, str):
        raise TypeError("MCP resource links must include a URI")
    if (
        not isinstance(mime_type, str)
        or mime_type.lower() not in _SUPPORTED_IMAGE_MEDIA_TYPES
    ):
        raise ValueError("Only JPEG, PNG, and WebP MCP resource links are supported")
    _validate_public_image_url(uri)
    return {"type": "image_url", "image_url": {"url": uri}}


def mcp_content_to_parts(content: Sequence[object]) -> list[ContentPart]:
    """Normalize MCP text, image, and resource blocks into Interactive Training content parts."""
    parts: list[ContentPart] = []
    for value in content:
        block = _as_mapping(value, label="MCP content block")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("MCP text content must include text")
            parts.append({"type": "text", "text": text})
        elif block_type == "image":
            parts.append(
                {
                    "type": "image",
                    "image": _image_data_uri(
                        block.get("data"),
                        _field(block, "mimeType", "mime_type"),
                    ),
                }
            )
        elif block_type == "resource":
            parts.append(_resource_to_part(block))
        elif block_type == "resource_link":
            parts.append(_resource_link_to_part(block))
        else:
            raise ValueError(f"Unsupported MCP content type: {block_type!r}")
    return parts


def mcp_call_result_to_tool_result(
    result: object,
    *,
    call_id: str = "",
    name: str = "",
) -> ToolResult:
    """Convert an MCP CallToolResult mapping/model into a Interactive Training ToolResult."""
    result_mapping = _as_mapping(result, label="MCP call result")
    content = result_mapping.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise TypeError("MCP call results must contain a content list")
    is_error = bool(_field(result_mapping, "isError", "is_error"))
    return ToolResult(
        messages=[
            Message(
                role="tool",
                content=mcp_content_to_parts(content),
                tool_call_id=call_id,
                name=name,
            )
        ],
        metadata={"mcp_is_error": True} if is_error else {},
    )


async def tools_from_mcp_session(session: MCPClientSession) -> list[MCPTool]:
    """Discover all tools in an MCP session and adapt them for Interactive Training."""
    tools: list[MCPTool] = []
    seen_names: set[str] = set()
    cursor: str | None = None

    while True:
        result = _as_mapping(
            await session.list_tools(cursor=cursor),
            label="MCP list tools result",
        )
        tool_definitions = result.get("tools")
        if not isinstance(tool_definitions, Sequence) or isinstance(
            tool_definitions, (str, bytes)
        ):
            raise TypeError("MCP list tools results must contain a tools list")

        for value in tool_definitions:
            definition = _as_mapping(value, label="MCP tool definition")
            name = definition.get("name")
            description = definition.get("description") or ""
            parameters_schema = _field(definition, "inputSchema", "input_schema")
            if not isinstance(name, str) or not name:
                raise ValueError("MCP tool definitions must include a name")
            if name in seen_names:
                raise ValueError(f"MCP returned duplicate tool name: {name}")
            if not isinstance(description, str):
                raise TypeError(f"MCP tool {name!r} has an invalid description")
            if not isinstance(parameters_schema, Mapping):
                raise TypeError(f"MCP tool {name!r} must include inputSchema")
            seen_names.add(name)
            tools.append(
                MCPTool(
                    session,
                    name=name,
                    description=description,
                    parameters_schema=dict(parameters_schema),
                )
            )

        next_cursor = _field(result, "nextCursor", "next_cursor")
        if next_cursor is None:
            return tools
        if not isinstance(next_cursor, str) or not next_cursor:
            raise ValueError("MCP nextCursor must be a non-empty string")
        cursor = next_cursor
