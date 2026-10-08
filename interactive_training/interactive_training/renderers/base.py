"""
Base types, utilities, and abstract Renderer class for message rendering.

Use viz_sft_dataset to visualize the output of different renderers. E.g.,
    python -m interactive_training.supervised.viz_sft_dataset dataset_path=Tulu3Builder renderer_name=role_colon
"""

import base64
import binascii
import hashlib
import http.client
import io
import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, NotRequired, Optional, Protocol, Required, TYPE_CHECKING, TypedDict, Union

import pydantic
import torch
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.image_types import ImageChunk
from interactive_training.tokenizer_utils import Tokenizer

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 25_000_000
IMAGE_FETCH_TIMEOUT_SECONDS = 10
IMAGE_FETCH_MAX_ATTEMPTS = 3
_ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
_ALLOWED_DATA_IMAGE_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
_IMAGE_READ_CHUNK_BYTES = 64 * 1024
_NAT64_WELL_KNOWN_NETWORK = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL_USE_NETWORK = ipaddress.ip_network("64:ff9b:1::/48")

# Tool types follow the OpenAI-style tool-call shape used by the cookbook.


class StrictBase(pydantic.BaseModel):
    """
    Pydantic base class that's immutable and doesn't silently ignore extra fields.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)


class ToolCall(StrictBase):
    """
    Structured tool invocation following OpenAI/kosong format.

    This represents a request to invoke a tool/function. The structure follows
    the OpenAI function calling format for compatibility with various LLM APIs.

    Example:
        tool_call = ToolCall(
            function=ToolCall.FunctionBody(
                name="search",
                arguments='{"query_list": ["python async", "pydantic validation"]}'
            ),
            id="call_abc123"
        )
    """

    class FunctionBody(pydantic.BaseModel):
        """
        Tool call function body containing the tool name and arguments.

        The arguments field must be a valid JSON string that will be parsed
        by the tool implementation.
        """

        name: str
        """The name of the tool to be called."""
        arguments: str
        """Arguments of the tool call in JSON string format."""

    type: Literal["function"] = "function"
    """Tool call type, must be 'function' for compatibility."""

    id: str | None = None
    """Optional unique identifier for tracking this specific tool call."""

    function: FunctionBody
    """The function body containing tool name and arguments."""


class UnparsedToolCall(StrictBase):
    """
    Represents a tool call that failed to parse from model output.

    When a model generates text that looks like a tool call but cannot be
    parsed (e.g., invalid JSON), this class captures the raw text and error
    for debugging and optional re-rendering.

    Example:
        unparsed = UnparsedToolCall(
            raw_text='<tool_call>{"name": "search", invalid json}</tool_call>',
            error="Invalid JSON: Expecting property name"
        )
    """

    raw_text: str
    """The original text from the model that failed to parse."""

    error: str
    """Description of what went wrong during parsing."""


class TextPart(TypedDict):
    """A chunk of text content in a message, usually meant to be visible to the user
    (unlike ThinkingPart, which is internal reasoning)."""

    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    """
    A chunk of image content in a message.
    """

    type: Literal["image"]
    image: "str | Image.Image"


class ImageURL(TypedDict, total=False):
    """OpenAI-compatible image URL descriptor used by Foundry JSONL data."""

    url: Required[str]
    detail: Literal["auto", "low", "high"]


class ImageURLPart(TypedDict):
    """An image referenced by a public URL or base64 data URI."""

    type: Literal["image_url"]
    image_url: ImageURL


class ThinkingPart(TypedDict):
    """Model's internal reasoning (chain-of-thought) as a content part."""

    type: Literal["thinking"]
    thinking: str  # The thinking/reasoning content


# Container for a part of a multimodal message content.
# Tool calls live exclusively in message["tool_calls"] / message["unparsed_tool_calls"].
ContentPart = TextPart | ImagePart | ImageURLPart | ThinkingPart


def get_image_source(part: ImagePart | ImageURLPart) -> "str | Image.Image":
    """Return the image value from an internal or OpenAI-compatible part."""
    if part["type"] == "image":
        return part["image"]
    return part["image_url"]["url"]


# Streaming types to enable incremental parsing of model output for real-time display.


@dataclass
class StreamingMessageHeader:
    """Emitted at the start of a new message during streaming.

    This signals that a new message is beginning and provides the author info.
    """

    role: str
    name: str | None = None


@dataclass
class StreamingTextDelta:
    """Incremental text content during streaming.

    Contains only the new text since the last delta, not the accumulated text.
    The recipient should concatenate deltas to build the full content.
    """

    text: str
    content_index: int = 0
    """Index of this content block within the message. Increments when content type changes."""


@dataclass
class StreamingThinkingDelta:
    """Incremental thinking/reasoning content during streaming.

    Contains only the new thinking text since the last delta.
    """

    thinking: str
    content_index: int = 0
    """Index of this content block within the message. Increments when content type changes."""


# Union of all streaming update types.
# A streaming parser yields these in sequence:
# 1. StreamingMessageHeader (once at start)
# 2. StreamingTextDelta / StreamingThinkingDelta (as content arrives)
# 3. Message (once at end, containing the complete parsed message)
MessageDelta = Union[StreamingMessageHeader, StreamingTextDelta, StreamingThinkingDelta, "Message"]


# Unicode replacement character - indicates incomplete/invalid UTF-8 sequence
_REPLACEMENT_CHAR = "\ufffd"


@dataclass
class Utf8TokenDecoder:
    """Handles incremental UTF-8 decoding from tokens.

    Tokens can split multi-byte UTF-8 sequences (e.g., a 3-byte character
    might be split across 2 tokens). This class buffers tokens until a
    valid UTF-8 string can be decoded.

    Detection strategy:
    1. Try decoding all pending + new tokens
    2. If result contains trailing U+FFFD (replacement char), it's incomplete
    3. Scan backwards to find longest prefix without trailing replacement chars
    4. Emit that prefix, buffer the rest

    This handles tiktoken-style tokenizers that return replacement chars
    instead of throwing exceptions for incomplete UTF-8.
    """

    tokenizer: "Tokenizer"
    _pending_tokens: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._pending_tokens is None:
            self._pending_tokens = []

    # Max tokens to try removing from the end when looking for decodable prefix.
    # UTF-8 chars are max 4 bytes, tokens typically 1-4 bytes each,
    # so 8 tokens is plenty to cover any incomplete trailing sequence.
    _MAX_TRAILING_TOKENS_TO_TRY: int = 8

    def _is_valid_decode(self, text: str) -> bool:
        """Check if decoded text represents a complete UTF-8 sequence.

        Returns False if the text ends with a replacement character,
        which indicates an incomplete multi-byte sequence that needs
        more tokens to complete.
        """
        return not text.endswith(_REPLACEMENT_CHAR)

    def decode(self, tokens: list[int]) -> str | None:
        """Decode tokens to string, buffering incomplete UTF-8 sequences.

        Args:
            tokens: New tokens to decode.

        Returns:
            Decoded string if complete UTF-8 sequences are available,
            None if all tokens were buffered (incomplete sequence).
        """
        self._pending_tokens.extend(tokens)

        # Try to decode all pending tokens (common case)
        try:
            text = self.tokenizer.decode(self._pending_tokens)
            if self._is_valid_decode(text):
                self._pending_tokens = []
                return text
            # Has trailing replacement chars - fall through to find valid prefix
        except Exception:
            pass

        # Scan backwards to find longest decodable prefix without replacement chars.
        # We only need to try removing a few tokens since UTF-8 sequences are at
        # most 4 bytes and tokens are typically 1-4 bytes each.
        for remove in range(
            1, min(len(self._pending_tokens), self._MAX_TRAILING_TOKENS_TO_TRY) + 1
        ):
            prefix = self._pending_tokens[:-remove]
            if not prefix:
                break
            try:
                text = self.tokenizer.decode(prefix)
                if self._is_valid_decode(text):
                    self._pending_tokens = self._pending_tokens[-remove:]
                    return text
            except Exception:
                continue

        # All tokens buffered - need more data
        return None

    def flush(self) -> str:
        """Force decode any remaining tokens.

        Call this at end of stream. May produce replacement characters
        for incomplete sequences.
        """
        if not self._pending_tokens:
            return ""
        try:
            text = self.tokenizer.decode(self._pending_tokens)
        except Exception:
            # Last resort: decode with errors='replace' behavior
            # Most tokenizers handle this, but fall back to empty string
            text = ""
        self._pending_tokens = []
        return text

    def reset(self) -> None:
        """Clear any buffered tokens."""
        self._pending_tokens = []

    def has_pending(self) -> bool:
        """Check if there are buffered tokens waiting for more data."""
        return len(self._pending_tokens) > 0


# NOTE: we use a broad type definition for the role to be flexible
# Common roles are "user", "assistant", "system", "tool"
Role = str

# Content is a string or a list of parts
Content = str | list[ContentPart]


class Message(TypedDict):
    """
    Container for a single turn in a multi-turn conversation.

    Args:

    role: Role
        String that denotes the source of the message, typically system, user, assistant, and tool.
    content: Content
        Content of the message, can be a string, or a list of ContentPart.
        When content is a list, it can contain TextPart, ImagePart, and ThinkingPart elements.
        ThinkingPart represents the model's internal reasoning (chain-of-thought).
    tool_calls: NotRequired[list[ToolCall]]
        Optional sequence of successfully parsed tool calls generated by the model.
    unparsed_tool_calls: NotRequired[list[UnparsedToolCall]]
        Optional sequence of tool calls that failed to parse (e.g., invalid JSON).
        The raw text is preserved for debugging or re-rendering.
    trainable: NotRequired[bool]
        Optional indicator whether this message should contribute to the training loss.
    tool_call_id: NotRequired[str]
        For tool result messages (role="tool"): ID correlating this result to a specific
        tool call. Used by renderers whose wire format references calls by ID.
        The value should match ToolCall.id from the
        assistant's tool_calls. Not all formats use IDs - GptOss/Harmony does not.
    name: NotRequired[str]
        For tool result messages (role="tool"): The function name that was called.
        Required by GptOss (renders "<|start|>functions.{name}..."), optional for others.
        When constructing tool results, include both name and tool_call_id when available
        since different renderers require different fields.

    """

    role: Role
    content: Content

    tool_calls: NotRequired[list[ToolCall]]
    unparsed_tool_calls: NotRequired[list["UnparsedToolCall"]]
    trainable: NotRequired[bool]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


@dataclass
class RenderContext:
    """
    Context passed to render_message for rendering a single message.

    This allows renderers to access information about the message's position
    in the conversation without changing the render_message signature for
    each new piece of context needed.
    """

    idx: int
    """Index of the message in the conversation (0-based)."""

    is_last: bool
    """Whether this is the last message in the conversation."""

    prev_message: Message | None = None
    """The previous message in the conversation, if any."""

    last_user_index: int = -1
    """Index of the last user message in the conversation, or -1 if none.

    Used by renderers (e.g. Qwen3.5) whose templates inject extra tokens
    into assistant messages that appear after the last user message.
    """


class ToolSpec(TypedDict):
    """
    Tool specification following the OpenAI function calling format.

    This represents a tool that can be called by the model, including its name,
    description, and parameter schema.

    Example:
        tool_spec: ToolSpec = {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }
    """

    name: str
    """The name of the tool."""
    description: str
    """A description of what the tool does."""
    parameters: dict
    """JSON Schema object describing the tool's parameters."""


def ensure_text(content: Content) -> str:
    """
    Assert that content is text-only and return it as a string.

    Raises ValueError if content contains images or multiple parts.
    Use this to validate that message content is text-only before
    processing it in code paths that don't support multimodal content.
    """
    if isinstance(content, str):
        return content
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    raise ValueError(f"Expected text content, got multimodal content with {len(content)} parts")


def ensure_list(content: Content) -> list[ContentPart]:
    """Normalize content to list form. Wraps string content in a TextPart."""
    if isinstance(content, str):
        return [TextPart(type="text", text=content)]
    return content


def remove_thinking(parts: list[ContentPart]) -> list[ContentPart]:
    """Filter out ThinkingPart elements from a content part list."""
    return [p for p in parts if p["type"] != "thinking"]


def get_text_content(message: Message) -> str:
    """Extract text content from message, stripping thinking parts.

    Use this after parse_response when you only need the text output,
    ignoring any thinking/reasoning content.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def format_content_as_string(content: Content, separator: str = "\n") -> str:
    """Format message content as a string, preserving all part types.

    Unlike get_text_content which only extracts text parts, this formats
    all content parts (thinking, text) as a readable string.

    This is useful for compatibility with APIs that expect string content
    (e.g., OpenAI Chat Completions API), but we don't recommend it if you
    need to ensure correctness - prefer working with structured content directly
    and using build_generation_prompt to convert to tokens.

    Args:
        content: Message content (string or list of ContentPart).
        separator: String to join parts with. Default is newline.

    Returns:
        Formatted string representation of all content parts.
    """
    if isinstance(content, str):
        return content

    parts = []
    for p in content:
        if p["type"] == "thinking":
            parts.append(f"<think>{p['thinking']}</think>")
        elif p["type"] == "text":
            parts.append(p["text"])
        else:
            raise ValueError(f"Unknown content part type: {p['type']}")
    return separator.join(parts)


def _parse_tool_call_json(tool_call_str: str, raw_text: str) -> ToolCall | UnparsedToolCall:
    """Parse tool call JSON. Returns UnparsedToolCall on failure."""
    try:
        tool_call = json.loads(tool_call_str.strip())
    except json.JSONDecodeError as e:
        return UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")

    if not isinstance(tool_call, dict):
        return UnparsedToolCall(raw_text=raw_text, error="Tool call is not a JSON object")

    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    tool_id = tool_call.get("id")

    if not isinstance(name, str):
        return UnparsedToolCall(raw_text=raw_text, error="Missing or invalid 'name' field")
    if not isinstance(arguments, dict):
        return UnparsedToolCall(raw_text=raw_text, error="Missing or invalid 'arguments' field")

    if tool_id is not None and not isinstance(tool_id, str):
        tool_id = None

    # TODO: arguments is already a dict from json.loads above, but ToolCall.FunctionBody.arguments
    # expects a JSON string. This round-trip (loads then dumps) is wasteful. Consider changing
    # FunctionBody.arguments to accept dict directly, or parse tool calls more lazily.
    # We may want to revisit the decision to store arguments as unparsed JSON string.
    return ToolCall(
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
        id=tool_id,
    )


def parse_content_blocks(
    content: str,
) -> tuple[list[ContentPart], list[ToolCall | UnparsedToolCall]] | None:
    """
    Parse a string with <think>...</think> and <tool_call>...</tool_call> tags.

    Handles interleaved thinking, tool call, and text blocks. Content parts
    (ThinkingPart, TextPart) are returned in the first element; tool calls
    (ToolCall, UnparsedToolCall) are returned separately in the second element,
    preserving their relative order.

    Whitespace in non-tool-call regions is preserved exactly - roundtrip
    (parse then render) is identity for the content parts.

    Args:
        content: String potentially containing <think> and/or <tool_call> blocks.

    Returns:
        Tuple of (content_parts, tool_calls), or None if no special tags are found.
        content_parts contains only ThinkingPart/TextPart.
        tool_calls contains ToolCall and UnparsedToolCall in order.

    Example:
        >>> parse_content_blocks("<think>step 1</think>answer<tool_call>{...}</tool_call>more")
        (
            [ThinkingPart(type="thinking", thinking="step 1"),
             TextPart(type="text", text="answer"),
             TextPart(type="text", text="more")],
            [ToolCall(...)],
        )
    """
    if "<think>" not in content and "<tool_call>" not in content:
        return None  # No special blocks, caller should use original string

    parts: list[ContentPart] = []
    tool_calls: list[ToolCall | UnparsedToolCall] = []
    pos = 0

    # Pattern to find both <think>...</think> and <tool_call>...</tool_call> blocks
    pattern = re.compile(r"<think>(.*?)</think>|<tool_call>(.*?)</tool_call>", re.DOTALL)

    for match in pattern.finditer(content):
        # Add any text before this block (preserve whitespace for identity roundtrip)
        text_before = content[pos : match.start()]
        if text_before:  # Skip only truly empty strings
            parts.append(TextPart(type="text", text=text_before))

        if match.group(1) is not None:
            # This is a <think> block
            thinking = match.group(1)
            if thinking:  # Skip empty thinking blocks
                parts.append(ThinkingPart(type="thinking", thinking=thinking))
        else:
            # This is a <tool_call> block — goes into separate tool_calls list
            tool_call_json = match.group(2)
            raw_text = match.group(0)  # Full match including tags
            tool_calls.append(_parse_tool_call_json(tool_call_json, raw_text))

        pos = match.end()

    # Add any remaining text after the last block
    remaining = content[pos:]
    if remaining:  # Skip only truly empty strings
        parts.append(TextPart(type="text", text=remaining))

    return parts, tool_calls


def parse_think_blocks(content: str) -> list[ContentPart] | None:
    """
    Parse a string with only <think>...</think> tags into ThinkingPart/TextPart list.

    This is a simpler version of parse_content_blocks for renderers that use
    non-standard tool call formats (like DeepSeek's <｜tool▁calls▁begin｜>).

    Whitespace is preserved exactly - roundtrip (parse then render) is identity.

    Args:
        content: String potentially containing <think>...</think> blocks.

    Returns:
        List of ThinkingPart and TextPart in order. None if no <think> tags found.
    """
    if "<think>" not in content:
        return None

    parts: list[ContentPart] = []
    pos = 0
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    for match in pattern.finditer(content):
        text_before = content[pos : match.start()]
        if text_before:  # Skip only truly empty strings
            parts.append(TextPart(type="text", text=text_before))

        thinking = match.group(1)
        if thinking:  # Skip empty thinking blocks
            parts.append(ThinkingPart(type="thinking", thinking=thinking))

        pos = match.end()

    remaining = content[pos:]
    if remaining:  # Skip only truly empty strings
        parts.append(TextPart(type="text", text=remaining))

    return parts


def _tool_call_payload(tool_call: ToolCall) -> dict[str, object]:
    """Minimal JSON payload for embedding in <tool_call> blocks."""
    # Convert from nested structure to flat format for compatibility
    return {
        "name": tool_call.function.name,
        "arguments": json.loads(tool_call.function.arguments),
    }


@dataclass(frozen=True)
class RenderedMessage:
    """
    Container for parts of a rendered message, structured for loss masking.

    A rendered message is split into header and output to control which tokens receive
    training loss. In the simplest case (where the full conversation is formed by
    concatenation), building a supervised example from messages [m_0, ..., m_{n-1}]
    produces:

        tokens = BOS + header_0 + output_0 + header_1 + output_1 + ... + header_{n-1} + output_{n-1}

    However, some renderers modify this structure. For example, Qwen3Renderer strips
    thinking blocks from historical assistant messages. Such renderers must override
    build_supervised_example to match their build_generation_prompt behavior.

    Attributes:
        output: What the model generates for this turn: the message text/images plus
            end-of-turn tokens. This is the trainable portion.
            Examples: " Hello world\\\\n\\\\n" (RoleColon), "Hello world<|eot_id|>" (Llama3).
        header: Role identifier and delimiters that introduce the turn. This is what the
            model sees but does not generate.
            Examples: "User:" (RoleColon), "<|start_header_id|>user<|end_header_id|>\\\\n\\\\n" (Llama3).
            Typically receives zero training weight.
        stop_overlap: Edge case field for formats where the stop sequence spans message
            boundaries. Most renderers (Llama3, Qwen3, DeepSeek, etc.) don't use this—their
            stop tokens are included in output.

            Only RoleColonRenderer uses this. Its stop sequence is "\\\\n\\\\nUser:", where "\\\\n\\\\n"
            ends the output but "User:" would duplicate the next message's header. To avoid
            duplication, "User:" is stored here and only appended for the last message in
            supervised training. The name "stop_overlap" reflects that these tokens are the
            overlap between the stop sequence and the next message's header.
    """

    output: list[ModelInputChunk]
    """What the model generates for this turn."""

    header: ModelInputChunk | None = None
    """Role identifier and delimiters that introduce the turn."""

    stop_overlap: ModelInputChunk | None = None
    """Tokens that overlap between stop sequence and next message's header."""


class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    LAST_ASSISTANT_TURN = "last_assistant_turn"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    ALL_USER_AND_SYSTEM_MESSAGES = "all_user_and_system_messages"
    CUSTOMIZED = "customized"


class Renderer(ABC):
    """
    Abstract base class for rendering message lists into training and sampling prompts.

    Subclasses must implement:
    - get_stop_sequences(): Return stop tokens/strings for sampling
    - render_message(): Break a message into header/output/stop_overlap components
    - parse_response(): Convert sampled tokens back into a Message

    The default build_generation_prompt and build_supervised_example implementations
    assume simple concatenation of rendered messages. Override these if your renderer
    modifies the conversation structure (e.g., stripping thinking blocks from history).
    """

    tokenizer: Tokenizer
    image_cache_dir: str | None = None

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def image_to_chunk(
        self,
        image_or_str: "Image.Image | str",
        image_processor: "ImageProcessorProtocol",
    ) -> ImageChunk:
        if self.image_cache_dir is None:
            return image_to_chunk(image_or_str, image_processor)
        return image_to_chunk(
            image_or_str,
            image_processor,
            cache_dir=self.image_cache_dir,
        )

    @property
    def has_extension_property(self) -> bool:
        """Whether this renderer satisfies the sequence extension property.

        A renderer has the extension property if, for any multi-turn conversation,
        calling build_generation_prompt at each successive assistant turn produces
        token sequences where each is a prefix of the next. This enables:
        - Merging multiple timesteps into a single training datum
        - KV-cache reuse during sampling
        - O(T) compute scaling instead of O(T^2) for T-turn trajectories

        Renderers that strip thinking blocks from history (like Qwen3Renderer with
        strip_thinking_from_history=True) do NOT have this property because the
        observation at timestep 2 is not a prefix of timestep 1's full sequence.

        See docs/rl/sequence-extension.mdx for details.
        """
        return False

    @property
    def _bos_tokens(self) -> list[int]:
        return []

    @abstractmethod
    def get_stop_sequences(self) -> list[str] | list[int]:
        """Return the stop sequences used when sampling from this renderer."""
        ...

    @abstractmethod
    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """
        Render a single message into its header/output/stop_overlap components.

        This method breaks down a message into parts for loss masking. See RenderedMessage
        for detailed semantics of each component.

        Args:
            message: The message to render.
            ctx: Context about the message's position in the conversation, including:
                - idx: The index of this message (0-based)
                - is_last: Whether this is the last message
                - prev_message: The previous message, if any

        Returns:
            RenderedMessage with header, output, and optionally stop_overlap.
        """
        ...

    @abstractmethod
    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """
        Parse sampled tokens back into a Message.

        Args:
            response: Token IDs returned from sampling.

        Returns:
            A tuple of (message, success). If success is False, the response could not
            be parsed (e.g., missing stop token), but a best-effort message is still returned.
        """
        ...

    def parse_response_with_tools(
        self, response: list[int], tools: list[ToolSpec]
    ) -> tuple[Message, bool]:
        """Parse a response with the tool schemas used to build its prompt.

        Renderers whose wire format preserves argument types can use the default
        implementation. Formats with untyped scalar values may override this to
        recover types from the declared JSON Schema.
        """
        return self.parse_response(response)

    def to_openai_message(self, message: Message) -> dict:
        """
        Convert a Message to OpenAI chat completions API format.

        The returned object can be passed into the transformers library's
        apply_chat_template function, which is useful for testing purposes.

        It's also useful for querying models that are being served through
        OpenAI-compatible APIs (OpenRouter, vLLM, etc.).

        The base implementation handles:
        - Basic role/content conversion
        - tool_calls conversion from ToolCall objects to OpenAI dict format
        - tool_call_id and name for tool response messages

        Subclasses should override this to handle model-specific features like
        reasoning_content for thinking models.

        Args:
            message: The Message to convert.

        Returns:
            A dict in OpenAI API message format.
        """
        result: dict = {"role": message["role"]}

        # Handle content
        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        elif any(p["type"] in {"image", "image_url"} for p in content):
            parts: list[dict[str, Any]] = []
            for p in content:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "thinking":
                    parts.append(
                        {
                            "type": "text",
                            "text": f"<think>{p['thinking']}</think>",
                        }
                    )
                elif p["type"] == "image_url":
                    parts.append(dict(p))
                elif isinstance(p["image"], str):
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": p["image"]},
                        }
                    )
                else:
                    raise ValueError(
                        "PIL images must be converted to a data URI before OpenAI serialization"
                    )
            result["content"] = parts
        else:
            # Structured content with ThinkingPart/TextPart/etc.
            # Base implementation: concatenate text parts, render thinking as <think> tags
            parts = []
            for p in content:
                if p["type"] == "text":
                    parts.append(p["text"])
                elif p["type"] == "thinking":
                    parts.append(f"<think>{p['thinking']}</think>")
            result["content"] = "".join(parts)

        # Handle tool_calls (convert ToolCall objects to OpenAI format)
        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message["tool_calls"]
            ]

        # Handle tool response fields
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create message(s) with tool specifications to prepend to conversations.

        Returns one or more messages to prepend to the conversation. This is the
        standard way to add tools - the returned messages should be placed at the
        start of your message list before user/assistant messages.

        Args:
            tools: List of tool specifications.
            system_prompt: The system prompt content.

        Returns:
            List of messages to prepend to the conversation.

        Raises:
            NotImplementedError: If the renderer doesn't support tool calling.
        """
        raise NotImplementedError

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Return tokens to append to the prompt for generation.

        This is called by build_generation_prompt to add the role header that
        precedes the model's response. The default implementation renders an
        empty message and extracts its header tokens.

        Args:
            role: The role to generate (usually "assistant")
            ctx: Context for the generation suffix. Note that ctx.is_last is True
                because we're rendering the header for the final (to-be-generated) message.

        Returns:
            List of token IDs for the role header. Examples in string form:
            - Llama3: "<|start_header_id|>assistant<|end_header_id|>\n\n"
            - Qwen3: "<|im_start|>assistant\n"
            - DeepSeek: "<｜Assistant｜>" (single special token)
        """
        # Default: render an empty message and use its header tokens
        rendered = self.render_message(Message(role=role, content=""), ctx)
        if rendered.header:
            return list(rendered.header.tokens)
        return []

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> ModelInput:
        """
        Generates tokens for sampling from the model.

        Args:
            messages: a list of messages to render.
            role: the role of the partial message to be completed.
            prefill: an optional string to prefill in the model's generation.
        """

        chunks: list[ModelInputChunk] = []
        if self._bos_tokens:
            chunks.append(ModelInputChunk(tokens=self._bos_tokens))
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"),
            default=-1,
        )
        for idx, message in enumerate(messages):
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(messages) - 1),
                prev_message=messages[idx - 1] if idx > 0 else None,
                last_user_index=last_user_idx,
            )
            rendered_message = self.render_message(message, ctx)
            header_chunk = rendered_message.header
            output_chunks = rendered_message.output
            if header_chunk:
                chunks.append(header_chunk)
            chunks.extend([x for x in output_chunks if x])

        suffix_ctx = RenderContext(
            idx=len(messages),
            is_last=True,
            prev_message=messages[-1] if messages else None,
            last_user_index=last_user_idx,
        )
        suffix_tokens = self._get_generation_suffix(role, suffix_ctx)
        if suffix_tokens:
            chunks.append(ModelInputChunk(tokens=suffix_tokens))

        if prefill:
            chunks.append(
                ModelInputChunk(
                    tokens=self.tokenizer.encode(prefill, add_special_tokens=False)
                )
            )
        return ModelInput(chunks=chunks)

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ) -> list[tuple[ModelInput, torch.Tensor]]:
        """
        Build tokens and per-token weights for supervised fine-tuning.
        This function returns a list of examples in the form of tuples, where each tuple contains a model input and a tensor of weights.
        This is needed because some renderers do not satisfy the extension property, so we need to return a list of examples instead of a single example.

        This default implementation concatenates rendered messages in order, which assumes the renderer satisfies the extension property.
        Override this method if your renderer does not satisfy the extension property.
        """

        if self.has_extension_property:
            return [self.build_supervised_example(messages, train_on_what=train_on_what)]
        else:
            # TODO: Add a default implementation that calls `build_supervised_example` for each message and merges examples with shared prefixes.
            raise NotImplementedError(
                "build_supervised_examples has not been implemented for this renderer."
            )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[ModelInput, torch.Tensor]:
        """
        Build tokens and per-token weights for supervised fine-tuning.

        This default implementation concatenates rendered messages in order. Override
        this method if your build_generation_prompt does anything that breaks the simple
        concatenation assumption—for example, if it strips thinking blocks from history
        (like Qwen3Renderer), injects default system prompts, or
        otherwise modifies the token sequence.

        The supervised example tokens should match what build_generation_prompt would
        produce for the same conversation prefix, so the model trains on the same
        distribution it sees at inference time.

        Args:
            messages: A list of messages to render.
            train_on_what: Controls which tokens receive non-zero training weight:
                - LAST_ASSISTANT_MESSAGE: Only the last assistant message
                - LAST_ASSISTANT_TURN: The last assistant message after the last user message
                - ALL_ASSISTANT_MESSAGES: All assistant messages
                - ALL_MESSAGES: All messages (but not headers)
                - ALL_TOKENS: Everything including headers
                - ALL_USER_AND_SYSTEM_MESSAGES: User and system messages only
                - CUSTOMIZED: Use the 'trainable' field on each message

        Returns:
            A tuple of (model_input, weights) where weights is a 1D tensor with the
            same length as the total number of tokens.
        """
        # Warn if training on multiple assistant messages with a renderer that doesn't
        # satisfy the extension property. In that case, each assistant message sees a
        # different context prefix, so they should be trained as separate examples.
        # NOTE: This warning only covers ALL_ASSISTANT_MESSAGES. Other modes that train
        # multiple assistant messages (e.g., ALL_MESSAGES, ALL_TOKENS, CUSTOMIZED) should
        # be used with caution when has_extension_property=False.
        if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES and not self.has_extension_property:
            logger.warning(
                "WARNING: Using train_on_what=ALL_ASSISTANT_MESSAGES with a renderer that "
                "does not satisfy the extension property (has_extension_property=False). "
                "This means earlier assistant messages in the conversation see a different "
                "token prefix than what build_generation_prompt would produce at that turn. "
                "You should instead create separate conversations for each assistant message "
                "and call build_supervised_example with train_on_what=LAST_ASSISTANT_MESSAGE "
                "for each one. See docs/rl/sequence-extension.mdx for details."
            )

        model_input_chunks_weights: list[tuple[ModelInputChunk, float]] = []
        if self._bos_tokens:
            model_input_chunks_weights.append(
                (ModelInputChunk(tokens=self._bos_tokens), 0.0)
            )

        last_user_idx = max(
            (idx for idx, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )

        for idx, message in enumerate(messages):
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                assert "trainable" in message, (
                    "When using CUSTOMIZED train_on_what, each message must have a trainable field: True if loss is applied on this message, False otherwise"
                )
            else:
                assert "trainable" not in message, (
                    "When using non-CUSTOMIZED train_on_what, each message must not have a trainable field. Either change train_on_what to CUSTOMIZED or remove the trainable field from the message"
                )

            is_last_message = idx == len(messages) - 1
            is_assistant = message["role"] == "assistant"
            is_user_or_system = message["role"] in ["user", "system"]
            is_after_last_user = last_user_idx == -1 or idx > last_user_idx

            # only apply weight to header if train_on_what is ALL_TOKENS
            ctx = RenderContext(
                idx=idx,
                is_last=is_last_message,
                prev_message=messages[idx - 1] if idx > 0 else None,
                last_user_index=last_user_idx,
            )
            rendered_message = self.render_message(message, ctx)
            header_part = rendered_message.header
            output_parts = rendered_message.output
            stop_overlap_part = rendered_message.stop_overlap

            header_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
            if header_part:
                model_input_chunks_weights += [(header_part, header_weight)]

            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    output_has_weight = is_last_message and is_assistant
                case TrainOnWhat.LAST_ASSISTANT_TURN:
                    output_has_weight = is_assistant and is_after_last_user
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    output_has_weight = is_assistant
                case TrainOnWhat.ALL_MESSAGES:
                    output_has_weight = True
                case TrainOnWhat.ALL_TOKENS:
                    output_has_weight = True
                case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                    output_has_weight = is_user_or_system
                case TrainOnWhat.CUSTOMIZED:
                    output_has_weight = message.get("trainable", False)
                case _:
                    raise ValueError(f"Unknown train_on_what: {train_on_what}")

            model_input_chunks_weights += [
                (output_part, int(output_has_weight)) for output_part in output_parts if output_part
            ]

            # stop_overlap completes the stop sequence for formats like RoleColon (e.g., "User:")
            # Only included for the last message.
            if is_last_message and stop_overlap_part:
                model_input_chunks_weights += [(stop_overlap_part, int(output_has_weight))]

        weights_data = [w for chunk, w in model_input_chunks_weights for _ in range(len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length)]
        # weights are conceptually a per-token weighting (often 0/1 mask, but
        # may be fractional). Force float32 so downstream code that combines
        # them with logprobs (also float) doesn't hit a dtype mismatch.
        weights_tensor = torch.tensor(weights_data, dtype=torch.float32)

        model_input_chunks = [chunk for chunk, _ in model_input_chunks_weights]
        return ModelInput(chunks=model_input_chunks), weights_tensor


def tokens_weights_from_strings_weights(
    strings_weights: list[tuple[str, float]],
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    strings, weights = zip(*strings_weights, strict=True)
    token_chunks = [tokenizer.encode(s, add_special_tokens=i == 0) for i, s in enumerate(strings)]
    weights = torch.cat(
        [torch.full((len(chunk),), w) for chunk, w in zip(token_chunks, weights, strict=True)]
    )
    tokens = torch.cat([torch.tensor(chunk) for chunk in token_chunks])
    assert tokens.dtype == torch.int64
    return tokens, weights


def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, bool]:
    """Parse response for a single stop token.

    We expect a properly rendered response to have exactly one stop token; but it may have zero if e.g. the model
    ran out of tokens when sampling, which will incur a format error. If there are > 1, there is likely a bug in the
    sampler and we should error.
    """
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return Message(role="assistant", content=str_response), False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=str_response), True
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )


# Image processing utilities (used by VL renderers)


class ImageProcessorProtocol(Protocol):
    merge_size: int
    patch_size: int

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: Optional[dict] = None
    ) -> int:
        raise NotImplementedError()

    def get_resize_config(self, image_data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError()


def _validate_image_url_syntax(url: str) -> urllib.parse.SplitResult:
    if len(url) > 8192:
        raise ValueError("Image URL is too long")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("Remote image URLs must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Image URLs must not contain credentials")
    if parsed.hostname is None:
        raise ValueError("Image URL must include a hostname")
    return parsed


def _resolve_public_image_url(url: str) -> str:
    parsed = _validate_image_url_syntax(url)

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("Image URL hostname could not be resolved") from exc
    if not addresses:
        raise ValueError("Image URL hostname could not be resolved")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%", 1)[0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError("Image URL must resolve to a public unicast address")
        if isinstance(ip, ipaddress.IPv6Address):
            if ip in _NAT64_LOCAL_USE_NETWORK:
                raise ValueError("Image URL must resolve to a public unicast address")
            if ip in _NAT64_WELL_KNOWN_NETWORK:
                translated_ip = ipaddress.IPv4Address(ip.packed[-4:])
                if not translated_ip.is_global or translated_ip.is_multicast:
                    raise ValueError(
                        "Image URL must resolve to a public unicast address"
                    )
    return addresses[0][4][0].split("%", 1)[0]


def _validate_public_image_url(url: str) -> None:
    _resolve_public_image_url(url)


class _PinnedAddressConnectionMixin:
    def __init__(self, *args, pinned_ip: str, deadline: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pinned_ip = pinned_ip
        self._deadline = deadline
        self._deadline_expired = threading.Event()
        self._deadline_timer: threading.Timer | None = None
        self._create_connection = self._create_pinned_connection

    def _create_pinned_connection(self, address, *args, **kwargs):
        sock = socket.create_connection((self._pinned_ip, address[1]), *args, **kwargs)
        self.sock = sock
        if self._deadline is not None:
            remaining_seconds = self._deadline - time.monotonic()
            if remaining_seconds <= 0:
                sock.close()
                raise TimeoutError("Image download exceeded the time limit")
            self._deadline_timer = threading.Timer(
                remaining_seconds, self._expire_deadline
            )
            self._deadline_timer.daemon = True
            self._deadline_timer.start()
        return sock

    def _expire_deadline(self) -> None:
        self._deadline_expired.set()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.close()

    def cancel_deadline(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._deadline_timer = None

    def connect(self) -> None:
        super().connect()
        if self._deadline_expired.is_set():
            self.close()
            raise TimeoutError("Image download exceeded the time limit")

    def close(self) -> None:
        self.cancel_deadline()
        super().close()


class _PinnedHTTPConnection(_PinnedAddressConnectionMixin, http.client.HTTPConnection):
    pass


class _PinnedHTTPSConnection(_PinnedAddressConnectionMixin, http.client.HTTPSConnection):
    pass


class _DeadlineConnectionHandlerMixin:
    def __init__(self, *args, deadline: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._deadline = deadline
        self._connections: list[_PinnedAddressConnectionMixin] = []

    def _new_connection(self, connection_type, host, pinned_ip, **kwargs):
        connection = connection_type(
            host,
            pinned_ip=pinned_ip,
            deadline=self._deadline,
            **kwargs,
        )
        self._connections.append(connection)
        return connection

    def cancel_deadlines(self) -> None:
        for connection in self._connections:
            connection.cancel_deadline()
        self._connections.clear()


class _SafeImageHTTPHandler(
    _DeadlineConnectionHandlerMixin, urllib.request.HTTPHandler
):
    def http_open(self, req):
        pinned_ip = _resolve_public_image_url(req.full_url)
        return self.do_open(
            lambda host, **kwargs: self._new_connection(
                _PinnedHTTPConnection, host, pinned_ip, **kwargs
            ),
            req,
        )


class _SafeImageHTTPSHandler(
    _DeadlineConnectionHandlerMixin, urllib.request.HTTPSHandler
):
    def https_open(self, req):
        pinned_ip = _resolve_public_image_url(req.full_url)
        return self.do_open(
            lambda host, **kwargs: self._new_connection(
                _PinnedHTTPSConnection, host, pinned_ip, **kwargs
            ),
            req,
            context=self._context,
        )


class _SafeImageRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if (
            urllib.parse.urlsplit(req.full_url).scheme == "https"
            and urllib.parse.urlsplit(newurl).scheme != "https"
        ):
            raise ValueError("Image redirects must not downgrade HTTPS to HTTP")
        _validate_public_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get_response_socket(response: Any) -> Any | None:
    pending = [response]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        sock = getattr(current, "_sock", None)
        if sock is not None:
            return sock
        pending.extend(
            [getattr(current, "fp", None), getattr(current, "raw", None)]
        )
    return None


def _read_bounded_image_response(
    response: Any, *, deadline: float | None = None
) -> bytes:
    content_length = response.headers.get("Content-Length") if response.headers else None
    parsed_content_length = None
    if content_length is not None:
        try:
            parsed_content_length = int(content_length)
        except ValueError as exc:
            raise ValueError("Image response has an invalid Content-Length") from exc
        if parsed_content_length < 0:
            raise ValueError("Image response has an invalid Content-Length")
        if parsed_content_length > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")

    if deadline is None:
        deadline = time.monotonic() + IMAGE_FETCH_TIMEOUT_SECONDS
    read = getattr(response, "read1", None) or response.read
    response_socket = _get_response_socket(response)
    chunks = []
    total_bytes = 0
    while parsed_content_length is None or total_bytes < parsed_content_length:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise ValueError("Image download exceeded the time limit")
        if response_socket is not None:
            response_socket.settimeout(remaining_seconds)
        read_size = min(
            _IMAGE_READ_CHUNK_BYTES,
            MAX_IMAGE_BYTES + 1 - total_bytes,
        )
        try:
            chunk = read(read_size)
        except TimeoutError as exc:
            raise ValueError("Image download exceeded the time limit") from exc
        if not chunk:
            break
        chunks.append(chunk)
        total_bytes += len(chunk)
        if total_bytes > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")
        if time.monotonic() > deadline:
            raise ValueError("Image download exceeded the time limit")
    return b"".join(chunks)


def _read_cached_image(cache_path: str) -> bytes | None:
    try:
        with open(cache_path, "rb") as cache_file:
            data = cache_file.read(MAX_IMAGE_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        os.unlink(cache_path)
        return None
    return data


def _write_cached_image(cache_path: str, data: bytes) -> None:
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    os.chmod(cache_dir, 0o700)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".image-",
        dir=cache_dir,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as cache_file:
            cache_file.write(data)
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _download_image_reference(reference: str) -> bytes:
    parsed = _validate_image_url_syntax(reference)
    host = parsed.hostname or "unknown host"
    last_error = "network error"
    attempts = 0
    for attempts in range(1, IMAGE_FETCH_MAX_ATTEMPTS + 1):
        deadline = time.monotonic() + IMAGE_FETCH_TIMEOUT_SECONDS
        _validate_public_image_url(reference)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise ValueError("Image download exceeded the time limit")
        http_handler = _SafeImageHTTPHandler(deadline=deadline)
        https_handler = _SafeImageHTTPSHandler(deadline=deadline)
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            http_handler,
            https_handler,
            _SafeImageRedirectHandler(),
        )
        request = urllib.request.Request(
            reference,
            headers={"User-Agent": "Interactive TrainingImageLoader/1.0"},
        )
        try:
            with opener.open(request, timeout=remaining_seconds) as response:
                return _read_bounded_image_response(response, deadline=deadline)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            last_error = "network error"
        finally:
            http_handler.cancel_deadlines()
            https_handler.cancel_deadlines()
        if attempts < IMAGE_FETCH_MAX_ATTEMPTS:
            time.sleep(0.1 * (2 ** (attempts - 1)))
    raise ValueError(
        f"Image download from {host} failed after {attempts} attempt(s): {last_error}"
    ) from None


def _load_image_reference(reference: str, *, cache_dir: str | None = None) -> bytes:
    if reference.startswith("data:"):
        header, separator, encoded = reference.partition(",")
        media_type = header[5:].removesuffix(";base64").lower()
        if (
            not separator
            or not header.lower().endswith(";base64")
            or media_type not in _ALLOWED_DATA_IMAGE_MEDIA_TYPES
        ):
            raise ValueError(
                "Image data URIs must contain a base64-encoded JPEG, PNG, or WebP image"
            )
        max_encoded_length = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
        if len(encoded) > max_encoded_length:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image data URI contains invalid base64") from exc
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")
        return data

    _validate_image_url_syntax(reference)
    cache_path = None
    if cache_dir is not None:
        source_key = hashlib.sha256(reference.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{source_key}.image")
        cached = _read_cached_image(cache_path)
        if cached is not None:
            return cached

    data = _download_image_reference(reference)
    with _open_verified_image(data):
        pass
    if cache_path is not None:
        _write_cached_image(cache_path, data)
    return data


def _get_encoded_image_format(image_data: bytes) -> str:
    if image_data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if (
        len(image_data) >= 12
        and image_data.startswith(b"RIFF")
        and image_data[8:12] == b"WEBP"
    ):
        return "WEBP"
    raise ValueError("Image must be encoded as JPEG, PNG, or WebP")


def _open_verified_image(image_data: bytes):
    from PIL import Image, UnidentifiedImageError

    encoded_format = _get_encoded_image_format(image_data)
    image = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_data)) as candidate:
                if candidate.format != encoded_format:
                    raise ValueError("Image must be encoded as JPEG, PNG, or WebP")
                if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit")
                if getattr(candidate, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                candidate.verify()

            image = Image.open(io.BytesIO(image_data))
            image.load()
    except ValueError:
        if image is not None:
            image.close()
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
    ) as exc:
        if image is not None:
            image.close()
        raise ValueError("Image is malformed or unsafe") from exc
    return image


def _loaded_image_to_chunk(
    pil_image: "Image.Image", image_processor: ImageProcessorProtocol
) -> ImageChunk:
    if pil_image.width * pil_image.height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit")
    if getattr(pil_image, "n_frames", 1) != 1:
        raise ValueError("Animated images are not supported")
    pil_image.load()

    normalized_image = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
    try:
        img_byte_arr = io.BytesIO()
        normalized_image.save(img_byte_arr, format="JPEG")
        image_data = img_byte_arr.getvalue()
        if len(image_data) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Normalized image exceeds the {MAX_IMAGE_BYTES}-byte limit"
            )

        if hasattr(image_processor, "get_number_of_image_patches"):
            width, height = normalized_image.size
            num_image_tokens = (
                image_processor.get_number_of_image_patches(
                    height, width, images_kwargs={}
                )
                // image_processor.merge_size**2
            )
        elif hasattr(image_processor, "get_resize_config"):
            config = image_processor.get_resize_config(
                {"type": "image", "image": normalized_image}
            )
            num_image_tokens = config["num_tokens"]
        else:
            raise ValueError(
                "Don't know how to get the number of image tokens for image "
                f"processor: {image_processor}"
            )

        if num_image_tokens < 1:
            raise ValueError("Image token count must be positive")

        return ImageChunk(
            data=image_data,
            format="jpeg",
            expected_tokens=num_image_tokens,
        )
    finally:
        if normalized_image is not pil_image:
            normalized_image.close()


def image_to_chunk(
    image_or_str: "Image.Image | str",
    image_processor: ImageProcessorProtocol,
    *,
    cache_dir: str | None = None,
) -> ImageChunk:
    """
    Convert a PIL Image to a ImageChunk for QwenVL
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for image / vision-language training. "
            "Install the image extra with: pip install -e .[image]"
        ) from e

    # Load and validate an image from a data URI or public URL.
    if isinstance(image_or_str, str):
        image_data = (
            _load_image_reference(image_or_str)
            if cache_dir is None
            else _load_image_reference(image_or_str, cache_dir=cache_dir)
        )
        with _open_verified_image(image_data) as pil_image:
            return _loaded_image_to_chunk(pil_image, image_processor)

    # Otherwise the image is a PIL image and can be loaded directly
    elif isinstance(image_or_str, Image.Image):
        return _loaded_image_to_chunk(image_or_str, image_processor)

    # Validate the provided data is actually a valid image type
    else:
        raise ValueError("The provided image must be a PIL.Image.Image, URL, or data URI.")
