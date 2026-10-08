"""
Renderers for converting message lists into training and sampling prompts.

Use viz_sft_dataset to visualize the output of different renderers. E.g.,
    python -m interactive_training.supervised.viz_sft_dataset dataset_path=Tulu3Builder renderer_name=role_colon
"""

from collections.abc import Callable
from typing import Any

from interactive_training.image_processing_utils import ImageProcessor

# Types and utilities used by external code
from interactive_training.renderers.base import (
    # Content part types
    ContentPart,
    ImagePart,
    ImageURLPart,
    Message,
    # Streaming types
    MessageDelta,
    # Renderer base
    RenderContext,
    Renderer,
    Role,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    Utf8TokenDecoder,
    # Utility functions
    ensure_text,
    format_content_as_string,
    get_text_content,
    parse_content_blocks,
)

# Renderer classes used directly by tests
from interactive_training.renderers.deepseek_v3 import DeepSeekV3ThinkingRenderer
from interactive_training.renderers.gpt_oss import GptOssRenderer
from interactive_training.renderers.qwen3 import Qwen3Renderer
from interactive_training.tokenizer_utils import Tokenizer

# Global registry for custom renderer factories
_CUSTOM_RENDERER_REGISTRY: dict[str, Callable[[Tokenizer, Any], Renderer]] = {}


def register_renderer(
    name: str,
    factory: Callable[[Tokenizer, Any], Renderer],
) -> None:
    """Register a custom renderer factory.

    Args:
        name: The renderer name
        factory: A callable that takes (tokenizer, image_processor) and returns a Renderer.

    Example:
        def my_renderer_factory(tokenizer, image_processor=None):
            return MyCustomRenderer(tokenizer)

        register_renderer("Foo/foo_renderer", my_renderer_factory)
    """
    _CUSTOM_RENDERER_REGISTRY[name] = factory


def get_registered_renderer_names() -> list[str]:
    """Return a list of all registered custom renderer names."""
    return list(_CUSTOM_RENDERER_REGISTRY.keys())


def is_renderer_registered(name: str) -> bool:
    """Check if a renderer name is registered."""
    return name in _CUSTOM_RENDERER_REGISTRY


def unregister_renderer(name: str) -> bool:
    """Unregister a custom renderer factory.

    Args:
        name: The renderer name to unregister.

    Returns:
        True if the renderer was unregistered, False if it wasn't registered.
    """
    if name in _CUSTOM_RENDERER_REGISTRY:
        del _CUSTOM_RENDERER_REGISTRY[name]
        return True
    return False


def get_renderer(
    name: str, tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    """Factory function to create renderers by name.

    Args:
        name: Renderer name. Supported values:
            - "role_colon": Simple role:content format
            - "llama3": Llama 3 chat format
            - "qwen3": Qwen3 with thinking enabled
            - "qwen3_vl": Qwen3 vision-language with thinking
            - "qwen3_vl_instruct": Qwen3 vision-language instruct (no thinking)
            - "qwen3_disable_thinking": Qwen3 with thinking disabled
            - "qwen3_instruct": Qwen3 instruct 2507 (no thinking)
            - "qwen3_5": Qwen3.5 with thinking enabled (XML tool calls)
            - "qwen3_5_disable_thinking": Qwen3.5 with thinking disabled
            - "qwen3_8": Qwen3.8 with thinking enabled (reasoning effort xhigh)
            - "qwen3_8_medium_reasoning": Qwen3.8 with reasoning effort medium
            - "qwen3_8_low_reasoning": Qwen3.8 with reasoning effort low
            - "qwen3_8_disable_thinking": Qwen3.8 with thinking disabled
            - "deepseekv3": DeepSeek V3 (defaults to non-thinking mode)
            - "deepseekv3_disable_thinking": DeepSeek V3 non-thinking (alias)
            - "deepseekv3_thinking": DeepSeek V3 thinking mode
            - "glm5": GLM-5.2 with thinking (Reasoning Effort: Max)
            - "glm5_high_reasoning": GLM-5.2 with Reasoning Effort: High
            - "glm5_disable_thinking": GLM-5.2 with thinking disabled
            - "gpt_oss_no_sysprompt": GPT-OSS without system prompt
            - "gpt_oss_low_reasoning": GPT-OSS with low reasoning
            - "gpt_oss_medium_reasoning": GPT-OSS with medium reasoning
            - "gpt_oss_high_reasoning": GPT-OSS with high reasoning
            - "muse_glimmer": Muse Glimmer with high reasoning strength
            - "muse_glimmer_low_reasoning": Muse Glimmer with low reasoning strength
            - "muse_glimmer_medium_reasoning": Muse Glimmer with medium reasoning strength
            - "muse_glimmer_xhigh_reasoning": Muse Glimmer with extra-high reasoning strength
            - Custom renderers registered via register_renderer()
        tokenizer: The tokenizer to use.
        image_processor: Required for VL renderers.

    Returns:
        A Renderer instance.

    Raises:
        ValueError: If the renderer name is unknown.
        AssertionError: If a VL renderer is requested without an image_processor.
    """
    # Check custom registry first
    if (renderer := _CUSTOM_RENDERER_REGISTRY.get(name)) is not None:
        return renderer(tokenizer, image_processor)

    # Import renderer classes lazily to avoid circular imports and keep exports minimal
    from interactive_training.renderers.deepseek_v3 import DeepSeekV3DisableThinkingRenderer
    from interactive_training.renderers.glm5 import (
        GLM5DisableThinkingRenderer,
        GLM5HighReasoningRenderer,
        GLM5Renderer,
    )
    from interactive_training.renderers.gpt_oss import GptOssRenderer
    from interactive_training.renderers.llama3 import Llama3Renderer
    from interactive_training.renderers.muse_glimmer import MuseGlimmerRenderer
    from interactive_training.renderers.qwen3 import (
        Qwen3DisableThinkingRenderer,
        Qwen3InstructRenderer,
        Qwen3VLInstructRenderer,
        Qwen3VLRenderer,
    )
    from interactive_training.renderers.qwen3_5 import (
        Qwen3_5DisableThinkingRenderer,
        Qwen3_5Renderer,
    )
    from interactive_training.renderers.qwen3_8 import (
        Qwen3_8DisableThinkingRenderer,
        Qwen3_8Renderer,
    )
    from interactive_training.renderers.role_colon import RoleColonRenderer

    if name == "role_colon":
        return RoleColonRenderer(tokenizer)
    elif name == "llama3":
        return Llama3Renderer(tokenizer)
    elif name == "qwen3":
        return Qwen3Renderer(tokenizer)
    elif name == "qwen3_vl":
        assert image_processor is not None, "qwen3_vl renderer requires an image_processor"
        return Qwen3VLRenderer(tokenizer, image_processor)
    elif name == "qwen3_vl_instruct":
        assert image_processor is not None, "qwen3_vl_instruct renderer requires an image_processor"
        return Qwen3VLInstructRenderer(tokenizer, image_processor)
    elif name == "qwen3_disable_thinking":
        return Qwen3DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_instruct":
        return Qwen3InstructRenderer(tokenizer)
    elif name == "qwen3_5":
        return Qwen3_5Renderer(tokenizer)
    elif name == "qwen3_5_disable_thinking":
        return Qwen3_5DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_8":
        return Qwen3_8Renderer(tokenizer)
    elif name == "qwen3_8_medium_reasoning":
        return Qwen3_8Renderer(tokenizer, reasoning_effort="medium")
    elif name == "qwen3_8_low_reasoning":
        return Qwen3_8Renderer(tokenizer, reasoning_effort="low")
    elif name == "qwen3_8_disable_thinking":
        return Qwen3_8DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3":
        # Default to non-thinking mode (matches HF template default behavior)
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_disable_thinking":
        # Alias for backward compatibility
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_thinking":
        return DeepSeekV3ThinkingRenderer(tokenizer)
    elif name == "glm5":
        return GLM5Renderer(tokenizer)
    elif name == "glm5_high_reasoning":
        return GLM5HighReasoningRenderer(tokenizer)
    elif name == "glm5_disable_thinking":
        return GLM5DisableThinkingRenderer(tokenizer)
    elif name == "gpt_oss_no_sysprompt":
        return GptOssRenderer(tokenizer, use_system_prompt=False)
    elif name == "gpt_oss_low_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="low")
    elif name == "gpt_oss_medium_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")
    elif name == "gpt_oss_high_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="high")
    elif name == "muse_glimmer":
        return MuseGlimmerRenderer(tokenizer, image_processor=image_processor)
    elif name == "muse_glimmer_low_reasoning":
        return MuseGlimmerRenderer(
            tokenizer,
            image_processor=image_processor,
            reasoning_strength="low",
        )
    elif name == "muse_glimmer_medium_reasoning":
        return MuseGlimmerRenderer(
            tokenizer,
            image_processor=image_processor,
            reasoning_strength="medium",
        )
    elif name == "muse_glimmer_xhigh_reasoning":
        return MuseGlimmerRenderer(
            tokenizer,
            image_processor=image_processor,
            reasoning_strength="xhigh",
        )
    else:
        raise ValueError(
            f"Unknown renderer: {name}. If this is a custom renderer, please register it via register_renderer()."
        )


__all__ = [
    # Types
    "ContentPart",
    "ImagePart",
    "ImageURLPart",
    "Message",
    "Role",
    "TextPart",
    "ThinkingPart",
    "ToolCall",
    "ToolSpec",
    # Streaming types
    "MessageDelta",
    "StreamingMessageHeader",
    "StreamingTextDelta",
    "StreamingThinkingDelta",
    "Utf8TokenDecoder",
    # Renderer base
    "RenderContext",
    "Renderer",
    "TrainOnWhat",
    # Utility functions
    "ensure_text",
    "format_content_as_string",
    "get_text_content",
    "parse_content_blocks",
    # Registry
    "register_renderer",
    "unregister_renderer",
    "get_registered_renderer_names",
    "is_renderer_registered",
    # Factory
    "get_renderer",
    # Renderer classes (used by tests)
    "DeepSeekV3ThinkingRenderer",
    "GptOssRenderer",
    "Qwen3Renderer",
]
