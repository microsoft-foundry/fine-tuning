"""
This module associates model names with metadata, which helps  training code choose good defaults.
"""

import os
from dataclasses import dataclass
from functools import cache

# Text recipes share one supported default; vision recipes choose separately.
DEFAULT_MODEL_NAME = "Qwen/Qwen3.8-27B"

#: Environment flag that gates GLM-5.2 / GLM-5.2-FP8 routing. The renderer classes
#: are always importable/registerable by name, but the cookbook only auto-routes
#: GLM-5.2 models to them (and the worker only enables the FP8/offload path) when
#: this flag is set truthy.
GLM5_FLAG_ENV = "INTERACTIVE_POST_TRAINING_ENABLE_GLM5"


def glm5_enabled() -> bool:
    """True iff the GLM-5.2 feature flag is enabled in the environment."""
    return os.environ.get(GLM5_FLAG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


#: Environment flag that gates DeepSeek-V4 Flash routing. The deepseekv3
#: renderers are always importable/registerable by name, but the cookbook only
#: auto-routes DeepSeek-V4 Flash to them (and the API accepts the base model)
#: when this flag is set truthy. DeepSeek-V4 Flash trains in full precision
#: (bf16); no 4-bit/NF4 quantization path is used.
DEEPSEEK_V4_FLAG_ENV = "INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4"


def deepseek_v4_enabled() -> bool:
    """True iff the DeepSeek-V4 Flash feature flag is enabled in the environment."""
    return os.environ.get(DEEPSEEK_V4_FLAG_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
@dataclass
class ModelAttributes:
    organization: str  # meta-llama, Qwen, etc.
    version_str: str  # just the version number e.g. "3.1", "2.5"
    size_str: str  # size of the model e.g. "8B", "72B", "1.5B"
    is_chat: bool  # is chat/instruct model
    is_vl: bool = False  # is vision-language model
    max_context_length: int | None = None


_CUSTOM_MODELS: dict[str, tuple[ModelAttributes, list[str]]] = {}


def register_model(
    model_name: str,
    attributes: ModelAttributes,
    renderer_names: list[str],
) -> None:
    """Register model metadata and recommended renderers from an external package."""
    _CUSTOM_MODELS[model_name] = (attributes, renderer_names)


@cache
def get_llama_info() -> dict[str, ModelAttributes]:
    org = "meta-llama"
    return {
        "Llama-3.2-1B-Instruct": ModelAttributes(org, "3.2", "1B", True),
        "Llama-3.2-3B-Instruct": ModelAttributes(org, "3.2", "3B", True),
        "Llama-3.1-8B-Instruct": ModelAttributes(org, "3.1", "8B", True),
        "Llama-3.2-1B": ModelAttributes(org, "3.2", "1B", False),
        "Llama-3.2-3B": ModelAttributes(org, "3.2", "3B", False),
        "Llama-3.1-8B": ModelAttributes(org, "3.1", "8B", False),
        "Llama-3.1-70B": ModelAttributes(org, "3.1", "70B", False),
        "Llama-3.3-70B-Instruct": ModelAttributes(org, "3.3", "70B", True),
    }


@cache
def get_meta_models_info() -> dict[str, ModelAttributes]:
    org = "meta-models"
    return {
        "Muse-Glimmer-30B": ModelAttributes(
            org, "1", "30B", True, is_vl=True, max_context_length=32_768
        ),
    }


def get_model_context_length(model_name: str, tokenizer=None) -> int | None:
    """Return a trustworthy model context length when one is available."""
    try:
        configured = get_model_attributes(model_name).max_context_length
    except ValueError:
        configured = None
    if configured is not None:
        return configured

    if tokenizer is None:
        return None
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_limit, int)
        and tokenizer_limit > 0
        and tokenizer_limit < 1_000_000_000
    ):
        return tokenizer_limit
    return None


def require_vision_language_model(model_name: str, *, workload: str) -> None:
    """Reject image-bearing workloads for known text-only models."""
    attributes = get_model_attributes(model_name)
    if not attributes.is_vl:
        raise ValueError(
            f"{workload} includes image inputs and requires a vision-language model, "
            f"but {model_name!r} is text-only. Choose a vision-language model with "
            "image fine-tuning enabled for your project; see docs/supported_models.md."
        )


@cache
def get_qwen_info() -> dict[str, ModelAttributes]:
    org = "Qwen"
    return {
        "Qwen3-VL-30B-A3B-Instruct": ModelAttributes(org, "3", "30B-A3B", True, is_vl=True),
        "Qwen3-VL-235B-A22B-Instruct": ModelAttributes(org, "3", "235B-A22B", True, is_vl=True),
        "Qwen3-4B-Base": ModelAttributes(org, "3", "4B", False),
        "Qwen3-8B-Base": ModelAttributes(org, "3", "8B", False),
        "Qwen3-14B-Base": ModelAttributes(org, "3", "14B", False),
        "Qwen3-30B-A3B-Base": ModelAttributes(org, "3", "30B-A3B", False),
        "Qwen3-0.6B": ModelAttributes(org, "3", "0.6B", True),
        "Qwen3-1.7B": ModelAttributes(org, "3", "1.7B", True),
        "Qwen3-4B": ModelAttributes(org, "3", "4B", True),
        "Qwen3-8B": ModelAttributes(org, "3", "8B", True),
        "Qwen3-14B": ModelAttributes(org, "3", "14B", True),
        "Qwen3-32B": ModelAttributes(org, "3", "32B", True),
        "Qwen3-32B-H100": ModelAttributes(org, "3", "32B", True),
        "Qwen3-30B-A3B": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-4B-Instruct-2507": ModelAttributes(org, "3", "4B", True),
        "Qwen3-30B-A3B-Instruct-2507": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-235B-A22B-Instruct-2507": ModelAttributes(org, "3", "235B-A22B", True),
        "Qwen3.5-4B": ModelAttributes(org, "3.5", "4B", True),
        "Qwen3.5-0.8B": ModelAttributes(org, "3.5", "0.8B", True),
        "Qwen3.5-9B": ModelAttributes(org, "3.5", "9B", True),
        "Qwen3.6-35B-A3B": ModelAttributes(org, "3.6", "35B-A3B", True),
        "Qwen3.8-27B": ModelAttributes(org, "3.8", "27B", True),
    }


@cache
def get_deepseek_info() -> dict[str, ModelAttributes]:
    org = "deepseek-ai"
    return {
        "DeepSeek-V3.1": ModelAttributes(org, "3", "671B-A37B", True),
        "DeepSeek-V3.1-Base": ModelAttributes(org, "3", "671B-A37B", False),
        "DeepSeek-V3.2-Exp": ModelAttributes(org, "3", "671B-A37B", True),
        # DeepSeek-V4 Flash (model_type "deepseek_v4"): MLA + 256-expert MoE.
        # Ships block-FP8 attention weights and FP4 routed experts, dequantized
        # per layer for full-precision training. Feature-flagged via
        # INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4; renderer routing (below) rejects it when the
        # flag is off. Shares the DeepSeek chat format, so it reuses the
        # deepseekv3 renderer.
        #
        # Sizes are counted from the checkpoint index, not inferred from the
        # 155 GiB on-disk footprint: the routed experts are FP4 (two values per
        # stored byte), so file size understates the parameter count by ~2x.
        # 43 layers x 256 experts x 3 x (4096x2048) = 277.0B routed + 7.3B
        # dense = 293B total; top-6-of-256 routing makes 13.8B active.
        # ``-0731`` differs only in its dropped MTP head, so both share sizes.
        "DeepSeek-V4-Flash": ModelAttributes(org, "4", "293B-A14B", True),
        "DeepSeek-V4-Flash-0731": ModelAttributes(org, "4", "293B-A14B", True),
    }


@cache
def get_gpt_oss_info() -> dict[str, ModelAttributes]:
    org = "openai"
    return {
        "gpt-oss-20b": ModelAttributes(org, "1", "21B-A3.6B", True),
        "gpt-oss-120b": ModelAttributes(org, "1", "117B-A5.1B", True),
    }


@cache
def get_glm_info() -> dict[str, ModelAttributes]:
    org = "zai-org"
    return {
        # GLM-5.2 (GlmMoeDsaForCausalLM): MLA + DSA indexer + 256-expert MoE.
        # The FP8 variant ships DeepSeek-style block-FP8 weights; both share the
        # same tokenizer / chat template and therefore the same renderer.
        "GLM-5.2": ModelAttributes(org, "5.2", "753B-A?", True),
        "GLM-5.2-FP8": ModelAttributes(org, "5.2", "753B-A?", True),
    }


@cache
def get_prism_info() -> dict[str, ModelAttributes]:
    org = "prism-ml"
    return {
        "Ternary-Bonsai-2-27B-gguf": ModelAttributes(org, "Bonsai-2", "27B", True),
    }


def get_model_attributes(model_name: str) -> ModelAttributes:
    """Look up model attributes by HF ID (``Org/Model_Name``) or lowercase name (``qwen3-32b``).

    Examples::

        get_model_attributes("Qwen/Qwen3-32B")       # HF ID
        get_model_attributes("qwen3-32b")             # lowercase shorthand
    """
    if model_name in _CUSTOM_MODELS:
        return _CUSTOM_MODELS[model_name][0]
    lowered = model_name.lower()
    for registered_name, (attributes, _) in _CUSTOM_MODELS.items():
        if registered_name.rsplit("/", 1)[-1].lower() == lowered:
            return attributes

    # Build a single flat lookup table: exact key → attrs, plus lowercased key → attrs.
    all_tables = [
        get_llama_info(),
        get_meta_models_info(),
        get_qwen_info(),
        get_deepseek_info(),
        get_gpt_oss_info(),
        get_glm_info(),
        get_prism_info(),
    ]
    org_for_table = {
        id(get_llama_info()): "meta-llama",
        id(get_meta_models_info()): "meta-models",
        id(get_qwen_info()): "Qwen",
        id(get_deepseek_info()): "deepseek-ai",
        id(get_gpt_oss_info()): "openai",
        id(get_glm_info()): "zai-org",
        id(get_prism_info()): "prism-ml",
    }

    # 1) Try HF ID: "Org/Model_Name"
    if "/" in model_name:
        org, key = model_name.split("/", 1)
        for table in all_tables:
            if org_for_table[id(table)] == org and key in table:
                return table[key]
        raise ValueError(f"Unknown model: {model_name}")

    # 2) Try lowercase shorthand: case-insensitive match against all known keys
    for table in all_tables:
        for key, attrs in table.items():
            if key.lower() == lowered:
                return attrs

    raise ValueError(
        f"Unknown model: {model_name}. Expected 'Org/Model_Name' or a known lowercase model name."
    )


def get_recommended_renderer_names(model_name: str) -> list[str]:
    """
    Return a list of renderers that are designed for the model.
    Used so we can emit a warning if you use a non-recommended renderer.
    The first result is the most recommended renderer for the model.
    """
    if model_name in _CUSTOM_MODELS:
        return list(_CUSTOM_MODELS[model_name][1])
    lowered = model_name.lower()
    for registered_name, (_, renderer_names) in _CUSTOM_MODELS.items():
        if registered_name.rsplit("/", 1)[-1].lower() == lowered:
            return list(renderer_names)

    attributes = get_model_attributes(model_name)
    if not attributes.is_chat:
        return ["role_colon"]
    elif attributes.organization == "meta-llama":
        return ["llama3"]
    elif attributes.organization == "meta-models":
        return ["muse_glimmer"]
    elif attributes.organization == "prism-ml":
        return ["qwen3_8", "qwen3_8_disable_thinking"]
    elif attributes.organization == "Qwen":
        if attributes.version_str == "3":
            if attributes.is_vl:
                if "-Instruct" in model_name:
                    return ["qwen3_vl_instruct"]
                else:
                    return ["qwen3_vl"]
            elif "-Instruct" in model_name:
                return ["qwen3_instruct"]
            else:
                return ["qwen3", "qwen3_disable_thinking"]
        elif attributes.version_str in {"3.5", "3.6"}:
            # Qwen3.5/Qwen3.6 share the dedicated renderer: always emits <think>
            # blocks and uses XML tool-call format.
            return [
                "qwen3_5",
                "qwen3_5_disable_thinking",
            ]
        elif attributes.version_str == "3.8":
            # Qwen3.8 keeps Qwen3.5's architecture and tool-call format but not
            # its chat template: it prepends a reasoning-effort instruction to
            # the system prompt and preserves thinking in history by default.
            # General cookbook runs prefer low reasoning. Explicit qwen3_8
            # retains its native xhigh behavior; all named modes stay available.
            return [
                "qwen3_8_low_reasoning",
                "qwen3_8",
                "qwen3_8_medium_reasoning",
                "qwen3_8_disable_thinking",
            ]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    elif attributes.organization == "deepseek-ai":
        # DeepSeek-V4 Flash is feature-flagged: only auto-route to the DeepSeek
        # renderer when the flag is enabled so the path stays inert otherwise.
        if attributes.version_str == "4" and not deepseek_v4_enabled():
            raise ValueError(
                f"{model_name} (DeepSeek-V4 Flash) support is feature-flagged. "
                f"Set {DEEPSEEK_V4_FLAG_ENV}=1 to enable it."
            )
        # deepseekv3 defaults to non-thinking mode (matches HF template)
        # Use deepseekv3_thinking for thinking mode
        return ["deepseekv3", "deepseekv3_thinking"]
    elif attributes.organization == "openai":
        return ["gpt_oss_no_sysprompt", "gpt_oss_medium_reasoning"]
    elif attributes.organization == "zai-org":
        # GLM-5.2 / GLM-5.2-FP8 is feature-flagged: only auto-route to the GLM
        # renderer when the flag is enabled so the path stays inert otherwise.
        if not glm5_enabled():
            raise ValueError(
                f"{model_name} (GLM-5.2) support is feature-flagged. "
                f"Set {GLM5_FLAG_ENV}=1 to enable it."
            )
        return ["glm5", "glm5_disable_thinking"]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_name(
    model_name: str, *, prefer_non_thinking: bool = False
) -> str:
    """
    Return the recommended renderer, optionally preferring its non-thinking variant.
    """
    names = get_recommended_renderer_names(model_name)
    if prefer_non_thinking:
        for name in names:
            if name.endswith("_disable_thinking"):
                return name
    return names[0]
