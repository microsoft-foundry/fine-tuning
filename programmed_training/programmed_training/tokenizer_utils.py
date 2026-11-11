"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

import json
import ntpath
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any

# Global registry for custom tokenizer factories
_CUSTOM_TOKENIZER_REGISTRY: dict[str, Callable[[], Tokenizer]] = {}

GLM5_FLAG_ENV = "INTERACTIVE_POST_TRAINING_ENABLE_GLM5"
_GLM5_MODEL_NAMES = {"zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8"}


def register_tokenizer(
    name: str,
    factory: Callable[[], Tokenizer],
) -> None:
    """Register a custom tokenizer factory.

    Args:
        name: The tokenizer name
        factory: A callable that takes no arguments and returns a Tokenizer.

    Example:
        def my_tokenizer_factory():
            return MyCustomTokenizer()

        register_tokenizer("Foo/foo_tokenizer", my_tokenizer_factory)
    """
    _CUSTOM_TOKENIZER_REGISTRY[name] = factory


def get_registered_tokenizer_names() -> list[str]:
    """Return a list of all registered custom tokenizer names."""
    return list(_CUSTOM_TOKENIZER_REGISTRY.keys())


def is_tokenizer_registered(name: str) -> bool:
    """Check if a tokenizer name is registered."""
    return name in _CUSTOM_TOKENIZER_REGISTRY


def unregister_tokenizer(name: str) -> bool:
    """Unregister a custom tokenizer factory.

    Args:
        name: The tokenizer name to unregister.

    Returns:
        True if the tokenizer was unregistered, False if it wasn't registered.
    """
    if name in _CUSTOM_TOKENIZER_REGISTRY:
        del _CUSTOM_TOKENIZER_REGISTRY[name]
        return True
    return False


def get_tokenizer(model_name: str) -> Tokenizer:
    """Get a tokenizer by name.

    Checks custom registry first, then falls back to HuggingFace AutoTokenizer.
    """
    # Check custom registry first (not cached, factory handles caching if needed)
    if (tokenizer := _CUSTOM_TOKENIZER_REGISTRY.get(model_name)) is not None:
        return tokenizer()

    return _get_hf_tokenizer(model_name)


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _strip_adapter_suffix(model_name: str) -> str:
    # Parse Windows drives on every host without treating their colon as a suffix.
    drive, path = ntpath.splitdrive(model_name)
    if drive.endswith(":"):
        return drive + path.split(":", 1)[0]
    return model_name.split(":", 1)[0]


def _is_glm5_model(model_name: str) -> bool:
    model_name = _strip_adapter_suffix(model_name)
    if model_name in _GLM5_MODEL_NAMES:
        return True

    model_path = Path(model_name)
    if not model_path.is_dir():
        return False

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    architectures = raw_config.get("architectures") or []
    return raw_config.get("model_type") == "glm_moe_dsa" or "GlmMoeDsaForCausalLM" in architectures


def _offline_mode() -> bool:
    return _flag_enabled("HF_HUB_OFFLINE") or _flag_enabled("TRANSFORMERS_OFFLINE")


def _resolve_glm5_tokenizer_dir(model_name: str) -> Path:
    path = Path(model_name)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_name,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "chat_template.jinja",
            ],
            local_files_only=_offline_mode(),
        )
    )


def _load_glm5_tokenizer_from_json(model_name: str) -> Tokenizer:
    """Load GLM-5.2's standard tokenizer.json when old Transformers lacks its alias."""
    from transformers import PreTrainedTokenizerFast

    tokenizer_dir = _resolve_glm5_tokenizer_dir(model_name)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"GLM-5.2 tokenizer.json not found under {tokenizer_dir}")
    if not tokenizer_config.is_file():
        raise FileNotFoundError(f"GLM-5.2 tokenizer_config.json not found under {tokenizer_dir}")

    raw_config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    eos_token = raw_config.get("eos_token", "<|endoftext|>")
    pad_token = raw_config.get("pad_token") or eos_token
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        eos_token=eos_token,
        pad_token=pad_token,
        model_max_length=raw_config.get("model_max_length", int(1e30)),
    )
    tokenizer.name_or_path = str(tokenizer_dir)
    return tokenizer


@cache
def _get_hf_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    model_name = _strip_adapter_suffix(model_name)

    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"

    kwargs: dict[str, Any] = {}
    if model_name == "prism-ml/Ternary-Bonsai-2-27B-gguf":
        # The GGUF repository has no standalone tokenizer; its companion is identical.
        model_name = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
        if Path(model_name).is_dir():
            raise ValueError(
                f"Local directory {model_name!r} shadows the pinned public tokenizer. "
                "Use an explicit local path or a registered tokenizer instead."
            )
        kwargs["revision"] = "3f926b415992eaa2ae9dd7b573706494d6bbf787"
        kwargs["fix_mistral_regex"] = False

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    except ValueError as exc:
        if (
            _flag_enabled(GLM5_FLAG_ENV)
            and _is_glm5_model(model_name)
            and "TokenizersBackend" in str(exc)
        ):
            return _load_glm5_tokenizer_from_json(model_name)
        raise
