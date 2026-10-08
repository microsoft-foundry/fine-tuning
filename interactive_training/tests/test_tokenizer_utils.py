import json
from unittest.mock import Mock, call

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from interactive_training import tokenizer_utils


def _write_tiny_tokenizer(path):
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<|endoftext|>": 1, "hello": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "model_max_length": 2048,
            }
        ),
        encoding="utf-8",
    )


def _write_glm5_config(path):
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def clear_tokenizer_cache():
    tokenizer_utils._get_hf_tokenizer.cache_clear()
    yield
    tokenizer_utils._get_hf_tokenizer.cache_clear()


@pytest.fixture
def auto_tokenizer_loader(monkeypatch):
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    loader = Mock(return_value=object())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    return loader


@pytest.mark.parametrize("suffix", ["", ":adapter"])
def test_bonsai_uses_pinned_equivalent_companion_tokenizer(auto_tokenizer_loader, suffix):
    tokenizer_utils.get_tokenizer("prism-ml/Ternary-Bonsai-2-27B-gguf" + suffix)
    auto_tokenizer_loader.assert_called_once_with(
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        use_fast=True,
        revision="3f926b415992eaa2ae9dd7b573706494d6bbf787",
        fix_mistral_regex=False,
    )


def test_bonsai_does_not_load_shadowing_companion(auto_tokenizer_loader, monkeypatch, tmp_path):
    (tmp_path / "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="shadows"):
        tokenizer_utils.get_tokenizer("prism-ml/Ternary-Bonsai-2-27B-gguf")
    auto_tokenizer_loader.assert_not_called()


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-32B",
        "openai/gpt-oss-20b",
        "other/model",
        "other/model-Instruct",
        "other/model.60",
        "other-mirror/model",
        "./other/model",
        "/models/other/model",
    ],
)
def test_other_tokenizers_do_not_receive_model_specific_trust_or_revision(auto_tokenizer_loader, model_name):
    assert tokenizer_utils.get_tokenizer(model_name) is auto_tokenizer_loader.return_value

    auto_tokenizer_loader.assert_called_once_with(model_name, use_fast=True)


@pytest.mark.parametrize("suffix", ["", ":adapter"])
@pytest.mark.parametrize(
    "model_name",
    [
        r"C:\models\other\model",
        "C:/models/other/model",
        r"d:\model cache\model",
        r"C:models\model",
        r"\\server\share\other\model",
        "//server/share/other/model",
        r"\\server\share",
        "//server/share",
        r"\\?\UNC\server\share",
        r"\\?\C:\models\model",
        r"\\?\UNC\server\share\model",
        "/models/other/model",
        "./models/other-model",
        "Qwen/Qwen3-32B",
    ],
)
def test_tokenizer_adapter_suffix_preserves_paths(
    auto_tokenizer_loader, model_name, suffix
):
    assert (
        tokenizer_utils.get_tokenizer(model_name + suffix)
        is auto_tokenizer_loader.return_value
    )

    auto_tokenizer_loader.assert_called_once_with(model_name, use_fast=True)


@pytest.mark.parametrize("suffix", ["", ":adapter"])
def test_glm5_detection_preserves_windows_drive(monkeypatch, tmp_path, suffix):
    _write_glm5_config(tmp_path)
    model_name = r"C:\models\glm5"
    path = Mock(return_value=tmp_path)
    monkeypatch.setattr(tokenizer_utils, "Path", path)

    assert tokenizer_utils._is_glm5_model(model_name + suffix)

    path.assert_called_once_with(model_name)


@pytest.mark.parametrize(
    "model_name",
    ["custom/tokenizer", "other/model", "other/model:adapter"],
)
def test_custom_registry_takes_precedence_over_cached_hf_tokenizer(
    monkeypatch, auto_tokenizer_loader, model_name
):
    monkeypatch.setattr(tokenizer_utils, "_CUSTOM_TOKENIZER_REGISTRY", {})
    assert tokenizer_utils.get_tokenizer(model_name) is auto_tokenizer_loader.return_value
    auto_tokenizer_loader.reset_mock()
    first, second = object(), object()
    factory = Mock(side_effect=[first, second])
    tokenizer_utils.register_tokenizer(model_name, factory)

    assert tokenizer_utils.get_tokenizer(model_name) is first
    assert tokenizer_utils.get_tokenizer(model_name) is second

    assert factory.call_args_list == [call(), call()]
    auto_tokenizer_loader.assert_not_called()


def test_custom_registry_lookup_does_not_strip_suffix(monkeypatch, auto_tokenizer_loader):
    factory = Mock(return_value=object())
    monkeypatch.setattr(
        tokenizer_utils, "_CUSTOM_TOKENIZER_REGISTRY", {"other/model": factory}
    )

    assert (
        tokenizer_utils.get_tokenizer("other/model:adapter")
        is auto_tokenizer_loader.return_value
    )

    factory.assert_not_called()
    auto_tokenizer_loader.assert_called_once_with(
        "other/model",
        use_fast=True,
    )


def test_glm5_tokenizers_backend_error_is_not_hidden_when_flag_off(monkeypatch, tmp_path):
    _write_tiny_tokenizer(tmp_path)

    from transformers.models.auto.tokenization_auto import AutoTokenizer

    def _raise_tokenizers_backend(*args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist")

    monkeypatch.delenv(tokenizer_utils.GLM5_FLAG_ENV, raising=False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raise_tokenizers_backend)

    with pytest.raises(ValueError, match="TokenizersBackend"):
        tokenizer_utils.get_tokenizer("zai-org/GLM-5.2-FP8")


def test_glm5_tokenizer_json_fallback_when_feature_flag_enabled(monkeypatch, tmp_path):
    _write_tiny_tokenizer(tmp_path)

    from transformers.models.auto.tokenization_auto import AutoTokenizer

    def _raise_tokenizers_backend(*args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist")

    monkeypatch.setenv(tokenizer_utils.GLM5_FLAG_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raise_tokenizers_backend)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *args, **kwargs: str(tmp_path),
    )

    tokenizer = tokenizer_utils.get_tokenizer("zai-org/GLM-5.2-FP8")

    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.pad_token == "<|endoftext|>"
    assert tokenizer.encode("hello", add_special_tokens=False) == [2]


def test_glm5_tokenizer_json_fallback_for_local_model_dir(monkeypatch, tmp_path):
    _write_tiny_tokenizer(tmp_path)
    _write_glm5_config(tmp_path)

    from transformers.models.auto.tokenization_auto import AutoTokenizer

    def _raise_tokenizers_backend(*args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist")

    monkeypatch.setenv(tokenizer_utils.GLM5_FLAG_ENV, "1")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raise_tokenizers_backend)

    tokenizer = tokenizer_utils.get_tokenizer(str(tmp_path))

    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.pad_token == "<|endoftext|>"
    assert tokenizer.encode("hello", add_special_tokens=False) == [2]
