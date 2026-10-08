import pytest
from interactive_training import model_info


@pytest.mark.parametrize(
    "name", ["prism-ml/Ternary-Bonsai-2-27B-gguf", "ternary-bonsai-2-27b-gguf"]
)
def test_bonsai_attributes_and_renderer(name):
    attributes = model_info.get_model_attributes(name)
    assert attributes.organization == "prism-ml"
    assert attributes.size_str == "27B"
    assert attributes.is_chat and not attributes.is_vl
    assert model_info.get_recommended_renderer_names(name) == [
        "qwen3_8",
        "qwen3_8_disable_thinking",
    ]
