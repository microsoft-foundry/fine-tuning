"""Feature-flag gating tests for DeepSeek-V4 Flash model routing.

DeepSeek-V4 Flash is gated behind ``INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4``: its attributes
are always resolvable, but the cookbook only auto-routes it to the DeepSeek
renderer (and the API only accepts the base model) when the flag is enabled.
It reuses the ``deepseekv3`` renderer because it shares the DeepSeek chat
format. It always trains in full precision (bf16); there is no NF4 path here.
"""

import pytest

from interactive_training import model_info


_DSV4 = "deepseek-ai/DeepSeek-V4-Flash"
_DSV4_0731 = "deepseek-ai/DeepSeek-V4-Flash-0731"

#: Every published DeepSeek-V4 release the cookbook routes. Changes here must
#: be checked against the service's supported-model allowlist before publishing.
_ALL_DSV4 = (_DSV4, _DSV4_0731)


@pytest.mark.parametrize("model_id", _ALL_DSV4)
def test_dsv4_attributes_resolve_regardless_of_flag(monkeypatch, model_id):
    monkeypatch.delenv("INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4", raising=False)
    attrs = model_info.get_model_attributes(model_id)
    assert attrs.organization == "deepseek-ai"
    assert attrs.version_str == "4"
    assert attrs.is_chat is True


def test_every_dsv4_release_is_registered():
    # A release that is downloaded and wired up but never added here resolves
    # to "Unknown model" at renderer selection, which is a runtime failure on
    # the first request rather than a build-time one.
    registered = {f"deepseek-ai/{name}" for name in model_info.get_deepseek_info() if "V4" in name}
    assert registered == set(_ALL_DSV4)


def test_dsv4_releases_report_the_same_size():
    # 0731 differs only in its dropped MTP head, so the trainable model is the
    # same size; divergent metadata here would misreport capacity planning.
    sizes = {model_info.get_model_attributes(m).size_str for m in _ALL_DSV4}
    assert sizes == {"293B-A14B"}


def test_dsv4_renderer_routing_requires_flag(monkeypatch):
    monkeypatch.delenv("INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4", raising=False)
    with pytest.raises(ValueError, match="feature-flagged"):
        model_info.get_recommended_renderer_names(_DSV4)


@pytest.mark.parametrize("model_id", _ALL_DSV4)
@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on"])
def test_dsv4_renderer_routing_enabled_by_flag(monkeypatch, truthy, model_id):
    monkeypatch.setenv("INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4", truthy)
    assert model_info.get_recommended_renderer_names(model_id) == [
        "deepseekv3",
        "deepseekv3_thinking",
    ]


@pytest.mark.parametrize("model_id", _ALL_DSV4)
@pytest.mark.parametrize("falsey", ["0", "false", "no", "off", ""])
def test_dsv4_renderer_routing_rejected_by_falsey_flag(monkeypatch, falsey, model_id):
    monkeypatch.setenv("INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4", falsey)
    with pytest.raises(ValueError, match="feature-flagged"):
        model_info.get_recommended_renderer_names(model_id)


def test_existing_deepseek_v3_models_are_not_gated(monkeypatch):
    # DeepSeek V3.x must keep working with the flag off (only V4 is gated).
    monkeypatch.delenv("INTERACTIVE_POST_TRAINING_ENABLE_DEEPSEEK_V4", raising=False)
    assert model_info.get_recommended_renderer_names(
        "deepseek-ai/DeepSeek-V3.2-Exp"
    ) == ["deepseekv3", "deepseekv3_thinking"]
