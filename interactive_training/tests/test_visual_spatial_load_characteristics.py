import io
import json
import sys
import time
from types import SimpleNamespace

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk
from PIL import Image

from interactive_training.image_types import ImageChunk
from interactive_training.recipes.visual_spatial.training import (
    _PreparedPromptCache,
    _model_input_payload_bytes,
)
from interactive_training.renderers import base as renderer_base


def _peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset

    import resource

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; other supported Unix platforms report KiB.
    return int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)


@pytest.mark.parametrize(
    ("platform", "reported_peak", "expected_bytes"),
    [("win32", 4096, 4096), ("linux", 4, 4096), ("darwin", 4096, 4096)],
)
def test_peak_rss_units(monkeypatch, platform, reported_peak, expected_bytes):
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(peak_wset=reported_peak)
    )
    resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: SimpleNamespace(ru_maxrss=reported_peak),
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: process))
        patch.setitem(sys.modules, "resource", resource)
        patch.setattr(sys, "platform", platform)
        actual_bytes = _peak_rss_bytes()

    assert actual_bytes == expected_bytes


def _png_bytes(side_length: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (side_length, side_length), color="red").save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


@pytest.mark.parametrize("image_count", [1, 5])
@pytest.mark.parametrize("side_length", [8, 256])
@pytest.mark.parametrize("repetitions", [1, 10])
def test_visual_spatial_cache_load_characteristics(
    monkeypatch,
    tmp_path,
    image_count,
    side_length,
    repetitions,
    capsys,
):
    image_data = _png_bytes(side_length)
    image_urls = [
        f"https://example.test/image-{index}.png?sig=test"
        for index in range(image_count)
    ]
    downloads = []
    monkeypatch.setattr(
        renderer_base,
        "_download_image_reference",
        lambda reference: downloads.append(reference) or image_data,
    )

    sample_prompt = ModelInput(
        chunks=[
            *(ImageChunk(data=image_data, format="png", expected_tokens=1)
              for _ in image_urls),
            ModelInputChunk(tokens=[1, 2, 3]),
        ]
    )
    payload_bytes = _model_input_payload_bytes(sample_prompt)
    prompt_cache = _PreparedPromptCache(max_bytes=payload_bytes * 2)
    factory_calls = 0

    def build_prompt() -> ModelInput:
        nonlocal factory_calls
        factory_calls += 1
        chunks = [
            ImageChunk(
                data=renderer_base._load_image_reference(
                    image_url, cache_dir=str(tmp_path)
                ),
                format="png",
                expected_tokens=1,
            )
            for image_url in image_urls
        ]
        chunks.append(ModelInputChunk(tokens=[1, 2, 3]))
        return ModelInput(chunks=chunks)

    started_at = time.perf_counter()
    for repetition in range(repetitions):
        prompt_cache.get_or_create(repetition % 2, build_prompt)
    wall_time_seconds = time.perf_counter() - started_at

    expected_misses = min(repetitions, 2)
    expected_hits = repetitions - expected_misses
    assert factory_calls == expected_misses
    assert prompt_cache.misses == expected_misses
    assert prompt_cache.hits == expected_hits
    assert downloads == image_urls
    assert prompt_cache.current_bytes == payload_bytes * expected_misses
    assert prompt_cache.current_bytes <= prompt_cache.max_bytes

    report = {
        "image_count": image_count,
        "image_side_pixels": side_length,
        "repetitions": repetitions,
        "wall_time_seconds": wall_time_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "prompt_cache_hit_rate": expected_hits / repetitions,
        "downloads": len(downloads),
        "serialized_bytes": prompt_cache.current_bytes,
    }
    with capsys.disabled():
        print(json.dumps(report, sort_keys=True))