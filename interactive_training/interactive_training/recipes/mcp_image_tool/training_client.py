"""Pixel-bounded submissions without changing the optimizer batch."""

from __future__ import annotations

import io
from types import SimpleNamespace

from PIL import Image

from interactive_training.rl.train_azure import AzureSDKTrainingClient


def split_image_batches(batch, max_pixels: int = 25_000_000):
    batches = []
    current = []
    current_pixels = 0
    for datum in batch:
        pixels = 0
        for chunk in datum.model_input.chunks:
            if hasattr(chunk, "tokens"):
                continue
            with Image.open(io.BytesIO(chunk.data)) as image:
                pixels += image.width * image.height
        if pixels > max_pixels:
            raise ValueError("A single training example exceeds the aggregate image-pixel limit")
        if current and current_pixels + pixels > max_pixels:
            batches.append(current)
            current = []
            current_pixels = 0
        current.append(datum)
        current_pixels += pixels
    if current:
        batches.append(current)
    return batches


class _CombinedForwardFuture:
    def __init__(self, futures):
        self.futures = futures

    async def result_async(self):
        outputs = []
        for future in self.futures:
            result = await future.result_async()
            outputs.extend(result.loss_fn_outputs)
        return SimpleNamespace(
            loss_fn_outputs=outputs,
            metrics={"forward_backward/pixel_bounded_requests": len(self.futures)},
        )


class MCPImageTrainingClient(AzureSDKTrainingClient):
    async def forward_backward_async(self, batch, loss_fn=None, loss_fn_config=None):
        batches = split_image_batches(batch)
        if len(batches) <= 1:
            return await super().forward_backward_async(batch, loss_fn, loss_fn_config)
        futures = []
        for microbatch in batches:
            futures.append(await super().forward_backward_async(
                microbatch, loss_fn, loss_fn_config,
            ))
        return _CombinedForwardFuture(futures)