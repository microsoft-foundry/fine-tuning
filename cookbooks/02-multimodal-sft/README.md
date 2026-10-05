# Multimodal supervised fine-tuning

This track contains one canonical notebook per demo:

1. [`01-chart-reasoning`](01-chart-reasoning/README.md) - ChartQA image-and-question reasoning.
2. [`02-image-classification`](02-image-classification/README.md) - Stanford Dogs classification with integrated latency and cost analysis.
3. [`03-video-action-recognition`](03-video-action-recognition/README.md) - UCF101 action recognition from ordered privacy-preserving frames.

Shared implementation mechanics live in [`src/multimodal_common.py`](src/multimodal_common.py). Data, recorded evidence, environment configuration, and generated outputs remain owned by the individual demo folders.

Install the centrally pinned environment from [`../requirements.lock`](../requirements.lock). Every demo defaults to `FOUNDRY_RUN_PAID_JOBS=false` and `FOUNDRY_RUN_LIVE_EVALUATION=false`. The notebooks show checked-in representative evidence without remote inference unless live evaluation is explicitly enabled.
