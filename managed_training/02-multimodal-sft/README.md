# Multimodal supervised fine-tuning

This track contains one canonical notebook per demo:

1. [`01-chart-reasoning`](01-chart-reasoning/README.md) - ChartQA image-and-question reasoning.
2. [`02-image-classification`](02-image-classification/README.md) - Stanford Dogs classification.
3. [`03-video-action-recognition`](03-video-action-recognition/README.md) - UCF101 action recognition from ordered video frames.

Reusable preprocessing and training helpers live in [`src/multimodal_common.py`](src/multimodal_common.py). Each demo owns its source-data recipe and blank environment template. Generated data, run state, and service training metrics are ignored, not customer shipping assets.

Install the centrally pinned environment from
[`../requirements.lock`](../requirements.lock). The notebooks prepare exact
datasets, submit fine-tuning jobs, and summarize service training
metrics.

Preparation cells do not upload files, create jobs, or resume remote monitoring.
Configure a project with access to the recipe's model before executing the
client, upload, and training cells, which can incur charges. These notebooks
do not deploy models or perform inference or base-model comparisons.

Copy the selected module's `.env.template` to `.env` in that same module,
not the repository root. Creation POSTs disable SDK retries; if submission
fails ambiguously, reconcile the remote job before attempting another run.
