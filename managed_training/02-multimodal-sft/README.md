# Multimodal supervised fine-tuning

This track contains one canonical notebook per demo:

1. [`01-chart-reasoning`](01-chart-reasoning/README.md) - ChartQA image-and-question reasoning.
2. [`02-image-classification`](02-image-classification/README.md) - Stanford Dogs classification.
3. [`03-video-action-recognition`](03-video-action-recognition/README.md) - UCF101 action recognition from ordered video frames.

Reusable preprocessing and training helpers live in [`src/multimodal_common.py`](src/multimodal_common.py). Each demo owns its source-data recipe and blank environment template. Generated data, run state, and service training metrics are ignored, not customer shipping assets.

Install the centrally pinned environment from
[`../requirements.lock`](../requirements.lock) and complete the editable
cookbook install in [Getting started](../GETTING_STARTED.md#create-the-central-environment)
so the shared submission lock is importable. The notebooks prepare exact
datasets, submit fine-tuning jobs, and summarize service training
metrics.

Preparation cells do not upload files, create jobs, or resume remote monitoring.
Configure a project with access to the recipe's model before executing the
client, upload, and training cells, which can incur charges. These notebooks
do not deploy models or perform inference or base-model comparisons.

Copy the selected module's `.env.template` to `.env` in that same module,
not the repository root. Creation POSTs disable SDK retries; if submission
fails ambiguously, reconcile the remote job before attempting another run.

All three notebooks hold the shared cross-process `run-state.json.lock` before
reading saved state, through pending intent, each upload/job POST, and returned-ID
persistence. Image uses that **same run lock for both** its file and job journals.
Chart and video retain the lock while monitoring because they continue updating
the journal; chart's later metrics cell reacquires it and reloads current state.
The second kernel fails before reading state or sending a POST; rerun after the
first writer exits to reuse its recorded IDs. A normal exit, exception, or
interrupt releases the lock. A hard-killed kernel can leave it behind: confirm
no writer is active and reconcile local/remote state before removing only the
stale lock. Never remove pending state or an active lock to force a resubmission.
This coordinates kernels sharing the **same state path on a filesystem with
atomic exclusive-file creation**, not independent copies or separate machines
with unrelated outputs directories; it is not service-wide exactly-once delivery.

Chart, image, and video uploads also disable SDK retries and record pending uploads
before sending a single POST. A known file ID is saved before waiting for
processing. If an upload response is lost, keep the pending state, reconcile
the remote file's contents and purpose, and record its confirmed ID in that
same state; a matching filename alone is not sufficient evidence to resume.
