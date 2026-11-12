# Multimodal SFT: Video Action Recognition

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to retrieve UCF101, preserve
the source demo's official train and validation partitions, sample clips
deterministically, extract ordered color frames, submit a GPT-4.1 fine-tuning
job, and inspect service training metrics.

Each selected color frame is processed with a 201 by 201 Gaussian blur at its
original resolution and JPEG-encoded. The notebook preserves temporal order
without resizing, edge extraction, or timestamp overlays. It uses 36 training
clips and 12 validation clips across three classes, with three frames per clip.

## Run

```powershell
cd managed_training\02-multimodal-sft\03-video-action-recognition
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
pip install -e ..\.. --no-deps
az login
jupyter lab notebooks\demo.ipynb
```

From macOS/Linux, use `cp .env.template .env` and `/` path separators.

The notebook records fine-tuning preprocessing summaries and the exact submitted
split alongside the result files. Its ignored run-state file prevents a rerun
from submitting a duplicate job; remove that file only when intentionally
starting a fresh run.

The shared cross-process `run-state.json.lock` is acquired before reading state
and held through uploads, pending intent, saved file/job IDs, and monitoring,
including later status/history updates. A second kernel fails before inspecting
state or sending a POST; retry after the writer exits to reload its saved IDs.
Normal exceptions/interrupts release ownership. After a hard kill, confirm no
writer remains and reconcile local/remote state before removing only a stale
lock, never pending state to bypass an unknown outcome. See the
[shared-state concurrency boundary](../README.md) for filesystem limits.

Preparation cells make no Foundry calls. Executing the client and training
cells authenticates, uploads data, and submits or resumes a paid job.
The recipe uses `gpt-4.1-2025-04-14`,
`GlobalStandard`, seed 0, and one supervised epoch.

Job creation disables automatic SDK retries. After an ambiguous submission
failure, reconcile the remote job before removing state or attempting another
submission.
The recipe persists `submission_pending` before the POST and refuses another
submission until the outcome is reconciled and the job ID is recorded.

File uploads also disable SDK retries and record pending state before a single
POST. Returned file IDs are saved before processing waits, so an interrupted
wait resumes that file rather than uploading it again. If the upload response
is lost, reconcile the remote contents and purpose and record the confirmed
`training_file_id` or `validation_file_id` in the same run-state file. Do not
select by filename alone or clear pending state to force another upload.

Job monitoring uses one 24-hour monotonic deadline. Each retrieve disables SDK
retries and uses a request timeout of at most 60 seconds or the remaining
budget, whichever is smaller. Transient retries and polling sleeps share that
deadline. The latest response, status history, and known job ID are saved on
every poll, including before a timeout is reported. A timeout never cancels the
remote job: rerun the training cell with the same state and input hashes to
resume monitoring, not to submit a replacement job. Upload processing and
post-terminal event/metric downloads are separate from this monitoring budget.

## Contents

- `data/` - source, split, preprocessing, and licensing documentation.
- `outputs/` - ignored downloads, generated JSONL, run state, and metrics.

Next: `../../03-core-rft`.

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
