# Multimodal SFT: Chart Reasoning

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to validate and submit the
original demo's exact ChartQA v4 training and validation files for multimodal
supervised fine-tuning. The workflow stops after the service reaches a terminal
job state and downloads the emitted training metrics.

## Run

```powershell
cd managed_training\02-multimodal-sft\01-chart-reasoning
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
pip install -e ..\.. --no-deps
az login
jupyter lab notebooks\demo.ipynb
```

From macOS/Linux, use `cp .env.template .env` and `/` path separators.

The notebook uses `DefaultAzureCredential`,
`AIProjectClient.get_openai_client()`, model `gpt-4.1-2025-04-14`, seed 42,
one epoch, and `GlobalStandard` training. Executing the upload/training cells
submits or resumes a paid job.
Batch size and learning-rate multiplier are omitted from the request so the
service selects its supported defaults; they are not sent as JSON `null`.

Source validation cells run locally without creating Foundry clients or
requiring a base-model deployment.

Each execution uses an ignored `outputs/runs/<run-id>/run-state.json`. Reusing
the same `FOUNDRY_RUN_ID` resumes the same uploaded files and job; a new run ID
creates fresh state. This prevents an interrupted rerun from submitting a
duplicate job.

The shared cross-process `run-state.json.lock` is acquired **before** loading
state and held through uploads, pending intent, saved file/job IDs, and monitoring.
The metrics cell reacquires the same lock and reloads current state before
updating it. A competing kernel fails before reading or changing the journal;
rerun after the first exits to resume its recorded IDs. Exceptions/interrupts
release ownership. After a hard kill, confirm no writer remains and reconcile
local/remote state before removing only a stale lock, never the pending journal.
See the [shared-state concurrency boundary](../README.md) before using separate
copies of the notebook or outputs directory.

Job creation disables automatic SDK retries. After an ambiguous submission
failure, reconcile the remote job before clearing state or using a new run ID.
Resume searches iterate SDK pages, including pending jobs, and reject a
matching job with a different seed or epoch recipe. An unresolved pending
submission is not automatically posted again.

File uploads also send a single POST with SDK retries disabled. Pending state
is saved before upload, and each returned file ID is saved before processing
is polled. If the upload response is lost, reconcile the remote contents and
purpose, then record the confirmed `training_file_id` or `validation_file_id`
in the same run-state file. Do not select by filename alone, clear pending
state, or choose a new run ID to bypass an unknown upload outcome.

The only effectiveness evidence produced is service-emitted training and
validation loss and mean token accuracy. The workflow does not deploy or invoke
the fine-tuned model.

## Contents

- `data/preserved/` - byte-identical v4 train and validation JSONL from the
  current-branch source demo.
- `data/hash-manifest.csv` - exact row counts, byte counts, and SHA-256 hashes.
- `outputs/` - ignored run state, service result files, and metric summaries.

Next: [`../02-image-classification`](../02-image-classification/README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
