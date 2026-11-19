# Multimodal SFT: Image Classification

Prepare Stanford Dogs image-classification data, fine-tune
`gpt-4o-2024-08-06`, and inspect service-produced training and validation
metrics. The notebook uses all 120 classes, the first 50 sorted images per
class, and a 40/5/5 train/validation/unused split: 4,800 training rows, 600
validation rows, and 600 unused provenance rows.

## Run

```powershell
cd managed_training\02-multimodal-sft\02-image-classification
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
pip install -e ..\.. --no-deps
az login
jupyter lab notebooks\demo.ipynb
```

From macOS/Linux, use `cp .env.template .env` and `/` path separators.

Set `FOUNDRY_PROJECT_ENDPOINT` in the module-local `.env`. Preparation cells
make no Foundry calls. Executing the client, upload, and training cells
authenticates and submits or resumes a paid job. The recipe uses
`GlobalStandard`, seed 42, two epochs, and learning-rate multiplier 0.5.

The notebook preserves the selected JPEG bytes and records split membership,
row counts, and hashes before upload. It monitors the job to a terminal state
and downloads service-emitted loss and token-accuracy metrics on success.

Generated JSONL, hashes, job state, and downloaded metrics are written under
ignored `outputs/training-only/`. Saved state is bound to the project,
dataset hashes, and training recipe so reruns can resume the same job.

One shared cross-process `run-state.json.lock` protects **both** the file and job
journals. It is acquired before checking either state file and retained until
pending intent and every returned upload/job ID are saved. A competing kernel
fails before reading state or posting; retry after the first exits to reload its
recorded IDs. Normal exceptions and interrupts release the lock. A hard kill may
leave it behind: confirm no writer remains and reconcile local/remote state
before removing only the stale lock, never pending state to force another POST.
See the [shared-state concurrency boundary](../README.md) for filesystem limits.

Job creation disables automatic SDK retries. After an ambiguous submission
failure, reconcile the remote job before removing state or attempting another
submission.
The retained recipe persists `submission_pending` before the POST and refuses
another submission until the outcome is reconciled and the job ID is recorded.

File uploads also disable SDK retries. The notebook atomically saves each
upload's pending intent and then its returned ID in the same ignored
`file-state.json` **before** polling processing or uploading the next split.
A processing timeout resumes that recorded ID. A missing/lost upload response
requires independent remote-file reconciliation; do not clear pending state,
select a file by filename alone, or blindly repeat the upload.

Next: [`../03-video-action-recognition`](../03-video-action-recognition/README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
