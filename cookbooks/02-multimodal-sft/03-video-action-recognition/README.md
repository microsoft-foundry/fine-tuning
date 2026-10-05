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
cd cookbooks\02-multimodal-sft\03-video-action-recognition
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
az login
jupyter lab notebooks\demo.ipynb
```

The notebook records fine-tuning preprocessing summaries and the exact submitted
split alongside the result files. Its ignored run-state file prevents a rerun
from submitting a duplicate job; remove that file only when intentionally
starting a fresh run.

Preparation cells make no Foundry calls. Executing the client and training
cells authenticates, uploads data, and submits or resumes a paid job.
The recipe uses `gpt-4.1-2025-04-14`,
`GlobalStandard`, seed 0, and one supervised epoch.

Job creation disables automatic SDK retries. After an ambiguous submission
failure, reconcile the remote job before removing state or attempting another
submission.
The recipe persists `submission_pending` before the POST and refuses another
submission until the outcome is reconciled and the job ID is recorded.

## Contents

- `data/` - source, split, preprocessing, and licensing documentation.
- `outputs/` - ignored downloads, generated JSONL, run state, and metrics.

Next: `../../03-core-rft`.

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
