# Multimodal SFT: Chart Reasoning

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to validate and submit the
original demo's exact ChartQA v4 training and validation files for multimodal
supervised fine-tuning. The workflow stops after the service reaches a terminal
job state and downloads the emitted training metrics.

## Run

```powershell
cd cookbooks\02-multimodal-sft\01-chart-reasoning
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
az login
jupyter lab notebooks\demo.ipynb
```

The notebook uses `DefaultAzureCredential`,
`AIProjectClient.get_openai_client()`, model `gpt-4.1-2025-04-14`, seed 42,
one epoch, and `GlobalStandard` training. Executing the upload/training cells
submits or resumes a paid job.

Source validation cells run locally without creating Foundry clients or
requiring a base-model deployment.

Each execution uses an ignored `outputs/runs/<run-id>/run-state.json`. Reusing
the same `FOUNDRY_RUN_ID` resumes the same uploaded files and job; a new run ID
creates fresh state. This prevents an interrupted rerun from submitting a
duplicate job.

Job creation disables automatic SDK retries. After an ambiguous submission
failure, reconcile the remote job before clearing state or using a new run ID.
Resume searches iterate SDK pages, including pending jobs, and reject a
matching job with a different seed or epoch recipe. An unresolved pending
submission is not automatically posted again.

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
