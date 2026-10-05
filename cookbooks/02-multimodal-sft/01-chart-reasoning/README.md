# Multimodal SFT: Chart Reasoning

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to prepare ChartQA, fine-tune or reuse a GPT-4.1 job, deploy externally in Foundry, and compare base versus fine-tuned quality on an untouched holdout.

The recorded run completed successfully but regressed from 90% to 80% on 10 held-out questions. This is intentionally retained as evidence that training success is not quality success.

## Run

```powershell
cd cookbooks\02-multimodal-sft\01-chart-reasoning
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
az login
jupyter lab notebooks\demo.ipynb
```

`FOUNDRY_RUN_PAID_JOBS=false` prepares and validates data without creating a job. Live preparation records split row IDs and hashes, and the guarded training cell verifies that a reused job's remote filenames contain those exact hashes. `FOUNDRY_RUN_LIVE_EVALUATION=false` displays checked-in representative evidence without remote inference; set it to `true` only after configuring both deployment names. Live evaluation scores all 50 untouched holdout examples, writes predictions and metrics under `outputs/`, and records an improvement, regression, or tie verdict.

## Contents

- `data/` - source documentation, byte-preserved committed JSONL, and SHA-256 manifest.
- `assets/` - sanitized ChartQA example and recorded regression evidence.
- `outputs/` - ignored live JSONL, predictions, and run artifacts.

Next: [`../02-image-classification`](../02-image-classification/README.md).

> Recorded accuracy varies with sampling, image encoding, model/deployment version, service conditions, and equivalence rules.
