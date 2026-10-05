# Multimodal SFT: Image Classification

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for deterministic Stanford Dogs preparation, managed safety-filter accounting, GPT-4o vision SFT, per-class and robustness evaluation, integrated latency benchmarking, and cost estimation.

The notebook incorporates all useful methodology and results from the former separate latency notebook; no second notebook is published.

## Recorded evidence

- Historical 120-class study: accuracy 73.67% to 82.67% (+9.0 pp).
- Bounded four-class rerun: both models 93.75% on 16 holdout images.
- Bounded latency smoke test: mean 2120.7 ms to 1525.0 ms over eight requests per model.
- Historical latency study: mean 1664.5 ms to 1505.7 ms and p99 2742.0 ms to 2303.2 ms over 599 successful requests per model.

For the bounded combined demo, acceptance means held-out quality does not regress and mean latency improves; an accuracy uplift is not required. Safety-filter attrition remains a limitation, and training is invalid if filtering removes all examples from any class.

## Run

```powershell
cd cookbooks\02-multimodal-sft\02-image-classification
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
az login
jupyter lab notebooks\demo.ipynb
```

`FOUNDRY_RUN_LIVE_EVALUATION=false` displays checked-in quality and latency evidence without remote inference. Setting it to `true` runs both the held-out base-versus-fine-tuned quality evaluation and the integrated latency benchmark, so both deployment names must be configured.

## Contents

- `data/` - source documentation, byte-preserved train/validation JSONL, hashes, and sanitized split membership.
- `assets/` - accuracy, latency, cost, aggregate metrics, and sanitized holdout predictions.
- `outputs/` - ignored downloads, generated JSONL, predictions, latency records, and plots.

Next: [`../03-video-action-recognition`](../03-video-action-recognition/README.md).

> Accuracy, safety acceptance, latency, tokens, and cost vary with sampling, preprocessing, service load, deployment configuration, and pricing date.
