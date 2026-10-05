# Multimodal SFT: Video Action Recognition

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to retrieve UCF101, preserve official CSV partitions, sample clips deterministically, extract ordered privacy-preserving low-frequency color frames, fine-tune or reuse GPT-4.1, and evaluate overall plus per-class action quality.

The recorded three-class run stayed flat at 73.33% overall. Fine-tuned recall varied from 1.00 for Basketball to 0.40 for TennisSwing, demonstrating why class-level analysis is required.

## Run

```powershell
cd cookbooks\02-multimodal-sft\03-video-action-recognition
Copy-Item .env.template .env
pip install -r ..\..\requirements.lock
az login
jupyter lab notebooks\demo.ipynb
```

`FOUNDRY_RUN_LIVE_EVALUATION=false` displays checked-in representative evidence without remote inference; set it to `true` only after configuring both deployment names.

The notebook prints fine-tuning preprocessing summaries when the service skips image rows for safety-policy reasons. Treat class balance after those skips as part of the model acceptance decision; a succeeded job does not imply that every submitted image row was used.

## Contents

- `data/` - source, split, privacy, and licensing documentation.
- `assets/` - sanitized frame samples and recorded quality evidence.
- `outputs/` - ignored downloaded data, generated JSONL, holdout manifests, predictions, and plots.

Next: `../../03-core-rft`.

> Video quality and safety acceptance vary with clip sampling, frame count, preprocessing, model/deployment version, and service conditions.
