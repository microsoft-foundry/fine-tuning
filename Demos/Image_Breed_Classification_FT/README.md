## VLM Image Classification with Fine-Tuning

This project demonstrates how to use a Vision Language Model (VLM) with **Azure OpenAI** for multi-class image classification (120 dog breeds) comparing three approaches:
1. Zero-shot classification with a base `gpt-4o-2024-08-06` deployment
2. A fine-tuned Azure OpenAI vision model (LoRA SFT) on a down-sampled subset
3. A classic CNN baseline (MobileNetV3-Small) for grounding

<img title="dogs" alt="dogs" src="public/dogs.png" width="400">

### At-a-Glance Summary
| Aspect | Base Zero‑Shot | Fine-Tuned | CNN Baseline |
|--------|----------------|-----------|--------------|
| Accuracy (mean) | 73.67% | 82.67% (+9.0 pp) | 61.67% (-12.0 pp vs base) |
| Mean Latency (ms) | 1665 | 1506 (-9.6%) | — (not benchmarked here) |
| Tail Latency p99 (ms) | 2737 | 2303 (-15.9%) | — |
| Hosting Cost | None (deployment cost only) | Hosting + training amortization | Local infra |
| Adaptation | Prompt only | Task-specific | Task-specific |

> The CNN baseline (MobileNetV3-Small) is included only for comparative grounding; detailed implementation is in `cnn_baseline.py` (kept concise to maintain focus on Azure OpenAI fine-tuning).

### Table of Contents
1. [Dataset](#dataset)
2. [Repository Structure](#repository-structure)
3. [Baselines & Experimental Setup](#baselines--experimental-setup)
4. [Results](#results)
5. [Latency Evaluation](#latency-evaluation)
6. [Why Fine-Tune?](#why-fine-tune)
7. [Cost Analysis & ROI](#cost-analysis--roi)
8. [Reproducibility & Setup](#reproducibility--setup)
9. [Dataset Citation & Licensing](#dataset-citation--licensing)
10. [License](#license)
11. [Disclaimer](#disclaimer)


## Dataset
Original Stanford Dogs stats:
* 120 breeds
* 20,580 images
* Extra annotations (not used)

For cost control: **50 images per breed** → 6,000 images total → split 40 train / 5 val / 5 test.

## Repository Structure
```
.
├─ README.md                                # This file
├─ requirements.txt                         # Python dependencies (pinned)
├─ .env.sample                              # Environment variable template
├─ images_classification_vlm.ipynb          # Prep + FT + evaluation (renamed)
├─ latency_base_ft_models.ipynb             # Latency benchmarking
├─ cnn_baseline.py                          # Classic CV baseline
├─ latency_outputs/                         # Latency measurement CSVs
└─ public/                                  # Plots & diagram assets
```

## Baselines & Experimental Setup
**Models Compared**
* **Zero-shot VLM**: Azure OpenAI `gpt-4o-2024-08-06` deployed for interactive + Batch inference.
* **Fine-tuned VLM**: Same base model, LoRA SFT over 4,800 training images (2 epochs) using chat-style JSONL with image + constrained label output.
* **CNN Baseline**: MobileNetV3-Small (ImageNet init) fine-tuned 8 epochs. Provides grounding for how a lightweight task-specific model fares under limited per-class data.

**Baseline Intent (Important Clarification)**
The CNN baseline is deliberately lightweight and minimally tuned. Its role is purely to supply a familiar, traditional computer vision reference point—not to establish a state-of-the-art classical model. Do **not** interpret the absolute performance gap as an inherent limit of conventional CV; a stronger backbone (e.g. ConvNeXt / ViT) or deeper training schedule could narrow it. The comparison is designed to highlight how much capability you get from:
1. A zero-shot Azure OpenAI vision model with only prompt engineering
2. The incremental uplift gained by fine-tuning that same model
3. A modest classic model under identical low-data constraints

This framing helps readers reason about ROI for adopting Azure OpenAI fine-tuning versus investing in further classic model engineering.

**Data Budget Constraint**: 40 training images per class simulates a low-data regime where pretrained multimodal priors can shine.

**Evaluation Metrics**: Mean accuracy, per-class accuracy deltas (CSV), latency distribution (mean, p50, p90, p95, p99), variance, throughput approximation.

**Prompt Strategy (Base)**: Constrained system prompt enumerating the 120 canonical breed tokens; single low-detail image encoding.

**Fine-Tuning Format**: Chat messages with `input_text` + `input_image` and a single class label assistant response.

## Results
| Model | Mean Accuracy | Δ vs Base |
|-------|---------------|----------|
| Base `gpt-4o-2024-08-06` (zero-shot) | 73.67% | — |
| Fine-tuned `gpt-4o-2024-08-06` | **82.67%** | **+9.00 pp** |
| CNN Baseline (MobileNetV3-Small) | 61.67% | -12.00 pp |

<img title="Accuracy comparison" alt="accuracy_base_vs_ft_models" src="public/accuracy_base_vs_ft_models.png" width="700">

**Interpretation**
* Zero-shot already outperforms a lightweight task-specific CNN (+12 pp) → strong semantic prior.
* Fine-tuning compounds gains (+9 pp further) with only 2 epochs.
* CNN underperformance largely attributable to limited per-class diversity and modest backbone capacity.

## Latency Evaluation
End-to-end client-observed latency over 600 sequential requests per model.

| Model | Mean (ms) | P50 | P90 | P95 | P99 | Min | Max | Throughput (req/s) |
|-------|-----------|-----|-----|-----|-----|-----|-----|--------------------|
| Base (`gpt-4o-2024-08-06`) | 1665.08 | 1629.17 | 1963.85 | 2182.58 | 2737.33 | 1053.66 | 4693.65 | 0.601 |
| Fine-tuned (`gpt-4o-2024-08-06-ft`) | 1505.97 | 1505.29 | 1706.80 | 1819.48 | 2302.56 | 824.95 | 3381.80 | 0.664 |

**Observations**
* Mean latency ↓ ~9.6%; p99 tail ↓ ~15.9% → tighter distribution.
* Slight throughput gain (sequential proxy) from reduced variance.
* Variance (std dev) shrinks (324 → 241 ms) improving predictability.

<img title="Latency Histogram" alt="latency_histogram" src="public/latency_histogram.png" width="600">  
<img title="Latency Distribution Boxplot" alt="latency_boxplot" src="public/latency_boxplot.png" width="400">

## Why Fine-Tune?
Fine-tuning delivers accuracy uplift, tighter latency distribution, and more stable per-class performance. Even modest SFT (2 epochs) yields tangible ROI in a constrained data regime. Additional epochs or richer prompts likely increase gains (at added cost).

## Cost Analysis & ROI
### Batch Inference (Base Model)
Pricing (Azure OpenAI reference): Input $1.25 / 1M tokens, Output $5 / 1M tokens.

Test Set (600 images):
* System prompt tokens ≈ 494
* Image tokens (detail=low) ≈ 85 per image
* Output ≈ 6 tokens
* Approx total cost ≈ $0.59 (≈ $0.00098 / image)

### Fine-Tuning Costs (`gpt-4o-2024-08-06` East US 2, Sept 2025)
* Training: $27.5 / 1M training tokens
* Hosting: $1.7 / hour (while deployed)
* Inference: $2.5 / 1M input; $10 / 1M output

Token Accounting (2 epochs, 4,800 examples):
```
Per example ≈ 494 (system) + 85 (image) + 6 (target) = 585
Tokens / epoch = 585 * 4,800 = 2,808,000
Total (2 epochs) = 5,616,000 ≈ 5.616M
Training cost ≈ 5.616 * 27.5 = $155
```
<img title="FT training cost" alt="ft_training_cost" src="public/FT_training_cost.png" width="600">

> Small discrepancy vs screenshot due to approximate label token count distribution.

### Trade-Off Interpretation
* **Accuracy Benefit**: +9 pp reduces misclassifications across 120 labels.
* **Latency Advantage**: Lower and tighter tail → better UX & capacity planning.
* **Operational Overhead**: Requires monitoring, possible periodic retraining.
* **ROI Heuristic**: `(Added Correct Predictions * Value per Correct) > (Training + Amortized Hosting)` → proceed with FT.

## Reproducibility & Setup
### Prerequisites
* Azure OpenAI resource (vision + fine-tuning access)
* Python 3.10+
* (Optional) Kaggle account to fetch dataset

### Environment
```
python -m venv .venv
./.venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

### Environment Variables (`.env`)
```
AZURE_OPENAI_API_KEY="<key>"
AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# Base interactive deployment
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-2024-08-06"
AZURE_OPENAI_DEPLOYMENT_NAME_MODEL_VERSION="2024-08-06"

# Batch deployment (if separate)
AZURE_OPENAI_BATCH_DEPLOYMENT_NAME="gpt-4o-2024-08-06"
AZURE_OPENAI_BATCH_DEPLOYMENT_NAME_MODEL_VERSION="2024-08-06"

# Fine-tuned deployment (after job completion)
AZURE_OPENAI_FT_DEPLOYMENT_NAME="<your-finetuned-deployment-name>"
```

### Dataset Down-Sampling Logic
1. Enumerate breeds (deterministic ordering)
2. Take first 50 images per breed
3. Split 40 / 5 / 5 (train / val / test)
4. Emit JSONL for fine-tuning + evaluation

### JSONL Schema (Azure OpenAI Vision FT)
The fine-tuning JSONL uses standard chat format with a single image per example and a single-label textual response. (Matches `jsonl/train_classification.jsonl`, `val_classification.jsonl`, etc.)

Minimal example (line-delimited JSON – one object per line):
```
{"messages": [
    {"role": "system", "content": "Classify the following input image into one of the following categories: [Affenpinscher, Afghan Hound, ... , Yorkshire Terrier]."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_BYTES>", "detail": "low"}}
    ]},
    {"role": "assistant", "content": "Springer Spaniel"}
]}
```

### Notebook Flow (`images_classification_vlm.ipynb`)
1. Credentials & imports
2. Dataset sampling & JSONL creation
3. Batch evaluation (base model)
4. Fine-tuning job creation & deployment
5. Fine-tuned evaluation
6. Latency benchmarking (separate notebook)
7. Comparative summary

## Troubleshooting

### Common Issues

**Content Filter Triggered**
- Some images may trigger Azure content filters (faces, people, CAPTCHA-like patterns)
- Use the [official form](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUMlBQNkZMR0lFRldORTdVQzQ0TEI5Q1ExOSQlQCN0PWcu) for policy-aligned adjustments

**Training Job Fails**
- Verify JSONL format matches the schema shown above
- Ensure base64 image encoding is correct
- Check that system prompt includes all 120 breed labels

**Authentication Error**
- Verify `.env` contains correct API key and endpoint
- Run `az login` to refresh credentials
- Check that you have vision + fine-tuning access enabled

**Quota Exceeded**
- Request additional quota in Azure Portal → Azure OpenAI → Quotas
- Try a different Azure region with available capacity

**Batch API Timeout**
- Large batch jobs may take time; check status in Azure Portal
- Consider splitting into smaller batches

## Dataset Citation & Licensing
**Stanford Dogs Dataset**: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. *Novel Dataset for Fine-Grained Image Categorization.* FGVC Workshop, CVPR 2011.  
Source: http://vision.stanford.edu/aditya86/ImageNetDogs/ (Derived from ImageNet; original image licenses apply).  
Raw images are excluded from this repository; only directory structure references are used.

Please ensure compliance with the dataset’s original terms when downloading or redistributing.

## License
MIT License (see `LICENSE`).

## Disclaimer
Costs and pricing are illustrative and may change. Always consult current [Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/).
* Some images triggered content filters (faces / people / CAPTCHA-like patterns); approved modifications via Azure process may be required. Use the [official form](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUMlBQNkZMR0lFRldORTdVQzQ0TEI5Q1ExOSQlQCN0PWcu) for policy-aligned adjustments.
* Demo scope only—not production hygiene (secrets rotation, monitoring, retraining pipeline) is shown.
* Protect API keys and respect data governance policies.

