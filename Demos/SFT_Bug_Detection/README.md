# SFT Bug Detection Demo

**Technique**: Supervised Fine-Tuning (SFT) — Distillation  
**Use Case**: Training-only validation of code bug detection SFT
**Base Model**: OpenAI-OSS `gpt-oss-120b`, catalog version `1`
**Training Tier**: `GlobalStandard` only
**Dataset**: 224 training / 20 validation examples across 10 bug categories  

## What You'll Learn

1. **Dataset integrity** — Verify the exact 224/20 source files and preserve the 10-row held-out scale
2. **Fine-tuning** — Submit one `GlobalStandard` SFT job with 2 epochs and learning rate 0.8
3. **Terminal monitoring** — Follow service events until success, failure, or cancellation
4. **Training validation** — Download service result CSVs and assess only validation-loss and token-accuracy trends

## Safety and Scope

- East US 2 is never used.
- No base or fine-tuned model is deployed.
- No inference or judge comparison is run.
- Runtime evidence is written under ignored `outputs/loom-model-runs/`.
- Conclusions are limited to service result-file training trends and do not claim downstream quality.

## Prerequisites

- Microsoft Foundry project with fine-tuning access
- Python 3.12
- `pip install -r requirements.txt`
- A credential supported by `DefaultAzureCredential` with Foundry User access

The notebook creates `AIProjectClient` with `DefaultAzureCredential` and obtains
the official OpenAI child client with `get_openai_client()`. Configure the
project before running:

```properties
AZURE_AI_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
AZURE_AI_REGION=<region>
```

## Files

| File | Description |
|------|-------------|
| `Bug_Detection_Fine_Tuning.ipynb` | Main notebook — run cells sequentially |
| `requirements.txt` | Python dependencies |
| `.env.template` | Authentication and scope notes; no secrets |

Training data is in `../../Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/`.

## Bug Categories Covered

The dataset covers 10 types of bugs across Python, JavaScript, Java, and C++:

1. Off-by-one errors
2. Null/undefined reference
3. Type mismatches
4. Resource leaks
5. Race conditions
6. Buffer overflows
7. Integer overflow
8. Logic errors
9. Unhandled exceptions
10. Security vulnerabilities (SQL injection, XSS)
