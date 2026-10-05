# RFT Countdown Foundry SDK 2.x demo

The two notebooks demonstrate training-only reinforcement fine-tuning of the
exact `qwen3.6-35b-a3b` catalog model with a model grader and a Python grader.
They use Microsoft Foundry SDK 2.x, Entra ID, and the authenticated child client
returned by `AIProjectClient.get_openai_client()`. They do not use API keys or
construct OpenAI clients directly.

## Prerequisites

- Python 3.10 or later
- Azure CLI signed in with `az login`
- Access to a supported Foundry project, preferring North Central US
- The `o3-mini` grader deployment for the model-grader variant

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<resource>.services.ai.azure.com/api/projects/<project>"
$env:FOUNDRY_BASE_MODEL = "qwen3.6-35b-a3b"
```

`FOUNDRY_PROJECT_ENDPOINT` is the project endpoint shown in the Foundry portal.
Authentication is exclusively through `DefaultAzureCredential`.

## Run

Open and run either notebook in order, or execute both from the directory:

```powershell
.\.venv\Scripts\python.exe scripts\configure_notebooks.py
.\.venv\Scripts\python.exe scripts\execute_notebooks.py
```

The notebooks hash-check the preserved 100-row training, 50-row validation, and
100-row evaluation files; upload the unchanged training and validation files;
submit or resume GlobalStandard RFT jobs; wait for terminal status; and download
result files. No deployment or inference is performed. Structured ignored
evidence is written under `outputs/loom-model-runs/`.
