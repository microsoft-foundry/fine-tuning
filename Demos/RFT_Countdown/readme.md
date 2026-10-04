# RFT Countdown Foundry SDK 2.x demo

The two notebooks demonstrate reinforcement fine-tuning with a model grader and
a Python grader. They use Microsoft Foundry SDK 2.x, Entra ID, and the
authenticated child client returned by `AIProjectClient.get_openai_client()`.
They do not use API keys or construct OpenAI clients directly.

## Prerequisites

- Python 3.10 or later
- Azure CLI signed in with `az login`
- Access to an East US 2 Foundry project with RFT enabled
- Existing baseline, grader, and fine-tuned model deployments used by the notebooks

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<resource>.services.ai.azure.com/api/projects/<project>"
$env:FOUNDRY_BASELINE_DEPLOYMENT = "<baseline-deployment>"
```

`FOUNDRY_PROJECT_ENDPOINT` is the project endpoint shown in the Foundry portal.
Authentication is exclusively through `DefaultAzureCredential`.

## Run

Open and run either notebook in order, or execute both from the directory:

```powershell
.\.venv\Scripts\python.exe scripts\configure_notebooks.py
.\.venv\Scripts\python.exe scripts\execute_notebooks.py
```

The notebooks upload data, submit or resume RFT jobs, wait for terminal job
status, verify the existing fine-tuned deployment through the Foundry project
client, and run exact Countdown inference validation.
