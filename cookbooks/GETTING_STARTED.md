# Getting started

## Prerequisites

- Python 3.11-3.14
- A Microsoft Foundry project and access to the models required by the selected demo
- Entra ID authentication available to `DefaultAzureCredential` (for example,
  `az login`, managed identity, Azure CLI, or workload identity)
- Quota and preview access explicitly required by the selected `catalog.yml` entry

## Create the central environment

From `cookbooks/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.lock
.\.venv\Scripts\python -m pip install -e . --no-deps
```

The lock consolidates retained notebook/helper requirements, and the editable install
makes the shared cookbook helpers importable from every demo. Notebook authors
must not add drifting per-demo requirements files. Update both `pyproject.toml`
and `requirements.lock` when a capability genuinely needs a new package.

## Configure without committing identifiers

```powershell
Copy-Item .env.template .env
```

Run this command from the selected cookbook directory, then open its
`notebooks/demo.ipynb`. Use that cookbook's template because RFT and the Retail
Capstone require additional settings.

Set `FOUNDRY_PROJECT_ENDPOINT` to the project endpoint shown by Foundry. Keep the
populated `.env` local. The shared loader validates HTTPS and the
`/api/projects/<project>` shape before constructing a client.

```python
from shared.auth import create_project_context
from shared.config import load_foundry_config

config = load_foundry_config()
with create_project_context(config) as context:
    jobs = list(context.openai_client.fine_tuning.jobs.list(limit=1))
```

No API key is required by the shared project context. Authentication failures are
reported by the Azure Identity SDK; they are not converted into anonymous or
key-based fallbacks.

Run preparation and validation cells before upload and training cells.
Executing the training cells submits or resumes live jobs and can incur
charges; there is no additional execution switch. For the Retail Capstone,
provision deployments through your approved management workflow before
running its inference comparison cells.

## Before a paid or long-running operation

1. Validate the exact runtime-generated or preserved JSONL files with
   `validate_jsonl(...).require_valid()`.
2. Verify train/validation/test isolation with `validate_split_isolation(...)`.
3. Record SHA-256 hashes in an `ExperimentManifest`.
4. Confirm the catalog prerequisites, model availability, cost tier, and runtime tier.
5. Use a meaningful `NameFactory` name and explicitly choose whether reuse is safe.

Runtime identifiers belong only below the demo's ignored `outputs/` directory.
Committed representative evidence must be sanitized and clearly labeled as a
past training run, not the reader's result.
