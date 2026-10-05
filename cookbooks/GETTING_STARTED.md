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

The lock consolidates direct source-demo requirements, and the editable install
makes the shared cookbook helpers importable from every demo. Notebook authors
must not add drifting per-demo requirements files. Update both `pyproject.toml`
and `requirements.lock` when a capability genuinely needs a new package.

## Configure without committing identifiers

```powershell
Copy-Item shared\templates\.env.template .env
```

Set `FOUNDRY_PROJECT_ENDPOINT` to the project endpoint shown by Foundry. Keep the
populated `.env` local. The shared loader validates HTTPS and the
`/api/projects/<project>` shape before constructing a client.

```python
from shared.auth import create_project_context
from shared.config import load_foundry_config

config = load_foundry_config()
with create_project_context(config) as context:
    models = list(context.project_client.deployments.list())
```

No API key is required by the shared project context. Authentication failures are
reported by the Azure Identity SDK; they are not converted into anonymous or
key-based fallbacks.

## Before a paid or long-running operation

1. Validate the exact runtime-generated or preserved JSONL files with
   `validate_jsonl(...).require_valid()`.
2. Verify train/validation/test isolation with `validate_split_isolation(...)`.
3. Record SHA-256 hashes in an `ExperimentManifest`.
4. Confirm the catalog prerequisites, availability, cost tier, and runtime tier.
5. Use a meaningful `NameFactory` name and explicitly choose whether reuse is safe.

Runtime identifiers belong only below the demo's ignored `outputs/` directory.
Committed representative evidence must be sanitized and clearly labeled as a
past run, not the reader's result.
