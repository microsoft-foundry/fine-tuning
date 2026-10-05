# Microsoft Foundry fine-tuning cookbooks

This clean-slate cookbook root provides the shared contracts for a six-stage
learning journey from a first supervised fine-tuning job through multimodal,
reinforcement, agentic, synthetic-data, and distillation workflows.

The track folders contain **14 canonical end-to-end notebooks**. `catalog.yml`
records `validation.status: partial`: structure, notebook syntax, metadata,
preserved dataset hashes, sanitized assets, and offline-safe paths are
validated. Running upload and training cells performs live Foundry operations
and can incur charges.

Notebooks demonstrate optimization through service-produced loss or reward
metrics. Only the Retail Capstone also includes deployment verification and
base-versus-fine-tuned comparisons when those cells are executed.

## Start here

1. Read [GETTING_STARTED.md](GETTING_STARTED.md) and install the central lock.
2. Follow [LEARNING_PATH.md](LEARNING_PATH.md) for the sequential curriculum.
3. Use [TASK_INDEX.md](TASK_INDEX.md) for task-oriented lookup.
4. Check [MODEL_AND_NOTEBOOK_INDEX.md](MODEL_AND_NOTEBOOK_INDEX.md) before
   interpreting model or notebook roles.

## Shared foundation

`shared/` contains deliberately small helpers for:

- validated `.env` configuration and `DefaultAzureCredential` project context;
- meaningful demo-derived names and redacted customer-readable logs;
- bounded transient retries with explicit terminal failures and timeouts;
- content-addressed file upload/reuse and fine-tuning job reuse/monitoring;
- JSONL validation, secret checks, duplicate detection, hashing, and split isolation;
- public experiment manifests and identifier-bearing runtime manifests confined to
  ignored `outputs/` directories;
- catalog/demo/manifest schemas and `.env`, metadata, manifest, and notebook templates.

The helpers return native SDK objects where notebook authors need full service
detail. They do not silently replace errors, retry terminal conditions, choose a
region, or create resources through an undisclosed mechanism.

## Environment and metadata contracts

- `pyproject.toml` is the readable direct-dependency definition.
- `requirements.lock` pins the direct dependencies used by the customer
  notebooks and helpers. It is one central environment, not a fully hashed transitive lock.
- `catalog.yml` is the root metadata source of truth.
- `shared/schemas/` contains machine-readable schemas.
- `shared/templates/` contains identifier-free starting points for track authors.

Never commit a populated `.env`, specific resource or project endpoint, region,
credential, tenant/subscription identifier, job URL, or runtime manifest.
