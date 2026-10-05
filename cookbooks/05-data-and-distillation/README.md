# Data and Distillation

This track contains four canonical customer notebooks:

| Notebook | Preserved training contract |
|---|---|
| [Teacher-Student Sarcasm](01-teacher-student-basics/notebooks/demo.ipynb) | 20 training / 6 validation labels; 10 source questions held out |
| [Code Distillation](02-code-distillation/notebooks/demo.ipynb) | 1,576 training / 83 validation examples |
| [Synthetic Tool Use](03-synthetic-tool-use/notebooks/demo.ipynb) | 24 training / 3 validation / 3 unused test examples |
| [Agent Traces to SFT](04-agent-traces-to-sft/notebooks/demo.ipynb) | 100 transformed traces; 80 training / 10 validation / 10 unused test examples |

Install `cookbooks/requirements.lock` and run from a module directory or its
`notebooks` directory. Copy that module's `.env.template` to its module-local,
ignored `.env`,
configure your own `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL`, and authenticate
with Azure CLI or another `DefaultAzureCredential` source. No API keys,
subscription IDs or project resource IDs are required.

The notebooks include their training workflows directly. They reuse the
customer-facing `shared.config` and `shared.auth` helpers and this track's
`stage05_runtime.py` for bounded processing/monitoring waits; no local validation
runner is required. Exact source data and generated customer datasets are
retained under `data/`, with hashes and preparation recipes. Generation is not
repeated and therefore requires no generation project, teacher or agent settings.
Git text conversion is disabled for the bundled JSONL files so byte-pinned
dataset hashes remain identical across platforms.

All template values start blank. Local data checks work without cloud access.
Executing the upload/training cells performs hosted calls and can incur
charges; available fine-tuning models depend on your project
and region. The source hyperparameters and `GlobalStandard` recipe are retained.
An optional expected model version checks service metadata when available; it
does not pin an undated model alias.

Runtime evidence is written only to ignored `outputs/<run-id>`. A blank
`FOUNDRY_RUN_ID` uses `default` and resumes its recorded job. Use a new run ID for
an intentional new paid job. State rejects changed project/model/data settings
and ambiguous interrupted submissions instead of silently creating duplicates.
Creation POSTs explicitly disable SDK retries (`max_retries=0`), matching the
shared helper's duplicate-avoidance policy. A recorded job is monitored through
pending/queued/running states without another creation attempt. The inline
submission retains the recipe and explicit `GlobalStandard` training type;
authentication and other service errors propagate without fallback.
Blank polling settings use file/job intervals of 5/60 seconds and file/job
timeouts of 900/86,400 seconds.

Results are limited to terminal training status, trained tokens and service
loss/token-accuracy curves. There is no deployment, fine-tuned inference or
base-model comparison. Missing loss observations and malformed metrics are
explicit errors, not fabricated conclusions.

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
