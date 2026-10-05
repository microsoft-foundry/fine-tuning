# Code Distillation: Natural Language to Python

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) trains on the exact source files
from `Demos/NL_to_Python_Distillation`: 1,576 training and 83 validation examples.
Read-only guards verify byte hashes, row counts, chat shape and split overlap.
The workflow never generates, filters, judges or replaces data.

| Source training setting | Value |
|---|---|
| Example model | Alibaba `qwen3.8-27b`, expected version `1` |
| Training type | `GlobalStandard` |
| Method | Supervised fine-tuning |
| Epochs | 1 |
| Batch size | 1 |
| Learning-rate multiplier | 1.3 |

## Run

1. Install `cookbooks/requirements.lock` and authenticate with `az login` or another `DefaultAzureCredential` source.
2. Copy `.env.template` to this module's ignored `.env` and configure your own `FOUNDRY_PROJECT_ENDPOINT` and available `FOUNDRY_MODEL`. Paid jobs default to false.
3. Open the notebook from this module or its `notebooks` directory. Run local validation, then execute the upload/training cells, which can incur charges.

The notebook submits or resumes one job, monitors with bounded timeouts,
disables SDK retries on creation POSTs, and requires reconciliation of ambiguous
submission responses rather than blind resubmission. Recorded pending jobs are
monitored without creating another job; authentication/service errors propagate.
It downloads result CSVs and plots service loss columns independently. Token
accuracy is included only when emitted. No deployment, fine-tuned inference or
base-model comparison is performed.

Ignored `outputs/<run-id>` contains state and fresh metrics. A blank
`FOUNDRY_RUN_ID` resumes `outputs/default`; use a new ID for an intentional new
paid job. The optional expected version checks returned metadata but does not
pin an undated model alias. See [data provenance](data/README.md) and
[track configuration](../README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
