# Teacher-Student Sarcasm

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) preserves the exact
`Demos/DistillingSarcasm` data and training recipe: shuffle 500 source questions
with seed 42, select 50, use the first 40 as teacher candidates and hold out the
last 10. The 26 accepted labels remain byte-for-byte unchanged as 20 training and
6 validation rows. Generation and judging are not repeated. Read-only guards
check hashes, chat shape, unique candidate membership, held-out separation and
the canonical Clippy system prompt.

The source student example is OpenAI-OSS `gpt-oss-120b`, version `1`, with
`GlobalStandard` and service-default supervised hyperparameters. Set an exact
fine-tuning model identifier available in your own project; an optional expected
version checks returned metadata but does not pin an undated alias.

## Run

1. Install the central environment by following [`GETTING_STARTED.md`](../../GETTING_STARTED.md), then authenticate with `az login` or another `DefaultAzureCredential` source.
2. Copy `.env.template` to this module's ignored `.env` and fill `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL`.
3. Open the notebook from this module or its `notebooks` directory. Run local validation, then execute the upload/training cells, which can incur charges.

The notebook contains the submission workflow; there is no separate runner.
Creation POSTs disable SDK retries; an ambiguous response requires reconciliation
instead of blind resubmission. Recorded pending jobs are monitored without
creating another job, and authentication/service errors propagate.
It uploads only the exact training/validation files, monitors with bounded
timeouts, and downloads fresh service loss/token-accuracy metrics. The 10
held-out questions are never uploaded or used after training. No deployment,
fine-tuned inference or base-model comparison is performed.

Ignored `outputs/<run-id>` stores submission state, lineage, result CSVs,
`training-summary.json` and a loss plot. A blank `FOUNDRY_RUN_ID` resumes
`outputs/default`; use a new run ID only for an intentional new paid job.
See [data provenance](data/README.md) and [track configuration](../README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.

**Next:** [Code Distillation](../02-code-distillation/README.md).
