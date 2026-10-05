# Synthetic Tool-Use SFT

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) preserves the exact source split
from `Demos/SyntheticDatagen-ToolUse`: 24 training, 3 validation and 3 unused
test rows. The original recipe converts the unchanged 15-tool catalog to
OpenAPI, requests three batches with `max_samples=60`, deduplicates complete
rows and splits shuffled indices 80/10/10 with seed 42. Returned samples can be
fewer than requested. The exact generated data are included; generation is not
repeated and requires no teacher or generation-project settings.

The source student example is `gpt-4.1-mini-2025-04-14`. The fixed supervised
recipe is 3 epochs, learning-rate multiplier 1.0 and `GlobalStandard`.
Read-only guards verify hashes, row counts, split separation, chat shape and
tool-call references. Final assistant tool calls are valid prediction targets;
no test rows are uploaded.

## Run

1. Install `cookbooks/requirements.lock` and authenticate with `az login` or another `DefaultAzureCredential` source.
2. Copy `.env.template` to this module's ignored `.env` and configure your own `FOUNDRY_PROJECT_ENDPOINT` and available `FOUNDRY_MODEL`. Paid jobs default to false.
3. Open the notebook from this module or its `notebooks` directory. Run local validation, then execute the upload/training cells, which can incur charges.

The inline workflow records uploaded IDs before processing waits, submits or
resumes one job, monitors with bounded timeouts and downloads fresh service
loss/token-accuracy metrics. No deployment, fine-tuned inference or base-model
comparison is performed. Creation POSTs disable SDK retries; an ambiguous
response requires reconciliation instead of blind resubmission. Recorded
pending jobs are monitored without another creation attempt, and authentication
or service errors propagate.
Runtime evidence stays under ignored `outputs/<run-id>`;
a blank `FOUNDRY_RUN_ID` resumes `outputs/default`.

See [data provenance](data/README.md) and [track configuration](../README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.

**Next:** [Agent Traces to SFT](../04-agent-traces-to-sft/README.md).
