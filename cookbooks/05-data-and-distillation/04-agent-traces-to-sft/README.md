# Agent Traces to SFT

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) uses the exact transformed
customer trace data from `Demos/TracesDistillation`, now retained under `data/`
rather than depending on ignored source-demo outputs.

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `traces-clean.jsonl` | 100 | `81083aadd6464293b6acfc9356216c6c685a1ba99ae3f6fe34ba9d7b5b864d71` |
| `train.jsonl` | 80 | `1371b81c70d7f8c20f8eff604932670a3eda8ca9437759264dcfce4c06aeeae3` |
| `val.jsonl` | 10 | `865ab59e51c81a29ecc89a0bde347a22d0929630f7cd3db1ffb12f1d406034fa` |
| `test.jsonl` | 10 | `7ea4284e67980cc2db8cfb2a4b5d403da4383bec9821be3fea27316cbd677040` |

The original seed-42 row-level split is exactly 80/10/10. Hashes, chat/tool
references, fixture agreement and split membership are checked read-only;
the notebook never repairs, filters or rewrites source data. The source
transformation recipe is documented in [data provenance](data/README.md).
The 10 test rows are never uploaded or used after training.

The source student example is `gpt-4.1-nano-2025-04-14`. The retained supervised
recipe is 3 epochs, learning-rate multiplier 1.0, batch size 1 and
`GlobalStandard`. No agent deployment, agent calls or trace generation is
needed to run the bundled training contract.

## Run

1. Install `cookbooks/requirements.lock` and authenticate with `az login` or another `DefaultAzureCredential` source.
2. Copy `.env.template` to this module's ignored `.env` and configure your own `FOUNDRY_PROJECT_ENDPOINT` and available `FOUNDRY_MODEL`. Paid jobs default to false.
3. Open the notebook from this module or its `notebooks` directory. Run local validation, then execute the upload/training cells, which can incur charges.

The notebook includes the entire submission workflow and requires no runner
or project resource ID. It submits or resumes one job, fails explicitly on
terminal service errors/timeouts, and downloads service loss/token-accuracy
metrics only. No fine-tuned inference or base-model comparison is performed.
Creation POSTs disable SDK retries; an ambiguous response requires reconciliation
instead of blind resubmission. Recorded pending jobs are monitored without
creating another job, and authentication/service errors propagate.
Ignored `outputs/<run-id>` stores local state and metrics; a blank
`FOUNDRY_RUN_ID` resumes `outputs/default`. See [track configuration](../README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
