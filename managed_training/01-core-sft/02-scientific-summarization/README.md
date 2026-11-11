# Domain SFT: Scientific Summarization

Fine-tune a model on the checked-in PubMed-derived summarization examples.

The canonical notebook is a sequential, training-only Microsoft Foundry workflow. It validates and uploads the exact checked-in files, submits one supervised fine-tuning job, monitors it to a terminal state, downloads the service result CSVs, and reports validation loss/token accuracy.

## Training recipe

| Setting | Value |
|---|---|
| Project | Your `FOUNDRY_PROJECT_ENDPOINT` from `.env` or the process environment |
| Reference region | Australia East; use a project supporting this model and training type |
| Model | Alibaba qwen3.8-27b, version `1` |
| Training type | `GlobalStandard` |
| Recipe | 3 epoch(s), batch size 1, learning-rate multiplier 1.0 |
| Training split | 1,000 rows |
| Validation split | 100 rows |

A prior demonstration run reported validation loss 1.168630 -> 1.117555. It is context only; this notebook submits and measures a fresh job.

## Run

1. Install the central environment by following [`GETTING_STARTED.md`](../../GETTING_STARTED.md).
2. Copy this module's `.env.template` to a module-local `.env`, set your own `FOUNDRY_PROJECT_ENDPOINT`, and sign in with a credential supported by `DefaultAzureCredential`.
3. Open `notebooks/demo.ipynb` from this cookbook or its `notebooks` directory. Preparation cells validate data locally; subsequent cells contact Azure.
4. Run preparation first, then execute the client, upload, training, and monitoring cells. Training can incur charges.

Runtime job IDs, uploaded file IDs, result CSVs, and summaries are written only under ignored `outputs/`. The notebook stops after service training metrics and makes no downstream quality claim.

Saved state binds the job to its project, model, recipe, and data hashes. Rerunning resumes it; a mismatched state raises an error rather than silently starting another paid job. Archive the state deliberately to start another run. Job-creation POSTs are submitted once with SDK retries disabled. Reconcile ambiguous job-creation failures in Foundry before retrying. Upload and job polling have explicit timeouts; failed or cancelled jobs raise errors.

## Dataset integrity

See [`data/README.md`](data/README.md) for the exact source mapping, row counts, byte counts, and SHA-256 values enforced by the notebook.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
