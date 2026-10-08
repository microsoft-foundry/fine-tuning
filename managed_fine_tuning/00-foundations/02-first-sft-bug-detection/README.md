# First SFT: Code Bug Detection

Fine-tune a model to learn the checked-in code bug detection examples.

The canonical notebook is a sequential, training-only Microsoft Foundry workflow. It validates and uploads the exact checked-in files, submits one supervised fine-tuning job, monitors it to a terminal state, downloads the service result CSVs, and reports validation loss/token accuracy.

## Training recipe

| Setting | Value |
|---|---|
| Project | Your `FOUNDRY_PROJECT_ENDPOINT` from `.env` or the process environment |
| Reference region | West US 3; use a project supporting this model and training type |
| Model | OpenAI-OSS gpt-oss-120b, version `1` |
| Training type | `GlobalStandard` |
| Recipe | 2 epoch(s), learning-rate multiplier 0.8 |
| Training split | 224 rows |
| Validation split | 20 rows |

A prior demonstration run reported validation loss 1.62558 -> 1.33327. It is context only; this notebook submits and measures a fresh job.

The source demo also tracked a 10-row held-out scale. Those rows are not copied, uploaded, or used here.

## Run

1. Install the central environment by following [`GETTING_STARTED.md`](../../GETTING_STARTED.md).
2. Copy this module's `.env.template` to a module-local `.env`, set your own `FOUNDRY_PROJECT_ENDPOINT`, and sign in with a credential supported by `DefaultAzureCredential`.
3. Open `notebooks/demo.ipynb` from this cookbook or its `notebooks` directory. Preparation cells validate data locally; subsequent cells contact Azure.
4. Run preparation first, then execute the client, upload, training, and monitoring cells. Training can incur charges.

Runtime job IDs, uploaded file IDs, result CSVs, and summaries are written only under ignored `outputs/`. The notebook stops after service training metrics and makes no downstream quality claim.

Use [`ADAPT_YOUR_DATA.md`](../../ADAPT_YOUR_DATA.md) to validate a separately
versioned dataset without changing this canonical notebook. This demo always
uploads the preserved 224-row training and 20-row validation files and enforces
their fixed row counts, byte counts, and hashes.

Saved state binds the job to its project, model, recipe, and data hashes. Rerunning resumes it; a mismatched state raises an error rather than silently starting another paid job. Archive the state deliberately to start another run. Job-creation POSTs are submitted once with SDK retries disabled. Reconcile ambiguous job-creation failures in Foundry before retrying. Upload and job polling have explicit timeouts; failed or cancelled jobs raise errors.

## Dataset integrity

See [`data/README.md`](data/README.md) for the exact source mapping, row counts, byte counts, and SHA-256 values enforced by the notebook.

After training, use [`POST_TRAINING.md`](../../POST_TRAINING.md) to identify the
`fine_tuned_model` artifact, evaluate held-out data, provision serving
separately, or explicitly cancel and clean up resources.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
