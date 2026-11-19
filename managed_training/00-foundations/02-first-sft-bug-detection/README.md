# First SFT: Code Bug Detection

Fine-tune a model to learn the checked-in code bug detection examples.

The canonical notebook is a sequential, training-only Microsoft Foundry workflow. It validates and uploads the exact checked-in files, submits one supervised fine-tuning job, monitors it to a terminal state, downloads the service result CSVs, and reports validation loss/token accuracy.

## Training recipe

| Setting | Value |
|---|---|
| Project | Your `FOUNDRY_PROJECT_ENDPOINT` from `.env` or the process environment |
| Reference region | West US 3; use a project supporting this model and training type |
| Model | OpenAI-OSS gpt-oss-120b; expected reference version `1`, actual version unverified |
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

### Version evidence is not a version pin

The OpenAI-compatible create request selects the model name; it does not send
`EXPECTED_MODEL_VERSION` as a version selector. The notebook does not assume an
undocumented response field or treat request metadata as service evidence.
Saved state and summaries distinguish `expected_model_version: "1"`,
`actual_model_version: null`, and `model_version_verification: "unverified"`.
Resuming a recorded job is not a claim of pinned-version equivalence. If your
workflow requires verified version pinning, establish it through documented
service metadata before relying on that claim; this notebook does not provide it.

### Submission recovery

The shared SFT helper binds state to the requested project, model name, expected
reference version, recipe, suffix, and canonical data hashes. It atomically
flushes pending intent **before each upload and job POST**, and records each
returned ID **before polling or another create**. All three POSTs use
`max_retries=0`. A read timeout resumes from known IDs rather than uploading
again. Saved files are checked against their canonical bytes before job creation;
a retrieved job must expose the matching model, file IDs, and supervised recipe.
Upload and job polling have explicit timeouts; terminal failures retain IDs.

An unknown acceptance outcome leaves `pending_operation` set to
`training_upload`, `validation_upload`, or `job_create`. Rerunning fails closed:
it never infers rejection from an empty list response and never repeats that
POST. Retain the journal and reconcile the exact project, accepted resources,
recipe, and file contents in Foundry. A known accepted ID must be recorded in its
matching top-level ID field before the operator resolves the pending marker;
do not clear it merely to retry. Corrupt, older-format, or mismatched state is
retained and rejected, not silently migrated or overwritten. A stale lock also
requires confirming that no writer is active and reconciling state before removal.
Archive state only after reconciliation when deliberately starting another run.

Top-level `job_id`, `training_file_id`, `validation_file_id`, and
`equivalence.project_endpoint` keep the cancellation/cleanup guide's state shape.
Partial uploads can leave just one known file ID; the guide's two-file deletion
example then needs to be restricted to that identified, confirmed-unshared file.
A pending POST without an ID must be reconciled first. Monitoring timeouts and
closing local clients do not cancel remote training or delete uploaded files.

## Dataset integrity

See [`data/README.md`](data/README.md) for the exact source mapping, row counts, byte counts, and SHA-256 values enforced by the notebook.

After training, use [`POST_TRAINING.md`](../../POST_TRAINING.md) to identify the
`fine_tuned_model` artifact, evaluate held-out data, provision serving
separately, or explicitly cancel and clean up resources.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
