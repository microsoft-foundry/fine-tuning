# Domain SFT: Scientific Summarization

Fine-tune a model on the checked-in PubMed-derived summarization examples.

The canonical notebook is a sequential, training-only Microsoft Foundry workflow. It validates and uploads the exact checked-in files, submits one supervised fine-tuning job, monitors it to a terminal state, downloads the service result CSVs, and reports validation loss/token accuracy.

## Training recipe

| Setting | Value |
|---|---|
| Project | Your `FOUNDRY_PROJECT_ENDPOINT` from `.env` or the process environment |
| Reference region | Australia East; use a project supporting this model and training type |
| Model | Alibaba qwen3.8-27b; expected reference version `1`, actual version unverified |
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

### Version evidence is not a version pin

The request selects the model name, not `EXPECTED_MODEL_VERSION`. Version `1`
is an expected reference only. The notebook assumes no undocumented response
field and never treats request metadata as service proof. State and summaries
record `expected_model_version: "1"`, `actual_model_version: null`, and
`model_version_verification: "unverified"`. Resuming the saved job does not
establish pinned-version equivalence. Workflows requiring verified pinning must
establish that through documented service metadata separately.

### Submission recovery

The shared SFT helper atomically flushes pending intent before each upload/job
POST, records every returned file/job ID before polling or another create, and
uses `max_retries=0` for all creates. Known IDs are reused after read failures;
saved files are byte-checked before job creation, and a retrieved job must expose
the matching model, file IDs, and supervised recipe. Upload and job polling have
explicit timeouts. Failed/cancelled resources retain their identifiers.

An unknown outcome leaves `pending_operation` as `training_upload`,
`validation_upload`, or `job_create`. The next run refuses further creates even
if no resource is visible in Foundry yet. Retain state and reconcile accepted
resources against the exact project/recipe/data; record a confirmed accepted ID
in its matching field before resolving that pending marker. Never clear or
archive ambiguous state merely to retry. Corrupt, older-format, or mismatched
state is rejected without overwrite. Remove a stale lock only after confirming
no active writer and reconciling state. Deliberate new runs require reconciled
old resources before archiving the journal.

State preserves top-level `job_id`, `training_file_id`, `validation_file_id`,
and `equivalence.project_endpoint` for
[`POST_TRAINING.md`](../../POST_TRAINING.md). A partial upload may have only one
known file; restrict cleanup to identified, confirmed-unshared files instead of
assuming the two-file example is complete. Pending unknown IDs require
reconciliation first. Closing clients or timing out monitoring does not cancel
training or delete files.

## Dataset integrity

See [`data/README.md`](data/README.md) for the exact source mapping, row counts, byte counts, and SHA-256 values enforced by the notebook.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
