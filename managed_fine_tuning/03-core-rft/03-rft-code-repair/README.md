# RFT Code Repair: MAI-Code-1.1-Flash (preview)

**Role:** TRAINING

**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

Train `MAI-Code-1.1-Flash-2026-09-15` to produce minimal, regression-safe Python code
repairs with managed reinforcement fine-tuning. This preview model supports
**RFT only**; it does not support SFT, DPO, or interactive training.

The tutorial uses [50 training and 15 validation examples](data/README.md)
and [score-model grader](graders/code_repair_quality.json).
The grader uses `o3-mini-2025-01-31`, rewards correctness, requirements,
minimality, and a usable patch, and returns a score from 0 to 1 with a pass
threshold of 0.7.

## Setup and execution

Follow the central [setup guide](../../GETTING_STARTED.md); there are no
per-demo dependencies. Select the **Foundry managed** kernel and run from this
demo directory or its `notebooks/` directory.

Copy `.env.template` to a demo-local `.env`. Set your project endpoint and
verified training region, and authenticate through a credential provider
supported by `DefaultAzureCredential`. Before remote execution, confirm:

- Preview and RFT access for the exact `MAI-Code-1.1-Flash-2026-09-15` identifier in your
  project and region, with sufficient training quota and permissions.
- A supported training type: MAI supports Global and Developer, not Standard or
  Data zone. These types do not guarantee that training stays in the project's
  region; review the [supported models and training types](https://learn.microsoft.com/azure/foundry/fine-tuning/overview#supported-models)
  and confirm the effective service defaults before uploading private data.
- Support for the `o3-mini-2025-01-31` grader and its sampling settings.
  [Grader hosting requirements](https://learn.microsoft.com/azure/foundry/openai/how-to/reinforcement-fine-tuning#model-grader-deployments)
  depend on the selected training model. Provision any required grader deployment
  separately; this notebook does not create one or change shared capacity.
- Data and model terms appropriate for your use case, privacy requirements for
  any custom data, and training and grading costs.

Sections 1-4 are offline: they verify exact file hashes, row schemas, duplicate
and prompt isolation, grader bindings, and the recipe. Sections 5 onward load
private configuration, connect, upload/reuse files, create/recover a job,
monitor, and download service metrics. **Do not use Run All as a smoke test.**
Executing the upload or submission cells performs live operations and can
incur charges; there is no hidden training-enable switch.

The recipe explicitly selects Global training (`trainingType=globalStandard`);
omitting it defaults to Standard, which this model rejects. Optional epochs,
batch size, reasoning effort, and evaluation settings use model-specific service
defaults rather than another model's configuration. Review those defaults and current
[pricing](https://aka.ms/oai/pricing) before submission. The grader's seed 42
is not a training seed or a guarantee of deterministic scoring.
The ten-minute file-processing and six-hour monitoring timeouts stop local
waiting, not remote work, and are not dollar caps.

## Recovery and private artifacts

Ignored `outputs/submission-state.json` records the project, region, recipe
fingerprint (including training type), file IDs, and job ID. Each uploaded file
ID is saved before waiting for processing. Reruns verify the saved input bytes
and recipe, and reuse a
recorded pending, validating, queued, running, or succeeded job. A different
project, region, recipe, or failed/cancelled job is rejected rather than
silently replaced.

An ambiguous upload leaves a `<split>_upload_state` marker. Reconcile the
content-addressed file, record a confirmed ID under `operation_ids.training`
or `operation_ids.validation`, and rerun upload verification. Clear a marker
only when reconciliation proves no file was created. Submission requires both
files' verified SHA-256 evidence, not merely the presence of file IDs.

If the project's file quota is full, do not delete shared inputs or results to
make room. An existing processed fine-tune file can be adopted only after
downloading it and verifying its full SHA-256 against the preserved split.
Record its confirmed ID and reused ownership in private state; rerunning the
upload cell verifies the bytes again before submission.

Upload and job-creation POSTs are not automatically retried. Before submission,
the notebook saves an in-flight marker. If the response is lost, reconcile the
job in Foundry and set `FOUNDRY_EXISTING_CODE_REPAIR_RFT_JOB_ID` to the confirmed
ID; the notebook verifies its recipe before adopting it. If reconciliation
proves no job exists, deliberately clear the `submission_state` marker in the
private state before an intentional retry. Do not delete state to bypass an
unknown submission outcome.

`outputs/experiment-manifest.json` records input and grader hashes and the
actual service training metrics. Result CSVs and service-produced artifact IDs
remain under `outputs/`; notebook output shows only redacted references.
Keep these files private and retain them for recovery and cleanup.

## What the results do and do not establish

The grader is an LLM code-review judgment. It does **not** apply the patch,
compile code, or execute the stated tests. Offline checks verify that the data
and grader contracts are wired correctly, not that the grader assigns trustworthy scores.
Calibrate separately on correct, incorrect, malformed, and reward-hacking
responses before relying on the reward.

The 15 validation rows participate in training-time evaluation and are not an
untouched final test set. This notebook stops at service reward/loss metrics:
it does not deploy a model, call an inference endpoint, or claim improved
code-repair quality. Before production, evaluate base and fine-tuned artifacts
on an independent test set with isolated executable tests and human patch
review.

Follow [post-training guidance](../../POST_TRAINING.md) for serving and
separately confirmed cancellation/deletion. Closing a local client or hitting
a timeout does not cancel training. Uploaded files can be reused by other
jobs; remove only verified owned resources, and reconcile any separately
provisioned grader deployment with its owner.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
