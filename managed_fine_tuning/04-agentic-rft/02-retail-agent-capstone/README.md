# Agentic Retail Capstone

The canonical lesson is [`notebooks/demo.ipynb`](notebooks/demo.ipynb): validate
the original Zava inputs, train SFT and RFT models, inspect service metrics,
then provision deployments through your approved process and optionally
verify, invoke, and compare base versus fine-tuned models.

The configured recipes are SFT `gpt-4.1-mini-2025-04-14` (386/103 rows, batch
size 1, learning-rate multiplier 2, three epochs) and RFT
`o4-mini-2025-04-16` (10/10 rows, `o3-mini` policy grader, remote
tools, one epoch).
`data/manifest.json` verifies their LF-canonical byte counts, SHA-256 hashes,
and row counts. CRLF is normalized only for lineage checks; files and upload
payloads are not rewritten.

Install the repository's shared dependencies as described in
[Getting started](../../GETTING_STARTED.md), copy `.env.template` to `.env`,
and run the notebook from this directory or `notebooks`. The `.env` is local
to this cookbook, not the repository or shared-helper directory. Standard
`python-dotenv` loading preserves environment variables already set by you.
The preparation cells validate local inputs without service calls.
Executing subsequent cells authenticates, uploads, trains, verifies deployments,
and runs inference comparisons. There is no local runner dependency.
The notebook uses the repository's `shared` configuration, retry, and
polling helpers.

**Training and inference can incur charges.** Execute only the cells for the
stages you intend to run; no additional execution switches are required.
Deployment checks are read-only and do not create or replace deployments.
Select customer-owned Foundry projects with access, quota, and RFT remote-tool
preview support. Training uses `DefaultAzureCredential`,
`AIProjectClient.get_openai_client()`, and GlobalStandard training. The notebook
does not discover alternate projects or silently substitute models.

RFT training retains the original mutating tools: point
`FOUNDRY_TOOLS_SERVER_URL` at an approved HTTPS server backed by an **isolated,
disposable copy** of the retail database, then set
`FOUNDRY_TOOL_SERVER_IS_SANDBOX=true`. Never use a production retail backend.
Comparison excludes validation scenario 7 and enforces a read-only tool
allowlist before every remote request, including model-generated calls.
Unknown and mutating calls fail before contacting the server.

After successful training, use the saved `fine_tuned_model` artifacts in
`outputs/notebook-state.json` to provision GlobalStandard deployments through
your approved Foundry portal or management workflow, with explicit unique
names and appropriate capacity. Provision SFT base/fine-tuned models in the
SFT account and RFT base/fine-tuned models in the RFT account. Supply your
subscription, resource group, account names, and four distinct deployment
names in `.env`. Existing deployments must match model, version, format, SKU,
and provisioning state; collisions and missing resources fail explicitly.
The notebook performs **read-only discovery**, never creation, updates, or
deletion. It makes no claim about ARM conditional-create support.

To complete comparison with preprovisioned deployments, preserve the
successful training state and its configuration. Run setup, local validation,
client initialization, the saved-state loading cell, deployment readiness
checks, and comparison cells; skip the upload/training and metric-download
cells if the jobs have already completed. Deployment readiness is verified
before inference. No creation-verification flag, UUID suffix, or capacity
setting is required. Execute the deployment readiness cell alone to check
readiness without inference.
Comparison also requires an existing `o3-mini` grader deployment in your
configured grader project. The notebook does not create the grader.

Runtime state, service CSVs, and customer comparison results are created only
when the corresponding cells execute under ignored `outputs/`. Preserve the state file to resume;
it is bound to endpoints, input hashes, and methods. An uncertain submission
stops for explicit reconciliation rather than silently submitting twice.
Retail submission uses the visible SDK call to preserve the required
`trainingType=globalStandard` extension, which the shared job helper does not
accept. SDK write retries are disabled, and the pending marker is saved before
the POST; no failed creation is automatically resubmitted.
SFT reports single-call original-grader reward alongside exact and
tool-name accuracy over all 280 held-out rows. RFT reports policy reward,
pass count, and exact tool sequence over the same nine read-only scenarios
for both models. No representative result is promised.

SFT and RFT comparison checkpoints bind reuse to the project endpoints,
deployment names, input and grader hashes, selected rows, and generation
settings. Each completed base or fine-tuned side is saved immediately. A
failure within one side repeats only that incomplete side; a changed comparison
context invalidates the saved results instead of presenting them as current.
If an RFT response attempts an unknown or mutating tool, no call in that batch
is executed and the scenario receives a deterministic zero without invoking
the policy grader. Attempted, executed, and blocked tool names are reported
separately.

The checked-in next-tool grader's multi-call matching
can double-credit an expected call, so the comparison validates that each of
the 280 rows expects exactly one next call and awards one row score
only when the model predicts exactly one call. Missing or extra calls receive
zero reward and zero exact/name accuracy; valid single-call predictions use
the grader's normalization and partial-credit rules. Reported
`mean_single_call_grader_reward` is not exact accuracy.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
