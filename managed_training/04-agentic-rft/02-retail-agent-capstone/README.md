# Agentic Retail Capstone

The canonical lesson is [`notebooks/demo.ipynb`](notebooks/demo.ipynb): validate
the preserved Zava provenance, prepare validated active copies, check the tool
contract, train SFT and RFT models, inspect service metrics,
then provision deployments through your approved process and optionally
verify, invoke, and compare base versus fine-tuned models.

The configured recipes are SFT `gpt-4.1-mini-2025-04-14` (386/103 rows, batch
size 1, learning-rate multiplier 2, three epochs) and RFT
`o4-mini-2025-04-16` (10/10 rows, `o3-mini` policy grader, remote
tools, one epoch).
[data/manifest.json](data/manifest.json) verifies LF-canonical byte counts,
SHA-256 hashes, and row counts of the **immutable committed sources**, not an
approval to upload their raw rows. It separately records original provenance
for the extended API contract; the original 15 operations are unchanged.
The source manifest and all seven committed datasets remain unchanged.

## Offline preparation and immutable provenance

**Never upload or evaluate the committed raw JSONL files directly.** Notebook
cell 5 calls the [offline preparation helper](src/retail_curation.py) before any
service stage. It verifies the original manifest, audits every row of all seven
sources, applies the nine reviewed record-level corrections, and builds only
five active SFT/RFT training, validation, and held-out next-tool copies under
ignored outputs/prepared-data. The two synthetic source splits are evidence
only: no prepared active artifact is created for either one. The table below
records the reviewed decisions; preparation also records them in its generated
manifest.

| Active split | Reviewed source rows | Prepared-copy correction |
| --- | --- | --- |
| SFT training | 16, 111 | End before unsupported payment actions with grounded refusals. |
| SFT training | 200 | Move the existing credentials and successful lookup before product access. |
| SFT validation | 54 | Apply the same authentication-first correction. |
| Next-tool evaluation | 10, 77, 99, 180 | Remove only each reviewed failed payment mutation and its matched error reply; retain the real profile lookup/reply and unchanged one-call target. |
| Next-tool evaluation | 162 | Move the existing authentication block, preserving the one-call target. |

No rows are dropped. The five prepared split counts remain 386, 103, 280, 10,
and 10 in SFT-train, SFT-validation, next-tool, RFT-train, and RFT-validation
order. Both RFT splits retain every source row unchanged except LF normalization.

Preparation computes its own LF-encoded output manifest alongside those
copies. It records actual canonical and physical source/output SHA-256 hashes,
byte/row counts, exact original changed rows and their line hashes, curation
decisions, and **all raw-source findings**, including unsafe synthetic rows.
Source findings are reported, not relabeled as passing or used as exceptions
to the prepared-active gate. No computed digest is guessed or committed.
Unchanged active records retain their original serialization except physical
CRLF-to-LF normalization; curated records alone are reserialized. Originals
retain their exact physical bytes, including their original line endings.

Every prepared active row must pass the authentication/payment audit before
any prepared output is published. Repeating preparation verifies and reuses
identical files without rewriting them; interrupted identical preparation can
finish missing artifacts, with the manifest written last. Unreviewed unsafe
active rows, source drift, blank/malformed records, path traversal, symbolic
links/junctions, hard-linked outputs, unexpected files, or conflicting existing
artifacts fail explicitly without overwriting inputs or existing outputs.
Reconcile stale local preparation artifacts explicitly, then rerun cell 5;
do not change source provenance or clear uncertain paid-submission state to
bypass a failure. There is no automatic adoption or source-rewrite step.

The [local trace audit](src/retail_tools.py) requires a matched, successful
email or name-plus-zip lookup reply before product, catalog, profile, or order
calls. A bearer token, claimed user ID, assistant assertion, tool declaration,
pending lookup, or RFT reference sequence is not observed customer
authentication. Action payment IDs must have appeared in earlier profile or
owned-order results for that same customer, and the action's order must be
owned. This is a bounded trace check, not a complete retail-policy engine.

The curator reuses existing authentication messages and results in the three
reviewed product-before-authentication records. For the two reviewed SFT
invented-card records, it supervises a refusal before the unsupported action,
not a fabricated payment method or result. The checked-out Ethan Kim evidence shows **one $37 gift
card**, not two, against a **$329.40** difference and no stored credit card.
The four reviewed evaluation histories contain unsupported payment calls that
failed before a real profile was retrieved. Removing those call/error pairs
leaves the actual profile retrieval as the next action; its observed customer,
order, method, and original evaluation target must still match the reviewed
evidence. A later profile never retroactively authorizes a historical mutation,
and no lookup or payment reply is invented or replaced.
The original synthetic splits remain source evidence and are included in the
seven-dataset audit, never silently promoted to approved training inputs.

Uploads, preflight scenario selection, and SFT/RFT comparisons use **only** the
explicit prepared path mapping. Before use, the notebook rechecks the persisted
manifest, immutable-source binding, output hashes/counts, and active trace
diagnostics. Prepared hashes are checked before record splitting or JSON
parsing, so corrupted bytes fail as a prepared-artifact integrity error before
they can be consumed. It uploads the same in-memory bytes whose hash was
checked, not a later reopening of a raw path. Training state and comparison
checkpoints bind to actual prepared hashes and the generated curation manifest; old raw-input
state is not silently reusable. A final SFT tool target still needs no invented
reply, and each wrapped evaluation target remains exactly one predicted call.
RFT reference names establish ordering only, not tool execution or proof of
historical policy outcomes.

The [trace regressions](tests/test_retail_trace_integrity.py) assert raw-source
immutability, actual curation decisions, full raw-finding retention, zero
prepared-active findings, independent authentication/payment oracles,
LF/hash/count fidelity, idempotence, containment, no-overwrite behavior, and
tamper detection. The [capstone tests](tests/test_retail_capstone.py) also mock
the actual extracted notebook preparation, byte-check, and upload helpers.
Their mock clients capture the submitted streams, check exact curated bytes
and saved hashes for all four training/validation uploads, and require no new
POST or state change after any prepared split or manifest is tampered with.
A separate snapshot check changes the path after validation and requires the
mock upload to consume only the previously checked immutable in-memory bytes.
Run both test modules to reproduce these checks; inspect their temporary
curation reports for all raw findings, exactly nine curated records, unchanged
source hashes/counts, and zero prepared-active findings from both independent
oracles. These are offline checks, not retail-server policy qualification or
evidence of successful managed training.

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

**No retail server is shipped with this capstone.** Supply an approved external
implementation of all 17 operations, including `list_user_orders` and
`policy_verify_return`, following the [external server contract](SERVER_CONTRACT.md).
The nearby catalog sample is not a retail backend. Preserved policy labels and
the synthetic database do not fully specify a deterministic return-policy
engine; the server owner must qualify that implementation without inventing
rules or changing the declared reason enum.

RFT training retains the original mutating tools: point
`FOUNDRY_TOOLS_SERVER_URL` at an approved HTTPS server backed by an **isolated,
disposable copy** of the retail database, then set
`FOUNDRY_TOOL_SERVER_IS_SANDBOX=true`. Never use a production retail backend.
Inject the same high-entropy `FOUNDRY_TOOL_AUTH_TOKEN` into the hosted server
and notebook environment or ignored local environment file. Every live remote
tool and local preflight/comparison POST uses `Authorization: Bearer ...`;
remote stages require the token. The original empty headers in the checked-in
configuration are templates, not the live method. Do not display live headers
or methods, enable HTTP/SDK wire-debug logging, or put credentials in URLs.
Offline preparation checks every configured and dataset-referenced operation
against the active API. Before uploads or RFT comparison, authenticated
read-only preflight checks the external dispatch envelope for the two required
RFT operations; this is not policy-engine qualification.
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
it is bound to endpoints, input hashes, redacted methods, and a credential
fingerprint. No live header values are serialized into state or a manifest.
Credential rotation and old state formats require explicit reconciliation,
not automatic reuse or clearing pending markers. An uncertain submission
stops for explicit reconciliation rather than silently submitting twice.
Retail submission uses a visible SDK call with the required
`trainingType=globalStandard` extension. SDK write retries are disabled, and
the pending marker is saved before the POST; no failed creation is automatically
resubmitted.
SFT reports single-call original-grader reward alongside exact and
tool-name accuracy over all 280 held-out rows. RFT reports policy reward,
pass count, and exact tool sequence over the same nine read-only scenarios
for both models. No representative result is promised.

SFT and RFT comparison checkpoints bind reuse to the project endpoints,
deployment names, input and grader hashes, selected rows, and generation
settings. RFT also binds the tool server URL, API hash, and auth fingerprint
without saving the credential. Each completed base or fine-tuned side is saved immediately. A
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

The [capstone-specific offline tests](tests/test_retail_capstone.py) cover local
operation coverage, exact RFT-only schemas, in-process authenticated transport,
mocked comparison requests, secret-free persisted state, credential/server
reuse invalidation, and local Git-ignore rules. They do not execute service-stage
cells, submit real jobs, or prove that an external retail server implements the
policy; extracted preparation/state/upload code uses temporary files and mock clients only.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
