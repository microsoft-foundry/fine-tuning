# Advanced RFT: Math Reasoning

**Role:** TRAINING  
**Level:** Advanced / quota-dependent  
**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

This notebook uses [100/10 curated OpenR1-Math-220k subsets](data/README.md),
including three explicitly labeled, self-contained local prompt adaptations,
an explicit runtime answer-only adaptation of training row 1, and a custom
boxed-answer grader. The full source row remains immutable: its two-part prompt
asks for all sides and a proof, which the stored `9` does not supply. The
[independent derivation](data/README.md#runtime-row-1-curation-and-independent-proof)
gives `(4, 6, 9)` and proves the square-side claim; the runtime task deliberately
grades only part (a), not proof quality. It includes adversarial grader tests,
bounded monitoring, and validation reward history.

## Safe execution

The training model is `qwen3.6-35b-a3b`, defined by `MODEL` in the notebook. Copy this module's `.env.template` to a module-local `.env` and configure your own project endpoint. Confirm model access, region support, and quota before training. Offline integrity and grader tests require no Azure resources. Executing the upload/training cell submits or resumes a paid job. Baseline and endpoint inference also incur charges when enabled. The notebook warns before the first such operation without adding an approval gate.

The ignored runtime state records the endpoint at `equivalence.project_endpoint`
and the top-level `job_id`, `training_file_id`, and `validation_file_id`, compatible
with [post-training cancellation and cleanup](../../POST_TRAINING.md). It also
binds the model, reinforcement method/grader, training type, suffix, canonical
file names, hashes, and sizes. An atomic, flushed checkpoint precedes each
upload/job POST; every returned ID is checkpointed before processing or further
requests. Keep this state on durable local storage. Do not commit it, copy its
identifiers into shared artifacts, or delete it merely to rerun. Clear populated
notebook outputs before committing.

**Resume and ambiguity:** rerunning Section 5 loads the saved state and retrieves
its IDs without another upload/job POST. Reuse accepts pending, validating,
queued, running, or succeeded jobs only when their model, input files, and
reinforcement recipe match. File content is downloaded and checked canonically;
a matching name alone is insufficient. Mismatched, failed, or cancelled jobs
raise rather than submitting replacements. An interrupted upload can be
reconciled automatically only when exactly one matching remote file is visible
and its content verifies. An empty listing is not proof that an upload failed.
An interrupted job POST with no saved ID requires a Foundry check and the
verified `FOUNDRY_EXISTING_MATH_RFT_JOB_ID`; the helper never blindly resubmits.
After a process crash, a retained submission lock also requires reconciliation
and confirmation that no writer remains before manual removal. If Foundry
confirms no resource was created, intentionally clear only the corresponding
pending marker after retaining the evidence; this authorizes a new attempt.

Upload, job-creation, and chat-completion POSTs are submitted once with SDK retries
disabled. Read and polling operations may retry transient errors. A manually
repeated inference cell can incur another charge; a timeout is not evidence that
the first request was unbilled. Baseline and endpoint evaluation remain optional
and separately enabled.

**Prepared data:** [prepare_math_data.py](prepare_math_data.py) verifies the
preserved source hashes and writes a separate 100/10 runtime split and computed
canonical-LF provenance report under ignored outputs. It changes only training
row 1 into a labeled answer-only subtask; validation and every other training
line are retained. The notebook uploads these prepared paths, not the preserved
files. Prepared content mismatches and old submission equivalence fail closed;
do not delete state or change its hashes to pretend an earlier job used this data.

**Inference identity:** before any notebook inference, the actual notebook
evaluation helper retrieves the configured deployment and requires an exact,
nonempty name/model/version match. The baseline must be `MODEL` at
`MODEL_VERSION` (`1`). Fine-tuned evaluation additionally requires the matching
succeeded job and its exact `fine_tuned_model` artifact. Set
`FOUNDRY_MATH_FINE_TUNED_MODEL_VERSION` only after independently inspecting that
registered artifact's version in Foundry, not by copying the candidate
deployment's version or assuming the base version. Missing/redacted identities
stop before client acquisition or chat POSTs. Deployment listing supplies only
candidates, not identity evidence; no deployment is provisioned here.

Before interpreting a completed job as equivalent, also verify agreement between
service scores and local regrading. [`live_validation.py`](live_validation.py)
contains the reusable submission-state and contract-checking helpers, not an
automation runner. It also parses reward and loss columns from result CSVs; the
notebook's recipe/content checks do not automatically establish local/service
regrade parity.

## Offline regression tests

From the managed training root, use the configured environment:

```powershell
python -B -m pytest -p no:cacheprovider 03-core-rft/02-rft-math-reasoning/tests shared/tests/test_cookbook_contract.py -q
```

These tests use local files, strict service fakes, and AST-selected notebook
functions only. They do not execute upload, training, or inference cells.
They include native filesystem state-replacement tests (including Windows),
CRLF upload/content checks, failure and resume cases, all 110 exact references,
and exact preservation of the earlier five-answer curation and grader fix.
Reference self-regrading checks only parser/label consistency, not correctness
of the full source questions. Additional tests independently derive row 1,
inspect actual prepared outputs/provenance, and exercise wrong-model,
wrong-version, missing-identity, and no-inference failure paths using the
notebook's own helper.

The representative source run completed training and produced validation reward
history `[0.05, 0.2]`.

See [`assets/charts/representative-validation-reward.svg`](assets/charts/representative-validation-reward.svg) and [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json). These assets are historical evidence, not a benchmark for the bundled dataset or guaranteed outcomes; model availability, quota, data sampling, grader behavior, and service updates can change results.
