# Countdown reinforcement fine-tuning

**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

This demo compares model-based and deterministic grading for the same Countdown reinforcement fine-tuning task. Both jobs use:

- `Alibaba/qwen3.6-35b-a3b`, catalog version `1`;
- `GlobalStandard` training;
- a `gpt-5.4-mini` `DataZoneStandard` deployment for the model grader;
- the checked-in 100-row training and 50-row validation datasets, normalized to canonical LF bytes;
- a separately preserved 100-row evaluation file for provenance only.

Install the central environment by following [`GETTING_STARTED.md`](../../GETTING_STARTED.md). Copy `.env.template` to a module-local `.env`, configure the project and ARM resource fields, run `az login`, and open the notebook from this cookbook directory or `notebooks/`.

## Grader deployment capacity

The model grader requires high throughput for the score-model sampling evaluation that runs before GPU training. The notebook:

1. verifies that the project endpoint, Azure AI Services account, resource group, subscription, and location agree;
2. rejects an existing deployment with the configured name unless its model, version, format, and SKU match exactly;
3. prints a read-only quota and capacity plan;
4. creates or updates the deployment only after `FOUNDRY_ALLOW_GRADER_DEPLOYMENT_UPDATE=true`;
5. verifies that the resulting rate limit is at least `FOUNDRY_GRADER_MIN_TPM`, which defaults to `100000` TPM.

`FOUNDRY_GRADER_CAPACITY=max` assigns all currently available matching quota. To preserve shared quota, set an explicit capacity-unit limit of at least 100 instead. The requested value cannot exceed the currently available maximum.

Azure CLI performs the ARM quota and deployment operations using the identity selected by `az login`. `DefaultAzureCredential` authenticates the Foundry SDK operations and may select that identity or another configured credential source. Both identities need the appropriate access.

The submitted score grader uses `FOUNDRY_GRADER_DEPLOYMENT` (default `gpt-5.4-mini`), not the underlying catalog model name. Deployment verification still requires `gpt-5.4-mini`, version `2026-03-17`, format `OpenAI`, and SKU `DataZoneStandard`. The configured deployment name and its model contract are part of the submission fingerprint.

Before the first ARM create/update, the notebook exclusively creates `outputs/grader-deployment-state.json`. This immutable operation-intent record saves the project/resource identity, whether the deployment already existed, its prior resource ID and capacity, and the first requested capacity. It is retained even if ARM fails, times out, or returns an insufficient/missing token limit. A rerun validates that configuration and the live deployment without resizing or replacing the original cleanup baseline, even when no job was recorded yet. A missing deployment or unexpected capacity requires manual reconciliation; the notebook does not blindly retry an uncertain operation.

If a model-grader job or submission intent is already recorded in `outputs/submission-state.json`, rerunning also verifies its recipe fingerprint, deployment, and minimum TPM without resizing. Legacy or mismatched deployment state fails closed: reconcile existing jobs and resources rather than deleting evidence to force a retry. The operation-intent record is not proof of successful creation or exclusive ownership. After both jobs are terminal, use independent resource/ownership checks and the first baseline to deliberately restore prior capacity or remove an owned deployment through your approved resource-management process. The notebook does not perform automatic cleanup.

## Training workflow

Executing the deployment, upload, and submission cells can incur charges and submits or resumes two jobs. The notebook validates both grader contracts, including the stable `countdown_python_grader` name, before deployment and again before the first job-creation POST. A single payload builder supplies the exact validated grader to submission. The notebook uploads the canonical dataset bytes, monitors each job with a timeout, downloads result CSVs, and summarizes the validation-reward curves produced by the current run.

Runtime state and downloaded results are written under ignored `outputs/`. The submission fingerprint includes the project, model, recipe, graders, response schema, and data hashes. The deterministic graders reject nonfinite literals, arithmetic intermediates, reported results, targets, and input numbers, including NaN, infinities, and overflow, before awarding any score. Both implementations also cap expressions at 4096 characters, ASTs at 256 nodes, and AST depth at 64 before recursive evaluation; parse/evaluation recursion failures earn zero. Finite scoring and exact input-number multiplicities are preserved, and the standalone and embedded grader have identical executable source.

### Durable submission intents and reconciliation

Before each job-creation POST, `submit_once(label, grader, suffix)` saves a separate immutable intent for `model-grader` or `python-grader`. Submission state uses schema version 2 with `fingerprint`, `jobs`, and `pending` maps. Each `pending[label]` contains:

- intent `schema_version: 1`, `record_type: job_creation_intent`, a UTC `recorded_at`, and its `label`;
- the exact `project_endpoint` and `run_fingerprint`;
- the complete `payload` passed to job creation and its `payload_sha256` (UTF-8 JSON, sorted keys, compact separators, no NaN);
- `files.training` and `files.validation`, each with the exact uploaded `file_id` and canonical LF `bytes`, `rows`, and `sha256`.

The notebook revalidates the local training inputs before recording an intent. The upload cell verifies reused remote file contents and uses the canonical dataset bytes for new uploads. The intent is evidence of the requested operation, not proof that the service accepted it. `save_state` writes a same-directory temporary file, flushes and fsyncs it, closes it, and atomically replaces state. A read-back check precedes the POST. A failed pre-POST save prevents creation. Job-creation SDK retries remain disabled. A returned job ID is saved in `jobs[label]` without removing or replacing `pending[label]`; failure to save that ID leaves the prior pending evidence and reports the returned ID for reconciliation.

An exclusive local `outputs/submission-state.json.lock` spans load, create/retrieve, and both saves. Another writer fails closed rather than racing or waiting. An interrupted process can leave a stale lock: stop all writers and independently reconcile state before removing **only** a confirmed stale lock. Do not break a live lock or remove submission evidence. Use one shared local state directory; this is not a distributed or service-side idempotency guarantee.

Valid legacy job-only state with a matching fingerprint remains readable without being rewritten during resume. Known IDs are retrieved rather than submitted again. Malformed, unverified, or mismatched fingerprints, pending schema, payloads, and file evidence are rejected without changing state. The corrected embedded grader changes the recipe fingerprint; do not relabel old state with a new fingerprint to bypass that check.

If `pending[label]` exists without a known job ID, **no POST is replayed**, even after a timeout, apparently definitive rejection, fresh kernel, changed suffix, or changed uploaded file IDs. Preserve a private copy of the original state bytes. Never clear, delete, or archive pending to resubmit. Independently verify the Foundry project and match the saved endpoint, payload/suffix, exact file IDs/hashes, and service request/audit evidence. Only after independently proving a matching job may an operator record its verified ID in `jobs[label]`, keeping the original intent intact and all writers stopped; rerunning then retrieves that ID. If a matching ID cannot be proved, stop and contact the service operator. Absence from a job listing does not prove that no job was accepted. The two variants retain their intents independently; reconciling one does not authorize replay of the other.

For stopping either remote job and reconciling shared files and grader deployment
capacity, follow the [Countdown cancellation and cleanup guide](CANCELLATION_AND_CLEANUP.md).
Closing the notebook or stopping monitoring does not cancel a job.
The demo-local [cancellation guard](cancellation.py) compares both input files,
suffix, model, and the full observable method/grader with the selected label's
durable intent before one non-retried cancellation request. Independent Foundry
endpoint/job-ID confirmation is required; job-only legacy state or unavailable
recipe identity stops this workflow, with a separate independent operator path.
Do not self-confirm from the state being checked. Shared authentication and
other tracks' cancellation helpers are unchanged.

Training reward is service telemetry, not held-out task-quality evidence. Actual results may vary by model version, data, configuration, region availability, and service conditions.

## Offline regression coverage

[`tests/test_countdown.py`](tests/test_countdown.py) extracts only explicitly allowlisted notebook definitions and contract assignments; it never runs the deployment, upload, or submission cells. Fake ARM and job clients cover custom deployment names, exact submitted payloads, immutable rerun baselines, partial failures, and capacity/identity safeguards. The job fake inspects pending evidence on disk at the POST boundary; regressions cover ambiguous outcomes without replay, atomic-save failures before and after creation, independent variants, locks, legacy resume, and schema/payload tampering. Parameterized grader cases use both the standalone implementation and the notebook's exact embedded source literal for NaN, infinities, overflow, finite reward tiers, exact number usage, deep chains, bounded balanced ASTs, and parser/evaluator recursion failures. [Cancellation tests](tests/test_cancellation.py) use AST-selected actual guard functions and strict fakes to reject stale/swapped IDs, file/suffix/method/grader mismatches, incomplete legacy identity, and non-independent confirmations without remote cancellation. Both LF-normalized artifact manifest checks remain enabled; the modified standalone grader pins are explicitly pending parent recomputation in [the grader manifest](graders/hashes.json). These are offline regression contracts, not evidence of a successful live deployment or training run. Run this module's tests explicitly from the central environment; the central default test paths do not include this directory.
