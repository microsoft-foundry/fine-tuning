# Agentic RFT: Tool Calling

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is a self-contained, training-only customer workflow for the Tool Calling RFT contract. It does not depend on a local validation runner.

It:

- uses the exact canonical 10-row training and 10-row validation files;
- verifies their row counts, SHA-256 hashes, and `search_catalog` tool schema;
- checks the remote tool request/response contract using `RFT_TOOL_SERVER_URL`;
- uploads both files and submits one reinforcement fine-tuning job;
- uses `Alibaba qwen3.6-35b-a3b`, model version `1`, `GlobalStandard`, and your own compatible project (reference region: Sweden Central);
- monitors the job to a terminal state;
- downloads service result metrics and summarizes training and validation reward curves.

The workflow ends with terminal job state and service-produced reward metrics.

## Run

1. Install the central environment by following [`GETTING_STARTED.md`](../../GETTING_STARTED.md).
2. Authenticate with Azure CLI or another credential supported by `DefaultAzureCredential`.
3. Copy this module's `.env.template` to a module-local `.env` and set your own `FOUNDRY_PROJECT_ENDPOINT` and HTTPS `RFT_TOOL_SERVER_URL`. Do not commit these values. The tool must be reachable by the training service. If it is protected, inject the same `FOUNDRY_TOOL_AUTH_TOKEN` into the hosted server and the notebook environment (or ignored local `.env`). Use a high-entropy secret from your approved secret store; never put it in the tool URL.
4. Run the notebook from this module directory or its `notebooks` directory. Data and local tool/contract checks run offline by default.
5. Execute the training cell to validate the remote tool, upload, and submit or resume one paid job.

The visible recipe uses the exact multi-grader (90% exact string match plus 10% fuzzy similarity, invalid grade 0), maximum 10 episode steps, evaluation every 3 steps with 5 samples, compute multiplier 1.0, and medium reasoning effort.

[`tools/catalog_server.py`](tools/catalog_server.py) is a reusable customer tool, not an automation runner. Host it with the checked-in catalog and request/response envelope; an optional `FOUNDRY_TOOL_AUTH_TOKEN` protects its HTTP endpoint. The notebook uses `Authorization: Bearer ...` for **both** its contract-check POST and the training method's remote-tool headers. An unset token retains the unprotected endpoint behavior. Tool URLs must not contain credentials, query strings, or fragments, and the check does not follow redirects. Do not display the live method/headers or enable HTTP/SDK wire-debug logging. [`src/cookbook_utils.py`](src/cookbook_utils.py) retains reusable data-validation and local grading utilities.

For HTTP requests, an explicitly supplied non-null top-level `top_k` takes precedence over `arguments.top_k`; otherwise the nested value is used, defaulting to 3 when omitted. The existing catalog-size limit and function-call output envelope are unchanged.

## Durable submission and recovery

[`src/agentic_rft_runtime.py`](src/agentic_rft_runtime.py) is a demo-local helper,
not a separate runner or an import from another recipe. It writes ignored
`outputs/submission-state.json` using a flushed, same-directory atomic replacement
and a single-writer lock. The state includes `equivalence.project_endpoint`,
`training_file_id`, `validation_file_id`, and `job_id`, compatible with the
[cancellation and cleanup guide](../../POST_TRAINING.md). It binds the model,
version, training type, full grader/tool recipe, canonical data hashes, and a
method fingerprint. **Only the redacted method and its fingerprint are saved,
never the tool credential or a secret-bearing request payload.** Service events
and error text are redacted before notebook output or local event persistence.

- A durable `pending_operation` is saved **before each upload or job POST**.
	SDK automatic retries are disabled for both kinds of POST.
- Each returned file ID is saved **before processing polls or the next upload**;
	the job ID is saved before returning to monitoring. Polling/read failures retain
	those IDs. Rerunning resumes saved progress, not a fresh pair of uploads.
- An interrupted upload is reconciled read-only using its content-addressed
	filename, purpose, downloaded canonical bytes, and raw byte count. The helper
	checks every SDK list page and refuses duplicate matches. An empty list does
	**not** prove the POST failed and never triggers another pending upload POST.
- An unidentified job submission is never automatically submitted again. Inspect
	the saved project, pending request, file IDs, and recipe in Foundry. Only after
	verifying the accepted job, set `FOUNDRY_RECONCILED_TOOL_RFT_JOB_ID` locally and
	rerun the training cell. The helper retrieves that ID and checks its model,
	both file IDs, and exposed redacted recipe before saving it. It does not infer
	credential equality from masked service headers; the operator must reconcile
	that pending request. If the service omits the necessary recipe fields,
	recovery stops for manual reconciliation instead of sending another POST.
- A hard process crash can leave `outputs/submission-state.json.lock`. Confirm
	that no writer remains and reconcile the state/remote requests before removing
	that lock. Never remove state or clear a pending marker merely because a list
	response is empty. Older state formats, changed credentials, and recipe/project
	mismatches fail closed; retain and reconcile their resources before deliberately
	archiving state for a new run. Credential rotation also changes the fingerprint.

File and job polling have timeouts, and failed or cancelled jobs raise errors.
Stopping monitoring or encountering an error does not cancel training. Retain
partial state for cleanup: a saved first-file ID remains useful even if validation
upload or job creation never completed. The guide's two-file deletion example
requires both IDs; for a partial submission, review only the non-empty saved IDs
in the same verified project, confirm each file is unshared, and require separate
per-file deletion confirmation. This notebook performs no cancellation/deletion.

The prior successful reference run had training and validation reward fixed at `1.0`. That value is context only; use the fresh job's downloaded metrics for the current conclusion.

Actual results may vary by model version, data, configuration, region availability, and service conditions.

**Next:** [`../02-retail-agent-capstone`](../02-retail-agent-capstone/README.md).
