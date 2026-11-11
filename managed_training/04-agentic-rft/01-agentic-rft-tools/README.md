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
3. Copy this module's `.env.template` to a module-local `.env` and set your own `FOUNDRY_PROJECT_ENDPOINT` and HTTPS `RFT_TOOL_SERVER_URL`. Do not commit these values. The tool must be reachable by the training service.
4. Run the notebook from this module directory or its `notebooks` directory. Data and local tool/contract checks run offline by default.
5. Execute the training cell to validate the remote tool, upload, and submit or resume one paid job.

The visible recipe uses the exact multi-grader (90% exact string match plus 10% fuzzy similarity, invalid grade 0), maximum 10 episode steps, evaluation every 3 steps with 5 samples, compute multiplier 1.0, and medium reasoning effort.

[`tools/catalog_server.py`](tools/catalog_server.py) is a reusable customer tool, not an automation runner. Host it with the checked-in catalog and request/response envelope; an optional `FOUNDRY_TOOL_AUTH_TOKEN` protects its HTTP endpoint. Configure any required authentication in your hosted tool URL before training. [`src/cookbook_utils.py`](src/cookbook_utils.py) retains reusable data-validation and local grading utilities.

For HTTP requests, an explicitly supplied non-null top-level `top_k` takes precedence over `arguments.top_k`; otherwise the nested value is used, defaulting to 3 when omitted. The existing catalog-size limit and function-call output envelope are unchanged.

Runtime evidence is written under ignored `outputs/`. The state file binds each job to its project, full recipe, grader, tool URL, and data hashes. Rerunning resumes that recorded job; a different configuration requires deliberate archival of the state. Job-creation POSTs are submitted once with SDK retries disabled. Reconcile ambiguous submission failures in Foundry before retrying. File and job polling have timeouts, and failed or cancelled jobs raise errors.

The prior successful reference run had training and validation reward fixed at `1.0`. That value is context only; use the fresh job's downloaded metrics for the current conclusion.

Actual results may vary by model version, data, configuration, region availability, and service conditions.

**Next:** [`../02-retail-agent-capstone`](../02-retail-agent-capstone/README.md).
