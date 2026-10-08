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

If a model-grader job is already recorded in `outputs/submission-state.json`, rerunning verifies the deployment and minimum TPM but does not resize it. A new deployment operation records its prior and requested capacity in `outputs/grader-deployment-state.json`. After both jobs are terminal, use that evidence to deliberately restore the prior capacity or remove the deployment through your approved resource-management process. The notebook does not perform automatic cleanup.

## Training workflow

Executing the deployment, upload, and submission cells can incur charges and submits or resumes two jobs. The notebook validates both grader contracts, uploads the canonical dataset bytes, monitors each job with a timeout, downloads result CSVs, and summarizes the validation-reward curves produced by the current run.

Runtime state and downloaded results are written under ignored `outputs/`. The submission fingerprint includes the project, model, recipe, graders, response schema, and data hashes. A mismatch requires deliberate archival of the previous state. Job-creation POSTs use disabled SDK retries; reconcile an ambiguous submission in Foundry before rerunning.

For stopping either remote job and reconciling shared files and grader deployment
capacity, follow the [Countdown cancellation and cleanup guide](CANCELLATION_AND_CLEANUP.md).
Closing the notebook or stopping monitoring does not cancel a job.

Training reward is service telemetry, not held-out task-quality evidence. Actual results may vary by model version, data, configuration, region availability, and service conditions.
