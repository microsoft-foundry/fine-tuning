# Countdown reinforcement fine-tuning

**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

The single customer-facing notebook contains two training sections: one for the model/score grader and one for the deterministic Python grader. Both use:

- `Alibaba/qwen3.6-35b-a3b`, catalog version `1`
- `GlobalStandard`
- your own compatible `FOUNDRY_PROJECT_ENDPOINT` (reference region: North Central US)
- exact restored 100-row training and 50-row validation content, normalized to canonical LF bytes
- a separately preserved 100-row evaluation file for provenance only

Install the central environment from `cookbooks/requirements.lock`. Copy this module's `.env.template` to a module-local `.env`, set your own `FOUNDRY_PROJECT_ENDPOINT`, run `az login`, open the notebook, and execute the cells in order. Authentication uses `DefaultAzureCredential` with `AIProjectClient`; no API key is required.

Data and grader checks run locally. Executing the client, upload, and submission cells authenticates and submits or resumes **two paid jobs**.

The notebook normalizes CRLF to LF before validating and uploading dataset bytes, validates both grader contracts, waits for file processing, submits both jobs, monitors them with a timeout, downloads result CSVs, and summarizes validation reward curves. Runtime state and downloaded results are written under ignored `outputs/`, allowing interrupted runs to resume the same recorded job IDs rather than create duplicates. The state fingerprint includes the project, model, recipe, both graders, response schema, and data hashes. A mismatch requires deliberate archival of the previous state. Job-creation POSTs are submitted once with SDK retries disabled. Reconcile ambiguous submission failures in Foundry before retrying.

## Reference training metrics

| Grader | Validation reward |
|---|---|
| Model/score grader | `0.67 -> 2.03` (max `2.36`) |
| Deterministic Python grader | `0.25 -> 2.53` (max `2.58`) |

These reward curves are training signals only.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
