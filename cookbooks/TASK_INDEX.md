# Task index

| I need to... | Shared starting point | Planned examples |
|---|---|---|
| Configure and authenticate | `shared/config.py`, `shared/auth.py`, `.env.template` | Every demo |
| Validate JSONL before upload | `shared/dataset_validation.py` | Every training demo |
| Preserve and prove dataset contents | `hash_file`, experiment manifest schema | Every migrated dataset |
| Upload once and safely reuse | `upload_or_reuse_file` | Every training demo |
| Create or resume an equivalent job | `create_or_reuse_fine_tuning_job` | SFT and RFT demos |
| Monitor a job with terminal errors | `monitor_fine_tuning_job` | SFT and RFT demos |
| Discover or explicitly create a deployment | `shared/deployment.py` | Deployment-capable demos |
| Invoke chat or Responses APIs | `invoke_chat`, `invoke_response` | Evaluation and capstone demos |
| Create and poll cloud evaluations | `shared/evaluation.py` | Baseline and final evaluation |
| Use meaningful remote names | `NameFactory` | Files, jobs, deployments, evaluations |
| Log without leaking customer context | `shared/logging.py` | Every service operation |
| Retry only transient failures | `RetryPolicy`, `retry_call` | Network and service propagation |
| Record an experiment decision | `ExperimentManifest` | Every canonical notebook |
| Persist job/resource IDs locally | `RuntimeManifest` under `outputs/` | Resumable customer runs |

The shared helpers intentionally do not select models, regions, quota, preview
features, or deployment SKUs. Those choices belong in demo metadata and explicit
notebook cells so customers can see and review them.
