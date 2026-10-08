# Task index

| I need to... | Shared starting point | Planned examples |
|---|---|---|
| Configure and authenticate | `shared/config.py`, `shared/auth.py`, `.env.template` | Every demo |
| Validate JSONL before upload | `shared/dataset_validation.py` | Every training demo |
| Preserve and prove dataset contents | `hash_file`, experiment manifest schema | Every migrated dataset |
| Upload once and safely reuse | `upload_or_reuse_file` | Every training demo |
| Create or resume an equivalent job | `create_or_reuse_fine_tuning_job` | SFT and RFT demos |
| Monitor a job with terminal errors | `monitor_fine_tuning_job` | SFT and RFT demos |
| Download service result files | Fine-tuning file APIs | Every completed job |
| Summarize validation loss or reward | Demo-local metric parser | Every completed job |
| Use meaningful remote names | `NameFactory` | Files and jobs |
| Log without leaking customer context | `shared/logging.py` | Every service operation |
| Retry only transient failures | `RetryPolicy`, `retry_call` | Network and service propagation |
| Record a training run | `ExperimentManifest` | Every canonical notebook |
| Persist job/resource IDs locally | `RuntimeManifest` under `outputs/` | Resumable customer runs |
| Validate a separately versioned custom dataset | [`ADAPT_YOUR_DATA.md`](ADAPT_YOUR_DATA.md), `python -m shared.validate_dataset` | First SFT adaptation |
| Identify, evaluate, deploy, or sample a trained artifact | [`POST_TRAINING.md`](POST_TRAINING.md) | Post-training handoff |
| Cancel training or clean up resources | [`POST_TRAINING.md`](POST_TRAINING.md) | Explicit lifecycle action |

The shared helpers intentionally do not select models, regions, quota, or
preview features. Those choices belong in demo metadata and explicit notebook
cells so customers can see and review them.
