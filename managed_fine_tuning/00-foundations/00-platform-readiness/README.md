# Training readiness

Use this short checklist before any fine-tuning notebook.

## Readiness checklist

| Area | Ready when | Evidence to retain in ignored runtime output |
|---|---|---|
| Project access | `DefaultAzureCredential` can open the intended Foundry project | timestamp and safe operation name |
| Roles | The identity can use project files and fine-tuning jobs | role review result, not tokens or credentials |
| Model support | The selected version supports SFT in the project region and tier | model/version and check date |
| Training quota | Current quota and capacity can cover the planned job | quota state and expected training scale |
| Metrics | The expected service loss, token-accuracy, or reward columns are known | metric names |
| Data | Source, schema, split membership, row counts, and hashes are known | dataset manifest |

## Configuration contract

1. Copy each demo's `.env.template` to ignored `.env`.
2. Supply your own project endpoint, model/version, and optional reuse identifiers.
3. Use Entra ID through `DefaultAzureCredential`; never commit keys, tokens, endpoints, or identifiers.
4. Keep region, subscription, resource, file, job, and result-model identifiers only under
   the demo's ignored `outputs/`.

Environment templates contain variable names and descriptions only. They intentionally provide no working
endpoint, region, model, subscription, tenant, resource group, or apparently usable placeholder.

## Data preflight

Before upload, validate every JSONL line, message role order, non-empty content, maximum size/token risks,
duplicates, obvious train/validation overlap, row counts, byte counts,
and SHA-256. Preserve the curriculum datasets exactly; create a separately versioned adaptation dataset if
you need filtering, relabeling, resampling, or a new split.

The notebook must upload the exact paths it validated. A live fetch or generation path must train on that
validated runtime output—not on an unrelated committed snapshot.

For a runnable custom-data workflow, use
[`ADAPT_YOUR_DATA.md`](../../ADAPT_YOUR_DATA.md). It keeps adaptations separate
from hash-pinned curriculum files and validates schema, duplicates, secrets,
hashes, and split isolation before any service client is created.

## Naming and restart behavior

Use a stable demo slug plus purpose:

- operation: `news-summarization-sft`
- local files: `news-summarization-train.jsonl`, `news-summarization-validation.jsonl`
- job suffix: `news-summarization-sft`

A rerun should reuse both file IDs together and a prior job ID when supplied.
Never silently create a duplicate job because a previous identifier was omitted
or malformed.

**Next:** [Training Metrics Contract](../01-evaluation-contract/README.md), then
[First SFT: Code Bug Detection](../02-first-sft-bug-detection/README.md).
