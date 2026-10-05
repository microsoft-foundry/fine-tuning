# Platform Readiness

Use this documentation module before any fine-tuning notebook. Its outcome is a go/no-go decision backed
by access, quota, data, cost, and cleanup evidence—not merely a successful sign-in.

## Readiness checklist

| Area | Ready when | Evidence to retain in ignored runtime output |
|---|---|---|
| Project access | `DefaultAzureCredential` can open the intended Foundry project | timestamp and safe operation name |
| Roles | The identity can use project inference, files, fine-tuning jobs, and deployment operations needed by the demo | role review result, not tokens or credentials |
| Model support | The selected version supports SFT in the project region and tier | model/version and check date |
| Training quota | Current quota and capacity can cover the planned job | quota state and expected training scale |
| Inference baseline | A base deployment is visible and accepts the task input shape | deployment alias and smoke-test status |
| Evaluation | Metrics, sample set, thresholds, and human review are defined before training | evaluation contract version |
| Data | Source, rights, schema, safety review, split membership, row counts, and hashes are known | dataset manifest |
| Cost control | Expected training and deployment costs are reviewed; paid creation has an explicit gate | approved cost tier and gate state |
| Cleanup | The owner knows what will be retained or deleted | cleanup decision and owner |

## Configuration contract

1. Copy each demo's `.env.template` to ignored `.env`.
2. Supply your own project endpoint, deployment aliases, model/version, and optional reuse identifiers.
3. Leave paid and deployment-creation gates blank until the preceding notebook cells pass.
4. Use Entra ID through `DefaultAzureCredential`; never commit keys, tokens, endpoints, or identifiers.
5. Keep region, subscription, resource, file, job, result-model, and deployment identifiers only under
   the demo's ignored `outputs/`.

Environment templates contain variable names and descriptions only. They intentionally provide no working
endpoint, region, model, deployment, subscription, tenant, resource group, or apparently usable placeholder.

## Data preflight

Before upload, validate every JSONL line, message role order, non-empty content, maximum size/token risks,
duplicates, obvious train/validation overlap, PII/secrets, harmful-content risk, row counts, byte counts,
and SHA-256. Preserve the curriculum datasets exactly; create a separately versioned adaptation dataset if
you need filtering, relabeling, resampling, or a new split.

The notebook must upload the exact paths it validated. A live fetch or generation path must train on that
validated runtime output—not on an unrelated committed snapshot.

## Naming and restart safety

Use a stable demo slug plus purpose:

- operation: `news-summarization-sft`
- local files: `news-summarization-train.jsonl`, `news-summarization-validation.jsonl`
- job suffix: `news-summarization-sft`
- deployment: `news-summarization-sft-ft-<runtime-suffix>`

A rerun should reuse both file IDs together, a succeeded job ID, and an existing deployment name when
supplied. Never silently create a second paid job because a previous identifier was omitted or malformed.

## Go/no-go rule

Proceed only when access, model support, quota, exact data validation, the base baseline, acceptance
thresholds, cost, and cleanup are understood. Otherwise stop before upload or job creation and record the
blocking condition plainly.

**Next:** [Evaluation Contract](../01-evaluation-contract/README.md), then
[First SFT: Code Bug Detection](../02-first-sft-bug-detection/README.md).
