# SFT Practice: News Summarization

Specialize a configuration-selected base model for this outcome: **Generate concise, factual, journalistic summaries of news articles.**

| Field | Value |
|---|---|
| Stage | `core-sft` |
| Technique | sft |
| Modality | text |
| Canonical notebook | [`notebooks/demo.ipynb`](notebooks/demo.ipynb) |
| Performs fine-tuning | Yes |
| Stability | Stable curriculum module; service/model availability remains date-sensitive |
| Runtime tier | medium |
| Cost tier | medium; check current pricing and quota before enabling job creation |
| Authentication | Microsoft Entra ID through `DefaultAzureCredential` |

## What you will learn

1. Validate platform access and the exact preserved dataset bytes.
2. Define and run a repeatable base-model baseline before training.
3. Upload or reuse meaningful dataset files and create or reuse an SFT job.
4. Monitor terminal state, resolve or create a deployment, and invoke the result.
5. Compare the base and fine-tuned model under the same evaluation contract.
6. Record gains, limitations, current-run identifiers, and cleanup decisions safely.

## Prerequisites

- A Microsoft Foundry project with supervised fine-tuning enabled for your selected model and region.
- Entra ID access through `DefaultAzureCredential` and permissions for project inference/files/jobs.
- Current fine-tuning quota plus deployment quota if you create a deployment.
- The repository's centrally pinned Python environment; this module intentionally has no local requirements file.
- A copied `.env.template` saved as ignored `.env`, with every required value supplied by you.
- Explicitly enable `FOUNDRY_RUN_PAID_JOBS` for training and
  `FOUNDRY_RUN_LIVE_EVALUATION` for paid inference; both default to disabled.
- Review [Platform Readiness](../../00-foundations/00-platform-readiness/README.md) and
  [Evaluation Contract](../../00-foundations/01-evaluation-contract/README.md) first.

No resource, region, subscription, tenant, file, job, model-result, or deployment identifier is committed.

## Dataset contract

**Dataset:** CNN/DailyMail curated RAI-filtered subset

**Source/provenance:** CNN/DailyMail-derived files preserved from Demos/SFT_CNN_DailyMail; source attribution is not a license grant.

| Split | Rows | File | SHA-256 |
|---|---:|---|---|
| Training | 1992 | `data/news-summarization-train.jsonl` | `cf7e7cc1ed8b02f9959c269113bf396546c1e959599376c17095cee7a14ef7fd` |
| Validation | 229 | `data/news-summarization-validation.jsonl` | `25901f4b80de3f6c052f026b5184fca4057dfc6be4ca2be61e2c41f18efe92b7` |

The files are byte-for-byte copies of the source material and retain exact split membership. The notebook
validates these hashes and uploads those same paths. It never fetches one dataset and trains on another.
Each row uses chat JSONL with exactly `system`, `user`, and `assistant` messages.

The source material does not include a separate independent final holdout. The canonical evaluation uses a
fixed prefix of the validation split, matching the original instructional pattern. For a real experiment,
add a separately governed holdout without changing these preserved curriculum files.

See [`data/README.md`](data/README.md) for source paths, byte counts, schema, licensing cautions, and safety notes.

## Evaluation and acceptance

The notebook measures the same fixed examples before and after fine-tuning. Acceptance requires:

- the exact dataset hashes and row counts to pass;
- the training job and deployment to reach valid states;
- no regression in the primary task metric;
- inspection of per-example failures, not just the average;
- a human review before promotion, especially for factual or domain-sensitive summaries;
- a written decision in ignored `outputs/run-summary.json`.

The detailed rubric is in [Evaluation Contract](../../00-foundations/01-evaluation-contract/README.md).

## Representative existing-run evidence

The sanitized assets describe a completed run dated **2026-10-03**:

- Scope: 15 training rows and 4 validation rows from a safety-screened historical execution subset.
- Terminal job status: `succeeded`.
- Approximate successful-job duration: 83 minutes.
- Outcome: The representative run verified training, deployment, and non-empty factual inference; it did not record a base-versus-fine-tuned quality baseline.
- Limitation: The evidence run used a small subset. The canonical notebook validates and uploads the complete preserved 1,992/229 splits, so runtime and outcomes will differ.



Open [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json) and
[`assets/charts/representative-run.svg`](assets/charts/representative-run.svg) without starting a paid run.

*Actual results may vary by model version, data, configuration, region availability, and service conditions.*

## Restart-safe behavior

- Set both existing file IDs to reuse prior uploads.
- Set `FOUNDRY_EXISTING_FINE_TUNE_JOB_ID` to replay monitoring/evaluation without creating a paid job.
- Set `FOUNDRY_FINE_TUNED_DEPLOYMENT` to reuse a deployment.
- Paid job creation remains blocked until `FOUNDRY_RUN_PAID_JOBS=true`.
- Optional deployment creation is separately gated and uses a meaningful `news-summarization-sft-ft-*` name.
- Runtime identifiers and metrics are written only under ignored `outputs/`.

## Cleanup

The final notebook cell closes clients. A deployment created by the notebook is retained unless
`FOUNDRY_DELETE_CREATED_DEPLOYMENT=true`. Delete unused deployments to stop hosting charges. Uploaded files, jobs,
and result models remain service records and should be managed under your organization's retention policy.

## Troubleshooting and limitations

- **Authentication:** sign in with an identity authorized for the Foundry project.
- **Unsupported model or region:** select a currently supported fine-tuning model/version and project region.
- **Quota/capacity:** reuse a job or wait/request quota; do not convert a terminal failure into a fake success.
- **File preprocessing or safety rejection:** inspect the reported row and policy result. Do not silently filter
  these preserved files; create a separately versioned adaptation dataset instead.
- **Long queue/runtime:** the preserved full dataset is intentional. Reuse completed work rather than silently
  replacing it with a different subset.
- **No independent holdout:** treat gains as instructional until validated on separately governed data.

For real-world use, add safety and privacy review, task-specific latency/cost measurement, broader topic coverage, and human review of representative errors.

## Gains, lessons, and next step

The representative evidence is useful but bounded by its recorded scope. Your promotion decision must come
from your current run's baseline, fine-tuned metrics, error review, safety, latency, token use, and cost.

The central lesson is that SFT is an evaluated experiment, not merely a successful job. Keep data lineage
exact, make expensive steps explicit, reuse prior work safely, and retain unfavorable results.

**Next curriculum step:** [Domain SFT: Scientific Summarization](../02-scientific-summarization/README.md)
