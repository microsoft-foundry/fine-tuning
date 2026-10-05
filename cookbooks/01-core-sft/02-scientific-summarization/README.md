# Domain SFT: Scientific Summarization

Specialize a configuration-selected base model for this outcome: **Generate concise scientific abstracts while preserving methods, findings, and limitations.**

| Field | Value |
|---|---|
| Stage | `core-sft` |
| Technique | sft, domain-adaptation |
| Modality | text |
| Canonical notebook | [`notebooks/demo.ipynb`](notebooks/demo.ipynb) |
| Performs fine-tuning | Yes |
| Stability | Stable curriculum module; service/model availability remains date-sensitive |
| Runtime tier | long |
| Cost tier | high; check current pricing and quota before enabling job creation |
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

**Dataset:** PubMed article summarization curated subset

**Source/provenance:** PubMed-derived files preserved from Demos/SFT_PubMed_Summarization; source attribution is not a license grant.

| Split | Rows | File | SHA-256 |
|---|---:|---|---|
| Training | 1000 | `data/scientific-summarization-train.jsonl` | `7e578bf92c922f849853e79e12f1bf23de9b24b56d64dea261093b826a1d5f8e` |
| Validation | 100 | `data/scientific-summarization-validation.jsonl` | `17349f913142c46e4c28cb10e5d326430d37567fa3ecabc851ccc09f00e08494` |

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

- Scope: 10 training rows and 10 validation rows from a disjoint historical execution subset.
- Terminal job status: `succeeded`.
- Approximate successful-job duration: 80 minutes.
- Outcome: The representative run verified training, deployment, and a structured scientific summary with the expected study details; it did not record an independent quality baseline.
- Limitation: The evidence run used a small subset. The canonical notebook validates and uploads the complete preserved 1,000/100 splits and requires human factuality review before any real medical use.



Open [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json) and
[`assets/charts/representative-run.svg`](assets/charts/representative-run.svg) without starting a paid run.

*Actual results may vary by model version, data, configuration, region availability, and service conditions.*

## Restart-safe behavior

- Set both existing file IDs to reuse prior uploads.
- Set `FOUNDRY_EXISTING_FINE_TUNE_JOB_ID` to replay monitoring/evaluation without creating a paid job.
- Set `FOUNDRY_FINE_TUNED_DEPLOYMENT` to reuse a deployment.
- Paid job creation remains blocked until `FOUNDRY_RUN_PAID_JOBS=true`.
- Optional deployment creation is separately gated and uses a meaningful `scientific-summarization-sft-ft-*` name.
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

Scientific and medical summarization requires domain-expert review, source-grounded factuality checks, privacy review, and clear non-clinical-use boundaries. Overlap metrics cannot establish medical safety.

## Gains, lessons, and next step

The representative evidence is useful but bounded by its recorded scope. Your promotion decision must come
from your current run's baseline, fine-tuned metrics, error review, safety, latency, token use, and cost.

The central lesson is that SFT is an evaluated experiment, not merely a successful job. Keep data lineage
exact, make expensive steps explicit, reuse prior work safely, and retain unfavorable results.

**Next curriculum step:** [Multimodal SFT: Chart Reasoning](../../02-multimodal-sft/01-chart-reasoning/README.md)
