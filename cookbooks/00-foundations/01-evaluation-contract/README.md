# Evaluation Contract

Define this contract before training so the fine-tuned model is judged against the same task, data,
metrics, and thresholds as the base model.

## Required contract

| Decision | Record before training |
|---|---|
| Task | Exact input, expected output, and prohibited behavior |
| Comparison | Base deployment, fine-tuned deployment, and teacher/reference only when relevant |
| Data | Fixed evaluation membership, source, hashes, relationship to training/validation, and limitations |
| Metrics | Primary task metric, secondary diagnostics, per-category results, and sample count |
| Acceptance | Minimum improvement, maximum tolerated regressions, and any hard safety/format gates |
| Human review | Sampling plan, reviewer expertise, disagreement handling, and factuality checks |
| Efficiency | Latency, token use, and cost collection under equivalent settings |
| Decision | Promote for broader evaluation, revise data/configuration, or do not promote |

## Curriculum contracts

### Code bug detection

Score bug identification, explanation, and fix quality from 1-10 using a fixed rubric. Report the average
and pass rate at `>= 7`. Inspect each failed example. A model should not advance if it improves wording but
misidentifies bugs or proposes unsafe fixes.

### News summarization

Report token overlap F1, ROUGE-L F1, length, and per-example output. Human review must check factual
consistency, omission of central facts, unsupported details, tone, sensitive-topic behavior, and whether
the summary is appropriately concise. Overlap alone is not acceptance.

### Scientific summarization

Report overlap plus domain-term coverage and inspect methods, cohort/sample size, primary outcomes,
uncertainty, and limitations. Require domain-expert factuality review. Never interpret a high lexical score
as evidence of clinical correctness or suitability for patient care.

## Split honesty

The migrated source demos contain training and validation files but no independent final holdout. Their
canonical notebooks use a fixed validation subset for an instructional before/after comparison and state
that limitation. Do not relabel validation as an independent test set. For a real project, create and
govern a separate holdout without modifying the preserved curriculum files.

## Reporting

Save the current run's hashes, model/configuration, safe identifiers, sample count, per-example results,
aggregate metrics, latency/token/cost observations, regressions, and decision under ignored `outputs/`.
Checked-in assets may contain only compact sanitized evidence with identifiers removed and scope stated.

Use this variability footnote on representative evidence:

*Actual results may vary by model version, data, configuration, region availability, and service conditions.*

**Next:** [First SFT: Code Bug Detection](../02-first-sft-bug-detection/README.md).
