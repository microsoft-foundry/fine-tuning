# Code Distillation: Natural Language to Python

This demo reproduces the explicitly documented historical winning recipe from
`Demos/NL_to_Python_Distillation` rather than the later small checked-in runs.
The teacher generates and judges synthetic examples, GPT-4.1-mini is fine-tuned
on the exact live output, and the same independent evaluator scores teacher,
base student, and fine-tuned student for correctness and conciseness.

## Primary parity contract

| Surface | Required value |
|---|---|
| Generation request | 2,000 live rows |
| Quality gate | Mean of Relevance, Pythonic, Readability, Efficiency >= 3.0/4 plus Python syntax validation |
| Deduplication | Normalized first 160 instruction characters, deterministic seed 42 |
| Training | Exactly 1,500 accepted live rows |
| Validation | Exactly 84 accepted live rows |
| Independent evaluation | The same 84 zero-overlap held-out rows used by the historical demo |
| Teacher | GPT-5.4 project deployment |
| Student/base | `gpt-4.1-mini-2025-04-14` and its matching base deployment |
| Evaluator | Independent GPT-5.4 project deployment, temperature 0 |
| Fine-tuning | Supervised, 1 epoch, learning-rate multiplier 1.3, batch size 1 |
| Primary metrics | Correctness 1-10, conciseness 1-10, combined mean, combined Pass@8 |
| Gain | `(fine_tuned_combined - base_combined) / base_combined * 100` from unrounded 84-row means |
| Secondary diagnostic | Hosted code-interpreter execution of the existing four-case APP_MODE check |

Live generation never falls back to preserved data. `generated-raw.jsonl` is
filtered, deterministically selected, split, hashed, uploaded, and trained in
one lineage. Generated code is parsed locally but is executed only in a
Foundry-hosted code-interpreter container, never in the notebook process.
Generation transport concurrency defaults to 12, and every 1,000-row batch or
retry gets a fresh Entra token. This balances East US 2 quota with token
lifetime without changing prompts, temperatures, scoring, filtering, scale,
split, or training method.
Data Designer 0.9.3 is invoked as two explicit 1,000-row batches because a
single 2,000-row call drops an internal row group; both batches are concatenated
before the one shared filtering, deduplication, split, and hashing pass.
The original notebook's UTF-8 Python validator patch is retained so valid
Unicode identifiers do not fail on Windows code pages.

## Source discrepancies

The original source contains several incompatible evidence regimes, retained
here as evidence rather than blended:

- The historical winner requests 2,000 rows and documents about 1,500 training
  rows plus 84 held-out rows, scores of teacher 9.0, base 8.6, and fine-tuned
  9.2, a reported +6.4% gain, and about 22% shorter code.
- The original README describes a 100-row request producing about 85 training
  rows and four validation rows, with no stable fine-tuned result.
- The checked-in notebook output now uses 11 training rows and one validation
  row.
- The historical prose says 70-80% acceptance and a 5% holdout. Those statements
  do not mathematically produce exactly 1,500 training plus 84 held-out rows.
  Parity mode therefore uses the explicitly reported winning scale as a fixed
  deterministic cohort after applying the original gates.
- The prose defines Pass@8 on combined score, while the old implementation
  counted correctness only. The runner records both and treats combined Pass@8
  as primary.
- Rounded scores 8.6 and 9.2 imply about 7.0%, not 6.4%. The historical +6.4%
  likely used unrounded means; the runner always computes gain from unrounded
  item-level aggregates.

## Run

Install `cookbooks/requirements.lock`, copy `.env.template` to `.env`, and set
the East US 2 project endpoint and exact deployment names. The runner defaults
to the fixed parity values above and rejects scale or method changes:

```powershell
python live_validate.py
```

Runtime identifiers, generated data, hashes, uploaded file records, evaluation
details, and reports are written below ignored `outputs/`. A prior job is reused
only when project, model, exact byte hashes, scale, and all hyperparameters
match.

**Next:** [`../03-synthetic-tool-use`](../03-synthetic-tool-use/README.md).

> Actual scores vary with service-side model versions and generated examples;
> the contract, lineage, and measurement method do not.
