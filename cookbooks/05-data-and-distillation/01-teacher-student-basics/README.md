# Teacher-Student Distillation Basics

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to generate sarcastic-but-correct teacher labels, calibrate an independent evaluator, filter candidates, fine-tune a smaller student, and compare teacher, base student, and distilled student behavior.

The canonical live path trains on the exact teacher output produced and validated in the notebook. Preserved source questions and baseline examples remain byte-identical and are documented in `data/SOURCE_DATA_SHA256SUMS.txt`.

## Acceptance contract

The primary goal is to improve the smaller student's teacher-style sarcasm while preserving factual correctness. On the same held-out rows and evaluator, the demo passes when the fine-tuned student's average sarcasm score is higher than the base student's and its average factual score does not regress. The evaluator's overall score is reported as a diagnostic, not as the primary acceptance metric, because its prompt does not define a fixed weighting between factuality and style.

The 2026-10-04 live calibration found that factual, low-sarcasm and high-sarcasm answers all received overall scores of 9–10 while sarcasm averages ranged from 2.75 to 5.42. This ceiling effect makes `overall_score` insufficiently sensitive to the style-transfer objective; inspect factual and sarcasm dimensions separately.

## Run

Install the central environment from `cookbooks/requirements.lock`, copy `.env.template` to `.env`, authenticate with Entra ID, and open the canonical notebook. Live generation, evaluation, and paid training require explicit opt-in.

For an unattended live validation, configure the same environment variables and run
`python live_validate.py`. The validator tries `FOUNDRY_PROJECT_ENDPOINTS` in order,
creates a new timestamped `outputs/` directory, generates and judges fresh teacher
answers, trains only on the accepted rows from that run, waits for uploaded files and
the fine-tuning job, creates and verifies the fine-tuned deployment, and evaluates the
teacher, base student, and fine-tuned student on one held-out set.

Representative sanitized metrics are in [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json). They show a small historical run where the distilled student matched the teacher's average score; they are evidence, not a guaranteed outcome.

**Next:** [`../02-code-distillation`](../02-code-distillation/README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
