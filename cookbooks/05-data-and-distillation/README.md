# Data and Distillation

This track contains four canonical, self-contained notebooks:

1. `01-teacher-student-basics/notebooks/demo.ipynb`
2. `02-code-distillation/notebooks/demo.ipynb`
3. `03-synthetic-tool-use/notebooks/demo.ipynb`
4. `04-agent-traces-to-sft/notebooks/demo.ipynb`

Run each notebook from its module directory so `data/` and `artifacts/` resolve locally. The
notebooks locate `stage05_runtime.py` in this track and use it to wait for uploads and fine-tuning
jobs with bounded timeouts. Configure `FOUNDRY_FILE_TIMEOUT_SECONDS`,
`FOUNDRY_JOB_TIMEOUT_SECONDS`, and `FOUNDRY_POLL_INTERVAL_SECONDS` only when the defaults are not
appropriate.

Preserved source datasets and fixtures are copied byte-for-byte and recorded in each module's
`data/SOURCE_DATA_SHA256SUMS.txt`. Generated artifacts are intentionally not committed. Paid
fine-tuning is disabled by default in every module. A submitted job is not treated as a trained,
deployed, or evaluated model: each notebook reports those outcomes separately.
