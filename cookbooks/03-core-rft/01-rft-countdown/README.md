# RFT Deterministic Reasoning: Countdown

**Role:** TRAINING VARIANTS  
**Level:** Advanced  
**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

The notebook consolidates the former model/score-grader and Python-grader notebooks. Both variants share the same immutable data, prompt, structured response, offline validator, baseline cases, job monitoring, deployment check, and final exact-answer evaluation.

## Safe execution

Copy `.env.template` to `.env` and fill only resources you intend to use. Offline data and grader tests require no Azure resources. Remote baseline and grader calibration are separately gated. `FOUNDRY_RUN_PAID_JOBS` is false unless explicitly enabled, so running all cells does not submit training by default.

The notebook uses `DefaultAzureCredential`; run `az login` before remote cells. `FOUNDRY_GRADER_MODEL` is the catalog model used by RFT (for example, `o3-mini`), while `FOUNDRY_GRADER_DEPLOYMENT` is a callable deployment used only for adversarial calibration. Keep them separate because a supported training grader model is not necessarily deployed for chat inference. Install dependencies from the repository's central pinned environment rather than creating a per-demo requirements file.

## Preserved evidence

- Every source Countdown JSONL split and grader input is copied byte-for-byte into `data/preserved/` or `graders/preserved/`.
- [`data/hashes.json`](data/hashes.json) records source paths, destination names, sizes, row counts, and SHA-256 values.
- [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json) records the existing representative result: baseline `1/3`, model-grader RFT `3/3`, and Python-grader RFT `3/3`.

Those three-case numbers demonstrate the workflow, not a statistical benchmark. Results vary with model snapshot, grader model, service version, region, random sampling, and deployment configuration.
