# Core Reinforcement Fine-Tuning

This track teaches reward design before training:

1. [`01-rft-countdown`](01-rft-countdown/README.md) compares a model/score grader with a deterministic Python grader on the same Countdown preparation, baseline, training, deployment, and evaluation contract.
2. [`02-rft-math-reasoning`](02-rft-math-reasoning/README.md) applies a hardened Python grader to longer advanced-math reasoning and separates training success, evaluation quality, deployment availability, and inference quality.

Both notebooks are safe to run through their offline preparation and grader-test sections. Remote inference, grader calibration, file upload, RFT submission, deployment checks, and hosted evaluation are individually gated. The committed notebooks never launch a paid job by default.

Shared track helpers live in [`src/rft_common.py`](src/rft_common.py). They are intentionally limited to concerns planned for the repository-wide shared layer: environment validation, content-addressed remote names, idempotent uploads, bounded retry/polling, hashes, JSONL validation, and training-event metric extraction.

