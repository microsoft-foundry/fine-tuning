# Core Reinforcement Fine-Tuning

This track teaches reward design before training:

1. [`01-rft-countdown`](01-rft-countdown/README.md) trains the same Countdown data with a model/score grader and a deterministic Python grader, then summarizes validation reward.
2. [`02-rft-math-reasoning`](02-rft-math-reasoning/README.md) applies a hardened Python grader to longer advanced-math reasoning and summarizes reward history.
3. [`03-rft-code-repair`](03-rft-code-repair/README.md) uses **MAI-Code-1.1-Flash (preview)** for managed RFT with a reference-patch score-model grader and the preserved 50/15 code-repair splits. This model does not support interactive training.

All three notebooks are safe to run through their offline preparation and grader-contract
sections. File upload and RFT submission occur only when their explicit cells
are executed; there is no hidden enable switch.

The code-repair tutorial checks the supplied grader's structure and field bindings
offline, not the grading model's accuracy. It stops at service training metrics;
the model grader does not execute candidate patches or tests.

Each module loads its own `.env` with process environment variables taking precedence. Copy the module-local blank `.env.template` and configure your own project. Job-creation POSTs are never automatically retried, including by the SDK; reconcile ambiguous submission failures before retrying. Pending jobs are reusable when their requested recipe matches.

Shared track helpers live in [`src/rft_common.py`](src/rft_common.py). They are intentionally limited to concerns planned for the repository-wide shared layer: environment validation, content-addressed remote names, idempotent uploads, bounded retry/polling, hashes, JSONL validation, and training-event metric extraction.
