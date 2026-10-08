# Core Reinforcement Fine-Tuning

This track teaches reward design before training:

1. [`01-rft-countdown`](01-rft-countdown/README.md) trains the same Countdown data with a model/score grader and a deterministic Python grader, then summarizes validation reward.
2. [`02-rft-math-reasoning`](02-rft-math-reasoning/README.md) applies a hardened Python grader to longer advanced-math reasoning and summarizes reward history.

Both notebooks are safe to run through their offline preparation and grader-test
sections. File upload and RFT submission occur only when their explicit cells
are executed; there is no hidden enable switch.

Each module loads its own `.env` with process environment variables taking precedence. Copy the module-local blank `.env.template` and configure your own project. Job-creation POSTs are never automatically retried, including by the SDK; reconcile ambiguous submission failures before retrying. Pending jobs are reusable when their requested recipe matches.

Shared track helpers live in [`src/rft_common.py`](src/rft_common.py). They are intentionally limited to concerns planned for the repository-wide shared layer: environment validation, content-addressed remote names, idempotent uploads, bounded retry/polling, hashes, JSONL validation, and training-event metric extraction.
