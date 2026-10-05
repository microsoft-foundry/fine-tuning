# SFT with Synthetic Data and Tool-Use

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to convert an OpenAI tool catalog to OpenAPI, submit a ToolUseFineTuning generation recipe, validate the downloaded examples, fine-tune a student, and evaluate tool-name and argument correctness.

The lineage rule is strict: when live generation is enabled, validation, splitting, evaluation, and training consume the exact downloaded output from that run. The preserved OpenAPI files are reference fixtures, never a silent fallback. The notebook submits multiple fresh generation batches by default because a single job can produce too few unique rows for a useful train/validation/test split.

For an unattended validation with regional failover, configure the primary, East US 2, and Sweden Central endpoint variables in `.env.template`, export them in the shell, and run `live_validate.py`. It tries configured projects in that order and writes hashes, service IDs, failures, and structural evaluation results only under ignored `outputs/`. Fine-tuning availability is regional; if generation succeeds but training is unsupported, retain the exact generated split when moving training to the next project.

Representative sanitized metrics are in [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json). Install `cookbooks/requirements.lock`, copy `.env.template` to `.env`, and explicitly opt in to generation or paid training.

**Next:** [`../04-agent-traces-to-sft`](../04-agent-traces-to-sft/README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
