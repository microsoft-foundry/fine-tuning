# Agentic RFT: Tool Calling

This module teaches reinforcement fine-tuning for a model that must call a catalog tool and return the correct SKU. The canonical lesson is [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

The notebook is safe by default: it validates preserved data, exercises a local in-process tool contract, builds the grader and job payload, and plots representative metrics without uploading files, starting paid training, or calling a remote endpoint. Opt-in cells perform the complete workflow: HTTPS tool-contract check, content-addressed upload/reuse, RFT submission or existing-job reuse, bounded terminal monitoring, caller-provided deployment resolution, endpoint invocation, and exact-match evaluation on the preserved validation split.

1. Create a virtual environment and install the centrally pinned `cookbooks/requirements.lock`.
2. Copy `.env.template` to `.env` and fill only the settings needed for the optional remote steps. `FOUNDRY_RUN_PAID_JOBS` and `FOUNDRY_RUN_LIVE_EVALUATION` are false unless explicitly enabled.
3. Run the notebook from this module directory.

`data/manifest.json` records the original source path, byte count, row count, split, and SHA-256 digest for each byte-preserved dataset. Grading stays in-notebook and uses the transparent composite reward shown in the lesson. The current SDK can discover deployments but cannot create them through the same client surface, so live evaluation requires `FOUNDRY_RFT_DEPLOYMENT` to name a deployment created through the caller's approved management path.

The notebook renders the local [`assets/charts/representative-reward-curve.svg`](assets/charts/representative-reward-curve.svg) and its machine-readable [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json), so the checkpoint-selection lesson can be followed without executing training.

**Next:** [`../02-retail-agent-capstone`](../02-retail-agent-capstone/README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
