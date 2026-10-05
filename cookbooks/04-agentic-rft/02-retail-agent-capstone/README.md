# Agentic Retail Capstone

This capstone combines policy-constrained tools, MCP integration, deterministic next-tool evaluation, SFT, and policy-reward RFT. The canonical lesson is [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

The committed notebook is offline-safe by default. Opt-in cells provide complete SFT and RFT paths: content-addressed upload/reuse, paid submission or existing-job reuse, bounded terminal monitoring, caller-provided deployment resolution, invocation, and held-out next-tool evaluation. The evaluation path predicts only the next tool call and never executes mutating retail tools.

All source datasets, policy files, tool schemas, OpenAPI definitions, and graders used by the lesson are byte-preserved. `data/manifest.json` records their source paths, roles, sizes, row counts where applicable, and SHA-256 hashes. Set `FOUNDRY_USE_LIVE_GENERATED_OUTPUT=true` together with both `FOUNDRY_LIVE_GENERATED_*_PATH` values to select live generated SFT data. Those exact files are validated and uploaded; missing or invalid live output raises an error and never falls back to preserved data.

The current SDK can discover deployments but cannot create them through the same client surface. Create deployments through your approved management path, then set `FOUNDRY_SFT_MODEL_NAME` and/or `FOUNDRY_RFT_MODEL_NAME` before enabling live evaluation.

For an operator-run, end-to-end validation with ordered regional failover, use
`scripts/live_validate.py` and pass the primary and repeated failover project
endpoints explicitly. It validates the committed lineage, submits both jobs before
polling, creates deployments through Azure CLI, and evaluates next-tool predictions
without executing any returned retail tool. The ignored
`outputs/live-validation.json` file is updated after every material state change.
Set `FOUNDRY_TOOLS_SERVER_URL` to the approved HTTPS retail simulator endpoint; no
credentials or resource-specific values are written to tracked files.
