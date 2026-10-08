# Advanced RFT: Math Reasoning

**Role:** TRAINING  
**Level:** Advanced / quota-dependent  
**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

This notebook uses [OpenR1-Math-220k subsets](data/README.md)
and a custom boxed-answer grader. It includes adversarial grader tests,
bounded monitoring, and validation reward history.

## Safe execution

The training model is `qwen3.6-35b-a3b`, defined by `MODEL` in the notebook. Copy this module's `.env.template` to a module-local `.env` and configure your own project endpoint. Confirm model access, region support, and quota before training. Offline integrity and grader tests require no Azure resources. Executing the upload/training cell submits or resumes a paid job. The existing-job option is consulted when that cell runs. Reuse accepts pending, validating, queued, running, or succeeded jobs only when their model, input file IDs, and reinforcement recipe match. A mismatched, failed, or cancelled job raises an error rather than submitting a replacement.

Before interpreting a completed job as equivalent, also verify training and validation file sizes and agreement between service scores and local regrading. [`live_validation.py`](live_validation.py) is a reusable contract-checking library, not an automation runner. It provides these additional checks and parses the reward and loss columns from the downloaded result CSV; the notebook's recipe check does not automatically perform local/service regrade parity.

Job-creation POSTs are submitted once, with SDK retries disabled. Reconcile an ambiguous submission failure in Foundry before retrying. Read and polling operations may retry transient errors. Baseline and endpoint evaluation are optional and separately enabled.

The representative source run completed training and produced validation reward
history `[0.05, 0.2]`.

See [`assets/charts/representative-validation-reward.svg`](assets/charts/representative-validation-reward.svg) and [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json). These assets are historical evidence, not a benchmark for the bundled dataset or guaranteed outcomes; model availability, quota, data sampling, grader behavior, and service updates can change results.
