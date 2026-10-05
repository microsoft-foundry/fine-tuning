# Advanced RFT: Math Reasoning

**Role:** TRAINING  
**Level:** Advanced / quota-dependent  
**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

This notebook preserves the OpenR1-Math-220k subsets and the custom boxed-answer grader while making the grader contract, adversarial tests, baseline, bounded monitoring, validation evaluation, deployment availability, and endpoint evaluation explicit.

## Safe execution

Copy `.env.template` to `.env`. Offline integrity and grader tests require no Azure resources. `FOUNDRY_RUN_PAID_JOBS` is false unless explicitly enabled. An existing job ID can be supplied to inspect a prior run without submitting a new one.

Before reusing a completed job, verify the base model, reinforcement method, full hyperparameter recipe, training and validation file sizes, and agreement between service scores and local regrading. [`live_validation.py`](live_validation.py) provides these checks and parses the reward and loss columns from the downloaded result CSV. A matching job avoids duplicate paid training; a partial match must not be treated as equivalent.

Training success does not imply deployment success or endpoint quality. The representative source run:

- completed training and registered a model;
- produced validation reward history `[0.05, 0.2]`;
- verified one 257-word correct validation response;
- could not create or find a deployment because the tested subscription had `0/90` required `GlobalProvisionedManaged` capacity.

See [`assets/charts/representative-validation-reward.svg`](assets/charts/representative-validation-reward.svg) and [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json). These values are evidence from one run, not guaranteed outcomes; model availability, quota, data sampling, grader behavior, and service updates can change results.
