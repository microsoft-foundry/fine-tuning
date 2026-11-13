# RFT Code Repair: MAI-Code-1.1-Flash and MAI-Code-2-Flash (preview)

**Role:** TRAINING

**Canonical notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

Use the Microsoft Foundry Project SDK to train
MAI-Code-1.1-Flash or MAI-Code-2-Flash to produce minimal, regression-safe Python code
repairs. Both preview models support **managed RFT only**, not SFT, DPO, or
programmed training.

The notebook walks through inspecting [50 training and 15 validation
examples](data/README.md), loading a [score-model grader](graders/README.md),
connecting with `AIProjectClient`, uploading files, creating an RFT job,
monitoring it, and downloading training metrics. Optional sections show
Serverless deployment creation with the Azure management SDK and a
code-repair inference request through the Project SDK. SDK calls are shown
directly; grader settings and RFT hyperparameters are explained alongside the
workflow.

## Model selection and support

| Model | Training model ID |
| :--- | :--- |
| MAI-Code-1.1-Flash | `MAI-Code-1.1-Flash` |
| MAI-Code-2-Flash | `mai-code-2-flash-2026-11-17` |

The notebook keeps MAI-Code-1.1-Flash as its default. To use MAI-Code-2-Flash,
replace the `MODEL` assignment in Section 1 with:

```python
MODEL = "mai-code-2-flash-2026-11-17"
```

Its catalog identifier is `Microsoft/mai-code-2-flash/2026-11-17`. The date
identifies the model version, not a launch date. Use the training model ID,
not the catalog path, in `fine_tuning.jobs.create`.

Both models use the same data schema, score-model grader, RFT submission,
monitoring, and optional deployment/inference workflow in this notebook.
They support **Global** and **Developer** training, not Standard or Data zone,
and **Serverless Global Standard** serving, not Managed Compute or Fireworks.
For supported regions, REST API, SDK, and Foundry portal workflows, follow
the [Foundry fine-tuning support reference](https://learn.microsoft.com/azure/foundry/fine-tuning/overview#supported-models)
and [RFT guide](https://learn.microsoft.com/azure/foundry/openai/how-to/reinforcement-fine-tuning).
The same model-grader prerequisites and safety/evaluation guidance below
apply to either model. Support parity does not imply identical prices,
numerical limits, or model quality.

## Setup

Follow the central [setup guide](../../GETTING_STARTED.md), select the
**Foundry managed** kernel, and run from this demo directory or its
`notebooks/` directory. Copy `.env.template` to a local `.env` and set
`FOUNDRY_PROJECT_ENDPOINT`. Authenticate through a provider supported by
`DefaultAzureCredential`.

Before remote execution, confirm preview/RFT access, model and grader
availability, region support, quota, and data terms for your project.
The notebook explicitly selects **Global training**
(`trainingType=globalStandard`): both MAI models support Global and Developer, not
Standard or Data zone. Review [training-type support](https://learn.microsoft.com/azure/foundry/fine-tuning/overview#supported-models)
and [grader hosting requirements](https://learn.microsoft.com/azure/foundry/openai/how-to/reinforcement-fine-tuning#model-grader-deployments).
Global training does not guarantee residency in the project's region.

## Run the notebook

| Sections | Scope |
| :--- | :--- |
| 1-3 | Inspect local data and understand the model grader. |
| 4-8 | Connect, upload, submit paid training, monitor, and inspect metrics. |
| 9-10 | Optionally create a deployment with the SDK, wait for provisioning, and run paid inference. |
| 11 | Close local clients and review evaluation and cleanup responsibilities. |

**Do not use Run All as an offline smoke test.** Each execution of the upload
or submission cell creates new resources; these cells do not automatically
discover or reuse earlier runs.

File and job IDs are saved under ignored `outputs/`. After an interruption,
use the saved IDs and the same project to retrieve existing resources.
Section 7 can resume monitoring without uploading or submitting again.
Check Foundry before retrying an operation whose outcome is unknown.
See [post-training guidance](../../POST_TRAINING.md) for the saved-file layout,
cancellation, and cleanup.

For serving, both MAI-Code-1.1-Flash and MAI-Code-2-Flash support **Serverless Global Standard**
deployments, not Managed Compute or Fireworks. Fill in the deployment-only
settings in your local `.env` for the same parent Foundry resource used in
training, a new deployment name, and capacity within your quota. Section 9
uses `CognitiveServicesManagementClient` to create the deployment and wait for
provisioning; Section 10 invokes it through the Project SDK client. Serving
requires project/deployment read and deployment-write permission and has
separate quota, hosting, and inference costs. The notebook does not update
existing deployments or create a new Foundry resource. You can stop after
training and skip these optional sections.

## Interpret the results

The `o3-mini-2025-01-31` grader returns a score from 0 to 1 with a pass threshold
of 0.7. It judges repairs against reference patches, expected behavior, and
requirements; it does not apply patches or execute tests.

The validation split is used during training. Service reward metrics do not
establish improvement on unseen tasks. The inference example is a single
repair request, not a base-versus-fine-tuned comparison or a held-out
evaluation. Use an independent test set, isolated executable tests, and human
review before making quality claims.

Actual results may vary by model version, data, configuration, region availability, and service conditions.
