# Model and notebook index

## Model roles

| Role | Meaning | Storage rule |
|---|---|---|
| Base/student model | The model submitted to the fine-tuning job. | Supported model ID in demo metadata or an explicit runtime configuration value. |
| Fine-tuned result model | The service-produced model identifier. | Record only in the user's ignored runtime manifest. |
| Teacher model | Produces labels, demonstrations, synthetic data, or traces. | Document the role separately from the model being optimized. |
| Grader | Computes reinforcement-training reward. | Version the definition and keep it attached to the training contract. |

`catalog.yml` records known source model evidence for routing.
`validation.status: partial` describes the portable notebook and dataset checks;
live outcomes still depend on the reader's model access, region, quota, and data.

## MAI-Code-1.1-Flash (preview)

The [code-repair RFT tutorial](03-core-rft/03-rft-code-repair/README.md) uses
`MAI-Code-1.1-Flash` as the base model and `o3-mini-2025-01-31` as the
score-model grader. The grader is not a teacher or the model being fine-tuned.
MAI-Code-1.1-Flash supports managed RFT only, not SFT, DPO, or interactive
training. Confirm preview access, the exact model identifier, region, grader
hosting requirements, and quota in your Foundry project before running remote
cells; the notebook's local checks do not establish service eligibility.
Optional serving sections create a Serverless Global Standard deployment
with the Azure management SDK, wait for provisioning, and send one repair
request through the Project SDK client. This is an inference demonstration,
not a held-out quality comparison.

## Notebook roles

The default is exactly one `notebooks/demo.ipynb` with role `end-to-end` and an
explicit `performs_fine_tuning` boolean. Other allowed roles are:

- `supporting`: required only when a genuine platform boundary prevents one notebook;
- `data-generation`: creates or acquires the exact validated data later uploaded.

The 15 catalog entries each resolve to one canonical end-to-end notebook.

## Result language

Report the terminal job status and the service-produced validation loss,
token-accuracy, or reward curve. Describe improvement only as a training signal;
do not infer serving quality or downstream task performance from those curves.
MAI Code Repair additionally demonstrates SDK deployment creation and inference.
The Retail Capstone includes explicitly enabled deployment verification and
base-versus-fine-tuned comparisons on the same preserved examples.
