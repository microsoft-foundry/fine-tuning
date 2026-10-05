# Model and notebook index

## Model roles

| Role | Meaning | Storage rule |
|---|---|---|
| Base/student model | The model submitted to the fine-tuning job. | Exact supported version in demo metadata or an explicit runtime configuration value. |
| Fine-tuned result model | The service-produced model identifier. | Record only in the user's ignored runtime manifest. |
| Teacher model | Produces labels, demonstrations, synthetic data, or traces. | Document the role separately from the model being optimized. |
| Grader | Computes reinforcement-training reward. | Version the definition and keep it attached to the training contract. |

`catalog.yml` records known source model evidence for routing.
`validation.status: partial` describes the portable notebook and dataset checks;
live outcomes still depend on the reader's model access, region, quota, and data.

## Notebook roles

The default is exactly one `notebooks/demo.ipynb` with role `end-to-end` and an
explicit `performs_fine_tuning` boolean. Other allowed roles are:

- `supporting`: required only when a genuine platform boundary prevents one notebook;
- `data-generation`: creates or acquires the exact validated data later uploaded.

The 14 catalog entries each resolve to one canonical end-to-end notebook.

## Result language

Report the terminal job status and the service-produced validation loss,
token-accuracy, or reward curve. Describe improvement only as a training signal;
do not infer serving quality or downstream task performance from those curves.
The Retail Capstone additionally includes explicitly enabled deployment and
base-versus-fine-tuned comparisons on the same preserved examples.
