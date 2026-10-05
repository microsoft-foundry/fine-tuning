# Model and notebook index

## Model roles

| Role | Meaning | Storage rule |
|---|---|---|
| Base/student model | The model submitted to the fine-tuning job. | Exact supported version in demo metadata or an explicit runtime configuration value. |
| Fine-tuned result model | The service-produced model identifier. | Record only in the user's ignored runtime manifest. |
| Teacher model | Produces labels, demonstrations, synthetic data, or traces. | Document the role separately from the model being optimized. |
| Evaluator model | Scores outputs independently of training. | Never describe it as the fine-tuned model. |
| Grader | Computes training reward or evaluation criteria. | Version the definition and keep final evaluation independent. |
| Deployment | A callable endpoint name for a base or fine-tuned model. | Runtime configuration/output only; no committed customer-specific name. |

`catalog.yml` records known source model evidence for routing.
`validation.status: partial` means static/offline validation is complete but the
clean-slate notebooks have not submitted new paid jobs to reproduce every
historical result.

## Notebook roles

The default is exactly one `notebooks/demo.ipynb` with role `end-to-end` and an
explicit `performs_fine_tuning` boolean. Other allowed roles are:

- `supporting`: required only when a genuine platform boundary prevents one notebook;
- `evaluation-only`: compares models without creating a fine-tuning job;
- `data-generation`: creates or acquires the exact validated data later uploaded.

The 14 catalog entries each resolve to one canonical end-to-end notebook.

## Result language

Keep these outcomes separate:

- **Platform success:** the file, job, deployment, or evaluation operation completed.
- **Model-quality success:** independent metrics improved enough for the hypothesis.
- **Deployment success:** the resulting model became callable under the selected quota/SKU.
- **Safety success:** safety and unrelated-capability checks met the demo's contract.

A succeeded job is not automatically a useful, safe, or deployable model.
