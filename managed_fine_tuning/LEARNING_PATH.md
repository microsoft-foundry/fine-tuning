# Learning path

The physical curriculum is ordered by prerequisite depth. The 15 demos below
have canonical notebooks and preserved data. Static and offline
validation is complete. Upload, training, deployment, inference, and Retail comparison cells perform
live Foundry operations when executed and can incur charges.

| Stage | Goal | Demos |
|---|---|---|
| `00-foundations` | Learn training readiness, metric interpretation, and complete a first small SFT job. | First SFT: Code Bug Detection |
| `01-core-sft` | Practice text SFT and interpret validation loss. | News Summarization; Scientific Summarization |
| `02-multimodal-sft` | Prepare image/video fine-tuning data and monitor training metrics. | Chart Reasoning; Image Classification; Video Action Recognition |
| `03-core-rft` | Design deterministic and model-based rewards for reasoning and code repair. | RFT Countdown; Advanced RFT Math Reasoning; [MAI-Code-1.1-Flash Code Repair](03-core-rft/03-rft-code-repair/README.md) |
| `04-agentic-rft` | Optimize tool calls and combine techniques in a capstone. | Agentic RFT Tool Calling; Agentic Retail Capstone |
| `05-data-and-distillation` | Produce trustworthy training data from teachers, synthetic workflows, and traces. | Teacher-Student Basics; NL-to-Python; Synthetic Tool Use; Agent Traces to SFT |

Each canonical notebook follows the outline in
`shared/templates/notebook-outline.md`: configuration, readiness, data
validation, upload/reuse, job creation, terminal monitoring, service training
metrics, and a scoped training-only conclusion.

The Retail Capstone extends this outline with deployment verification
and explicitly enabled base-versus-fine-tuned comparisons. MAI Code Repair adds
optional Serverless deployment creation and inference through SDK calls.
Neither serving flow is required to complete the training-metric workflow.

Use [Adapt your own data](ADAPT_YOUR_DATA.md) after completing the first SFT
with preserved inputs. Use
[Post-training evaluation, serving, cancellation, and cleanup](POST_TRAINING.md)
to identify the service-produced model artifact and plan held-out evaluation,
deployment, inference, or teardown. These are separate from training completion,
including when a notebook provides optional serving steps.

Do not skip directly to an advanced track unless its catalog prerequisites are
satisfied. In particular, preview access, RFT access, model quota, Application
Insights, and tool schemas are explicit gates rather than hidden setup.
