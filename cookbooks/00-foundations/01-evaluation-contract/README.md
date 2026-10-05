# Training Metrics Contract

Each notebook records the same compact training evidence:

| Item | Record |
|---|---|
| Data | Exact train/validation paths, row counts, byte counts, and SHA-256 |
| Model | Exact catalog model and version |
| Method | SFT or RFT, training type, and explicit hyperparameters |
| Job | Terminal status and trained-token count when available |
| Metrics | Validation loss/token accuracy for SFT, or validation reward for RFT |
| Conclusion | Whether the service training curve improved, declined, or remained flat |

Training metrics describe optimization behavior only. They do not establish
serving quality or downstream task performance.

Save current-run identifiers and result files under ignored `outputs/`.

**Next:** [First SFT: Code Bug Detection](../02-first-sft-bug-detection/README.md).
