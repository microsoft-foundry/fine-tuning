# Bug-detection demonstration data

These training and validation examples demonstrate bug identification,
explanation, and suggested fixes. They are sample data, not a production
quality code-correctness benchmark.

## Published source snapshot

The reference is the [Microsoft fine-tuning Text-Bug-Detection directory](https://github.com/microsoft-foundry/fine-tuning/tree/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection)
at Git revision `cc042d12f6704be36c70f9f4b2457aeaa12b97c7`.

| Published source file | Git blob SHA-1 | Source bytes |
|---|---|---:|
| [training_data.jsonl](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/training_data.jsonl) | `fc9a739cbf84c5c3de3c5912fba087a51004c270` | 213931 |
| [validation_data.jsonl](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/validation_data.jsonl) | `d2a408dd90a0a7388ad6e338befc234c4d2f54d8` | 20895 |

These identify upstream Git blobs, not newly computed local SHA-256 values.
Checkout line endings can differ. This notice does not claim that local/source
byte comparisons have been executed, and it changes no JSONL records.

## Retained license

The [upstream repository license](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/LICENSE)
is MIT, with `Copyright (c) 2025 Azure AI Foundry`. The complete copyright,
permission, retention condition, and disclaimer are included in
[LICENSE.md](LICENSE.md). Keep that notice with redistributed copies, including
the managed cookbook's renamed copies.