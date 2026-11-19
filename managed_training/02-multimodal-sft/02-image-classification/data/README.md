# Stanford Dogs data

The notebook downloads immutable Kaggle dataset
`jessicali9530/stanford-dogs-dataset/versions/2` and applies the source demo's
split rule:

- all 120 breed folders;
- the first 50 image paths per breed after sorting by cleaned breed name and
  image path;
- positions 0-39 for training, 40-44 for validation, and 45-49 retained only
  as unused split provenance;
- 4,800 training rows, 600 validation rows, and 600 unused rows.

The training and validation JSONL embed the original JPEG bytes as base64 data
URIs with image detail `low`. Before serialization, the notebook hashes each
selected source JPEG and records its relative path, split, and SHA-256 in
`split-membership.csv`; serialization rejects a JPEG changed since selection.
The generated dataset manifest records version 2 and the membership-file hash,
alongside JSONL row counts, byte sizes, and SHA-256 hashes. All generated files
remain under ignored `outputs/training-only/`. Existing unversioned download
directories are not reused as evidence of the pinned version.

Stanford Dogs is derived from ImageNet. Follow the source dataset terms when
using or redistributing it.

For supported image formats and service requirements, see the
[vision fine-tuning documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning-vision).
