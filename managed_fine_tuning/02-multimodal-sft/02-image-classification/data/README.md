# Stanford Dogs data

The notebook downloads the Stanford Dogs Kaggle mirror used by the current
branch's original demo and reproduces its split exactly:

- all 120 breed folders;
- the first 50 image paths per breed after sorting by cleaned breed name and
  image path;
- positions 0-39 for training, 40-44 for validation, and 45-49 retained only
  as unused split provenance;
- 4,800 training rows, 600 validation rows, and 600 unused rows.

The training and validation JSONL embed the original JPEG bytes as base64 data
URIs with image detail `low`. Generated JSONL, split membership, row counts,
byte sizes, and SHA-256 hashes are stored under ignored
`outputs/training-only/`.

Stanford Dogs is derived from ImageNet. Follow the source dataset terms when
using or redistributing it.

For supported image formats and service requirements, see the
[vision fine-tuning documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning-vision).
