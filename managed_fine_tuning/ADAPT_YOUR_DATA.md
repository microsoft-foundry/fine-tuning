# Adapt your own data

Keep customer adaptations separate from the hash-pinned curriculum datasets.
Do not overwrite files under a demo's `data/` directory or remove its fixed
integrity checks.

## 1. Create a versioned input

Start at the repository root, enter the first-SFT cookbook, copy the offline
example, and give the copy a stable version:

```powershell
Set-Location managed_fine_tuning\00-foundations\02-first-sft-bug-detection
New-Item -ItemType Directory -Force outputs\adaptations | Out-Null
Copy-Item -Recurse adaptations\example-v1 outputs\adaptations\my-domain-v1
```

```bash
cd managed_fine_tuning/00-foundations/02-first-sft-bug-detection
mkdir -p outputs/adaptations
cp -R adaptations/example-v1 outputs/adaptations/my-domain-v1
```

Replace `train.jsonl` and `validation.jsonl` in the copied directory. Each line
must be one JSON object with a non-empty `messages` array, supported roles, and
at least one assistant message. Keep held-out evaluation examples in a third
file that is never uploaded for training or validation.

Confirm that you have the right to use the data and that it contains no secrets
or unsupported personal data. The repository validator detects credential-like
strings but is not a substitute for privacy, safety, or license review.
The `outputs/` location is ignored by Git so customer data and generated
manifests are not offered for commit.

## 2. Validate before any service operation

Return to `managed_fine_tuning`, then run the validator with the central
environment. The command writes LF-normalized upload-ready copies separately
from the source files:

```powershell
Set-Location ..\..
.\.venv\Scripts\python -m shared.validate_dataset `
  --train .\00-foundations\02-first-sft-bug-detection\outputs\adaptations\my-domain-v1\train.jsonl `
  --validation .\00-foundations\02-first-sft-bug-detection\outputs\adaptations\my-domain-v1\validation.jsonl `
  --canonical-output-dir .\00-foundations\02-first-sft-bug-detection\outputs\adaptations\my-domain-v1\prepared `
  --output .\00-foundations\02-first-sft-bug-detection\outputs\adaptations\my-domain-v1\validation-manifest.json
```

```bash
cd ../..
.venv/bin/python -m shared.validate_dataset \
  --train ./00-foundations/02-first-sft-bug-detection/outputs/adaptations/my-domain-v1/train.jsonl \
  --validation ./00-foundations/02-first-sft-bug-detection/outputs/adaptations/my-domain-v1/validation.jsonl \
  --canonical-output-dir ./00-foundations/02-first-sft-bug-detection/outputs/adaptations/my-domain-v1/prepared \
  --output ./00-foundations/02-first-sft-bug-detection/outputs/adaptations/my-domain-v1/validation-manifest.json
```

The command fails on invalid JSONL, unsupported message shape, detected
credential-like content, or exact records shared by train and validation. The
manifest records row counts, source and upload byte counts/SHA-256 hashes, role
counts, duplicates, split-isolation evidence, and the prepared output paths.
CRLF source files therefore have different source evidence but the same
LF-normalized upload evidence as equivalent LF files. Review duplicate warnings
rather than assuming duplicated examples are intentional.

## 3. Preserve the canonical demo contract

The canonical first-SFT notebook remains pinned to its preserved 224-row
training and 20-row validation files. It does not read adaptation paths from
`.env`, and the example adaptation files are never uploaded by any canonical
notebook.

Use the generated manifest as the offline starting point for a separately
versioned customer workflow. Before implementing that workflow, choose a
supported model and method, document bounded hyperparameters and stop
conditions, retain a held-out evaluation split, and ensure the upload code uses
the prepared `upload_path` files and `upload_sha256` values recorded by the
validator. Do not edit a canonical notebook's fixed hashes or paths to make
different data pass.
