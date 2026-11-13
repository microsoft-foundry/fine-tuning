# UCF101 data

The notebook retrieves immutable Kaggle dataset
`matthewjansen/ucf101-action-recognition/versions/4` and verifies the SHA-256
hashes of its official `train.csv` and `val.csv` before parsing either CSV.
It selects the
three most common training classes and samples 12 training and 4 validation
clips per class with seed 0. The source test partition is not used by this
training-only cookbook.

The pinned source CSV hashes are:

| Source | SHA-256 |
|---|---|
| `train.csv` | `e67f43569701064a568a1e56bcfb2cce5646caa017c1f45171bcb347aeed6ea4` |
| `val.csv` | `df8125e4170c3567ee89db96a860be524c6e031fb35aa7c96925e3dd8f4fcabf` |

The generated `lineage.json` records the requested version and handle, verified
CSV hashes, and selected relative clip paths and labels for each split. It does
not infer a source version from the mirror's mutable latest download.

All frames from a source clip stay in one split. Frame positions exactly match
the source helper: three indices are selected with
`linspace(0, duration, 3, endpoint=False) * fps`.

Each decoded color frame receives a 201 by 201 Gaussian blur at the original
resolution and is encoded with the default OpenCV JPEG encoder. There is no
edge transform or timestamp text strip; temporal order carries the frame
sequence. The generated split contains 36 training rows and 12 validation
rows, each with three ordered frames.

The source demo committed no generated UCF101 train or validation JSONL, so
there are no preserved files to migrate. Live preparation is authoritative and
training consumes the exact generated paths.

Review the [UCF101 dataset](https://www.crcv.ucf.edu/research/data-sets/ucf101/), its citation, and the Kaggle mirror's terms before use.
