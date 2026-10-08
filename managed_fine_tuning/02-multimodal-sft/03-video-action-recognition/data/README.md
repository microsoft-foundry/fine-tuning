# UCF101 data

The notebook retrieves the UCF101 Kaggle dataset used by the source demo and
starts from its official `train.csv` and `val.csv` partitions. It selects the
three most common training classes and samples 12 training and 4 validation
clips per class with seed 0. The source test partition is not used by this
training-only cookbook.

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
