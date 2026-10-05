# UCF101 data

The notebook retrieves the UCF101 Kaggle dataset used by the source demo and starts from its official `train.csv`, `val.csv`, and `test.csv` partitions. It selects the three most common training classes and samples 12 train, 4 validation, and 5 holdout clips per class with seed 0.

All frames from a source clip stay in one split. Three evenly spaced frames are converted to RGB, reduced to 8 by 8 pixels, smoothly reconstructed to 256 by 256, and strongly blurred. This preserves coarse scene color and temporal change while obscuring identity details and removes the CAPTCHA-like high-frequency and block-edge patterns that the live fine-tuning preprocessor rejected from earlier transforms.

The source demo committed no generated UCF101 train, validation, or holdout JSONL, so there are no preserved files to migrate. Live preparation is authoritative and training consumes the exact generated train and validation paths.

Review the [UCF101 dataset](https://www.crcv.ucf.edu/research/data-sets/ucf101/), its citation, and the Kaggle mirror's terms before use.
