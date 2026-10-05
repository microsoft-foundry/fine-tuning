# Stanford Dogs data

The notebook downloads the Stanford Dogs Kaggle mirror used by the source demo. It uses four fixed breed folders and the first 20 sorted JPEG filenames per class: indices 0-11 train, 12-15 validation, and 16-19 holdout.

Images are converted to RGB JPEG thumbnails with a maximum size of 384 by 384 and quality 82. Image IDs are asserted disjoint across splits. Managed vision safety preprocessing can reject rows; record submitted and accepted counts, and fail if a rejection manifest removes an entire class.

`preserved/` contains the committed training and validation JSONL byte-for-byte. [`hash-manifest.csv`](hash-manifest.csv) verifies hashes, row counts, and sizes. [`split-membership.csv`](split-membership.csv) retains all 80 bounded-run IDs, filenames, labels, and assignments without absolute workstation paths.

Stanford Dogs is derived from ImageNet. Review the original image licenses and dataset terms before use or redistribution.
