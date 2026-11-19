# Orca DPO sample: sources and license scope

These training and validation files are demonstration data, not a production
quality or independently validated preference benchmark. This notice changes
attribution and license documentation only; it does not change either JSONL file.

## Published Microsoft sample snapshot

The public [Microsoft fine-tuning sample directory](https://github.com/microsoft-foundry/fine-tuning/tree/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Direct_Preference_Optimization/orca_dpo_pairs)
is pinned to Git revision `cc042d12f6704be36c70f9f4b2457aeaa12b97c7`.
Its published Git blob identifiers are:

| Sample | Source Git blob SHA-1 | Source bytes |
|---|---|---:|
| [training.jsonl](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Direct_Preference_Optimization/orca_dpo_pairs/training.jsonl) | `d410ff3cff9ee1724f6e3be7aedb9cf28ce734ac` | 8986685 |
| [validation.jsonl](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Direct_Preference_Optimization/orca_dpo_pairs/validation.jsonl) | `9846f0ebc564f037935c607115c305a69c26dfc7` | 957910 |

These identify the published reference files, not freshly computed hashes of
this checkout. Git blob identifiers are not raw-file SHA-256 values.

## Intel dataset reference

The relevant preference-pair reference is
[Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs/tree/624952e3f420ae18d88b31977ee2ea436c833abb),
at the verified Hugging Face revision
`624952e3f420ae18d88b31977ee2ea436c833abb`. Its
[pinned dataset card](https://huggingface.co/datasets/Intel/orca_dpo_pairs/raw/624952e3f420ae18d88b31977ee2ea436c833abb/README.md)
declares `license: apache-2.0` and describes 12k examples from the Orca-style
[Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset,
with a link to the [Orca paper](https://arxiv.org/abs/2306.02707).

The pinned Intel tree records its full dataset as a 36310081-byte LFS object
with content SHA-256
`ccfb85f1dc86bb7a016db1f3e96ccc74d3f28dc9bd4bd65a1f861c2773e0848f`.
That is the upstream LFS object identifier, **not** a hash of either bundled
sample and not evidence of a completed local download or row comparison.

### Provenance limitation

The historical Microsoft notice says the samples were curated based on
OpenOrca. It does not record the original Intel acquisition revision, selection
or split recipe, transformations, or row-level mapping. The Intel revision above
is a verified **reference snapshot**, not a claim that the historical import used
that revision. The two snapshot IDs belong to different repositories and must
not be substituted for one another. Reproducing or expanding the samples
requires independently establishing that missing derivation chain and reviewing
the applicable source-data terms; the pin alone does not establish it.

## License notices

Retain [Apache License 2.0](APACHE-2.0.txt), declared by the pinned Intel card,
with material obtained from that dataset. The verified Intel tree contains no
standalone LICENSE or NOTICE file; no Intel copyright holder or additional
upstream notice has been invented here. The card's attribution is retained above.

The pre-existing [Microsoft MIT notice](LICENSE.md) is retained for applicable
Microsoft material. It does **not** relicense Intel/OpenOrca data or replace
upstream terms. This notice does not establish rights for every underlying
example or resolve the historical provenance limitation.
