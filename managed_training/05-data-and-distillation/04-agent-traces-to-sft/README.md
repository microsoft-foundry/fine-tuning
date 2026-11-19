# Agent Traces to SFT

[notebooks/demo.ipynb](notebooks/demo.ipynb) trains only on explicitly prepared,
source-traceable, conversation-disjoint data. The originals under [data/](data/)
remain unchanged for provenance, not as an approved evaluation split.

## Preserved input, not a validated split

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `traces-clean.jsonl` | 100 | `81083aadd6464293b6acfc9356216c6c685a1ba99ae3f6fe34ba9d7b5b864d71` |
| `train.jsonl` | 80 | `1371b81c70d7f8c20f8eff604932670a3eda8ca9437759264dcfce4c06aeeae3` |
| `val.jsonl` | 10 | `865ab59e51c81a29ecc89a0bde347a22d0929630f7cd3db1ffb12f1d406034fa` |
| `test.jsonl` | 10 | `7ea4284e67980cc2db8cfb2a4b5d403da4383bec9821be3fea27316cbd677040` |

The offline audit verified the source counts and hashes. The historical seed-42
split is row-level: there are 44 cross-split conversation-overlap pairs and 35
normalized message-prefix pairs across all three splits. Exact
row/byte inequality is not sufficient: generated function-call IDs differ
between snapshots of the same conversation.

The original policy also permits inventory checks only for exchanges, although
the recorded supervision includes replacement stock checks. See the explicit
policy revision below. No prepared JSONL artifacts are bundled; offline tests
generate and verify them in temporary directories, and operators prepare their
own separate versioned outputs before upload.

## Explicit offline preparation

[trace_integrity.py](trace_integrity.py) is a standalone standard-library module;
it performs no hosted calls and does not import Azure clients. From the repository
root, run it with Python using `prepare --output-dir
managed_training/05-data-and-distillation/04-agent-traces-to-sft/outputs/prepared`.
Use `audit` for a read-only report of original split leakage, and `check` to
verify the prepared directory. `audit` exits unsuccessfully if leakage exists.
Preparation refuses an existing output directory instead of overwriting it.

The input fixture byte hashes are preserved by this module's
[.gitattributes](.gitattributes); the track already preserves JSONL bytes.
A source hash mismatch is an error to investigate, not permission to replace
the expected digest with the current bytes.

Preparation:

1. Requires the exact original hashes/counts and fixture agreement.
2. Retains **every source trace**, including shorter snapshots. No new replies,
	answers, tool results or conversations are invented; none are dropped.
3. Changes only the system/tool **descriptions**: `check_inventory` applies to
	**exchanges and replacements**, including lost-package replacements, before
	calculating that action. Refund-only resolutions do not require stock.
	The action examples now include `replacement`. Tool names, parameter schemas,
	eligibility/refund rules and all recorded calls/results stay unchanged.
	This is an explicit cookbook policy amendment, not a claim about the original
	agent or a newly tested live tool implementation.
4. Alpha-renames generated function-call IDs and matching replies for identity,
	canonicalizes JSON arguments/results, and excludes the common system prompt.
	It does **not** remove order IDs, SKU values or ordinary message text. Rows
	sharing their first user request are conservatively one conversation even
	when assistant continuations differ. Equal/prefix normalized non-system
	message sequences are joined into the same transitive group as well.
5. Assigns whole groups deterministically using seed-42 SHA-256 ordering and
	greedy 80/10/10 row **targets**, requiring nonempty train/validation/test.
	Group boundaries take priority: exact output counts are computed, not fixed.
	Too few independent groups is a hard error, never a row-level fallback.
6. Writes separate prepared data/fixtures and a manifest with actual counts,
	byte hashes, original leakage pairs, group identity and every source-line/hash
	to output split/line/hash mapping. The manifest is written last.

The notebook independently rebuilds this preparation and checks exact artifacts,
manifest and conversation/prefix disjointness read-only before upload. Missing,
partial, altered or stale preparation fails closed. It rechecks immediately
before hosted calls. Only prepared train/validation files are uploaded; prepared
test rows are retained solely for lineage. The historical 80/10/10 table must
not be reused as prepared output counts.

Run the offline tests under [tests/](tests/) and inspect the emitted manifest
before using outputs. Offline preparation tests passed: 100 source traces form
40 groups and produce disjoint 80/10/10 splits with no source trace lost. Exact
prepared hashes are recorded in [validation-status.json](validation-status.json).
No hosted job, agent call, or tool execution was performed.

The catalog and demo metadata describe this offline preparation contract.
These checks do not establish live service eligibility or task-quality results.

The source student example is `gpt-4.1-nano-2025-04-14`. The retained supervised
recipe is 3 epochs, learning-rate multiplier 1.0, batch size 1 and
`GlobalStandard`. No agent deployment, agent calls or trace generation is
needed to run the bundled training contract.

## Run

1. Install the central environment by following [`GETTING_STARTED.md`](../../GETTING_STARTED.md), then authenticate with `az login` or another `DefaultAzureCredential` source.
2. Copy `.env.template` to this module's ignored `.env` and configure your own `FOUNDRY_PROJECT_ENDPOINT` and available `FOUNDRY_MODEL`.
3. Complete explicit offline preparation and tests above. Open the notebook from this module or its `notebooks` directory. Its validation is read-only; later upload/training cells can incur charges.

The notebook includes the entire submission workflow and requires no runner
or project resource ID. It submits or resumes one job, fails explicitly on
terminal service errors/timeouts, and downloads service loss/token-accuracy
metrics only. No fine-tuned inference or base-model comparison is performed.
Creation POSTs disable SDK retries; an ambiguous response requires reconciliation
instead of blind resubmission. Recorded pending jobs are monitored without
creating another job, and authentication/service errors propagate.
Ignored `outputs/<run-id>` stores local state and metrics; a blank
`FOUNDRY_RUN_ID` resumes `outputs/default`. See [track configuration](../README.md).

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
