# Historical trace-distillation source data

The original manifest records 100 transformed traces and a seed-42 row-level
80/10/10 split. These immutable files are **provenance input**, not an approved
evaluation split: differing generated call IDs can conceal the same conversation,
and shorter snapshots can be prefixes of training conversations.
[SOURCE_DATA_SHA256SUMS.txt](SOURCE_DATA_SHA256SUMS.txt) records the original
JSONL raw hashes and CRLF-to-LF text-fixture hashes; they must not be relabelled
as corrected-data hashes. Historical CRLF fixture archive digests are retained
separately. Text-fixture canonicalization does not change embedded trace text.

The original prompt/tool fixtures also say inventory checks are only for
exchanges, while retained replacement examples call that tool. The fixtures here
are preserved as historical evidence. The explicit
[preparation module](../trace_integrity.py) changes the system/tool descriptions
together in separate outputs to permit exchange **or replacement** stock checks,
without changing recorded actions/results or inventing supervision.

The original trace-generation workflow requests
approximately 100 traces from the configured source agent/time window using
the Foundry Data Generation API. Its transformation:

1. Removes overlapping snapshots by first-occurrence message deduplication using role, content, tool-call ID, and function-call identity.
2. Discards fragments without assistant tool-call supervision.
3. Merges adjacent assistant tool-call turns and omits assistant tool-call content.
4. Restores missing function-call IDs deterministically as `call_trace_<row>_<message>_<call>`, associating tool replies in order.
5. Restores the exact Zava system prompt, tool catalog and `parallel_tool_calls=true`.

Historically, the resulting rows were serialized unchanged. Python `random.Random(42)`
shuffles row indices, with the first 80 going to `train.jsonl`, the next 10 to
`val.jsonl`, and the final 10 to `test.jsonl`. This is retained only to explain
the original assets. The notebook no longer accepts this row split for upload.
It requires separately prepared whole-conversation groups, normalization of call
IDs and explicit prefix-disjointness checks, described in the
[module README](../README.md). All source traces are retained in that preparation,
but their split memberships change. Offline preparation tests verified 40 groups
and an 80/10/10 split with no cross-split conversation or message-prefix overlap.
Measured hashes are in [validation-status.json](../validation-status.json).

The source implementation can skip orphan tool-response rows during
transformation; that historical recipe is not a new fallback in the cookbook.
Only the already transformed, hash-guarded data are preparation inputs here.
No hosted agent calls or fresh generation are needed. Original JSONL files are
not rewritten, and historical agent behavior is not certified by the policy
amendment. Tool schemas/recorded calls establish data shape, not live eligibility
or end-to-end policy correctness.

The preserved supervised training recipe is 3 epochs, learning-rate multiplier
1.0 and batch size 1 with `GlobalStandard`.
