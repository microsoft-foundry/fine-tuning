# External retail tool server contract

## What is and is not supplied

This capstone supplies preserved synthetic retail data, function definitions,
an [OpenAPI contract](configs/retail-openapi-with-policy.json), and an
[authenticated client](src/retail_tools.py). **It does not supply or deploy a
retail server or a complete return-policy engine.** The nearby
[catalog server](../01-agentic-rft-tools/tools/catalog_server.py) demonstrates the
function-call envelope and bearer convention, but implements only
`search_catalog`. It cannot serve this capstone's retail tools.

The two RFT-only operations are required, not optional examples:

| Operation | Exact request contract |
| --- | --- |
| `list_user_orders` | Required string `user_id`; optional ISO-date strings `from_date`, `to_date`; optional integer `limit` (default 10). |
| `policy_verify_return` | Required string `order_id`, array of string `item_ids`, and `reason` in `unwanted`, `wrong_item`, `size_issue`, `defective`, `other`. |

The active OpenAPI parameter schemas match every declaration in the preserved
RFT training and validation rows. All 20 RFT rows declare `policy_verify_return`
and contain policy grading arguments; 19 reference call sequences name it
(validation scenario 7 instead contains mutations). Fourteen reference
sequences name `list_user_orders`. The local validator
checks every configured tool and every dataset declaration/reference against
the published operations before a service stage. These checks establish
contract coverage, **not the existence of an external implementation**.

## Customer-owned implementation and isolation

Before running a remote stage, supply an approved HTTPS dispatcher implementing
**all 17 configured operations**, including the original mutations. Back it
with a separate, disposable copy of [the synthetic fixture](data/retail-database.json),
not production data or customer accounts. Only then acknowledge
`FOUNDRY_TOOL_SERVER_IS_SANDBOX=true`. No server, data store, permission, or
deployment is created by the notebook.

The external implementation must enforce the
[original retail policy](data/retail-policy.md), including:

- Locate the customer through email or name plus zip even if a user ID was
  supplied. A tool-server bearer secret authenticates the calling application;
  it is **not** customer authentication or proof of order ownership.
- Serve only that customer in a conversation. Verify user/order ownership on
  every relevant read and write; do not treat a caller-supplied `user_id`,
  `order_id`, or arbitrary `trace_id` as an authorization grant. Define trusted
  conversation isolation at the approved server/gateway. The supplied envelope
  does not by itself establish an authenticated retail session.
- Reject unknown identifiers, invalid dates, reversed date ranges, invalid
  limits, and missing/extra item occurrences rather than inventing records or
  silently coercing values. Document date boundaries, ordering, and the
  deterministic clock used for historical fixture evaluation. Duplicate item
  IDs refer to actual ordered occurrences, not unlimited return quantities.
- Require delivered status for returns, explicit confirmation before any
  mutation, and refunds only to the original payment method or an existing
  gift card. Preserve pending-order, same-product replacement, payment balance,
  and once-per-order mutation restrictions. A policy-verification response
  never constitutes confirmation or a completed return/refund.

The RFT reference outcomes mention 30-day windows, electronics rules, VIP
extensions, marketplace redirects, opened items, fees, and abuse-related store
credit. The preserved generic policy and database do **not** fully specify those
rules, fee calculations, all necessary item-condition facts, or an evaluation
clock. Some grading-only `reference_policy_args.reason` values also differ from
the declared API enum (for example, `size issue`, `exchange`, and `wrong_size`).
Do not broaden that enum, infer missing facts, hard-code reference outcomes as
tool results, or invent a policy engine to make the comparison pass. The server
owner must provide and qualify the complete policy/data contract and reconcile
these label inconsistencies separately; this change does not rewrite datasets
or promise that the historical policy labels are reproducible.

## Authentication and HTTP envelope

Inject the same high-entropy `FOUNDRY_TOOL_AUTH_TOKEN` from an approved secret
store into the hosted server and the notebook environment or ignored local
environment file. Require `Authorization: Bearer <secret>` on the server;
compare credentials securely and reject missing/incorrect credentials before
dispatch. Unlike the catalog sample's optional local protection, this
capstone's remote stages require a token and have no anonymous fallback.
Never put credentials in URLs, committed configs, notebook outputs, or HTTP/SDK
wire-debug logs. URLs must use HTTPS without credentials, query, or fragment.

The checked-in [remote configuration](configs/rft-tools.json) intentionally
retains its original empty header objects. The notebook builds **every** live
entry with the same bearer header in memory; it does not write secrets into the
preserved configuration or its lineage manifest.

The configured `FOUNDRY_TOOLS_SERVER_URL` is a dispatcher URL, not an operation
base URL. Both the training service and local comparison POST a JSON object
containing `type: function_call`, `id`, `call_id`, `name`, JSON-string
`arguments`, and `trace_id` to that exact URL. The response is a JSON object
with `type: function_call_output`, the corresponding `id` and `call_id`, and
`output` (a string or a JSON value). The operation paths in OpenAPI describe
the function contracts; the client does not append those paths to the dispatcher.
Keep secrets out of response bodies and errors too.

The local client generates one `trace_id` for an entire preflight or model
conversation and reuses it for every operation, model turn, and read retry.
Separate preflights, scenarios, and deployment comparisons receive fresh IDs;
tool `call_id` values still identify individual calls. The transport requires
the owning caller to pass this trace explicitly instead of silently generating
one per request. Bind this correlation to trusted conversation state at the
approved gateway; it is not itself customer authentication or an ownership
grant. The comparison context versions this trace scope so results produced
with the old per-call behavior are not silently reused.

The notebook's authenticated preflight uses only four read-only calls for one
fixture identity: `find_user_id_by_email`, `list_user_orders`,
`get_order_details`, and `policy_verify_return`. It checks the output envelope,
identity match, nonempty JSON results, and explicit tool errors before uploads
or RFT comparison. A generic success object is not evidence of correct policy
semantics; independently qualify ownership, item accounting, dates, reason
validation, policy boundaries, fees, and mutation behavior at the server.
Preflight never creates a customer, changes an order, or tests a refund.
Redirects are refused, including on authentication failures. Local comparison
permits only the read-only allowlist; if a model proposes any unknown or
mutating call in a batch, none of that batch is sent.

## Private state and recovery

All generated state/results and temporary state replacements belong under
the locally ignored outputs directory. The notebook saves only redacted
methods and an auth fingerprint for reuse equivalence, never a bearer token or
secret-bearing request payload. RFT comparison equivalence also includes the
server URL and active API hash. Changing the token or server invalidates reuse;
old state formats fail closed. Preserve and reconcile existing files/jobs and
pending requests before deliberately archiving state for a new run. Never
delete state merely to bypass a pending POST or a credential mismatch.

Submission and tool-transport errors deliberately omit underlying HTTP/service
text that might echo a secret. Keep wire-debug logging off. If an older local
state file or log already contains a credential, treat it as exposed, rotate
the credential, and reconcile the associated jobs; new redaction does not
retroactively sanitize old files. Never add populated runtime state to Git.

The workflow retains one paid-resource warning followed by execution of the
selected stage, not another budget approval gate. Neither local contract checks
nor the offline tests submit jobs or invoke a remote tool server.