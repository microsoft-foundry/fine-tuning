"""Parent-run offline regressions for immutable provenance and prepared-only use."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "sft-train.jsonl", "sft-validation.jsonl", "next-tool-evaluation.jsonl",
    "rft-train.jsonl", "rft-validation.jsonl",
    "synthetic-source-train.jsonl", "synthetic-source-validation.jsonl",
)
ACTIVE_DATASETS = DATASETS[:5]
KNOWN_RAW_FINDINGS = {
    ("sft-train.jsonl", 16): "payment.unobserved-method",
    ("sft-train.jsonl", 111): "payment.unobserved-method",
    ("sft-train.jsonl", 200): "authentication.required-first",
    ("sft-validation.jsonl", 54): "authentication.required-first",
    ("next-tool-evaluation.jsonl", 10): "payment.unobserved-method",
    ("next-tool-evaluation.jsonl", 77): "payment.unobserved-method",
    ("next-tool-evaluation.jsonl", 99): "payment.unobserved-method",
    ("next-tool-evaluation.jsonl", 162): "authentication.required-first",
    ("next-tool-evaluation.jsonl", 180): "payment.unobserved-method",
    ("synthetic-source-train.jsonl", 76): "authentication.required-first",
    ("synthetic-source-train.jsonl", 134): "authentication.required-first",
    ("synthetic-source-train.jsonl", 153): "authentication.required-first",
}
REVIEWED_EVALUATION_PAYMENTS = (
    (10, "call_82056c5fb8984cb58d28e4d2", "call_f051f4d829d14c4b80f5295e", "credit_card_1034663"),
    (77, "call_b41864d72a774a818b4dbdc3", "call_585bd161ed704eae8f635736", "paypal_6651356"),
    (99, "call_379037066c31444486ee6c30", "call_2276c4bc41a24d5b9ca7f657", "credit_card_3577130"),
    (180, "call_4efdf36d7ffc4c8ab1b2b12f", "call_6d4771a4e703464890f0abc6", "paypal_1191071"),
)


@pytest.fixture(scope="module")
def retail():
    spec = importlib.util.spec_from_file_location(
        "retail_trace_integrity_under_test", ROOT / "src" / "retail_tools.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset_report(retail):
    return retail.audit_retail_datasets(ROOT)


@pytest.fixture(scope="module")
def curation(retail):
    # Resolve the sibling import to the exact module under test, not another checkout.
    spec = importlib.util.spec_from_file_location(
        "retail_reviewed_curation_under_test", ROOT / "src" / "retail_curation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "retail_tools", retail)
        spec.loader.exec_module(module)
    return module


def immutable_inputs(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for directory in ("data", "configs", "graders")
            for path in (root / directory).rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def prepared(curation, tmp_path_factory):
    root = tmp_path_factory.mktemp("retail-prepared-traces")
    for directory in ("data", "configs", "graders"):
        shutil.copytree(ROOT / directory, root / directory)
    before = immutable_inputs(root)
    paths, report = curation.prepare_active_datasets(root)
    assert immutable_inputs(root) == before
    assert immutable_inputs(ROOT) == before
    return root, paths, report


def call(name, arguments=None, identifier="fixture-call"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {})},
    }]}


def reply(identifier, result):
    return {"role": "tool", "tool_call_id": identifier,
            "content": result if isinstance(result, str) else json.dumps(result)}


def authenticated(*, object_result=False):
    result = {"user_id": "fixture_a", "email": "a@example.invalid"} if object_result else "fixture_a"
    return [
        {"role": "user", "content": "My email is a@example.invalid."},
        call("find_user_id_by_email", {"email": "a@example.invalid"}, "lookup"),
        reply("lookup", result),
    ]


def observed_profile():
    return authenticated() + [
        call("get_user_details", {"user_id": "fixture_a"}, "profile"),
        reply("profile", {
            "user_id": "fixture_a", "email": "a@example.invalid", "orders": ["#fixture-order"],
            "payment_methods": {"gift_card_fixture": {"id": "gift_card_fixture", "balance": 37}},
        }),
    ]


def action(method="gift_card_fixture", order="#fixture-order", **extra):
    return call("exchange_delivered_order_items", {
        "order_id": order, "item_ids": ["old-fixture"], "new_item_ids": ["new-fixture"],
        "payment_method_id": method, **extra,
    }, "action")


def codes(retail, messages):
    return {problem["code"] for problem in retail.retail_trace_diagnostics({"messages": messages})}


@pytest.mark.parametrize("name,arguments", [
    ("list_all_product_types", {}), ("get_product_details", {"product_id": "fixture-product"}),
    ("get_order_details", {"order_id": "#fixture-order"}),
    ("get_user_details", {"user_id": "fixture_a"}),
])
def test_catalog_product_order_and_claimed_user_id_require_prior_authentication(retail, name, arguments):
    messages = [{"role": "user", "content": "My user ID is fixture_a."}, call(name, arguments)]
    assert "authentication.required-first" in codes(retail, messages)


@pytest.mark.parametrize("prefix", [
    [{"role": "assistant", "content": "I have authenticated your account."}],
    [call("find_user_id_by_email", {"email": "a@example.invalid"}, "lookup")],
    [call("find_user_id_by_email", {"email": "a@example.invalid"}, "lookup"),
     reply("lookup", {"error": "Customer not found"})],
    [call("find_user_id_by_email", {"email": "a@example.invalid"}, "lookup"),
     reply("different-call", "fixture_a")],
])
def test_claims_pending_failed_and_unmatched_lookups_never_authenticate(retail, prefix):
    assert "authentication.required-first" in codes(retail, [*prefix, call("list_all_product_types")])


def test_declarations_bearer_and_rft_reference_names_are_not_observed_authentication(retail):
    row = {
        "messages": [call("list_all_product_types")],
        "tools": [call("find_user_id_by_email")["tool_calls"][0]],
        "headers": {"Authorization": "Bearer test-only-not-a-customer"},
        "reference_tool_calls": ["find_user_id_by_email", "list_all_product_types"],
    }
    assert "authentication.required-first" in {p["code"] for p in retail.retail_trace_diagnostics(row)}


@pytest.mark.parametrize("name,arguments,result", [
    ("find_user_id_by_email", {"email": "a@example.invalid"}, "fixture_a"),
    ("find_user_id_by_email", {"email": "A@example.invalid"},
     {"user_id": "fixture_a", "email": "a@example.invalid"}),
    ("find_user_id_by_name_zip", {"first_name": "Unit", "last_name": "Fixture", "zip": "00001"},
     "fixture_a"),
    ("find_user_id_by_name_zip", {"first_name": "Unit", "last_name": "Fixture", "zip": "00001"},
     {"user_id": "fixture_a", "name": {"first_name": "Unit", "last_name": "Fixture"},
      "address": {"zip": "00001"}}),
    ("find_user_id_by_name_zip", {"first_name": "Unit", "last_name": "Fixture", "zip": "00001"},
     {"user_id": "fixture_a", "name": "Unit Fixture", "zip": "00001"}),
])
def test_matched_successful_lookup_allows_a_final_catalog_prediction_without_a_result(
    retail, name, arguments, result,
):
    messages = [call(name, arguments, "lookup"), reply("lookup", result), call("list_all_product_types")]
    assert retail.retail_trace_diagnostics({"messages": messages}) == []


def test_failed_lookup_can_retry_and_no_customer_context_survives_between_rows(retail):
    messages = [
        call("find_user_id_by_email", {"email": "missing@example.invalid"}, "failed"),
        reply("failed", {"success": False}), *authenticated(), call("list_all_product_types"),
    ]
    assert retail.retail_trace_diagnostics({"messages": messages}) == []
    assert "authentication.required-first" in codes(retail, [call("list_all_product_types")])


@pytest.mark.parametrize("result", [
    {"user_id": "fixture_a", "email": "different@example.invalid"},
    {"user_id": "fixture_a", "email": []},
])
def test_contradictory_or_malformed_lookup_identity_does_not_authenticate(retail, result):
    messages = authenticated()
    messages[-1] = reply("lookup", result)
    problems = codes(retail, [*messages, call("list_all_product_types")])
    assert {"authentication.identity-mismatch", "authentication.required-first"} <= problems


def test_second_customer_and_foreign_profile_cannot_supply_payment_evidence(retail):
    messages = authenticated() + [
        call("find_user_id_by_email", {"email": "b@example.invalid"}, "other-lookup"),
        reply("other-lookup", {"user_id": "fixture_b", "email": "b@example.invalid"}),
        call("get_user_details", {"user_id": "fixture_a"}, "profile"),
        reply("profile", {"user_id": "fixture_b", "orders": ["#fixture-order"],
                          "payment_methods": {"gift_card_fixture": {}}}),
        action(),
    ]
    assert {"authentication.customer-switch", "authentication.foreign-result",
            "payment.unobserved-method", "authentication.unproven-action-owner"} <= codes(retail, messages)


def test_pre_authentication_profile_reply_cannot_be_laundered_through_a_later_lookup(retail):
    profile = call("get_user_details", {"user_id": "fixture_a"}, "profile")["tool_calls"][0]
    lookup = call("find_user_id_by_email", {"email": "a@example.invalid"}, "lookup")["tool_calls"][0]
    messages = [
        {"role": "assistant", "tool_calls": [lookup, profile]}, reply("lookup", "fixture_a"),
        reply("profile", {"user_id": "fixture_a", "orders": ["#fixture-order"],
                          "payment_methods": {"gift_card_fixture": {}}}), action(),
    ]
    assert {"authentication.required-first", "payment.unobserved-method"} <= codes(retail, messages)


def test_every_action_payment_is_observed_in_the_correct_prior_customer_context(retail):
    assert retail.retail_trace_diagnostics({"messages": [*observed_profile(), action()]}) == []
    assert "payment.unobserved-method" in codes(retail, [*observed_profile(), action("credit_card_0000000")])
    assert "authentication.unproven-action-owner" in codes(retail, [*observed_profile(), action(order="#other-order")])
    assert "authentication.foreign-user" in codes(retail, [*observed_profile(), action(user_id="fixture_b")])
    # A method seen in a different record cannot leak into a new record.
    assert "payment.unobserved-method" in codes(retail, [*authenticated(), action()])


@pytest.mark.parametrize("field", ["id", "payment_method_id"])
def test_observed_profile_list_supports_both_existing_method_identifier_shapes(retail, field):
    messages = observed_profile()
    profile = json.loads(messages[-1]["content"])
    profile["payment_methods"] = [{field: "gift_card_fixture", "type": "gift_card"}]
    messages[-1] = reply("profile", profile)
    assert retail.retail_trace_diagnostics({"messages": [*messages, action()]}) == []


def test_failed_action_result_never_excuses_an_invented_payment_argument(retail):
    messages = [*observed_profile(), action("credit_card_0000000"),
                reply("action", "Error: payment method not found")]
    assert "payment.unobserved-method" in codes(retail, messages)


def test_original_order_payment_history_is_grounding_only_for_an_owned_matching_order(retail):
    messages = authenticated() + [
        call("get_order_details", {"order_id": "#fixture-order"}, "order"),
        reply("order", {"order_id": "#fixture-order", "user_id": "fixture_a",
                        "payment_history": [{"payment_method_id": "gift_card_fixture"}]}), action(),
    ]
    assert retail.retail_trace_diagnostics({"messages": messages}) == []
    for changes, expected in [
        ({"user_id": "fixture_b"}, "authentication.foreign-result"),
        ({"order_id": "#other-order"}, "authentication.order-mismatch"),
        ({"user_id": None}, "authentication.unproven-order-owner"),
    ]:
        altered = copy.deepcopy(messages)
        result = json.loads(altered[-2]["content"])
        result.update(changes)
        altered[-2] = reply("order", result)
        assert {expected, "payment.unobserved-method"} <= codes(retail, altered)


def test_future_results_and_foreign_user_requests_are_not_payment_evidence(retail):
    messages = [*authenticated(), action(), reply("action", {"success": False}), *observed_profile()[3:]]
    assert "payment.unobserved-method" in codes(retail, messages)
    messages = authenticated() + [
        call("get_user_details", {"user_id": "fixture_b"}, "profile"),
        reply("profile", {"payment_methods": {"gift_card_fixture": {}}, "orders": ["#fixture-order"]}),
        action(),
    ]
    assert {"authentication.foreign-user", "payment.unobserved-method"} <= codes(retail, messages)


def test_final_target_exception_never_allows_partial_or_interrupted_reply_batches(retail):
    first = call("list_all_product_types", identifier="first")["tool_calls"][0]
    second = call("calculate", {"expression": "2+2"}, "second")["tool_calls"][0]
    batch = {"role": "assistant", "tool_calls": [first, second]}
    messages = [*authenticated(), batch, reply("first", {})]
    assert "tool.missing-reply" in codes(retail, messages)
    assert "tool.missing-reply" in codes(retail, [*messages, call("get_product_details")])
    assert retail.retail_trace_diagnostics({"messages": [*messages, reply("second", 4)]}) == []


def test_wrapped_next_tool_prediction_is_exactly_one_call_and_never_gets_a_fake_reply(retail):
    item = {"messages": observed_profile(), "expected_output": action()}
    before = copy.deepcopy(item)
    assert retail.retail_trace_diagnostics({"item": item}) == []
    assert item == before
    item["expected_output"]["tool_calls"].append(copy.deepcopy(item["expected_output"]["tool_calls"][0]))
    assert retail.retail_trace_diagnostics({"item": item})[0]["code"] == "prediction.one-call-required"
    item["expected_output"] = None
    assert retail.retail_trace_diagnostics({"item": item})[0]["code"] == "prediction.one-call-required"


def test_rft_references_check_order_not_fabricated_executions_or_policy_outcomes(retail):
    row = {"messages": [{"role": "user", "content": "Find my order."}],
           "reference_tool_calls": ["find_user_id_by_email", "list_user_orders"],
           "reference_policy_outcome": [{"eligibility": True}]}
    assert retail.retail_trace_diagnostics(row) == []
    row["reference_tool_calls"].reverse()
    assert retail.retail_trace_diagnostics(row)[0]["code"] == "authentication.reference-required-first"


@pytest.mark.parametrize("row", [
    [], {"messages": None}, {"messages": [None]},
    {"messages": [reply([], "fixture_a")]},
    {"messages": [*observed_profile(), action([])]},
    {"messages": [*observed_profile(), action(order=[])]},
])
def test_malformed_records_produce_diagnostics_instead_of_type_errors(retail, row):
    assert retail.retail_trace_diagnostics(row)


def independent_first_tool_violations(row):
    """Small ordering oracle: declarations/prose/references cannot grant access."""
    item = row.get("item", row)
    messages = list(item["messages"])
    if "expected_output" in item:
        messages.append(item["expected_output"])
    lookups = {"find_user_id_by_email", "find_user_id_by_name_zip"}
    pending = {}
    authenticated_customer = False
    failures = []
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            if pending.pop(message.get("tool_call_id"), None) not in lookups:
                continue
            result = message.get("content")
            try:
                result = json.loads(result)
            except (TypeError, ValueError):
                pass
            if isinstance(result, dict):
                if "error" in result or result.get("success") is False:
                    continue
                result = result.get("user_id")
            if isinstance(result, str) and result and not result.lower().startswith("error"):
                authenticated_customer = True
            continue
        for target in message.get("tool_calls", []):
            name = target["function"]["name"]
            if name not in lookups and not authenticated_customer:
                failures.append(index)
            pending[target["id"]] = name
    return failures


def independent_payment_violations(row):
    """Separate chronological oracle; no production audit/curation helpers."""
    item = row.get("item", row)
    messages = list(item["messages"])
    if "expected_output" in item:
        messages.append(item["expected_output"])
    pending = {}
    customer = None
    methods, orders, failures = set(), set(), []
    for index, message in enumerate(messages):
        if message["role"] != "tool":
            pending.clear()
            for target in message.get("tool_calls", []):
                name = target["function"]["name"]
                arguments = json.loads(target["function"]["arguments"])
                if "payment_method_id" in arguments and (
                    customer is None or arguments["payment_method_id"] not in methods
                    or arguments.get("order_id") not in orders
                    or arguments.get("user_id", customer) != customer
                ):
                    failures.append(index)
                pending[target["id"]] = (name, arguments, customer)
            continue
        observed = pending.pop(message.get("tool_call_id"), None)
        if observed is None:
            continue
        name, arguments, customer_at_call = observed
        result = message.get("content")
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            pass
        if isinstance(result, dict) and ("error" in result or result.get("success") is False):
            continue
        if name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
            identity = result.get("user_id") if isinstance(result, dict) else result
            if isinstance(identity, str) and identity and not identity.lower().startswith("error"):
                if customer is None:
                    customer = identity
            continue
        if (
            customer_at_call is None or customer_at_call != customer or not isinstance(result, dict)
            or arguments.get("user_id", customer) != customer or result.get("user_id", customer) != customer
        ):
            continue
        if name == "get_user_details":
            orders.update(order for order in result.get("orders", []) if isinstance(order, str))
            payment_methods = result.get("payment_methods", {})
            if isinstance(payment_methods, dict):
                methods.update(payment_methods)
            else:
                methods.update(method.get("id", method.get("payment_method_id"))
                               for method in payment_methods if isinstance(method, dict))
        elif name == "get_order_details":
            order = arguments["order_id"]
            if result.get("order_id", order) != order:
                continue
            if result.get("user_id") != customer and order not in orders:
                continue
            orders.add(order)
            methods.update(payment["payment_method_id"] for payment in result.get("payment_history", [])
                           if isinstance(payment, dict) and "payment_method_id" in payment)
    return failures


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_wide_audit_covers_every_row_and_independent_authentication_order(retail, dataset_report, name):
    # The original synthetic files remain evidence, not trusted training inputs.
    # This test does not waive their findings or assert that all source rows pass.
    assert set(dataset_report) == set(DATASETS)
    observed = {}
    for number, line in enumerate((ROOT / "data" / name).read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        problems = retail.retail_trace_diagnostics(row)
        if problems:
            observed[number] = problems
        independent = independent_first_tool_violations(row)
        reported = {problem["message_index"] for problem in problems
                    if problem["code"] == "authentication.required-first"}
        assert set(independent) <= reported, (name, number, independent, problems)
    flattened = [{"row": number, **problem} for number, problems in observed.items() for problem in problems]
    assert dataset_report[name] == flattened


@pytest.mark.parametrize("name", ACTIVE_DATASETS)
def test_every_prepared_active_row_passes_production_and_independent_oracles(retail, prepared, name):
    root, paths, report = prepared
    assert retail.audit_retail_datasets(root, dataset_paths=paths)[name] == []
    entry = report["files"][f"data/{name}"]
    rows = [json.loads(line) for line in paths[name].read_bytes().splitlines()]
    assert len(rows) == entry["source"]["records"] == entry["output"]["records"]
    for number, row in enumerate(rows, 1):
        assert retail.retail_trace_diagnostics(row) == [], (name, number)
        assert independent_first_tool_violations(row) == [], (name, number)
        assert independent_payment_violations(row) == [], (name, number)
        if name == "next-tool-evaluation.jsonl":
            assert len(row["item"]["expected_output"]["tool_calls"]) == 1


def test_raw_findings_are_reported_in_full_not_waived_or_promoted(dataset_report, prepared):
    _, paths, report = prepared
    assert set(paths) == set(ACTIVE_DATASETS)
    for name in DATASETS:
        entry = report["files"][f"data/{name}"]
        assert entry["source_diagnostics"] == dataset_report[name]
        if name not in ACTIVE_DATASETS:
            assert entry["role"] == "preserved-source-evidence-not-active"
            assert entry["output"] is None and entry["output_diagnostics"] is None
            assert entry["curation"] == [] and name not in paths
    for (name, number), code in KNOWN_RAW_FINDINGS.items():
        assert any(problem["row"] == number and problem["code"] == code
                   for problem in dataset_report[name]), (name, number, code)
        original = read_row(name, number)
        oracle = independent_payment_violations if code.startswith("payment.") else independent_first_tool_violations
        assert oracle(original), (name, number)


def test_independent_payment_oracle_detects_future_foreign_and_ungrounded_evidence():
    assert independent_payment_violations({"messages": [*observed_profile(), action()]}) == []
    assert independent_payment_violations({"messages": [*observed_profile(), action("credit_card_0000000")]})
    assert independent_payment_violations({"messages": [*observed_profile(), action(order="#foreign")]})
    assert independent_payment_violations({"messages": [*observed_profile(), action(user_id="fixture_b")]})
    assert independent_payment_violations({"messages": [*authenticated(), action(), *observed_profile()[3:]]})


@pytest.mark.parametrize("unsafe_name", ACTIVE_DATASETS)
@pytest.mark.parametrize("violation,expected_code", [
    ("authentication", "authentication.required-first"),
    ("payment", "payment.unobserved-method"),
    ("order-owner", "authentication.unproven-action-owner"),
    ("foreign-user", "authentication.foreign-user"),
])
def test_local_preparation_rejects_unsafe_active_trace_before_any_live_stage(
    retail, tmp_path, unsafe_name, violation, expected_code,
):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "data").mkdir()
    messages = [call("list_all_product_types")]
    if violation != "authentication":
        mutation = {
            "payment": action("credit_card_unobserved_fixture"),
            "order-owner": action(order="#foreign-fixture-order"),
            "foreign-user": action(user_id="fixture_b"),
        }[violation]
        messages = [*observed_profile(), mutation]
    row = {"messages": messages, "tools": []}
    directory = tmp_path / "outputs" / "prepared-data"
    directory.mkdir(parents=True)
    paths = {name: directory / name for name in ACTIVE_DATASETS}
    for name, path in paths.items():
        value = row if name == unsafe_name else {"messages": [], "tools": []}
        path.write_bytes((json.dumps(value) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match=f"Retail trace validation failed.*{expected_code}"):
        retail.validate_local_tool_contract(tmp_path, dataset_paths=paths)


def test_raw_sources_can_never_be_selected_as_active_by_contract_validation(retail):
    with pytest.raises(ValueError, match="Explicit prepared active dataset paths"):
        retail.validate_local_tool_contract(ROOT)
    with pytest.raises(ValueError, match="under outputs/prepared-data"):
        retail.validate_local_tool_contract(ROOT, dataset_paths={
            name: ROOT / "data" / name for name in ACTIVE_DATASETS
        })


def read_row(name, number):
    with (ROOT / "data" / name).open(encoding="utf-8") as stream:
        return next(json.loads(line) for index, line in enumerate(stream, 1) if index == number)


@pytest.mark.parametrize("name,number,lookup_id,catalog_id", [
    ("sft-train.jsonl", 200, "call_3fb27ba67f3240c48a982c95", "call_5805cd701ac84b60a7c7d2a1"),
    ("sft-validation.jsonl", 54, "call_349fb7e2c7bf40eea73e227b", "call_1d2d61c31ff64373b418873b"),
    ("next-tool-evaluation.jsonl", 162, "call_786d9ae8758549e9abc4d0bc", "call_69f39243f05e4b7db95902bd"),
])
def test_reviewed_authentication_curation_reuses_real_credentials_and_results_only(
    curation, retail, name, number, lookup_id, catalog_id,
):
    original = read_row(name, number)
    unchanged = copy.deepcopy(original)
    curated, decisions = curation.curate_reviewed_row(original)
    assert original == unchanged
    before = original.get("item", original)
    after = curated.get("item", curated)
    assert decisions == ["authentication-before-product-access"]
    assert independent_first_tool_violations(original)
    assert independent_first_tool_violations(curated) == []
    assert retail.retail_trace_diagnostics(curated) == []
    assert after.get("tools") == before.get("tools")
    assert after.get("expected_output") == before.get("expected_output")
    before_results = [message for message in before["messages"] if message["role"] == "tool"]
    after_results = [message for message in after["messages"] if message["role"] == "tool"]
    assert sorted(before_results, key=lambda message: message["tool_call_id"]) == sorted(
        after_results, key=lambda message: message["tool_call_id"]
    )
    assert curation._call_index(after["messages"], lookup_id) < curation._call_index(after["messages"], catalog_id)
    assert all(message.get("content") != "" for message in after["messages"] if not message.get("tool_calls"))
    assert curation.curate_reviewed_row(curated) == (curated, [])


@pytest.mark.parametrize("number,forbidden_id,known_method", [
    (111, "call_9e5a869f9b624b98ab119d32", "gift_card_5701566"),
    (16, "call_7b03739a5a9d4602bc3cdd3a", "gift_card_1888303"),
])
def test_reviewed_payment_curation_refuses_instead_of_inventing_or_submitting_a_method(
    curation, retail, number, forbidden_id, known_method,
):
    original = read_row("sft-train.jsonl", number)
    before = copy.deepcopy(original)
    curated, decisions = curation.curate_reviewed_row(original)
    assert original == before
    assert retail.retail_trace_diagnostics(curated) == []
    assert curated["tools"] == original["tools"]
    assert curation._call_index(curated["messages"], forbidden_id) is None
    target = curated["messages"][-1]
    assert target["role"] == "assistant" and not target.get("tool_calls")
    assert known_method in target["content"] and "cannot" in target["content"]
    if number == 111:
        assert "$37" in target["content"] and "$329.40" in target["content"]
        assert curation._call_index(curated["messages"], "call_ef150e73c5e14f0e888ca256") is None
    assert decisions == ["refuse-unobserved-payment-without-submitting-mutation"]
    assert independent_payment_violations(original)
    assert independent_payment_violations(curated) == []
    assert curated["messages"][:-1] == original["messages"][:len(curated["messages"]) - 1]
    assert curation.curate_reviewed_row(curated) == (curated, [])


@pytest.mark.parametrize("number,profile_id", [
    (111, "call_b7d75f8e3de74d6591f62b30"),
    (16, "call_fad448d0cb44480990b0d876"),
])
@pytest.mark.parametrize("failure", [{"error": "fixture profile lookup failed"}, {"success": False}])
def test_payment_refusal_requires_successful_profile_evidence(curation, number, profile_id, failure):
    row = read_row("sft-train.jsonl", number)
    index = curation._call_index(row["messages"], profile_id)
    assert index is not None
    profile = json.loads(row["messages"][index + 1]["content"])
    profile.update(failure)  # Retained method fields cannot turn a failed reply into evidence.
    row["messages"][index + 1]["content"] = json.dumps(profile)
    before = copy.deepcopy(row)
    with pytest.raises(ValueError, match="Reviewed profile requires a successful observed result"):
        curation.curate_reviewed_row(row)
    assert row == before


@pytest.mark.parametrize("defect", ["missing-profile", "missing-calculation", "wrong-operation", "missing-method"])
def test_ethan_payment_curation_fails_closed_on_missing_or_changed_evidence(curation, defect):
    row = read_row("sft-train.jsonl", 111)
    messages = row["messages"]
    profile = curation._call_index(messages, "call_b7d75f8e3de74d6591f62b30")
    calculation = curation._call_index(messages, "call_543dd16e45f844efac6bccdf")
    assert profile is not None and calculation is not None
    if defect == "missing-profile":
        del messages[profile:profile + 2]
    elif defect == "missing-calculation":
        del messages[calculation:calculation + 2]
    elif defect == "wrong-operation":
        messages[calculation]["tool_calls"][0]["function"]["name"] = "get_product_details"
    else:
        result = json.loads(messages[profile + 1]["content"])
        result["payment_methods"] = None
        messages[profile + 1]["content"] = json.dumps(result)
    before = copy.deepcopy(row)
    with pytest.raises(ValueError, match="Reviewed"):
        curation.curate_reviewed_row(row)
    assert row == before


@pytest.mark.parametrize("number,failed_id,profile_id,observed_method", REVIEWED_EVALUATION_PAYMENTS)
def test_reviewed_evaluation_curation_removes_only_the_failed_pair_not_observed_evidence_or_target(
    curation, retail, number, failed_id, profile_id, observed_method,
):
    original = read_row("next-tool-evaluation.jsonl", number)
    before = copy.deepcopy(original)
    item = original["item"]
    index = curation._call_index(item["messages"], failed_id)
    assert index is not None
    assert item["messages"][index + 1] == reply(failed_id, "Error: payment method not found")
    assert curation._call_index(item["messages"], profile_id) == index + 2
    profile = json.loads(item["messages"][index + 3]["content"])
    assert profile["payment_methods"][observed_method]["id"] == observed_method
    target = item["expected_output"]["tool_calls"]
    assert len(target) == 1
    assert json.loads(target[0]["function"]["arguments"])["payment_method_id"] == observed_method
    assert independent_payment_violations(original) == [index]

    curated, decisions = curation.curate_reviewed_row(original)
    assert original == before
    expected = copy.deepcopy(original)
    del expected["item"]["messages"][index:index + 2]
    assert curated == expected  # Every other message, reply, declaration, and target is unchanged.
    assert decisions == ["remove-reviewed-failed-payment-attempt-preserve-observed-profile-and-target"]
    assert curated["item"]["expected_output"] == item["expected_output"]
    assert retail.retail_trace_diagnostics(curated) == []
    assert independent_first_tool_violations(curated) == []
    assert independent_payment_violations(curated) == []
    assert curation.curate_reviewed_row(curated) == (curated, [])


@pytest.mark.parametrize("number,failed_id,profile_id,observed_method", REVIEWED_EVALUATION_PAYMENTS)
@pytest.mark.parametrize("defect", [
    "failure-reply", "profile-customer", "profile-method", "profile-error", "profile-unsuccessful", "target",
])
def test_evaluation_curation_rejects_changed_failure_profile_or_target_without_fabrication(
    curation, number, failed_id, profile_id, observed_method, defect,
):
    row = read_row("next-tool-evaluation.jsonl", number)
    item = row["item"]
    messages = item["messages"]
    index = curation._call_index(messages, failed_id)
    profile_index = curation._call_index(messages, profile_id)
    assert index is not None and profile_index is not None
    if defect == "failure-reply":
        messages[index + 1]["content"] = json.dumps({"success": True})
    elif defect == "target":
        function = item["expected_output"]["tool_calls"][0]["function"]
        arguments = json.loads(function["arguments"])
        arguments["payment_method_id"] = "credit_card_unobserved_fixture"
        function["arguments"] = json.dumps(arguments)
    else:
        profile = json.loads(messages[profile_index + 1]["content"])
        if defect == "profile-customer":
            profile["user_id"] = "different_fixture_customer"
        elif defect == "profile-error":
            profile["error"] = "fixture profile lookup failed"
        elif defect == "profile-unsuccessful":
            profile["success"] = False
        else:
            del profile["payment_methods"][observed_method]
        messages[profile_index + 1]["content"] = json.dumps(profile)
    before = copy.deepcopy(row)
    with pytest.raises(ValueError, match="Reviewed"):
        curation.curate_reviewed_row(row)
    assert row == before


def test_preparation_returns_only_reviewed_row_changes_and_computed_provenance_without_writes(curation):
    before = immutable_inputs(ROOT)
    payloads, report = curation.prepare_reviewed_datasets(ROOT)
    assert immutable_inputs(ROOT) == before
    assert report["status"] == "validated-for-prepared-use"
    assert set(payloads) == {f"data/{name}" for name in ACTIVE_DATASETS}
    expected_changes = {
        "sft-train.jsonl": {16, 111, 200}, "sft-validation.jsonl": {54},
        "next-tool-evaluation.jsonl": {10, 77, 99, 162, 180},
        "rft-train.jsonl": set(), "rft-validation.jsonl": set(),
    }
    for name in DATASETS:
        relative = f"data/{name}"
        original = before[relative]
        entry = report["files"][relative]
        assert entry["source"]["raw_sha256"] == hashlib.sha256(original).hexdigest()
        assert entry["source"]["sha256"] == hashlib.sha256(original.replace(b"\r\n", b"\n")).hexdigest()
        if name not in ACTIVE_DATASETS:
            assert relative not in payloads and entry["output"] is None and entry["curation"] == []
            continue
        content = payloads[relative]
        assert b"\r\n" not in content
        assert entry["output"]["sha256"] == entry["output"]["raw_sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["output"]["bytes"] == entry["output"]["raw_bytes"] == len(content)
        source_lines = original.splitlines(keepends=True)
        output_lines = content.splitlines(keepends=True)
        assert len(source_lines) == len(output_lines) == entry["output"]["records"] == entry["source"]["records"]
        changed = {record["row"] for record in entry["curation"]}
        assert changed == expected_changes[name]
        for number, (source, output) in enumerate(zip(source_lines, output_lines), 1):
            if number not in changed:
                assert source.replace(b"\r\n", b"\n") == output
        for record in entry["curation"]:
            source = record["source_raw_line"].encode("utf-8")
            assert source == source_lines[record["row"] - 1]
            assert record["source_raw_line_sha256"] == hashlib.sha256(source).hexdigest()
            assert record["source_lf_line_sha256"] == hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()
            assert record["output_raw_line_sha256"] == hashlib.sha256(output_lines[record["row"] - 1]).hexdigest()
            assert record["source_diagnostics"] and record["output_diagnostics"] == []


def test_reviewed_authentication_repair_refuses_to_invent_a_changed_lookup_result(curation):
    row = read_row("sft-validation.jsonl", 54)
    messages = row["messages"]
    index = curation._call_index(messages, "call_349fb7e2c7bf40eea73e227b")
    catalog = curation._call_index(messages, "call_1d2d61c31ff64373b418873b")
    assert index is not None and catalog is not None
    assert index > catalog  # Committed originals stay uncurated provenance.
    messages[index + 1]["content"] = "different_fixture_customer"
    before = copy.deepcopy(row)
    with pytest.raises(ValueError, match="authentication result changed"):
        curation.curate_reviewed_row(row)
    assert row == before


def fixture_preparation_inputs(root, *, eol=b"\n", final_newline=True):
    """Write synthetic unit fixtures only, never the checked-in retail records."""
    data = root / "data"
    data.mkdir()
    row = {"messages": [{"role": "user", "content": "Offline fixture with literal \\r\\n text."},
                        {"role": "assistant", "content": "No tool was invoked."}]}
    encoded = b"  " + json.dumps(row).encode("utf-8")
    raw = encoded + eol + encoded + (eol if final_newline else b"")
    canonical = raw.replace(b"\r\n", b"\n")
    entries = []
    for name in DATASETS:
        (data / name).write_bytes(raw)
        entries.append({"path": f"data/{name}", "bytes": len(canonical), "records": 2,
                        "sha256": hashlib.sha256(canonical).hexdigest()})
    manifest = {"algorithm": "sha256", "text_normalization": "crlf-to-lf", "files": entries}
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw, manifest


@pytest.mark.parametrize("eol,final_newline", [
    (b"\n", True), (b"\r\n", True), (b"\n", False), (b"\r\n", False),
])
def test_preparation_preserves_raw_bytes_and_writes_lf_active_copies_and_manifest(
    curation, tmp_path, eol, final_newline,
):
    raw, manifest = fixture_preparation_inputs(tmp_path, eol=eol, final_newline=final_newline)
    paths, report = curation.prepare_active_datasets(tmp_path)
    for expected in manifest["files"]:
        relative = expected["path"]
        assert (tmp_path / relative).read_bytes() == raw
        entry = report["files"][relative]
        assert entry["source"]["sha256"] == expected["sha256"]
        assert entry["source"]["bytes"] == expected["bytes"] and entry["source"]["records"] == 2
        assert entry["source"]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["curation"] == []
        name = Path(relative).name
        if name in ACTIVE_DATASETS:
            assert paths[name].read_bytes() == raw.replace(b"\r\n", b"\n")
            assert entry["output"]["sha256"] == entry["output"]["raw_sha256"] == expected["sha256"]
        else:
            assert name not in paths and entry["output"] is None
    generated = tmp_path / "outputs" / "prepared-data" / "manifest.json"
    assert b"\r\n" not in generated.read_bytes()
    assert json.loads(generated.read_bytes()) == report
    assert curation.validated_prepared_paths(tmp_path, report) == paths


@pytest.mark.parametrize("defect,match", [
    ("source-hash", "Source provenance differs"),
    ("record-count", "Source record count differs"),
    ("active-trace", "Unreviewed active trace failures remain"),
    ("active-payment", "Unreviewed active trace failures remain"),
    ("active-order-owner", "Unreviewed active trace failures remain"),
])
def test_preparation_fails_closed_on_source_drift_count_drift_and_unreviewed_active_rows(
    curation, tmp_path, defect, match,
):
    raw, manifest = fixture_preparation_inputs(tmp_path)
    first = tmp_path / manifest["files"][0]["path"]
    if defect == "source-hash":
        first.write_bytes(raw + b" ")
    elif defect == "record-count":
        manifest["files"][0]["records"] = 3
    else:
        messages = [call("list_all_product_types")]
        if defect == "active-payment":
            messages = [*observed_profile(), action("credit_card_unobserved_fixture")]
        elif defect == "active-order-owner":
            messages = [*observed_profile(), action(order="#foreign-fixture-order")]
        raw = (json.dumps({"messages": messages}) + "\n").encode("utf-8")
        first.write_bytes(raw)
        manifest["files"][0].update(bytes=len(raw), records=1, sha256=hashlib.sha256(raw).hexdigest())
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in (tmp_path / "data").iterdir()}
    with pytest.raises(ValueError, match=match):
        curation.prepare_active_datasets(tmp_path)
    assert {path.name: path.read_bytes() for path in (tmp_path / "data").iterdir()} == before
    assert not (tmp_path / "outputs").exists()


def test_preparation_is_idempotent_and_does_not_rewrite_sources_or_identical_outputs(curation, tmp_path):
    fixture_preparation_inputs(tmp_path, eol=b"\r\n", final_newline=False)
    before = immutable_inputs(tmp_path)
    paths, report = curation.prepare_active_datasets(tmp_path)
    files = list((tmp_path / "outputs" / "prepared-data").iterdir())
    artifacts = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    again, repeated = curation.prepare_active_datasets(tmp_path)
    assert again == paths and repeated == report
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in files} == artifacts
    assert immutable_inputs(tmp_path) == before


@pytest.mark.parametrize("output", ["data", "data/sft-train.jsonl", "..", "outputs/elsewhere",
                                         "outputs/prepared-data/../prepared-data"])
def test_output_path_cannot_escape_or_overwrite_immutable_inputs(curation, tmp_path, output):
    fixture_preparation_inputs(tmp_path)
    before = immutable_inputs(tmp_path)
    with pytest.raises(ValueError, match="contained in outputs/prepared-data"):
        curation.prepare_active_datasets(tmp_path, output_dir=tmp_path / output)
    assert immutable_inputs(tmp_path) == before and not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("name", [*ACTIVE_DATASETS, "manifest.json"])
def test_conflicting_existing_artifact_is_never_overwritten_or_partially_published(curation, tmp_path, name):
    fixture_preparation_inputs(tmp_path)
    before = immutable_inputs(tmp_path)
    directory = tmp_path / "outputs" / "prepared-data"
    directory.mkdir(parents=True)
    path = directory / name
    path.write_bytes(b"existing local artifact, do not clobber\n")
    with pytest.raises(ValueError, match="will not overwrite"):
        curation.prepare_active_datasets(tmp_path)
    assert list(directory.iterdir()) == [path]
    assert path.read_bytes() == b"existing local artifact, do not clobber\n"
    assert immutable_inputs(tmp_path) == before


@pytest.mark.parametrize("kind", ["file-symlink", "file-hardlink", "parent-symlink"])
def test_linked_output_can_never_alias_or_modify_inputs(curation, tmp_path, kind):
    fixture_preparation_inputs(tmp_path)
    before = immutable_inputs(tmp_path)
    directory = tmp_path / "outputs" / "prepared-data"
    source = tmp_path / "data" / "sft-train.jsonl"
    try:
        if kind == "parent-symlink":
            (tmp_path / "outputs").symlink_to(tmp_path / "data", target_is_directory=True)
        else:
            directory.mkdir(parents=True)
            destination = directory / "sft-train.jsonl"
            if kind == "file-symlink":
                destination.symlink_to(source)
            else:
                os.link(source, destination)
    except OSError as error:
        pytest.skip(f"This platform cannot create the link fixture: {error}")
    with pytest.raises(ValueError, match="contain|link"):
        curation.prepare_active_datasets(tmp_path)
    assert immutable_inputs(tmp_path) == before


def test_interrupted_identical_preparation_resumes_only_missing_files(curation, tmp_path):
    fixture_preparation_inputs(tmp_path)
    paths, report = curation.prepare_active_datasets(tmp_path)
    retained = paths["sft-train.jsonl"]
    evidence = retained.read_bytes(), retained.stat().st_mtime_ns
    paths["rft-validation.jsonl"].unlink()
    (tmp_path / "outputs" / "prepared-data" / "manifest.json").unlink()
    again, repeated = curation.prepare_active_datasets(tmp_path)
    assert again == paths and repeated == report
    assert (retained.read_bytes(), retained.stat().st_mtime_ns) == evidence
    assert curation.validated_prepared_paths(tmp_path, repeated) == paths


@pytest.mark.parametrize("name", [*ACTIVE_DATASETS, "manifest.json"])
@pytest.mark.parametrize("defect", ["blank-record", "malformed-json", "empty"])
def test_consumer_rechecks_tampered_prepared_bytes_before_use(curation, tmp_path, monkeypatch, name, defect):
    fixture_preparation_inputs(tmp_path)
    _, report = curation.prepare_active_datasets(tmp_path)
    path = tmp_path / "outputs" / "prepared-data" / name
    original = path.read_bytes()
    corrupted = original + b" " if defect == "blank-record" else b"{not-json}\n" if defect == "malformed-json" else b""
    path.write_bytes(corrupted)
    records = curation._jsonl_records

    def reject_corrupted_parser_input(content, dataset_name):
        assert content != corrupted, "Tampered prepared bytes reached the record parser before their hash was checked."
        return records(content, dataset_name)

    monkeypatch.setattr(curation, "_jsonl_records", reject_corrupted_parser_input)
    with pytest.raises(ValueError, match="Prepared"):
        curation.validated_prepared_paths(tmp_path, report)
    assert path.read_bytes() == corrupted


def test_source_drift_after_preparation_invalidates_consumption_without_repair(curation, tmp_path):
    fixture_preparation_inputs(tmp_path)
    _, report = curation.prepare_active_datasets(tmp_path)
    source = tmp_path / "data" / "synthetic-source-train.jsonl"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Source provenance differs"):
        curation.validated_prepared_paths(tmp_path, report)


def test_synthetic_findings_survive_and_only_five_outputs_are_ever_created(curation, retail, tmp_path):
    _, manifest = fixture_preparation_inputs(tmp_path)
    name = "synthetic-source-validation.jsonl"
    relative = f"data/{name}"
    raw = (json.dumps({"messages": [call("list_all_product_types")]}) + "\n").encode("utf-8")
    (tmp_path / relative).write_bytes(raw)
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry.update(bytes=len(raw), records=1, sha256=hashlib.sha256(raw).hexdigest())
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    paths, report = curation.prepare_active_datasets(tmp_path)
    assert report["files"][relative]["source_diagnostics"] == retail.audit_retail_datasets(tmp_path)[name]
    assert report["files"][relative]["source_diagnostics"]
    assert report["files"][relative]["output"] is None and name not in paths
    assert {path.name for path in (tmp_path / "outputs" / "prepared-data").iterdir()} == {
        *ACTIVE_DATASETS, "manifest.json",
    }
    assert (tmp_path / relative).read_bytes() == raw


@pytest.mark.parametrize("defect", ["blank", "malformed", "manifest-traversal", "duplicate-manifest-path"])
def test_invalid_source_structure_fails_before_creating_outputs(curation, tmp_path, defect):
    raw, manifest = fixture_preparation_inputs(tmp_path)
    if defect in {"blank", "malformed"}:
        raw = b"\n" if defect == "blank" else b"{not-json}\n"
        (tmp_path / "data" / "sft-train.jsonl").write_bytes(raw)
        manifest["files"][0].update(bytes=len(raw), records=1, sha256=hashlib.sha256(raw).hexdigest())
    elif defect == "manifest-traversal":
        manifest["files"][0]["path"] = "data/../data/sft-train.jsonl"
    else:
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = immutable_inputs(tmp_path)
    with pytest.raises(ValueError):
        curation.prepare_active_datasets(tmp_path)
    assert immutable_inputs(tmp_path) == before and not (tmp_path / "outputs").exists()