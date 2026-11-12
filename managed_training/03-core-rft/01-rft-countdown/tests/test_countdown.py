"""Offline regressions for Countdown's actual notebook contracts and call paths.

Only allowlisted assignments and function definitions are compiled. No notebook
cell, Azure import, environment loader, subprocess, upload, or live client runs.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "demo.ipynb"
ASSIGNMENTS = {
    "MODEL",
    "MODEL_PUBLISHER",
    "MODEL_VERSION",
    "TRAINING_TYPE",
    "GRADER_MODEL",
    "GRADER_MODEL_VERSION",
    "GRADER_DEPLOYMENT_SKU",
    "GRADER_QUOTA_NAME",
    "DATASETS",
    "RESPONSE_FORMAT",
    "MODEL_GRADER",
    "PYTHON_GRADER_SOURCE",
    "PYTHON_GRADER",
    "HYPERPARAMETERS",
    "RUN_FINGERPRINT",
}
FUNCTIONS = {
    "validate_jsonl",
    "validate_grader_contracts",
    "build_job_payload",
    "recipe_fingerprint",
    "canonical_json_sha256",
    "submission_reconciliation_message",
    "submission_file_evidence",
    "validate_submission_intent",
    "build_submission_intent",
    "load_state",
    "normalized_location",
    "validate_existing_grader",
    "choose_grader_capacity",
    "deployment_token_limit",
    "grader_account_name",
    "grader_deployment_identity",
    "load_grader_deployment_state",
    "record_grader_deployment_operation",
    "get_grader_deployment",
    "plan_grader_deployment",
    "ensure_grader_deployment",
    "submission_state_lock",
    "save_state",
    "submit_once",
}


def code_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, str):
            sources.append(source)
        elif any(line.endswith(("\n", "\r")) for line in source):
            sources.append("".join(source))
        else:
            # Retained cells may store one line per element without terminators.
            sources.append("\n".join(source))
    return sources


def load_contracts(tmp_path: Path, deployment_name: str = "countdown-custom-grader") -> dict:
    def unexpected_cloud_call(*args, **kwargs):
        raise AssertionError("A real cloud or subprocess call is not allowed in these tests")

    namespace = {
        "__builtins__": __builtins__,
        "Path": Path,
        "datetime": datetime,
        "timezone": timezone,
        "urlparse": urlparse,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "tempfile": tempfile,
        "contextmanager": contextmanager,
        "DEMO_ROOT": ROOT,
        "OUTPUT_ROOT": tmp_path,
        "STATE_PATH": tmp_path / "submission-state.json",
        "GRADER_DEPLOYMENT_STATE_PATH": tmp_path / "grader-deployment-state.json",
        "PROJECT_ENDPOINT": "https://countdown-offline.services.ai.azure.com/api/projects/offline",
        "SUBSCRIPTION_ID": "offline-subscription",
        "RESOURCE_GROUP": "offline-resource-group",
        "RESOURCE_LOCATION": "Offline Region",
        "GRADER_DEPLOYMENT": deployment_name,
        "GRADER_CAPACITY_SETTING": "100",
        "GRADER_MIN_TPM": 100_000,
        "ALLOW_GRADER_DEPLOYMENT_UPDATE": True,
        "run_az_json": unexpected_cloud_call,
    }
    found_assignments = set()
    found_functions = set()
    for index, source in enumerate(code_sources(), start=1):
        tree = ast.parse(source)
        safe_nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                safe_nodes.append(node)
                found_functions.add(node.name)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ASSIGNMENTS:
                    safe_nodes.append(node)
                    found_assignments.add(target.id)
        module = ast.Module(body=safe_nodes, type_ignores=[])
        exec(compile(module, f"countdown-offline-code-{index}", "exec"), namespace)
    assert found_assignments == ASSIGNMENTS
    assert found_functions == FUNCTIONS
    return namespace


@pytest.fixture
def contracts(tmp_path: Path) -> dict:
    return load_contracts(tmp_path)


def deployment(namespace: dict, capacity: int = 100, tpm: int = 100_000) -> dict:
    return {
        "id": "/offline/deployments/" + namespace["GRADER_DEPLOYMENT"],
        "name": namespace["GRADER_DEPLOYMENT"],
        "sku": {"name": namespace["GRADER_DEPLOYMENT_SKU"], "capacity": capacity},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": namespace["GRADER_MODEL"],
                "version": namespace["GRADER_MODEL_VERSION"],
            },
            "rateLimits": [{"key": "token", "count": tpm}],
        },
    }


class FakeArm:
    def __init__(self, namespace: dict, current: dict | None):
        self.namespace = namespace
        self.current = copy.deepcopy(current)
        self.calls = []
        self.create_calls = 0
        self.failure = None
        self.quota_limit = 200

    def __call__(self, *arguments):
        self.calls.append(arguments)
        if arguments[:3] == ("cognitiveservices", "account", "show"):
            return {"name": "countdown-offline", "location": "offlineregion"}
        if arguments[:3] == ("cognitiveservices", "usage", "list"):
            capacity = (self.current or {}).get("sku", {}).get("capacity", 0)
            return [{
                "name": {"value": self.namespace["GRADER_QUOTA_NAME"]},
                "limit": self.quota_limit,
                "currentValue": capacity,
            }]
        if arguments[:4] == ("cognitiveservices", "account", "deployment", "list"):
            return [copy.deepcopy(self.current)] if self.current is not None else []
        if arguments[:4] == ("cognitiveservices", "account", "deployment", "show"):
            return copy.deepcopy(self.current)
        if arguments[:4] != ("cognitiveservices", "account", "deployment", "create"):
            raise AssertionError(f"Unexpected offline ARM call: {arguments}")

        # Inspect the on-disk intent at the exact mutation boundary, not afterward.
        evidence = self.namespace["load_grader_deployment_state"]()
        assert evidence is not None
        assert evidence["previous_capacity"] == (
            (self.current or {}).get("sku", {}).get("capacity", 0)
        )
        assert evidence["previously_existed"] is (self.current is not None)
        assert evidence["previous_deployment_id"] == (self.current or {}).get("id")
        assert arguments[arguments.index("--deployment-name") + 1] == evidence["deployment_name"]
        capacity = int(arguments[arguments.index("--sku-capacity") + 1])
        assert capacity == evidence["requested_capacity"]
        self.create_calls += 1
        if self.failure == "before-update":
            raise TimeoutError("Offline ambiguous ARM timeout before the update")
        self.current = deployment(self.namespace, capacity=capacity)
        if self.failure == "after-update":
            raise TimeoutError("Offline ambiguous ARM timeout after the update")
        if self.failure == "low-tpm":
            self.current["properties"]["rateLimits"][0]["count"] = 99_999
        if self.failure == "missing-tpm":
            self.current["properties"]["rateLimits"] = []
        if self.failure == "wrong-capacity":
            self.current["sku"]["capacity"] = capacity + 1
        return copy.deepcopy(self.current)


class FakeJobs:
    def __init__(self, namespace: dict):
        self.namespace = namespace
        self.payloads = []
        self.retrieved = []
        self.options = []
        self.pending_at_create = []
        self.accepted_ids = []
        self.failure = None
        self.retrieve_failure = None
        self.on_create = None
        self.mutate_payload = False
        self.fine_tuning = SimpleNamespace(jobs=self)

    def with_options(self, **options):
        self.options.append(options)
        return self

    def create(self, **payload):
        label = {
            "countdown_model_grader": "model-grader",
            "countdown_python_grader": "python-grader",
        }[payload["method"]["reinforcement"]["grader"]["name"]]
        path = self.namespace["STATE_PATH"]
        original_bytes = path.read_bytes()
        state = self.namespace["load_state"]()
        intent = state["pending"][label]
        # Inspect durable evidence and serialization at the actual POST boundary.
        assert path.with_name(path.name + ".lock").is_file()
        assert state["schema_version"] == 2
        assert state["fingerprint"] == self.namespace["RUN_FINGERPRINT"]
        assert label not in state["jobs"]
        assert intent["schema_version"] == 1
        assert intent["record_type"] == "job_creation_intent"
        assert intent["label"] == label
        assert intent["project_endpoint"] == self.namespace["PROJECT_ENDPOINT"]
        assert intent["run_fingerprint"] == state["fingerprint"]
        assert intent["payload"] == payload
        assert intent["payload_sha256"] == hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        assert intent["files"] == {
            kind: {
                "file_id": payload[f"{kind}_file"],
                **{
                    key: self.namespace["DATASETS"][kind][key]
                    for key in ("bytes", "rows", "sha256")
                },
            }
            for kind in ("training", "validation")
        }
        self.pending_at_create.append((label, original_bytes))
        self.payloads.append(copy.deepcopy(payload))
        if self.on_create is not None:
            self.on_create()
        if self.failure == "before-response":
            raise TimeoutError("Offline failure before acceptance/response")
        if self.failure == "definitive-error":
            raise ValueError("Offline definitive rejection still retains intent")
        job_id = f"offline-job-{len(self.payloads)}"
        self.accepted_ids.append(job_id)
        if self.failure == "after-response":
            raise TimeoutError("Offline accepted job with lost response")
        if self.mutate_payload:
            payload["method"]["reinforcement"]["hyperparameters"]["eval_samples"] = 999
        if self.failure == "missing-id":
            job_id = None
        return SimpleNamespace(id=job_id, status="queued")

    def retrieve(self, job_id):
        path = self.namespace["STATE_PATH"]
        assert path.with_name(path.name + ".lock").is_file()
        self.retrieved.append(job_id)
        if self.retrieve_failure is not None:
            raise self.retrieve_failure
        return SimpleNamespace(id=job_id, status="queued")


def attach_jobs(namespace: dict) -> FakeJobs:
    jobs = FakeJobs(namespace)
    namespace.update({
        "openai_client": jobs,
        "training_file_id": "offline-training-file",
        "validation_file_id": "offline-validation-file",
    })
    return jobs


def submit_variant(namespace: dict, label: str = "model-grader", suffix: str | None = None):
    grader = namespace["MODEL_GRADER" if label == "model-grader" else "PYTHON_GRADER"]
    return namespace["submit_once"](
        label, grader, suffix or f"qwen36-countdown-{label}",
    )


@pytest.fixture(params=["standalone", "embedded-literal"])
def grader_function(request, contracts):
    if request.param == "standalone":
        source = (ROOT / "graders" / "countdown_python_grader.py").read_text(encoding="utf-8")
    else:
        assignments = [
            node
            for cell_source in code_sources()
            for node in ast.parse(cell_source).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PYTHON_GRADER_SOURCE"
                    for target in node.targets)
        ]
        assert len(assignments) == 1
        assert isinstance(assignments[0].value, ast.Constant)
        source = ast.literal_eval(assignments[0].value)
        assert source == contracts["PYTHON_GRADER_SOURCE"] == contracts["PYTHON_GRADER"]["source"]
    namespace = {}
    exec(compile(source, f"countdown-offline-{request.param}", "exec"), namespace)
    return namespace["grade"]


def test_notebook_python_cells_parse_without_execution() -> None:
    for source in code_sources():
        ast.parse(source)


def test_notebook_remains_unexecuted() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")


@pytest.mark.parametrize("field", ["reported", "target", "input"])
@pytest.mark.parametrize("nonfinite", [
    "NaN", "nan", "Infinity", "-Infinity", "inf", "-inf", "1e309", "-1e309",
    float("nan"), float("inf"), float("-inf"), 10 ** 400,
])
def test_both_graders_reject_nonfinite_values(grader_function, field, nonfinite):
    sample = {"output_json": {"expression": "(1 + 3) * (2 + 4)", "result": "24"}}
    item = {"target": "24", "nums": "[1, 2, 3, 4]"}
    if field == "reported":
        sample["output_json"]["result"] = nonfinite
    elif field == "target":
        item["target"] = nonfinite
    else:
        item["nums"] = json.dumps([1, 2, 3, nonfinite])
    assert grader_function(sample, item) == 0


@pytest.mark.parametrize(("expression", "result", "target", "numbers"), [
    ("1e309 + 1", "1", "1", "[1e309, 1]"),
    ("0 - 1e309", "-Infinity", "0", "[0, 1e309]"),
    ("1e309 - 1e309", "NaN", "0", "[1e309, 1e309]"),
    ("1e308 * 1e308", "Infinity", "1", "[1e308, 1e308]"),
    ("(1e308 * 1e308) / 1e308", "1e308", "1e308", "[1e308, 1e308, 1e308]"),
    ("(1e308 * 1e308) - (1e308 * 1e308)", "0", "0", "[1e308, 1e308, 1e308, 1e308]"),
    ("1e308 + 0", "1e308", "-1e308", "[1e308, 0]"),
    ("1e308 + 0", "-1e308", "1e308", "[1e308, 0]"),
    (str(10 ** 400) + " + 1", "1", "1", json.dumps([10 ** 400, 1])),
])
def test_both_graders_reject_nonfinite_literals_arithmetic_and_overflow(
    grader_function, expression, result, target, numbers,
):
    assert grader_function(
        {"output_json": {"expression": expression, "result": result}},
        {"target": target, "nums": numbers},
    ) == 0


@pytest.mark.parametrize(("expression", "result", "target", "numbers", "score"), [
    ("(1 + 3) * (2 + 4)", "24", "24", "[1, 2, 3, 4]", 5),
    ("(1 + 3) * (2 + 4)", "24", "25", "[1, 2, 3, 4]", 4),
    ("(1 + 3) * (2 + 4)", "24", "26", "[1, 2, 3, 4]", 3),
    ("(1 + 3) * (2 + 4)", "24", "29", "[1, 2, 3, 4]", 3),
    ("(1 + 3) * (2 + 4)", "24", "30", "[1, 2, 3, 4]", 2),
    ("(1 + 3) * (2 + 4)", "25", "24", "[1, 2, 3, 4]", 1),
    ("(8 - 4) / (3 + 1)", "1", "1", "[8, 4, 3, 1]", 5),
    ("(2 + 2) * (3 + 3)", "24", "24", "[2, 2, 3, 3]", 5),
    ("4 * 4 + 3 + 2", "21", "21", "[1, 2, 3, 4]", 0),
    ("1 + 2 + 3", "6", "6", "[1, 2, 3, 4]", 0),
    ("1 + 2 + 3 + 4 + 0", "10", "10", "[1, 2, 3, 4]", 0),
    ("1 / (2 - 2)", "0", "0", "[1, 2, 2]", 0),
    ("2 ** 3", "8", "8", "[2, 3]", 0),
    ("sum([1, 2, 3, 4])", "10", "10", "[1, 2, 3, 4]", 0),
])
def test_both_graders_preserve_finite_scores_and_exact_number_usage(
    grader_function, expression, result, target, numbers, score,
):
    assert grader_function(
        {"output_json": {"expression": expression, "result": result}},
        {"target": target, "nums": numbers},
    ) == score


def test_standalone_output_text_fallback_rejects_nan_and_preserves_valid_score():
    namespace = {}
    source = (ROOT / "graders" / "countdown_python_grader.py").read_text(encoding="utf-8")
    exec(compile(source, "countdown-offline-text-fallback", "exec"), namespace)
    item = {"target": "24", "nums": "[1, 2, 3, 4]"}
    for result, score in (("24", 5), ("NaN", 0), ("Infinity", 0), ("-Infinity", 0)):
        sample = {"output_text": json.dumps({
            "expression": "(1 + 3) * (2 + 4)", "result": result,
        })}
        assert namespace["grade"](sample, item) == score


def test_standalone_and_embedded_grader_have_identical_executable_source(contracts):
    standalone = (ROOT / "graders/countdown_python_grader.py").read_text(encoding="utf-8")
    assert ast.dump(ast.parse(standalone)) == ast.dump(ast.parse(contracts["PYTHON_GRADER_SOURCE"]))


@pytest.mark.parametrize("count", [80, 1500])
def test_both_graders_reject_deep_left_associative_chains(grader_function, count):
    # 1500 operands fit the character limit but exceeded Python evaluation depth
    # before the fix; 80 isolates our deterministic depth bound on a smaller AST.
    assert grader_function(
        {"output_json": {"expression": "+".join(["1"] * count), "result": str(count)}},
        {"target": str(count), "nums": json.dumps([1] * count)},
    ) == 0


@pytest.mark.parametrize(("count", "expected"), [(85, 5), (86, 0)])
def test_both_graders_bound_balanced_ast_size_not_only_depth(grader_function, count, expected):
    parts = ["1"] * count
    while len(parts) > 1:
        parts = [
            f"({parts[index]}+{parts[index + 1]})" if index + 1 < len(parts) else parts[index]
            for index in range(0, len(parts), 2)
        ]
    assert grader_function(
        {"output_json": {"expression": parts[0], "result": str(count)}},
        {"target": str(count), "nums": json.dumps([1] * count)},
    ) == expected


def test_both_graders_reject_oversize_input_before_parsing(grader_function, monkeypatch):
    def forbidden_parse(*args, **kwargs):
        raise AssertionError("Oversize input must be rejected before AST allocation")

    namespace = grader_function.__globals__
    monkeypatch.setitem(namespace, "ast", SimpleNamespace(parse=forbidden_parse))
    expression = "1" + " " * namespace["_MAX_EXPRESSION_CHARS"]
    assert grader_function(
        {"output_json": {"expression": expression, "result": "1"}},
        {"target": "1", "nums": "[1]"},
    ) == 0


@pytest.mark.parametrize("phase", ["parse", "evaluate"])
def test_both_graders_convert_recursion_failures_to_zero(grader_function, monkeypatch, phase):
    def recursion_failure(*args, **kwargs):
        raise RecursionError("Offline parser/evaluator recursion failure")

    namespace = grader_function.__globals__
    if phase == "parse":
        monkeypatch.setitem(namespace, "ast", SimpleNamespace(parse=recursion_failure))
    else:
        monkeypatch.setitem(namespace, "_evaluate", recursion_failure)
    assert grader_function(
        {"output_json": {"expression": "1+2", "result": "3"}},
        {"target": "3", "nums": "[1, 2]"},
    ) == 0


@pytest.mark.parametrize("deployment_name", ["gpt-5.4-mini", "countdown-custom-grader"])
def test_custom_deployment_routes_grader_and_arm_consistently(tmp_path, deployment_name):
    namespace = load_contracts(tmp_path, deployment_name)
    arm = FakeArm(namespace, None)
    namespace["run_az_json"] = arm
    result = namespace["ensure_grader_deployment"]()
    assert result["name"] == namespace["MODEL_GRADER"]["model"] == deployment_name
    assert result["properties"]["model"] == {
        "name": "gpt-5.4-mini",
        "format": "OpenAI",
        "version": "2026-03-17",
    }
    assert result["sku"] == {"name": "DataZoneStandard", "capacity": 100}
    evidence = namespace["load_grader_deployment_state"]()
    assert evidence["deployment_name"] == deployment_name
    assert evidence["run_fingerprint"] == namespace["recipe_fingerprint"]()
    assert evidence["previously_existed"] is False


def test_configured_deployment_name_changes_recipe_fingerprint(tmp_path):
    default = load_contracts(tmp_path / "default", "gpt-5.4-mini")
    custom = load_contracts(tmp_path / "custom", "countdown-custom-grader")
    assert default["RUN_FINGERPRINT"] != custom["RUN_FINGERPRINT"]
    custom["MODEL_GRADER"]["model"] = custom["GRADER_MODEL"]
    with pytest.raises(ValueError, match="configured deployment"):
        custom["recipe_fingerprint"]()


@pytest.mark.parametrize("previously_existed", [False, True])
def test_rerun_before_any_job_preserves_first_zero_capacity_baseline(contracts, previously_existed):
    current = deployment(contracts, capacity=0, tpm=0) if previously_existed else None
    arm = FakeArm(contracts, current)
    contracts["run_az_json"] = arm
    contracts["ensure_grader_deployment"]()
    state_path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    first_bytes = state_path.read_bytes()
    first = json.loads(first_bytes)
    assert first["previous_capacity"] == 0
    assert first["requested_capacity"] == 100
    assert first["previously_existed"] is previously_existed
    assert not contracts["STATE_PATH"].exists()

    # Reruns are verification-only even if update authorization is no longer set.
    contracts["ALLOW_GRADER_DEPLOYMENT_UPDATE"] = False
    contracts["ensure_grader_deployment"]()
    contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 1
    assert state_path.read_bytes() == first_bytes


def test_max_quota_rerun_does_not_expand_the_first_request(contracts):
    contracts["GRADER_CAPACITY_SETTING"] = "max"
    arm = FakeArm(contracts, deployment(contracts, capacity=40, tpm=40_000))
    arm.quota_limit = 100
    contracts["run_az_json"] = arm
    contracts["ensure_grader_deployment"]()
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    baseline = path.read_bytes()
    assert json.loads(baseline)["previous_capacity"] == 40
    assert json.loads(baseline)["requested_capacity"] == 100
    arm.quota_limit = 200
    assert contracts["ensure_grader_deployment"]()["sku"]["capacity"] == 100
    assert arm.create_calls == 1
    assert path.read_bytes() == baseline


def test_first_operation_record_cannot_be_overwritten(contracts):
    arm = FakeArm(contracts, deployment(contracts, capacity=0, tpm=0))
    contracts["run_az_json"] = arm
    contracts["ensure_grader_deployment"]()
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    baseline = path.read_bytes()
    plan = contracts["plan_grader_deployment"]()
    with pytest.raises(RuntimeError, match="already recorded"):
        contracts["record_grader_deployment_operation"](plan, arm.current)
    assert path.read_bytes() == baseline
    assert arm.create_calls == 1


@pytest.mark.parametrize("previously_existed", [False, True])
@pytest.mark.parametrize("failure", [
    "before-update", "after-update", "low-tpm", "missing-tpm", "wrong-capacity",
])
def test_partial_failure_retains_baseline_and_never_retries_arm(contracts, previously_existed, failure):
    current = deployment(contracts, capacity=0, tpm=0) if previously_existed else None
    arm = FakeArm(contracts, current)
    arm.failure = failure
    contracts["run_az_json"] = arm
    with pytest.raises((TimeoutError, RuntimeError)):
        contracts["ensure_grader_deployment"]()
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    baseline = path.read_bytes()
    evidence = json.loads(baseline)
    assert evidence["previous_capacity"] == 0
    assert evidence["previously_existed"] is previously_existed
    assert evidence["requested_capacity"] == 100
    assert not contracts["STATE_PATH"].exists()

    if failure == "after-update":
        assert contracts["ensure_grader_deployment"]()["sku"]["capacity"] == 100
    else:
        with pytest.raises(RuntimeError):
            contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 1
    assert path.read_bytes() == baseline


@pytest.mark.parametrize("field", [
    "PROJECT_ENDPOINT", "SUBSCRIPTION_ID", "RESOURCE_GROUP", "RESOURCE_LOCATION",
    "GRADER_DEPLOYMENT", "GRADER_MODEL", "GRADER_MODEL_VERSION", "GRADER_DEPLOYMENT_SKU",
    "GRADER_CAPACITY_SETTING", "GRADER_MIN_TPM", "RUN_FINGERPRINT",
])
def test_changed_configuration_rejects_saved_operation_without_mutation(contracts, field):
    arm = FakeArm(contracts, deployment(contracts, capacity=0, tpm=0))
    contracts["run_az_json"] = arm
    contracts["ensure_grader_deployment"]()
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    baseline = path.read_bytes()
    value = contracts[field]
    contracts[field] = value + 1 if isinstance(value, int) else value + "-changed"
    with pytest.raises(RuntimeError, match="different configuration"):
        contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 1
    assert path.read_bytes() == baseline


def test_saved_operation_and_plan_reject_missing_or_externally_resized_deployment(contracts):
    arm = FakeArm(contracts, None)
    contracts["run_az_json"] = arm
    contracts["ensure_grader_deployment"]()
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    baseline = path.read_bytes()
    arm.current = deployment(contracts, capacity=150, tpm=150_000)
    with pytest.raises(RuntimeError, match="capacity differs"):
        contracts["ensure_grader_deployment"]()
    arm.current = None
    with pytest.raises(RuntimeError, match="no longer exists"):
        contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 1
    assert path.read_bytes() == baseline


@pytest.mark.parametrize("evidence", [
    {"previous_capacity": 0, "requested_capacity": 100},
    {"schema_version": 1, "previous_capacity": -1},
])
def test_legacy_or_invalid_evidence_fails_closed_and_is_not_replaced(contracts, evidence):
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    path.write_text(json.dumps(evidence), encoding="utf-8")
    baseline = path.read_bytes()
    with pytest.raises(RuntimeError):
        contracts["ensure_grader_deployment"]()
    assert path.read_bytes() == baseline


@pytest.mark.parametrize("field", ["previous_capacity", "requested_capacity", "previously_existed"])
def test_invalid_capacity_or_existence_evidence_is_not_rewritten(contracts, field):
    arm = FakeArm(contracts, None)
    contracts["run_az_json"] = arm
    contracts["ensure_grader_deployment"]()
    path = contracts["GRADER_DEPLOYMENT_STATE_PATH"]
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence[field] = "invalid-value"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    baseline = path.read_bytes()
    with pytest.raises(RuntimeError, match="Invalid saved deployment baseline"):
        contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 1
    assert path.read_bytes() == baseline


def test_recipe_drift_blocks_deployment_before_the_first_arm_call(contracts):
    contracts["HYPERPARAMETERS"]["eval_samples"] = 3
    with pytest.raises(RuntimeError, match="recipe changed"):
        contracts["ensure_grader_deployment"]()
    assert not contracts["GRADER_DEPLOYMENT_STATE_PATH"].exists()


def test_mismatched_saved_job_recipe_blocks_deployment_before_arm(contracts):
    contracts["STATE_PATH"].write_text(json.dumps({
        "fingerprint": "a-different-recipe",
        "jobs": {"model-grader": "offline-existing-job"},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different or unverified recipe"):
        contracts["ensure_grader_deployment"]()
    assert not contracts["GRADER_DEPLOYMENT_STATE_PATH"].exists()


def test_recorded_job_remains_read_only_without_fabricating_operation_baseline(contracts):
    contracts["STATE_PATH"].write_text(json.dumps({
        "fingerprint": contracts["RUN_FINGERPRINT"],
        "jobs": {"model-grader": "offline-existing-job"},
    }), encoding="utf-8")
    arm = FakeArm(contracts, deployment(contracts))
    contracts["run_az_json"] = arm
    contracts["ALLOW_GRADER_DEPLOYMENT_UPDATE"] = False
    contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 0
    assert not contracts["GRADER_DEPLOYMENT_STATE_PATH"].exists()


@pytest.mark.parametrize("failure", ["account", "location", "quota"])
def test_account_location_and_quota_checks_block_mutation(contracts, failure):
    arm = FakeArm(contracts, None)

    def checked_arm(*arguments):
        result = arm(*arguments)
        if arguments[:3] == ("cognitiveservices", "account", "show"):
            if failure == "account":
                result["name"] = "another-offline-account"
            if failure == "location":
                result["location"] = "another-offline-region"
        if arguments[:3] == ("cognitiveservices", "usage", "list") and failure == "quota":
            return []
        return result

    contracts["run_az_json"] = checked_arm
    with pytest.raises(RuntimeError):
        contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 0
    assert not contracts["GRADER_DEPLOYMENT_STATE_PATH"].exists()


@pytest.mark.parametrize(("field", "value"), [
    ("name", "another-deployment"),
    ("format", "AnotherFormat"),
    ("model", "another-model"),
    ("version", "another-version"),
    ("sku", "Standard"),
])
def test_existing_deployment_identity_checks_are_preserved(contracts, field, value):
    current = deployment(contracts)
    if field == "name":
        current["name"] = value
    elif field == "sku":
        current["sku"]["name"] = value
    else:
        current["properties"]["model"]["name" if field == "model" else field] = value
    with pytest.raises(RuntimeError):
        contracts["validate_existing_grader"](current)


def test_capacity_bounds_and_update_authorization_are_preserved(contracts):
    choose = contracts["choose_grader_capacity"]
    assert choose(200, 60, 10, "max") == 150
    assert choose(200, 60, 10, "100") == 100
    for setting in ("99", "0", "-1", "100.0", "not-a-capacity"):
        with pytest.raises(ValueError):
            choose(200, 60, 10, setting)
    with pytest.raises(RuntimeError, match="exceeds"):
        choose(200, 60, 10, "151")
    with pytest.raises(RuntimeError, match="below"):
        choose(99, 0, 0, "max")

    arm = FakeArm(contracts, None)
    contracts["run_az_json"] = arm
    contracts["ALLOW_GRADER_DEPLOYMENT_UPDATE"] = False
    with pytest.raises(RuntimeError, match="authorize"):
        contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 0
    assert not contracts["GRADER_DEPLOYMENT_STATE_PATH"].exists()


def test_actual_two_job_payloads_match_validated_contracts_before_paid_submission(contracts):
    jobs = attach_jobs(contracts)
    variants = [
        ("model-grader", contracts["MODEL_GRADER"], "qwen36-countdown-model-grader"),
        ("python-grader", contracts["PYTHON_GRADER"], "qwen36-countdown-python-grader"),
    ]
    for label, grader, suffix in variants:
        contracts["submit_once"](label, grader, suffix)
    assert jobs.options == [{"max_retries": 0}, {"max_retries": 0}]
    assert len(jobs.payloads) == 2
    assert jobs.retrieved == []
    for payload, (_, grader, suffix) in zip(jobs.payloads, variants, strict=True):
        assert payload == {
            "model": "qwen3.6-35b-a3b",
            "training_file": "offline-training-file",
            "validation_file": "offline-validation-file",
            "method": {
                "type": "reinforcement",
                "reinforcement": {
                    "hyperparameters": {
                        "eval_interval": 5,
                        "eval_samples": 2,
                        "reasoning_effort": "medium",
                        "n_epochs": 1,
                        "batch_size": 4,
                        "learning_rate_multiplier": 2,
                    },
                    "grader": grader,
                    "response_format": contracts["RESPONSE_FORMAT"],
                },
            },
            "suffix": suffix,
            "extra_body": {"trainingType": "globalStandard"},
        }
    assert jobs.payloads[0]["method"]["reinforcement"]["grader"]["model"] == "countdown-custom-grader"
    python_grader = jobs.payloads[1]["method"]["reinforcement"]["grader"]
    assert set(python_grader) == {"name", "type", "source"}
    assert python_grader["name"] == "countdown_python_grader"
    assert python_grader["type"] == "python"
    source_namespace = {}
    exec(python_grader["source"], source_namespace)
    assert source_namespace["grade"](
        {"output_json": {"expression": "(1 + 3) * (2 + 4)", "result": "24"}},
        {"target": "24", "nums": "[1, 2, 3, 4]"},
    ) == 5

    path = contracts["STATE_PATH"]
    submitted_bytes = path.read_bytes()
    state = contracts["load_state"]()
    assert set(state["jobs"]) == set(state["pending"]) == {"model-grader", "python-grader"}
    first_model_intent = json.loads(jobs.pending_at_create[0][1])["pending"]["model-grader"]
    assert state["pending"]["model-grader"] == first_model_intent
    for label, grader, suffix in variants:
        contracts["submit_once"](label, grader, suffix)
    assert len(jobs.payloads) == 2
    assert jobs.retrieved == ["offline-job-1", "offline-job-2"]
    assert path.read_bytes() == submitted_bytes


@pytest.mark.parametrize("invalid_name", [None, "", " ", "unstable-grader"])
def test_invalid_python_grader_blocks_the_first_model_job_post(contracts, invalid_name):
    jobs = attach_jobs(contracts)
    if invalid_name is None:
        contracts["PYTHON_GRADER"].pop("name")
    else:
        contracts["PYTHON_GRADER"]["name"] = invalid_name
    with pytest.raises(ValueError, match="stable countdown_python_grader name"):
        contracts["submit_once"](
            "model-grader", contracts["MODEL_GRADER"], "qwen36-countdown-model-grader",
        )
    assert jobs.payloads == []
    assert jobs.options == []
    assert not contracts["STATE_PATH"].exists()


def test_wrong_variant_and_recipe_drift_never_submit(contracts):
    jobs = attach_jobs(contracts)
    with pytest.raises(ValueError, match="label"):
        contracts["submit_once"](
            "model-grader", contracts["PYTHON_GRADER"], "qwen36-countdown-model-grader",
        )
    contracts["HYPERPARAMETERS"]["eval_samples"] = 3
    with pytest.raises(RuntimeError, match="recipe changed"):
        contracts["submit_once"](
            "model-grader", contracts["MODEL_GRADER"], "qwen36-countdown-model-grader",
        )
    assert jobs.payloads == []
    assert jobs.options == []


@pytest.mark.parametrize("label", ["model-grader", "python-grader"])
@pytest.mark.parametrize("failure", ["before-response", "after-response", "definitive-error", "missing-id"])
def test_pending_at_post_boundary_blocks_every_replay_and_preserves_bytes(contracts, label, failure):
    jobs = attach_jobs(contracts)
    jobs.failure = failure
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts, label)
    path = contracts["STATE_PATH"]
    original_bytes = path.read_bytes()
    assert original_bytes == jobs.pending_at_create[0][1]
    assert contracts["load_state"]()["jobs"] == {}
    assert set(contracts["load_state"]()["pending"]) == {label}
    if failure in {"after-response", "missing-id"}:
        assert jobs.accepted_ids == ["offline-job-1"]
    else:
        assert jobs.accepted_ids == []

    jobs.failure = None
    for suffix in (None, "a-different-suffix-cannot-authorize-replay"):
        with pytest.raises(RuntimeError, match="pending submission has no verified job ID") as caught:
            submit_variant(contracts, label, suffix)
        assert "Never clear, delete, or archive pending to resubmit" in str(caught.value)
        assert path.read_bytes() == original_bytes
    # A fresh notebook namespace/client must see the same durable refusal.
    restarted = load_contracts(path.parent)
    restarted_jobs = attach_jobs(restarted)
    with pytest.raises(RuntimeError, match="no POST was replayed"):
        submit_variant(restarted, label)
    assert restarted_jobs.payloads == restarted_jobs.options == []
    assert jobs.options == [{"max_retries": 0}]
    assert len(jobs.payloads) == 1
    assert jobs.retrieved == []
    assert path.read_bytes() == original_bytes
    assert not path.with_name(path.name + ".lock").exists()


@pytest.mark.parametrize("label", ["model-grader", "python-grader"])
@pytest.mark.parametrize("failure_point", ["fsync", "replace"])
def test_atomic_save_failure_before_post_prevents_create(contracts, monkeypatch, label, failure_point):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    original_bytes = b'{"fingerprint": null, "jobs": {}}\n'
    path.write_bytes(original_bytes)

    if failure_point == "fsync":
        actual_fsync = os.fsync
        calls = 0

        def fail_temporary_flush(fd):
            nonlocal calls
            calls += 1
            # First flush persists the lock; second flush is the state temp file.
            if calls == 2:
                raise OSError("offline state fsync failure")
            return actual_fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_temporary_flush)
    else:
        def fail_replace(*args, **kwargs):
            raise OSError("offline state replace failure")

        monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="pre-POST submission intent failed; no POST was sent") as caught:
        submit_variant(contracts, label)
    assert isinstance(caught.value.__cause__, OSError)
    assert jobs.payloads == jobs.options == []
    assert path.read_bytes() == original_bytes
    assert not path.with_name(path.name + ".lock").exists()
    assert not list(path.parent.glob(path.name + ".*.tmp"))


def test_unverified_noop_save_cannot_reach_post(contracts):
    jobs = attach_jobs(contracts)
    contracts["save_state"] = lambda state: None
    with pytest.raises(RuntimeError, match="pre-POST submission intent failed; no POST was sent") as caught:
        submit_variant(contracts)
    assert "not persisted exactly" in str(caught.value.__cause__)
    assert jobs.payloads == jobs.options == []
    assert not contracts["STATE_PATH"].exists()


@pytest.mark.parametrize("label", ["model-grader", "python-grader"])
def test_atomic_save_failure_after_known_id_keeps_original_pending(contracts, monkeypatch, label):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    actual_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("offline known-ID save failed")
        return actual_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(RuntimeError, match="service returned a job ID but saving it failed") as caught:
        submit_variant(contracts, label)
    assert "offline-job-1" in str(caught.value)
    original_bytes = path.read_bytes()
    assert original_bytes == jobs.pending_at_create[0][1]
    assert contracts["load_state"]()["jobs"] == {}
    assert set(contracts["load_state"]()["pending"]) == {label}
    with pytest.raises(RuntimeError, match="no POST was replayed"):
        submit_variant(contracts, label)
    assert len(jobs.payloads) == 1
    assert jobs.retrieved == []
    assert path.read_bytes() == original_bytes
    assert not list(path.parent.glob(path.name + ".*.tmp"))


@pytest.mark.parametrize("first_label", ["model-grader", "python-grader"])
def test_per_variant_pending_is_independent_and_never_overwritten(contracts, first_label):
    jobs = attach_jobs(contracts)
    second_label = "python-grader" if first_label == "model-grader" else "model-grader"
    jobs.failure = "after-response"
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts, first_label)
    first_intent = copy.deepcopy(contracts["load_state"]()["pending"][first_label])
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts, second_label)
    state = contracts["load_state"]()
    assert state["jobs"] == {}
    assert set(state["pending"]) == {first_label, second_label}
    assert state["pending"][first_label] == first_intent
    original_bytes = contracts["STATE_PATH"].read_bytes()
    jobs.failure = None
    for label in (first_label, second_label):
        with pytest.raises(RuntimeError, match="no POST was replayed"):
            submit_variant(contracts, label)
    assert len(jobs.payloads) == 2
    assert jobs.accepted_ids == ["offline-job-1", "offline-job-2"]
    assert contracts["STATE_PATH"].read_bytes() == original_bytes


def test_existing_job_resumes_while_other_variant_has_unknown_pending(contracts):
    jobs = attach_jobs(contracts)
    submit_variant(contracts, "model-grader")
    jobs.failure = "after-response"
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts, "python-grader")
    original_bytes = contracts["STATE_PATH"].read_bytes()
    # Resume uses the saved job, not newly uploaded IDs or a new suffix.
    contracts["training_file_id"] = "different-newly-uploaded-training-file"
    contracts["validation_file_id"] = "different-newly-uploaded-validation-file"
    assert submit_variant(contracts, "model-grader", "different-suffix").id == "offline-job-1"
    assert jobs.retrieved == ["offline-job-1"]
    assert len(jobs.payloads) == 2
    assert contracts["STATE_PATH"].read_bytes() == original_bytes


@pytest.mark.parametrize("label", ["model-grader", "python-grader"])
def test_verified_operator_job_id_resumes_without_removing_pending(contracts, label):
    jobs = attach_jobs(contracts)
    jobs.failure = "after-response"
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts, label)
    path = contracts["STATE_PATH"]
    state = contracts["load_state"]()
    original_intent = copy.deepcopy(state["pending"][label])
    # Simulate independently verified operator reconciliation, not automatic lookup.
    with contracts["submission_state_lock"]():
        state["jobs"][label] = jobs.accepted_ids[0]
        contracts["save_state"](state)
    reconciled_bytes = path.read_bytes()
    assert submit_variant(contracts, label).id == "offline-job-1"
    assert contracts["load_state"]()["pending"][label] == original_intent
    assert jobs.retrieved == ["offline-job-1"]
    assert len(jobs.payloads) == 1
    assert path.read_bytes() == reconciled_bytes


def test_active_lock_blocks_reentrant_create_and_retrieve(contracts):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]

    def another_submitter():
        baseline = path.read_bytes()
        with pytest.raises(RuntimeError, match="Submission state is locked"):
            submit_variant(contracts, "python-grader")
        assert path.read_bytes() == baseline

    jobs.on_create = another_submitter
    submit_variant(contracts)
    baseline = path.read_bytes()
    lock_path = path.with_name(path.name + ".lock")
    with contracts["submission_state_lock"]():
        lock_bytes = lock_path.read_bytes()
        with pytest.raises(RuntimeError, match="Submission state is locked"):
            submit_variant(contracts)
        assert lock_path.read_bytes() == lock_bytes
    assert len(jobs.payloads) == 1
    assert jobs.retrieved == []
    assert path.read_bytes() == baseline


def test_stale_lock_is_not_removed_or_bypassed(contracts):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    lock_path = path.with_name(path.name + ".lock")
    lock_bytes = b'offline interrupted owner; operator reconciliation required\n'
    lock_path.write_bytes(lock_bytes)
    with pytest.raises(RuntimeError, match="Do not break a live lock"):
        submit_variant(contracts)
    assert lock_path.read_bytes() == lock_bytes
    assert not path.exists()
    assert jobs.payloads == jobs.retrieved == jobs.options == []


def test_failed_retrieve_preserves_state_and_releases_lock(contracts):
    jobs = attach_jobs(contracts)
    submit_variant(contracts)
    path = contracts["STATE_PATH"]
    original_bytes = path.read_bytes()
    jobs.retrieve_failure = TimeoutError("offline retrieval timeout")
    with pytest.raises(TimeoutError, match="offline retrieval timeout"):
        submit_variant(contracts)
    assert len(jobs.payloads) == 1
    assert path.read_bytes() == original_bytes
    assert not path.with_name(path.name + ".lock").exists()


def test_sdk_cannot_mutate_saved_payload_or_recipe_through_aliases(contracts):
    jobs = attach_jobs(contracts)
    jobs.mutate_payload = True
    submit_variant(contracts)
    state = contracts["load_state"]()
    assert state["pending"]["model-grader"]["payload"] == jobs.payloads[0]
    assert contracts["HYPERPARAMETERS"]["eval_samples"] == 2
    assert contracts["recipe_fingerprint"]() == contracts["RUN_FINGERPRINT"]


@pytest.mark.parametrize("current_deployment", ["ready", "missing", "low-tpm"])
def test_pending_model_submission_requires_read_only_deployment_validation(contracts, current_deployment):
    jobs = attach_jobs(contracts)
    jobs.failure = "after-response"
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts)
    path = contracts["STATE_PATH"]
    original_bytes = path.read_bytes()
    current = None if current_deployment == "missing" else deployment(
        contracts, tpm=99_999 if current_deployment == "low-tpm" else 100_000,
    )
    arm = FakeArm(contracts, current)
    contracts["run_az_json"] = arm
    contracts["ALLOW_GRADER_DEPLOYMENT_UPDATE"] = False
    if current_deployment == "ready":
        assert contracts["ensure_grader_deployment"]()["name"] == contracts["GRADER_DEPLOYMENT"]
    else:
        with pytest.raises(RuntimeError):
            contracts["ensure_grader_deployment"]()
    assert arm.create_calls == 0
    assert path.read_bytes() == original_bytes
    assert not contracts["GRADER_DEPLOYMENT_STATE_PATH"].exists()


@pytest.mark.parametrize(("field", "value"), [
    ("training_file_id", None), ("validation_file_id", " "),
])
def test_invalid_uploaded_file_ids_never_record_intent_or_submit(contracts, field, value):
    jobs = attach_jobs(contracts)
    contracts[field] = value
    with pytest.raises(RuntimeError, match="Invalid or unverified pending submission intent"):
        submit_variant(contracts)
    assert jobs.payloads == jobs.options == []
    assert not contracts["STATE_PATH"].exists()


def test_changed_local_dataset_blocks_pending_and_post(contracts):
    jobs = attach_jobs(contracts)
    contract = contracts["DATASETS"]["training"]
    changed_path = contracts["OUTPUT_ROOT"] / "offline-changed-training.jsonl"
    changed_path.write_bytes(contract["path"].read_bytes() + b"\n")
    contracts["DATASETS"]["training"] = {**contract, "path": changed_path}
    with pytest.raises(ValueError, match="training integrity mismatch"):
        submit_variant(contracts)
    assert jobs.payloads == jobs.options == []
    assert not contracts["STATE_PATH"].exists()


@pytest.mark.parametrize("label", ["model-grader", "python-grader"])
def test_verified_legacy_job_only_state_resumes_without_rewriting(contracts, label):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    original_bytes = (json.dumps({
        "fingerprint": contracts["RUN_FINGERPRINT"],
        "jobs": {label: "offline-legacy-job"},
        "job_endpoints": {label: contracts["PROJECT_ENDPOINT"]},
    }, indent=4) + "\n").encode("utf-8")
    path.write_bytes(original_bytes)
    assert submit_variant(contracts, label).id == "offline-legacy-job"
    assert jobs.payloads == jobs.options == []
    assert jobs.retrieved == ["offline-legacy-job"]
    assert path.read_bytes() == original_bytes
    assert contracts["load_state"]()["pending"] == {}


def test_new_variant_preserves_verified_legacy_job_and_endpoint(contracts):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    path.write_text(json.dumps({
        "fingerprint": contracts["RUN_FINGERPRINT"],
        "jobs": {"model-grader": "offline-legacy-job"},
        "job_endpoints": {"model-grader": contracts["PROJECT_ENDPOINT"]},
    }), encoding="utf-8")
    submit_variant(contracts, "python-grader")
    state = contracts["load_state"]()
    assert state["jobs"] == {
        "model-grader": "offline-legacy-job", "python-grader": "offline-job-1",
    }
    assert state["job_endpoints"] == {"model-grader": contracts["PROJECT_ENDPOINT"]}
    assert set(state["pending"]) == {"python-grader"}
    assert submit_variant(contracts, "model-grader").id == "offline-legacy-job"
    assert len(jobs.payloads) == 1


@pytest.mark.parametrize("invalid_state", [
    [], {}, {"jobs": []}, {"jobs": {"model-grader": "offline-job"}},
    {"jobs": {}, "pending": {}},
    {"schema_version": 1, "jobs": {}, "pending": {}},
    {"schema_version": True, "jobs": {}, "pending": {}},
    {"schema_version": 2, "jobs": {}},
    {"schema_version": 2, "jobs": {}, "pending": []},
    {"jobs": {"unknown-variant": "offline-job"}},
    {"jobs": {"model-grader": ""}}, {"jobs": {"model-grader": " "}},
    {"jobs": {"model-grader": 123}},
    {"jobs": {}, "fingerprint": ["not-a-fingerprint"]},
    {"jobs": {}, "job_endpoints": []},
    {"jobs": {}, "job_endpoints": {"model-grader": "another-project"}},
    {"jobs": {}, "pending_intents": {"model-grader": "unrecognized evidence"}},
])
def test_invalid_or_unverified_submission_state_is_never_rewritten(contracts, invalid_state):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    original_bytes = (json.dumps(invalid_state) + "\n").encode("utf-8")
    path.write_bytes(original_bytes)
    with pytest.raises(RuntimeError):
        submit_variant(contracts)
    assert jobs.payloads == jobs.retrieved == jobs.options == []
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("invalid_bytes", [b'{"jobs":', b'\xff\xfe-not-utf8'])
def test_unreadable_submission_state_fails_closed_without_repair(contracts, invalid_bytes):
    jobs = attach_jobs(contracts)
    path = contracts["STATE_PATH"]
    path.write_bytes(invalid_bytes)
    with pytest.raises(RuntimeError, match="Invalid submission state"):
        submit_variant(contracts)
    assert jobs.payloads == jobs.options == []
    assert path.read_bytes() == invalid_bytes


@pytest.mark.parametrize("change", [
    "missing-pending-field", "legacy-pending", "pending-schema", "pending-type",
    "intent-type", "intent-schema", "intent-record-type", "intent-extra-field",
    "intent-label", "intent-endpoint", "intent-fingerprint", "intent-timestamp",
    "intent-payload", "intent-payload-hash", "payload-rehashed", "payload-bool",
    "file-id", "file-hash", "file-bytes", "file-rows", "file-evidence-type",
    "state-fingerprint", "unverified-fingerprint", "unknown-label", "known-id-endpoint",
])
def test_tampered_pending_evidence_blocks_all_submission_without_byte_changes(contracts, change):
    jobs = attach_jobs(contracts)
    jobs.failure = "after-response"
    with pytest.raises(RuntimeError, match="pending intent was retained"):
        submit_variant(contracts)
    path = contracts["STATE_PATH"]
    state = json.loads(path.read_text(encoding="utf-8"))
    intent = state["pending"]["model-grader"]
    if change == "missing-pending-field":
        state.pop("pending")
    elif change == "legacy-pending":
        state.pop("schema_version")
    elif change == "pending-schema":
        state["schema_version"] = 3
    elif change == "pending-type":
        state["pending"] = []
    elif change == "intent-type":
        state["pending"]["model-grader"] = None
    elif change == "intent-schema":
        intent["schema_version"] = True
    elif change == "intent-record-type":
        intent["record_type"] = "job_created"
    elif change == "intent-extra-field":
        intent["unverified_job_id"] = "offline-job-1"
    elif change == "intent-label":
        intent["label"] = "python-grader"
    elif change == "intent-endpoint":
        intent["project_endpoint"] = "https://another-project.invalid"
    elif change == "intent-fingerprint":
        intent["run_fingerprint"] = "another-recipe"
    elif change == "intent-timestamp":
        intent["recorded_at"] = "2026-10-20T00:00:00"
    elif change == "intent-payload":
        intent["payload"] = None
    elif change == "intent-payload-hash":
        intent["payload_sha256"] = "0" * 64
    elif change in {"payload-rehashed", "payload-bool"}:
        intent["payload"]["method"]["reinforcement"]["hyperparameters"]["n_epochs"] = (
            True if change == "payload-bool" else 2
        )
        intent["payload_sha256"] = contracts["canonical_json_sha256"](intent["payload"])
    elif change.startswith("file-"):
        if change == "file-evidence-type":
            intent["files"] = []
        else:
            field = {
                "file-id": "file_id", "file-hash": "sha256", "file-bytes": "bytes", "file-rows": "rows",
            }[change]
            intent["files"]["training"][field] = "unverified"
    elif change == "state-fingerprint":
        state["fingerprint"] = "another-recipe"
    elif change == "unverified-fingerprint":
        state["fingerprint"] = None
    elif change == "unknown-label":
        state["pending"]["unknown-variant"] = state["pending"].pop("model-grader")
    elif change == "known-id-endpoint":
        state["jobs"]["model-grader"] = "offline-job-1"
        state["job_endpoints"] = {"model-grader": "https://another-project.invalid"}
    else:
        raise AssertionError(change)
    original_bytes = (json.dumps(state, indent=4) + "\n").encode("utf-8")
    path.write_bytes(original_bytes)
    with pytest.raises(RuntimeError):
        submit_variant(contracts)
    assert len(jobs.payloads) == 1
    assert jobs.retrieved == []
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("manifest_relative", ["data/hashes.json", "graders/hashes.json"])
def test_every_manifest_entry_matches_actual_canonical_lf_bytes(manifest_relative):
    manifest = json.loads((ROOT / manifest_relative).read_text(encoding="utf-8"))
    base = ROOT if manifest_relative.startswith("data/") else ROOT / "graders"
    for entry in manifest["artifacts"]:
        content = (base / entry["path"]).read_bytes().replace(b"\r\n", b"\n")
        actual_bytes = len(content)
        actual_sha256 = hashlib.sha256(content).hexdigest()
        evidence = f"{entry['path']}: canonical-LF bytes={actual_bytes}, sha256={actual_sha256}"
        assert actual_bytes == entry["bytes"], evidence
        assert actual_sha256 == entry["sha256"], evidence
        if "rows" in entry:
            assert len([line for line in content.splitlines() if line.strip()]) == entry["rows"]


def test_duplicate_grader_manifest_entries_agree_and_preserved_grader_remains_unmodified():
    graders = json.loads((ROOT / "graders" / "hashes.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "data" / "hashes.json").read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in data["artifacts"]}
    for entry in graders["artifacts"]:
        if entry["path"].startswith("preserved/"):
            duplicate = entries["graders/" + entry["path"]]
            for key in ("source", "bytes", "sha256"):
                assert entry[key] == duplicate[key]
    content = (ROOT / "graders" / "preserved" / "countdown-model-score-grader.json").read_bytes()
    content = content.replace(b"\r\n", b"\n")
    assert len(content) == 1731
    assert hashlib.sha256(content).hexdigest() == (
        "3d6e6983a70b7861ae7603f5f3d679ff83598134513685064ae065cc2d396794"
    )
    preserved = json.loads(content)
    assert preserved["name"] == "custom_grader"
    assert preserved["model"] == "gpt-5.4-mini"


def test_training_validation_and_evaluation_quantities_are_unchanged(contracts):
    for label, expected_rows in (("training", 100), ("validation", 50), ("evaluation_provenance", 100)):
        rows, evidence = contracts["validate_jsonl"](label, contracts["DATASETS"][label])
        assert len(rows) == evidence["rows"] == expected_rows
    assert contracts["MODEL_PUBLISHER"] == "Alibaba"
    assert contracts["MODEL_VERSION"] == "1"


def test_grader_names_are_validated_before_deployment_cell():
    sources = code_sources()
    contracts_index = next(index for index, source in enumerate(sources) if "def recipe_fingerprint" in source)
    deployment_index = next(index for index, source in enumerate(sources) if "grader_deployment = ensure_grader_deployment()" in source)
    assert contracts_index < deployment_index