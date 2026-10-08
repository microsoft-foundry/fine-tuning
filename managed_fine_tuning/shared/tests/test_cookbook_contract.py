from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import nbformat
import pytest
import yaml

VARIABILITY_NOTE = (
    "Actual results may vary by model version, data, configuration, "
    "region availability, and service conditions."
)
EXTERNAL_ENV_NAMES = {"HF_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "RFT_TOOL_SERVER_URL"}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
LOCAL_DIRECTORIES = {
    "outputs", "downloads", "build", "dist", "__pycache__", ".pytest_cache",
    ".ipynb_checkpoints", ".venv", ".git"
}


def _root() -> Path:
    return Path(__file__).parents[2]


def _source_text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def _customer_files(root: Path):
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories
            if name not in LOCAL_DIRECTORIES and not name.endswith(".egg-info")
        ]
        for filename in filenames:
            if filename == ".env":
                continue
            yield Path(directory) / filename


def test_demo_folder_and_notebook_contracts() -> None:
    root = _root()
    catalog = yaml.safe_load((root / "catalog.yml").read_text(encoding="utf-8"))

    for demo in catalog["demos"]:
        demo_root = root / demo["path"]
        assert (demo_root / "README.md").is_file()
        assert (demo_root / "demo.yaml").is_file()
        assert (demo_root / ".env.template").is_file()
        assert (demo_root / "data").is_dir()
        assert (demo_root / "assets").is_dir()

        notebooks = list((demo_root / "notebooks").glob("*.ipynb"))
        assert notebooks == [demo_root / "notebooks" / "demo.ipynb"]

        raw = json.loads(notebooks[0].read_text(encoding="utf-8-sig"))
        nbformat.validate(nbformat.from_dict(raw))
        cell_ids = [cell.get("id") for cell in raw["cells"]]
        assert all(cell_ids)
        assert len(cell_ids) == len(set(cell_ids))

        combined_text = "\n".join(
            _source_text(cell.get("source", "")) for cell in raw["cells"]
        )
        assert VARIABILITY_NOTE in combined_text
        for cell in raw["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
            assert not set(cell.get("metadata", {})) & {"execution", "ExecuteTime"}
            ast.parse(_source_text(cell.get("source", "")))


def test_environment_templates_are_empty_and_portable() -> None:
    for path in _customer_files(_root()):
        if path.name != ".env.template":
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name, separator, value = stripped.partition("=")
            assert separator
            assert not value.strip(), f"Populated template value in {path}: {name}"
            assert name.startswith("FOUNDRY_") or name in EXTERNAL_ENV_NAMES


def test_notebook_execution_switches_are_removed() -> None:
    configs = {"FOUNDRY_RUN_" + name for name in ("PAID_JOBS", "DEPLOYMENT", "INFERENCE")}
    variables = {"RUN_TRAINING", "RUN_DEPLOYMENT", "RUN_INFERENCE", "SUBMIT_TRAINING"}
    for path in _customer_files(_root()):
        if path.name == ".env.template":
            keys = {
                line.partition("=")[0].strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            assert not keys & configs
        elif path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8-sig"))
            for cell in notebook["cells"]:
                source = _source_text(cell.get("source", ""))
                assert not any(config in source for config in configs)
                if cell["cell_type"] == "code":
                    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
                    assert not names & variables


def test_setup_text_has_no_retired_paths_or_phantom_switches() -> None:
    forbidden = (
        "cookbooks/requirements.lock",
        "cookbooks\\requirements.lock",
        "cookbooks/shared",
        "cookbooks\\shared",
        "paid jobs default to false",
        "blank means false. explicitly set true to upload and submit",
        "blank means false. explicitly set true to submit both",
    )
    for path in _customer_files(_root()):
        if path.suffix not in {".md", ".ipynb", ".template"}:
            continue
        text = path.read_text(encoding="utf-8-sig").casefold()
        for phrase in forbidden:
            assert phrase not in text, f"Retired setup guidance in {path}: {phrase}"


def test_first_sft_remains_pinned_to_canonical_data() -> None:
    root = _root()
    notebook_path = (
        root
        / "00-foundations"
        / "02-first-sft-bug-detection"
        / "notebooks"
        / "demo.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    source = "\n".join(_source_text(cell.get("source", "")) for cell in notebook["cells"])

    assert "FOUNDRY_TRAINING_DATA_PATH" not in source
    assert "FOUNDRY_VALIDATION_DATA_PATH" not in source
    assert 'TRAINING_PATH = DATA_DIR / "bug-detection-train.jsonl"' in source
    assert 'VALIDATION_PATH = DATA_DIR / "bug-detection-validation.jsonl"' in source
    assert '"rows": 224' in source
    assert '"rows": 20' in source


def test_adaptation_guide_uses_ignored_outputs_and_explicit_cwd() -> None:
    guide = (_root() / "ADAPT_YOUR_DATA.md").read_text(encoding="utf-8")
    assert "Set-Location managed_fine_tuning" in guide
    assert "cd managed_fine_tuning/" in guide
    assert "Set-Location ..\\.." in guide
    assert "cd ../.." in guide
    assert "outputs\\adaptations\\my-domain-v1" in guide
    assert "outputs/adaptations/my-domain-v1" in guide
    assert "--canonical-output-dir" in guide


def test_cancellation_guide_reconstructs_verified_context() -> None:
    guide = (_root() / "POST_TRAINING.md").read_text(encoding="utf-8")
    assert 'Path("outputs/submission-state.json")' in guide
    assert 'load_foundry_config(".env")' in guide
    assert "config.project_endpoint != saved_endpoint" in guide
    assert "confirmed_job_id != job_id" in guide
    assert "with create_project_context(config) as context:" in guide
    assert "fine_tuning.jobs.cancel(job_id)" in guide
    assert 'input(f"Type DELETE {file_id}' in guide
    assert "openai_client.files.retrieve(file_id)" in guide
    assert "openai_client.files.delete(file_id)" in guide


def test_adaptation_upload_evidence_matches_notebook_normalization() -> None:
    root = _root()
    validator = (root / "shared" / "validate_dataset.py").read_text(encoding="utf-8")
    notebook = (
        root
        / "00-foundations"
        / "02-first-sft-bug-detection"
        / "notebooks"
        / "demo.ipynb"
    ).read_text(encoding="utf-8")
    normalization = 'replace(b\\"\\\\r\\\\n\\", b\\"\\\\n\\")'
    assert 'replace(b"\\r\\n", b"\\n")' in validator
    assert normalization in notebook


def test_customer_content_excludes_test_run_failure_narratives() -> None:
    for path in _customer_files(_root()):
        if path.suffix not in {".md", ".ipynb", ".json", ".yml", ".yaml", ".template"}:
            continue
        text = path.read_text(encoding="utf-8-sig").casefold()
        for phrase in (
            "tested subscription", "blocked_by_quota", "known source-data blocker",
            "training is blocked", "recent preprocessing failure",
        ):
            assert phrase not in text, f"Internal run narrative in {path}"


def test_byte_pinned_inputs_disable_git_text_conversion() -> None:
    root = _root()
    inputs = list((root / "05-data-and-distillation").glob("*/data/*.jsonl"))
    inputs += list(
        (root / "02-multimodal-sft" / "01-chart-reasoning" / "data" / "preserved").glob("*.jsonl")
    )
    assert inputs
    result = subprocess.run(
        ["git", "check-attr", "-z", "text", "--", *[
            str(path.relative_to(root.parent)) for path in inputs
        ]],
        cwd=root.parent, check=True, capture_output=True,
    )
    fields = result.stdout.rstrip(b"\0").split(b"\0")
    assert len(fields) == len(inputs) * 3
    for index in range(0, len(fields), 3):
        assert fields[index + 1:index + 3] == [b"text", b"unset"], fields[index]


def test_dependencies_and_local_links_are_centralized() -> None:
    root = _root()
    customer_files = list(_customer_files(root))
    assert not [
        path for path in customer_files
        if path.name.startswith("requirements") and path.suffix == ".txt"
    ]

    documents = [path for path in customer_files if path.suffix in {".md", ".ipynb"}]
    for path in documents:
        if path.suffix == ".ipynb":
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            texts = [
                _source_text(cell.get("source", ""))
                for cell in raw["cells"]
                if cell["cell_type"] == "markdown"
            ]
        else:
            texts = [path.read_text(encoding="utf-8-sig")]

        for text in texts:
            for target in MARKDOWN_LINK.findall(text):
                local_target = target.strip().split("#", 1)[0]
                if not local_target or local_target.startswith(
                    ("data:", "http://", "https://", "mailto:")
                ):
                    continue
                assert (path.parent / local_target.replace("%20", " ")).exists(), (
                    f"Broken local link in {path}: {target}"
                )


def test_customer_content_excludes_local_execution_artifacts() -> None:
    forbidden_names = {"live_validate.py", "execute_notebook.py", "prepare_execution.py"}
    private_details = re.compile(
        r"prakharg-demo-\d{4}|ftjob-[0-9a-f]{32}|"
        r"(?:C:\\\\|C:/)(?:Data[/\\]AIP|Users[/\\])|"
        r"/subscriptions/[0-9a-f]{8}-[0-9a-f-]{27,}"
    )
    for path in _customer_files(_root()):
        assert path.name not in forbidden_names, f"Local runner shipped: {path}"
        assert not path.name.endswith((".executed.ipynb", ".log", ".tmp", ".bak"))
        if path.suffix not in {
            ".py", ".md", ".ipynb", ".yaml", ".yml", ".json", ".jsonl",
            ".csv", ".txt", ".toml", ".template",
        }:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".ipynb":
            notebook = json.loads(text)
            text = "\n".join(_source_text(cell["source"]) for cell in notebook["cells"])
            assert not re.search(r"(?:from|import)\s+live_validate\b", text)
            assert "live_validate.py" not in text
        # This test's regex intentionally describes the excluded patterns.
        if path == Path(__file__):
            continue
        assert not private_details.search(text), f"Local runtime details shipped: {path}"


def test_retail_comparison_is_resumable_and_safely_scores_blocked_tools() -> None:
    notebook_path = (
        _root()
        / "04-agentic-rft"
        / "02-retail-agent-capstone"
        / "notebooks"
        / "demo.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    source = "\n".join(
        _source_text(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert source.count("policy=COMPARISON_RETRY_POLICY") >= 3
    assert "sft-comparison-state.json" in source
    assert "rft-comparison-state.json" in source
    assert "save_comparison_result" in source
    assert "'context': context" in source
    assert "'status': 'blocked' if blocked else 'graded'" in source
    assert "0.0 if blocked else policy_score" in source
    assert "'attempted_tools'" in source
    assert "'executed_tools'" in source
    assert "'blocked_tools'" in source
    assert (
        "item['status'] == 'graded' and "
        "item['executed_tools'] == item['reference_tools']"
    ) in source


def _notebook_function_namespace(notebook_path: Path, function_names: set[str], **values):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    source = "\n".join(
        _source_text(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = dict(values)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(notebook_path), "exec"), namespace)
    return namespace


def test_retail_comparison_checkpoint_context_and_blocked_score(tmp_path: Path) -> None:
    notebook_path = (
        _root()
        / "04-agentic-rft"
        / "02-retail-agent-capstone"
        / "notebooks"
        / "demo.ipynb"
    )
    policy_calls = []

    def policy_score(row, output):
        policy_calls.append((row, output))
        return 1.0

    namespace = _notebook_function_namespace(
        notebook_path,
        {"load_comparison_results", "save_comparison_result", "score_rft_outcome"},
        json=json,
        policy_score=policy_score,
    )
    checkpoint = tmp_path / "comparison.json"
    first_context = {"contract_version": 2, "input_sha256": "first"}
    second_context = {"contract_version": 2, "input_sha256": "second"}

    results = namespace["save_comparison_result"](
        checkpoint, first_context, "base", {"score": 0.5}
    )
    assert results == {"base": {"score": 0.5}}
    results = namespace["save_comparison_result"](
        checkpoint, first_context, "fine_tuned", {"score": 0.75}
    )
    assert results == {
        "base": {"score": 0.5},
        "fine_tuned": {"score": 0.75},
    }
    assert namespace["load_comparison_results"](checkpoint, second_context) == {}

    outcome = namespace["score_rft_outcome"](
        {"reference_tool_calls": ["calculate"]},
        {
            "output": None,
            "attempted_tools": ["calculate", "cancel_order"],
            "executed_tools": [],
            "blocked_tools": ["cancel_order"],
        },
    )
    assert outcome["score"] == 0.0
    assert outcome["status"] == "blocked"
    assert outcome["attempted_tools"] == ["calculate", "cancel_order"]
    assert outcome["executed_tools"] == []
    assert policy_calls == []


def test_countdown_retries_transient_cli_credential_timeouts() -> None:
    notebook_path = (
        _root()
        / "03-core-rft"
        / "01-rft-countdown"
        / "notebooks"
        / "demo.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    source = "\n".join(
        _source_text(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert '"CredentialUnavailableError"' in source
    assert '"TimeoutExpired"' in source
    assert 'GRADER_MODEL = "gpt-5.4-mini"' in source
    assert 'MODEL_GRADER["model"] != "gpt-5.4-mini"' in source
    assert 'GRADER_QUOTA_NAME = "OpenAI.DataZoneStandard.gpt-5.4-mini"' in source
    assert 'GRADER_CAPACITY_SETTING = os.getenv("FOUNDRY_GRADER_CAPACITY", "")' in source
    assert 'positive_integer("FOUNDRY_GRADER_MIN_TPM", 100_000)' in source
    assert "validate_existing_grader" in source
    assert "choose_grader_capacity" in source
    assert "grader_job_is_recorded" in source
    assert "grader_deployment = ensure_grader_deployment()" in source


def test_countdown_capacity_selection_and_collision_checks() -> None:
    notebook_path = (
        _root()
        / "03-core-rft"
        / "01-rft-countdown"
        / "notebooks"
        / "demo.ipynb"
    )
    namespace = _notebook_function_namespace(
        notebook_path,
        {"choose_grader_capacity", "validate_existing_grader"},
        math=__import__("math"),
        GRADER_MIN_TPM=100_000,
        GRADER_MODEL="gpt-5.4-mini",
        GRADER_MODEL_VERSION="2026-03-17",
        GRADER_DEPLOYMENT="grader",
        GRADER_DEPLOYMENT_SKU="DataZoneStandard",
    )

    assert namespace["choose_grader_capacity"](200, 50, 10, "max") == 160
    assert namespace["choose_grader_capacity"](200, 50, 10, "120") == 120
    with pytest.raises(ValueError, match="at least 100"):
        namespace["choose_grader_capacity"](200, 50, 10, "99")
    with pytest.raises(RuntimeError, match="exceeds"):
        namespace["choose_grader_capacity"](200, 50, 10, "161")

    matching = {
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": "gpt-5.4-mini",
                "version": "2026-03-17",
            }
        },
        "sku": {"name": "DataZoneStandard", "capacity": 120},
    }
    namespace["validate_existing_grader"](matching)
    mismatched = json.loads(json.dumps(matching))
    mismatched["sku"]["name"] = "GlobalStandard"
    with pytest.raises(RuntimeError, match="Expected DataZoneStandard"):
        namespace["validate_existing_grader"](mismatched)


def test_countdown_and_retail_customer_copy_has_no_run_history() -> None:
    root = _root()
    paths = [
        root / "03-core-rft" / "01-rft-countdown" / "README.md",
        root / "03-core-rft" / "01-rft-countdown" / "notebooks" / "demo.ipynb",
        root / "04-agentic-rft" / "02-retail-agent-capstone" / "README.md",
        root / "04-agentic-rft" / "02-retail-agent-capstone" / "notebooks" / "demo.ipynb",
    ]
    forbidden = (
        "successful demo submission",
        "previous session",
        "fake probe",
        "verified locally",
        "local test",
        "historical run output",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8-sig").casefold()
        for phrase in forbidden:
            assert phrase not in text, f"Prior-run narrative in {path}: {phrase}"


def test_git_candidates_exclude_runtime_paths() -> None:
    root = _root()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", root.name],
        cwd=root.parent,
        check=True,
        capture_output=True,
    )
    candidates = [raw_path for raw_path in result.stdout.split(b"\0") if raw_path]
    assert candidates, f"No git candidates found under {root.name}"
    for raw_path in candidates:
        relative = Path(raw_path.decode("utf-8"))
        if not (root.parent / relative).exists():
            continue
        assert not set(relative.parts) & LOCAL_DIRECTORIES, f"Runtime path staged: {relative}"
        assert relative.name != ".env", f"Populated configuration staged: {relative}"
        assert not relative.name.endswith((".pyc", ".log", ".tmp", ".bak", ".executed.ipynb"))
