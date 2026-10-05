from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import nbformat
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


def test_git_candidates_exclude_runtime_paths() -> None:
    root = _root()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "cookbooks"],
        cwd=root.parent,
        check=True,
        capture_output=True,
    )
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        if not (root.parent / relative).exists():
            continue
        assert not set(relative.parts) & LOCAL_DIRECTORIES, f"Runtime path staged: {relative}"
        assert relative.name != ".env", f"Populated configuration staged: {relative}"
        assert not relative.name.endswith((".pyc", ".log", ".tmp", ".bak", ".executed.ipynb"))
