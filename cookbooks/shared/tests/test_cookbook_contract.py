from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import nbformat
import yaml

VARIABILITY_NOTE = (
    "Actual results may vary by model version, data, configuration, "
    "region availability, and service conditions."
)
EXTERNAL_ENV_NAMES = {"HF_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _root() -> Path:
    return Path(__file__).parents[2]


def _source_text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


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
            ast.parse(_source_text(cell.get("source", "")))


def test_environment_templates_are_empty_and_portable() -> None:
    for path in _root().glob("**/.env.template"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name, separator, value = stripped.partition("=")
            assert separator
            assert not value.strip()
            assert name.startswith("FOUNDRY_") or name in EXTERNAL_ENV_NAMES


def test_dependencies_and_local_links_are_centralized() -> None:
    root = _root()
    assert not list(root.glob("**/requirements*.txt"))

    documents = list(root.glob("**/*.md")) + list(root.glob("**/*.ipynb"))
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
