"""The cookbook namespace changes without renaming its separate Azure SDK."""

import inspect
import json
from pathlib import Path
import tomllib

import programmed_training
from programmed_training.utils.code_state import code_state


ROOT = Path(__file__).resolve().parents[1]
OLD_PACKAGES = ("interactive_" + "training", "interactive_" + "post_training")


def test_cookbook_distribution_and_wheel_use_current_namespace():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "programmed-training"
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "programmed_training"
    ]
    assert "azure-ai-finetuningsessions==1.0.0b1" in metadata["project"]["dependencies"]
    assert Path(programmed_training.__file__).resolve() == ROOT / "programmed_training" / "__init__.py"
    assert all(not (ROOT / package).exists() for package in OLD_PACKAGES)


def test_code_snapshot_discovers_renamed_package_by_default():
    assert inspect.signature(code_state).parameters["modules"].default == (
        "programmed_training",
    )


def test_documentation_and_modules_have_no_stale_package_references():
    files = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "dashboard_server.py"]
    for directory in ("programmed_training", "docs", "tests", "notebooks"):
        files.extend(
            path for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".ipynb"}
        )
    assert len(files) >= 200, "Cookbook rename inventory must not pass vacuously"
    for path in files:
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        else:
            text = path.read_text(encoding="utf-8")
        if path == ROOT / "docs" / "troubleshooting.md":
            text = text.split("### Updating an older cookbook checkout\n", 1)[0] + text.split(
                "### Replacing an older SDK preview\n", 1
            )[1]
        assert all(package not in text for package in OLD_PACKAGES), path.relative_to(ROOT)