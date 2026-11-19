"""Offline checks for the public cookbook's training names and entrypoints."""

import ast
import json
from pathlib import Path
import re
import tomllib
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
OLD_PACKAGE = "interactive_" + "training"
OLD_MANAGED = "managed_" + "fine_tuning"
PACKAGE_ROOT = ROOT / "programmed_training" / "programmed_training"


class TrainingRenameTests(unittest.TestCase):
    def test_track_paths_and_package_metadata(self):
        self.assertTrue(PACKAGE_ROOT.is_dir())
        self.assertTrue((ROOT / "managed_training" / "catalog.yml").is_file())
        self.assertFalse((ROOT / OLD_PACKAGE).exists())
        self.assertFalse((ROOT / OLD_MANAGED).exists())
        metadata = tomllib.loads(
            (ROOT / "programmed_training" / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["project"]["name"], "programmed-training")
        self.assertEqual(
            metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"],
            ["programmed_training"],
        )
        self.assertIn(
            "azure-ai-finetuningsessions==1.0.0b2", metadata["project"]["dependencies"]
        )

    def test_python_imports_use_existing_renamed_modules(self):
        paths = [
            path
            for track in ("programmed_training", "managed_training")
            for path in (ROOT / track).rglob("*.py")
            if "__pycache__" not in path.parts
        ]
        self.assertGreater(len(paths), 150)
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    with self.subTest(path=path.relative_to(ROOT), module=module):
                        self.assertNotEqual(module.split(".")[0], OLD_PACKAGE)
                        if module == "programmed_training" or module.startswith(
                            "programmed_training."
                        ):
                            target = PACKAGE_ROOT.joinpath(*module.split(".")[1:])
                            self.assertTrue(
                                target.is_dir() or target.with_suffix(".py").is_file()
                            )

    def test_repository_and_dataset_navigation_resolves(self):
        paths = [ROOT / "README.md", ROOT / "AGENTS.md"] + sorted(
            (ROOT / "Sample_Datasets").rglob("*.md")
        )
        self.assertGreater(len(paths), 2)
        for path in paths:
            source = path.read_text(encoding="utf-8")
            source = re.sub(r"(?ms)^```.*?^```\s*$", "", source)
            for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", source):
                target = urlsplit(link)
                if target.scheme or target.netloc or not target.path:
                    continue
                with self.subTest(path=path.relative_to(ROOT), link=link):
                    self.assertTrue((path.parent / unquote(target.path)).exists())

    def test_programmed_ci_uses_renamed_paths(self):
        workflows = ROOT / ".github" / "workflows"
        self.assertFalse((workflows / ("interactive-" + "training-checks.yml")).exists())
        source = (workflows / "programmed-training-checks.yml").read_text(encoding="utf-8")
        self.assertIn("working-directory: programmed_training", source)
        self.assertIn("cache-dependency-path: programmed_training/pyproject.toml", source)
        self.assertIn("'programmed_training/**'", source)
        self.assertIn("python -m pytest -q --tb=short", source)
        self.assertNotIn(OLD_PACKAGE, source)

    def test_managed_ci_checks_root_navigation(self):
        source = (ROOT / ".github/workflows/documentation-checks.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("working-directory: managed_training", source)
        self.assertIn("'managed_training/**'", source)
        self.assertIn('python -m unittest discover -s tests -p "test_*.py"', source)
        self.assertNotIn(OLD_MANAGED, source)
        self.assertNotIn(OLD_PACKAGE, source)

    def test_paid_smoke_keeps_existing_opt_in_and_environment(self):
        source = (ROOT / ".github/workflows/smoke-testing.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("if: vars.ENABLE_DAILY_TRAINING == 'true'", source)
        self.assertIn("environment: interactive-training-smoke", source)
        self.assertIn("working-directory: programmed_training", source)
        self.assertIn("cron: '0 0 * * *'", source)

    def test_notebook_sources_have_no_retired_paths(self):
        paths = [
            path
            for track in ("programmed_training", "managed_training")
            for path in (ROOT / track).rglob("*.ipynb")
        ]
        self.assertGreater(len(paths), 15)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8-sig"))
            for index, cell in enumerate(notebook["cells"], 1):
                with self.subTest(path=path.relative_to(ROOT), cell=index):
                    source = "".join(cell.get("source", []))
                    self.assertNotIn(OLD_PACKAGE, source)
                    self.assertNotIn(OLD_MANAGED, source)


if __name__ == "__main__":
    unittest.main()
