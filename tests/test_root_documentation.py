"""Root entry-point contracts, with no SDK, network, or paid operations."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def prose(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def anchors(text):
    result = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', text))
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+)$", prose(text), re.MULTILINE):
        slug = re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        result.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return result


def check_links(document):
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", prose(document.read_text(encoding="utf-8"))):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        destination = document.parent / unquote(parsed.path) if parsed.path else document
        if not destination.exists():
            raise AssertionError(f"{document.name}: missing target {target}")
        if parsed.fragment and (
            not destination.is_file()
            or unquote(parsed.fragment) not in anchors(destination.read_text(encoding="utf-8"))
        ):
            raise AssertionError(f"{document.name}: missing anchor {target}")


def model_rows(text):
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines() if line.startswith("|")
    ]


def check_models(readme, registry):
    available, legacy = set(), set()
    for row in model_rows(registry):
        if len(row) == 3 and row[1].startswith("`"):
            name = row[1].strip("`").rsplit("/", 1)[-1].casefold()
            if row[2] == "Available":
                available.add(name)
            elif "Legacy" in row[2]:
                legacy.add(name)
    assert available and legacy, "Empty current/legacy registry inventory"
    assert not available.intersection(legacy), "Conflicting registry entries"
    listed = {}
    for row in model_rows(readme):
        if len(row) == 3 and row[0].startswith("`"):
            name = re.search(r"`([^`]+)`", row[0]).group(1).casefold().replace(" ", "-")
            assert name not in listed, f"Duplicate model: {name}"
            assert row[2] in {"✅", "❌"}, f"Invalid support marker: {name}"
            listed[name] = row[2]
    assert listed, "Empty root model inventory"
    assert available.issubset(listed), "Current models missing from root table"
    for name, marker in listed.items():
        assert (marker == "✅") == (name in available), f"Incorrect interactive eligibility: {name}"


class RootDocumentationTests(unittest.TestCase):
    def test_root_navigation_and_anchors(self):
        documents = list(ROOT.glob("*.md"))
        self.assertTrue(documents)
        for document in documents:
            with self.subTest(document=document.name):
                check_links(document)

    def test_checker_rejects_missing_paths_and_fragments(self):
        with TemporaryDirectory() as directory:
            document = Path(directory) / "README.md"
            document.write_text("# Start\n[Go](#start)\n", encoding="utf-8")
            check_links(document)
            for target in ("missing.md", "#missing", "#todo", "README.md#missing"):
                document.write_text(f"# Start\n[Go]({target})\n", encoding="utf-8")
                with self.subTest(target=target), self.assertRaises(AssertionError):
                    check_links(document)

    def test_duplicate_heading_anchors(self):
        self.assertEqual(anchors("# Start\n## Start\n"), {"start", "start-1"})

    def test_models_follow_current_and_legacy_registry(self):
        check_models(
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "interactive_training/docs/supported_models.md").read_text(encoding="utf-8"),
        )

    def test_model_regressions_fail_without_exceptions(self):
        registry = "| New | `org/New` | Available |\n| Old | `org/Old` | Legacy |\n"
        readme = "| `New` | SFT | ✅ |\n| `Old` | SFT | ❌ |\n"
        check_models(readme, registry)
        for broken, source in (
            (readme.replace("| ❌ |", "| ✅ |"), registry),
            (readme.replace("| ✅ |", "| ❌ |"), registry),
            ("| `Old` | SFT | ❌ |\n", registry),
            (readme + readme, registry),
            (readme, ""),
            ("", registry),
        ):
            with self.subTest(readme=broken), self.assertRaises(AssertionError):
                check_models(broken, source)

    def test_no_retired_namespace_or_placeholder_links_in_entry_points(self):
        for name in ("README.md", "AGENTS.md", "PAID_SMOKE_TESTING.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("interactive_post_training", text)
            self.assertNotRegex(text, r"(?i)\]\([^)]*(?:#todo|placeholder)[^)]*\)")

    def test_root_json_examples_parse(self):
        for document in ROOT.glob("*.md"):
            for block in re.findall(r"```json\s*\n(.*?)```", document.read_text(encoding="utf-8"), re.DOTALL):
                with self.subTest(document=document.name):
                    json.loads(block)


if __name__ == "__main__":
    unittest.main()
