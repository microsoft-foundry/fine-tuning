import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "Demos"

TEXT_SUFFIXES = {".py", ".md", ".txt", ".template", ".sample"}
IGNORED_PARTS = {
    ".venv",
    "__pycache__",
    ".jupyter",
    ".kagglehub",
    "site-packages",
    "analysis_charts",
    "latency_outputs",
    "outputs",
    "run",
}

RULES = {
    "direct OpenAI import": re.compile(r"(?m)^\s*(?:from\s+openai\b|import\s+openai\b)"),
    "direct OpenAI client constructor": re.compile(
        r"(?<!Azure )\bOpenAI\s*\(|\bAzureOpenAI\s*\("
    ),
    "API-key authentication": re.compile(
        r"\b(?:AZURE_OPENAI_API_KEY|OPENAI_API_KEY)\b|(?<![\w])api_key\s*="
    ),
    "raw OpenAI API route": re.compile(r"/openai/(?:v\d+|deployments|fine_tuning|files|responses|evals)"),
    "account-level OpenAI endpoint": re.compile(
        r"https://[^\"'\s]+(?:\.openai\.azure\.com|\.cognitiveservices\.azure\.com)"
    ),
}


def ignored(path: Path) -> bool:
    if any(part in IGNORED_PARTS for part in path.parts):
        return True
    name = path.name.lower()
    return ".executed." in name or ".blocked." in name


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") in {"code", "markdown"}
    )


def source_text(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return notebook_source(path)
    return path.read_text(encoding="utf-8", errors="replace")


def check_requirements(path: Path, text: str):
    issues = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"(?i)^openai(?:\s*[<>=!~].*)?$", stripped):
            issues.append((line_number, "explicit OpenAI SDK dependency", stripped))
        if stripped.lower().startswith("azure-ai-projects"):
            version = re.search(r"(\d+)(?:\.\d+)?", stripped)
            if not version or int(version.group(1)) < 2:
                issues.append((line_number, "pre-2.x Foundry SDK dependency", stripped))
    return issues


def main() -> int:
    findings = []
    checked = 0
    for path in DEMOS.rglob("*"):
        if not path.is_file() or ignored(path):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES | {".ipynb"}:
            continue
        if path.name.startswith("requirements") and path.suffix == ".txt":
            pass
        elif path.suffix.lower() == ".txt" and "requirements" not in path.name.lower():
            continue
        try:
            text = source_text(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            findings.append((path, 0, "unreadable source", str(error)))
            continue
        checked += 1
        lines = text.splitlines()
        for rule, pattern in RULES.items():
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                excerpt = lines[line_number - 1].strip()[:240] if lines else ""
                findings.append((path, line_number, rule, excerpt))
        if path.name.startswith("requirements") and path.suffix == ".txt":
            findings.extend((path, *issue) for issue in check_requirements(path, text))

    if findings:
        print(f"Found {len(findings)} Foundry SDK upgrade violation(s) in {checked} source files:")
        for path, line, rule, excerpt in findings:
            relative = path.relative_to(ROOT)
            print(f"{relative}:{line}: {rule}: {excerpt}")
        return 1
    print(f"Foundry SDK upgrade validation passed for {checked} canonical demo source files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
