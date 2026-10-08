from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_central_dependency_manifest_matches_lock() -> None:
    root = Path(__file__).parents[2]
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = (
        manifest["project"]["dependencies"]
        + manifest["project"]["optional-dependencies"]["test"]
    )
    locked = {}
    for line in (root / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifications = list(requirement.specifier)
        assert len(specifications) == 1 and specifications[0].operator == "=="
        name = requirement.name.casefold().replace("_", "-")
        assert name not in locked
        locked[name] = specifications[0].version
    names = set()
    for line in requirements:
        requirement = Requirement(line)
        name = requirement.name.casefold().replace("_", "-")
        names.add(name)
        assert name in locked
        assert requirement.specifier.contains(locked[name])
    assert names == set(locked)


def test_hidden_environment_template_is_included_in_package_data() -> None:
    root = Path(__file__).parents[2]
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "templates/.env.template" in manifest["tool"]["setuptools"]["package-data"]["shared"]
