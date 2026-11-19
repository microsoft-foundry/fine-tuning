"""Offline publisher-metadata regressions; these do not resolve or install packages."""

import json
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
import pytest


ROOT = Path(__file__).parents[2]
PYTHON_MINORS = ("3.11", "3.12", "3.13", "3.14")


def _snapshot() -> dict:
    path = Path(__file__).parent / "data" / "dependency_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _locked_versions() -> dict[str, str]:
    versions = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        specifications = list(requirement.specifier)
        assert requirement.marker is None
        assert len(specifications) == 1 and specifications[0].operator == "=="
        name = canonicalize_name(requirement.name)
        assert name not in versions
        versions[name] = specifications[0].version
    return versions


def test_publisher_snapshot_covers_every_exact_direct_pin() -> None:
    snapshot = _snapshot()
    versions = _locked_versions()
    metadata = snapshot["direct_pins"]
    assert tuple(snapshot["python_minors"]) == PYTHON_MINORS
    assert len(versions) == 35
    assert set(metadata) == set(versions)
    for name, version in versions.items():
        assert metadata[name]["version"] == version
        assert metadata[name]["metadata_url"] == (
            f"https://pypi.org/pypi/{name}/{version}/json"
        )
    unknown = {name for name, item in metadata.items() if item["requires_python"] is None}
    assert unknown == {"jupyter"}, "Missing publisher metadata must remain explicit."


@pytest.mark.parametrize("minor", PYTHON_MINORS)
def test_recorded_direct_python_floors_cover_each_declared_minor(minor: str) -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    supported = SpecifierSet(manifest["project"]["requires-python"])
    assert supported.contains(f"{minor}.0")
    assert not supported.contains("3.10.99")
    assert not supported.contains("3.15.0")
    for name, item in _snapshot()["direct_pins"].items():
        requires_python = item["requires_python"]
        if requires_python is None:
            # The Jupyter metapackage is unknown here, not verified compatible.
            assert name == "jupyter"
            continue
        assert SpecifierSet(requires_python).contains(f"{minor}.0"), (
            name, item["version"], minor, requires_python
        )


@pytest.mark.parametrize("minor", PYTHON_MINORS)
def test_numpy_satisfies_the_recorded_opencv_dependency(minor: str) -> None:
    metadata = _snapshot()["direct_pins"]
    requirements = [
        Requirement(line) for line in metadata["opencv-python"]["requires_dist"]
    ]
    selected = [
        requirement for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(
            {"python_version": minor, "python_full_version": f"{minor}.0"}
        )
    ]
    assert len(selected) == 1
    assert canonicalize_name(selected[0].name) == "numpy"
    assert selected[0].specifier.contains(_locked_versions()["numpy"])