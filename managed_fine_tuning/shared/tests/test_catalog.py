from __future__ import annotations

from pathlib import Path
import json

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
import yaml

from shared.catalog_sync import synchronize


def test_catalog_matches_implemented_demo_tree() -> None:
    root = Path(__file__).parents[2]
    catalog = yaml.safe_load((root / "catalog.yml").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "1.0"
    assert catalog["validation"]["status"] == "partial"
    demos = catalog["demos"]
    assert len(demos) == 14
    assert len({demo["id"] for demo in demos}) == 14
    assert all(demo["validation"]["status"] == "partial" for demo in demos)
    assert all((root / demo["path"] / "notebooks" / "demo.ipynb").is_file() for demo in demos)
    assert synchronize(root, check=True) == []


def test_catalog_and_demo_metadata_match_schemas() -> None:
    root = Path(__file__).parents[2]
    catalog = yaml.safe_load((root / "catalog.yml").read_text(encoding="utf-8"))
    schemas = [
        json.loads((root / "shared" / "schemas" / filename).read_text(encoding="utf-8"))
        for filename in ("catalog.schema.json", "demo.schema.json")
    ]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas
    )
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(schemas[0], registry=registry).validate(catalog)
    validator = Draft202012Validator(schemas[1], registry=registry)
    for demo in catalog["demos"]:
        validator.validate(demo)


def test_retail_is_the_only_deployment_comparison_exception() -> None:
    root = Path(__file__).parents[2]
    demos = yaml.safe_load((root / "catalog.yml").read_text(encoding="utf-8"))["demos"]
    for demo in demos:
        sections = demo["notebooks"][0]["sections"]
        if demo["id"] == "agentic-retail-capstone":
            assert "deploy" in sections and "compare" in sections
        else:
            assert not set(sections) & {"deploy", "compare", "evaluate", "baseline"}
