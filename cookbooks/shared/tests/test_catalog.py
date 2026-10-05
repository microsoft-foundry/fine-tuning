from __future__ import annotations

from pathlib import Path

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
