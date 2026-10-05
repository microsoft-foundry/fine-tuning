"""Synchronize per-demo metadata from the root cookbook catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_catalog(cookbook_root: Path) -> dict[str, Any]:
    catalog_path = cookbook_root / "catalog.yml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not isinstance(catalog.get("demos"), list):
        raise ValueError(f"Invalid catalog structure: {catalog_path}")
    return catalog


def render_demo_metadata(demo: dict[str, Any]) -> str:
    return yaml.safe_dump(demo, sort_keys=False, allow_unicode=False, width=100)


def synchronize(cookbook_root: Path, *, check: bool) -> list[Path]:
    mismatches: list[Path] = []
    for demo in load_catalog(cookbook_root)["demos"]:
        relative_path = demo.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Catalog demo has no path: {demo!r}")
        metadata_path = cookbook_root / relative_path / "demo.yaml"
        expected = render_demo_metadata(demo)
        actual = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else None
        if actual == expected:
            continue
        mismatches.append(metadata_path)
        if not check:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(expected, encoding="utf-8")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Cookbook root containing catalog.yml.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail when demo.yaml files drift.")
    mode.add_argument("--write", action="store_true", help="Rewrite demo.yaml files from catalog.yml.")
    args = parser.parse_args()

    mismatches = synchronize(args.root.resolve(), check=args.check)
    if args.check and mismatches:
        for path in mismatches:
            print(f"metadata drift: {path}")
        return 1
    action = "updated" if args.write else "validated"
    print(f"{action} {len(mismatches) if args.write else 14} demo metadata files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
