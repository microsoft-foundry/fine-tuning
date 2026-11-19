"""Resolve this demo and its shared track helpers without installing packages."""

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
for root in (DEMO_ROOT.parent / "src", DEMO_ROOT):
    sys.path.insert(0, str(root))