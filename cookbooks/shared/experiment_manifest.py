"""Serializable experiment and runtime manifests."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


@dataclass(slots=True)
class ExperimentManifest:
    demo_id: str
    hypothesis: str
    base_model: str
    dataset_hashes: dict[str, str]
    code_version: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=_utc_now)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        return _write_json_atomic(Path(path), self.to_dict())


@dataclass(slots=True)
class RuntimeManifest:
    """Identifier-bearing state that may be written only below ignored outputs."""

    demo_id: str
    operation_ids: dict[str, str] = field(default_factory=dict)
    resource_context: dict[str, str] = field(default_factory=dict)
    sdk_versions: dict[str, str] = field(default_factory=dict)
    started_at_utc: str = field(default_factory=_utc_now)
    completed_at_utc: str | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        selected = Path(path)
        if "outputs" not in {part.casefold() for part in selected.parts}:
            raise ValueError(
                "Runtime manifests can contain service identifiers and must be "
                "written below an ignored outputs/ directory"
            )
        return _write_json_atomic(selected, self.to_dict())
