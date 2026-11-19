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
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def runtime_output_path(
    path: str | Path, *, outputs_root: str | Path | None = None,
) -> Path:
    """Resolve an identifier-bearing destination inside its intended outputs root.

    Existing callers may omit the root when there is one unambiguous lexical
    ``outputs`` ancestor, not whichever ``outputs`` survives traversal.
    New callers should pass their demo's explicit, ignored ``outputs`` directory.
    A redirected root itself, an escaping descendant symlink, and traversal out
    of that boundary are rejected before directories or temporary files exist.
    """
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    if outputs_root is None:
        positions = [index for index, part in enumerate(selected.parts) if part == "outputs"]
        if len(positions) != 1:
            raise ValueError("Runtime state requires an unambiguous ignored outputs/ root")
        root = Path(*selected.parts[: positions[0] + 1])
    else:
        root = Path(outputs_root)
        if not root.is_absolute():
            root = Path.cwd() / root
    if root.name != "outputs" or ".." in root.parts:
        raise ValueError("The intended ignored outputs/ root must be explicit and unambiguous")
    try:
        intended_root = root.parent.resolve() / root.name
        resolved = selected.resolve()
        if root.resolve() != intended_root:
            raise ValueError("The outputs/ root must not be redirected by a symlink")
        if resolved == intended_root or not resolved.is_relative_to(intended_root):
            raise ValueError("Runtime state must remain inside the intended ignored outputs/ root")
    except (OSError, RuntimeError) as error:
        raise ValueError("Cannot resolve the intended ignored outputs/ boundary") from error
    return resolved


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

    def write(
        self, path: str | Path, *, outputs_root: str | Path | None = None,
    ) -> Path:
        selected = runtime_output_path(path, outputs_root=outputs_root)
        return _write_json_atomic(selected, self.to_dict())
