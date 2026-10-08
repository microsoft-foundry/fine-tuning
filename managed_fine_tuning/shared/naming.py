"""Meaningful local and remote naming conventions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, max_length: int = 64) -> str:
    """Convert a display value into a non-empty lowercase kebab-case slug."""
    slug = _NON_SLUG.sub("-", value.strip().casefold()).strip("-")
    if not slug:
        raise ValueError("A name must contain at least one letter or number")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    return slug[:max_length].rstrip("-")


@dataclass(frozen=True, slots=True)
class NameFactory:
    """Build names that identify the demo, purpose, and run."""

    demo_slug: str
    run_label: str

    @classmethod
    def for_demo(
        cls,
        demo_slug: str,
        *,
        run_label: str | None = None,
        now: datetime | None = None,
    ) -> "NameFactory":
        instant = now or datetime.now(UTC)
        label = run_label or instant.strftime("%Y%m%d-%H%M%S")
        return cls(slugify(demo_slug, max_length=40), slugify(label, max_length=24))

    def remote(self, purpose: str, *, max_length: int = 64) -> str:
        return slugify(
            f"{self.demo_slug}-{purpose}-{self.run_label}",
            max_length=max_length,
        )

    def local(self, purpose: str, extension: str) -> str:
        suffix = extension.strip().lstrip(".")
        if not suffix or not re.fullmatch(r"[a-zA-Z0-9]+", suffix):
            raise ValueError("extension must contain only letters and numbers")
        return f"{self.remote(purpose, max_length=100)}.{suffix.casefold()}"
