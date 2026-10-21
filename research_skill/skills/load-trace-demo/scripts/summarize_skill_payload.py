"""Tiny helper loaded by the skill demo."""

from __future__ import annotations

from typing import Any


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    paths = [str(x.get("path", "")) for x in payload.get("loaded_files", []) if isinstance(x, dict)]
    return {"count": len(paths), "paths": paths, "has_all_parts": all(any(part in path for path in paths) for part in ("/references/", "/scripts/", "/assets/"))}
