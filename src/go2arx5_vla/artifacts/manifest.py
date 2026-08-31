from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_manifest_keys(text: str, keys: tuple[str, ...]) -> list[str]:
    """Return missing top-level YAML keys without requiring PyYAML."""
    missing = []
    for key in keys:
        if not any(line.startswith(f"{key}:") for line in text.splitlines()):
            missing.append(key)
    return missing
