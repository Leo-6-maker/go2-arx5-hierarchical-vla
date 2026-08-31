#!/usr/bin/env python3
"""Dependency-free validation of the committed logical artifact manifests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from go2arx5_vla.artifacts.manifest import required_manifest_keys


REQUIRED = ("logical_id", "role", "local_path_template", "provenance", "status", "external_private", "redistributable")


def main() -> int:
    root = Path(__file__).parents[1] / "artifact_manifests"
    failures = []
    for path in sorted(root.glob("*.yaml")):
        missing = required_manifest_keys(path.read_text(encoding="utf-8"), REQUIRED)
        if missing:
            failures.append(f"{path}: missing {missing}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"validated {len(list(root.glob('*.yaml')))} artifact manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
