"""Resolve content-pinned V5 inputs into the archive without editing old records."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from v5_storage_archive import ACTIVE_ROOT, ARCHIVE_ROOT
from v5_storage_tree import file_sha, require


B1 = {
    "sla-d4-native-paired-20261119-01",
    "sla-d4-core-e0-interfaces-20261119-01",
}


def plan(spec_path, *, inspect_targets):
    original = json.loads(spec_path.read_bytes())
    updated = deepcopy(original)
    matches = []

    def walk(value, label):
        if isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, (*label, str(i)))
            return
        if not isinstance(value, dict):
            return
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            old = Path(value["path"])
            if old.is_relative_to(ACTIVE_ROOT):
                relative = old.relative_to(ACTIVE_ROOT)
                if len(relative.parts) > 1 and relative.parts[0] in B1:
                    new = ARCHIVE_ROOT / "ssd/historic" / relative
                    prospective_root = ARCHIVE_ROOT / "ssd/historic" / relative.parts[0]
                    digest = value["sha256"]
                    require(old.resolve(strict=True) == old and file_sha(old) == digest,
                            f"Original reference pin changed: {old}")
                    if inspect_targets:
                        require(new.resolve(strict=True).is_relative_to(prospective_root) and file_sha(new) == digest,
                                f"Relocated reference pin changed: {new}")
                        value["path"] = str(new)
                    matches.append(dict(field=".".join(label), original=str(old), archive=str(new),
                                        sha256=digest, relocated_bytes_verified=inspect_targets))
        for key, child in value.items():
            walk(child, (*label, key))

    walk(updated, ())
    require(matches, "No approved B1 references found in this input")
    return dict(original=str(spec_path), original_sha256=file_sha(spec_path), inputs=matches,
                all_archive_inputs_verified=inspect_targets, native_launches=0), updated


def emit_new(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, help="Create a rebound copy only after verifying archive pins")
    args = parser.parse_args()
    proof, rewritten = plan(args.spec, inspect_targets=args.output is not None)
    if args.output is not None:
        emit_new(args.output, rewritten)
        proof["output"] = str(args.output)
        proof["output_sha256"] = file_sha(args.output)
    print(json.dumps(proof, indent=2))
