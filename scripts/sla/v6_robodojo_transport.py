"""Stream exactly one sealed public source release into a fresh receiving directory."""

import argparse
from pathlib import Path
import sys
import tarfile

from v6_robodojo_intake import digest, read
from v6_ville_pilot import need


def send(root, ready_sha):
    ready_file = root / "READY.json"
    need(digest(ready_file) == ready_sha, "Release receipt changed")
    ready = read(ready_file)
    manifest = root / "MANIFEST.private.json"
    need(digest(manifest) == ready["manifest_sha256"] and ready["core_readback"] is True
         and ready["hopper_readback"] is False, "Expected an exact Core public release")
    source = read(manifest)
    episodes = source["episodes"]
    need(len(episodes) == ready["newly_admitted_train"] and sum(x["windows"] for x in episodes) == ready["windows"],
         "Release and manifest populations disagree")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", dereference=True) as archive:
        for src, name in ((ready_file, "CORE_READY.json"), (manifest, manifest.name),
                          (root / "STATISTICS.json", "STATISTICS.json")):
            archive.add(src, arcname=name, recursive=False)
        for episode in episodes:
            sha = episode["raw_sha256"]
            need(len(bytes.fromhex(sha)) == 32 and episode["split"] == "TRAIN",
                 "Expected a sealed train-only raw identity")
            numeric = Path(episode["arrays"]["file"])
            need(not numeric.is_absolute() and numeric.parts == ("numeric", numeric.name),
                 "Numeric member must stay in the release")
            source_path = Path(episode["raw_path"])
            need(source_path.resolve(strict=True).is_relative_to(root.parent.resolve(strict=True)),
                 "Raw public archive must belong to the staged corpus")
            archive.add(root / numeric, arcname=numeric.as_posix(), recursive=False)
            archive.add(source_path, arcname=f"raw/{sha}.hdf5", recursive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("ready_sha")
    args = parser.parse_args()
    send(args.root, args.ready_sha)
