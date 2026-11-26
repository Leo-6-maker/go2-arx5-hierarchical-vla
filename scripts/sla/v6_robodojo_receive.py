"""Independently bind and read Core's unchanged source archive on Hopper."""

import argparse
from pathlib import Path

from v6_robodojo_intake import digest, read
from v6_robodojo_pilot import REV
from v6_robodojo_seal import R48_STATS
from v6_ville_pilot import need, write


def receive(root, core_ready_sha):
    import numpy as np
    from go2arx5_vla.sla_adapter.scale20k_data import RobodojoDataset
    from openpi_client.image_tools import resize_with_pad

    class ExpansionDataset(RobodojoDataset):
        RECIPE = "SLA_V6_EXISTING_ROBODOJO_CODEC_EXTENSION"

    core_ready = root / "CORE_READY.json"
    need(digest(core_ready) == core_ready_sha, "Core release bytes changed")
    ready = read(core_ready)
    manifest_path = root / "MANIFEST.private.json"
    manifest_sha = ready["manifest_sha256"]
    need(digest(manifest_path) == manifest_sha and ready["core_readback"] is True
         and ready["hopper_readback"] is False and ready["training_visits"] == 0,
         "Expected an actual Core release without optimizer visits")
    manifest = read(manifest_path)
    need(manifest["HF_revision"] == REV and manifest["test_labels_read"] is False
         and manifest["statistics"]["sha256"] == R48_STATS
         and digest(root / "STATISTICS.json") == R48_STATS, "RoboDojo native source ABI changed")
    raw = {}
    for ep in manifest["episodes"]:
        src = ep["raw_sha256"]
        relative = f"raw/{src}.hdf5"
        need((root / relative).resolve(strict=True).is_relative_to(root.resolve(strict=True)),
             "Received member left the release root")
        raw[ep["id"]] = dict(file=relative, sha256=src)

    binding = root / "RAW_BINDING.private.json"
    write(binding, dict(manifest_sha256=manifest_sha, raw=raw, split="TRAIN"))
    reader = ExpansionDataset(root, dict(path=str(manifest_path), sha256=manifest_sha),
        raw_binding=dict(path=str(binding), sha256=digest(binding)), split="TRAIN",
        resize_with_pad=resize_with_pad)
    checked = []
    offset = 0
    for ep in reader.episodes.values():
        for row in (offset, offset + ep["windows"] - 1):
            item = reader[row]
            need(item["actions"].shape == (50, 32) and np.isfinite(item["actions"]).all()
                 and np.isfinite(item["state"]).all() and not np.any(item["actions"][:, 14:])
                 and all(item["image_mask"].values()), "Hopper source learner input changed")
            checked.append(reader.identity(row))
        offset += ep["windows"]
    need(offset == ready["windows"] and len(reader.verified_raw) == ready["newly_admitted_train"]
         and len(checked) == ready["readback_samples"], "Hopper readback population differs")
    checks = root / "HOPPER_SOURCE_READBACK.json"
    write(checks, dict(checked=checked, raw_sha256_verified=len(reader.verified_raw), training_visits=0))
    write(root / "READY.json", {**ready, "hopper_readback": True, "core_ready_sha256": core_ready_sha,
          "raw_binding_sha256": digest(binding), "hopper_readback_sha256": digest(checks)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("core_ready_sha")
    args = parser.parse_args()
    receive(args.root, args.core_ready_sha)
