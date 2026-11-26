"""Apply the verified R4 coordinate representation to the 16 added Go2 episodes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v6_reanchored_teacher_labels import reexpress
from v6_reference_anchors import case_evidence


ORIGINAL = "1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21"
LATEST = "beedab75cafa45de30013ab815e84f249fdbc26b3d09dee950271f928e0a9780"


def build(native_root, factory, original, output):
    from scripts.pi0292.durable import write

    first = factory / "SNAPSHOTS/TRAIN-LEGACY-064"
    latest = factory / "SNAPSHOTS/TRAIN-LEGACY-068/MANIFEST.json"
    read(pin(latest, LATEST))
    earlier = read(pin(original / "MANIFEST.json", ORIGINAL))
    base = read(first / "MANIFEST.json")
    base_ready = read(first / "READY.json")
    require(base_ready["manifest_sha256"] == sha(first / "MANIFEST.json")
            and base_ready["episodes"] == 64 and base_ready["receiver_ready"], "Unknown base data version")
    blocked = set(base["excluded_dev_scenes"])
    members = [(r, first / r["labels"], r["sha256"]) for r in base["episodes"]
               if r["origin"] == "NEW_LEGACY_READMISSION"]
    for name in ("LEGACY_RECOVERY", "LEGACY-RED-PLACE-REREAD-01"):
        path = factory / name
        proof = read(path / "ACTUAL.json")
        require(not proof["new_native_requests"] and not proof["new_model_forwards"]
                and all(r["split"] == "TRAIN" for r in proof["admissions"]), "Unexpected Teacher recovery")
        members.extend((r, path / r["label"], r["label_sha256"]) for r in proof["admissions"])
    identities = {r["episode"]["id"] if isinstance(r["episode"], dict) else r["episode"] for r, _, _ in members}
    require(len(identities) == len(members) == 16 and len(earlier["episodes"]) == 52
            and not identities.intersection(r["episode"] for r in earlier["episodes"]), "The delta population changed")

    output.mkdir(mode=0o700)
    ledger, phases = [], Counter()
    for member, label, label_sha in members:
        original_id = member["episode"]
        identity = original_id["id"] if isinstance(original_id, dict) else original_id
        row = dict(member, episode=identity)
        require(row["scene"] not in blocked and row["task"].startswith(("HOLD-", "PLACE-")),
                "Held-out scene or unsupported original task")
        spec, queries, query_sha = case_evidence(native_root, row)
        with np.load(pin(label, label_sha), allow_pickle=False) as source:
            recorded = {key: source[key] for key in ("action", "valid", "phase", "native_step")}
        action, error, _ = reexpress(spec, queries, recorded)
        require(float(error.max()) < .001, "Reconstruction changed the executed physical geometry")
        target = output / f"{identity}.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **dict(recorded, action=action))
        mask = recorded["valid"]
        one = Counter(recorded["phase"][mask.all(axis=1)].tolist())
        phases.update(one)
        ledger.append(dict(episode=identity, task=row["task"], scene_sha256=row["scene"], label=target.name,
            sha256=sha(target), original_label_sha256=label_sha, query_sha256=query_sha,
            windows=len(queries), full_H40=int(mask.all(axis=1).sum()), supported_atoms=int(mask.sum()),
            max_reconstructed_mm_deg=error.tolist(), full_H40_by_phase=dict(one)))
    sums = {key: sum(row[key] for row in ledger) for key in ("windows", "full_H40", "supported_atoms")}
    require(sums["windows"] + earlier["windows"] == 6034 and sums["full_H40"] + earlier["full_H40"] == 5830,
            "Actual reconstructed windows no longer match the signed 68-episode dataset")
    manifest = output / "MANIFEST.json"
    write(manifest, dict(schema="SLA_R4_REEXPRESSED_LEGACY_DELTA", source_dataset_sha256=LATEST,
                         inherited_corrected_manifest_sha256=ORIGINAL, episodes=ledger,
                         gripper_and_mask_bitwise_preserved=True, **sums))
    write(output / "ACTUAL.json", dict(manifest_sha256=sha(manifest), episodes=16, corrected_total=68,
         full_H40_by_phase=dict(phases), max_reconstructed_mm_deg=np.max(
             [r["max_reconstructed_mm_deg"] for r in ledger], axis=0).tolist(),
         optimizer_updates=0, model_forwards=0, native_integrations=0, **sums))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("native_root", "factory", "original", "output"):
        parser.add_argument(field, type=Path)
    build(**vars(parser.parse_args()))
