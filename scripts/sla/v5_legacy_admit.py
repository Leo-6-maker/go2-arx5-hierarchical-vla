"""Recheck unused TRAIN-069 members with the original actual G7 label reader."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_consumed_data import PHASES, TRAIN_SEAL, consume


def check(archive, previous, raw, out):
    from scripts.pi0292.durable import write

    seal = read(pin(archive / "SEALED.private.json", TRAIN_SEAL))
    catalog = read(pin(archive / "MANIFEST.private.json", seal["manifest_sha256"]))
    certified = read(pin(archive / "ACTUAL.private.json", seal["admission_sha256"]))
    admitted = {row["id"]: row for row in certified["cases"] if row["admitted"]}
    baseline = read(previous / "MANIFEST.json")
    base = {row["episode"]: row for row in baseline["episodes"]}
    heldout = set(baseline["excluded_dev_scenes"])
    original = catalog["episodes"]
    require(catalog["split"] == "TRAIN" and seal["count"] == len(original) == len(admitted) == 69
            and len(base) == 52 and len(heldout) == 2
            and baseline["signed_original_manifest_sha256"] == seal["manifest_sha256"],
            "Wrong signed legacy campaign or comparison exclusion")
    require(set(base) <= set(admitted) and len({e["id"] for e in original}) == len(original),
            "Missing or duplicate signed episode identity")
    out.mkdir(mode=0o700)
    current = []
    sites = sorted(raw.glob("sla-core-*"))
    receipts = []
    for episode in original:
        identity = episode["id"]
        actual = admitted[identity]
        require(episode["split"] == "TRAIN" and episode["individual_pass"]
                and actual["scene_sha256"] == episode["scene_sha256"]
                and episode["native_result_sha256"] == actual["native_result_sha256"],
                "A signed legacy candidate has mismatched provenance")
        if identity in base:
            row = base[identity]
            require(row["scene"] not in heldout and row["result"] == actual["native_result_sha256"],
                    "A previously accepted training scene conflicts with a held-out scene")
            pin(previous / row["labels"], row["sha256"])
            current.append(dict(row, origin="LEGACY_V4"))
            continue
        row = dict(id=identity, task=episode["task"], scene_sha256=episode["scene_sha256"],
                   signed_result=actual["native_result_sha256"])
        if episode["scene_sha256"] in heldout:
            receipts.append(dict(row, status="HELD_OUT_SCENE"))
            continue
        cases = [d / "cases" / identity for d in sites if (d / "cases" / identity / "RESULT.private.json").is_file()]
        cases = [candidate for candidate in cases
                 if sha(candidate / "RESULT.private.json") == actual["native_result_sha256"]]
        if len(cases) != 1:
            receipts.append(dict(row, status="ACTUAL_NATIVE_SOURCE_NOT_UNIQUE", sources=len(cases)))
            continue
        try:
            arrays, proof = consume(cases[0], episode, actual, archive)
            require(set(arrays["phase"].tolist()) <= PHASES.keys(), "Unsupported task phase")
        except (KeyError, ValueError, OSError) as exc:
            receipts.append(dict(row, status="ORIGINAL_CONSUMPTION_CHECK_FAILED", reason=str(exc)))
            continue
        file = out / (identity + ".npz")
        with file.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        complete = int(arrays["valid"].all(axis=1).sum())
        current.append(dict(proof, labels=file.name, sha256=sha(file), origin="NEW_LEGACY_READMISSION",
                            complete_H40=complete, actual_label_atoms=int(arrays["valid"].sum())))
        receipts.append(dict(row, status="ACTUAL_CONSUMED_G7_PASS", windows=len(arrays["valid"]),
                             complete_H40=complete, label_sha256=sha(file)))
    candidates = [row for row in current if row["origin"] == "NEW_LEGACY_READMISSION"]
    tally = Counter("HOLD" if row["task"].startswith("HOLD-") else "PLACE" for row in current)
    summary = dict(schema="CORE_EMBODIMENT_V5_LEGACY_READBACK", source_seal_sha256=TRAIN_SEAL,
                   previous_sha256=sha(previous / "MANIFEST.json"), original_admitted=52,
                   recovered=len(candidates), admitted=len(current), tasks=dict(tally),
                   windows=sum(row["windows"] for row in current), source_considered=len(original),
                   reason_counts=dict(Counter(row["status"] for row in receipts)), receipts=receipts,
                   new_native_requests=0, new_physics_ticks=0, actual_training_visits=0)
    write(out / "ACTUAL.json", summary)
    write(out / "CANDIDATE_MANIFEST.json", dict(schema="V3_MAIN_CONSUMED_G7",
          training_seal=TRAIN_SEAL, signed_original_manifest_sha256=seal["manifest_sha256"],
          excluded_dev_scenes=sorted(heldout), episodes=current, legacy_revision_only=True))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("archive", "previous", "raw", "out"):
        parser.add_argument(argument, type=Path)
    result = check(**vars(parser.parse_args()))
    print({k: v for k, v in result.items() if k != "receipts"})
