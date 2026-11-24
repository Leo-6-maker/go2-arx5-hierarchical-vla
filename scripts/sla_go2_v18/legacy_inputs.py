"""Resolve historical source evidence without pooling codec-specific windows."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"changed archived input {path.name}")
    return path


def read(path):
    return json.loads(path.read_bytes())


def inventory(root, wm02, readback, report):
    primary = read(report / "TRAIN_QUALIFICATION.json")["episodes"]
    verified = {
        v["episode_sha256"]: name
        for name, v in primary.items()
        if v["status"].startswith("verified_")
    }
    conf = read(readback / "CONFIG.json")
    ref = read(checked(Path(conf["prototype"]), conf["prototype_sha"]))
    seal = read(checked(Path(ref["path"]), ref["sha256"]))
    ids = seal["episode_sha256"]
    assert (
        ids == ref["membership"]
        and len(set(ids)) == len(ids) == 69
        and len(verified) == 57
    )
    assert set(verified) <= set(ids)
    cohorts = {
        "TRAIN-069": {
            "membership": [
                {"episode_sha256": h, "verified_primary_episode": verified.get(h)}
                for h in ids
            ],
            "windows": ref["windows"],
            "exact_primary_duplicates": len(verified),
            "seal_sha256": ref["sha256"],
        }
    }
    for key, mf in (
        (
            "TRAIN100",
            wm02 / "native-camera-train100-20261112-01/data/DATASET_MANIFEST.json",
        ),
        (
            "C20",
            wm02
            / "mfree-c20-learning-pipeline-v1-20261116-02/dataset/MANIFEST.private.json",
        ),
    ):
        d = read(mf)
        records = []
        for r in d["episodes"]:
            if key == "C20":
                label = r["labels"]
                f = checked(mf.parent / label["file"], label["sha256"])
                result = r["native_result"]
                capture = r["original_live_capture"]
                n = r["windows"]
                scene_key = r["scene_lineage"]
            else:
                f = checked(mf.parent / r["file"], r["sha256"])
                result = r["proof"]["native"]
                capture = r["proof"]["capture_index"]
                n = r["supervised_policy_decisions"]
                scene_key = r["parent"]
            checked(Path(result["path"]), result["sha256"])
            checked(Path(capture["path"]), capture["sha256"])
            records.append(
                {
                    "episode": r["id"],
                    "split": r["split"],
                    "lineage": scene_key,
                    "label_bundle_sha256": digest(f),
                    "capture_sha256": capture["sha256"],
                    "native_result_sha256": result["sha256"],
                    "windows": n,
                }
            )
        cohorts[key] = {
            "manifest_sha256": digest(mf),
            "provenance_rows": records,
            "native_result_duplicates_with_primary": sum(
                r["native_result_sha256"]
                in {v["result_sha256"] for v in primary.values()}
                for r in records
            ),
            "qualification": "Native result, capture receipt and old label bundle verified. Memory history, same-episode clock reconstruction and official label conversion are outstanding.",
        }
    wm = read(root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json")
    parent = root / "vendor/XPolicyLab/policy/SLA_Memory/checkpoints"
    weights = {}
    for r in wm["verified_checkpoint_files"]:
        checked(parent / r["path"], r["local_content_sha256"])
        weights[r["path"]] = r["local_content_sha256"]
    return {
        "qualified_primary": 57,
        "official_revision": wm["model_revision"],
        "official_source": wm["source_commit"],
        "official_checkpoint_files": weights,
        "official_bytes": wm["verified_bytes"],
        "data": cohorts,
        "newly_admitted_training_episodes": 0,
        "new_inference": 0,
        "physics_ticks": 0,
        "updates": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "wm02", "readback", "report", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    result = inventory(a.root, a.wm02, a.readback, a.report)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
