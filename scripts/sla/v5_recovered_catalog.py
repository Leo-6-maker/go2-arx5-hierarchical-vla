"""Read the closed recovery tranche and seal the next disjoint V5 legacy version."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha


def composite_members(snapshot, base_sha):
    path = snapshot / "MANIFEST.json"
    status = read(snapshot / "READY.json")
    digest = status["manifest_sha256"]
    total = status["total"]
    added = []
    while total > 64:
        manifest = read(pin(path, digest))
        group = manifest["actual_added"]
        require(manifest["schema"] == f"V5_LEGACY_COMPOSITE_{total}"
                and manifest["original_manifest_sha256"] == base_sha and 0 < len(group) <= total - 64,
                "Broken cumulative legacy release")
        added.extend(group)
        total -= len(group)
        digest = manifest.get("previous_composite_sha256")
        path = snapshot.parent / f"TRAIN-LEGACY-{total:03d}/MANIFEST.json"
        require((digest is None) == (total == 64), "Missing or extraneous legacy predecessor")
    require(len({r["episode"]["id"] for r in added}) == len(added)
            and len({r["result_sha256"] for r in added}) == len(added), "Duplicate prior recovery members")
    return status, added


def prepare(first, recovery, archive, actions, states, out, *, composite=None):
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32
    from go2arx5_vla.sla_adapter.multitask_data_v6 import TeacherDataset
    from scripts.pi0292.durable import write

    old = read(first / "MANIFEST.json")
    ready = read(first / "READY.json")
    addition = read(recovery / "ACTUAL.json")
    require(ready["manifest_sha256"] == sha(first / "MANIFEST.json") and ready["receiver_ready"]
            and len(old["episodes"]) == ready["episodes"] == 64 and addition["recovered"] > 0
            and (addition.get("terminal_and_processes_absent") or addition["expired_and_processes_absent"])
            and not addition["new_native_requests"], "Expected the verified base and a reconciled terminal tranche")
    known = {entry["episode"] for entry in old["episodes"]}
    scenes = set(old["excluded_dev_scenes"])
    old_results = {entry["result"] for entry in old["episodes"]}
    count = Counter(row["task"].split("-")[0] for row in old["episodes"])
    prior_episodes, prior_windows, prior_full = ready["episodes"], ready["windows"], ready["complete_H40"]
    previous = {}
    if composite is not None:
        status, prior_added = composite_members(composite, ready["manifest_sha256"])
        require(status["total"] > 64, "Expected a predecessor with real recovered members")
        prior_episodes, prior_windows, prior_full = status["total"], status["total_windows"], status["full_H40"]
        known.update(row["episode"]["id"] for row in prior_added)
        old_results.update(row["result_sha256"] for row in prior_added)
        count = Counter(status["task_counts"])
        previous["previous_composite_sha256"] = status["manifest_sha256"]
    rows = addition["admissions"]
    require(len({row["episode"]["id"] for row in rows}) == len({row["result"] for row in rows})
            == len(rows) == addition["recovered"],
            "Recovered identities are ambiguous")
    for row in rows:
        ep = row["episode"]
        source_file = pin(archive / ep["id"] / "EPISODE.private.json", row["readmission_origin"])
        require(read(source_file) == ep and ep["id"] not in known
                and row["scene"] not in scenes and row["result"] not in old_results and ep["split"] == "TRAIN",
                "Recovered row conflicts with a signed original or held-out scene")
    original_stats = dict(actions=read(actions)["actions"], state=read(states)["state"])

    class Recovered(TeacherDataset):
        def __init__(self):
            self.root = archive
            self.manifest = dict(episodes=[row["episode"] for row in rows], windows=addition["recovered_windows"])
            self.statistics = {key: Quantiles(info["q01"], info["q99"]) for key, info in original_stats.items()}
            self._load_episodes(resize_with_pad, 96 * 1024**2)
            self.actual = {}
            for row in rows:
                eid = row["episode"]["id"]
                with np.load(pin(recovery / row["label"], row["label_sha256"]), allow_pickle=False) as f:
                    arrays = {k: f[k] for k in ("action", "valid", "phase", "native_step")}
                require(np.array_equal(arrays["phase"], self.labels[eid]["phase"])
                        and np.array_equal(arrays["native_step"], self.labels[eid]["native_step"])
                        and arrays["action"].shape == (row["windows"], 40, 7)
                        and arrays["valid"].shape == (row["windows"], 40)
                        and arrays["valid"].dtype == bool and arrays["valid"][:, :10].all()
                        and np.isfinite(arrays["action"]).all(), "Recovered action or mask lost indexed physical support")
                self.actual[eid] = arrays["action"], arrays["valid"]

        def __getitem__(self, index):
            result = super().__getitem__(index)
            ep, local = self.rows[index]
            actions, mask = self.actual[ep["id"]]
            value = pad32(self.statistics["actions"].normalize(actions[local])).astype(np.float32)
            result["actions"] = np.where(mask[local, :, None], value, 0)
            return result

    reader = Recovered()
    added = []
    offset = 0
    for row in rows:
        ep = row["episode"]
        for index in (offset, offset + ep["windows"] - 1):
            item = reader[index]
            require(item["actions"].shape == (40, 32) and item["state"].shape == (32,)
                    and np.isfinite(item["actions"]).all()
                    and item["image_mask"] == dict(base_0_rgb=True, left_wrist_0_rgb=True, right_wrist_0_rgb=False),
                    "Recovery training reader did not reach real boundary observations")
        offset += ep["windows"]
        added.append(dict(episode=ep, result_sha256=row["result"], actual_label=pin(
            recovery / row["label"], row["label_sha256"]).name, actual_label_sha256=row["label_sha256"],
            complete_H40=row["complete_H40"]))
    require(offset == len(reader) == addition["recovered_windows"], "Recovery window population changed")
    out.mkdir(mode=0o700)
    count.update(addition["recovered_tasks"])
    manifest = out / "MANIFEST.json"
    total = prior_episodes + len(rows)
    write(manifest, dict(schema=f"V5_LEGACY_COMPOSITE_{total}", original_manifest_sha256=ready["manifest_sha256"],
          recovered_archive_receipt_sha256=sha(recovery / "ACTUAL.json"), actual_added=added,
          allocation="CLOSED_LEGACY_PRODUCER", task_counts=dict(count), new_physics_ticks=0, **previous))
    proof = dict(manifest_sha256=sha(manifest), total=total,
                 total_windows=prior_windows + offset, full_H40=prior_full + addition["complete_H40"],
                 core_new_boundary_reads=2 * len(rows), task_counts=dict(count),
                 hopper_independent_readback=False, new_native_requests=0, optimizer_visits=0)
    write(out / "READY.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "recovery", "archive", "actions", "states", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--composite", type=Path)
    print(prepare(**vars(parser.parse_args())))
