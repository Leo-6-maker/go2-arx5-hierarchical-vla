"""Assemble the A74 addition from already admitted, physically consumed TRAIN records."""

import argparse
import errno
import json
import os
from pathlib import Path
import shutil

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_bank import PINS


PLAN = "161707b864f7d7eb3e27f48889276fbc915bc599"
LEGACY_AUDIT = "f2dfa1034421b0c1d13f8685f6135b5be7e13ea1cf292d14b4fb1ea84bef8ab7"
LEGACY_CATALOG = "451b824ced10bc802ebd4a0528cb4e8dcf002ec2c05f65ae8559126da6356381"
RECENT_BATCHES = {
    "GO2-R4-DELTA-0004-01": "e87e09c17d52f302232d8eb03da2cd9bf393695bc27a0a1164347190d14babc8",
    "DIVERSE-PLACE-TRAIN-02-LABELS-R4": "7e5c429d5a28c57ced42b175942adabe8e6501ad53ed1ac1553b088c9ff401e9",
}


def receipt(path):
    return dict(file=path.name, sha256=sha(path))


def materialize(source, dest):
    dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with source.open("rb") as original, dest.open("xb") as received:
            shutil.copyfileobj(original, received, length=4 * 1024**2)


def signed_case(root, row):
    pinned, = (p for p in root.glob(f"*/cases/{row['episode']}/EXECUTION_SPEC.json")
               if sha(p) == row["original_markers"])
    case = pinned.parent
    terminal = read(pin(case / "RESULT.private.json", row["result"]))
    require(read(case / "control/LEASE.json")["status"] == "CLOSED"
            and terminal["shutdown_clean"] and not terminal["unsafe"] and not terminal["technical_failure"]
            and terminal["detail"]["score"]["task_success"], "A true, clean native TRAIN terminal is required")
    return case


def zip_episode(root, episode, added, extras, output):
    from go2arx5_vla.sla_adapter.dense_dataset_v4 import member

    require(episode["id"] == added["episode"] and episode["split"] == "TRAIN"
            and episode["scene_sha256"] == added["scene"] and episode["individual_pass"]
            and episode["windows"] == added["windows"], "The original RGB and TRAIN episode do not match")
    for row in [episode["labels"], *episode["frame_shards"]]:
        original = pin(member(root, row["file"]), row["sha256"])
        target = output / row["file"]
        materialize(original, target)
        pin(target, row["sha256"])
    retained = output / episode["id"] / "R4_ACTION.npz"
    materialize(pin(extras["path"], extras["sha256"]), retained)
    pin(retained, extras["sha256"])
    return dict(episode=episode, corrected=dict(file=str(retained.relative_to(output)), sha256=extras["sha256"]),
                native_result_sha256=added["result"], original_execution_spec_sha256=added["original_markers"],
                population="newest" if episode["id"].startswith("EF5_") else "legacy")


def physical_aux(case, corrected, destination):
    from go2arx5_vla.sla_adapter.gripper_latent_v2 import label_latent

    with np.load(corrected, allow_pickle=False) as data:
        ticks, mask, actions = (data[k] for k in ("native_step", "valid", "action"))
    require(mask.dtype == bool and mask[:, :10].all() and not np.any(np.diff(mask.astype(np.int8), axis=1) > 0),
            "Actual query support must be a contiguous 50 ms prefix")
    source = case / "native-01/MOBILE_SAMPLES.private.jsonl"
    atoms = {}
    with source.open() as f:
        for atom in map(json.loads, f):
            tick = atom["native_start_step"]
            require(tick not in atoms and atom["native_end_step"] == tick + 25,
                    "Expected one actual jaw command per physical 50 ms atom")
            if atom["safety_valid"]:
                atoms[tick] = atom["installed_EEF10"][9] * .044
    width = np.zeros(mask.shape)
    previous = np.asarray([atoms[int(tick) - 25] for tick in ticks])
    for i, j in np.argwhere(mask):
        width[i, j] = atoms[int(ticks[i]) + 25 * int(j)]
    grip = (width[mask] - .002) / .040
    require(np.isfinite(previous).all() and np.all((previous >= .002 - 1e-10) & (previous <= .042 + 1e-10))
            and np.all((grip >= -1e-10) & (grip <= 1. + 1e-10))
            and np.array_equal(label_latent(np.clip(grip, 0., 1.)).astype(np.float32), actions[..., 6][mask]),
            "Exact issued jaw widths must reproduce the trained gripper latent")
    with destination.open("xb") as f:
        np.savez_compressed(f, actual_width_m=width, prior_actual_width_m=previous, source_tick=ticks, valid=mask)
    return dict(file=str(destination.relative_to(destination.parents[1])), sha256=sha(destination),
                native_log_sha256=sha(source), windows=len(ticks),
                complete_H40=int(mask.all(axis=1).sum()), actual_support=int(mask.sum()))


def read_added(root, manifest, statistics, resize_with_pad):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32
    from go2arx5_vla.sla_adapter.dense_dataset_v4 import member
    from go2arx5_vla.sla_adapter.multitask_data_v6 import TeacherDataset

    class Added(TeacherDataset):
        def __init__(self):
            self.root = root
            self.manifest = dict(episodes=[r["episode"] for r in manifest["episodes"]],
                                 windows=manifest["windows"])
            self.statistics = {k: Quantiles(statistics[k]["q01"], statistics[k]["q99"]) for k in ("state", "actions")}
            self._load_episodes(resize_with_pad, 256*1024**2)
            self.corrected, self.widths = {}, {}
            for record in manifest["episodes"]:
                eid = record["episode"]["id"]
                action, aux = record["corrected"], record["aux"]
                with np.load(pin(member(root, action["file"]), action["sha256"]), allow_pickle=False) as f:
                    labels = {k: f[k] for k in ("action", "valid", "phase", "native_step")}
                with np.load(pin(member(root, aux["file"]), aux["sha256"]), allow_pickle=False) as f:
                    widths = {k: f[k] for k in ("actual_width_m", "prior_actual_width_m", "source_tick", "valid")}
                capture = self.labels[eid]
                n = record["episode"]["windows"]
                mask = labels["valid"]
                require(labels["action"].shape == (n,40,7) and np.isfinite(labels["action"]).all()
                        and mask.shape == (n,40) and mask.dtype == bool and mask[:, :10].all()
                        and np.array_equal(labels["phase"], capture["phase"])
                        and np.array_equal(labels["native_step"], capture["native_step"])
                        and np.array_equal(mask, widths["valid"])
                        and np.array_equal(widths["source_tick"], capture["native_step"])
                        and np.isfinite(widths["actual_width_m"][mask]).all()
                        and np.isfinite(widths["prior_actual_width_m"]).all(), "Portable action, image clock or aux differ")
                self.corrected[eid] = labels
                self.widths[eid] = widths

        def __getitem__(self, index):
            item = super().__getitem__(index)
            ep, k = self.rows[index]
            labels = self.corrected[ep["id"]]
            value = pad32(self.statistics["actions"].normalize(labels["action"][k])).astype(np.float32)
            item["actions"] = np.where(labels["valid"][k, :, None], value, 0.)
            return item

        def support(self, index):
            ep, k = self.rows[index]
            return self.corrected[ep["id"]]["valid"][k]

        def jaw(self, index):
            ep, k = self.rows[index]
            aux = self.widths[ep["id"]]
            require(aux["valid"][k].all(), "Auxiliary examples need complete physical support")
            return aux["prior_actual_width_m"][k], aux["actual_width_m"][k]

    return Added()


def prepare(base, native, archive, out):
    from openpi_client.image_tools import resize_with_pad

    factory = base / "core-factory-v5-20261126-01"
    retention = base / "v6-retention-20261126-01"
    source = base / "v61-corrected-retention-20261126-01"
    work = base / "v62-science-first-20261127-01"
    original = read(pin(archive / "MANIFEST.private.json", PINS["target_archive"]))
    old = read(pin(retention / "REEXPRESSED-G7-V1/MANIFEST.json", PINS["target_corrected"]))
    closed = source / "CORRECTED-17-CORE-READBACK.json"
    evidence = read(pin(closed, LEGACY_AUDIT))
    require(evidence["legacy_69_manifest_sha256"] == LEGACY_CATALOG, "Legacy audit belongs to another population")
    legacy = {r["episode"]: r for r in evidence["episodes"]}
    snapshot = factory / "SNAPSHOTS/TRAIN-LEGACY-064"
    catalog = read(snapshot / "MANIFEST.json")
    require(read(snapshot / "READY.json")["manifest_sha256"] == sha(snapshot / "MANIFEST.json"),
            "Use the closed legacy snapshot")
    protected = set(catalog["excluded_dev_scenes"])
    previous = {e["id"]: e for e in original["episodes"]}
    base_ids = {r["episode"] for r in old["episodes"]}
    require(len(base_ids) == 52 and base_ids <= set(previous) and len(legacy) == 17, "The fixed R4 parent population changed")
    delta = retention / "REEXPRESSED-LEGACY-DELTA-68-01"
    fixed = {r["episode"]: dict(path=str(delta / r["label"]), sha256=r["sha256"])
             for r in read(delta / "MANIFEST.json")["episodes"]}
    extra = retention / "REEXPRESSED-LEGACY-69-INCREMENT-01"
    inc = read(extra / "ACTUAL.json")
    fixed[inc["episode"]] = dict(path=str(extra / (inc["episode"] + ".npz")), sha256=inc["corrected_labels_sha256"])
    stage = []
    for row in catalog["episodes"]:
        if row["origin"] == "NEW_LEGACY_READMISSION":
            stage.append((archive, previous[row["episode"]], row, fixed[row["episode"]], signed_case(native, row)))
    for name in ("LEGACY_RECOVERY", "LEGACY-RED-PLACE-REREAD-01", "LEGACY-N1-069-REAL-READBACK-01"):
        for entry in read(factory / name / "ACTUAL.json")["admissions"]:
            ep = entry["episode"]
            row = dict(entry, episode=ep["id"])
            case = signed_case(factory if name == "LEGACY-N1-069-REAL-READBACK-01" else native, row)
            marker, = (r["episode"] for r in read(case.parents[1] / "RECORDS.private.json")
                       if r["condition"] == ep["id"])
            ep_file = pin(marker["path"], entry["readmission_origin"])
            require(read(ep_file) == ep, "The recovered episode differs from its original admission")
            stage.append((ep_file.parent.parent, ep, row, fixed[ep["id"]], case))
    require({ep["id"] for _, ep, _, _, _ in stage} == set(fixed) == set(legacy)
            and len(stage) == len(legacy), "17 legacy additions must have full geometry and admissible RGB")
    for name, digest in RECENT_BATCHES.items():
        for row in read(pin(work / name / "MANIFEST.json", digest))["episodes"]:
            marker = row["original_episode"]
            ep = read(pin(marker["path"], marker["sha256"]))
            stage.append((Path(row["archive"]), ep, row, row["corrected"], signed_case(work, row)))
    require(len(stage) == 22 and len({e["id"] for _, e, _, _, _ in stage} | base_ids) == 74
            and all(e["scene_sha256"] not in protected for _, e, _, _, _ in stage),
            "Duplicate episode identity or held-out scene in A74")
    stats = read(pin(work / "DIVERSE-PLACE-TRAIN-02-LABELS-R4/MANIFEST.json",
                     RECENT_BATCHES["DIVERSE-PLACE-TRAIN-02-LABELS-R4"]))
    bindings = {k: stats[k] for k in ("state_statistics", "action_statistics")}
    statistics = {role: read(pin(bindings[key]["path"], bindings[key]["sha256"]))[role]
                  for role, key in (("state", "state_statistics"), ("actions", "action_statistics"))}
    out.mkdir(mode=0o700)
    members = []
    for root, ep, row, corrected, case in stage:
        stored = zip_episode(root, ep, row, corrected, out)
        stored["aux"] = physical_aux(case, out / stored["corrected"]["file"], out / ep["id"] / "R4_AUX.npz")
        if ep["id"] in legacy:
            proof = legacy[ep["id"]]
            require(stored["corrected"]["sha256"] == proof["corrected_sha256"]
                    and stored["aux"]["actual_support"] == proof["supported_points"], "Legacy seal changed")
        members.append(stored)
    manifest = dict(plan=PLAN, corpus="A74", episodes=members, added=22, inherited=52,
                    statistics_sha256={key: item["sha256"] for key, item in bindings.items()},
                    windows=sum(r["episode"]["windows"] for r in members),
                    full_H40=sum(r["aux"]["complete_H40"] for r in members),
                    actual_support=sum(r["aux"]["actual_support"] for r in members),
                    native_integrations=0, model_forwards=0, optimizer_visits=0)
    dataset = read_added(out, manifest, statistics, resize_with_pad)
    offset, readbacks = 0, []
    for row in members:
        ep = row["episode"]
        for index in (offset, offset + ep["windows"] - 1):
            sample = dataset[index]
            mask = dataset.support(index)
            require(sample["actions"].shape == (40,32) and sample["state"].shape == (32,)
                    and not sample["actions"][~mask].any()
                    and sample["image_mask"] == dict(base_0_rgb=True, left_wrist_0_rgb=True, right_wrist_0_rgb=False),
                    "Actual TRAIN reader changed the two RGB streams or the physical mask")
            readbacks.append(dict(episode=ep["id"], index=index, support=int(mask.sum())))
        offset += ep["windows"]
    write(out / "MANIFEST.json", manifest)
    write(out / "CORE_READBACK.json", dict(manifest=receipt(out / "MANIFEST.json"), actual_RGB_boundaries=readbacks,
          old52_archive_sha256=PINS["target_archive"], old52_corrected_sha256=PINS["target_corrected"],
          legacy_audit_sha256=LEGACY_AUDIT, optimizer_visits=0))
    print(sha(out / "MANIFEST.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "native", "archive", "out"):
        parser.add_argument(name, type=Path)
    prepare(**vars(parser.parse_args()))
