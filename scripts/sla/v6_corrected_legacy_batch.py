"""Read a real 64-row target batch and four physical auxiliary contexts from legacy 68."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import input_contract
from v6_reexpress_legacy_delta import LATEST, ORIGINAL


def preflight(native, factory, retained, original_aux, archive, parent_gate, out):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32
    from go2arx5_vla.sla_adapter.multitask_data_v6 import TeacherDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.pi0292.durable import write

    final = factory / "SNAPSHOTS/TRAIN-LEGACY-068/MANIFEST.json"
    read(pin(final, LATEST))
    original = retained / "REEXPRESSED-G7-V1"
    delta = retained / "REEXPRESSED-LEGACY-DELTA-68-01"
    extra_aux = retained / "REEXPRESSED-LEGACY-DELTA-AUX-01"
    old = read(pin(original / "MANIFEST.json", ORIGINAL))["episodes"]
    extension = read(pin(delta / "MANIFEST.json", "acdaa698c0c30bc1e68ea4f48556cd0f69c6c434da407c245d10c118a85f4027"))["episodes"]
    base = read(factory / "SNAPSHOTS/TRAIN-LEGACY-064/MANIFEST.json")
    full = read(pin(archive / "MANIFEST.private.json", base["signed_original_manifest_sha256"]))
    members = {r["episode"] for r in base["episodes"]}
    extra_ids = {r["episode"] for r in extension}
    groups = {archive: [r for r in full["episodes"] if r["id"] in members]}
    recovered = defaultdict(list)
    for tranche in ("LEGACY_RECOVERY", "LEGACY-RED-PLACE-REREAD-01"):
        for row in read(factory / tranche / "ACTUAL.json")["admissions"]:
            record = row["episode"]
            eid = record["id"]
            match = [s for s in native.glob(f"*/cases/{eid}/EXECUTION_SPEC.json") if sha(s) == row["original_markers"]]
            require(len(match) == 1, "The recovered RGB source identity changed")
            entry, = (r for r in read(match[0].parents[2] / "RECORDS.private.json") if r["condition"] == eid)
            declared = entry["episode"]
            episode_path = pin(declared["path"], row["readmission_origin"])
            source = episode_path.parent.parent
            require(declared["sha256"] == row["readmission_origin"] and read(episode_path) == record,
                    "Existing RGB episode mismatch")
            recovered[source].append(record)
    groups.update(recovered)
    require(sum(map(len, groups.values())) == 68, "Incomplete combined TRAIN membership")
    gate = read(parent_gate)["gate"]
    _, _, statistics = input_contract(gate, Path(gate["package"]["path"]))
    stats = {key: Quantiles(value["q01"], value["q99"]) for key, value in statistics.items()}
    labels = {r["episode"]: (original / r["labels"], r["sha256"]) for r in old}
    labels.update({r["episode"]: (delta / r["label"], r["sha256"]) for r in extension})
    selected: dict[str, list] = {}

    class Reader(TeacherDataset):
        def __init__(self, root, episodes):
            self.root = root
            self.manifest = dict(episodes=episodes, windows=sum(e["windows"] for e in episodes))
            self.statistics = stats
            self._load_episodes(resize_with_pad, 112 * 1024**2)

    for path, episodes in groups.items():
        reader = Reader(path, episodes)
        for i, (ep, anchor) in enumerate(reader.rows):
            eid = ep["id"]
            selected.setdefault(eid, []).append((reader, i, anchor))
    require(set(selected) == set(labels) and sum(map(len, selected.values())) == 6034,
            "Actual direct reader did not reach the 68 sealed episodes")
    old_ids = [r["episode"] for r in old]
    preview = [selected[eid][0] for eid in old_ids[:32]]
    preview.extend(point for identity in sorted(extra_ids) for point in (selected[identity][0], selected[identity][-1]))
    require(len(preview) == 64, "Expected one complete learner-width CPU preview")

    auxiliary = {}
    for root in (original_aux, extra_aux):
        for r in read(root / "MANIFEST.json")["episodes"]:
            require(r["episode"] not in auxiliary, "Overlapping auxiliary provenance")
            auxiliary[r["episode"]] = (root / r["file"], r["sha256"])
    aux_counts = Counter()
    window_rows, actions, widths = [], [], []
    for i, (reader, index, anchor) in enumerate(preview):
        ep = reader.rows[index][0]
        eid = ep["id"]
        item = reader[index]
        with np.load(pin(*labels[eid]), allow_pickle=False) as packet:
            raw, mask, tick = packet["action"][anchor], packet["valid"][anchor], int(packet["native_step"][anchor])
        action = np.where(mask[:, None], pad32(stats["actions"].normalize(raw)).astype(np.float32), 0.)
        require(action.shape == (40, 32) and item["state"].shape == (32,) and np.isfinite(action).all()
                and np.isfinite(item["state"]).all() and mask[:10].all() and not action[~mask].any()
                and item["image_mask"] == dict(base_0_rgb=True, left_wrist_0_rgb=True, right_wrist_0_rgb=False),
                "A real RGB example or corrected target label lost its learner ABI")
        require(all(item["image"][key].shape == (224, 224, 3) and item["image"][key].dtype == np.uint8
                    for key in ("base_0_rgb", "left_wrist_0_rgb")), "The original learner RGB views changed")
        actions.append(action)
        row = dict(slot=i, episode=eid, query=anchor, native_step=tick, supported=int(mask.sum()), auxiliary=False)
        population = "added" if eid in extra_ids else "original"
        if mask.all() and aux_counts[population] < 2:
            with np.load(pin(*auxiliary[eid]), allow_pickle=False) as source:
                require(source["source_tick"][anchor] == tick and source["valid"][anchor].all()
                        and np.isfinite(source["actual_width_m"][anchor]).all()
                        and np.isfinite(source["prior_actual_width_m"][anchor]), "Physical auxiliary context changed")
                widths.append(source["actual_width_m"][anchor])
            aux_counts[population] += 1
            row["auxiliary"] = True
        window_rows.append(row)
    batch, grip = np.stack(actions), np.stack(widths)
    require(batch.shape == (64, 40, 32) and grip.shape == (4, 40)
            and aux_counts == {"added": 2, "original": 2}, "Need four complete physical auxiliary slots")
    write(out, dict(data_manifest_sha256=LATEST, windows=6034, episodes=68, preview_rows=window_rows,
          actual_two_rgb_pairs_read=64, actual_learner_action_shape=list(batch.shape), physical_aux_shape=list(grip.shape),
          original_and_added_aux=dict(aux_counts), actual_optimizer_visits=0, model_forwards=0, native_integrations=0))
    print(sha(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "factory", "retained", "original_aux", "archive", "parent_gate", "out"):
        parser.add_argument(name, type=Path)
    preflight(**vars(parser.parse_args()))
