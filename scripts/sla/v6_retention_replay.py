"""Select a fixed phase- and mask-matched old-skill replay contrast."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import data_scope, input_contract


STEPS = 200
REPLAY_PER_STEP = 8
SEED = 26112602


def read_replay(site, original, recovered, base_dataset, evidence, recovered_labels=None):
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.codec_v1 import pad32
    from go2arx5_vla.sla_adapter.multitask_data_v6 import TeacherDataset

    first = site / "SNAPSHOTS/TRAIN-LEGACY-064"
    manifest = read(pin(site / "SNAPSHOTS/TRAIN-LEGACY-067/MANIFEST.json", evidence["manifest_sha256"]))
    require(sha(first / "MANIFEST.json") == manifest["original_manifest_sha256"],
            "Wrong signed source for replay")
    original_set = read(original / "SEALED.private.json")
    signed = read(pin(original / "MANIFEST.private.json", original_set["manifest_sha256"]))
    candidates = {row["episode"]: row for row in evidence["rows"]}
    from_legacy = {e["id"]: e for e in signed["episodes"] if e["id"] in candidates}
    from_closed = {row["episode"]["id"]: row["episode"] for row in manifest["actual_added"]
                   if row["episode"]["id"] in candidates}
    require(len(from_legacy) == 2 and len(from_closed) == 1 and set(from_legacy | from_closed) == set(candidates),
            "Unexpected old skill roster")

    class Replay(TeacherDataset):
        def __init__(self, root, episodes, labels):
            self.root, self.statistics = root, base_dataset.statistics
            self.manifest = dict(episodes=list(episodes.values()), windows=sum(e["windows"] for e in episodes.values()))
            self._load_episodes(resize_with_pad, 96 * 1024**2)
            self.actual = {}
            for identity, episode in episodes.items():
                row = candidates[identity]
                require(episode["split"] == "TRAIN" and episode["scene_sha256"] == row["scene_sha256"],
                        "Unexpected raw TRAIN episode or layout")
                with np.load(pin(labels / (identity + ".npz"), row["sha256"]), allow_pickle=False) as data:
                    true = data["action"]
                    mask = data["valid"]
                    require(np.array_equal(data["phase"], self.labels[identity]["phase"])
                            and np.array_equal(data["native_step"], self.labels[identity]["native_step"])
                            and true.shape == (episode["windows"], 40, 7)
                            and mask.shape == (episode["windows"], 40)
                            and mask.dtype == bool, "Replay RGB timeline lost its actual command labels")
                    self.actual[identity] = true, mask

        def support(self, index):
            episode, query = self.rows[index]
            return self.actual[episode["id"]][1][query]

        def __getitem__(self, index):
            item = super().__getitem__(index)
            episode, query = self.rows[index]
            true, mask = self.actual[episode["id"]]
            value = pad32(self.statistics["actions"].normalize(true[query])).astype(np.float32)
            item["actions"] = np.where(mask[query, :, None], value, 0.)
            return item

    return (Replay(original, from_legacy, first),
            Replay(recovered, from_closed, recovered_labels or site / "LEGACY_RECOVERY"))


class ReplayDataset:
    def __init__(self, target, replay):
        self.original = target
        self.extra = [(reader, i) for reader in replay for i in range(len(reader))]
        self.statistics = target.statistics
        self.rows = list(target.rows) + [reader.rows[i] for reader, i in self.extra]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index < len(self.original):
            return self.original[index]
        reader, query = self.extra[index - len(self.original)]
        return reader[query]

    def support(self, index):
        if index < len(self.original):
            return self.original.support(index)
        reader, query = self.extra[index - len(self.original)]
        return reader.support(query)


def bind(original, original_auxiliary, target, fixed):
    require(fixed["updates"] == STEPS and fixed["replaced_per_update"] == REPLAY_PER_STEP
            and fixed["original_hold_exposure_unchanged"] and len(fixed["rows"]) == STEPS
            and fixed["original_bank_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
            and fixed["original_auxiliary_sha256"] == hashlib.sha256(original_auxiliary.tobytes()).hexdigest(),
            "The same original exposure and accepted phase-matched roster are required")
    lookup = {(episode["id"], query): i for i, (episode, query) in enumerate(target.rows)
              if i >= len(target.original)}
    result = original.copy()
    for step, row in enumerate(fixed["rows"]):
        replacement = row["replacements"]
        slots = [r["slot"] for r in replacement]
        require(row["update"] == step + 1 and row["auxiliary_original"] == original_auxiliary[step].tolist()
                and len(slots) == len(set(slots)) == REPLAY_PER_STEP
                and all(0 <= i < original.shape[1] and i not in row["auxiliary_original"] for i in slots),
                "Replay must preserve the original physical auxiliary inputs")
        for entry in replacement:
            i = entry["slot"]
            previous = int(original[step, i])
            incoming = lookup[entry["incoming_episode"], entry["incoming_query"]]
            before, after = (target.rows[n] for n in (previous, incoming))
            require(entry["from_train_index"] == previous and before[0]["task"].startswith("PLACE-")
                    and before[0]["expert_stage_audit_only"][before[1]] == entry["phase"]
                    and after[0]["expert_stage_audit_only"][after[1]] == entry["phase"]
                    and target.support(previous).all() and target.support(incoming).all(),
                    "Losing task, phase, or true full-H40 matching would change the contrast")
            result[step, i] = incoming
    return result


def prepare(original_archive, consumed, parent_gate, data_factory, original_bank, evidence, output):
    gate = read(parent_gate)["gate"]
    _, _, stats = input_contract(gate, Path(gate["package"]["path"]))
    baseline = data_scope(original_archive, consumed, stats)
    candidates = read(evidence)
    require(len(baseline) == 4450 and len(candidates["rows"]) == 3 and candidates["trained"] == 0,
            "Use the original target and the three actual closed TRAIN candidates")
    first = data_factory / "SNAPSHOTS/TRAIN-LEGACY-064"
    composite = data_factory / "SNAPSHOTS/TRAIN-LEGACY-067/MANIFEST.json"
    manifest = read(pin(composite, candidates["manifest_sha256"]))
    require(sha(first / "MANIFEST.json") == manifest["original_manifest_sha256"],
            "The real first catalog no longer matches the composite release")
    with np.load(original_bank, allow_pickle=False) as saved:
        target, auxiliary = saved["target_indices"], saved["auxiliary_slots"]
    require(target.shape == (2000, 64) and auxiliary.shape == (2000, 4),
            "Wrong original exposure record")

    pools = defaultdict(lambda: defaultdict(list))
    for row in sorted(candidates["rows"], key=lambda r: r["episode"]):
        label = row["episode"] + ".npz"
        directory = first if (first / label).is_file() else data_factory / "LEGACY_RECOVERY"
        with np.load(pin(directory / label, row["sha256"]), allow_pickle=False) as data:
            full = data["valid"].all(axis=1)
            require(int(full.sum()) == row["complete_H40"], "Replay no longer has original actual masks")
            for local in np.flatnonzero(full):
                pools[str(data["phase"][local])][row["episode"]].append(int(local))

    rng = np.random.default_rng(SEED)
    replacements, episode_visits, phase_visits = [], Counter(), Counter()
    for i, (row, auxiliary_slots) in enumerate(zip(target[:STEPS], auxiliary[:STEPS], strict=True)):
        selected = []
        for slot in rng.permutation(len(row)):
            if slot in auxiliary_slots or not baseline.support(row[slot]).all():
                continue
            episode, local = baseline.rows[row[slot]]
            if not episode["task"].startswith("PLACE-"):
                continue
            phase = episode["expert_stage_audit_only"][local]
            if phase not in pools:
                continue
            options = pools[phase]
            # Balance by episode where that actual task phase is available.
            smallest = min(episode_visits[e] for e in options)
            eligible = [e for e in options if episode_visits[e] == smallest]
            replay = str(rng.choice(eligible))
            selected.append(dict(slot=int(slot), from_train_index=int(row[slot]), phase=phase,
                                 incoming_episode=replay, incoming_query=int(rng.choice(options[replay]))))
            episode_visits[replay] += 1
            phase_visits[phase] += 1
            if len(selected) == REPLAY_PER_STEP:
                break
        require(len(selected) == REPLAY_PER_STEP, f"Not enough matching H40 rows at update {i + 1}")
        replacements.append(dict(update=i+1, auxiliary_original=auxiliary_slots.tolist(), replacements=selected))

    result = dict(updates=STEPS, per_update_target=64, per_update_auxiliary=4,
                  replaced_per_update=REPLAY_PER_STEP, phase_and_full_mask_matched=True,
                  original_hold_exposure_unchanged=True,
                  original_bank_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                  original_auxiliary_sha256=hashlib.sha256(auxiliary.tobytes()).hexdigest(),
                  replay_candidate_sha256=sha(evidence), replay_episode_visits=dict(episode_visits),
                  replay_phase_visits=dict(phase_visits), rows=replacements, optimizer_updates=0,
                  physics_ticks=0, seed=SEED)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original_archive", "consumed", "parent_gate", "data_factory", "original_bank", "evidence", "output"):
        parser.add_argument(name, type=Path)
    prepare(**vars(parser.parse_args()))
