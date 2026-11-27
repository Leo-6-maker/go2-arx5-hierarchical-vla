"""Read the sealed SOURCE608 candidate with the existing native TRAIN codecs."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from v64_source608_public_bank import CATALOGUE, EPISODES, NATIVE, WIDTH, digest, read_pinned, write

JOINT_RECEIPT = "cb65c9fec9bcc26f2acf3aa3d97b6dfede26f59d19d0d4072b108ed14ff0c270"
TOKENIZER = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
LEGACY_DUAL = "5a558e07034f2f683a26705585716760930e73acca1c396898ef6a01c3b25e71"


class ScheduledRows:
    """Resolve each sealed slot to one exact release and native reader row."""

    def __init__(self, records, readers, offsets):
        self.manifest = readers[records[0]["version"]].manifest
        self.episodes = {i: readers[row["version"]].episodes[row["reader_episode"]]
                         for i, row in enumerate(records)}
        self.rows = [(i, row["anchor"]) for i, row in enumerate(records)]
        self.entries = [(readers[row["version"]], offsets[row["version"]][row["reader_episode"]] + row["anchor"])
                        for row in records]
        for i, ((reader, index), row) in enumerate(zip(self.entries, records, strict=True)):
            if reader.rows[index] != (row["reader_episode"], row["anchor"]) or self.episodes[i]["split"] != "TRAIN":
                raise ValueError("Scheduled slot is outside the exact TRAIN episode")

    def __getitem__(self, slot):
        reader, index = self.entries[slot]
        return reader[index]


def video_root(root, manifest):
    path = Path(manifest["video_root"])
    return path if path.is_absolute() else root / path


def release_reader(role, ref, norm, legacy_stats, resize):
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptRobodojoDataset
    from v6_robodojo_reader import ExpansionDataset
    from v6_ville_reader import SingleArmReader
    from v62_sc_source import CAMERAS, HORIZON, NATIVE_HZ, SCHEMA

    path = Path(ref["path"])
    root, expected = path.parent, ref["sha256"]
    manifest = read_pinned(path, expected)
    if role == "robodojo" and expected == LEGACY_DUAL:
        return AdaptRobodojoDataset(dict(root=root, manifest=ref, inherited_statistics={
            "path": str(legacy_stats), "sha256": norm}), resize_with_pad=resize)
    if manifest["statistics"]["sha256"] != norm:
        raise ValueError("The source statistics differ from the sealed schedule")
    if role == "robodojo":
        return ExpansionDataset(root, expected, raw_binding=None, resize_with_pad=resize)
    options = dict(schema=SCHEMA, native_hz=NATIVE_HZ, horizon=HORIZON, cameras=CAMERAS) if role == "sc_vla" else {}
    return SingleArmReader(root, expected, video_root(root, manifest), split="TRAIN", resize=resize,
                           cache_episodes=1, **options)


def public_batch(role, positions, records, readers, offsets, tokenizer):
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch
    from v62_sc_training_inputs import prepare

    dataset = ScheduledRows(records, readers, offsets)
    slots = range(len(records))
    if role == "sc_vla":
        return prepare(dataset, tokenizer, positions, slots)
    return prepare_host_batch(dataset, tokenizer, "target" if role == "ville" else role, positions, slots)


def batch_receipt(role, batch):
    actions = np.asarray(batch["actions"])
    obs = batch["observation"]
    mask = {key: np.asarray(value) for key, value in obs["image_mask"].items()}
    active = 14 if role == "robodojo" else 7
    if (actions.shape != (WIDTH, NATIVE[role][1], 32) or not np.isfinite(actions).all()
            or np.any(actions[..., active:]) or
            any(not rgb.any() for rgb in map(np.asarray, (obs["image"][key] for key in mask if mask[key].all())))
            or not mask["base_0_rgb"].all() or not mask["right_wrist_0_rgb"].all()
            or np.any(mask["left_wrist_0_rgb"] != (role == "robodojo"))):
        raise ValueError("Model-ready source tensors do not match the recorded camera and action contract")
    arrays = {"actions": actions, "state": obs["state"], "tokens": obs["tokenized_prompt"],
              "token_mask": obs["tokenized_prompt_mask"]}
    for field in ("image", "image_mask", "image_padding_mask"):
        arrays.update({f"{field}/{camera}": value for camera, value in obs[field].items()})
    if "action_support" in batch:
        arrays["action_support"] = batch["action_support"]
    fingerprints = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                    for name, value in arrays.items()}
    return dict(shape=list(actions.shape), active_channels=active, actual_native_hz=NATIVE[role][0],
                valid_cameras={key: int(value.sum()) for key, value in mask.items()}, tensors_sha256=fingerprints)


def replay_receipt(bank, log):
    matches = dict(target=0, libero=0, auxiliary=0)
    lines = log.read_text().splitlines()
    if len(lines) != 250:
        raise ValueError("Expected the complete historical SOURCE250 ledger")
    for i, line in enumerate(lines):
        row = json.loads(line)
        if row["step"] != i + 1:
            raise ValueError("Historical SOURCE250 ledger has a gap")
        parts = row["components"]
        for role in ("target", "libero"):
            if (parts[role]["visits"] == 16 and
                    parts[role]["visit_sha256"] == hashlib.sha256(bank[role][i].tobytes()).hexdigest()):
                matches[role] += 1
        if parts["auxiliary"]["visits"] == 4 and parts["auxiliary"]["slots"] == bank["auxiliary_slots"][i].tolist():
            matches["auxiliary"] += 1
    return dict(historical_updates=len(lines), matches=matches, sha256=digest(log),
                historical_access_only=True)


def run(catalogue, public, joint, tokenizer_path, log, output, row):
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from openpi_client.image_tools import resize_with_pad

    start = time.perf_counter()
    catalog = read_pinned(catalogue, CATALOGUE)
    sealed = read_pinned(joint / "ACTUAL.json", JOINT_RECEIPT)
    index = read_pinned(public / "INDEX.json", sealed["public_episode_index_sha256"])
    if (digest(joint / "BANK.npz") != sealed["joint_bank_sha256"]
            or digest(tokenizer_path) != TOKENIZER or not 0 <= row < sealed["steps"]):
        raise ValueError("SOURCE608 joint bank or tokenizer changed")
    by_version = {release["manifest"]["sha256"]: release for release in catalog["versions"]}
    inherited = next(release for release in catalog["versions"]
                     if release["domain"] == "robodojo" and release["episodes"] == 107)
    inherited_stats = Path(inherited["manifest"]["path"]).with_name("STATISTICS.json")
    tokenizer = Tokenizer(str(tokenizer_path))
    roles = {}
    with np.load(joint / "BANK.npz", allow_pickle=False) as bank:
        replay = replay_receipt(bank, log)
        for role in EPISODES:
            records = [dict(index[role][int(i)], anchor=int(anchor)) for i, anchor in zip(
                bank[f"{role}_episode"][row], bank[f"{role}_anchor"][row], strict=True)]
            readers, offsets = {}, {}
            for record in records:
                version = record["version"]
                if version in readers:
                    continue
                release = by_version[version]
                if release["domain"] != role or release["normalized_by"] != record["normalization_sha256"]:
                    raise ValueError("Scheduled release has a different domain or normalization")
                reader = release_reader(role, release["manifest"], release["normalized_by"],
                                        inherited_stats, resize_with_pad)
                readers[version] = reader
                offsets[version] = {episode: i for i, (episode, anchor) in enumerate(reader.rows) if anchor == 0}
            print(f"{role} {len(records)} visits across {len(readers)} prepared releases", flush=True)
            batch = public_batch(role, range(row * WIDTH, (row + 1) * WIDTH), records, readers, offsets, tokenizer)
            roles[role] = dict(batch_receipt(role, batch), visits=records)
    write(output, dict(kind="SOURCE608_CORELAB_ACTUAL_PUBLIC_HOST_BATCH", schedule_row=row,
                      joint_bank_sha256=sealed["joint_bank_sha256"], tokenizer_sha256=TOKENIZER,
                      actual_public_windows=sum(len(role["visits"]) for role in roles.values()), roles=roles,
                      historical_replay=replay, new_optimizer_visits=0, new_model_forwards=0,
                      wall_seconds=time.perf_counter() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("catalogue", "public", "joint", "tokenizer", "historical_log", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--row", type=int, default=0)
    options = parser.parse_args()
    run(options.catalogue, options.public, options.joint, options.tokenizer,
        options.historical_log, options.output, options.row)
