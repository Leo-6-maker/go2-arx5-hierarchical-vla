"""Bind native command supervision and original RoboDojo/LIBERO input views."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import MAIN, data_scope
from v4_long_engine import BATCH


def inputs(custody, short_capsule, prepared, parent):
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptRobodojoDataset, LIBERO_MANIFEST, LiberoDataset
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from scripts.sla.hopper_inputs import inputs as original_inputs

    reference = short_capsule / "incoming/reference"
    checkpoint = read(reference / "CHECKPOINT.json")
    target_stats = dict(state=read(pin(reference / "STATE.json", checkpoint["state_sha256"]))["state"],
                        actions=read(pin(reference / "ACTION.json", checkpoint["action_sha256"]))["actions"])
    tokenizer = Tokenizer(str(pin(reference / "TOKENIZER.model", checkpoint["tokenizer_sha256"])))
    commands = short_capsule / "incoming/V3-MAIN-CONSUMED-DATA-02"
    target = data_scope(custody / "foundation-layout-v2/train-ready/TRAIN-069", commands, target_stats)
    plan = original_inputs(custody / "assets.verified")["data"]
    lib = plan["libero"]
    require(lib["manifest"]["sha256"] == LIBERO_MANIFEST, "Original LIBERO training manifest changed")
    datasets = dict(target=target,
                    robodojo=AdaptRobodojoDataset(plan["robodojo"], resize_with_pad=resize_with_pad),
                    libero=LiberoDataset(lib["root"], lib["manifest"], split="TRAIN", resize_with_pad=resize_with_pad))
    require({k: len(d) for k, d in datasets.items()} == {"target": 4450, "robodojo": 210398, "libero": 209146},
            "An original admitted input window is missing")

    bank_root = prepared / "prepared"
    binding = read(bank_root / "ACTUAL.json")
    with np.load(pin(bank_root / "BANK.npz", binding["archive_sha256"]), allow_pickle=False) as saved:
        indices, slots = saved["target_indices"], saved["auxiliary_slots"]
    require(indices.shape == (2000, 64) and slots.shape == (2000, 4)
            and hashlib.sha256(indices.tobytes()).hexdigest() == binding["target_visit_bank_sha256"]
            and hashlib.sha256(slots.tobytes()).hexdigest() == binding["aux_slot_bank_sha256"],
            "The fixed 2K target and auxiliary banks changed")
    native = prepared / "native_aux-01"
    receipt = read(native / "MANIFEST.json")
    require(receipt["command_manifest_sha256"] == binding["consumed_command_manifest_sha256"]
            == sha(commands / "MANIFEST.json"), "Target sampling and native auxiliary labels disagree")
    widths = {}
    for row in receipt["episodes"]:
        with np.load(pin(native / row["file"], row["sha256"]), allow_pickle=False) as f:
            widths[row["episode"]] = (f["prior_actual_width_m"], f["actual_width_m"], f["valid"])

    def auxiliary(selected):
        pair = []
        for i in selected:
            ep, k = target.rows[i]
            prior, label, mask = widths[ep["id"]]
            require(mask[k].all() and np.array_equal(mask[k], target.support(i)), "Native H40 support differs")
            pair.append((prior[k], label[k]))
        return tuple(np.asarray(a, dtype=np.float32) for a in zip(*pair, strict=True))

    banks = {"target": indices}
    progress = read(parent / "COMPLETE.json")["progress"]
    for domain in ("robodojo", "libero"):
        start = progress["role_visits"][domain]
        require(start == progress["phase_step"] * BATCH[domain], "Source parent phases differ")
        original = np.load(custody / "banks.verified" / (domain + ".npy"), mmap_mode="r", allow_pickle=False)
        banks[domain] = original[start:start + 2000 * BATCH[domain]].reshape(2000, BATCH[domain])
        require(banks[domain].min() >= 0 and banks[domain].max() < len(datasets[domain]), "Visit out of TRAIN")
    original_main = custody / "HOPPER-ADAPT20K-V1/MAIN"
    pin(original_main / "checkpoints/phase03800/COMPLETE.json", MAIN)
    previous = read(original_main / "updates/phase03800.json")
    key = float(previous["result"]["learning_rate"])
    require(previous["status"] == "COMPLETE" and 0 < key < .01, "The original main endpoint LR is missing")
    warm = key * np.arange(1, 101) / 100
    rest = key * (.1 + .45 * (1 + np.cos(np.linspace(0., np.pi, 1900))))
    return datasets, tokenizer, banks, slots, auxiliary, progress["lineage_cursors"], np.concatenate((warm, rest))


def cpu_preflight(custody, short_capsule, prepared, parent):
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch

    datasets, tokenizer, banks, slots, auxiliary, cursors, rates = inputs(custody, short_capsule, prepared, parent)
    shapes = {}
    for domain, batch in banks.items():
        cursor = cursors[domain]
        host = prepare_host_batch(datasets[domain], tokenizer, domain, range(cursor, cursor+BATCH[domain]), batch[0])
        shapes[domain] = list(host["actions"].shape)
    preceding, executed = auxiliary(banks["target"][0, slots[0]])
    require(preceding.shape == (4,) and executed.shape == (4,40) and np.isfinite(executed).all(),
            "Physical auxiliary context does not match first true batch")
    return dict(logical_action_shapes=shapes, auxiliary_shape=list(executed.shape), optimizer_updates=0,
                target_windows=len(datasets["target"]), scheduled_rates=[float(rates[i]) for i in (0,99,100,1999)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("custody", "short_capsule", "prepared", "parent"):
        parser.add_argument(name, type=Path)
    print(json.dumps(cpu_preflight(**vars(parser.parse_args()))))
