"""Audit PLACE release support and actual V7-A visits using the fixed TRAIN split."""

import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_core_endpoint import ARMS, ARRAYS, paired_ledger
from v7_matched_bank import PUBLIC_INDEX
from v7_place_release_readout import OPEN_FRACTION, OPEN_RANGE_M


MATCHED = "ec5e7436b839e8452b61fc137f4fe6f659af2aabb522165c7604413b12d20024"
LATE_PHASES = ("CARRY", "LOWER", "OPEN", "WITHDRAW")


def label_rows(root, index, matched):
    threshold = OPEN_FRACTION * OPEN_RANGE_M
    phases, tasks, episodes, opens, supported = [], [], [], [], []
    start = 0
    for entry, registered in zip(index["episodes"], matched["episodes"], strict=True):
        name, n, split = (entry[key] for key in ("episode", "windows", "split"))
        require(registered == dict(episode=name, offset=start, windows=n, split=split)
                and split in ("TRAIN", "TRAIN_RECOVERY"), "Unexpected index order or non-TRAIN sample")
        bundle = (root / entry["path"]).resolve()
        require(bundle.is_relative_to(root.resolve()) and bundle.stat().st_size == entry["archive_bytes"],
                "Unexpected episode archive")
        with tarfile.open(pin(bundle, entry["archive_sha256"]), "r") as tar, tar.extractfile("labels.npz") as stream:
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as labels:
                phase, valid, prior, jaw = (labels[field] for field in
                                            ("phase", "valid", "prior_actual_width_m", "actual_width_m"))
        require(phase.shape == prior.shape == (n,) and jaw.shape == valid.shape == (n, 40)
                and valid.dtype == bool and valid[:, :10].all() and np.isfinite(prior).all()
                and np.isfinite(jaw[valid]).all() and dict(Counter(phase)) == entry["phases"], "Original label support changed")
        # Count transitions from a below-threshold prior command. Exclude maintenance of an already open jaw.
        crossing = (prior[:, None] < threshold) & valid & (jaw >= threshold)
        opens.extend(zip(crossing[:, :10].any(axis=1), crossing.any(axis=1), strict=True))
        supported.extend(valid.sum(axis=1).tolist())
        phases.extend(phase.tolist())
        tasks.extend([entry["task"]] * n)
        episodes.extend([name] * n)
        start += n
    require(start == 8031 and len(set(episodes)) == 96, "Incomplete registered A93 and Recovery3 population")
    return np.asarray(phases), np.asarray(tasks), np.asarray(episodes), np.asarray(opens), np.asarray(supported)


def p10_mask_audit(support, target, slots):
    sizes = support[target]
    require(sizes.shape == (500, 96) and np.all((sizes >= 10) & (sizes <= 40)), "Changed actual Go2 FM masks")
    unweighted = 10. / sizes
    weighted = 20. / (sizes + 10.)
    groups = np.repeat(("old", "new", "recovery"), (4450, 3466, 115))[target[250]]
    return dict(
        actual_go2_fm_support_histogram=dict(sorted(Counter(sizes.ravel().tolist()).items())),
        actual_aux_support_histogram=dict(sorted(Counter(np.take_along_axis(sizes, slots, axis=1).ravel().tolist()).items())),
        step_251_support_histogram=dict(sorted(Counter(sizes[250].tolist()).items())),
        step_251_by_pool={group: dict(sorted(Counter(sizes[250, groups == group].tolist()).items()))
                          for group in ("old", "new", "recovery")},
        step_251_original_prefix_weight_fraction=float(unweighted[250].mean()),
        step_251_candidate_prefix_weight_fraction=float(weighted[250].mean()),
        actual_prefix_weight_fraction_range=[float(unweighted.mean(axis=1).min()), float(unweighted.mean(axis=1).max())],
        candidate_prefix_weight_fraction_range=[float(weighted.mean(axis=1).min()), float(weighted.mean(axis=1).max())],
        fully_supported_steps=int((sizes == 40).all(axis=1).sum()), candidate_normalization="32 * (valid_count + 10)")


def audit(root, bank, capsule):
    index = read(pin(root / "index.json", PUBLIC_INDEX))
    matched = read(pin(bank / "MATCHED_INPUTS.json", MATCHED))
    require(matched["data_index_sha256"] == PUBLIC_INDEX and matched["bank_sha256"] == ARRAYS, "Wrong matched dataset")
    for arm in ARMS:
        run = capsule / arm
        terminal = read(run / "TERMINAL.json")
        require(terminal["formal_updates"] == 500 and terminal["updates_not_in_ledger"] == 0,
                "Requires the complete formal pair")
        pin(run / "UPDATES.jsonl", terminal["optimizer_ledger_sha256"])
    training = paired_ledger(capsule, bank)
    phase, task, episode, opened, support = label_rows(root, index, matched)
    with np.load(pin(bank / "BANK.npz", ARRAYS), allow_pickle=False) as payload:
        target, slots = payload["target"], payload["auxiliary_slots"]
    fm = np.bincount(target.ravel(), minlength=len(phase))
    aux = np.bincount(np.take_along_axis(target, slots, axis=1).ravel(), minlength=len(phase))
    require(fm.sum() == 48000 and aux.sum() == 2000 and (support[aux > 0] == 40).all(), "Actual target or aux support changed")
    groups = np.repeat(("old", "new", "recovery"), (4450, 3466, 115))
    output = defaultdict(dict)
    for role in ("old", "new", "recovery"):
        for instruction in sorted(set(task[(groups == role) & np.char.startswith(task, "PLACE-")])):
            in_task = (groups == role) & (task == instruction)
            for category in LATE_PHASES + ("ALL_PHASES",):
                pick = in_task if category == "ALL_PHASES" else in_task & (phase == category)
                first, full = opened[:, 0] & pick, opened[:, 1] & pick
                output[role].setdefault(instruction, {})[category] = dict(
                    windows=int(pick.sum()), episodes=len(set(episode[pick])), fm_visits=int(fm[pick].sum()),
                    aux_visits=int(aux[pick].sum()), open_transition_P10_windows=int(first.sum()),
                    open_transition_H40_supported_windows=int(full.sum()),
                    open_transition_P10_episodes=len(set(episode[first])), open_transition_P10_fm_visits=int(fm[first].sum()),
                    open_transition_P10_aux_visits=int(aux[first].sum()), open_transition_H40_supported_fm_visits=int(fm[full].sum()))
    return dict(public_index_sha256=PUBLIC_INDEX, matched_inputs_sha256=MATCHED, bank_sha256=ARRAYS,
                training_receipt={"formal_updates": training["formal_updates"], "matched_cuda_steps": training["matched_cuda_steps"],
                                  "visits_each": training["visits_each"]},
                open_threshold_mm=1000 * OPEN_FRACTION * OPEN_RANGE_M, target_fm_visits_per_arm=int(fm.sum()),
                target_aux_visits_per_arm=int(aux.sum()), place_by_source_and_instruction=dict(output),
                p10_mask_audit=p10_mask_audit(support, target, slots),
                uses_evaluation_windows=False, new_integrations=0, new_model_forwards=0, new_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "bank", "capsule", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    write(args.output, audit(args.root, args.bank, args.capsule))
