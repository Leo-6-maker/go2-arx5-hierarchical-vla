"""Seal one parent-independent, matched V7-A sample bank from the proven V6.5 inputs."""

import argparse
from collections import Counter
import io
from pathlib import Path
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_bank import index_identity
from v65_recovery_course import AUX, BATCH, PARTS, PEAK
from v7_previous_command import PRECEDING_INTERVAL_TICKS, encode_previous_width


PLAN = "08153842af4bfa778b0a43094e300de430c09c65"
V65_BANK = "b8a29eea650acc2eb9f1ab8802b95c0d0a4ab88459c1e1fd7d54f2740ee9cb66"
PUBLIC_INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
CONTEXT_RECEIPTS = {
    "legacy": "163d0958842be585fb1b64854652234c36b8edd7efbdc498f4a7b6bc8bf5228a",
    "ef5": "04fcdfa6110eb89bedd249f53c5a0fdb77144cac2c948b08c3999808b5e99824",
    "recovery": "8fabd1e61a6d724ac71f107a0372d002b622b95fa6998b6eb84758e487a48008",
}
STEPS, WARMUP, MODEL_NOISE_SEED = 500, 50, 529_202611
NUM_LABELS = 8031


def rates():
    step = np.arange(1, STEPS + 1, dtype=np.float64)
    cosine = .1 + .45 * (1 + np.cos(np.pi * (step - WARMUP) / (STEPS - WARMUP)))
    return PEAK * np.where(step <= WARMUP, step / WARMUP, cosine)


def training_context(archive, reader_sha, proofs):
    index = read(pin(archive / "index.json", PUBLIC_INDEX))
    roster = index["episodes"]
    require(index_identity((ep["episode"], anchor) for ep in roster for anchor in range(ep["windows"]))
            == reader_sha, "Public episode order must agree with the exact training reader")
    old = {row["episode"]: row["windows"] for row in proofs["legacy"]["episodes"]}
    new = {row["episode"]: row["windows"] for row in proofs["ef5"]["confirmed"]}
    recovery = {row["id"]: row["rows"] for row in proofs["recovery"]["episodes"]}
    require(len(old) == 69 and len(new) == 24 and len(recovery) == 3
            and not (old.keys() & new.keys()), "Incomplete directly verified context population")
    metadata, ticks, widths, full = [], [], [], []
    offset = 0
    for row in roster:
        name, count = row["episode"], row["windows"]
        require(row["split"] in ("TRAIN", "TRAIN_RECOVERY"), "Only the fixed training splits contribute context")
        if row["split"] == "TRAIN":
            require((old | new)[name] == count, "TRAIN context does not match the admitted episode")
        path = archive / row["path"]
        require(path.resolve().is_relative_to(archive.resolve()) and path.stat().st_size == row["archive_bytes"],
                "Wrong portable episode payload")
        with tarfile.open(path, "r") as bundle, bundle.extractfile("labels.npz") as member:
            with np.load(io.BytesIO(member.read()), allow_pickle=False) as labels:
                query, previous, valid = (labels[key] for key in ("native_step", "prior_actual_width_m", "valid"))
        require(query.shape == previous.shape == (count,) and valid.shape == (count, 40)
                and np.issubdtype(query.dtype, np.integer) and valid.dtype == bool
                and valid[:, :10].all() and (np.diff(query) == 250).all(), "Invalid admitted query clock or support")
        if row["split"] == "TRAIN_RECOVERY":
            context = recovery[name]
            require(np.array_equal(query, [r["query_tick"] for r in context])
                    and np.array_equal(previous, [r["previous_consumed_width_m"] for r in context])
                    and all(r["valid"] for r in context), "Recovery context no longer matches actual metadata")
        metadata.append(dict(episode=name, offset=offset, windows=count, split=row["split"]))
        offset += count
        ticks.extend(query.tolist())
        widths.extend(previous.tolist())
        full.extend(valid.all(axis=1).tolist())
    require(offset == NUM_LABELS, "A93 and Recovery3 must retain every bound training window")
    query = np.asarray(ticks, dtype=np.int64)
    previous = np.asarray(widths, dtype=np.float64)
    source = query - PRECEDING_INTERVAL_TICKS
    valid = np.ones(offset, dtype=bool)
    encode_previous_width(previous, query, source, valid, valid, treatment=True)
    return metadata, dict(width_m=previous, query_tick=query, consumed_start_tick=source, command_valid=valid,
                          full_h40=np.asarray(full, dtype=bool))


def prepare_go2_batch(dataset, tokenizer, cursors, dataset_indices, command_rows, *, treatment, training=True):
    """Prepare the original Go2 tensors and a separate arm-specific conditioning column."""
    from go2arx5_vla.sla_adapter import batch_v2
    from go2arx5_vla.sla_adapter.scale20k_training import DOMAINS, prepare_host_batch

    cursors = tuple(cursors)
    batch = prepare_host_batch(dataset, tokenizer, "target", cursors, dataset_indices, training=training)
    cursor_offset = 1_000_000 * DOMAINS.index("target")
    kept = np.asarray([not training or not batch_v2.drop_state(int(cursor) + cursor_offset) for cursor in cursors])
    keys = ("width_m", "query_tick", "consumed_start_tick", "command_valid")
    field = encode_previous_width(*(command_rows[key] for key in keys), kept, treatment=treatment)
    return batch, field


def seal(source_bank, archive, legacy, ef5, recovery, output):
    original = read(pin(source_bank / "ACTUAL.json", V65_BANK))
    proofs = {key: read(pin(path, CONTEXT_RECEIPTS[key]))
              for key, path in (("legacy", legacy), ("ef5", ef5), ("recovery", recovery))}
    metadata, context = training_context(archive, original["reader_order"]["target"], proofs)
    with np.load(pin(source_bank / "BANK.npz", original["bank_sha256"]), allow_pickle=False) as inherited:
        banks = {role: inherited[role][:STEPS].copy() for role in BATCH}
        auxiliary_slots = inherited["auxiliary_slots"][:STEPS].copy()
    require(all(bank.shape == (STEPS, BATCH[role]) and bank.dtype == np.int64 and (bank >= 0).all()
                for role, bank in banks.items()), "The fixed source visit schedule changed")
    target = banks["target"]
    groups = np.repeat(("old", "new", "recovery"), (4450, 3466, 115))
    require(target.max() < NUM_LABELS and np.all(np.diff(np.sort(target, axis=1), axis=1) > 0)
            and all(np.all(np.sum(groups[target] == key, axis=1) == width) for key, width in PARTS.items()),
            "Go2 batch composition has changed")
    require(auxiliary_slots.shape == (STEPS, 4) and (auxiliary_slots >= 0).all()
            and (auxiliary_slots < BATCH["target"]).all(), "Invalid auxiliary slots")
    selected = np.take_along_axis(target, auxiliary_slots, axis=1)
    require((groups[selected] == ["old", "old", "new", "new"]).all()
            and np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)
            and context["full_h40"][selected].all(), "Require four independently supported H40 slots")
    output.mkdir(mode=0o700)
    bank = output / "BANK.npz"
    fields = output / "PREVIOUS_COMMAND.npz"
    with bank.open("xb") as stream:
        np.savez_compressed(stream, **banks, auxiliary_slots=auxiliary_slots, rates=rates())
    with fields.open("xb") as stream:
        np.savez_compressed(stream, **context)
    steps = rates()
    write(output / "MATCHED_INPUTS.json", dict(
        plan=PLAN, source_v65_bank=V65_BANK, v65_original_schedule_retained=[1, STEPS],
        data_index_sha256=PUBLIC_INDEX, context_proofs=CONTEXT_RECEIPTS, reader_order=original["reader_order"],
        bank_sha256=sha(bank), previous_command_sha256=sha(fields), episodes=metadata, context_windows=NUM_LABELS,
        arms={"CONTROL_NEUTRAL": False, "TREATMENT_PREVIOUS_EXECUTED_GRIPPER_COMMAND": True},
        optimizer_initialization="FRESH_PER_ARM", candidate_model_noise_seed=MODEL_NOISE_SEED,
        v65_noise_reused=False, steps=STEPS, warmup_steps=WARMUP, first_lr=steps[0], peak_lr=steps[WARMUP - 1],
        final_lr=steps[-1], aux_coefficient=AUX, parent_selected=False,
        fm_visits_per_arm={role: STEPS * count for role, count in BATCH.items()},
        go2_visits_per_arm=dict(Counter(groups[target].ravel())), aux_visits_per_arm=STEPS * 4,
        optimizer_updates=0, model_forwards=0, native_integrations=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for member in ("source_bank", "archive", "legacy", "ef5", "recovery", "output"):
        cli.add_argument(member, type=Path)
    seal(**vars(cli.parse_args()))
