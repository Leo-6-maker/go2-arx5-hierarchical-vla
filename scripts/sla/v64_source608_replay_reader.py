"""Read the sealed Go2 and LIBERO lanes of one SOURCE608 host batch."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from v64_source608_public_bank import digest, read_pinned, write
from v64_source608_reader import JOINT_RECEIPT, TOKENIZER


COMMANDS = "ea8b99355b0a8fc2367d17b5e2f18e8a4d583c73ddb39b3848b1add25e143f96"
CORRECTED = "1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21"
AUXILIARY = "584fe14655538cbe22509cd27f5fc432026808d5c577c84b6ab9ed81d2e7838b"
LIBERO = "ba86ee2f88803fff79bda950ee54add91dcefaa105575ac22dade5d650b3fa8a"
REFERENCE = "d987be63fdf39ded1825fa37703511fd9449589c9b20d84814ac85d5b2854584"
PUBLIC_BATCH = "837d5d379c71c411ca3a4828770b91fc15221f645499171cd67855af7ca2697d"


def target_reader(package, archive, corrected):
    from v3_main_place_learn import data_scope
    from v6_reanchored_teacher_labels import bind_dataset

    model = package / "reference"
    checkpoint = read_pinned(model / "CHECKPOINT.json", REFERENCE)
    stats = {field: read_pinned(model / f"{file.upper()}.json", checkpoint[f"{file}_sha256"])[field]
             for file, field in (("state", "state"), ("action", "actions"))}
    commands = package / "V3-MAIN-CONSUMED-DATA-02"
    original = commands / "MANIFEST.json"
    read_pinned(original, COMMANDS)
    reader = data_scope(archive, commands, stats)
    bind_dataset(reader, corrected, CORRECTED, original)
    return reader


def auxiliary_for(reader, source, selected):
    manifest = read_pinned(source / "MANIFEST.json", AUXILIARY)
    evidence = {row["episode"]: row for row in manifest["episodes"]}
    values = []
    for i in selected:
        episode, anchor = reader.rows[int(i)]
        row = evidence[episode["id"]]
        path = source / row["file"]
        if digest(path) != row["sha256"]:
            raise ValueError("Actual native width bytes changed")
        with np.load(path, allow_pickle=False) as data:
            support, preceding, actual = (data[key][anchor] for key in
                                          ("valid", "prior_actual_width_m", "actual_width_m"))
        if not (support.shape == actual.shape == (40,) and support.all()
                and np.array_equal(support, reader.support(int(i)))
                and np.isfinite(preceding) and np.isfinite(actual).all()):
            raise ValueError("Expected complete actual prior and future jaw-width evidence")
        values.append((preceding, actual))
    return {name: hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()
            for name, values in zip(("preceding_width", "actual_width_H40"), zip(*values, strict=True), strict=True)}


def run(package, archive, corrected, library_root, tokenizer_path, joint, public_batch, output):
    from go2arx5_vla.sla_adapter.libero_scale20k import LiberoDataset
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch
    from openpi_client.image_tools import resize_with_pad
    from v61_source_bank import source_identity, target_identity

    started = time.perf_counter()
    sealed = read_pinned(joint / "ACTUAL.json", JOINT_RECEIPT)
    public = read_pinned(public_batch, PUBLIC_BATCH)
    if (digest(joint / "BANK.npz") != sealed["joint_bank_sha256"]
            or public["joint_bank_sha256"] != sealed["joint_bank_sha256"]
            or public["actual_public_windows"] != 96 or digest(tokenizer_path) != TOKENIZER):
        raise ValueError("The replay and public rows must use the exact same candidate")
    target = target_reader(package, archive, corrected)
    manifest = package / "sla-scale20k-vanda-libero-train-assets-20261120-01" / "MANIFEST.private.json"
    library = LiberoDataset(library_root, {"path": str(manifest), "sha256": LIBERO},
                           split="TRAIN", resize_with_pad=resize_with_pad)
    identities = dict(target=target_identity(target), libero=source_identity(library))
    if identities != sealed["replay_reader_order_sha256"]:
        raise ValueError("A live replay reader does not preserve the sealed SOURCE order")
    tokenizer = Tokenizer(str(tokenizer_path))
    row = public["schedule_row"]
    results = {}
    with np.load(joint / "BANK.npz", allow_pickle=False) as bank:
        for role, reader, horizon, views in (
            ("target", target, 40, (True, True, False)), ("libero", library, 10, (True, True, False)),
        ):
            indices = bank[role][row]
            host = prepare_host_batch(reader, tokenizer, role, range(row * 16, row * 16 + 16), indices)
            actions, obs = host["actions"], host["observation"]
            keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            if (actions.shape != (16, horizon, 32) or not np.isfinite(actions).all()
                    or any(not np.all(obs["image_mask"][view] == active) or
                           not np.all(np.asarray(obs["image"][view]).any(axis=(1, 2, 3)) == active)
                           for view, active in zip(keys, views, strict=True))):
                raise ValueError("The real replay tensors differ from their native clock or camera mask")
            results[role] = dict(shape=list(actions.shape), indices=indices.tolist(),
                                 actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                                 images_sha256={view: hashlib.sha256(obs["image"][view].tobytes()).hexdigest()
                                                for view in keys})
        slots = bank["auxiliary_slots"][row]
        selected = bank["target"][row][slots]
        if len(np.unique(selected)) != 4:
            raise ValueError("Four independent H40 auxiliary source windows are required")
        auxiliary = auxiliary_for(target, package / "native_aux-01", selected)
    write(output, dict(kind="SOURCE608_CORELAB_ACTUAL_FIVE_DOMAIN_HOST_INPUT", schedule_row=row,
                      joint_bank_sha256=sealed["joint_bank_sha256"], public_batch_sha256=PUBLIC_BATCH,
                      real_host_windows=96 + sum(len(role["indices"]) for role in results.values()),
                      roles=results, actual_H40_auxiliary_windows=len(selected), auxiliary_sha256=auxiliary,
                      reader_order_sha256=identities, new_optimizer_visits=0, new_model_forwards=0,
                      wall_seconds=time.perf_counter() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("package", "archive", "corrected", "library_root", "tokenizer_path",
                "joint", "public_batch", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
