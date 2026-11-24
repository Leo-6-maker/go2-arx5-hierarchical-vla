"""Recover missing pre-query camera history from a sealed Teacher journal on CPU."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np


def replay(case, folder, *, split="TRAIN"):
    import mujoco
    from demonstrated_grip import pin, read, require
    from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from target_labels import checksum, history

    require(
        os.environ["CUDA_VISIBLE_DEVICES"] == ""
        and os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1",
        "archived software render only",
    )
    spec = read(case / "EXECUTION_SPEC.json")
    native = case / "native-01"
    result = read(case / "RESULT.private.json")
    capture = result["detail"]["capture"]
    manifest = read(pin(Path(capture["path"]), capture["sha256"]))
    qsource = manifest["labels"]
    queries = read(pin(Path(qsource["path"]), qsource["sha256"]))
    ticks = [row["native_step"] for row in queries]
    scene = pin(Path(spec["model_path"]), manifest["frozen_scene_sha256"])
    require(
        split in ("TRAIN", "VAL")
        and spec["collection_split"] == split
        and result["detail"]["task_success"],
        "wrong replay cohort",
    )
    missing = {t for tick in ticks for t in history(tick)} - set(ticks)
    # All new observation times precede the first actual query. Verify the unmodified
    # camera pipeline at both ends against both archived sensors and the physical TCP.
    require(missing and max(missing) < ticks[0], "unexpected internal missing history")
    anchors = {ticks[0]: queries[0], ticks[-1]: queries[-1]}
    expected = {}
    shards = [
        manifest["frame_shards"][i]
        for i in sorted({0, len(manifest["frame_shards"]) - 1})
    ]
    for shard in shards:
        with np.load(
            pin(Path(shard["path"]), shard["sha256"]), allow_pickle=False
        ) as z:
            for t, im in zip(z["native_step"], z["rgb"]):
                if int(t) in anchors:
                    expected[int(t)] = im.copy()
    require(expected.keys() == anchors.keys(), "archived anchor missing")
    initfile = native / "initialization.jsonl"
    journal = native / "state.bin.journal"
    with initfile.open() as f:
        init = json.loads(next(f))
    require(init["time"] == 0 and init["kind"] == "actual_startup_V1", "unbound reset")
    first = {k: np.asarray(init[k]) for k in ("qpos", "qvel", "ctrl")}
    first.update(time_s=0.0, step=0)
    wanted = missing | anchors.keys()
    ticks_out, images, rows = [], [], []
    prev = None
    writer = TwoObjectBodyRenderer(scene)
    try:
        with forbid_integration(mujoco) as attempts:
            for a in itertools.chain([first], packets(journal)):
                t = a["step"]
                if t == prev == 0:
                    require(
                        all(
                            np.array_equal(a[k], first[k])
                            for k in ("qpos", "qvel", "ctrl")
                        ),
                        "reset disagrees",
                    )
                    continue
                require(prev is None or t == prev + 1, "journal clock gap")
                prev = t
                if t not in wanted:
                    continue
                require(abs(a["time_s"] - t * 0.002) < 1e-8, "replay state clock")
                body, _, _, _ = writer.render(a, audit_masks=False)
                writer.renderer.update_scene(writer.data, camera="wrist_rgb")
                pair = np.stack([body, writer.renderer.render().copy()])
                if t in anchors:
                    require(
                        np.array_equal(pair, expected[t]),
                        "stereo replay anchor differs",
                    )
                    tip = writer.data.site("x5_ee")
                    tcp = np.asarray(anchors[t]["actual_state"]["tcp"])
                    require(
                        np.allclose(tip.xpos, tcp[:3, 3], rtol=0, atol=2e-10)
                        and np.allclose(
                            tip.xmat.reshape(3, 3), tcp[:3, :3], rtol=0, atol=2e-10
                        ),
                        "TCP anchor",
                    )
                images.append(pair)
                ticks_out.append(t)
                rows.append(
                    {
                        "tick": t,
                        "rgb_sha256": [
                            hashlib.sha256(im.tobytes()).hexdigest() for im in pair
                        ],
                    }
                )
                if t == ticks[-1]:
                    break
            require(
                not attempts and set(ticks_out) == wanted,
                "incomplete sensor-only replay",
            )
    finally:
        writer.close()
    folder.mkdir()
    rgbfile = folder / "history.npz"
    np.savez_compressed(rgbfile, tick=ticks_out, rgb=np.stack(images))
    ledger = folder / "frames.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {
        "id": case.name,
        "shards": [
            {"name": rgbfile.name, "count": len(rows), "sha256": checksum(rgbfile)}
        ],
        "frames_sha256": checksum(ledger),
        "missing_count": len(missing),
        "anchor_ticks": sorted(anchors),
        "scene_sha256": manifest["frozen_scene_sha256"],
        "integration_ticks": 0,
        "source_files": {
            "LIVE_DATA/QUERIES.private.json": qsource["sha256"],
            "initialization.jsonl": checksum(initfile),
            "state.bin.journal": checksum(journal),
            "EXECUTION_SPEC.json": checksum(case / "EXECUTION_SPEC.json"),
        },
    }
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--split", choices=("TRAIN", "VAL"), default="TRAIN")
    args = parser.parse_args()
    replay(args.case, args.folder, split=args.split)
