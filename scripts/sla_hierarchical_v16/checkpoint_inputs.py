"""Inspect the pinned input transforms on real observations, without invoking the model."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import h5py
import numpy as np
from demonstrated_grip import pin, read, require


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def run(root, visibility):
    from XPolicyLab.policy.SLA_Memory.model import Model, _policy_config
    from XPolicyLab.utils.process_data import decode_image_bit
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.checkpoints import load_norm_stats

    repo = Path(__file__).resolve().parents[2]
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    e = root / "evidence"
    manifest = read(e / "SOURCE_AND_WEIGHT_MANIFEST.actual.json")
    vendor = root / "vendor/XPolicyLab"
    for r in manifest["source_files"]:
        pin(vendor / r["path"], r["sha256"])
    pin(Path(manifest["tokenizer"]["path"]), manifest["tokenizer"]["sha256"])
    # Stop the adapter at its resolved loader boundary. No parameters or policy methods are called.
    with patch.object(
        _policy_config,
        "create_trained_policy",
        side_effect=lambda config, path: (config, path),
    ):
        cfg, ckpt = Model(
            read(
                pin(e / "P0_EFFECTIVE_CONFIG.json", manifest["effective_config_sha256"])
            )
        )._policy
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    statsfile = ckpt / "assets" / data.asset_id / "norm_stats.json"
    pin(statsfile, "cf6ddf4f89180e762e5e6210881a763f8e29d3499fb1cc69a440de7aa585d276")
    stats = load_norm_stats(ckpt / "assets", data.asset_id)
    normalize = transforms.Normalize(stats, use_quantiles=data.use_quantile_norm)
    prepare = transforms.compose(
        [transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs]
    )
    tokenizer = next(
        t
        for t in data.model_transforms.inputs
        if isinstance(t, transforms.TokenizePrompt)
    )
    untruncated = dataclasses.replace(tokenizer, tokenizer=PaligemmaTokenizer(4096))
    low, high = np.array(stats["state"].q01), np.array(stats["state"].q99)
    require(
        low.shape == high.shape == (14,) and data.use_quantile_norm,
        "wrong state representation",
    )

    def inspect(obs):
        transformed = prepare(obs)
        state = transformed["state"].copy()
        norm = normalize(transformed)
        normalized = norm["state"].copy()
        reference = untruncated(dict(norm))
        final = transforms.compose(data.model_transforms.inputs)(dict(norm))
        short = final["tokenized_prompt"][final["tokenized_prompt_mask"]]
        long = reference["tokenized_prompt"][reference["tokenized_prompt_mask"]]
        require(
            np.array_equal(short, long[: len(short)])
            and len(short) == min(len(long), cfg.model.max_token_len),
            "wrong tokens",
        )
        require(
            np.array_equal(final["state"][:14], normalized)
            and all(final["image_mask"].values())
            and all(im.shape == (224, 224, 3) for im in final["image"].values())
            and final["extra_images"].shape == (9, 224, 224, 3),
            "wrong network input",
        )
        return dict(
            state=state.tolist(),
            normalized=normalized.tolist(),
            right_low=(state[7:] < low[7:]).tolist(),
            right_high=(state[7:] > high[7:]).tolist(),
            tokens_used=len(short),
            tokens_full=len(long),
            prompt_truncated=len(short) != len(long),
            prompt=obs["prompt"],
            tokens_sha256=digest(short),
            network_image_sha256={k: digest(im) for k, im in final["image"].items()},
            history_images_sha256=[digest(im) for im in final["extra_images"]],
        )

    capture_file = root / "v12/A2_O2_DENSE_CAPTURE.json"
    capture = read(pin(capture_file, pins[str(capture_file)]))["windows"]
    target = []
    for r in read(visibility)["target"]:
        require(
            capture[r["id"]]["input_sha256"] == r["input_sha256"],
            "input registration drift",
        )
        f = pin(root / "v12/STUDENT_INPUT" / (r["id"] + ".npz"), r["input_sha256"])
        with np.load(f, allow_pickle=False) as z:
            obs = dict(
                state=z["state"],
                prompt=str(z["prompt"]),
                images=dict(
                    cam_high=z["head"],
                    cam_left_wrist=np.zeros((480, 640, 3), np.uint8),
                    cam_right_wrist=z["right_wrist"],
                ),
            )
            require(
                np.array_equal(obs["state"], r["sensor_state"])
                and obs["prompt"] == r["prompt"],
                "stream identity",
            )
        target.append(dict(id=r["id"], episode=r["episode"], **inspect(obs)))
    o0 = read(repo / "reports/SLA_OFFICIAL_GO2_TRANSFER/O0_ACTUAL.json")
    demo_files = read(e / "P0_SOURCE_FILES.actual.json")["files"]
    source = []
    tracks = []
    for entry in demo_files:
        if not entry["path"].endswith(".hdf5"):
            continue
        name = Path(entry["path"]).name
        with h5py.File(
            pin(root / "data" / name, entry["published_lfs_sha256"]), "r"
        ) as z:
            measured = np.concatenate(
                [
                    z["state/" + k][:]
                    for k in (
                        "left_ee_poses",
                        "left_ee_joint_states",
                        "right_ee_poses",
                        "right_ee_joint_states",
                    )
                ],
                axis=1,
            )
            # The state-only diagnostic is insensitive to image contents. Run the same
            # official pose/rotation transforms on every measured frame.
            dummy = np.zeros((4, 4, 3), np.uint8)
            coords = prepare(
                dict(
                    images={
                        k: dummy
                        for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                    },
                    state=measured.astype(np.float32),
                    prompt="source",
                )
            )["state"][:, 7:]
            tracks.append(
                dict(
                    episode=name,
                    frames=len(coords),
                    minimum=coords.min(0).tolist(),
                    maximum=coords.max(0).tolist(),
                    checkpoint_low_count=(coords < low[7:]).sum(0).tolist(),
                    checkpoint_high_count=(coords > high[7:]).sum(0).tolist(),
                )
            )
            for row in o0["source"]:
                if row["episode"] != name or row["mode"] != "primary":
                    continue
                folder = e / "O0_PANEL_01"
                f = folder / f"call_{row['call']:02d}_actions.npz"
                with np.load(pin(f, row["output_sha256"]), allow_pickle=False) as a:
                    orig_state = a["state"]
                bind = read(folder / f"call_{row['call']:02d}_input.json")
                fr = row["frame"]
                require(
                    np.array_equal(measured[fr].astype(np.float32), orig_state)
                    and bind["frame"] == fr
                    and bind["episode"] == name,
                    "source state identity",
                )

                def rgb(view, j):
                    return decode_image_bit(z["vision/" + view + "/colors"][j])

                head = np.stack([rgb("cam_head", j) for j in bind["head_frames"]])
                imgs = dict(
                    cam_high=head,
                    cam_left_wrist=rgb("cam_left_wrist", fr),
                    cam_right_wrist=rgb("cam_right_wrist", fr),
                )
                require(
                    [digest(im) for im in head] == bind["head_sha256"]
                    and all(
                        digest(imgs[k]) == bind["current_rgb_sha256"][k]
                        for k in ("cam_left_wrist", "cam_right_wrist")
                    ),
                    "source pixel identity",
                )
                item = inspect(
                    dict(
                        state=orig_state.astype(np.float32),
                        prompt=bind["prompt"],
                        images=imgs,
                    )
                )
                require(
                    np.allclose(item["state"][7:], coords[fr], rtol=0, atol=2e-6),
                    "batch source transform",
                )
                source.append(dict(episode=name, frame=fr, **item))
    require(
        len(source) == 6 and len(target) == 16 and len(tracks) == 2, "wrong population"
    )
    return dict(
        target=target,
        source=source,
        full_source_tracks=tracks,
        state_fields=[
            "x",
            "y",
            "z",
            "rotation_vector_x",
            "rotation_vector_y",
            "rotation_vector_z",
            "gripper",
        ],
        checkpoint_right_q01=low[7:].tolist(),
        checkpoint_right_q99=high[7:].tolist(),
        model_input_steps=[
            type(t).__name__
            for t in [
                *data.data_transforms.inputs,
                normalize,
                *data.model_transforms.inputs,
            ]
        ],
        discrete_state_input=tokenizer.discrete_state_input,
        token_capacity=cfg.model.max_token_len,
        new_forwards=0,
        native_integrations=0,
        training_updates=0,
    )


if __name__ == "__main__":
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("JAX_PLATFORMS") == "cpu",
        "CPU preprocessing only",
    )
    result = run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
    with Path(sys.argv[3]).open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
