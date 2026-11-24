"""Bind a local first-fruit right-arm diagnostic from two real ARX X5 episodes."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from sea1_archive_readout import alignment, pose_error, pose_rows
from sea1b_panel import PLAN_SHA, recorded
from source_custody import O0, REV, SOURCE_H5, VIEWS, digest, pin, read, require, source_history, write_new
from policy_query import input_identity

ORIGINAL = "Place all the fruits into the blue bowl, then pour the fruits from the blue bowl into the large white bowl."
PARAPHRASE = "Put every fruit into the blue bowl. Then empty the blue bowl into the large white bowl."
EPISODES = (
    ("episode_0000000.hdf5", (252, 266), 139, 272, 337, (252, 271, 280, 294, 312, 337)),
    ("episode_0000001.hdf5", (263, 276), 121, 282, 310, (263, 281, 289, 297, 308, 319)),
)


def saw_threshold(width, frame, threshold, closing):
    a, b = width[frame-1:frame+1]
    return a > threshold >= b if closing else a < threshold <= b


def payload(h, frame, prompt):
    from XPolicyLab.utils.process_data import decode_image_bit
    history = source_history(frame)
    return dict(
        images=dict(cam_high=np.stack([decode_image_bit(h[f"vision/{VIEWS[0]}/colors"][i]) for i in history]),
                    **{view: decode_image_bit(h[f"vision/{view}/colors"][frame]) for view in VIEWS[1:]}),
        state=recorded(h, "state", slice(frame, frame+1))[0].astype(np.float32), prompt=prompt,
    )


def run(root, dest):
    readback = read(O0)
    source_archive = root / "evidence/O0_PANEL_01"
    historical = {(r["episode"], r["frame"]): r for r in readback["source"] if r["mode"] == "primary"}
    rows, archives = [], []
    for episode, frames, bowl, close, release, visible in EPISODES:
        path = pin(root / "data" / episode, SOURCE_H5[episode])
        with h5py.File(path) as h:
            require(int(h["additional_info/frequency"][()]) == 30
                    and h["instruction"].asstr()[()] == ORIGINAL, "recording language or clock changed")
            widths = h["state/right_ee_joint_states"][:, 0]
            require(saw_threshold(widths, bowl, .55, True) and saw_threshold(widths, close, .55, True)
                    and saw_threshold(widths, release, .65, False), "preselected grasp events changed")
            knots = recorded(h, "state", slice(close, release+1))
            raised = float(1000 * np.max(knots[:, 10]-knots[0, 10]))
            require(raised > 0 and len(knots) == release-close+1, "right link6 source Z does not increase")
            archives.append(dict(episode=episode, source_sha256=SOURCE_H5[episode], bowl_rim_close_frame=bowl,
                                 fruit_close_frame=close, fruit_release_frame=release, reviewed_rgb_frames=list(visible),
                                 closed_interval_s=(release-close)/30., right_link6_max_z_rise_mm=raised,
                                 localized_visual_evidence="right hand surrounds and transports first fruit; left hand visible away from the fruit"))
            for frame in frames:
                obs = payload(h, frame, ORIGINAL)
                expert = recorded(h, "state", slice(frame+1, frame+51))
                original = recorded(h, "action", slice(frame, frame+50))
                require(expert.shape == (50,16) and np.array_equal(original, expert)
                        and frame < close <= frame+50, "invalid causal source window")
                measured, labels = pose_rows(obs["state"][None, 8:]), pose_rows(expert[:, 8:])
                baseline = np.repeat(measured, 50, axis=0)
                left = recorded(h, "state", slice(frame, frame+51))[:, :8]
                row = dict(
                    anchor=f"FRUIT{len(rows):02d}", episode=episode, frame=frame, history=source_history(frame),
                    source_sha256=SOURCE_H5[episode], right_close_successor_offset=close-frame,
                    expert_right_link6_peak_from_query_mm=1000*float(np.linalg.norm(labels[:, :3, 3]-measured[0, :3, 3], axis=1).max()),
                    left_link6_peak_from_query_mm=1000*float(np.linalg.norm(left[:,:3]-left[0,:3], axis=1).max()),
                    left_grip_fraction_range=float(np.ptp(left[:,7])), hold_xyz_rms_mm=pose_error(baseline, labels)["xyz_rms_mm"],
                    t0_input_sha256=input_identity(obs)[0],
                    t1_input_sha256=input_identity({**obs, "prompt":PARAPHRASE})[0],
                    historical_forward=None,
                )
                if (episode, frame) in historical:
                    o0 = historical[episode, frame]
                    archive = pin(source_archive / f"call_{o0['call']:02d}_actions.npz", o0["output_sha256"])
                    with np.load(archive, allow_pickle=False) as z:
                        prediction = pose_rows(z["native_absolute_arms"][:,8:])
                        require(np.array_equal(z["expert_absolute_arms"], expert), "different historical source labels")
                    row["historical_forward"] = dict(
                        call=o0["call"], output_sha256=o0["output_sha256"],
                        xyz_rms_mm=pose_error(prediction, labels)["xyz_rms_mm"],
                        direction_cos=alignment(prediction[-1,:3,3]-measured[0,:3,3], labels[-1,:3,3]-measured[0,:3,3]),
                        formal_matched_noise_reuse=False,
                    )
                rows.append(row)
    cells = [dict(anchor=x["anchor"], seed=[0,seed], condition=c,
                  input_sha256=x[f"{c.lower()}_input_sha256"])
             for seed in range(2) for x in rows for c in ("T0", "T1")]
    write_new(dest, dict(
        status="SOURCE_DIAGNOSTIC_ROSTER_BOUND_NO_SCIENCE_ALLOCATION", registry_sha256=PLAN_SHA,
        task="pack_and_pour_fruit", dataset_revision=REV, arm="right",
        scope="first local fruit pickup and lift only, bimanual full task including pouring excluded",
        observations=archives, anchors=rows, prompts={"T0": ORIGINAL, "T1": PARAPHRASE},
        t0_t1_cells=cells, t0_t1_forwards_cap=16, exact_reuse=0,
        t2_status="BLOCKED_NO_MATCHING_SEALED_GO2_OBSERVATION",
        t3_status="HISTORICAL_REACH_ONLY", allocation=None, model_forwards=0, physics_ticks=0, training_updates=0,
        historical_identity=digest(O0),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU source data only")
    run(args.root.resolve(), args.output.resolve())
