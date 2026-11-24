"""Score registered V4 outputs against the measured TCP and red cube at the capture tick."""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sealed_panel import configure_site


def world_frame(x, r):
    frame = np.eye(4)
    frame[:3, 3] = x
    frame[:3, :3] = np.reshape(r, (3, 3))
    return frame


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, repo, score_file, output):
    import mujoco
    from compiler import SourceCurve
    from demonstrated_grip import pin, read, require
    from feedback_policy_panel import arm_diff, inputs, write_new
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from policy_query import tensor_sha
    from scipy.spatial.transform import Rotation
    from verify_grip_card import states

    score = read(score_file)
    population = inputs(root)
    result = {}
    frames = {}
    source = root / "v16/V4-FEEDBACK-01"
    original = root / "v16/sla-v16-student-gripper-native-01b/V2-GRIP-01B"
    native = original / read(original / "RESULT.private.json")["episode"]
    spec = read(original / "EXECUTION_SPEC.json")
    observed = read(repo / "reports/SLA_HIERARCHICAL_V16/OWNED_FEEDBACK_ACTUAL.json")
    scene = pin(Path(spec["model_path"]), observed["scene_sha256"])
    regfile = root / "v12/TARGET_REGISTRATION.json"
    hashes = read(root / "v13/run01/S0_ACTUAL.json")["inputs"]
    calib = read(pin(regfile, hashes[str(regfile)]))
    tool = np.asarray(calib["T_link6_TCP"])
    local = np.eye(4)
    local[:3, 3] = calib["source_origin"]
    model = mujoco.MjModel.from_binary_path(str(scene))
    data = mujoco.MjData(model)
    wanted = {info["tick"] for _, info in population.values()}
    base = model.body("x5_base_link").id
    tip = model.site("x5_ee").id
    cube = model.geom("task_cube_geom").id
    finger_q = [
        model.joint("task_" + k + "_slide").qposadr[0] for k in ("left", "right")
    ]
    require(len(wanted) == 4, "wrong native panel")
    with forbid_integration(mujoco) as steps:
        for packet in states(native):
            tick = packet["step"]
            if tick not in wanted:
                continue
            data.qpos[:] = packet["qpos"]
            data.qvel[:] = packet["qvel"]
            data.ctrl[:] = packet["ctrl"]
            data.time = packet["time_s"]
            mujoco.mj_forward(model, data)
            frames[tick] = (
                world_frame(data.xpos[base], data.xmat[base]) @ local,
                world_frame(data.site_xpos[tip], data.site_xmat[tip]),
                data.geom_xpos[cube].copy(),
                data.qpos[finger_q].mean() / 0.044,
            )
            if len(frames) == len(wanted):
                break
        require(not steps and frames.keys() == wanted, "missing measured states")
    arrays = {}
    for key, (obs, info) in population.items():
        r = score["calls"][key]
        require(r["input_sha256"] == info["input_sha256"], "input identity")
        origin, actual, center, fraction = frames[info["tick"]]
        measured = np.linalg.inv(origin) @ actual @ np.linalg.inv(tool)
        state = obs["state"]
        require(
            np.allclose(measured[:3, 3], state[8:11], atol=2e-7, rtol=0)
            and np.allclose(
                Rotation.from_quat(state[[12, 13, 14, 11]]).as_matrix(),
                measured[:3, :3],
                atol=2e-7,
                rtol=0,
            ),
            "query transform disagrees with model input",
        )
        if info["mode"] == "observed":
            require(abs(fraction - state[15]) < 1e-7, "different measured width")
        with np.load(
            pin(source / (key + ".npz"), r["file_sha256"]), allow_pickle=False
        ) as z:
            absolute = z["absolute"]
            raw = z["actions"]
            raw_dtype = str(z["actions_dtype"])
            arrays[key] = (raw, absolute)
            sampler_state_sha = tensor_sha(z["sampler_state"])
        path = SourceCurve(absolute, origin, tool).knots[:, :3, 3]
        distance = 1000 * np.linalg.norm(path - center, axis=1)
        result[key] = {
            "tick": info["tick"],
            "mode": info["mode"],
            "field_sha256": info["field_sha256"],
            "archive_sha256": info["archive_sha256"],
            "sampler_state_storage_sha256": sampler_state_sha,
            "raw_dtype": raw_dtype,
            "raw_storage_sha256": tensor_sha(raw),
            "absolute_sha256": tensor_sha(absolute),
            "measured_TCP_to_red_mm": float(
                1000 * np.linalg.norm(actual[:3, 3] - center)
            ),
            "first_to_red_mm": float(distance[0]),
            "last_to_red_mm": float(distance[-1]),
            "minimum_to_red_mm": float(distance.min()),
            "first_to_last_gain_mm": float(distance[0] - distance[-1]),
            "first_to_last_z_mm": float(1000 * (path[-1, 2] - path[0, 2])),
            "actual_to_last_gain_mm": float(
                1000 * np.linalg.norm(actual[:3, 3] - center) - distance[-1]
            ),
            "source_width_min_mm": float(2 + 40 * absolute[:, 15].min()),
            "source_width_max_mm": float(2 + 40 * absolute[:, 15].max()),
        }
    pairs = []
    knots = []
    for pair in score["pairs"]:
        a, b = pair["first"], pair["second"]
        ar, aa = arrays[a]
        br, ba = arrays[b]
        dr = ar.astype(float) - br.astype(float)
        left, right = arm_diff(aa[:, :8], ba[:, :8]), arm_diff(aa[:, 8:], ba[:, 8:])
        require(all(right[k] == pair[k] for k in right), "panel score disagreement")
        pairs.append(
            {
                "first": a,
                "second": b,
                "kind": pair["kind"],
                "left": left,
                "right": right,
                "sampler_active14_rms_by_channel": np.sqrt(
                    np.mean(dr[:, :14] ** 2, axis=0)
                ).tolist(),
                "changed_active14_values": int(np.count_nonzero(dr[:, :14])),
                "changed_padding_values": int(np.count_nonzero(dr[:, 14:])),
            }
        )
        knots.append(
            {
                "first": a,
                "second": b,
                "source_hz": 30,
                "raw_sampler_delta_50x32": dr.tolist(),
                "first_absolute_50x16": aa.tolist(),
                "second_absolute_50x16": ba.tolist(),
            }
        )
    output.mkdir(exist_ok=False)
    knot_file = output / "FEEDBACK_KNOTS.json.gz"
    with knot_file.open("xb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:
        z.write(json.dumps(knots, separators=(",", ":"), allow_nan=False).encode())
    write_new(
        output / "FEEDBACK_GOAL_DIRECTION.json",
        {
            "panel_score_sha256": digest(score_file),
            "scene_sha256": digest(scene),
            "native_state_journal_sha256": digest(native / "state.bin.journal"),
            "per_knot_sha256": digest(knot_file),
            "input_count": len(result),
            "calls": result,
            "pairs": pairs,
            "physics_integrations": 0,
            "additional_forwards": 0,
            "notes": "Object center only enters scoring. Distances cover the raw 50-knot world TCP proposal in the parked V2 scene; no pose executed.",
        },
    )


if __name__ == "__main__":
    root, score_file, output = (Path(s).resolve() for s in sys.argv[1:])
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    run(root, repo, score_file, output)
