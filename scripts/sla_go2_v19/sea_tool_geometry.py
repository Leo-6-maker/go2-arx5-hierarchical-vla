"""Read sealed gripper units and common-tool geometry without stepping either robot."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

from source_custody import BASE, SOURCE_H5, digest, pin, read, require, target_case, write_new
from demonstrated_grip import rows

SEA = BASE / "reports/SLA_GO2_V19"
V2 = BASE / "reports/SLA_HIERARCHICAL_V16/V2_STUDENT_GRIPPER_OWNER_ACTUAL.json"
S0_SHA = "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9"
DIAGNOSTIC_SLIDE_M = (0., .002, .038, .042, .044)
DT_S = .002


class ParallelJaw:
    def __init__(self, model):
        import mujoco as mj

        self.mj, self.model = mj, model
        self.data = mj.MjData(model)
        self.joints = [model.joint(f"task_{side}_slide").id for side in ("left", "right")]
        self.pads = [model.geom(f"task_{side}_pad").id for side in ("left", "right")]
        self.slides = model.jnt_qposadr[self.joints]
        require(np.all(model.geom_type[self.pads] == mj.mjtGeom.mjGEOM_BOX)
                and np.all(model.jnt_type[self.joints] == mj.mjtJoint.mjJNT_SLIDE)
                and np.array_equal(model.geom_bodyid[self.pads], model.jnt_bodyid[self.joints]),
                "unexpected slide or collision pad geometry")
        require(np.array_equal(model.jnt_range[self.joints], [[0., .044], [0., .044]]),
                "unexpected frozen slide travel")

    def observe(self, qpos):
        d, m = self.data, self.model
        d.qpos[:] = qpos
        self.mj.mj_kinematics(m, d)
        axis = d.xaxis[self.joints[0]]
        require(np.allclose(d.xaxis[self.joints[1]], -axis, rtol=0, atol=1e-10), "slides no longer oppose")
        # Each pad is a box with one face normal parallel to the slide axis.
        # Its inner face center lies one projected half-extents inward.
        rot = d.geom_xmat[self.pads].reshape(2, 3, 3)
        axis_in_pad = abs(rot.transpose(0, 2, 1) @ axis)
        require(np.allclose(axis_in_pad.max(axis=1), 1., rtol=0, atol=1e-10), "pad face does not align")
        half = np.sum(axis_in_pad * m.geom_size[self.pads], axis=1)
        inner = d.geom_xpos[self.pads] - np.array([1., -1.])[:, None] * half[:, None] * axis
        tip = d.site("x5_ee")
        tip_rotation = tip.xmat.reshape(3, 3)
        flange = d.body("x5_link6")
        flange_rotation = flange.xmat.reshape(3, 3)
        flange_to_tip = np.eye(4)
        flange_to_tip[:3, :3] = flange_rotation.T @ tip_rotation
        flange_to_tip[:3, 3] = flange_rotation.T @ (tip.xpos - flange.xpos)
        world_tip = np.eye(4)
        world_tip[:3, :3], world_tip[:3, 3] = tip_rotation, tip.xpos
        return dict(
            slide_m=d.qpos[self.slides].copy(), pad_plane_gap_m=float((inner[0] - inner[1]) @ axis),
            pad_midpoint_from_tcp_m=(inner.mean(axis=0) - tip.xpos) @ tip_rotation,
            world_tcp=world_tip, flange_to_tcp=flange_to_tip,
        )

    def symmetric_geometry(self, qpos):
        values = []
        for per_slide in DIAGNOSTIC_SLIDE_M:
            code = qpos.copy()
            code[self.slides] = per_slide
            obs = self.observe(code)
            values.append(dict(per_slide_mm=1000 * per_slide, pad_gap_mm=1000 * obs["pad_plane_gap_m"],
                               pad_midpoint_from_tcp_mm=(1000 * obs["pad_midpoint_from_tcp_m"]).tolist()))
        intercept = values[0]["pad_gap_mm"]
        require(all(abs(v["pad_gap_mm"] - intercept - 2 * v["per_slide_mm"]) < 1e-7 for v in values),
                "pad-gap travel does not equal the sum of opposing slides")
        return values


def registered_tool(root):
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0_SHA))["inputs"]
    calibration = root / "v12/TARGET_REGISTRATION.json"
    return np.asarray(read(pin(calibration, pins[str(calibration)]))["T_link6_TCP"])


def v2_replay(root, calibration):
    import mujoco as mj

    sys.path.insert(0, str(BASE))
    from verify_grip_card import states

    original = read(V2)
    pins = original["source_sha256"]
    case = root / "v16/sla-v16-student-gripper-native-01b/V2-GRIP-01B"
    spec = read(pin(case / "EXECUTION_SPEC.json", pins["EXECUTION_SPEC.json"]))
    result = read(pin(case / "RESULT.private.json", pins["RESULT.private.json"]))
    lease = read(pin(case / "control/LEASE.json", pins["LEASE.json"]))
    require(lease["status"] == "CLOSED" and original["non_support_contacts"] == 0, "V2 archive not closed")
    pin(case / "student/receiver.csv", pins["receiver.csv"])
    pin(case / "student/issuer.jsonl", pins["issuer.jsonl"])
    native = case / result["episode"]
    pin(native / "state.bin.journal", pins["state.bin.journal"])
    scene = Path(spec["model_path"])
    scene_sha = spec["frozen_files"][str(scene)]
    jaw = ParallelJaw(mj.MjModel.from_binary_path(str(pin(scene, scene_sha))))
    schedule = original["verified_command_intervals"]
    archived_samples, gaps = {}, [None]
    wanted = {r[k] for r in schedule for k in ("source_tick", "end_tick")}
    maximum_gap_model_error = 0.
    for p in states(native):
        t = p["step"]
        sample = jaw.observe(p["qpos"])
        if t == 1:
            working = jaw.symmetric_geometry(p["qpos"])
            zero_gap = working[0]["pad_gap_mm"] / 1000
        maximum_gap_model_error = max(maximum_gap_model_error,
                                      abs(sample["pad_plane_gap_m"] - zero_gap - sum(sample["slide_m"])))
        require(np.allclose(sample["flange_to_tcp"], calibration, rtol=0, atol=2e-7),
                "archive TCP site changed from its registered transform")
        gaps.append(sample["pad_plane_gap_m"])
        if t in wanted:
            archived_samples[t] = sample
    require(len(gaps) == original["native_ticks"] + 1 and set(archived_samples) == wanted,
            "V2 native archive incomplete")
    require(maximum_gap_model_error < 1e-9, "gap mapping changed during V2")
    commands = []
    for row in schedule:
        start, end = row["source_tick"], row["end_tick"]
        require(row["receiver_ack_tick"] == start and end - start == row["applied_native_steps"] == 25,
                "V2 consumption clock changed")
        a, b = archived_samples[start], archived_samples[end]
        for sample, label in ((a, "start"), (b, "end")):
            require(np.allclose(1000 * sample["slide_m"], row[f"measured_slide_{label}_mm"], rtol=0, atol=1e-10),
                    "measured V2 slide differs from the sealed command receipt")
        commands.append(dict(sequence=row["sequence"], start_tick=start, end_tick=end,
                             commanded_per_slide_mm=1000 * row["width_m"], wire_sha256=row["wire_sha256"],
                             measured_start_gap_mm=1000 * a["pad_plane_gap_m"],
                             measured_end_gap_mm=1000 * b["pad_plane_gap_m"],
                             measured_end_slide_mm=(1000 * b["slide_m"]).tolist(),
                             end_pad_midpoint_from_tcp_mm=(1000 * b["pad_midpoint_from_tcp_m"]).tolist()))
    tabs = {name: commands[i] for name, i in (("initial_open_hold", 127), ("closed_hold", 143), ("reopen_hold", 159))}
    return dict(
        model_sha256=scene_sha, actual_report_sha256=digest(V2), closed_historical_card=True,
        historical_native_ticks=original["native_ticks"], historical_verified_intervals=len(commands),
        synthetic_geometry=working, maximum_gap_formula_error_m=float(maximum_gap_model_error),
        original_per_slide_command_limit_mm_s=40.,
        historical_applied_per_slide_target_peak_mm_s=original["peak_applied_rate_mm_s"],
        measured_pad_gap_rate_peak_mm_s=float(1000 * np.max(abs(np.diff(gaps[schedule[0]["source_tick"]:]))) / DT_S),
        phase_endpoints=tabs, commands=commands,
    )


def go2_layouts(root, calibration):
    import mujoco as mj

    anchors = read(SEA / "SEA0/SEA0_PAIRED_ROSTER.json")["go2"]
    archive = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", archive["member_manifest_sha256"]))

    def sealed(path):
        return pin(path, members[str(path.relative_to(root / "v18"))])

    result = []
    for a in anchors:
        case = target_case(root, a["layout"])
        spec = read(sealed(case / "EXECUTION_SPEC.json"))
        episode = read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
        scene = pin(Path(spec["model_path"]), a["scene_sha256"])
        jaw = ParallelJaw(mj.MjModel.from_binary_path(str(scene)))
        packet = next(p for p in rows(sealed(case / episode / "MOBILE_SAMPLES.private.jsonl"))
                      if p["native_end_step"] == a["tick"])
        sensor = np.asarray(packet["qpos"])
        observed = jaw.observe(sensor)
        require(np.allclose(observed["flange_to_tcp"], calibration, rtol=0, atol=2e-7)
                and np.allclose(observed["world_tcp"], packet["actual_TCP"], rtol=0, atol=2e-7),
                "target tool calibration or sample identity changed")
        q = case / f"query-{int(a['id'][-2:]):02d}/INPUT.npz"
        with np.load(pin(sealed(q), a["input_file_sha256"]), allow_pickle=False) as z:
            state_fraction = float(z["state"][15])
        require(state_fraction == np.float32(observed["slide_m"].mean() / .044),
                "legacy feedback no longer represents mean slide over full travel")
        result.append(dict(
            id=a["id"], tick=a["tick"], scene_sha256=a["scene_sha256"],
            pad_gap_mm=1000 * observed["pad_plane_gap_m"],
            measured_slide_mm=(1000 * observed["slide_m"]).tolist(),
            pad_midpoint_from_tcp_mm=(1000 * observed["pad_midpoint_from_tcp_m"]).tolist(),
            legacy_policy_state_fraction=state_fraction, symmetric_geometry=jaw.symmetric_geometry(sensor),
        ))
    return result


def source_signals(root):
    import h5py

    episodes = []
    for name, sha in SOURCE_H5.items():
        with h5py.File(pin(root / "data" / name, sha)) as h:
            signals = {}
            for group in ("state", "action"):
                for side in ("left", "right"):
                    key = f"{group}/{side}_ee_joint_states"
                    field = h[key]
                    signal = np.asarray(field)
                    require(signal.ndim == 2 and signal.shape[1] == 1 and np.isfinite(signal).all(),
                            "expected one raw gripper value per arm and frame")
                    values = signal[:, 0]
                    signals[key] = dict(n=len(values), min=float(values.min()), max=float(values.max()),
                                        percentiles=np.percentile(values, [5, 50, 95]).tolist(),
                                        outside_0_1=int(np.count_nonzero((values < 0) | (values > 1))),
                                        dataset_attribute_keys=sorted(field.attrs))
            clock_checks = {}
            for side in ("left", "right"):
                for field in ("ee_joint_states", "ee_poses", "arm_joint_states"):
                    key = f"{side}_{field}"
                    actions, states = (h[f"{name}/{key}"][:] for name in ("action", "state"))
                    require(actions.ndim == 2 and actions.shape == states.shape and len(actions) > 2
                            and actions.dtype == states.dtype and np.isfinite(actions).all()
                            and np.isfinite(states).all(), "source action/state clock or encoding changed")
                    same = (actions[:-1] == states[1:]).all(axis=1)
                    clock_checks[key] = dict(
                        comparable_next_rows=len(same), exact_equal_next_rows=int(same.sum()),
                        all_next_rows_byte_exact=actions[:-1].tobytes() == states[1:].tobytes(),
                        exact_equal_same_index_rows=int((actions == states).all(axis=1).sum()),
                        terminal_action_equals_last_state_bytes=actions[-1].tobytes() == states[-1].tobytes(),
                        first_different_next_row=int(np.flatnonzero(~same)[0]) if not same.all() else None,
                    )
            episodes.append(dict(episode=name, sha256=sha, fps=int(h["additional_info/frequency"][()]),
                                 additional_info_keys=sorted(h["additional_info"]), signals=signals,
                                 action_state_clock_checks=clock_checks))
    return dict(episodes=episodes, calibrated_source_pad_gap_available=False,
                calibrated_source_link6_to_common_tool_available=False)


def run(root, output, domain):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only tool readback")
    if domain == "go2":
        calibration = registered_tool(root)
        result = dict(source_scope="SEALED_MUJOCO_GEOMETRY", go2_flange_to_tcp=calibration.tolist(),
                      v2=v2_replay(root, calibration), go2_developers=go2_layouts(root, calibration))
    else:
        result = dict(source_scope="REAL_ROBODOJO_HDF5_GRIPPER_SIGNALS", source=source_signals(root))
    write_new(output, dict(**result, new_integrations=0, source_sampler_forwards=0, training_updates=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("domain", choices=("go2", "source"))
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.domain)
