"""Read the target tool from the exact H0 scenes and test the physical codec seam."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from sea_physical_tool import COMMON, InvertibleJaw, PhysicalToolBridge, SourceTool, TargetTool
from source_calibration import SCHEMA
from sea_tool_geometry import ParallelJaw
from sea_visual_support import EVIDENCE, H0_DIR, H0_SHA
from source_custody import SOURCE_H5, digest, pin, read, require, write_new
from policy_query import input_identity, tensor_sha


def refusal(action, expected):
    try:
        action()
    except ValueError as error:
        require(expected in str(error), "wrong physical-admission refusal")
        return str(error)
    raise ValueError("physical-admission negative control unexpectedly passed")


def fixture(target, actual_tcp, fingers):
    # A made-up, noncommuting transform proves the direction of the inverses.
    g = np.eye(4)
    g[:3, :3] = Rotation.from_euler("z", .31).as_matrix()
    g[:3, 3] = [.121, -.011, .007]
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_euler("xy", [.17, -.21]).as_matrix()
    origin[:3, 3] = [.09, -.13, .26]
    src = SourceTool(g, InvertibleJaw([[0., .002], [.2, .019], [.65, .054], [1., .095]]),
                     "FIXTURE_ONLY", "FIXTURE_ONLY")
    bridge = PhysicalToolBridge(src, target)
    state = bridge.right_observation(origin, actual_tcp, fingers)
    expect = np.linalg.inv(origin) @ actual_tcp @ target.tcp_from_common @ np.linalg.inv(g)
    expected_code = np.interp(target.zero_gap_m+sum(fingers), src.jaw.gap_m, src.jaw.code)
    require(np.allclose(state[:3], expect[:3, 3], atol=2e-7, rtol=0) and abs(state[-1] - expected_code) < 1e-7
            and np.allclose(Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix(),
                            expect[:3, :3], atol=2e-7, rtol=0), "wrong calibrated observation transform")
    # The test descriptor is an in-memory fixture, never a task observation.
    observation = dict(images={"fixture": np.zeros((1, 1, 3), np.uint8)},
                       state=np.r_[np.zeros(8, np.float32), state], prompt="fixture")
    raw = np.tile(observation["state"], (50, 1))
    raw[:, 15] = np.linspace(.2, .75, 50, dtype=np.float32)
    raw[:, 8] += np.linspace(0., .016, 50, dtype=np.float32)
    key = input_identity(observation)[0]
    args = dict(expected_input_sha256=key, returned_input_sha256=key, decoded_sha256=tensor_sha(raw),
                world_from_policy=origin, world_from_tcp=actual_tcp, measured_slides_m=fingers)
    bound = bridge.decode(observation, raw, **args)
    error = np.max(abs(bound["tcp_world"][0] - actual_tcp))
    delta = origin[:3, :3] @ np.array([.016, 0., 0.])
    require(error < 2e-7
            and np.allclose(bound["tcp_world"][-1, :3, 3]-bound["tcp_world"][0, :3, 3], delta, atol=2e-7, rtol=0)
            and np.array_equal(raw, bound["raw"]) and not bound["raw"].flags.writeable
            and np.array_equal(bound["source_times_s"], np.arange(50)/30), "wrong physical source row binding")
    physical_gap = np.interp(raw[:, 15], [.0, .2, .65, 1.], [.002, .019, .054, .095])
    require(np.array_equal(bound["physical_gap_m"], physical_gap)
            and np.allclose(2*bound["per_finger_stroke_m"]+target.zero_gap_m, physical_gap, atol=1e-12, rtol=0),
            "physical per-finger stroke no longer realizes the measured gap")
    wrong = raw.copy()
    wrong[-1, 15] = 1.  # Fixture jaw gap .095 m exceeds the real target stroke.
    stale = refusal(lambda: bridge.decode(observation, raw, **(args | {"returned_input_sha256": "wrong"})),
                    "ACTION_OBSERVATION_MISMATCH")
    excessive = refusal(lambda: bridge.decode(observation, wrong, **(args | {"decoded_sha256": tensor_sha(wrong)})),
                        "TARGET_WORKING_STROKE_UNREACHABLE")
    limited = InvertibleJaw([[.1, .03], [.7, .044]])
    unmeasured_code = refusal(lambda: limited.gap_from_code(.75), "SOURCE_ACTION_CODE_UNCOVERED")
    unmeasured_gap = refusal(lambda: limited.code_from_gap(.05), "SOURCE_PHYSICAL_GAP_UNCOVERED")
    return dict(synthetic_source_tool=True, maximum_initial_TCP_transform_error=float(error),
                preserved_30hz_rows=50, refusals=[stale, excessive, unmeasured_code, unmeasured_gap])


def prove_missing_source_refusals():
    episode, sha = next(iter(SOURCE_H5.items()))
    with TemporaryDirectory() as location:
        outputs = []
        for name, archive, reason in (
            ("absent_measurement", sha, "SOURCE_PHYSICAL_TOOL_OR_PAD_MEASUREMENTS_MISSING"),
            ("different_recording", "0"*64, "source tool is unbound to this actual recorder"),
        ):
            fixture_path = Path(location) / (name+".json")
            write_new(fixture_path, dict(
                schema=SCHEMA, world_units="m", common_tool_definition=COMMON,
                episodes={episode: dict(archive_sha256=archive, fps=30, arms={"right": {}})},
            ))
            outputs.append(refusal(lambda: SourceTool.read_metrology(fixture_path, digest(fixture_path), episode), reason))
    outputs.append(refusal(lambda: InvertibleJaw([[.1, .03], [.6, .03]]), "strictly increasing source table"))
    return outputs


def run(root, output):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from sea_fruit_h0_capture import samples
    from sea_three_view import native_sample_identity

    summary = read(pin(EVIDENCE / "GRIP_H0/COMPACT_H0_ACTUAL.json", H0_SHA))
    campaign = root / "v19" / H0_DIR
    pin(campaign / "PANEL_CLOSE.json", summary["campaign_sha256"]["PANEL_CLOSE.json"])
    results, geometries = {}, {}
    with forbid_integration(mujoco) as attempted:
        for layout, record in summary["rows"].items():
            case = campaign / layout
            seals = record["evidence_sha256"]
            lease = read(pin(case / "control/LEASE.json", seals["LEASE.json"]))
            actual = read(pin(case / "RESULT.private.json", seals["RESULT.private.json"]))
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            require(lease["status"] == "CLOSED" and record["qualified"] and actual["absent"], "open H0 case")
            native = case / actual["episode"]
            pin(native / "state.bin.journal", seals["state.bin.journal"])
            scene = Path(spec["model_path"])
            scene_sha = spec["frozen_files"][str(scene)]
            jaw = ParallelJaw(mujoco.MjModel.from_binary_path(str(pin(scene, scene_sha))))
            packets = samples(native, actual["observation_events"])
            items = {}
            for event, sealed in record["capture"]["events"].items():
                pkt = packets[sealed["tick"]]
                require(native_sample_identity(native.name, pkt) == sealed["current_packet_sha256"],
                        "target observation differs from the sealed native packet")
                got = jaw.observe(pkt["qpos"])
                jaw_axis = got["world_tcp"][:3, :3].T @ jaw.data.xaxis[jaw.joints[0]]
                require(np.allclose(jaw_axis, [0., 1., 0.], atol=1e-9, rtol=0), "target opening axis differs from common +Y")
                symmetric = jaw.symmetric_geometry(pkt["qpos"])
                offset = symmetric[0]["pad_gap_mm"]/1000
                relative = np.eye(4)
                relative[:3, 3] = np.asarray(symmetric[0]["pad_midpoint_from_tcp_mm"])/1000
                require(all(np.allclose(r["pad_midpoint_from_tcp_mm"], relative[:3, 3]*1000, atol=1e-8, rtol=0)
                            for r in symmetric), "target symmetric common tool frame depends on opening")
                target = TargetTool(relative, offset)
                gap = target.gap_from_measured_slides(got["slide_m"])
                require(abs(gap - got["pad_plane_gap_m"]) < 1e-10
                        and abs(gap*1000-sealed["measured_go2_inner_pad_gap_mm"]) < 1e-8,
                        "measured actual target gap disagrees")
                geometries[f"{layout}/{event}"] = np.r_[relative[:3, 3], offset]
                items[event] = dict(measured_target_gap_mm=1000*gap, target_tcp_from_common=relative.tolist(),
                                    zero_target_gap_m=offset, effective_jaw_axis_in_tcp=jaw_axis.tolist(),
                                    pose_and_grip_fixture=fixture(target, got["world_tcp"], got["slide_m"]))
            results[layout] = dict(scene_sha256=scene_sha, events=items)
    require(not attempted and len(geometries) == 4 and all(
        np.allclose(a, next(iter(geometries.values())), atol=1e-10, rtol=0) for a in geometries.values()),
        "target tool differs across frozen scenes or states")
    write_new(output, dict(
        status="TARGET_PHYSICAL_TOOL_AND_CODEC_SEAM_VERIFIED", common_tool_definition=COMMON,
        compact_h0_receipt_sha256=H0_SHA, target_rows=results, real_source_codec_verified=False,
        source_intake_negative_controls=prove_missing_source_refusals(),
        still_required=["recording_matched_right_link6_from_common", "measured_right_code_to_pad_gap_m"],
        hardware_qualification=False, student_contact_qualification=False, model_forwards=0,
        new_native_integrations=0, new_rgb_renders=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU readback only")
    run(a.root.resolve(), a.output)
