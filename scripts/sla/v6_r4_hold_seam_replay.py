"""Replay the two counted R4 HOLD seam violations through the original L1 compiler."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from m38_a_native_v1 import bind_native
from m38_gpu_preflight_v1 import pin, require, sha


CASES = ("CONFIRM_N02_201", "CONFIRM_N02_202")


def read(path, digest=None):
    path = Path(path)
    return json.loads((path if digest is None else pin(path, digest)).read_text())


def replay(root, result_digest):
    result_path = root / "R4_STEP0200_SCREEN/RESULT.json"
    result = read(result_path, result_digest)
    rows = [r for r in result["rows"] if r["metrics"]["jaw"]["original_execution_speed_exceeded"]]
    require(tuple(r["id"] for r in rows) == CASES and len(result["rows"]) == 24,
            "Expected the two fixed post-grasp R4-200 seams")
    parent_path = root / "STEP0000_RETENTION/RESULT.json"
    parent = read(parent_path)
    require(result["roster"] == parent["roster"], "Parent used a different feedback roster")
    parent_rows = {r["id"]: r for r in parent["rows"]}
    first_query = Path(rows[0]["request"]["path"]).parent
    bind_native(read(first_query.parents[2] / "EXECUTION_SPEC.json"))
    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    from go2arx5_vla.world_model import sla_dense_policy_v5, task_learning_grasp_v1
    from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
    from scripts.pi0292.eval_contract import matrix10
    from scripts.sla import m38_command_continuity_v1 as continuity, m38_native_link_v1 as link

    cases = []
    for row in rows:
        q = Path(row["request"]["path"]).parent
        case, native = q.parents[2], q.parents[1]
        spec = read(case / "EXECUTION_SPEC.json")
        require(spec["interface_comparison_link"] == "L1" and not spec.get("comparison_jaw_bridge")
                and read(case / "control/LEASE.json")["status"] == "CLOSED", "Wrong archived L1 condition")
        for module in (sla_dense_policy_v5, task_learning_grasp_v1):
            path = Path(module.__file__)
            relative = path.relative_to(spec["source_root"])
            pin(path, spec["source_files"][str(relative)])
        req = read(row["request"]["path"], row["request"]["sha256"])
        meta = read(q / "INPUT_META.private.json")
        tick, previous = req["native_step"], req["previous_reference"]
        control = parent_rows[row["id"]]
        require(req["input"] == meta["input"] == row["input"] == control["input"]
                and row["request"] == control["request"] and row["noise_sha256"] == control["noise_sha256"]
                and tick == meta["native_step"] == row["tick"] and previous == meta["previous_reference"],
                "Changed feedback input, consumed grip or noise")
        with np.load(pin(row["input"]["path"], row["input"]["sha256"]), allow_pickle=False) as packet:
            state, reference = packet["state_eef10"], packet["accepted_reference_TCP"]
        incoming = read(q.parent / f"{row['query'] - 1:04d}/QUERY_ACTUAL.private.json")
        intervals = incoming["confirmed_intervals"][-2:]
        ticks = [tick - 25, tick]
        require(len(intervals) == 2 and [r["native_end_step"] for r in intervals] == ticks
                and incoming["terminal_native_step"] == tick and incoming["full_P10"], "Missing confirmed intervals")
        trace = native / "MOBILE_SAMPLES.private.jsonl"
        with trace.open() as stream:
            history = [r for r in map(json.loads, stream) if r["native_end_step"] in ticks]
        require([r["native_end_step"] for r in history] == ticks, "Missing exact archived native consumption")
        travel = continuity.native.MECHANICAL_TRAVEL_M
        history_poses = []
        for interval, sample in zip(intervals, history, strict=True):
            pose = np.asarray(interval["accepted_boundary"]["accepted_pose7"])
            accepted = np.eye(4)
            accepted[:3, :3], accepted[:3, 3] = quat_to_matrix(pose[3:]), pose[:3]
            require(interval["native_end_step"] - interval["native_start_step"] == sample["ticks"] == 25
                    and interval["accepted_boundary"]["exact_receiver_knot"]
                    and np.allclose(accepted, matrix10(sample["target_endpoint_EEF10"]), rtol=0., atol=2e-7)
                    and abs(interval["emitted_gripper_width_m"] - sample["installed_EEF10"][9] * travel) <= 1e-12,
                    "Accepted receiver pose or emitted jaw differs from the consumption archive")
            history_poses.append(dict(end_native_step=sample["native_end_step"], pose=accepted))
        require(np.allclose(history_poses[-1]["pose"], reference, rtol=0., atol=2e-7)
                and abs(previous["width_m"] - intervals[-1]["emitted_gripper_width_m"]) <= 1e-12,
                "Captured query anchor differs from the receiver")
        last = history[-1]
        prepared = dict(metadata=meta, previous_reference=previous, observation=SimpleNamespace(
            values_for=lambda _: dict(state_eef10=state)))
        sample = dict(tcp=np.asarray(last["actual_TCP"]))

        def compile_for(action, mode):
            adapter.spec = dict(spec, interface_comparison_link=mode)
            try:
                program, evidence, speed = link.compile_link(adapter, prepared, sample, reference, dict(action=action))
            except StudentProposalRejected as error:
                return dict(status="REJECT", reason=str(error))
            if speed["original_execution_speed_exceeded"] and not evidence.get("link_retiming_qualified"):
                return dict(status="G7_REJECT", first_width_mm=1000. * program.preview(0)[0, 9] * travel,
                            first_seam_mm_s=1000. * speed["first_seam_speed_m_s"])
            if not evidence.get("link_retiming_qualified"):
                return dict(status="UNCHANGED", first_seam_mm_s=1000. * speed["first_seam_speed_m_s"],
                            preview_hashes=[hashlib.sha256(program.preview(i).tobytes()).hexdigest() for i in (0, 7)])
            raw_indices = evidence["original_knot_atoms"]
            require(len(raw_indices) == 40 and np.allclose(program.full[raw_indices],
                    evidence["original_command_rows"], rtol=0., atol=2e-7), "Retiming altered model knots")
            return dict(status="L1_RETIMED", preserved_original_knots=len(raw_indices), first_raw_atom=raw_indices[0],
                        raw_knots_in_p10=sum(i < continuity.P10 for i in raw_indices),
                        source_stretch=evidence["source_time_stretch"], peak_jaw_mm_s=1000. * evidence["candidate_peak_jaw_m_s"],
                        p10_scheduled_width_mm=(1000. * travel * program.preview(0)[:10, 9]).tolist(),
                        p10_z_delta_mm=1000. * (program.preview(0)[9, 2] - reference[2, 3]))

        with TemporaryDirectory(prefix="sla-r4-seam-cpu-") as scratch:
            (Path(scratch) / trace.name).write_text("".join(json.dumps(r) + "\n" for r in history))
            adapter = SimpleNamespace(output=scratch, base_pose_hold=np.asarray(last["qpos"][:7]),
                                      reference_history=history_poses)
            r4_raw, r4_l1 = (compile_for(row["action"], mode) for mode in ("L0", "L1"))
            parent_raw, parent_l1 = (compile_for(control["action"], mode) for mode in ("L0", "L1"))
        cases.append(dict(id=row["id"], phase=row["phase"], tick=tick, input_sha256=row["input"]["sha256"],
                          r4_action_sha256=row["action"]["sha256"], parent_action_sha256=control["action"]["sha256"],
                          trace_sha256=sha(trace), previous_width_mm=1000. * previous["width_m"],
                          r4_raw=r4_raw, r4_l1=r4_l1, parent_raw=parent_raw, parent_l1=parent_l1))
    return dict(screen_sha256=result_digest, parent_result_sha256=sha(parent_path), link_sha256=sha(link.__file__),
                continuity_sha256=sha(continuity.__file__), original_G7_limit_mm_s=1000. * continuity.native.MAX_FINGER_SPEED_M_S,
                new_model_forwards=0, new_physics_ticks=0, new_training_updates=0, cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--screen-sha256", required=True)
    args = parser.parse_args()
    result = replay(args.root, args.screen_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
