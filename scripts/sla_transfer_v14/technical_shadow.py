"""Materialize both proposed B2 holds through the guarded site receiver at a recorded pose."""

import sys
from pathlib import Path

import numpy as np
from clock_guard import new_receiver_output
from clock_replay import GuardedManager, episode, signal
from identity import read, record, sha
from ocs2_msgs.msg import MpcObservation
from shadow import check, message, packet

from go2arx5_vla.trajectory.so3 import matrix_to_quat


def main(root, out, registration):
    plan = read(registration)
    old = read(root / "v13/run01/S0_ACTUAL.json")
    spec = read(
        next(Path(k) for k in old["inputs"] if k.endswith("/EXECUTION_SPEC.json"))
    )
    history = Path(spec["mobile_interface_qualification"]["path"])
    assert sha(history) == plan["card_source_sha256"]
    detail = read(history)["detail"]
    tcp, base = (np.array(detail[k]) for k in ("initial_TCP", "initial_mount"))
    ee_q = matrix_to_quat(tcp[:3, :3]).tolist()
    base_q = matrix_to_quat(base[:3, :3]).tolist()
    yaw_pitch_roll = [
        np.arctan2(base[1, 0], base[0, 0]),
        np.arctan2(-base[2, 0], np.hypot(base[0, 0], base[1, 0])),
        np.arctan2(base[2, 1], base[2, 2]),
    ]
    data = []
    for card in plan["cards"]:
        clock = card["initialization_tick_cap"] * card["physics_dt_s"]
        directory = new_receiver_output(out)
        with episode(directory.name, directory):
            manager = GuardedManager()
            before = None
            # The physical output is unknown. Both measured inputs in this replay
            # are held at the recorded pose exclusively to check the reference.
            for seq in range(card["after_initialization_tick_cap"] // 50):
                now = clock + seq * 0.1
                ee = MpcObservation()
                ee.time = now
                ee.mode = 1
                ee.state.value = [*tcp[:3, 3], *ee_q]
                manager.on_ee_state(ee)
                obs = MpcObservation()
                obs.time = now
                obs.state.value = [
                    *base[:3, 3],
                    *yaw_pitch_roll,
                    *([0.0] * 6),
                ]
                manager.on_observation(obs)
                doc = packet(now, seq, base=True)
                doc["_pi0294"]["episode"] = directory.name
                for part, pose, quat in (("ee", tcp, ee_q), ("base", base, base_q)):
                    for point in doc[part]["waypoints"]:
                        point["position"] = pose[:3, 3].tolist()
                        point["quaternion_xyzw"] = quat
                manager.on_trajectory_json(message(doc))
                field = check(manager.target_pub[-1], 14)
                assert np.allclose(
                    field,
                    [*tcp[:3, 3], *ee_q, *base[:3, 3], *base_q],
                    atol=1e-6,
                    rtol=0,
                )
                assert len(field) == 41 and (
                    before is None or np.array_equal(before, field)
                )
                before = field
            targets = len(manager.target_pub)
            if card["id"] == "T1_CLOCK_FAULT_STOP":
                signal(manager, "mpc", clock - 0.1)
                manager.publish()
                assert manager.clock_fault and len(manager.target_pub) == targets
            else:
                doc["_pi0294"].update(operation="cancel", sequence=seq + 1)
                manager.on_trajectory_json(message(doc))
            assert [m.data for m in manager._r4_mode] == [
                "HOLD"
            ] and not manager.trajectory_started
            data.append(
                {
                    "card": card["id"],
                    "targets": targets,
                    "points_per_target": 41,
                    "reference_width": 14,
                    "target_translation_change_m": 0.0,
                    "target_rotation_change_rad": 0.0,
                    "callback_hold_requests": 1,
                    "live_hold_ack": None,
                    "new_native_ticks": 0,
                }
            )
    record(
        out / "TECHNICAL_SHADOW_ACTUAL.json",
        {
            "cards": data,
            "physics": "NOT_RUN",
            "source_pose_sha256": sha(history),
            "actual_tracking_error": None,
            "integrations_after_physical_STOP": None,
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
