"""CPU examples cannot qualify a physical panel; test real-output gate logic."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.target_mujoco.reference import (
    CONDITIONS, interface_contract, native_document, panel_chunks, panel_pose,
    pose_error, pose_matrix, score_reference,
)

INITIAL = [.4, -.1, .6, .2, -.3, .4, .8]
EVIDENCE = dict(real_output=True, native_abort_record_complete=True,
    joint_bounds_record_complete=True, contact_record_complete=True,
    clock_record_complete=True, dispatch_record_complete=True, native_abort=False,
    numerical_bound_warning=False, illegal_contact=False, same_time_pose_alignment=True)


def rows(condition):
    result = []
    for i in range(201):
        t = i * .05
        pose = panel_pose(condition, INITIAL, t)
        result.append(dict(t=t, requested_pose=pose, installed_pose=pose, achieved_pose=pose,
            ref_id="episode:0", sim_time_s=t, observation_time_s=t, install_time_s=0,
            dispatch_time_s=t, achieved_time_s=t, safety_ok=True))
    return result


def test_registered_panel_geometry_and_speed():
    for condition in CONDITIONS:
        samples = [panel_pose(condition, INITIAL, i * .01) for i in range(1001)]
        deltas = [pose_error(a, b) for a, b in zip(samples, samples[1:])]
        assert max(p for p, _ in deltas) / .01 <= .05
        assert max(a for _, a in deltas) / .01 <= .20
    rotation = pose_matrix(INITIAL)[:3, :3].T @ pose_matrix(panel_pose(CONDITIONS[3], INITIAL, 2))[:3, :3]
    assert np.allclose(rotation[:, 2], [0, 0, 1])
    assert np.isclose(np.arctan2(rotation[1, 0], rotation[0, 0]), .15)
    assert np.isclose(panel_pose(CONDITIONS[1], INITIAL, 2)[0] - INITIAL[0], .05)
    assert pose_error(panel_pose(CONDITIONS[4], INITIAL, 6), INITIAL)[0] < 1e-12


def test_native_h40_keys_and_common_future_identity():
    chunks = panel_chunks(CONDITIONS[5], INITIAL, "episode")
    assert len(chunks) == 25 and len({x["ref_id"] for x in chunks}) == 25
    assert len(chunks[0]["document"]["waypoints"]) == 40
    assert chunks[0]["document"]["waypoints"][-1]["time"] == 1.95
    for a, b in zip(chunks, chunks[1:]):
        assert a["document"]["frame"] == "world"
        for old, new in zip(a["document"]["waypoints"][8:], b["document"]["waypoints"]):
            assert np.allclose(old["position"], new["position"], atol=1e-12)
            assert np.allclose(old["quaternion_xyzw"], new["quaternion_xyzw"], atol=1e-12)
    assert set(native_document(CONDITIONS[0], INITIAL)) == {"version", "frame", "start_delay", "lead_in", "align_orientation", "waypoints"}
    assert interface_contract()["native_ids_ack_cancel_TTL"].startswith("none")
    with pytest.raises(ValueError):
        panel_pose(CONDITIONS[0], [0] * 7, 0)


def test_scorer_real_evidence_direction_full_duration_and_jumps():
    for condition in CONDITIONS[:5]:
        assert score_reference(condition, INITIAL, rows(condition), runtime_evidence=EVIDENCE)["pass"]
    assert not score_reference(CONDITIONS[0], INITIAL, rows(CONDITIONS[0]))["pass"]
    assert not score_reference(CONDITIONS[0], INITIAL, rows(CONDITIONS[0])[:190], runtime_evidence=EVIDENCE)["pass"]
    immobile = rows(CONDITIONS[1])
    for x in immobile:
        x["achieved_pose"] = INITIAL
    assert "DIRECTIONAL_TRANSLATION_FAIL" in score_reference(CONDITIONS[1], INITIAL, immobile, runtime_evidence=EVIDENCE)["reasons"]
    warning = {**EVIDENCE, "numerical_bound_warning": True}
    assert not score_reference(CONDITIONS[0], INITIAL, rows(CONDITIONS[0]), runtime_evidence=warning)["pass"]
    chunks = panel_chunks(CONDITIONS[5], INITIAL, "episode")
    accepted = [{"ref_id": x["ref_id"], "issue_time_s": x["issue_time_s"], "accept_time_s": x["issue_time_s"], "install_time_s": x["issue_time_s"]} for x in chunks]
    comparisons = []
    for a, b in zip(chunks, chunks[1:]):
        future = b["issue_time_s"] + .05
        pose = panel_pose(CONDITIONS[5], INITIAL, future)
        comparisons.append(dict(old_ref_id=a["ref_id"], new_ref_id=b["ref_id"], future_time_s=future, old_pose=pose, new_pose=pose))
    arguments = dict(issued=chunks, accepted=accepted, common_future=comparisons, runtime_evidence=EVIDENCE)
    assert score_reference(CONDITIONS[5], INITIAL, rows(CONDITIONS[5]), **arguments)["pass"]
    comparisons[0]["new_pose"] = [*comparisons[0]["new_pose"]]
    comparisons[0]["new_pose"][0] += .002
    assert "COMMON_FUTURE_REFERENCE_JUMP" in score_reference(CONDITIONS[5], INITIAL, rows(CONDITIONS[5]), **arguments)["reasons"]


def test_doctor_does_not_admit_missing_clock_or_files(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/pi0288_mujoco/doctor.py"
    spec = importlib.util.spec_from_file_location("pi0288_doctor", path)
    doctor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor)
    assert doctor.inspect({"private_root": str(tmp_path)})["status"] == "ENV_READY_INTERFACE_BLOCKED"
    assert doctor.check_clock(None)["status"] == "BLOCKED_LIVE_CLOCK_RECEIPT"
    receipt = dict(real_output=True, reset_isolation_checked=True, expired_reference_hold_checked=True,
        episode_id="episode", sim_to_mpc=dict(slope=1., offset_s=3.),
        samples=[dict(sim_time_s=i*.1, mpc_time_s=3+i*.1, ros_time_s=100+i*.1, wall_monotonic_s=500+i*.1) for i in range(3)])
    assert doctor.check_clock(receipt)["status"] == "CLOCK_CONTRACT_BOUND"
    receipt["samples"][-1]["sim_time_s"] = -.1
    assert doctor.check_clock(receipt)["status"] == "BLOCKED_CLOCK_OR_RESET"


def check_actual_planner_observer(native_root, observer_text, trace_path):
    """Run real private parser/callbacks without constructing a ROS node."""
    import sys
    sys.path.insert(0, str(Path(native_root) / "src/planning/go2_x5_planning"))
    from go2_x5_planning.ee_trajectory_node import EETrajectoryManager as manager_type
    native_accept, native_send = manager_type.on_trajectory_json, manager_type._send
    native_arm = manager_type._arm_trajectory
    calls = {"accept": 0, "send": 0, "publish": 0, "arm": 0}
    def counted_accept(self, msg):
        calls["accept"] += 1
        return native_accept(self, msg)
    def counted_send(self, msg):
        calls["send"] += 1
        return native_send(self, msg)
    manager_type.on_trajectory_json, manager_type._send = counted_accept, counted_send
    old_trace = os.environ.get("PI0288_PLANNER_TRACE")
    os.environ["PI0288_PLANNER_TRACE"] = str(trace_path)
    try:
        exec(compile(observer_text, "sitecustomize.py", "exec"), {})
        warnings = []
        logger = SimpleNamespace(info=lambda _: None, warning=warnings.append)
        manager = SimpleNamespace(trajectory_source=None, trajectory=None,
            observation=SimpleNamespace(time=1.2), bound_start=None,
            reference_enabled=True, goal=None, base_goal=None, trajectory_started=False,
            get_logger=lambda: logger)
        def arm():
            calls["arm"] += 1
            return native_arm(manager)
        manager._arm_trajectory = arm
        def publish(_):
            calls["publish"] += 1
        manager.target_pub = SimpleNamespace(publish=publish)
        wire = json.dumps(native_document(CONDITIONS[0], INITIAL))
        manager_type.on_trajectory_json(manager, SimpleNamespace(data=wire))
        accepted_source = manager.trajectory_source
        assert accepted_source is not None and manager.trajectory_started
        assert calls == dict(accept=1, send=0, publish=0, arm=1)
        manager_type.on_trajectory_json(manager, SimpleNamespace(data='{"waypoints":[]}'))
        assert manager.trajectory_source is accepted_source and len(warnings) == 1
        assert calls == dict(accept=2, send=0, publish=0, arm=1)
        target = SimpleNamespace(time_trajectory=[1.2],
            state_trajectory=[SimpleNamespace(value=INITIAL)],
            input_trajectory=[SimpleNamespace(value=[0.] * 12)])
        manager_type._send(manager, target)
        assert calls == dict(accept=2, send=1, publish=1, arm=1)
        assert manager.target_has_base is False
        recorded = [json.loads(x) for x in Path(trace_path).read_text().splitlines()]
        assert [x["kind"] for x in recorded] == ["accept", "accept", "installed"]
        assert recorded[0]["accepted"] is True and recorded[1]["accepted"] is False
        assert recorded[2]["ref_id"] == recorded[0]["ref_id"]
        assert recorded[2]["poses"] == [INITIAL] and recorded[2]["inputs"] == [[0.] * 12]
        return {"status": "CPU_NATIVE_CALLBACK_SEMANTICS_PASS", "calls": calls,
                "invalid_JSON_preserves_active_reference": True, "trace": str(trace_path),
                "physics_integrations": 0, "WBC_forwards": 0}
    finally:
        manager_type.on_trajectory_json, manager_type._send = native_accept, native_send
        if old_trace is None:
            os.environ.pop("PI0288_PLANNER_TRACE", None)
        else:
            os.environ["PI0288_PLANNER_TRACE"] = old_trace


def test_actual_private_planner_observer(tmp_path):
    native_root = os.environ.get("PI0288_PRIVATE_ROOT")
    if not native_root:
        pytest.skip("actual private native source is on Corelab; run source CPU check there")
    observer = os.environ.get("PI0288_OBSERVER_SITE") or str(Path(__file__).resolve().parents[2] / "scripts/pi0288_mujoco/planner_observer/sitecustomize.py")
    result = check_actual_planner_observer(native_root, Path(observer).read_text(), tmp_path / "planner.jsonl")
    assert result["status"] == "CPU_NATIVE_CALLBACK_SEMANTICS_PASS"
