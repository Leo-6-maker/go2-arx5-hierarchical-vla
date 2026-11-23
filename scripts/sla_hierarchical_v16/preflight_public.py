"""Read-only gate for V1.6 from *public* sealed O1, O3 and F2 JSON.

This check DOES NOT authenticate private checkpoint bytes, run SLA, send ROS
commands, qualify dynamic A1/contact, or grant native/GPU experiment budgets.
Only independently signed native receipts may qualify those later gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class EvidenceMismatch(ValueError):
    pass


def require(value: bool, description: str) -> None:
    if not value:
        raise EvidenceMismatch(description)


def assess(o1: dict, o3: dict, f2: dict, registration: dict) -> dict:
    require(registration["status"] == "PLAN_ONLY_NOT_AUTHORIZED", "unexpected plan authority")
    require(registration["base_parent_sha"] == "d6639cb03af1ff98519b2566cee2c9ee29627ba8", "incorrect parent experiment")
    require(o1["denominator"] == 16 and len(o1["windows"]) == 16, "different O1 science population")
    ids = [row["id"] for row in o1["windows"]]
    require(len(set(ids)) == len(ids), "duplicate O1 window ID")
    require(o1["labels"]["DIRECT_OFFICIAL_RAW"]["TASK_LEGAL"] == 0, "legacy score changed")
    require(o1["labels"]["OFFICIAL_WITH_CONTINUOUS_REF"]["TASK_LEGAL"] == 0, "new baseline score changed")
    require(o1["labels"]["OFFICIAL_WITH_CONTINUOUS_REF"]["DYNAMICS_PASS"] == 0, "dynamic science gate changed")
    require(o1["labels"]["OFFICIAL_WITH_CONTINUOUS_REF"]["NONTRIVIAL_PROGRESS"] == 0, "first-P10 progress reinterpreted")
    require(o1["gripper_commands_issued"] == 0 and o1["heldout_episodes"] == 0, "unexpected student task or gripper")
    require(
        all(o1[k] == 0 for k in ("new_gpu_forwards", "new_physics_ticks", "training_updates")),
        "unexpected O1 resource consumption",
    )
    require(o3["archived_student_owner_receipts"] == 0 and o3["blocked_windows"] == 16, "gripper ownership changed")
    require(o3["new_native_runs"] == 0, "unexpected gripper native activity")
    require(f2["decision"] == "FROZEN_WBC_KNOWN_GOAL_DWELL_QUALIFIED", "unexpected F2 status")
    require(len(f2["matched_comparisons"]) == 4, "F2 matched denominator changed")
    require(f2["native_student_grasps_attempted"] == 0, "unrecorded student grasps")
    require(f2["student_grasp_success_rate"] is None, "invalid unmeasured task rate")
    require(
        f2["SLA_forwards"] == 0
        and f2["training_updates"] == 0
        and f2["physical_stops"] == 0,
        "unexpected F2 activity",
    )
    require(
        all(registration["proposed_resources"][k] == 0 for k in (
            "new_gpu_forwards_authorized", "new_native_requests_authorized",
            "new_physics_ticks_authorized", "training_updates_authorized",
            "real_hardware_runs_authorized",
        )),
        "registration silently granted resources",
    )
    return {
        "scope": "V1.6 PUBLIC_READ_ONLY_PREFLIGHT",
        "status": "PUBLIC_EVIDENCE_CONSISTENT",
        "real_student_grasp_success_rate": None,
        "source_model": "official_real_posttrained_SLA_Memory",
        "prior_replay_windows": 16,
        "old_H40_student_legal": 0,
        "continuous_reference_student_legal": 0,
        "affected_dynamics_windows": 16,
        "archived_student_gripper_owner": 0,
        "static_feedback_matched_pairs": 4,
        "static_feedback_is_not_dynamic_tracking_qualification": True,
        "native_or_GPU_permission_granted": False,
        "next_parallel_CPU_stages": ["V1_TRAJECTORY_COMPILER", "V2_GRIPPER_CONTRACT", "V3_A1_DYNAMIC_PREFLIGHT", "V4_POLICY_SKILL_DIAGNOSTIC"],
        "native_student_control": "BLOCKED_PENDING_NEW_SCIENCE_AND_RESOURCE_APPROVALS",
        "student_contact": "BLOCKED_PENDING_OWNER_DYNAMIC_AND_CONTACT_SAFETY",
        "evidence_limitations": [
            "No direct policy-selected MRT ID",
            "F2 only proved slow stationary endpoint dwell, not moving contact",
            "Go2 observations have a virtual left camera and arm",
            "Original student task success is unmeasured, not zero percent",
        ],
    }


def read_source(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o1", type=Path, required=True)
    parser.add_argument("--o3", type=Path, required=True)
    parser.add_argument("--f2", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = {key: read_source(path) for key, path in [
        ("o1", args.o1), ("o3", args.o3), ("f2", args.f2),
        ("registration", args.registration),
    ]}
    result = assess(*(sources[name][0] for name in ("o1", "o3", "f2", "registration")))
    result["source_sha256"] = {key: value[1] for key, value in sources.items()}
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
