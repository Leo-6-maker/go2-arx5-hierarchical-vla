"""Independent prelaunch exercise against a closed A0 card. No executor is opened."""

import argparse
from pathlib import Path

from native_owner import baseline, configure, pinned, read, record, sha


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("closed_card", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    base = baseline(root)
    configure(base)
    import mujoco
    from feedback import START, require
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from native_servo import preflight, verify_finger_file
    from verify_native import verify_card

    rdir = Path(__file__).resolve().parents[2] / "reports/SLA_FROZEN_FEEDBACK"
    cfg = read(rdir / "F0_REGISTRATION.json")
    plan = read(rdir / "F1_F2_REGISTRATION.json")
    case = a.closed_card.resolve()
    result = read(case / "RESULT.private.json")
    require(
        result["outcome"] == "A0_COMPLETE"
        and case.parent.name == plan["campaigns"]["F1"],
        "closed known target",
    )
    modelpath = Path(base["model_path"])
    pinned(modelpath, plan["scene_sha256"])
    row = dict(
        result,
        result_sha256=sha(case / "RESULT.private.json"),
        baseline_reproduced=None,
    )
    with forbid_integration(mujoco) as calls:
        faults = preflight(root, cfg)
        verify_finger_file(
            case / result["episode"], result["ticks"], 127 + result["atoms"]
        )
        audit = verify_card(
            case, row, "F1", mujoco.MjModel.from_binary_path(str(modelpath)), None
        )
    require(
        not calls
        and audit["post_init_ticks"] == result["ticks"] - START
        and audit["prohibited_contact_records"]
        == audit["robot_cube_contact_records"]
        == 0,
        "closed example",
    )
    record(
        a.output,
        {
            "phase": "F2_PRELAUNCH_CPU",
            "fault_receipts": faults,
            "example": audit,
            "source_sha256": {
                p.name: sha(p) for p in Path(__file__).parent.glob("*.py")
            },
            "native_requests": 0,
            "physics_ticks": 0,
        },
    )


if __name__ == "__main__":
    main()
