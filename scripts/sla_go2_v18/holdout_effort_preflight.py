"""Replay the deployed effort guard on private copies of the new VAL scenes, without physics."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require

REG = BASE / "reports/SLA_GO2_V18/P1_P2"
OLD = "sla-core-hopper-v1-expert-chain-20261121-02"


def check_contact_roles(spec, instruction):
    from go2arx5_vla.world_model import multitask_contact_v6 as contacts

    selected = instruction.split("-")[1]
    target = contacts.OBJECT_GEOMS[selected]
    other = next(g for c, g in contacts.OBJECT_GEOMS.items() if c != selected)
    module = Path(contacts.__file__)
    source = "src/go2arx5_vla/world_model/multitask_contact_v6.py"
    require(
        module.resolve() == (Path(spec["source_root"]) / source).resolve(),
        "wrong Python contact predicate",
    )
    pin(module, spec["source_files"][source])
    binary = Path(spec["observer_library"])
    pin(binary, spec["frozen_files"][str(binary)])
    role = ctypes.CDLL(str(binary)).WM02MultitaskTaskPairV6
    role.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
    role.restype = ctypes.c_int
    tests = (
        (
            "selected_pad_approach",
            "task_left_pad",
            target,
            -0.001,
            "approach",
            (1, True, False),
        ),
        (
            "selected_pad_close",
            "task_left_pad",
            target,
            -0.001,
            "close",
            (1, True, True),
        ),
        (
            "selected_pad_maximum",
            "task_right_pad",
            target,
            -0.004,
            "close",
            (1, True, True),
        ),
        (
            "selected_pad_exceeds_maximum",
            "task_right_pad",
            target,
            -0.004001,
            "close",
            (1, False, False),
        ),
        (
            "wrong_object_pad",
            "task_right_pad",
            other,
            -0.001,
            "close",
            (0, False, False),
        ),
        (
            "pad_hits_table",
            "task_left_pad",
            "task_table",
            -0.001,
            "close",
            (0, False, False),
        ),
        (
            "passive_cube_table",
            other,
            "task_table",
            -0.001,
            "approach",
            (1, True, True),
        ),
    )
    results = []
    for name, a, b, pen, stage, expected in tests:
        actual = (
            role(a.encode(), b.encode(), selected.encode()),
            contacts.observer_permitted(a, b, pen, selected),
            contacts.preview_permitted(a, b, pen, stage, selected),
        )
        require(
            actual == expected,
            f"contact classification changed for {instruction}: {name}",
        )
        results.append(
            dict(
                case=name,
                native_role=actual[0],
                observer_allowed=actual[1],
                preview_allowed=actual[2],
            )
        )
    return dict(
        observer_sha256=spec["frozen_files"][str(binary)],
        selected_color=selected,
        checks=results,
    )


def check_one(archive, compiled, row):
    import mujoco
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    src = archive / OLD / "cases" / row["historical_positive"]["id"]
    old = read(
        pin(src / "RESULT.private.json", row["historical_positive"]["result_sha256"])
    )
    require(
        old["completed"] and old["shutdown_clean"] and not old["unsafe"],
        "unverified historical carrier",
    )
    lease = read(src / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "unclosed historical input")
    spec = read(pin(src / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    scene_name = next(
        x["cluster"] for x in compiled["conditions"]["VAL"] if x["id"] == row["slot"]
    )
    catalog = next(x["scene"] for x in compiled["clusters"] if x["id"] == scene_name)
    require(catalog["sha256"] == row["geometry_sha256"], "different VAL geometry")
    model = mujoco.MjModel.from_binary_path(
        str(pin(Path(catalog["path"]), catalog["sha256"]))
    )
    data = mujoco.MjData(model)
    fixture = next(
        p for p in packets(src / "native-01/state.bin.journal") if p["step"] == 5703
    )
    require(
        model.nq == 41
        and model.nv == 38
        and model.nu == 20
        and model.opt.timestep == 0.002
        and model.jnt_qposadr[model.joint("task_cube_free").id] == 27
        and model.jnt_dofadr[model.joint("task_cube_free").id] == 26,
        "frozen state split",
    )
    require(mujoco.get_mjcb_control() is None, "use an isolated inert replay process")
    data.qpos[:27] = fixture["qpos"][:27]
    data.qvel[:26] = fixture["qvel"][:26]
    data.ctrl[:] = fixture["ctrl"]
    data.time = fixture["time_s"]
    qualification = spec["mobile_guard_qualification"]
    guard = read(pin(Path(qualification["path"]), qualification["sha256"]))
    lib = pin(Path(spec["mobile_safety_library"]), guard["library_sha256"])
    binding = pin(Path(spec["mobile_safety_binding"]), guard["binding_sha256"])
    qid = model.joint("x5_joint4").id
    j = [
        i
        for i in range(model.nu)
        if model.actuator_trnid[i, 0] == qid
        and model.actuator_trntype[i] == int(mujoco.mjtTrn.mjTRN_JOINT)
    ]
    require(
        len(j) == 1
        and model.actuator_gear[j[0], 0] == 1.0
        and guard["checks"]["over_effort"] == {"actual": 6, "expected": 6},
        "wrong deployed effort ABI",
    )
    actuator = j[0]
    test = (
        (None, 0),
        (7.0, 0),
        (-7.0, 0),
        (7.000001, 6),
        (-7.000001, 6),
        (-7.013501112271807, 6),
    )
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="sla-heldout-guard-") as farm:
        d = Path(farm)
        os.environ.update(
            MOBILE_SAFETY_BINDING=str(binding),
            MOBILE_SAFETY_ARMED=str(d / "ARMED"),
            MOBILE_SAFETY_STOP=str(d / "STOP"),
            MOBILE_SAFETY_TRACE=str(d / "guard.jsonl"),
        )
        fn = ctypes.CDLL(str(lib)).mobile_safety_evaluate_v1
        fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        fn.restype = ctypes.c_int
        saved = data.qpos.copy()
        with forbid_integration(mujoco) as calls:
            mujoco.mj_forward(model, data)
            for command, expected in test:
                if command is not None:
                    data.ctrl[actuator] = command
                code = fn(model._address, data._address)
                outcomes.append(
                    dict(
                        joint4_command_nm=float(data.ctrl[actuator]),
                        expected=expected,
                        actual=code,
                    )
                )
            require(
                not calls
                and data.time == fixture["time_s"]
                and np.array_equal(data.qpos, saved)
                and not (d / "STOP").exists(),
                "static guard replay advanced or signaled an execution",
            )
        require(
            (d / "guard.jsonl").is_file(),
            "deployed guard did not initialize its own binding",
        )
    return dict(
        slot=row["slot"],
        historical_robot_fixture=row["historical_positive"]["id"],
        fixture_tick=fixture["step"],
        scene_sha256=row["geometry_sha256"],
        guard_sha256=guard["library_sha256"],
        outcomes=outcomes,
        contact_roles=check_contact_roles(spec, row["task"]),
        matched=all(v["actual"] == v["expected"] for v in outcomes),
        integration_ticks=0,
        world_starts=0,
    )


def main(args):
    plan = read(REG / "VAL_CANDIDATES.json")
    scope = read(REG / "HELDOUT_COMPILED.json")
    compiled = read(pin(args.compiled, scope["compiled_manifest_sha256"]))
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "no GPU probes")
    if args.slot:
        row = next(x for x in plan["rows"] if x["slot"] == args.slot)
        return check_one(args.archive, compiled, row)
    records = []
    for row in plan["rows"]:
        p = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.archive),
                str(args.compiled),
                "--slot",
                row["slot"],
            ],
            check=True,
            timeout=45,
            capture_output=True,
            text=True,
        )
        records.append(json.loads(p.stdout))
    require(
        len(records) == 4 and all(r["matched"] for r in records),
        "frozen guard outcome changed",
    )
    return dict(
        status="REFERENCE_EFFORT_AND_CONTACT_CLASSIFIER_REPLAY_PASS",
        slots=records,
        training_updates=0,
        integration_ticks=0,
        new_SLA_forwards=0,
        native_dispatch_qualified=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("compiled", type=Path)
    p.add_argument("--slot")
    print(json.dumps(main(p.parse_args()), indent=2, allow_nan=False))
