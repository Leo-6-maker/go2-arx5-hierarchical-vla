"""Build a separately identifiable two-object receiver and exercise its real STOP on archived states."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_hierarchical_v16"))
from build_student_receiver import replace_once
from demonstrated_grip import pin, read, require
from moving_stop_card import baseline, configure, record, sha
from reach_profile import BASE, SLOTS
from receiver_fault_replay import replay_treatments


def run(root, out):
    old = root / "v17/cap-21203-replay02/receiver"
    native = read(
        root / "feedback/sla-feedback-f2-a1-20261123-01/F2-01/EXECUTION_SPEC.json"
    )
    lib = Path(native["outer_library"])
    argv = next(
        v["argv"]
        for v in read(lib.parent / "COMMANDS.private.json")
        if v["label"] == "task_actuation_r1_compile"
    )
    engine = Path(next(x for x in argv if x.endswith("/libmujoco.so")))
    inc = next(x for x in argv if x.startswith("-I") and "/library/mujoco/include" in x)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    original = pin(
        old / lib.with_suffix(".cpp").name,
        "19c564d8c931c690f54b790dbb18b9a36dfffc2317676807c5fab7172ef05ffa",
    )
    path = out / original.name
    for h in ("init_trace.hpp", "student_receiver.hpp"):
        shutil.copyfile(old / h, out / h)
    receiver = out / "student_grip.so"
    build = [
        str(path)
        if x == str(lib.with_suffix(".cpp"))
        else str(receiver)
        if x == str(lib)
        else x
        for x in argv
    ]
    path.write_bytes(original.read_bytes())
    subprocess.run(build, check=True)
    pin(receiver, "7a0d94a7cd39e468107525e7415936cceeebb931eeef2d70f7d65f067020039d")
    s = replace_once(
        path.read_text(), "m->nq!=34 || m->nv!=32", "m->nq!=41 || m->nv!=38"
    )
    for marker in (
        'std::strcmp(cap,"21203")',
        'std::getenv("V16_GRIP_EPOCH"),21203,target',
    ):
        s = replace_once(s, marker, marker.replace("21203", "20000"))
    s = replace_once(
        s,
        'return !std::strcmp(y,"task_cube_geom");',
        'return !std::strcmp(y,"task_cube_geom") || !std::strcmp(y,"task_blue_cube_geom");',
    )
    s = replace_once(
        s,
        "if(!support(a,b) && !support(b,a))contact_ok=false;",
        """if(!support(a,b) && !support(b,a))contact_ok=false;
   if(a && b && (!std::strcmp(a,"task_table") || !std::strcmp(b,"task_table")) && d->contact[i].dist < -.004)contact_ok=false;""",
    )
    path.write_text(s)
    subprocess.run(build, check=True)
    code = BASE / "scripts/sla_hierarchical_v16"
    writer = Path(native["trace_library"])
    pin(writer, native["frozen_files"][str(writer)])
    common = ["g++", "-std=c++17", "-O2", "-DV18_TWO_OBJECT_ARCHIVE", inc]
    recorded = out / "librecorded_step.so"
    exe = out / "fault_probe"
    subprocess.run(
        [
            *common,
            "-fPIC",
            "-shared",
            str(code / "recorded_step.cc"),
            str(engine),
            "-Wl,-rpath," + str(engine.parent),
            "-o",
            str(recorded),
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            "-rdynamic",
            str(code / "receiver_fault_probe.cc"),
            "-Wl,--no-as-needed",
            str(receiver),
            str(recorded),
            str(engine),
            str(writer),
            "-ldl",
            "-Wl,-rpath," + str(engine.parent),
            "-o",
            str(exe),
        ],
        check=True,
    )
    source = Path(
        read(root / "evidence/P1_REGISTRATION.json")["target_windows"][0][
            "episode_pin"
        ]["path"]
    ).parents[3]
    info = read(root / "v18/reach-profile05.json")
    rows = []
    for name, r in zip(SLOTS, info["results"]):
        case = source / "cases" / name
        spec = read(pin(case / "EXECUTION_SPEC.json", r["source_spec_sha256"]))
        scene = pin(Path(spec["model_path"]), r["compiled_scene"])
        task = out / name
        task.mkdir(mode=0o700)
        with (case / "native-01/initialization.jsonl").open() as f:
            import json

            init = json.loads(next(f))
        require(
            init["time"] == 0 and len(init["qpos"]) == 41 and len(init["qvel"]) == 38,
            "reset identity",
        )
        reset = task / "reset.bin"
        np.r_[init["qpos"], init["qvel"]].astype("<f8").tofile(reset)
        journal = case / "native-01/state.bin.journal"
        v = replay_treatments(
            task,
            exe,
            receiver,
            recorded,
            writer,
            scene,
            reset,
            journal,
            20000,
            code / "receiver_fault_probe.cc",
            "BRIDGE",
        )
        rows.append(
            {
                "episode": name,
                "scene_sha256": sha(scene),
                "journal_sha256": sha(journal),
                "treatments": v,
            }
        )
    return {
        "receiver_sha256": sha(receiver),
        "new_source_sha256": sha(path),
        "frozen_trace_sha256": sha(writer),
        "test_step_sha256": sha(recorded),
        "parent_receiver_rebuilt_byte_identical": True,
        "results": rows,
        "whole_system_qualified": False,
        "new_integrations": 0,
        "new_native_starts": 0,
        "training_updates": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("root", "build", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    configure(baseline(root))
    sys.path.insert(0, str(root / "v12/CPU_LIB"))
    os.umask(0o077)
    record(a.out, run(root, a.build.resolve()))
